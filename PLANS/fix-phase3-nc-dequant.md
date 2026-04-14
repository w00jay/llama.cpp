# Fix Phase 3: Non-Contiguous TBQ Dequant for Flash Attention

## Context

Phase 3 added CUDA flash attention support for TBQ types, but the implementation
only handles **contiguously allocated** K/V tensors. In llama.cpp, KV cache K/V
tensors are **always views** of a larger cache tensor (e.g., [128, 128, 4, 1] view
of [128, 256, 4, 1]), which means `ggml_is_contiguously_allocated()` returns false.

Result: `ggml_cuda_get_best_fattn_kernel` returns `BEST_FATTN_KERNEL_NONE` for all
TBQ tensors at runtime, causing `GGML_ABORT("fatal error")`. The FA tests also all
show "not supported" — **they never actually ran**.

## Root Cause

The existing quant types (Q4_0, Q8_0, etc.) handle non-contiguous views via
`ggml_get_to_fp16_nc_cuda()`, which returns a stride-aware dequant function.
TBQ is not registered there, so the Phase 3 code added a guard that rejects
non-contiguous tensors instead. The guard needs to be removed, and a proper
NC dequant kernel added.

## Fix Plan

### 1. Add NC dequant CUDA kernels to `convert.cu`

**File:** `ggml/src/ggml-cuda/convert.cu`

Add two new kernel + launcher pairs following the existing NC pattern.

**Kernel signature** (matches the `dequantize_block` template pattern):
```cuda
template<typename dst_t>
static __global__ void k_dequantize_nc_tbq3_0(
    const void * __restrict__ vx, dst_t * __restrict__ y,
    const int64_t ne00, const int64_t ne01,
    const int64_t ne0203, const uint3 ne02_fdv,
    const int64_t s01, const int64_t s02, const int64_t s03)
```

**Key differences from existing `dequantize_block` template:**
- Existing template: per-element (2 elements per thread), uses `dequantize_kernel_t` function pointer
- TBQ kernel: per-block (128 elements per thread), calls `dequantize_f32_tbq3_0_block` from tbq-quants.cuh

**Index computation per CUDA thread:**
```
// One TBQ block = 128 output elements = 1 "row" in dim 0
// Each thread handles one TBQ block at position (i01, i02, i03)
thread_id = blockIdx.x;  // one CUDA thread per TBQ block

// Map flat thread to (i01, i02, i03) — same loop pattern as dequantize_block
for i01 in [blockIdx.y .. ne01, step gridDim.y]:
  for i0203 in [blockIdx.z .. ne0203, step gridDim.z]:
    i02 = i0203 % ne02  (via fast_div_modulo)
    i03 = i0203 / ne02

    // Source block in strided tensor (strides in block units):
    src_block_idx = i03*s03 + i02*s02 + i01*s01
    // (no + i00/QK_TBQ term: ne00=128=QK_TBQ, so always 0)

    // block_in_row for PRNG seeding:
    block_in_row = 0  // ne00/QK_TBQ = 1 always

    // Output: contiguous, one block = QK_TBQ fp16 values
    y_offset = (i0203*ne01 + i01) * QK_TBQ
```

**Launcher** (matches `to_fp16_nc_cuda_t` signature):
```cuda
template<typename dst_t>
static void dequantize_nc_tbq3_0_cuda(
    const void * vx, dst_t * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream)
{
    GGML_ASSERT(ne00 == QK_TBQ);  // TBQ requires head_dim == 128
    const int64_t ne0203 = ne02 * ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    // 1 CUDA block per TBQ block along dim 0 (always 1 block since ne00=QK_TBQ)
    const dim3 num_blocks(1, (int)std::min(ne01, (int64_t)65535),
                             (int)std::min(ne0203, (int64_t)65535));
    k_dequantize_nc_tbq3_0<<<num_blocks, 1, 0, stream>>>(
        vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}
```

Same pattern for TBQ4_0.

### 2. Register NC dequant in `ggml_get_to_fp16_nc_cuda`

**File:** `ggml/src/ggml-cuda/convert.cu` (~line 881)

Add cases to the switch:
```cpp
case GGML_TYPE_TBQ3_0:
    return dequantize_nc_tbq3_0_cuda;
case GGML_TYPE_TBQ4_0:
    return dequantize_nc_tbq4_0_cuda;
```

### 3. Remove the contiguously-allocated guard from `fattn.cu`

**File:** `ggml/src/ggml-cuda/fattn.cu` (~line 398-414)

Replace the TBQ case block:
```cpp
// BEFORE (buggy):
case GGML_TYPE_TBQ3_0:
case GGML_TYPE_TBQ4_0: {
    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }
    if (!ggml_is_contiguously_allocated(K) || !ggml_is_contiguously_allocated(V)) {
        return BEST_FATTN_KERNEL_NONE;
    }
    if (turing_mma_available(cc) && K->ne[0] != 40 && K->ne[0] != 72) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }
    return BEST_FATTN_KERNEL_TILE;
}

// AFTER (fixed — contiguity guard removed, fall through to general path):
case GGML_TYPE_TBQ3_0:
case GGML_TYPE_TBQ4_0:
    break;
```

By falling through to `break`, TBQ enters the general path below (lines 420+)
which handles mask checks, VEC/TILE/MMA selection, and GQA optimization —
all of which work correctly since the NC dequant now handles the view layout.

### 4. Add TBQ thresholds in `test-quantize-fns.cpp`

**File:** `tests/test-quantize-fns.cpp`

Add TBQ-specific error thresholds (TBQ has higher per-element error by design):
```cpp
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_TBQ = 0.015f;   // ~0.0095 for TBQ3
constexpr float MAX_DOT_PRODUCT_ERROR_TBQ = 1.5f;            // ~1.1 observed
```

Wire into the threshold selection:
```cpp
type == GGML_TYPE_TBQ3_0 || type == GGML_TYPE_TBQ4_0
    ? MAX_QUANTIZATION_TOTAL_ERROR_TBQ : ...

type == GGML_TYPE_TBQ3_0 || type == GGML_TYPE_TBQ4_0
    ? MAX_DOT_PRODUCT_ERROR_TBQ : ...
```

## Files Modified

| File | Change |
|------|--------|
| `ggml/src/ggml-cuda/convert.cu` | Add NC dequant kernels + launchers; register in `ggml_get_to_fp16_nc_cuda` |
| `ggml/src/ggml-cuda/fattn.cu` | Remove `ggml_is_contiguously_allocated` guard for TBQ; fall through to general path |
| `tests/test-quantize-fns.cpp` | Add TBQ-specific error thresholds |

## Verification

1. **Build:** `cmake --build build -j$(nproc)` on build server
2. **FA tests:** `LD_LIBRARY_PATH=build/bin CUDA_VISIBLE_DEVICES=1 ./build/bin/test-backend-ops -o FLASH_ATTN_EXT 2>&1 | grep tbq`
   - Expected: TBQ tests show `OK` (not "not supported")
3. **SET_ROWS regression:** `LD_LIBRARY_PATH=build/bin CUDA_VISIBLE_DEVICES=1 ./build/bin/test-backend-ops -o SET_ROWS`
   - Expected: 141/141 still pass
4. **quantize-fns:** `LD_LIBRARY_PATH=build/bin ./build/bin/test-quantize-fns`
   - Expected: TBQ tests pass with new thresholds
5. **Phase 0 math:** `g++ -std=c++17 -O2 -o test-tbq-math tests/test-tbq-math.cpp -lm && ./test-tbq-math`
   - Expected: 7/7 still pass (sanity check)
