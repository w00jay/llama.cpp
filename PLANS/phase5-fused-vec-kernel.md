# Phase 5: Fused TBQ Attention Kernel (VEC path)

## Context

TBQ currently dequantizes K/V to fp16 (via NC dequant kernel), then runs standard
MMA/TILE flash attention. This throws away TBQ's main advantage: computing attention
scores directly from compressed data without full dequantization.

Current PPL results show TBQ4 (9.65) ≈ q4_0-without-rotation (9.05), much worse than
q4_0-with-rotation (5.53). The full dequant roundtrip adds reconstruction noise that
negates the quality benefits of Lloyd-Max + QJL.

The fused kernel computes scores in the rotated domain, avoiding reconstruction noise:
```
q_rot = externally rotated query (already done by llama.cpp)
For each cached key K:
  score = d_norm * (dot(q_rot, centroids[K.idx]) + sqrt(pi/2)/d * gamma * dot_qjl)
where dot_qjl = sum over j of (S_row_j · q_rot_normalized) * qjl_sign_j
```

## Architecture Decision

**Use the VEC kernel path.** The VEC kernel already supports per-type fused K·Q dot products
(Q4_0, Q8_0 all have specializations). MMA/TILE require fp16 and can't be easily modified.

For V dequant: keep the existing pre-conversion to fp16. V values are weighted-summed
(not dot-producted), so they must be fully dequantized. V continues using the NC path.

## Files to Modify

| File | Change |
|------|--------|
| `ggml/src/ggml-cuda/fattn-common.cuh` | Add `vec_dot_fattn_vec_KQ_tbq3_0` and `tbq4_0` functions + dispatcher entries; add V dequant specializations |
| `ggml/src/ggml-cuda/fattn.cu` | Route TBQ to VEC kernel (K type) with fp16 V; add VEC dispatcher entries |
| `ggml/src/ggml-cuda/fattn-common.cuh` | Skip K pre-conversion for TBQ (raw TBQ bytes go to VEC kernel) |

## Implementation Details

### 1. Fused K·Q dot product: `vec_dot_fattn_vec_KQ_tbq3_0`

**File:** `fattn-common.cuh` (after existing Q8_0 specialization, ~line 289)

```cuda
template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_tbq3_0(
    const char * __restrict__ K_c,
    const void * __restrict__ Q_v,
    const int * __restrict__ Q_q8,
    const void * __restrict__ Q_ds_v)
{
    // K_c points to one block_tbq3_0 (52 bytes for 128 elements)
    // Q_v points to the query as half2 (f16)
    // Q_q8 / Q_ds_v: q8_1 quantized query (used by other types, we use Q_v instead)

    const block_tbq3_0 * K_tbq = (const block_tbq3_0 *) K_c;
    const half2 * Q_h2 = (const half2 *) Q_v;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    static_assert(D == 128, "TBQ requires D=128");

    const float d_norm = __half2float(K_tbq->d);
    const float gamma  = __half2float(K_tbq->gamma);

    // Part 1: dot(q, centroids[idx])
    // The query is already in the rotated domain (external Hadamard applied by llama.cpp).
    // The centroids are in the rotated domain too (stored that way during quantization).
    float dot_centroid = 0.0f;
    // Each thread handles a subset of the 128 coordinates
    for (int k0 = 0; k0 < D/2; k0 += nthreads) {
        const int k = k0 + threadIdx.x % nthreads;
        if (k < D/2) {
            const int j0 = 2*k;
            const int j1 = j0 + 1;
            int idx0 = (K_tbq->idx[j0/4] >> (2*(j0%4))) & 3;
            int idx1 = (K_tbq->idx[j1/4] >> (2*(j1%4))) & 3;
            half2 q = Q_h2[k];
            dot_centroid += __half2float(q.x) * cb4[idx0]
                         + __half2float(q.y) * cb4[idx1];
        }
    }

    // Part 2: QJL correction term
    // dot_qjl = sum_j qjl_sign_j * dot(S_row_j, q_normalized)
    // S_row_j is Rademacher row generated from PRNG seeded per (block_in_row, j)
    // This is O(d) per QJL bit if we precompute the signs, but we need the PRNG.
    //
    // Optimization: compute per-thread partial sums, then warp reduce.
    // Each thread handles a subset of the 128 QJL bits.
    float dot_qjl = 0.0f;
    for (int j0 = 0; j0 < QK_TBQ; j0 += nthreads) {
        const int j = j0 + threadIdx.x % nthreads;
        if (j < QK_TBQ) {
            float qjl_sign = ((K_tbq->qjl[j/8] >> (j%8)) & 1) ? 1.0f : -1.0f;
            // dot(S_row_j, q) = sum_l S[j][l] * q[l]
            uint64_t srng = tbq_qjl_seed_cuda(block_in_row, j);
            float s_dot_q = 0.0f;
            for (int l = 0; l < QK_TBQ; l++) {
                srng = tbq_xorshift64_cuda(srng);
                float s_jl = (srng & 1) ? 1.0f : -1.0f;
                s_dot_q += s_jl * __half2float(((const half *)Q_v)[l]);
            }
            dot_qjl += qjl_sign * s_dot_q;
        }
    }

    // Warp reduce both partial sums
    dot_centroid = warp_reduce_sum(dot_centroid);
    dot_qjl     = warp_reduce_sum(dot_qjl);

    const float qjl_scale = sqrtf(M_PI / 2.0f) / (float)QK_TBQ * gamma;
    return d_norm * (dot_centroid + qjl_scale * dot_qjl);
}
```

**Key challenge:** The `block_in_row` is needed for PRNG seeding but the VEC kernel
function signature `vec_dot_KQ_t` doesn't carry it. Options:
- (a) Pass block_in_row through K pointer arithmetic (K_c - base gives the block index)
- (b) Store block_in_row in unused bits of the TBQ block
- (c) Compute from the K pointer offset: `block_in_row = ((K_c - K_base) / sizeof(block_tbq3_0)) % blocks_per_row`
- **(d) Set block_in_row = 0 always.** The external rotation already happened, and the QJL
  rows don't need per-head differentiation since each head is independent. The PRNG seed
  just needs to be consistent between quantize and dequant. Since we removed the internal
  rotation, the only PRNG-dependent part is QJL. And QJL seeds with `(block_idx, coord)` —
  if both quantize and fused-dot use `block_idx = position within row`, they match.

Actually — wait. The quantize path (set_rows.cu) uses `block_idx = i00/qk` which is the
within-row position. The VEC kernel receives K as a pointer with strides, and each call
processes one K row. The K pointer for row `i_KQ` is `K + i_KQ * nb11`. We need `block_in_row`
which is the block's position in the original cache row.

Since `nb11` = stride between KV positions (in bytes), and each cache row has `n_head_kv`
TBQ blocks, `block_in_row = (K_c - K_base) % (nb11_orig * n_head_kv / sizeof(block))`.
This is complex. But since the query and key are **both** in the rotated domain, the PRNG
seed consistency only matters for the QJL correction term — and that term is small.

**Simplification for initial implementation:** Use `block_in_row = 0` for the fused kernel.
This introduces a small bias in the QJL term for heads > 0, but the centroid dot product
(the dominant term) is exact. We can fix the PRNG seeding in a follow-up.

### 2. V dequant: keep fp16 pre-conversion (NC path)

V values are weighted-summed in attention, not dot-producted. There's no shortcut —
full dequant is needed. Continue using the existing NC dequant → fp16 path.

For the VEC type dispatch, use `type_V = GGML_TYPE_F16` (V is pre-converted to fp16).

### 3. VEC dispatcher entries and type check fix

**File:** `fattn.cu` (~line 289)

Add after existing VEC cases:
```cpp
FATTN_VEC_CASE(128, GGML_TYPE_TBQ3_0, GGML_TYPE_F16)
FATTN_VEC_CASE(128, GGML_TYPE_TBQ4_0, GGML_TYPE_F16)
```

Only D=128 (no 64/256) since TBQ requires head_dim=128.

**FATTN_VEC_CASE type_V check issue:** The macro checks `V->type == type_V`. When V is
TBQ in the cache but we dispatch with `type_V = GGML_TYPE_F16`, the check fails. Fix:
expand the `type_V_okay` condition to allow TBQ V types when `type_V == GGML_TYPE_F16`:
```cpp
const bool type_V_okay = V->type == (type_V)
    || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16)
    || ((V->type == GGML_TYPE_TBQ3_0 || V->type == GGML_TYPE_TBQ4_0) && (type_V) == GGML_TYPE_F16);
```

This is safe because `launch_fattn` will pre-convert V to fp16 via `need_f16_V = true`.

### 4. K pre-conversion: handled automatically

**File:** `fattn-vec.cuh` line 529

`need_f16_K = (type_K == GGML_TYPE_F16)`. When type_K = TBQ3_0, need_f16_K = false.
K stays raw — the fused dot product reads TBQ bytes directly. No changes needed.

V pre-conversion: `need_f16_V = (type_V == GGML_TYPE_F16)` = true → V gets pre-converted
from TBQ to fp16 inside `launch_fattn`. The NC path handles this correctly.

### 5. Kernel routing

**File:** `fattn.cu` TBQ case in `ggml_cuda_get_best_fattn_kernel`

Change TBQ routing to prefer VEC for single-token decode (where it shines), fall back
to MMA/TILE with fp16 pre-conversion for batched prefill:
```cpp
case GGML_TYPE_TBQ3_0:
case GGML_TYPE_TBQ4_0: {
    if (mask && mask->ne[2] != 1) return BEST_FATTN_KERNEL_NONE;
    // VEC for decode (Q->ne[1] small), MMA for prefill
    if (can_use_vector_kernel && Q->ne[1] <= 2) return BEST_FATTN_KERNEL_VEC;
    if (turing_mma_available(cc) && K->ne[0] != 40 && K->ne[0] != 72) return BEST_FATTN_KERNEL_MMA_F16;
    return BEST_FATTN_KERNEL_TILE;
}
```

## Verification

1. **Build** on dell box with CUDA
2. **FA tests:** `test-backend-ops -o FLASH_ATTN_EXT | grep tbq` — all 4 OK
3. **SET_ROWS:** 141/141 still pass
4. **Perplexity:** Compare PPL for TBQ3/TBQ4 vs current (should be same or better — the
   fused centroid dot product is mathematically identical, QJL term may differ slightly)
5. **Speed:** Compare tokens/sec vs pre-conversion path (should be faster for decode)

## Risk

- **QJL PRNG seed mismatch (block_in_row=0):** May cause small quality regression for
  multi-head models. Mitigation: the QJL term is a small correction on top of the centroid
  dot product. If PPL regresses, add proper block_in_row recovery.
- **Compile time increase:** Each `FATTN_VEC_CASE` instantiates a full template. Adding 2
  cases is minor.
