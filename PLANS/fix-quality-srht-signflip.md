# Fix TBQ Quality Gap: Add Random Sign-Flip (SRHT)

## Context

TBQ4 PPL=9.65 vs q4_0 PPL=5.53. Root cause: llama.cpp's external Hadamard is a plain
WHT without random sign flips. TBQ's codebook assumes coordinates are ~N(0, 1/d), which
requires the full SRHT (sign-flip + WHT). Without sign-flips, coordinates retain
model-specific structure and the fixed codebook is a poor fit.

## Approach

Add a per-block random sign-flip inside TBQ quantize/dequant. Combined with the
external WHT, this gives the full SRHT: `(1/√d) * H * D * x`. Just 128 multiplies
by ±1 per block — O(d), no FWHT needed. Uses existing `tbq_rot_seed` PRNG.

## Files to Modify

| File | Change |
|------|--------|
| `ggml/src/ggml-quants.c` | Add sign-flip to quantize + dequant (CPU, 4 functions) |
| `ggml/src/ggml-cuda/tbq-quants.cuh` | Add sign-flip to CUDA quantize + dequant |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | Add shared `tbq_kv_head` variable for sign-flip seed |
| `ggml/src/ggml-cuda/fattn-common.cuh` | Fused VEC dot: sign-flip query before dot product |

## Key Detail: Sign-Flip in Fused VEC Kernel

The fused dot product computes `dot(q, centroids[idx])` where centroids were stored
based on sign-flipped coordinates. Query must also be sign-flipped with the SAME seed.

The sign-flip seed is `tbq_rot_seed(block_in_row)` where `block_in_row` = KV head index.
In the VEC kernel, head index = `(head / gqa_ratio)`, available at line 99 of fattn-vec.cuh.

Pass it via a `__shared__` variable:
```cuda
// In fattn-vec.cuh, before K-loop:
__shared__ int tbq_kv_head;
if (threadIdx.x == 0 && threadIdx.y == 0) {
    tbq_kv_head = head / gqa_ratio;
}
__syncthreads();
```

The fused dot function reads `tbq_kv_head` from shared memory to seed the sign-flip.

## Verification

1. Build on build server
2. FA tests: 4/4 TBQ OK
3. PPL: TBQ4 should improve from 9.65 toward 5.5-6.5
4. SET_ROWS: 141/141 pass
