# TBQ Quality Trials Log

Model: Selene-1-Mini-Llama-3.1-8B-Q6_K (head_dim=128, n_head_kv=8)
Dataset: wikitext-2 (10-chunk estimates unless noted)
GPU: RTX 3090 (sm_86)

## Baselines

| Config | bpw | PPL | Notes |
|--------|-----|-----|-------|
| f16 K+V | 16 | 5.46 | Gold standard |
| q4_0 K+V (with rotation) | ~4.5 | 5.53 | llama.cpp default for quantized KV |
| q4_0 K+V (no rotation) | ~4.5 | 9.05 | LLAMA_ATTN_ROT_DISABLE=1 |

## TBQ Trials

### Trial 1: Initial working implementation (after Phase 3 fixes)
- Commit: `799bae7d7`
- Config: TBQ internal rotation removed, external WHT rotation, no sign-flip, no per-block scale
- **TBQ4_0: PPL = 9.65** | TBQ3_0: PPL = 18.27
- Finding: Comparable to q4_0-without-rotation (9.05), much worse than q4_0-with-rotation (5.53)
- Diagnosis: Plain WHT without sign-flip doesn't give N(0,1/d) distribution for codebook

### Trial 2: SRHT sign-flip added
- Commit: `e6962c78e`
- Config: Added per-block random sign-flip (Rademacher diagonal) to complete SRHT
- **TBQ4_0: PPL = 9.65** (unchanged) | **TBQ3_0: PPL = 16.74** (-8%)
- Finding: Sign-flip helps TBQ3 but not TBQ4. Distribution was already reasonable for 8 centroids.

### Trial 3: Per-block adaptive scale
- Commit: `77a9c405f`
- Config: Added fp16 scale field to struct. Coords divided by per-block RMS before centroid lookup.
- Block sizes: TBQ3 52→54 bytes, TBQ4 68→70 bytes
- **TBQ4_0: PPL = 9.62** (-0.3%) | **TBQ3_0: PPL = 16.81** (unchanged)
- Finding: Per-block scale doesn't help — codebook already matches post-rotation distribution

### Trial 4: QJL disabled entirely (experiment, not committed)
- Config: Set gamma=0 in CUDA dequant (both K and V)
- **TBQ4_0: PPL = 9.53** (-1%) | **TBQ3_0: PPL = 13.54** (-19%)
- **KEY FINDING: QJL reconstruction noise is the primary quality bottleneck**
- QJL adds stochastic correction that's unbiased for IP estimation but adds noise to each
  coordinate. V vectors are weighted-summed (not dot-producted), so noise accumulates.

### Trial 5: Asymmetric QJL — V only no-QJL
- Commit: first attempt at `8db195b8d`
- Config: Skip QJL in V pre-conversion, keep QJL in K pre-conversion and fused VEC dot
- **TBQ4_0: PPL = 9.84** (+2%) | **TBQ3_0: PPL = 15.77** (-6%)
- Finding: V improvement offset by K pre-conversion still using QJL (MMA/TILE prefill path)

### Trial 6: No-QJL on both K and V pre-conversion paths
- Commit: `7b6c83ce8`
- Config: Skip QJL in NC dequant for both K and V. Fused VEC dot (decode only) still uses QJL.
- **TBQ4_0: PPL = 9.53** | **TBQ3_0: PPL = 13.54**
- Matches Trial 4 exactly — confirms QJL noise was the bottleneck in pre-conversion path
- Fused VEC QJL (decode only) has negligible impact on PPL

### Trial 7: Asymmetric K/V types (TBQ K + q4_0 V) — PENDING
- Config: `--cache-type-k tbq4_0 --cache-type-v q4_0`
- Status: Times out due to slow NC dequant kernel (single-threaded, even without QJL)
- Expected: PPL ~5.5-6.0 (V quality dominates, q4_0 V is much better than TBQ V)

## Summary of Findings

1. **QJL hurts V quality** — the stochastic correction noise accumulates in weighted sums.
   Skipping QJL in pre-conversion improved TBQ3 by 20%.
2. **Per-block scale doesn't help** — post-rotation distribution already matches codebook.
3. **Sign-flip helps marginally** — completes SRHT but coordinates were already well-distributed.
4. **The remaining gap (9.53 vs 5.53)** is the fundamental limit of fixed Lloyd-Max centroids
   (4 or 8 levels) vs q4_0's adaptive uniform quantization (16 levels per block).
5. **Asymmetric K/V** (TBQ K + q4_0 V) is the most promising path but blocked by slow
   NC dequant kernel performance.
