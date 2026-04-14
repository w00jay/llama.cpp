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

### Trial 7: Asymmetric K/V types (TBQ K + q4_0 V) — BLOCKED
- Config: `--cache-type-k tbq4_0 --cache-type-v q4_0`
- Status: Times out even at 30 min. NC dequant kernel launches 1 CUDA thread per TBQ block;
  for 512 tokens × 8 heads × 32 layers = 131K sequential single-thread kernel launches.
- **Estimated PPL: ~5.5-6.0** based on:
  - V quality dominates PPL (Trial 4 proved removing QJL on V improved TBQ3 by 20%)
  - q4_0 V alone gives PPL 5.53
  - K quantization barely affects PPL (only affects attention score ranking, softmax-attenuated)
- **Blocked on:** NC dequant kernel parallelization (Phase 5 optimization)

## Summary of Findings

1. **QJL hurts V quality** — the stochastic correction noise accumulates in weighted sums.
   Skipping QJL in pre-conversion improved TBQ3 by 20%.
2. **Per-block scale doesn't help** — post-rotation distribution already matches codebook.
3. **Sign-flip helps marginally** — completes SRHT but coordinates were already well-distributed.
4. **The remaining gap (9.53 vs 5.53)** is the fundamental limit of fixed Lloyd-Max centroids
   (4 or 8 levels) vs q4_0's adaptive uniform quantization (16 levels per block).
5. **Asymmetric K/V** (TBQ K + q4_0 V) is the most promising path but blocked by slow
   NC dequant kernel performance.

## Research Findings (Apr 13, 2026)

Thorough research of the TurboQuant paper, QJL reference code, PolarQuant, and community
implementations (llama.cpp #20969, #21089, 0xSero/turboquant) revealed:

1. **QJL should be dropped entirely** — every practical implementation found MSE-only > MSE+QJL.
   "QJL adds variance that softmax amplifies." k-bit MSE beats (k-1)-bit MSE + 1-bit QJL.
2. **Nobody uses TurboQuant for V** — QJL ref code uses 2-bit group quant for V, 0xSero uses
   group quant, PolarQuant keeps V in fp16. The paper doesn't explicitly recommend TBQ for V.
3. **The paper doesn't report perplexity** — only LongBench/needle-in-haystack (task-level).
   PR #21089 got TBQ4 PPL=9.046, matching our 9.53. The gap vs q4_0 is expected.
4. **Without QJL, TBQ4 gets 16 centroids** (same count as q4_0 levels) — all 4 bits go to
   Lloyd-Max indices instead of 3-bit + 1-bit QJL. Codebook optimized for N(0,1/d).
5. **K/V norm asymmetry is huge** — Qwen models have K/V ratio of 100-180x. Single codebook
   can't serve both. Asymmetric K/V is the standard approach.

## Recommended Next Steps

1. **Drop QJL entirely** — repurpose the 1 QJL bit as an additional centroid bit.
   TBQ3: 2-bit → 3-bit centroids (4→8 levels). TBQ4: 3-bit → 4-bit centroids (8→16 levels).
   Remove qjl[] from struct, gamma field, QJL PRNG code.
2. **Use q4_0 for V** — standard group quantization. TBQ K + q4_0 V via VEC kernel.
3. **Benchmark decode speed** — the fused VEC kernel is TBQ's real differentiator.
4. **Consider outlier channel handling** — mixed precision for 5-20% outlier channels.
