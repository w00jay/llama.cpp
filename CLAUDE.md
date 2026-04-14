# Llamaq.cpp — TurboQuant for llama.cpp

## Project Overview

This project implements TurboQuant KV cache quantization in llama.cpp.
TurboQuant compresses the KV cache ~5x at 3 bits/coordinate using randomized
Hadamard rotation + Lloyd-Max scalar quantization + QJL residual correction.

This is a fork of ggml-org/llama.cpp. Work happens on the `turboquant` branch.
Upstream is tracked via the `upstream` remote for rebasing.

## Plan

Read [PLAN.md](PLAN.md) before starting any work. It contains the full
implementation plan with phases, file targets, struct layouts, and algorithms.

## Workflow (follow for each phase)

1. **Read PLAN.md** — understand the phase goals, files to modify, and exit criteria
2. **Read existing code** — study the reference files and patterns before writing
3. **Implement** — write the code, mirroring existing conventions
4. **Build** — `cmake --build build -j$(nproc)` and fix any errors
5. **Test** — run relevant tests (`test-backend-ops`, `test-quantize-fns`, etc.)
6. **Fix failures** — diagnose, fix, rebuild, retest until all pass
7. **Update docs** — update Progress in this file + Phase Results in PLAN.md
8. **Commit** — descriptive message with `turboquant:` prefix
9. **Push** — `git push origin turboquant`

## Key Architecture Decisions

- **Rotation:** External WHT (llama.cpp's `attn_rot_k/v`) + internal random sign-flip
  (SRHT). The sign-flip makes coordinates ~N(0, 1/d) for optimal codebook fit.
  No FWHT inside TBQ — external rotation handles that.
- **QJL:** Rademacher (+/-1) generated on-the-fly from seeded PRNG. Used for K
  attention scores (fused VEC dot), **skipped for V** (QJL noise hurts weighted sums).
- **Per-block scale:** fp16 scale factor adapts codebook to actual coordinate variance.
- **Block size:** QK_TBQ = 128 (matches head dimension of Llama/Mistral/Qwen).
  Hard constraint — models with non-128 head dims won't work.
- **Types:** GGML_TYPE_TBQ3_0 (3.25 bpw, 52 bytes/128 elem, 8 centroids) and
  GGML_TYPE_TBQ4_0 (4.25 bpw, 68 bytes/128 elem, 16 centroids).
  Each block stores: fp16 norm, fp16 per-block scale, packed centroid indices.

## Build

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j4   # -j2 for fattn.cu changes (high memory usage)
```

**Build time warning:** fattn.cu VEC template instantiations are extremely slow
(~1-2 hours). Each `FATTN_VEC_CASE` adds ~20 min compile time. ptxas can use
36GB+ RAM. See [PLANS/build-times.md](PLANS/build-times.md) for details.
TBQ VEC cases should ideally be split into separate `template-instances/` files.

## Test

```bash
# Phase 0 standalone math validation (no cmake needed)
g++ -std=c++17 -O2 -o test-tbq-math tests/test-tbq-math.cpp -lm && ./test-tbq-math

# Phase 1+ integration test
./build/bin/llama-cli --cache-type-k tbq3_0 --cache-type-v tbq3_0 -fa -m model.gguf -p "Hello"

# Perplexity benchmark
./build/bin/llama-perplexity --cache-type-k tbq3_0 --cache-type-v tbq3_0 -fa -m model.gguf -f wikitext-2-raw/wiki.test.raw
```

## Progress

- **Phase 0** — DONE. Standalone math validation: 7/7 tests pass. All core algorithms
  (FWHT, Lloyd-Max, TBQ quantize/dequant, IP estimation) validated. Key finding:
  per-pair IP correlation at 3-bit is ~0.92 (matches theory); real quality emerges
  from softmax averaging in attention.
- **Phase 1** — DONE. GGML types TBQ3_0 (52 bytes/128 elem) and TBQ4_0 (68 bytes/128 elem)
  registered. CPU quantize/dequantize/vec_dot implemented. Builds clean, roundtrip verified.
- **Phase 2** — DONE. CUDA write path: `tbq-quants.cuh` device functions + custom
  `k_set_rows_tbq` kernel. All SET_ROWS tests pass (GPU matches CPU bit-exact).
- **Phase 3** — DONE. CUDA read path (flash attention). TBQ K/V dequantized to fp16 via
  contiguous and non-contiguous (NC) dequant kernels in convert.cu, routed to MMA/TILE
  kernels via the general FA path in fattn.cu. 4/4 TBQ FLASH_ATTN_EXT tests pass on GPU,
  2838 total FA tests OK, 0 FAIL. Files: tbq-quants.cuh, convert.cu, fattn.cu.
- **Phase 4** — Skipped (benchmarking deferred until fused kernel proves TBQ's value).
- **Phase 5** — IN PROGRESS. Key results:
  - Fused VEC kernel: `vec_dot_fattn_vec_KQ_tbq3_0/4_0` computes attention scores
    directly from TBQ K without dequant (decode path)
  - Removed internal FWHT rotation; uses llama.cpp external WHT + internal sign-flip (SRHT)
  - Added per-block adaptive scale (fp16 `s` field in struct)
  - **Asymmetric QJL**: skip QJL in V dequant (noise hurts weighted sums)
  - Parallelized NC dequant kernel (128 threads per block)
  - PPL (10-chunk, Llama-3.1-8B-Q6K): TBQ4=9.53, TBQ3=13.54 (vs f16=5.46, q4_0=5.53)
  - Asymmetric K/V (TBQ K + q4_0 V) in progress — expected PPL ~5.5-6.0
  - See [PLANS/quality-trials.md](PLANS/quality-trials.md) for full experiment log

## Key Files

### Upstream files we modify
- `ggml/include/ggml.h` — type enum (add TBQ3_0, TBQ4_0)
- `ggml/src/ggml-common.h` — block struct definitions
- `ggml/src/ggml.c` — type_traits[] table + ggml_quantize_chunk
- `ggml/src/ggml-quants.c` — CPU quantize/dequantize implementations
- `ggml/src/ggml-quants.h` — CPU quantize/dequantize declarations
- `ggml/src/ggml-cpu/ggml-cpu.c` — type_traits_cpu[] table
- `ggml/src/ggml-cpu/quants.c` — CPU from_float + vec_dot wrappers
- `ggml/src/ggml-cpu/quants.h` — CPU wrapper declarations
- `ggml/src/ggml-cuda/set-rows.cu` — GPU quantize-on-write dispatch
- `ggml/src/ggml-cuda/ggml-cuda.cu` — SET_ROWS type allowlist
- `ggml/src/ggml-cuda/fattn-common.cuh` — flash attention K/V dequant (Phase 3)
- `ggml/src/ggml-cuda/fattn.cu` — type allowlist for FA kernels (Phase 3)
- `common/arg.cpp` — kv_cache_types vector
- `tests/test-backend-ops.cpp` — TBQ SET_ROWS test cases

### New files we create
- `ggml/src/ggml-cuda/tbq-quants.cuh` — CUDA TBQ device functions (quantize, dequant, PRNG)
- `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-tbq*.cu` — VEC template instances (TODO)
- `tests/test-tbq-math.cpp` — standalone algorithm tests (Phase 0, all passing)
- `PLANS/quality-trials.md` — PPL experiment log with all trials and findings
- `PLANS/build-times.md` — CUDA build time tracking and optimization notes

### Reference files (read these to understand patterns)
- `ggml/src/ggml-common.h:177-200` — block_q1_0, block_q4_0 struct patterns
- `ggml/src/ggml.c:609-915` — type_traits[] table (copy pattern from Q4_0)
- `ggml/src/ggml-cpu/ggml-cpu.c:207-402` — type_traits_cpu[] table
- `ggml/src/ggml-cuda/set-rows.cu:7-70` — k_set_rows_quant template pattern
- `ggml/src/ggml-cuda/fattn-common.cuh:580-622` — get_vec_dot_KQ / get_dequantize_V dispatch
- `src/llama-kv-cache.cpp:289-323` — existing Hadamard rotation setup

## Code Style

Follow llama.cpp conventions (see llama.cpp/CONTRIBUTING.md):
- snake_case everywhere
- Simple C-style patterns, minimal STL
- 4-space indent, no trailing whitespace
- Prefix names with module: `tbq_`, `block_tbq3_0`, etc.
- Keep CUDA kernels self-contained with device functions in .cuh files

## Algorithm Quick Reference (current implementation — TBQ-MSE, no QJL)

### Quantize (set_rows CUDA kernel):
1. `d = ||x||; x_hat = x / d` — normalize (external WHT rotation already applied by llama.cpp)
2. `x_sf = sign_flip(x_hat)` — per-block Rademacher diagonal (completes SRHT)
3. `s = rms(x_sf) / sqrt(1/d)` — per-block adaptive scale
4. `x_scaled = x_sf / s` — scale to match codebook distribution N(0, 1/d)
5. For each coord j: `idx[j] = nearest_centroid(x_scaled[j])` — 8 or 16 Lloyd-Max centroids
6. Norm correction: `d_corrected = d / ||recon||` — compensate for quantization norm change
7. Store: `{d_corrected, s, idx[128]}`

### Dequantize (NC kernel, for fp16 pre-conversion):
1. `val[j] = s * centroid[idx[j]]` — scaled centroid lookup
2. `val[j] = sign_flip_undo(val[j])` — undo Rademacher diagonal
3. `y[j] = d * val[j]` — scale by corrected norm

### Fused K·Q dot product (VEC kernel, decode path):
1. `q_sf = sign_flip(q)` — sign-flip query to match K's quantized domain
2. `score = d * s * sum_j(centroid[idx[j]] * q_sf[j])` — direct dot, no dequant

## Results (10-chunk wikitext-2, Llama-3.1-8B-Q6K, RTX 3090)

| Config | bpw | PPL |
|--------|-----|-----|
| f16 | 16 | 5.46 |
| q4_0 | ~4.5 | 5.53 |
| TBQ4_0 | 4.25 | 8.93 |
| TBQ3_0 | 3.25 | 9.53 |

See [PLANS/quality-trials.md](PLANS/quality-trials.md) for the full experiment log.

## Key Findings

1. **QJL hurts more than it helps** — every practical implementation found MSE-only
   (no QJL) outperforms MSE+QJL. QJL adds variance that softmax amplifies.
2. **Nobody uses TurboQuant for V** — all successful implementations use standard
   group quantization for V (values). TBQ is best suited for K (keys) only.
3. **TBQ3 at 3.25 bpw** fills a gap where llama.cpp has no good option between
   q2_K (~2.6 bpw) and q4_0 (~4.5 bpw).
4. **The remaining PPL gap** (8.93 vs 5.53 for TBQ4 vs q4_0) is fundamental:
   fixed Lloyd-Max codebook vs per-block adaptive scale.