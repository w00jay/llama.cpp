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
- **Types:** GGML_TYPE_TBQ3_0 (3.38 bpw, 54 bytes/128 elem) and GGML_TYPE_TBQ4_0
  (4.38 bpw, 70 bytes/128 elem). Includes norm, gamma, scale, indices, QJL signs.

## Build

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86  # RTX 3090 only
cmake --build build -j4   # -j2 for fattn.cu changes, never higher (OOM risk)
```

**Build time warning:** fattn.cu VEC template instantiations are extremely slow.
Each `FATTN_VEC_CASE` takes ~20+ min to compile. ptxas can use 36GB+ RAM.
TBQ VEC cases should go in `template-instances/fattn-vec-instance-tbq*.cu` files
(separate compilation units) not inline in fattn.cu. See [PLANS/build-times.md](PLANS/build-times.md).

Dell box (192.168.1.91): use GPU UUID for 3090:
```bash
export CUDA_VISIBLE_DEVICES=GPU-d5346770-8aa7-8069-e875-bf874014dfaa
export LD_LIBRARY_PATH=build/bin
```

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

## Algorithm Quick Reference (current implementation)

### Quantize (set_rows path, b=3 example, d=128):
1. `d_norm = ||x||; x_hat = x / d_norm` — normalize (external WHT already applied)
2. `x_sf = sign_flip(x_hat)` — per-block Rademacher diagonal (SRHT completion)
3. `s = rms(x_sf) / sqrt(1/d)` — per-block adaptive scale
4. `x_scaled = x_sf / s`
5. For each coord j: `idx[j] = nearest_centroid_2bit(x_scaled[j])` — 4 centroids for N(0,1/d)
6. `r = x_scaled - centroids[idx]` — residual in scaled domain
7. `gamma = ||r||`
8. For each coord j: `qjl[j] = sign(dot(S_row_j, r))` — Rademacher PRNG for S rows
9. Store: `{d_norm, gamma, s, idx[128], qjl[128]}`

### Dequantize (V path — no QJL, for weighted sums):
1. `val[j] = s * centroids[idx[j]]` — scaled centroid lookup
2. `val[j] = sign_flip_undo(val[j])` — undo Rademacher diagonal
3. `y[j] = d_norm * val[j]`

### Fused K·Q dot product (VEC kernel, decode path — with QJL):
1. `q_sf = sign_flip(q)` — sign-flip query to match K domain
2. `dot_centroid = sum_j(s * centroid[idx[j]] * q_sf[j])`
3. `dot_qjl = sum_j(qjl_sign[j] * dot(S_row_j, q_sf))` — QJL IP correction
4. `score = d_norm * (dot_centroid + sqrt(pi/2)/d * gamma * dot_qjl)`


<claude-mem-context>
# Recent Activity

<!-- This section is auto-generated by claude-mem. Edit content outside the tags. -->

### Apr 11, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #330 | 8:40 PM | ✅ | Comprehensive TurboQuant implementation plan created defining 6-phase integration roadmap | ~1389 |
</claude-mem-context>