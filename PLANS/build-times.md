# TBQ Build Times Log

Machine: Dell R730xd, 56 cores, 125GB RAM, CUDA 12.5
GPU target: sm_86 only (RTX 3090)

## Build Times

| Date | Change | -j | Time | Notes |
|------|--------|-----|------|-------|
| Apr 12 | Phase 3 (convert.cu + fattn.cu) | 56 | ~2 min | No VEC templates |
| Apr 12 | Fused VEC kernel (2 TBQ cases) | 56 | OOM killed | fattn.cu template explosion |
| Apr 12 | Fused VEC kernel (2 TBQ cases) | 4 | OOM killed | Still too much memory |
| Apr 12 | Fused VEC kernel (2 TBQ cases) | 1 | OOM killed | Single nvcc exceeds memory?! |
| Apr 13 | Reconfigure sm_86 only + rebuild | 4 | ~20 min | Succeeded after dropping sm_61 |
| Apr 13 | Sign-flip (tbq-quants.cuh + fattn-common.cuh) | 4 | ~15 min | Only CUDA files changed |
| Apr 13 | Per-block scale (struct change + all CUDA) | 4 | ~20 min | Full rebuild from ggml-common.h |
| Apr 13 | Asymmetric QJL (fattn-common.cuh only) | 4 | ~12 min | Only fattn templates recompile |
| Apr 13 | Parallel NC dequant (convert.cu only) | 4 | ~5 min | No fattn template change |
| Apr 13 | VEC guard fix (fattn.cu) | 4 | ~12 min | fattn.cu recompile |
| Apr 13 | +4 VEC cases (TBQ×q4_0/q8_0) in fattn.cu | 2 | 2h+ (ongoing) | ptxas uses 36GB, 1 core |

## Root Cause: All TBQ VEC Templates in fattn.cu

llama.cpp already splits VEC templates into **49 separate .cu files** in
`template-instances/fattn-vec-instance-{type_k}-{type_v}.cu`. Each compiles
independently (~5 min) and in parallel. Our TBQ cases were added directly to
`fattn.cu` instead of using this system, forcing a single monolithic compilation.

## Fix: Move TBQ Cases to Instance Files

Create 6 files in `template-instances/`:
```
fattn-vec-instance-tbq3_0-f16.cu
fattn-vec-instance-tbq3_0-q4_0.cu
fattn-vec-instance-tbq3_0-q8_0.cu
fattn-vec-instance-tbq4_0-f16.cu
fattn-vec-instance-tbq4_0-q4_0.cu
fattn-vec-instance-tbq4_0-q8_0.cu
```

Each file: ~7 lines, one `DECL_FATTN_VEC_CASE` invocation.
Register in CMakeLists.txt. Remove cases from fattn.cu.

**Expected improvement:** 2h+ single-file build → 6 × ~5 min parallel builds.

## Other Optimization Notes

- **ptxas is the real bottleneck**: 99.9% CPU, 36GB RAM, runs for hours on large .cu files
- **NVCC flags not currently used**: `--threads` (intra-file parallelism, CUDA 11.6+)
- Use -j4 for non-fattn changes, -j2 for fattn.cu changes
- sm_86 only (dropping P40's sm_61) halves compile time per architecture
