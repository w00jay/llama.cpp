# TurboQuant for llama.cpp — Implementation Plan

## Goal

Implement TurboQuant KV cache quantization in llama.cpp so that running
`llama-cli --cache-type-k tbq3 --cache-type-v tbq3 -fa ...` compresses the
KV cache ~5x at 3 bits/coordinate with negligible quality loss, enabling
longer contexts and larger models on consumer GPUs.

---

## Background

**TurboQuant** (ICLR 2026, Zandieh et al.) is a training-free, online KV cache
quantization algorithm with near-optimal distortion. It composes three techniques:

1. **Random rotation** — multiply each KV vector by a fixed orthogonal matrix Pi
   to spread information uniformly (eliminates outliers, induces a known Beta
   distribution on coordinates).
2. **Lloyd-Max scalar quantization** — each rotated coordinate is independently
   quantized using precomputed optimal centroids for the Beta/Gaussian distribution.
3. **QJL residual correction** — the quantization residual is projected through a
   random Gaussian matrix S and reduced to sign bits, providing an unbiased 1-bit
   correction that eliminates inner-product bias.

For `TurboQuant_prod` at b bits/coordinate:
- (b-1) bits: Lloyd-Max index per coordinate
- 1 bit: QJL sign bit per coordinate
- 1 float16: vector norm
- **Total: b*d + 16 bits per vector** (e.g., 3*128+16 = 400 bits for d=128, b=3)

### Key references
- TurboQuant paper: arxiv.org/abs/2504.19874
- PolarQuant paper: arxiv.org/abs/2502.02617
- QJL paper + reference CUDA code: github.com/amirzandieh/QJL
- PolarQuant reference Python code: github.com/ericshwu/PolarQuant
- llama.cpp PR #21089 (CPU-only TBQ attempt, not merged): github.com/ggml-org/llama.cpp/pull/21089

### Existing llama.cpp infrastructure we leverage
- KV cache already supports quantized types (Q4_0, Q5_0, Q8_0, etc.)
- Hadamard pre-rotation already exists for quantized KV (`attn_rot_k/v` in llama-kv-cache.cpp:289-299)
- `ggml_set_rows` quantizes on write; flash attention dequantizes on read
- CUDA flash attention dispatches per-type via C++ templates (fattn-common.cuh)
- CLI flags `--cache-type-k`/`--cache-type-v` already wired

---

## Architecture Decision: Rotation Strategy

llama.cpp already uses **Walsh-Hadamard rotation** (O(d log d), no matrix storage)
for KV cache quantization. TurboQuant's paper specifies a random orthogonal matrix
(O(d^2), requires storing d*d floats).

**Decision: Use Hadamard + random diagonal signs** (the "randomized Hadamard transform").
This is the standard practical substitute — used by QuiP#, QUIP, and already partially
implemented in llama.cpp. It gives the same distributional properties (uniform on
sphere after projection) at O(d log d) cost and zero matrix storage.

This means TurboQuant's rotation step piggybacks on the existing `attn_rot_k`/`attn_rot_v`
infrastructure with a minor modification: add a random sign-flip vector (d random
+1/-1 values, seeded deterministically) applied before the Hadamard.

---

## Phases

### Phase 0: Standalone Math Validation (no llama.cpp changes)

**Goal:** Validate the core algorithms in isolation before touching ggml.

**Deliverables:**
- `test/test_tbq_math.cpp` — standalone C++ test (compiles independently)

**Tasks:**
1. Implement Lloyd-Max codebook solver for the Gaussian distribution N(0, 1/d)
   - For b-1 = 1,2,3 bits (i.e., 2, 4, 8 centroids)
   - Use iterative Lloyd-Max on the known PDF: f(x) = sqrt(d/(2*pi)) * exp(-d*x^2/2)
   - Validate against known values: b=1 centroids at +/-sqrt(2/(pi*d))
   - Output: static codebook tables (small, can be constexpr)

2. Implement randomized Hadamard transform
   - In-place FWHT (Fast Walsh-Hadamard Transform) on a float vector of length d
   - Random sign-flip vector: d values of +1/-1, seeded with (layer_id, head_id)
   - Apply signs, then FWHT, then scale by 1/sqrt(d)
   - Validate: rotated unit vectors should have coordinates ~N(0, 1/d)

3. Implement TurboQuant_prod quantize/dequantize for a single vector
   - Quantize: sign-flip → FWHT → nearest centroid (b-1 bits) → residual → QJL sign bits
   - Dequantize: centroid lookup → inverse FWHT → inverse sign-flip → add QJL reconstruction
   - QJL matrix: use a seeded PRNG to generate S rows on-the-fly (avoids storing d*d)

4. Implement inner product estimation (the attention-critical path)
   - Given quantized key and full-precision query:
     - Rotate query: q_rot = FWHT(sign_flip(q))
     - MSE part: dot(q_rot, centroids[idx])  — O(d)
     - QJL part: (sqrt(pi/2)/d) * gamma * dot(S*q, qjl_signs)  — O(d) if S*q precomputed
   - Validate: estimate vs true inner product over many random vectors

5. Test roundtrip accuracy at 2-bit, 3-bit, 4-bit for d=128
   - Measure MSE, inner product correlation, max error
   - Compare against Q4_0 baseline (uniform quantization with scale)

**Exit criteria:** Inner product correlation > 0.99 at 3-bit for d=128.

---

### Phase 1: GGML Type Registration (CPU-only, no CUDA)

**Goal:** Register TBQ3_0 and TBQ4_0 as new GGML types with CPU quantize/dequantize.

**Files to modify:**

#### 1.1 Block struct definition
**File:** `llama.cpp/ggml/src/ggml-common.h`

```c
// TurboQuant block: d=128 values per block
// TBQ3_0: 3 bits/coord = 2-bit Lloyd-Max index + 1-bit QJL sign
//   128 * 2 bits = 256 bits = 32 bytes (indices)
//   128 * 1 bit  = 128 bits = 16 bytes (QJL signs)
//   1 * fp16 = 2 bytes (norm)
//   1 * fp16 = 2 bytes (QJL residual norm / gamma)
//   Total: 52 bytes per 128 elements
#define QK_TBQ 128
typedef struct {
    ggml_half d;              // vector norm ||x||
    ggml_half gamma;          // residual norm ||r||
    uint8_t   idx[QK_TBQ/4]; // 2-bit centroid indices, packed 4 per byte (32 bytes)
    uint8_t   qjl[QK_TBQ/8]; // 1-bit QJL signs, packed 8 per byte (16 bytes)
} block_tbq3_0;
// 2 + 2 + 32 + 16 = 52 bytes for 128 values = 3.25 bits/value

typedef struct {
    ggml_half d;              // vector norm ||x||
    ggml_half gamma;          // residual norm ||r||
    uint8_t   idx[3*QK_TBQ/8]; // 3-bit centroid indices, packed (48 bytes)
    uint8_t   qjl[QK_TBQ/8];   // 1-bit QJL signs (16 bytes)
} block_tbq4_0;
// 2 + 2 + 48 + 16 = 68 bytes for 128 values = 4.25 bits/value
```

Note: block size QK_TBQ=128 matches typical head dimensions (Llama, Mistral, Qwen).
This is larger than the standard QK=32 block size used by Q4_0 etc., but required
because TurboQuant operates on whole head-dimension vectors.

#### 1.2 Enum registration
**File:** `llama.cpp/ggml/include/ggml.h`

Add after GGML_TYPE_Q1_0 = 41:
```c
GGML_TYPE_TBQ3_0  = 42,
GGML_TYPE_TBQ4_0  = 43,
GGML_TYPE_COUNT   = 44,
```

#### 1.3 Type traits table
**File:** `llama.cpp/ggml/src/ggml.c` — `type_traits[]` array (~line 609)

Add entries with:
- `.type_name = "tbq3_0"` / `"tbq4_0"`
- `.blck_size = QK_TBQ` (128)
- `.type_size = sizeof(block_tbq3_0)` / `sizeof(block_tbq4_0)`
- `.is_quantized = true`
- `.to_float = dequantize_row_tbq3_0` / `dequantize_row_tbq4_0`
- `.from_float_ref = quantize_row_tbq3_0_ref` / `quantize_row_tbq4_0_ref`

#### 1.4 CPU quantize/dequantize kernels
**New file:** `llama.cpp/ggml/src/ggml-cpu/tbq-quants.h` + `tbq-quants.cpp`

Functions to implement:
- `quantize_row_tbq3_0_ref(const float * x, void * y, int64_t k)`
  - For each block of 128 floats:
    1. Compute and store norm d = ||x||
    2. Normalize: x_hat = x / d
    3. Apply randomized Hadamard: sign-flip + FWHT
    4. For each coordinate: find nearest 2-bit centroid, store 2-bit index
    5. Compute residual r = rotated - centroid_values
    6. Store gamma = ||r||
    7. QJL: for each coordinate, generate S row from seeded PRNG, compute dot(S_row, r), store sign bit
- `dequantize_row_tbq3_0(const void * x, float * y, int64_t k)`
  - For each block:
    1. Look up centroid values from 2-bit indices
    2. QJL reconstruct: for each coord, generate S row, multiply by sign * (sqrt(pi/2)/d) * gamma
    3. Add to centroid values
    4. Apply inverse FWHT + inverse sign-flip
    5. Scale by norm d
- Same for tbq4_0 but with 3-bit indices (8 centroids)

**Shared state:** The sign-flip vectors and PRNG seeds need to be deterministic.
Use a simple hash of (block_index) as the seed. The Hadamard transform is
parameterless. The codebook centroids are compile-time constants.

**Critical detail:** The QJL matrix S is never stored. Each row of S is generated
on-the-fly from a seeded xorshift PRNG: `s_ij = (prng_next() & 1) ? 1.0f : -1.0f`.
This is a Rademacher matrix (random +/-1), which is a valid JL projection and
much faster to generate than Gaussian.

#### 1.5 CPU vec_dot kernel (for non-flash attention path)
**File:** `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` — `type_traits_cpu[]`

The vec_dot computes dot(dequantized_block, float_block). For TBQ this is:
1. Dequantize the TBQ block to float (reuse dequantize function)
2. Standard dot product

Later optimization: compute the dot in the rotated domain to avoid inverse FWHT.

#### 1.6 Enable as KV cache type
**File:** `llama.cpp/common/arg.cpp` — `kv_cache_types` vector (line 383)

Add `GGML_TYPE_TBQ3_0` and `GGML_TYPE_TBQ4_0` to the vector.

#### 1.7 Handle block size mismatch
**File:** `llama.cpp/src/llama-context.cpp` — validation around line 2964

The existing validation checks that `n_embd_head_k % blck_size == 0`. Since
QK_TBQ=128 and typical head dimensions are 128, this should pass for most models.
Need to verify and add a clear error message for models with non-128 head dims.

**Exit criteria:** `llama-cli --cache-type-k tbq3_0 -fa -m model.gguf -p "test"` runs
on CPU and produces coherent output. Perplexity within 0.5 of Q4_0 KV cache baseline.

---

### Phase 2: CUDA Quantization Kernel (write path)

**Goal:** Quantize KV vectors on GPU when writing to cache.

**File:** `llama.cpp/ggml/src/ggml-cuda/set-rows.cu`

#### 2.1 Device-side TBQ quantize function

Add a CUDA kernel `k_set_rows_tbq3` that:
1. Each thread block processes one 128-element vector
2. Compute norm via parallel reduction
3. Normalize
4. Apply sign-flip (precomputed sign vector in constant memory)
5. Apply FWHT in shared memory (7 butterfly stages for d=128)
6. Each thread quantizes its coordinates to 2-bit centroids (4 centroids in registers)
7. Compute residual
8. Parallel reduction for residual norm
9. QJL: each thread generates its S row element from a fast PRNG, computes sign of dot
10. Pack bits and write to global memory

**Memory layout considerations:**
- Sign-flip vectors: stored in device constant memory (128 floats = 512 bytes per head config)
- Codebook centroids: 4 or 8 floats in constant memory
- FWHT: entirely in shared memory (128 floats = 512 bytes)

#### 2.2 Dispatch in set-rows.cu

Add to `ggml_cuda_op_set_rows`:
```cpp
} else if (dst->type == GGML_TYPE_TBQ3_0) {
    k_set_rows_tbq3<<<...>>>(src, dst, ...);
} else if (dst->type == GGML_TYPE_TBQ4_0) {
    k_set_rows_tbq4<<<...>>>(src, dst, ...);
}
```

**Exit criteria:** GPU quantization produces identical results to CPU reference.

---

### Phase 3: CUDA Flash Attention Kernels (read path)

**Goal:** Dequantize TBQ-encoded K/V during flash attention on GPU.

**Files:**
- `llama.cpp/ggml/src/ggml-cuda/fattn-common.cuh`
- `llama.cpp/ggml/src/ggml-cuda/fattn.cu`

#### 3.1 K dequantize + dot product for attention scores

In `fattn-common.cuh`, add a `get_vec_dot_KQ` specialization for TBQ3_0:
- Load the TBQ block (52 bytes) into shared memory
- Look up centroids from indices
- Apply inverse FWHT to the centroid-reconstructed vector
- Scale by norm
- Dot product with query vector
- Add QJL correction term

**Optimization:** Instead of full dequant + dot, compute in rotated domain:
1. Pre-rotate query with FWHT (done once, stored in shared memory)
2. Dot rotated query directly with centroid values — no inverse FWHT needed
3. Add QJL term: (sqrt(pi/2)/d) * gamma * dot(S*q_rot, qjl_signs)

This reduces the per-key cost from O(d log d) to O(d).

#### 3.2 V dequantize for value accumulation

In `fattn-common.cuh`, add a `get_dequantize_V` specialization:
- Full dequantize is needed for V (values are weighted-summed, not dot-producted)
- Load block → centroid lookup → QJL reconstruction → inverse FWHT → scale

#### 3.3 Type allowlist

In `fattn.cu`, add TBQ3_0 and TBQ4_0 to the switch statement that selects
flash attention kernels (around line 381).

#### 3.4 Fallback path

If the MMA/WMMA kernel doesn't support TBQ natively, the existing fallback
pre-converts to FP16 via the dequantize path (fattn-common.cuh:965-1010).
This works automatically if `to_float` is implemented. Initial implementation
can rely on this fallback; fused kernels are an optimization for later.

**Exit criteria:** Flash attention with TBQ3_0 K/V produces correct output on GPU.
Benchmark: measure tokens/sec vs Q4_0 KV cache and FP16 KV cache.

---

### Phase 4: Validation and Benchmarking

**Goal:** Prove quality and performance.

#### 4.1 Perplexity evaluation
- Run `llama-perplexity` on wikitext-2 with:
  - `--cache-type-k f16 --cache-type-v f16` (baseline)
  - `--cache-type-k q4_0 --cache-type-v q4_0` (existing quantized baseline)
  - `--cache-type-k tbq3_0 --cache-type-v tbq3_0` (our implementation)
  - `--cache-type-k tbq4_0 --cache-type-v tbq4_0`
- Models: Llama-3.1-8B-Instruct (primary), Mistral-7B, Qwen2-7B

#### 4.2 Long-context evaluation
- Needle-in-haystack at various context lengths
- Compare memory usage (peak GPU memory) vs baseline

#### 4.3 Performance benchmarking
- `llama-bench` with various batch sizes and context lengths
- Measure: tokens/sec (prompt processing and generation), peak memory

#### 4.4 Correctness tests
- Bit-exact CPU vs GPU quantization roundtrip
- Attention score comparison: TBQ vs FP16 reference (correlation > 0.99)

**Exit criteria:** Perplexity within 0.3 of FP16 at 3-bit. Memory reduction > 4x.
No correctness regressions on standard benchmarks.

---

### Phase 5: Optimization and Polish

#### 5.1 Fused CUDA kernels
- Fused FWHT + quantize kernel (eliminate shared memory round-trip)
- Fused dequant + dot kernel for K (rotated-domain optimization from 3.1)
- Benchmark and compare to fallback path

#### 5.2 Asymmetric K/V quantization
- Research suggests K benefits more from TBQ than V
- Support `--cache-type-k tbq3_0 --cache-type-v q4_0` (mix types)
- This already works via llama.cpp's separate K/V type configuration

#### 5.3 Metal backend (Apple Silicon)
- Port the FWHT and quantize/dequant kernels to Metal shaders
- Similar structure to CUDA but using Metal threadgroup memory

#### 5.4 Documentation
- Update llama.cpp docs with TBQ usage instructions
- Add to README model compatibility table
- Document memory savings and quality trade-offs

---

## File Change Summary

| File | Change | Phase |
|------|--------|-------|
| `test/test_tbq_math.cpp` | **New** — standalone algorithm validation | 0 |
| `ggml/include/ggml.h` | Add GGML_TYPE_TBQ3_0, TBQ4_0 enum values | 1 |
| `ggml/src/ggml-common.h` | Add block_tbq3_0, block_tbq4_0 struct defs | 1 |
| `ggml/src/ggml.c` | Add type_traits[] entries | 1 |
| `ggml/src/ggml-cpu/tbq-quants.h` | **New** — CPU quantize/dequant declarations | 1 |
| `ggml/src/ggml-cpu/tbq-quants.cpp` | **New** — CPU quantize/dequant + FWHT + codebook | 1 |
| `ggml/src/ggml-cpu/ggml-cpu.c` | Add type_traits_cpu[] entries | 1 |
| `common/arg.cpp` | Add TBQ types to kv_cache_types | 1 |
| `ggml/src/ggml-cuda/set-rows.cu` | Add TBQ quantize dispatch + CUDA kernel | 2 |
| `ggml/src/ggml-cuda/tbq-quants.cuh` | **New** — CUDA TBQ device functions | 2 |
| `ggml/src/ggml-cuda/fattn-common.cuh` | Add TBQ K dequant+dot and V dequant | 3 |
| `ggml/src/ggml-cuda/fattn.cu` | Add TBQ to type allowlist | 3 |

---

## Dependency Graph

```
Phase 0 (standalone math)
    |
    v
Phase 1 (GGML types + CPU kernels) ──> Phase 4.4 (correctness tests)
    |
    ├──> Phase 2 (CUDA write path)
    |        |
    |        v
    └──> Phase 3 (CUDA read/FA path) ──> Phase 4 (full benchmarks)
                                              |
                                              v
                                         Phase 5 (optimization)
```

Phases 2 and 3 can be worked in parallel once Phase 1 is complete.
Phase 4 benchmarks require both Phase 2 and Phase 3.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Block size 128 incompatible with some models | Medium | Blocks those models | Check n_embd_head_k at init; fall back to Q4_0 with clear error |
| QJL PRNG-based S matrix too slow on CPU | Low | CPU perf hit | Use Rademacher matrix (+/-1) instead of Gaussian; fast xorshift |
| Hadamard rotation differs from paper's random orthogonal | Low | Slight quality delta | Randomized Hadamard is theoretically equivalent; validate empirically |
| Large block size (128) hurts partial-sequence updates | Medium | Wastes cache space | llama.cpp already handles non-contiguous KV via set_rows; block alignment may waste up to 127 positions at boundaries |
| Upstream rejection (llama.cpp AI policy) | High if upstreaming | Can't merge PR | Keep as local fork; focus on correctness not upstreaming |

---

## Open Questions (resolve during Phase 0)

1. **Rademacher vs Gaussian for QJL matrix S:** Paper uses Gaussian, but Rademacher
   (+/-1) is faster and has the same JL guarantee. Validate empirically that quality
   is equivalent at d=128.

2. **Codebook precision:** Should centroids be stored as float32 or float16? At d=128,
   the quantization is coarse enough that FP16 centroids should suffice.

3. **Seed strategy for QJL PRNG:** Per-block seed = hash(layer, head, position)?
   Must ensure determinism for correctness but sufficient randomness for JL guarantee.

4. **Interaction with existing Hadamard rotation:** llama.cpp already applies Hadamard
   to K/V for quantized types. Should TBQ disable `attn_rot_k/v` and handle rotation
   internally, or layer on top of it? Internal handling is cleaner — avoids double rotation.
