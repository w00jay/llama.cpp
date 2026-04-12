#pragma once

#include "ggml-common.h"

// TurboQuant CUDA device functions
// Mirrors the CPU implementation in ggml-quants.c exactly.
//
// Design: single-threaded per block (one CUDA thread quantizes one 128-element
// vector). Arrays spill to local memory (~1.5KB) which is fine for correctness.
// Phase 5 optimization: cooperative thread block with shared memory FWHT.
//
// The block_idx parameter is the block's index within the row (0-based).
// It seeds the PRNG for rotation and QJL, so CPU/GPU produce identical output.

// --- Helpers (same algorithms as CPU, __device__ versions) ---

static __device__ __forceinline__ uint64_t tbq_xorshift64_cuda(uint64_t state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

static __device__ void tbq_fwht_inplace_cuda(float * x, int d) {
    for (int stride = 1; stride < d; stride *= 2) {
        for (int i = 0; i < d; i += stride * 2) {
            for (int j = i; j < i + stride; j++) {
                float a = x[j];
                float b = x[j + stride];
                x[j]          = a + b;
                x[j + stride] = a - b;
            }
        }
    }
}

static __device__ __forceinline__ uint64_t tbq_rot_seed_cuda(int64_t block_idx) {
    return 0x9E3779B97F4A7C15ULL + (uint64_t)block_idx * 0x6C62272E07BB0142ULL;
}

static __device__ __forceinline__ uint64_t tbq_qjl_seed_cuda(int64_t block_idx, int coord) {
    return 0xCAFEBABE00000000ULL + (uint64_t)block_idx * 0x517CC1B727220A95ULL
         + (uint64_t)coord * 0x6364136223846793ULL;
}

// --- TBQ3_0: 2-bit Lloyd-Max + 1-bit QJL = 3.25 bpw ---

static __device__ void quantize_f32_tbq3_0_block(const float * __restrict__ x,
                                                   block_tbq3_0 * __restrict__ y,
                                                   int64_t block_idx) {
    const float cb[4] = { -1.335033e-01f, -4.002048e-02f, 4.002048e-02f, 1.335033e-01f };

    float tmp[QK_TBQ];

    // Step 1: norm
    float norm = 0.0f;
    for (int j = 0; j < QK_TBQ; j++) {
        norm += x[j] * x[j];
    }
    norm = sqrtf(norm);
    y->d = __float2half(norm);

    // Step 2: normalize + randomized Hadamard
    float inv_norm = (norm > 1e-10f) ? 1.0f / norm : 0.0f;
    for (int j = 0; j < QK_TBQ; j++) {
        tmp[j] = x[j] * inv_norm;
    }

    uint64_t rng = tbq_rot_seed_cuda(block_idx);
    for (int j = 0; j < QK_TBQ; j++) {
        rng = tbq_xorshift64_cuda(rng);
        if (rng & 1) tmp[j] = -tmp[j];
    }

    tbq_fwht_inplace_cuda(tmp, QK_TBQ);
    const float scale = 1.0f / sqrtf((float)QK_TBQ);
    for (int j = 0; j < QK_TBQ; j++) {
        tmp[j] *= scale;
    }

    // Step 3: nearest centroid (2-bit = 4 centroids)
    for (int j = 0; j < QK_TBQ / 4; j++) { y->idx[j] = 0; }

    float centroid_vals[QK_TBQ];
    for (int j = 0; j < QK_TBQ; j++) {
        int best = 0;
        float best_dist = fabsf(tmp[j] - cb[0]);
        for (int c = 1; c < 4; c++) {
            float dist = fabsf(tmp[j] - cb[c]);
            if (dist < best_dist) { best_dist = dist; best = c; }
        }
        y->idx[j / 4] |= (uint8_t)(best << (2 * (j % 4)));
        centroid_vals[j] = cb[best];
    }

    // Step 4: residual
    float residual[QK_TBQ];
    float gamma_sq = 0.0f;
    for (int j = 0; j < QK_TBQ; j++) {
        residual[j] = tmp[j] - centroid_vals[j];
        gamma_sq += residual[j] * residual[j];
    }
    float gamma = sqrtf(gamma_sq);
    y->gamma = __float2half(gamma);

    // Step 5: QJL sign bits
    for (int j = 0; j < QK_TBQ / 8; j++) { y->qjl[j] = 0; }

    for (int j = 0; j < QK_TBQ; j++) {
        uint64_t srng = tbq_qjl_seed_cuda(block_idx, j);
        float proj = 0.0f;
        for (int l = 0; l < QK_TBQ; l++) {
            srng = tbq_xorshift64_cuda(srng);
            float s = (srng & 1) ? 1.0f : -1.0f;
            proj += s * residual[l];
        }
        if (proj >= 0.0f) {
            y->qjl[j / 8] |= (1 << (j % 8));
        }
    }
}

// --- TBQ4_0: 3-bit Lloyd-Max + 1-bit QJL = 4.25 bpw ---

static __device__ void quantize_f32_tbq4_0_block(const float * __restrict__ x,
                                                   block_tbq4_0 * __restrict__ y,
                                                   int64_t block_idx) {
    const float cb[8] = {
        -1.902069e-01f, -1.187859e-01f, -6.682206e-02f, -2.166347e-02f,
         2.166347e-02f,  6.682206e-02f,  1.187859e-01f,  1.902069e-01f,
    };

    float tmp[QK_TBQ];

    // Step 1: norm
    float norm = 0.0f;
    for (int j = 0; j < QK_TBQ; j++) {
        norm += x[j] * x[j];
    }
    norm = sqrtf(norm);
    y->d = __float2half(norm);

    // Step 2: normalize + randomized Hadamard
    float inv_norm = (norm > 1e-10f) ? 1.0f / norm : 0.0f;
    for (int j = 0; j < QK_TBQ; j++) {
        tmp[j] = x[j] * inv_norm;
    }

    uint64_t rng = tbq_rot_seed_cuda(block_idx);
    for (int j = 0; j < QK_TBQ; j++) {
        rng = tbq_xorshift64_cuda(rng);
        if (rng & 1) tmp[j] = -tmp[j];
    }

    tbq_fwht_inplace_cuda(tmp, QK_TBQ);
    const float scale = 1.0f / sqrtf((float)QK_TBQ);
    for (int j = 0; j < QK_TBQ; j++) {
        tmp[j] *= scale;
    }

    // Step 3: nearest centroid (3-bit = 8 centroids)
    for (int j = 0; j < 3 * QK_TBQ / 8; j++) { y->idx[j] = 0; }

    float centroid_vals[QK_TBQ];
    for (int j = 0; j < QK_TBQ; j++) {
        int best = 0;
        float best_dist = fabsf(tmp[j] - cb[0]);
        for (int c = 1; c < 8; c++) {
            float dist = fabsf(tmp[j] - cb[c]);
            if (dist < best_dist) { best_dist = dist; best = c; }
        }
        int bit_pos = j * 3;
        int byte_pos = bit_pos / 8;
        int bit_off = bit_pos % 8;
        y->idx[byte_pos] |= (uint8_t)(best << bit_off);
        if (bit_off + 3 > 8) {
            y->idx[byte_pos + 1] |= (uint8_t)(best >> (8 - bit_off));
        }
        centroid_vals[j] = cb[best];
    }

    // Step 4: residual
    float residual[QK_TBQ];
    float gamma_sq = 0.0f;
    for (int j = 0; j < QK_TBQ; j++) {
        residual[j] = tmp[j] - centroid_vals[j];
        gamma_sq += residual[j] * residual[j];
    }
    y->gamma = __float2half(sqrtf(gamma_sq));

    // Step 5: QJL sign bits
    for (int j = 0; j < QK_TBQ / 8; j++) { y->qjl[j] = 0; }

    for (int j = 0; j < QK_TBQ; j++) {
        uint64_t srng = tbq_qjl_seed_cuda(block_idx, j);
        float proj = 0.0f;
        for (int l = 0; l < QK_TBQ; l++) {
            srng = tbq_xorshift64_cuda(srng);
            float s = (srng & 1) ? 1.0f : -1.0f;
            proj += s * residual[l];
        }
        if (proj >= 0.0f) {
            y->qjl[j / 8] |= (1 << (j % 8));
        }
    }
}
