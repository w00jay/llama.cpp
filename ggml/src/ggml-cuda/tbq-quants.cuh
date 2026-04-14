#pragma once

#include "ggml-common.h"

// TurboQuant-MSE CUDA device functions
// All bits go to Lloyd-Max centroids (no QJL). Simple and fast.

// --- Helpers ---

static __device__ __forceinline__ uint64_t tbq_xorshift64_cuda(uint64_t state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

static __device__ __forceinline__ uint64_t tbq_rot_seed_cuda(int64_t block_idx) {
    return 0x9E3779B97F4A7C15ULL + (uint64_t)block_idx * 0x6C62272E07BB0142ULL;
}

// Codebook lookup helpers
static __device__ __forceinline__ float tbq_cb8(int idx) {
    constexpr float cb[8] = {
        -1.902069e-01f, -1.187859e-01f, -6.682206e-02f, -2.166347e-02f,
         2.166347e-02f,  6.682206e-02f,  1.187859e-01f,  1.902069e-01f,
    };
    return cb[idx];
}

static __device__ __forceinline__ float tbq_cb16(int idx) {
    constexpr float cb[16] = {
        -2.4159188e-01f, -1.8294797e-01f, -1.4308883e-01f, -1.1110441e-01f,
        -8.3350925e-02f, -5.8095365e-02f, -3.4327652e-02f, -1.1358431e-02f,
         1.1358431e-02f,  3.4327652e-02f,  5.8095365e-02f,  8.3350925e-02f,
         1.1110441e-01f,  1.4308883e-01f,  1.8294797e-01f,  2.4159188e-01f,
    };
    return cb[idx];
}

// --- TBQ3_0 quantize: 3-bit Lloyd-Max (8 centroids) ---

static __device__ void quantize_f32_tbq3_0_block(const float * __restrict__ x,
                                                   block_tbq3_0 * __restrict__ y,
                                                   int64_t block_idx) {
    float tmp[QK_TBQ];

    // Step 1: norm
    float norm = 0.0f;
    for (int j = 0; j < QK_TBQ; j++) norm += x[j] * x[j];
    norm = sqrtf(norm);
    y->d = __float2half(norm);

    // Step 2: normalize + sign-flip
    float inv_norm = (norm > 1e-10f) ? 1.0f / norm : 0.0f;
    for (int j = 0; j < QK_TBQ; j++) tmp[j] = x[j] * inv_norm;
    uint64_t rng = tbq_rot_seed_cuda(block_idx);
    for (int j = 0; j < QK_TBQ; j++) {
        rng = tbq_xorshift64_cuda(rng);
        if (rng & 1) tmp[j] = -tmp[j];
    }

    // Step 3: per-block scale
    float sum_sq = 0.0f;
    for (int j = 0; j < QK_TBQ; j++) sum_sq += tmp[j] * tmp[j];
    float actual_rms = sqrtf(sum_sq / (float)QK_TBQ);
    float expected_rms = 1.0f / sqrtf((float)QK_TBQ);
    float block_scale = (actual_rms > 1e-10f) ? actual_rms / expected_rms : 1.0f;
    y->s = __float2half(block_scale);
    float inv_scale = 1.0f / block_scale;
    for (int j = 0; j < QK_TBQ; j++) tmp[j] *= inv_scale;

    // Step 4: nearest centroid (3-bit = 8 centroids)
    for (int j = 0; j < 3 * QK_TBQ / 8; j++) y->idx[j] = 0;
    for (int j = 0; j < QK_TBQ; j++) {
        int best = 0;
        float best_dist = fabsf(tmp[j] - tbq_cb8(0));
        for (int c = 1; c < 8; c++) {
            float dist = fabsf(tmp[j] - tbq_cb8(c));
            if (dist < best_dist) { best_dist = dist; best = c; }
        }
        int bit_pos = j * 3;
        int byte_pos = bit_pos / 8;
        int bit_off = bit_pos % 8;
        y->idx[byte_pos] |= (uint8_t)(best << bit_off);
        if (bit_off + 3 > 8) {
            y->idx[byte_pos + 1] |= (uint8_t)(best >> (8 - bit_off));
        }
    }
}

// --- TBQ4_0 quantize: 4-bit Lloyd-Max (16 centroids) ---

static __device__ void quantize_f32_tbq4_0_block(const float * __restrict__ x,
                                                   block_tbq4_0 * __restrict__ y,
                                                   int64_t block_idx) {
    float tmp[QK_TBQ];

    // Step 1: norm
    float norm = 0.0f;
    for (int j = 0; j < QK_TBQ; j++) norm += x[j] * x[j];
    norm = sqrtf(norm);
    y->d = __float2half(norm);

    // Step 2: normalize + sign-flip
    float inv_norm = (norm > 1e-10f) ? 1.0f / norm : 0.0f;
    for (int j = 0; j < QK_TBQ; j++) tmp[j] = x[j] * inv_norm;
    uint64_t rng = tbq_rot_seed_cuda(block_idx);
    for (int j = 0; j < QK_TBQ; j++) {
        rng = tbq_xorshift64_cuda(rng);
        if (rng & 1) tmp[j] = -tmp[j];
    }

    // Step 3: per-block scale
    float sum_sq = 0.0f;
    for (int j = 0; j < QK_TBQ; j++) sum_sq += tmp[j] * tmp[j];
    float actual_rms = sqrtf(sum_sq / (float)QK_TBQ);
    float expected_rms = 1.0f / sqrtf((float)QK_TBQ);
    float block_scale = (actual_rms > 1e-10f) ? actual_rms / expected_rms : 1.0f;
    y->s = __float2half(block_scale);
    float inv_scale = 1.0f / block_scale;
    for (int j = 0; j < QK_TBQ; j++) tmp[j] *= inv_scale;

    // Step 4: nearest centroid (4-bit = 16 centroids)
    for (int j = 0; j < QK_TBQ / 2; j++) y->idx[j] = 0;
    for (int j = 0; j < QK_TBQ; j++) {
        int best = 0;
        float best_dist = fabsf(tmp[j] - tbq_cb16(0));
        for (int c = 1; c < 16; c++) {
            float dist = fabsf(tmp[j] - tbq_cb16(c));
            if (dist < best_dist) { best_dist = dist; best = c; }
        }
        y->idx[j / 2] |= (uint8_t)(best << (4 * (j % 2)));
    }
}

// --- Dequantize device functions ---

static __device__ void dequantize_f32_tbq3_0_block(const block_tbq3_0 * __restrict__ x,
                                                     float * __restrict__ y,
                                                     int64_t block_idx) {
    const float d_norm = __half2float(x->d);
    const float block_scale = __half2float(x->s);

    float tmp[QK_TBQ];

    // Centroid lookup (3-bit), scaled
    for (int j = 0; j < QK_TBQ; j++) {
        int bit_pos = j * 3;
        int byte_pos = bit_pos / 8;
        int bit_off = bit_pos % 8;
        int idx = (x->idx[byte_pos] >> bit_off);
        if (bit_off + 3 > 8) {
            idx |= ((int)x->idx[byte_pos + 1] << (8 - bit_off));
        }
        idx &= 7;
        tmp[j] = block_scale * tbq_cb8(idx);
    }

    // Undo sign-flip
    uint64_t rng = tbq_rot_seed_cuda(block_idx);
    for (int j = 0; j < QK_TBQ; j++) {
        rng = tbq_xorshift64_cuda(rng);
        if (rng & 1) tmp[j] = -tmp[j];
    }

    // Scale by norm
    for (int j = 0; j < QK_TBQ; j++) {
        y[j] = d_norm * tmp[j];
    }
}

static __device__ void dequantize_f32_tbq4_0_block(const block_tbq4_0 * __restrict__ x,
                                                     float * __restrict__ y,
                                                     int64_t block_idx) {
    const float d_norm = __half2float(x->d);
    const float block_scale = __half2float(x->s);

    float tmp[QK_TBQ];

    // Centroid lookup (4-bit = nibble packed), scaled
    for (int j = 0; j < QK_TBQ; j++) {
        int idx = (x->idx[j / 2] >> (4 * (j % 2))) & 0xF;
        tmp[j] = block_scale * tbq_cb16(idx);
    }

    // Undo sign-flip
    uint64_t rng = tbq_rot_seed_cuda(block_idx);
    for (int j = 0; j < QK_TBQ; j++) {
        rng = tbq_xorshift64_cuda(rng);
        if (rng & 1) tmp[j] = -tmp[j];
    }

    // Scale by norm
    for (int j = 0; j < QK_TBQ; j++) {
        y[j] = d_norm * tmp[j];
    }
}
