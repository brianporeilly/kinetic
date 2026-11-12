#pragma once

#include "attention.cuh"

// Attention for large batches without tensor cores, flash style in f32: a
// block takes 64 queries of one head and walks their keys in tiles of 64,
// with an online softmax, so no scores are stored and every key and value
// is read once per 64 queries rather than once per query. The scores Q K^T
// are a register-tiled product over slices of 32 head dims staged in shared
// memory, each thread 4 queries by 4 keys; the weights then go to shared
// memory, and P V is a second product, each thread 8 queries by VD / 32
// value dims, over sub-tiles of 16 keys whose values take the slices' place.
// Queries were normalized and rotated, and keys and values cached, by
// `attn_prepare`; the gated result goes to the f32 values of `out`,
// quantized afterwards. Heads of 512 leave half their value dims to each of
// two blocks, as all 512 sums would not fit the registers.

constexpr unsigned TILE_Q = 64;
constexpr unsigned TILE_K = 64;
constexpr unsigned TILE_D = 32;
constexpr unsigned TILE_VK = 16;
constexpr unsigned TILE_THREADS = 256;
// f32 values of padding per row of weights, keeping float4 alignment.
constexpr unsigned TILE_PAD = 4;

__host__ __device__ constexpr unsigned tile_values(unsigned hd) { return hd == 512 ? 256 : hd; }

// Shared memory of a block: query and key slices, or a sub-tile of values,
// in one buffer; the weights; and per query the rescaling of its sums and
// at the end its total weight.
struct TileShared {
    float slices[TILE_D * (TILE_Q + TILE_K)];
    float weights[TILE_K][TILE_Q + TILE_PAD];
    float alpha[TILE_Q];
    float total[TILE_Q];
};
static_assert(TILE_VK * 256 <= TILE_D * (TILE_Q + TILE_K), "values fit the slices");

// Values at..at + 8 of an f16 or q8_0 cache as f32.
__device__ __forceinline__ void tile_load8(float (&x)[8], const void* cache, const __half* scale, size_t at) {
    float2 f[4];
    if (!scale) {
        AttnRow<8, false> r;
        r.load(cache, scale, at);
#pragma unroll
        for (unsigned e = 0; e < 4; ++e) f[e] = r.pair(e);
    } else {
        AttnRow<8, true> r;
        r.load(cache, scale, at);
#pragma unroll
        for (unsigned e = 0; e < 4; ++e) f[e] = r.pair(e);
    }
#pragma unroll
    for (unsigned e = 0; e < 4; ++e) {
        x[2 * e] = f[e].x;
        x[2 * e + 1] = f[e].y;
    }
}

template <unsigned HD>
__device__ __forceinline__ void attn_tile_impl(const AttnArgs& a, unsigned n, TileShared& sh) {
    constexpr unsigned VD = tile_values(HD), VQ = VD / 128;
    float* qs = sh.slices;                    // [TILE_D][TILE_Q]
    float* ks = sh.slices + TILE_D * TILE_Q;  // [TILE_D][TILE_K]
    float* vs = sh.slices;                    // [TILE_VK][VD]

    const unsigned tid = threadIdx.x;
    const unsigned t0 = blockIdx.x * TILE_Q, h = blockIdx.y, v0 = blockIdx.z * VD;
    const unsigned kvh = h / (a.n_head / a.n_head_kv), kv_dim = a.n_head_kv * HD;
    const unsigned pos0 = a.batch[0];
    const unsigned last = min(t0 + TILE_Q, n) - 1;
    // tiles start on multiples of TILE_K whatever the batch's first position,
    // so where a prompt is cut changes no sum; keys before `seen` are masked
    const unsigned seen = attn_start(a, pos0 + t0), lo = seen / TILE_K * TILE_K, hi = pos0 + last + 1;
    const float scale = a.scale * 1.4426950408889634f;
    const auto slot = [&](unsigned p) {
        return static_cast<size_t>(attn_slot(a, min(max(p, seen), hi - 1))) * kv_dim + kvh * HD;
    };

    // scores: queries 4sq.., keys 4sk..; products: queries 8pq.., value
    // dims 4pd + 128j
    const unsigned sq = tid / 16, sk = tid % 16;
    const unsigned pq = tid / 32, pd = tid % 32;
    unsigned qpos[4], qstart[4];
    float m[4], l[4];
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        qpos[i] = pos0 + t0 + 4 * sq + i;
        qstart[i] = attn_start(a, qpos[i]);
        m[i] = -CUDART_INF_F;
        l[i] = 0.0f;
    }
    float o[8][4 * VQ] = {};

    for (unsigned k0 = lo; k0 < hi; k0 += TILE_K) {
        float s[4][4] = {};
        for (unsigned d0 = 0; d0 < HD; d0 += TILE_D) {
            // the slices were last read by the previous slice or sub-tile
            __syncthreads();
            for (unsigned i = tid; i < TILE_Q * TILE_D / 4; i += TILE_THREADS) {
                const unsigned q = i % TILE_Q, d = 4 * (i / TILE_Q);
                const unsigned tok = min(t0 + q, n - 1);
                const float4 x =
                    *reinterpret_cast<const float4*>(a.q + tok * a.q_stride + h * a.q_head_stride + d0 + d);
                qs[d * TILE_Q + q] = x.x;
                qs[(d + 1) * TILE_Q + q] = x.y;
                qs[(d + 2) * TILE_Q + q] = x.z;
                qs[(d + 3) * TILE_Q + q] = x.w;
            }
            for (unsigned i = tid; i < TILE_K * TILE_D / 8; i += TILE_THREADS) {
                const unsigned k = i % TILE_K, d = 8 * (i / TILE_K);
                float x[8];
                tile_load8(x, a.k_cache, a.k_scale, slot(k0 + k) + d0 + d);
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) ks[(d + e) * TILE_K + k] = x[e];
            }
            __syncthreads();
#pragma unroll 8
            for (unsigned d = 0; d < TILE_D; ++d) {
                const float4 qv = *reinterpret_cast<const float4*>(&qs[d * TILE_Q + 4 * sq]);
                const float4 kv = *reinterpret_cast<const float4*>(&ks[d * TILE_K + 4 * sk]);
                const float qa[4] = {qv.x, qv.y, qv.z, qv.w}, ka[4] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
                for (unsigned i = 0; i < 4; ++i)
#pragma unroll
                    for (unsigned j = 0; j < 4; ++j) s[i][j] = fmaf(qa[i], ka[j], s[i][j]);
            }
        }

        // online softmax; a query's keys are spread over the 16 lanes of
        // its half warp, each of which keeps the sum of its own weights
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            float mx = m[i];
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const unsigned kp = k0 + 4 * sk + j;
                const bool visible = kp <= qpos[i] && kp >= qstart[i];
                s[i][j] = visible ? s[i][j] * scale : -CUDART_INF_F;
                mx = fmaxf(mx, s[i][j]);
            }
#pragma unroll
            for (unsigned w = 1; w < 16; w *= 2) mx = fmaxf(mx, __shfl_xor_sync(FULL_MASK, mx, w));
            // A query that has seen no visible key yet keeps nothing to rescale.
            const float alpha = mx == -CUDART_INF_F ? 1.0f : exp2f(m[i] - mx);
            const float base = mx == -CUDART_INF_F ? 0.0f : mx;
            float sum = 0.0f;
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                s[i][j] = exp2f(s[i][j] - base);
                sum += s[i][j];
            }
            l[i] = fmaf(l[i], alpha, sum);
            m[i] = mx;
            if (sk == 0) sh.alpha[4 * sq + i] = alpha;
        }
#pragma unroll
        for (unsigned j = 0; j < 4; ++j)
            *reinterpret_cast<float4*>(&sh.weights[4 * sk + j][4 * sq]) =
                make_float4(s[0][j], s[1][j], s[2][j], s[3][j]);
        // the weights and rescalings are written, and the slices free
        __syncthreads();
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            const float alpha = sh.alpha[8 * pq + i];
#pragma unroll
            for (unsigned c = 0; c < 4 * VQ; ++c) o[i][c] *= alpha;
        }

        for (unsigned kk = 0; kk < TILE_K; kk += TILE_VK) {
            if (kk > 0) __syncthreads();
            for (unsigned i = tid; i < TILE_VK * VD / 8; i += TILE_THREADS) {
                const unsigned k = i / (VD / 8), col = 8 * (i % (VD / 8));
                float x[8];
                tile_load8(x, a.v_cache, a.v_scale, slot(k0 + kk + k) + v0 + col);
                *reinterpret_cast<float4*>(&vs[k * VD + col]) = make_float4(x[0], x[1], x[2], x[3]);
                *reinterpret_cast<float4*>(&vs[k * VD + col + 4]) = make_float4(x[4], x[5], x[6], x[7]);
            }
            __syncthreads();
#pragma unroll 4
            for (unsigned k = 0; k < TILE_VK; ++k) {
                const float4 p0 = *reinterpret_cast<const float4*>(&sh.weights[kk + k][8 * pq]);
                const float4 p1 = *reinterpret_cast<const float4*>(&sh.weights[kk + k][8 * pq + 4]);
                const float p[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
#pragma unroll
                for (unsigned j = 0; j < VQ; ++j) {
                    const float4 v = *reinterpret_cast<const float4*>(&vs[k * VD + 4 * pd + 128 * j]);
#pragma unroll
                    for (unsigned i = 0; i < 8; ++i) {
                        o[i][4 * j] = fmaf(p[i], v.x, o[i][4 * j]);
                        o[i][4 * j + 1] = fmaf(p[i], v.y, o[i][4 * j + 1]);
                        o[i][4 * j + 2] = fmaf(p[i], v.z, o[i][4 * j + 2]);
                        o[i][4 * j + 3] = fmaf(p[i], v.w, o[i][4 * j + 3]);
                    }
                }
            }
        }
    }

#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        float sum = l[i];
#pragma unroll
        for (unsigned w = 1; w < 16; w *= 2) sum += __shfl_xor_sync(FULL_MASK, sum, w);
        if (sk == 0) sh.total[4 * sq + i] = sum;
    }
    __syncthreads();
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        const unsigned tok = t0 + 8 * pq + i;
        if (tok >= n) continue;
        const float inv = 1.0f / sh.total[8 * pq + i];
        const float* gates = a.q + tok * a.q_stride + h * a.q_head_stride + HD;
        const float head_gate = a.head_gate ? sigmoid(a.head_gate[tok * a.n_head + h]) : 1.0f;
        float* out = a.out.x + static_cast<size_t>(tok) * a.n_head * HD + h * HD;
#pragma unroll
        for (unsigned j = 0; j < VQ; ++j) {
            const unsigned col = v0 + 4 * pd + 128 * j;
            float v[4];
#pragma unroll
            for (unsigned c = 0; c < 4; ++c)
                v[c] = o[i][4 * j + c] * inv * (a.interleaved_gate ? sigmoid(gates[col + c]) : head_gate);
            *reinterpret_cast<float4*>(out + col) = make_float4(v[0], v[1], v[2], v[3]);
        }
    }
}

// Blocks per multiprocessor the registers are fitted for: two where the
// shared memory holds two, as it does on the GTX 10 series and the V100,
// which pays for a few spilled registers; one on the P100, whose 64 KB hold
// only one.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 600
constexpr unsigned TILE_BLOCKS = 1;
#else
constexpr unsigned TILE_BLOCKS = 2;
#endif

// grid (ceil(n / TILE_Q), n_head, head_dim / tile_values(head_dim)), block
// TILE_THREADS; heads of 128, 256 or 512; `n` is the batch size.
extern "C" __global__ void __launch_bounds__(TILE_THREADS, TILE_BLOCKS) attn_tile(AttnArgs a, unsigned n) {
    __shared__ __align__(16) TileShared sh;
    switch (a.head_dim) {
        case 128: attn_tile_impl<128>(a, n, sh); break;
        case 256: attn_tile_impl<256>(a, n, sh); break;
        default: attn_tile_impl<512>(a, n, sh); break;
    }
}
