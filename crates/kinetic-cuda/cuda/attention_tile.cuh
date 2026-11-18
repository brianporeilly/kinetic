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

// The P100 (sm_60) runs f16 at twice its f32 rate, so its build takes the
// tile in f16 pairs: queries and keys staged as pairs of head dims, each
// score's terms summed sixteen dims at a time in f16 and those sums in f32;
// the weights in f16, and P V over pairs of value dims, sixteen keys at a
// time in f16 and those sums in f32. Slices of 64 dims and sub-tiles of 32
// keys halve the barriers per key tile, and the shared memory leaves room
// for two blocks per multiprocessor. Results are deterministic and keep
// the tiles' positions, so prompt cuts still change no bit; they round to
// about one part in a thousand of their terms, as the products do there.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 600
#define ATTN_TILE_HALF 1
#endif

constexpr unsigned TH_D = 64;   // head dims per slice
constexpr unsigned TH_VK = 32;  // keys per sub-tile of values
constexpr unsigned TH_RUN = 8;  // pairs summed in f16 before f32
constexpr unsigned TH_WPAD = 8; // f16 padding per row of weights

struct TileHalfShared {
    // query and key slices, [TH_D / 2][TILE_Q] and [TH_D / 2][TILE_K]
    // pairs, or a sub-tile of values, [TH_VK][VD / 2] pairs
    __half2 slices[TH_D / 2 * (TILE_Q + TILE_K)];
    __half weights[TILE_K][TILE_Q + TH_WPAD];
    float alpha[TILE_Q];
    float total[TILE_Q];
};
static_assert(TH_VK * 256 / 2 <= TH_D / 2 * (TILE_Q + TILE_K), "values fit the slices");

template <unsigned HD>
__device__ __forceinline__ void attn_tile_half_impl(const AttnArgs& a, unsigned n, TileHalfShared& sh) {
    constexpr unsigned VD = tile_values(HD), VQ = VD / 128, PAIRS = TH_D / 2;
    __half2* qs = sh.slices;                   // [PAIRS][TILE_Q]
    __half2* ks = sh.slices + PAIRS * TILE_Q;  // [PAIRS][TILE_K]
    __half2* vs = sh.slices;                   // [TH_VK][VD / 2]

    const unsigned tid = threadIdx.x;
    const unsigned t0 = blockIdx.x * TILE_Q, h = blockIdx.y, v0 = blockIdx.z * VD;
    const unsigned kvh = h / (a.n_head / a.n_head_kv), kv_dim = a.n_head_kv * HD;
    const unsigned pos0 = a.batch[0];
    const unsigned last = min(t0 + TILE_Q, n) - 1;
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
        for (unsigned d0 = 0; d0 < HD; d0 += TH_D) {
            // the slices were last read by the previous slice or sub-tile
            __syncthreads();
            for (unsigned i = tid; i < TILE_Q * TH_D / 4; i += TILE_THREADS) {
                const unsigned q = i % TILE_Q, d = 4 * (i / TILE_Q);
                const unsigned tok = min(t0 + q, n - 1);
                const float4 x =
                    *reinterpret_cast<const float4*>(a.q + tok * a.q_stride + h * a.q_head_stride + d0 + d);
                qs[(d / 2) * TILE_Q + q] = __floats2half2_rn(x.x, x.y);
                qs[(d / 2 + 1) * TILE_Q + q] = __floats2half2_rn(x.z, x.w);
            }
            for (unsigned i = tid; i < TILE_K * TH_D / 8; i += TILE_THREADS) {
                const unsigned k = i % TILE_K, d = 8 * (i / TILE_K);
                float x[8];
                tile_load8(x, a.k_cache, a.k_scale, slot(k0 + k) + d0 + d);
#pragma unroll
                for (unsigned e = 0; e < 4; ++e) ks[(d / 2 + e) * TILE_K + k] = __floats2half2_rn(x[2 * e], x[2 * e + 1]);
            }
            __syncthreads();
#pragma unroll
            for (unsigned r = 0; r < PAIRS; r += TH_RUN) {
                __half2 acc[4][4];
#pragma unroll
                for (unsigned i = 0; i < 4; ++i)
#pragma unroll
                    for (unsigned j = 0; j < 4; ++j) acc[i][j] = __float2half2_rn(0.0f);
#pragma unroll
                for (unsigned p = r; p < r + TH_RUN; ++p) {
                    const uint4 qv = *reinterpret_cast<const uint4*>(&qs[p * TILE_Q + 4 * sq]);
                    const uint4 kv = *reinterpret_cast<const uint4*>(&ks[p * TILE_K + 4 * sk]);
                    const __half2* qa = reinterpret_cast<const __half2*>(&qv);
                    const __half2* ka = reinterpret_cast<const __half2*>(&kv);
#pragma unroll
                    for (unsigned i = 0; i < 4; ++i)
#pragma unroll
                        for (unsigned j = 0; j < 4; ++j) acc[i][j] = __hfma2(qa[i], ka[j], acc[i][j]);
                }
#pragma unroll
                for (unsigned i = 0; i < 4; ++i)
#pragma unroll
                    for (unsigned j = 0; j < 4; ++j) {
                        const float2 f = __half22float2(acc[i][j]);
                        s[i][j] += f.x + f.y;
                    }
            }
        }

        // online softmax, as the f32 tile
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
            const float alpha = mx == -CUDART_INF_F ? 1.0f : exp2f(m[i] - mx);
            const float base = mx == -CUDART_INF_F ? 0.0f : mx;
            float sum = 0.0f;
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                // the sums take the weights as the products will, in f16
                s[i][j] = __half2float(__float2half_rn(exp2f(s[i][j] - base)));
                sum += s[i][j];
            }
            l[i] = fmaf(l[i], alpha, sum);
            m[i] = mx;
            if (sk == 0) sh.alpha[4 * sq + i] = alpha;
        }
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            const __half2 w01 = __floats2half2_rn(s[0][j], s[1][j]), w23 = __floats2half2_rn(s[2][j], s[3][j]);
            uint2 w;
            w.x = *reinterpret_cast<const unsigned*>(&w01);
            w.y = *reinterpret_cast<const unsigned*>(&w23);
            *reinterpret_cast<uint2*>(&sh.weights[4 * sk + j][4 * sq]) = w;
        }
        // the weights and rescalings are written, and the slices free
        __syncthreads();
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            const float alpha = sh.alpha[8 * pq + i];
#pragma unroll
            for (unsigned c = 0; c < 4 * VQ; ++c) o[i][c] *= alpha;
        }

        for (unsigned kk = 0; kk < TILE_K; kk += TH_VK) {
            if (kk > 0) __syncthreads();
            for (unsigned i = tid; i < TH_VK * VD / 8; i += TILE_THREADS) {
                const unsigned k = i / (VD / 8), col = 8 * (i % (VD / 8));
                float x[8];
                tile_load8(x, a.v_cache, a.v_scale, slot(k0 + kk + k) + v0 + col);
                __half2 hx[4];
#pragma unroll
                for (unsigned e = 0; e < 4; ++e) hx[e] = __floats2half2_rn(x[2 * e], x[2 * e + 1]);
                *reinterpret_cast<uint4*>(&vs[k * (VD / 2) + col / 2]) = *reinterpret_cast<const uint4*>(hx);
            }
            __syncthreads();
            // a column group at a time, so only its sums in f16 stay live
#pragma unroll
            for (unsigned j = 0; j < VQ; ++j) {
                const unsigned col = (4 * pd + 128 * j) / 2;
#pragma unroll
                for (unsigned k1 = 0; k1 < TH_VK; k1 += 2 * TH_RUN) {
                    __half2 acc[8][2];
#pragma unroll
                    for (unsigned i = 0; i < 8; ++i) acc[i][0] = acc[i][1] = __float2half2_rn(0.0f);
#pragma unroll 4
                    for (unsigned k = k1; k < k1 + 2 * TH_RUN; ++k) {
                        const uint4 pw = *reinterpret_cast<const uint4*>(&sh.weights[kk + k][8 * pq]);
                        const __half* pa = reinterpret_cast<const __half*>(&pw);
                        const uint2 vw = *reinterpret_cast<const uint2*>(&vs[k * (VD / 2) + col]);
                        const __half2* v = reinterpret_cast<const __half2*>(&vw);
#pragma unroll
                        for (unsigned i = 0; i < 8; ++i) {
                            const __half2 p = __half2half2(pa[i]);
                            acc[i][0] = __hfma2(p, v[0], acc[i][0]);
                            acc[i][1] = __hfma2(p, v[1], acc[i][1]);
                        }
                    }
#pragma unroll
                    for (unsigned i = 0; i < 8; ++i)
#pragma unroll
                        for (unsigned c = 0; c < 2; ++c) {
                            const float2 f = __half22float2(acc[i][c]);
                            o[i][4 * j + 2 * c] += f.x;
                            o[i][4 * j + 2 * c + 1] += f.y;
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
// and on the P100 for the f16 tile.
constexpr unsigned TILE_BLOCKS = 2;

// grid (ceil(n / TILE_Q), n_head, head_dim / tile_values(head_dim)), block
// TILE_THREADS; heads of 128, 256 or 512; `n` is the batch size.
extern "C" __global__ void __launch_bounds__(TILE_THREADS, TILE_BLOCKS) attn_tile(AttnArgs a, unsigned n) {
#ifdef ATTN_TILE_HALF
    __shared__ __align__(16) TileHalfShared sh;
    switch (a.head_dim) {
        case 128: attn_tile_half_impl<128>(a, n, sh); break;
        case 256: attn_tile_half_impl<256>(a, n, sh); break;
        default: attn_tile_half_impl<512>(a, n, sh); break;
    }
#else
    __shared__ __align__(16) TileShared sh;
    switch (a.head_dim) {
        case 128: attn_tile_impl<128>(a, n, sh); break;
        case 256: attn_tile_impl<256>(a, n, sh); break;
        default: attn_tile_impl<512>(a, n, sh); break;
    }
#endif
}
