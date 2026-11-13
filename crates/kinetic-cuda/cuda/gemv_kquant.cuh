#pragma once

#include "kquant.cuh"

// K-quant GEMV with dp4a, in the order kquant.cuh defines: one warp per
// row, where lane group v = lane / 8 is partial sum v. Its eight lanes split
// a 256-element block; each computes the term of one sub-block (Q4_K, Q5_K:
// after adding its partner's half of the integer products) or the sum of
// two runs (Q6_K), and three exchanges within the group finish the tree.

// Q4_K and Q5_K: lane j holds qs bytes 32c + 16h .. +16 (c = j / 2,
// h = j % 2), elements 16h..16h+15 of sub-blocks 2c (low nibbles) and 2c+1
// (high nibbles), and computes the term of sub-block 2c + h.
struct K4Lane {
    struct Block {
        Words head;
        Words lo, hi;
    };

    __device__ static float term(const Block& k, unsigned b, unsigned j, Q8Input x) {
        const unsigned c = j / 2, h = j % 2;
        const int8_t* xp = x.qs + b * 256 + 64 * c + 16 * h;
        int lo = dot16(k.lo, load_cached(xp));
        int hi = dot16(k.hi, load_cached(xp + 32));
        lo += __shfl_xor_sync(FULL_MASK, lo, 1);
        hi += __shfl_xor_sync(FULL_MASK, hi, 1);
        const unsigned s = 2 * c + h;
        const float2 d = __ldg(x.ds + b * 8 + s);
        const float2 g = k4_group(k.head, s);
        return kq_term(g.x, g.y, d.x, d.y, h ? hi : lo);
    }
};

struct KqQ4K : K4Lane {
    __device__ static Block load(const uint8_t* w, size_t, size_t blk, unsigned j) {
        const uint8_t* p = w + blk * 144;
        Block k;
        k.head = load_weights(p);
        const Words q = load_weights(p + 16 + 16 * j);
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            k.lo.w[i] = q.w[i] & NIBBLES;
            k.hi.w[i] = (q.w[i] >> 4) & NIBBLES;
        }
        return k;
    }
};

// Bit s of qh byte l is the fifth bit of element l of sub-block s.
struct KqQ5K : K4Lane {
    __device__ static Block load(const uint8_t* w, size_t, size_t blk, unsigned j) {
        const uint8_t* p = w + blk * 176;
        const unsigned c = j / 2;
        Block k;
        k.head = load_weights(p);
        const Words qh = load_weights(p + 16 + 16 * (j % 2));
        const Words q = load_weights(p + 48 + 16 * j);
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const unsigned bits = static_cast<unsigned>(qh.w[i]);
            k.lo.w[i] = (q.w[i] & NIBBLES) | (((bits >> (2 * c)) & 0x01010101) << 4);
            k.hi.w[i] = ((q.w[i] >> 4) & NIBBLES) | (((bits >> (2 * c + 1)) & 0x01010101) << 4);
        }
        return k;
    }
};

// Q6_K, planar as layout.rs stores it: lane j holds elements 32j..32j+31,
// group g = j % 4 of half n = j / 4, which are runs 2j and 2j + 1.
struct KqQ6K {
    // The two runs' scales and their quants as stored (+ 32).
    struct Block {
        float scale[2];
        Words q[2];
    };

    __device__ static Block load(const uint8_t* w, size_t total, size_t blk, unsigned j) {
        const unsigned n = j / 4, g = j % 4;
        const uint8_t* ql = w + blk * 128 + 64 * n + 32 * (g & 1);
        const uint8_t* qh = w + total * 128 + blk * 64 + 32 * n;
        const unsigned scales =
            __ldg(reinterpret_cast<const unsigned*>(w + total * 192 + blk * 16) + j / 2);
        const float d = __half2float(__ldg(reinterpret_cast<const __half*>(w + total * 208) + blk));
        Block k;
#pragma unroll
        for (unsigned r = 0; r < 2; ++r) {
            k.scale[r] = k6_group(d, scales, 2 * j + r);
            const Words low = load_weights(ql + 16 * r);
            const Words high = load_weights(qh + 16 * r);
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) k.q[r].w[i] = k6_stored(low.w[i], high.w[i], g);
        }
        return k;
    }

    __device__ static float term(const Block& k, unsigned b, unsigned j, Q8Input x) {
        const int8_t* xp = x.qs + b * 256 + 32 * j;
        const float xd = __ldg(x.ds + b * 8 + j).x;
        float t[2];
#pragma unroll
        for (unsigned r = 0; r < 2; ++r) {
            const Words a = load_cached(xp + 16 * r);
            t[r] = kq_term(k.scale[r], 0.0f, xd, 0.0f, unbiased_dot(k.q[r], a));
        }
        return __fadd_rn(t[0], t[1]);
    }

    // The integer dot product of quants stored + 32 with the input.
    __device__ static int unbiased_dot(const Words& q, const Words& a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610
        // Without dp4a a second dot product with the bias doubles the work.
        // The stored values are below 64, so flipping bit 5 and shifting the
        // word left by 2 makes each byte 4 (q - 32) as a signed byte, with
        // nothing crossing into the next; the sum is then exactly 4 times
        // the dot product.
        int s = 0;
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const unsigned w = (static_cast<unsigned>(q.w[i]) ^ 0x20202020u) << 2;
            const unsigned x = static_cast<unsigned>(a.w[i]);
            s = mad_halves(byte_halves<EVEN_BYTES>(w), byte_halves<EVEN_BYTES>(x), s);
            s = mad_halves(byte_halves<ODD_BYTES>(w), byte_halves<ODD_BYTES>(x), s);
        }
        return s >> 2;
#else
        // Dotting the stored +32 with the input takes it back out.
        const Words bias = {{0x20202020, 0x20202020, 0x20202020, 0x20202020}};
        return dot16(q, a) - dot16(bias, a);
#endif
    }
};

// Y = W X for N inputs; output i of row r goes to y[i * rows + r].
template <class F, unsigned N>
__device__ __forceinline__ void gemv_kq(const uint8_t* w, Q8Input x, float* y, unsigned rows,
                                        unsigned blocks_per_row, size_t total) {
    const unsigned row = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= rows) return;
    const unsigned lane = threadIdx.x % WARP, v = lane / 8, j = lane % 8;
    const unsigned len = blocks_per_row * 256;
    const size_t first = static_cast<size_t>(row) * blocks_per_row;

    float acc[N] = {};
    for (unsigned base = 0; base < blocks_per_row; base += KQ_SPLIT) {
        const unsigned b = base + v;
        const bool active = b < blocks_per_row;
        typename F::Block blk = {};
        if (active) blk = F::load(w, total, first + b, j);
#pragma unroll
        for (unsigned i = 0; i < N; ++i) {
            // Idle groups read block 0 of the input; their sums are dropped.
            float t = F::term(blk, active ? b : 0, j, batch_row(x, i, len));
#pragma unroll
            for (unsigned o = 1; o < 8; o *= 2) t = __fadd_rn(t, __shfl_xor_sync(FULL_MASK, t, o));
            if (active) acc[i] = __fadd_rn(acc[i], t);
        }
    }
#pragma unroll
    for (unsigned i = 0; i < N; ++i) {
        float s = __shfl_sync(FULL_MASK, acc[i], 0);
#pragma unroll
        for (unsigned p = 1; p < KQ_SPLIT; ++p) s = __fadd_rn(s, __shfl_sync(FULL_MASK, acc[i], 8 * p));
        if (lane == 0) y[static_cast<size_t>(i) * rows + row] = s;
    }
}

// The Tesla P100 (sm_60) has no dp4a, and the integer dot products above
// cost it about six instructions a weight, which keeps one-token decode
// below its memory bandwidth. There the products run in f16 pairs at twice
// its f32 rate. A weight's unsigned integer, put in the low bits of an f16
// as it is, is the subnormal weight * 2^-24, exact and free to form; the
// int8 inputs are exact f16 integers. Each run of 16 products adds into an
// f16 pair, eight terms a half, and the halves are added in f32 and scaled
// back by 2^24, giving what the integer dot product gives, then the same
// scales and order after it (Q6_K's stored + 32 comes out as 32 times the
// run's input sum, exact in f16). The products and sums round to about one
// part in a thousand of their terms; each input still gets the same bits
// alone or batched. A warp takes KQ_HALF_ROWS rows, so each input is
// widened to f16 once for all of them. Must match Kernels::half_decode.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 600
constexpr bool KQ_HALF = true;
#else
constexpr bool KQ_HALF = false;
#endif
// Rows per warp of the f16 kernels; must match KQ_HALF_ROWS in gemv.rs.
constexpr unsigned KQ_HALF_ROWS = 2;

// Bytes 0 and 2 (EVEN) or 1 and 3 (ODD) of v, each in the low byte of a
// 16-bit half above `high`.
constexpr unsigned EVEN_F16 = 0x4240;
constexpr unsigned ODD_F16 = 0x4341;

template <unsigned SELECT>
__device__ __forceinline__ unsigned bytes_over(unsigned v, unsigned high) {
    unsigned r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(v), "r"(high), "n"(SELECT));
    return r;
}

// Unsigned bytes as the f16 pair 1024 + byte, less `offset`.
template <unsigned SELECT>
__device__ __forceinline__ __half2 bytes_f16(unsigned v, __half2 offset) {
    const unsigned r = bytes_over<SELECT>(v, 0x64646464u);
    return __hsub2(*reinterpret_cast<const __half2*>(&r), offset);
}

// Unsigned bytes as the subnormal f16 pair byte * 2^-24.
template <unsigned SELECT>
__device__ __forceinline__ __half2 bytes_tiny(unsigned v) {
    const unsigned r = bytes_over<SELECT>(v, 0u);
    return *reinterpret_cast<const __half2*>(&r);
}

// Sixteen int8 inputs as f16 pairs: even[i] holds elements 4i and 4i + 2,
// odd[i] 4i + 1 and 4i + 3.
struct HalfRun {
    __half2 even[4], odd[4];
};

__device__ __forceinline__ HalfRun input_run(const int8_t* p) {
    const Words a = load_cached(p);
    const __half2 offset = __float2half2_rn(1152.0f);  // 1024 + 128
    HalfRun r;
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const unsigned u = static_cast<unsigned>(a.w[i]) ^ 0x80808080u;
        r.even[i] = bytes_f16<EVEN_F16>(u, offset);
        r.odd[i] = bytes_f16<ODD_F16>(u, offset);
    }
    return r;
}

// Sixteen unsigned weights as subnormal f16 pairs, weight * 2^-24.
__device__ __forceinline__ HalfRun weight_run(const Words& q) {
    HalfRun r;
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        r.even[i] = bytes_tiny<EVEN_F16>(static_cast<unsigned>(q.w[i]));
        r.odd[i] = bytes_tiny<ODD_F16>(static_cast<unsigned>(q.w[i]));
    }
    return r;
}

// The dot product of a run of weights from weight_run with 16 inputs, its
// halves added in f32 and scaled back by 2^24.
__device__ __forceinline__ float half_dot(const HalfRun& w, const HalfRun& x) {
    __half2 s = __float2half2_rn(0.0f);
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        s = __hfma2(w.even[i], x.even[i], s);
        s = __hfma2(w.odd[i], x.odd[i], s);
    }
    const float2 f = __half22float2(s);
    return __fmul_rn(__fadd_rn(f.x, f.y), 16777216.0f);
}

// The sum of 16 inputs, exact: at most 16 * 127.
__device__ __forceinline__ float input_sum(const HalfRun& x) {
    __half2 s = __hadd2(x.even[0], x.odd[0]);
#pragma unroll
    for (unsigned i = 1; i < 4; ++i) s = __hadd2(s, __hadd2(x.even[i], x.odd[i]));
    const float2 f = __half22float2(s);
    return __fadd_rn(f.x, f.y);
}

__device__ __forceinline__ float kq_term_f(float scale, float min, float xd, float xs, float q) {
    return fmaf(__fmul_rn(scale, xd), q, -__fmul_rn(min, xs));
}

// Q4_K and Q5_K in f16: lane j's two runs (low and high nibbles, see
// K4Lane) and the scale and min of its sub-block 2c + h.
struct K4Half {
    struct Block {
        HalfRun lo, hi;
        float2 group;
    };
    struct Input {
        HalfRun lo, hi;
        float2 d;
    };

    template <class F>
    __device__ static Block load(const uint8_t* w, size_t total, size_t blk, unsigned j) {
        const typename F::Block k = F::load(w, total, blk, j);
        return {weight_run(k.lo), weight_run(k.hi), k4_group(k.head, j)};
    }

    __device__ static Input input(unsigned b, unsigned j, Q8Input x) {
        const unsigned c = j / 2, h = j % 2;
        const int8_t* xp = x.qs + b * 256 + 64 * c + 16 * h;
        return {input_run(xp), input_run(xp + 32), __ldg(x.ds + b * 8 + 2 * c + h)};
    }

    __device__ static float term(const Block& k, const Input& x, unsigned j) {
        float lo = half_dot(k.lo, x.lo);
        float hi = half_dot(k.hi, x.hi);
        lo = __fadd_rn(lo, __shfl_xor_sync(FULL_MASK, lo, 1));
        hi = __fadd_rn(hi, __shfl_xor_sync(FULL_MASK, hi, 1));
        return kq_term_f(k.group.x, k.group.y, x.d.x, x.d.y, j % 2 ? hi : lo);
    }
};

struct HQ4K : K4Half {
    __device__ static Block load(const uint8_t* w, size_t total, size_t blk, unsigned j) {
        return K4Half::load<KqQ4K>(w, total, blk, j);
    }
};

struct HQ5K : K4Half {
    __device__ static Block load(const uint8_t* w, size_t total, size_t blk, unsigned j) {
        return K4Half::load<KqQ5K>(w, total, blk, j);
    }
};

// Q6_K in f16: lane j's two runs of 16 as stored (+ 32).
struct HQ6K {
    struct Block {
        HalfRun q[2];
        float scale[2];
    };
    struct Input {
        HalfRun run[2];
        // 32 times each run's input sum, the stored bias's share.
        float bias[2];
        float d;
    };

    __device__ static Block load(const uint8_t* w, size_t total, size_t blk, unsigned j) {
        const KqQ6K::Block k = KqQ6K::load(w, total, blk, j);
        return {{weight_run(k.q[0]), weight_run(k.q[1])}, {k.scale[0], k.scale[1]}};
    }

    __device__ static Input input(unsigned b, unsigned j, Q8Input x) {
        const int8_t* xp = x.qs + b * 256 + 32 * j;
        Input in;
        in.run[0] = input_run(xp);
        in.run[1] = input_run(xp + 16);
#pragma unroll
        for (unsigned r = 0; r < 2; ++r) in.bias[r] = __fmul_rn(32.0f, input_sum(in.run[r]));
        in.d = __ldg(x.ds + b * 8 + j).x;
        return in;
    }

    __device__ static float term(const Block& k, const Input& x, unsigned) {
        float t[2];
#pragma unroll
        for (unsigned r = 0; r < 2; ++r) {
            const float q = __fsub_rn(half_dot(k.q[r], x.run[r]), x.bias[r]);
            t[r] = kq_term_f(k.scale[r], 0.0f, x.d, 0.0f, q);
        }
        return __fadd_rn(t[0], t[1]);
    }
};

// gemv_kq in f16 pairs, KQ_HALF_ROWS rows to a warp sharing each input's
// widening; a row's arithmetic does not depend on N or on its neighbour.
template <class H, unsigned N>
__device__ __forceinline__ void gemv_kq_half(const uint8_t* w, Q8Input x, float* y, unsigned rows,
                                             unsigned blocks_per_row, size_t total) {
    constexpr unsigned R = KQ_HALF_ROWS;
    const unsigned row0 = (blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP) * R;
    if (row0 >= rows) return;
    const unsigned lane = threadIdx.x % WARP, v = lane / 8, j = lane % 8;
    const unsigned len = blocks_per_row * 256;

    float acc[R][N] = {};
    for (unsigned base = 0; base < blocks_per_row; base += KQ_SPLIT) {
        const unsigned b = base + v;
        const bool active = b < blocks_per_row;
        typename H::Block blk[R];
#pragma unroll
        for (unsigned r = 0; r < R; ++r) {
            // Rows past the end repeat the last one; their sums are dropped.
            const size_t row = min(row0 + r, rows - 1);
            blk[r] = H::load(w, total, row * blocks_per_row + (active ? b : 0), j);
        }
#pragma unroll
        for (unsigned i = 0; i < N; ++i) {
            // Idle groups read block 0; their sums are dropped.
            const typename H::Input in = H::input(active ? b : 0, j, batch_row(x, i, len));
#pragma unroll
            for (unsigned r = 0; r < R; ++r) {
                float t = H::term(blk[r], in, j);
#pragma unroll
                for (unsigned o = 1; o < 8; o *= 2) t = __fadd_rn(t, __shfl_xor_sync(FULL_MASK, t, o));
                if (active) acc[r][i] = __fadd_rn(acc[r][i], t);
            }
        }
    }
#pragma unroll
    for (unsigned r = 0; r < R; ++r)
#pragma unroll
        for (unsigned i = 0; i < N; ++i) {
            float s = __shfl_sync(FULL_MASK, acc[r][i], 0);
#pragma unroll
            for (unsigned p = 1; p < KQ_SPLIT; ++p) s = __fadd_rn(s, __shfl_sync(FULL_MASK, acc[r][i], 8 * p));
            if (lane == 0 && row0 + r < rows) y[static_cast<size_t>(i) * rows + row0 + r] = s;
        }
}

// The kernels take the f16 path where KQ_HALF holds, with KQ_HALF_ROWS rows
// a warp (the host launches blocks of 4 warps either way).
#define GEMV_KERNEL_KQ(name, F, N)                                                          \
    extern "C" __global__ void __launch_bounds__(128)                                       \
        name##_##N(const uint8_t* w, const int8_t* xq, const float2* xds, float* y,        \
                   unsigned rows, unsigned blocks_per_row, size_t total) {                  \
        if constexpr (KQ_HALF)                                                              \
            gemv_kq_half<H##F, N>(w, Q8Input{xq, xds}, y, rows, blocks_per_row, total);     \
        else                                                                                \
            gemv_kq<Kq##F, N>(w, Q8Input{xq, xds}, y, rows, blocks_per_row, total);         \
    }

GEMV_KERNELS(GEMV_KERNEL_KQ, gemv_q4_k, Q4K)
GEMV_KERNELS(GEMV_KERNEL_KQ, gemv_q5_k, Q5K)
GEMV_KERNELS(GEMV_KERNEL_KQ, gemv_q6_k, Q6K)
