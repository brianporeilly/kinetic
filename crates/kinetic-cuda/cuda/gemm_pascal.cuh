#pragma once

#include <type_traits>

#include "gemm.cuh"

// Quantized matrix products for GPUs without tensor cores (Pascal, Volta's
// sm_70 build): Y = W X for batches larger than a GEMV's, in f32.
//
// A block of 256 threads takes a tile of 16 TR rows and 16 TC inputs. The k
// axis goes in stages of GEMM_K: the decoders of gemm.cuh write a stage's
// weights to shared memory as int8 with their scales, exactly as for the
// tensor-core kernels, and each sub-block of 32 is then expanded once into
// f32 weights next to the inputs of the same 32 elements, so that each value
// read from shared memory feeds TR or TC multiply-adds. Every thread owns
// TR rows by TC inputs, in groups of four 64 apart, and adds their terms in
// k order, one f32 fused multiply-add each; the tile a batch lands in does
// not change an output's bits. Narrow tiles (4 by 4) suit small matrices,
// wide ones (8 by 8, or 8 by 4 in f32, whose expanded sub-blocks would not
// fit the shared memory) read half as much shared memory per product; the
// host takes the narrow ones for f16 pairs, where the wide ones measured
// slower on the P100.
//
// The inputs are the GEMV input's int8 copy times its block scales, the
// values the GEMV and the tensor-core kernels multiply; the sums may differ
// from theirs in the last bits.
//
// GPUs whose f16 arithmetic runs at twice the f32 rate (the Tesla P100,
// sm_60, and the V100, sm_70) multiply in f16 pairs instead: the weights
// expanded to f16, the inputs as their int8 values, which f16 holds exactly,
// each pair of products of consecutive k added into an f16 pair. At the end
// of each sub-block of 32, which holds 16 terms in each half, the two halves
// are added in f32, times the input's block scale, into the f32 sums. Only
// the weights and those short sums round; the results differ from the f32
// kernel's by about one part in a thousand of the sums' terms. Must match
// Kernels::half_products.

constexpr unsigned PGEMM_THREADS = 256;
// Padding of the k-major tiles, which keeps their rows 16-byte aligned.
constexpr unsigned PGEMM_PAD = 4;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 600 || __CUDA_ARCH__ == 700)
constexpr bool PGEMM_HALF = true;
#else
constexpr bool PGEMM_HALF = false;
#endif

// The int8 stage the decoders write: the layout of GemmStage's weights and
// scales (gemm_put_scale), without the int8 inputs.
template <unsigned ROWS>
struct __align__(16) PGemmStage {
    int8_t w[ROWS][GEMM_ROW];
    float4 scale[GEMM_SUBS][ROWS / 2];
};

// One sub-block of 32 elements, k-major: weights by row, inputs by input,
// in f32, or in f16 pairs of consecutive k with the inputs' block scales.
template <unsigned ROWS, unsigned COLS>
struct __align__(16) PGemmF32 {
    float w[32][ROWS + PGEMM_PAD];
    float x[32][COLS + PGEMM_PAD];
};

template <unsigned ROWS, unsigned COLS>
struct __align__(16) PGemmF16 {
    __half2 w[16][ROWS + PGEMM_PAD];
    __half2 x[16][COLS + PGEMM_PAD];
    float d[COLS];
};

template <unsigned TR, unsigned TC, bool HALF>
struct PGemmTile {
    static constexpr unsigned ROWS = 16 * TR, COLS = 16 * TC;
    static_assert(TR % 4 == 0 && TC % 4 == 0, "patches are read as groups of four");
    struct __align__(16) Shared {
        PGemmStage<ROWS> st;
        typename std::conditional<HALF, PGemmF16<ROWS, COLS>, PGemmF32<ROWS, COLS>>::type t;
    };
};

// Scale slot `slot` (0: a, 1: b) of row `row` in sub-block `s`; the read
// that matches gemm_put_scale.
template <unsigned ROWS>
__device__ __forceinline__ float pgemm_scale(const PGemmStage<ROWS>& st, unsigned s, unsigned slot,
                                             unsigned row) {
    const float* f = &st.scale[s][row / 16 * 8 + row % 8].x;
    return f[2 * slot + row / 8 % 2];
}

// Expands sub-block `s` of the stage into f32 weights: a q + b with the
// sub-block's scale a and negated min b (Q4_K, Q5_K), a q (one run), or
// each run of 16 times its own scale (Q6_K). Thread t takes rows t / 4,
// t / 4 + 64, ... and elements 8 (t % 4) .. + 7.
template <class F, class T, bool HALF>
__device__ __forceinline__ void pgemm_expand(typename T::Shared& sh, unsigned s) {
    const unsigned part = threadIdx.x % 4;
    const unsigned run = F::RUNS == 2 ? part / 2 : 0;
#pragma unroll
    for (unsigned r0 = 0; r0 < T::ROWS; r0 += PGEMM_THREADS / 4) {
        const unsigned row = r0 + threadIdx.x / 4;
        const float a = pgemm_scale(sh.st, s, run, row);
        const float b = F::MINS ? pgemm_scale(sh.st, s, 1, row) : 0.0f;
        const int2 q = *reinterpret_cast<const int2*>(&sh.st.w[row][32 * s + 8 * part]);
        const unsigned words[2] = {static_cast<unsigned>(q.x), static_cast<unsigned>(q.y)};
        float v[8];
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            const float e = static_cast<float>(static_cast<int8_t>(words[i / 4] >> (8 * (i % 4))));
            v[i] = F::MINS ? fmaf(a, e, b) : a * e;
        }
        if constexpr (HALF) {
#pragma unroll
            for (unsigned i = 0; i < 4; ++i)
                sh.t.w[4 * part + i][row] = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
        } else {
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) sh.t.w[8 * part + i][row] = v[i];
        }
    }
}

// A thread's share of one sub-block of the block's inputs, read a sub-block
// ahead: thread t takes inputs t / 4, t / 4 + 64, ... and elements
// 8 (t % 4) .. + 7 of the int8 copy, and their blocks' scales. Inputs past
// the batch repeat the last one; their results are not stored.
template <class T>
struct PGemmInputs {
    static constexpr unsigned N = T::COLS / (PGEMM_THREADS / 4);
    int2 q[N];
    float d[N];

    __device__ __forceinline__ void load(Q8Input x, unsigned cols, unsigned in0, unsigned n,
                                         unsigned k0) {
        const unsigned part = threadIdx.x % 4;
#pragma unroll
        for (unsigned c = 0; c < N; ++c) {
            const unsigned in = c * (PGEMM_THREADS / 4) + threadIdx.x / 4;
            const size_t at = static_cast<size_t>(min(in0 + in, n - 1)) * cols + k0;
            q[c] = __ldg(reinterpret_cast<const int2*>(x.qs + at + 8 * part));
            d[c] = __ldg(x.ds + at / 32).x;
        }
    }

    // Writes them k-major: as f32 values, or as their int8 values in f16
    // pairs and the scale apart.
    template <bool HALF>
    __device__ __forceinline__ void store(typename T::Shared& sh) const {
        const unsigned part = threadIdx.x % 4;
#pragma unroll
        for (unsigned c = 0; c < N; ++c) {
            const unsigned in = c * (PGEMM_THREADS / 4) + threadIdx.x / 4;
            const unsigned words[2] = {static_cast<unsigned>(q[c].x), static_cast<unsigned>(q[c].y)};
            float v[8];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i)
                v[i] = static_cast<float>(static_cast<int8_t>(words[i / 4] >> (8 * (i % 4))));
            if constexpr (HALF) {
#pragma unroll
                for (unsigned i = 0; i < 4; ++i)
                    sh.t.x[4 * part + i][in] = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
                if (part == 0) sh.t.d[in] = d[c];
            } else {
#pragma unroll
                for (unsigned i = 0; i < 8; ++i) sh.t.x[8 * part + i][in] = d[c] * v[i];
            }
        }
    }
};

// Row i of a thread's patch, or input i: groups of four 64 apart.
__device__ __forceinline__ unsigned pgemm_at(unsigned t, unsigned i) { return 64 * (i / 4) + 4 * t + i % 4; }

// Adds the products of the sub-block in shared memory to `acc`.
template <unsigned TR, unsigned TC, class S>
__device__ __forceinline__ void pgemm_sub(float (&acc)[TR][TC], const S& sh, unsigned tr, unsigned tc,
                                          std::false_type) {
#pragma unroll 4
    for (unsigned e = 0; e < 32; ++e) {
        float wv[TR], xv[TC];
#pragma unroll
        for (unsigned g = 0; g < TR / 4; ++g)
            *reinterpret_cast<float4*>(&wv[4 * g]) = *reinterpret_cast<const float4*>(&sh.t.w[e][pgemm_at(tr, 4 * g)]);
#pragma unroll
        for (unsigned g = 0; g < TC / 4; ++g)
            *reinterpret_cast<float4*>(&xv[4 * g]) = *reinterpret_cast<const float4*>(&sh.t.x[e][pgemm_at(tc, 4 * g)]);
#pragma unroll
        for (unsigned i = 0; i < TR; ++i)
#pragma unroll
            for (unsigned j = 0; j < TC; ++j) acc[i][j] = fmaf(wv[i], xv[j], acc[i][j]);
    }
}

template <unsigned TR, unsigned TC, class S>
__device__ __forceinline__ void pgemm_sub(float (&acc)[TR][TC], const S& sh, unsigned tr, unsigned tc,
                                          std::true_type) {
    __half2 h[TR][TC];
#pragma unroll
    for (unsigned i = 0; i < TR; ++i)
#pragma unroll
        for (unsigned j = 0; j < TC; ++j) h[i][j] = __float2half2_rn(0.0f);
#pragma unroll 4
    for (unsigned e = 0; e < 16; ++e) {
        unsigned wu[TR], xu[TC];
#pragma unroll
        for (unsigned g = 0; g < TR / 4; ++g)
            *reinterpret_cast<uint4*>(&wu[4 * g]) = *reinterpret_cast<const uint4*>(&sh.t.w[e][pgemm_at(tr, 4 * g)]);
#pragma unroll
        for (unsigned g = 0; g < TC / 4; ++g)
            *reinterpret_cast<uint4*>(&xu[4 * g]) = *reinterpret_cast<const uint4*>(&sh.t.x[e][pgemm_at(tc, 4 * g)]);
#pragma unroll
        for (unsigned i = 0; i < TR; ++i)
#pragma unroll
            for (unsigned j = 0; j < TC; ++j)
                h[i][j] = __hfma2(*reinterpret_cast<const __half2*>(&wu[i]),
                                  *reinterpret_cast<const __half2*>(&xu[j]), h[i][j]);
    }
    float d[TC];
#pragma unroll
    for (unsigned g = 0; g < TC / 4; ++g)
        *reinterpret_cast<float4*>(&d[4 * g]) = *reinterpret_cast<const float4*>(&sh.t.d[pgemm_at(tc, 4 * g)]);
#pragma unroll
    for (unsigned i = 0; i < TR; ++i)
#pragma unroll
        for (unsigned j = 0; j < TC; ++j) {
            const float2 f = __half22float2(h[i][j]);
            acc[i][j] = fmaf(d[j], f.x + f.y, acc[i][j]);
        }
}

// grid (ceil(n / COLS), ceil(rows / ROWS)), block PGEMM_THREADS. `cols` is
// a multiple of GEMM_K; output i of row r goes to y[i * rows + r].
template <class F, unsigned TR, unsigned TC>
__device__ __forceinline__ void pgemm(const uint8_t* w, Q8Input x, float* y, unsigned rows,
                                      unsigned cols, size_t total, unsigned n) {
    using T = PGemmTile<TR, TC, PGEMM_HALF>;
    constexpr unsigned DECODES = T::ROWS / (PGEMM_THREADS / GEMM_PARTS);
    __shared__ typename T::Shared sh;
    const unsigned row0 = blockIdx.y * T::ROWS, in0 = blockIdx.x * T::COLS;
    const unsigned stages = cols / GEMM_K;
    // Decoding: thread (own, u) reads part u of rows own, own + 64, ...;
    // rows past the end repeat the last one, and their results are not
    // stored.
    const unsigned own = threadIdx.x / GEMM_PARTS, u = threadIdx.x % GEMM_PARTS;
    size_t blk[DECODES];
#pragma unroll
    for (unsigned r = 0; r < DECODES; ++r)
        blk[r] = static_cast<size_t>(min(row0 + own + 64 * r, rows - 1)) * (cols / F::BLOCK);
    // Multiplying: rows pgemm_at(tr, i), inputs pgemm_at(tc, j); a warp's
    // lanes read 16 distinct groups of inputs and 2 of rows.
    const unsigned tr = threadIdx.x / 16, tc = threadIdx.x % 16;

    float acc[TR][TC] = {};
    typename F::Raw raw[DECODES];
    PGemmInputs<T> xin;
#pragma unroll
    for (unsigned r = 0; r < DECODES; ++r) F::load(raw[r], w, total, blk[r], 0, u);
    xin.load(x, cols, in0, n, 0);
    for (unsigned k = 0; k < stages; ++k) {
        __syncthreads();  // the previous stage is no longer read
#pragma unroll
        for (unsigned r = 0; r < DECODES; ++r) {
            F::store(raw[r], sh.st, own + 64 * r, k, u);
            if (k + 1 < stages) F::load(raw[r], w, total, blk[r], k + 1, u);
        }
#pragma unroll 1
        for (unsigned s = 0; s < GEMM_SUBS; ++s) {
            __syncthreads();  // the stage is written, the previous sub-block read
            pgemm_expand<F, T, PGEMM_HALF>(sh, s);
            xin.template store<PGEMM_HALF>(sh);
            const unsigned next = k * GEMM_K + 32 * (s + 1);
            if (next < cols) xin.load(x, cols, in0, n, next);
            __syncthreads();
            pgemm_sub<TR, TC>(acc, sh, tr, tc, std::integral_constant<bool, PGEMM_HALF>{});
        }
    }

#pragma unroll
    for (unsigned i = 0; i < TR; ++i)
#pragma unroll
        for (unsigned j = 0; j < TC; ++j) {
            const unsigned row = row0 + pgemm_at(tr, i), in = in0 + pgemm_at(tc, j);
            if (row < rows && in < n) y[static_cast<size_t>(in) * rows + row] = acc[i][j];
        }
}

// Wide tiles: 8 by 8 in f16 pairs, 8 by 4 in f32.
constexpr unsigned PGEMM_WIDE_TC = PGEMM_HALF ? 8 : 4;

// The signature of gemm.cuh's kernels, so that gemm.rs launches both alike;
// `up`, `yq`, `yds` and `gelu_gate` are unused (gated products are not
// fused here). `name` takes 64 by 64 tiles, `name_wide` 128 by 128 (128
// by 64 in f32); must match PASCAL_TILES in gemm.rs.
#define PGEMM_KERNEL(name, F)                                                                   \
    extern "C" __global__ void __launch_bounds__(PGEMM_THREADS, 2)                              \
        name(const uint8_t* w, const uint8_t*, const int8_t* xq, const float2* xds, float* y,   \
             int8_t*, float2*, unsigned rows, unsigned cols, size_t total, unsigned n,          \
             unsigned) {                                                                        \
        pgemm<F, 4, 4>(w, Q8Input{xq, xds}, y, rows, cols, total, n);                           \
    }                                                                                           \
    extern "C" __global__ void __launch_bounds__(PGEMM_THREADS, 1)                              \
        name##_wide(const uint8_t* w, const uint8_t*, const int8_t* xq, const float2* xds,      \
                    float* y, int8_t*, float2*, unsigned rows, unsigned cols, size_t total,     \
                    unsigned n, unsigned) {                                                     \
        pgemm<F, 8, PGEMM_WIDE_TC>(w, Q8Input{xq, xds}, y, rows, cols, total, n);               \
    }

PGEMM_KERNEL(pgemm_q4_k, GemmK4<false>)
PGEMM_KERNEL(pgemm_q5_k, GemmK4<true>)
PGEMM_KERNEL(pgemm_q6_k, GemmQ6K)
PGEMM_KERNEL(pgemm_q8_0, GemmQ8_0)
PGEMM_KERNEL(pgemm_q4_0, GemmQ4_0)
PGEMM_KERNEL(pgemm_pq2_0, GemmPQ2_0)
