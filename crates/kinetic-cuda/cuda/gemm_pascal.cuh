#pragma once

#include <type_traits>

#include "gemm.cuh"

// Quantized matrix products for GPUs without tensor cores (Pascal, Volta's
// sm_70 build): Y = W X for batches larger than a GEMV's, in f32.
//
// A block takes a tile of 64 rows and 64 inputs. The k axis goes in stages
// of GEMM_K: the decoders of gemm.cuh write a stage's weights to shared
// memory as int8 with their scales, exactly as for the tensor-core kernels,
// and each sub-block of 32 is then expanded once into f32 weights next to
// the inputs of the same 32 elements, so that each value read from shared
// memory feeds four multiply-adds. Every thread owns four rows by four
// inputs and adds their terms in k order, one f32 fused multiply-add each;
// the tile a batch lands in does not change an output's bits.
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

constexpr unsigned PGEMM_ROWS = 64;
constexpr unsigned PGEMM_COLS = 64;
constexpr unsigned PGEMM_THREADS = GEMM_PARTS * PGEMM_ROWS;
// Each thread's patch: PGEMM_TR rows by PGEMM_TC inputs.
constexpr unsigned PGEMM_TR = 4, PGEMM_TC = 4;
// Padding of the k-major f32 tiles, which keeps their rows 16-byte aligned.
constexpr unsigned PGEMM_PAD = 4;
static_assert((PGEMM_ROWS / PGEMM_TR) * (PGEMM_COLS / PGEMM_TC) == PGEMM_THREADS, "a patch each");
static_assert(PGEMM_TR == 4 && PGEMM_TC == 4, "patches are read as float4");

// The int8 stage the decoders write: the layout of GemmStage's weights and
// scales (gemm_put_scale), without the int8 inputs.
struct __align__(16) PGemmStage {
    int8_t w[PGEMM_ROWS][GEMM_ROW];
    float4 scale[GEMM_SUBS][PGEMM_ROWS / 2];
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 600 || __CUDA_ARCH__ == 700)
constexpr bool PGEMM_HALF = true;
#else
constexpr bool PGEMM_HALF = false;
#endif

// One sub-block of 32 elements, k-major: weights by row, inputs by input,
// in f32, or in f16 pairs of consecutive k with the inputs' block scales.
struct __align__(16) PGemmF32 {
    float w[32][PGEMM_ROWS + PGEMM_PAD];
    float x[32][PGEMM_COLS + PGEMM_PAD];
};

struct __align__(16) PGemmF16 {
    __half2 w[16][PGEMM_ROWS + PGEMM_PAD];
    __half2 x[16][PGEMM_COLS + PGEMM_PAD];
    float d[PGEMM_COLS];
};

template <bool HALF>
struct __align__(16) PGemmShared {
    PGemmStage st;
    typename std::conditional<HALF, PGemmF16, PGemmF32>::type t;
};

// Scale slot `slot` (0: a, 1: b) of row `row` in sub-block `s`; the read
// that matches gemm_put_scale.
__device__ __forceinline__ float pgemm_scale(const PGemmStage& st, unsigned s, unsigned slot,
                                             unsigned row) {
    const float* f = &st.scale[s][row / 16 * 8 + row % 8].x;
    return f[2 * slot + row / 8 % 2];
}

// Expands sub-block `s` of the stage into f32 weights: a q + b with the
// sub-block's scale a and negated min b (Q4_K, Q5_K), a q (one run), or
// each run of 16 times its own scale (Q6_K). Thread t takes row t / 4 and
// elements 8 (t % 4) .. + 7.
template <class F, bool HALF>
__device__ __forceinline__ void pgemm_expand(PGemmShared<HALF>& sh, unsigned s) {
    const unsigned row = threadIdx.x / 4, part = threadIdx.x % 4;
    const unsigned run = F::RUNS == 2 ? part / 2 : 0;
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

// A thread's share of one sub-block of the block's inputs, read a sub-block
// ahead: thread t takes input t / 4 and elements 8 (t % 4) .. + 7 of the
// int8 copy, and the block's scale. Inputs past the batch repeat the last
// one; their results are not stored.
struct PGemmInputs {
    int2 q;
    float d;

    __device__ __forceinline__ void load(Q8Input x, unsigned cols, unsigned in0, unsigned n,
                                         unsigned k0) {
        const unsigned in = threadIdx.x / 4, part = threadIdx.x % 4;
        const size_t at = static_cast<size_t>(min(in0 + in, n - 1)) * cols + k0;
        q = __ldg(reinterpret_cast<const int2*>(x.qs + at + 8 * part));
        d = __ldg(x.ds + at / 32).x;
    }

    // Writes them k-major: as f32 values, or as their int8 values in f16
    // pairs and the scale apart.
    template <bool HALF>
    __device__ __forceinline__ void store(PGemmShared<HALF>& sh) const {
        const unsigned in = threadIdx.x / 4, part = threadIdx.x % 4;
        const unsigned words[2] = {static_cast<unsigned>(q.x), static_cast<unsigned>(q.y)};
        float v[8];
#pragma unroll
        for (unsigned i = 0; i < 8; ++i)
            v[i] = static_cast<float>(static_cast<int8_t>(words[i / 4] >> (8 * (i % 4))));
        if constexpr (HALF) {
#pragma unroll
            for (unsigned i = 0; i < 4; ++i)
                sh.t.x[4 * part + i][in] = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
            if (part == 0) sh.t.d[in] = d;
        } else {
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) sh.t.x[8 * part + i][in] = d * v[i];
        }
    }
};

// Adds the products of the sub-block in shared memory to `acc`.
__device__ __forceinline__ void pgemm_sub(float (&acc)[PGEMM_TR][PGEMM_TC],
                                          const PGemmShared<false>& sh, unsigned tr,
                                          unsigned tc) {
#pragma unroll 8
    for (unsigned e = 0; e < 32; ++e) {
        const float4 w4 = *reinterpret_cast<const float4*>(&sh.t.w[e][PGEMM_TR * tr]);
        const float4 x4 = *reinterpret_cast<const float4*>(&sh.t.x[e][PGEMM_TC * tc]);
        const float wv[PGEMM_TR] = {w4.x, w4.y, w4.z, w4.w};
        const float xv[PGEMM_TC] = {x4.x, x4.y, x4.z, x4.w};
#pragma unroll
        for (unsigned i = 0; i < PGEMM_TR; ++i)
#pragma unroll
            for (unsigned j = 0; j < PGEMM_TC; ++j) acc[i][j] = fmaf(wv[i], xv[j], acc[i][j]);
    }
}

__device__ __forceinline__ void pgemm_sub(float (&acc)[PGEMM_TR][PGEMM_TC],
                                          const PGemmShared<true>& sh, unsigned tr,
                                          unsigned tc) {
    __half2 h[PGEMM_TR][PGEMM_TC];
#pragma unroll
    for (unsigned i = 0; i < PGEMM_TR; ++i)
#pragma unroll
        for (unsigned j = 0; j < PGEMM_TC; ++j) h[i][j] = __float2half2_rn(0.0f);
#pragma unroll 8
    for (unsigned e = 0; e < 16; ++e) {
        const uint4 w4 = *reinterpret_cast<const uint4*>(&sh.t.w[e][PGEMM_TR * tr]);
        const uint4 x4 = *reinterpret_cast<const uint4*>(&sh.t.x[e][PGEMM_TC * tc]);
        const unsigned wu[PGEMM_TR] = {w4.x, w4.y, w4.z, w4.w};
        const unsigned xu[PGEMM_TC] = {x4.x, x4.y, x4.z, x4.w};
#pragma unroll
        for (unsigned i = 0; i < PGEMM_TR; ++i)
#pragma unroll
            for (unsigned j = 0; j < PGEMM_TC; ++j)
                h[i][j] = __hfma2(*reinterpret_cast<const __half2*>(&wu[i]),
                                  *reinterpret_cast<const __half2*>(&xu[j]), h[i][j]);
    }
    const float4 d4 = *reinterpret_cast<const float4*>(&sh.t.d[PGEMM_TC * tc]);
    const float d[PGEMM_TC] = {d4.x, d4.y, d4.z, d4.w};
#pragma unroll
    for (unsigned i = 0; i < PGEMM_TR; ++i)
#pragma unroll
        for (unsigned j = 0; j < PGEMM_TC; ++j) {
            const float2 f = __half22float2(h[i][j]);
            acc[i][j] = fmaf(d[j], f.x + f.y, acc[i][j]);
        }
}

// grid (ceil(n / PGEMM_COLS), ceil(rows / PGEMM_ROWS)), block PGEMM_THREADS.
// `cols` is a multiple of GEMM_K; output i of row r goes to y[i * rows + r].
template <class F>
__device__ __forceinline__ void pgemm(const uint8_t* w, Q8Input x, float* y, unsigned rows,
                                      unsigned cols, size_t total, unsigned n) {
    __shared__ PGemmShared<PGEMM_HALF> sh;
    const unsigned row0 = blockIdx.y * PGEMM_ROWS, in0 = blockIdx.x * PGEMM_COLS;
    const unsigned stages = cols / GEMM_K;
    // Decoding: thread (own, u) reads part u of row own's stage. Rows past
    // the end repeat the last one; their results are not stored.
    const unsigned own = threadIdx.x / GEMM_PARTS, u = threadIdx.x % GEMM_PARTS;
    const size_t blk = static_cast<size_t>(min(row0 + own, rows - 1)) * (cols / F::BLOCK);
    // Multiplying: rows 4 tr .. + 3 and inputs 4 tc .. + 3, each four read
    // as one float4; a warp's lanes read 16 distinct inputs and 2 rows.
    const unsigned tr = threadIdx.x / (PGEMM_COLS / PGEMM_TC);
    const unsigned tc = threadIdx.x % (PGEMM_COLS / PGEMM_TC);

    float acc[PGEMM_TR][PGEMM_TC] = {};
    typename F::Raw raw;
    PGemmInputs xin;
    F::load(raw, w, total, blk, 0, u);
    xin.load(x, cols, in0, n, 0);
    for (unsigned k = 0; k < stages; ++k) {
        __syncthreads();  // the previous stage is no longer read
        F::store(raw, sh.st, own, k, u);
        if (k + 1 < stages) F::load(raw, w, total, blk, k + 1, u);
#pragma unroll 1
        for (unsigned s = 0; s < GEMM_SUBS; ++s) {
            __syncthreads();  // the stage is written, the previous sub-block read
            pgemm_expand<F>(sh, s);
            xin.store(sh);
            const unsigned next = k * GEMM_K + 32 * (s + 1);
            if (next < cols) xin.load(x, cols, in0, n, next);
            __syncthreads();
            pgemm_sub(acc, sh, tr, tc);
        }
    }

#pragma unroll
    for (unsigned i = 0; i < PGEMM_TR; ++i)
#pragma unroll
        for (unsigned j = 0; j < PGEMM_TC; ++j) {
            const unsigned row = row0 + PGEMM_TR * tr + i, in = in0 + PGEMM_TC * tc + j;
            if (row < rows && in < n) y[static_cast<size_t>(in) * rows + row] = acc[i][j];
        }
}

// The signature of gemm.cuh's kernels, so that gemm.rs launches both alike;
// `up`, `yq`, `yds` and `gelu_gate` are unused (gated products are not
// fused here).
#define PGEMM_KERNEL(name, F)                                                                   \
    extern "C" __global__ void __launch_bounds__(PGEMM_THREADS, 2)                              \
        name(const uint8_t* w, const uint8_t*, const int8_t* xq, const float2* xds, float* y,   \
             int8_t*, float2*, unsigned rows, unsigned cols, size_t total, unsigned n,          \
             unsigned) {                                                                        \
        pgemm<F>(w, Q8Input{xq, xds}, y, rows, cols, total, n);                                 \
    }

PGEMM_KERNEL(pgemm_q4_k, GemmK4<false>)
PGEMM_KERNEL(pgemm_q5_k, GemmK4<true>)
PGEMM_KERNEL(pgemm_q6_k, GemmQ6K)
PGEMM_KERNEL(pgemm_q8_0, GemmQ8_0)
PGEMM_KERNEL(pgemm_q4_0, GemmQ4_0)
PGEMM_KERNEL(pgemm_pq2_0, GemmPQ2_0)
