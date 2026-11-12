#pragma once

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

struct __align__(16) PGemmShared {
    PGemmStage st;
    // One sub-block of 32 elements, k-major: weights by row, inputs by input.
    float w[32][PGEMM_ROWS + PGEMM_PAD];
    float x[32][PGEMM_COLS + PGEMM_PAD];
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
template <class F>
__device__ __forceinline__ void pgemm_expand(PGemmShared& sh, unsigned s) {
    const unsigned row = threadIdx.x / 4, part = threadIdx.x % 4;
    const unsigned run = F::RUNS == 2 ? part / 2 : 0;
    const float a = pgemm_scale(sh.st, s, run, row);
    const float b = F::MINS ? pgemm_scale(sh.st, s, 1, row) : 0.0f;
    const int2 q = *reinterpret_cast<const int2*>(&sh.st.w[row][32 * s + 8 * part]);
    const unsigned words[2] = {static_cast<unsigned>(q.x), static_cast<unsigned>(q.y)};
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        const int v = static_cast<int8_t>(words[i / 4] >> (8 * (i % 4)));
        sh.w[8 * part + i][row] = F::MINS ? fmaf(a, static_cast<float>(v), b)
                                          : a * static_cast<float>(v);
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

    // Writes them as f32 values, k-major.
    __device__ __forceinline__ void store(PGemmShared& sh) const {
        const unsigned in = threadIdx.x / 4, part = threadIdx.x % 4;
        const unsigned words[2] = {static_cast<unsigned>(q.x), static_cast<unsigned>(q.y)};
#pragma unroll
        for (unsigned i = 0; i < 8; ++i)
            sh.x[8 * part + i][in] =
                d * static_cast<float>(static_cast<int8_t>(words[i / 4] >> (8 * (i % 4))));
    }
};

// grid (ceil(n / PGEMM_COLS), ceil(rows / PGEMM_ROWS)), block PGEMM_THREADS.
// `cols` is a multiple of GEMM_K; output i of row r goes to y[i * rows + r].
template <class F>
__device__ __forceinline__ void pgemm(const uint8_t* w, Q8Input x, float* y, unsigned rows,
                                      unsigned cols, size_t total, unsigned n) {
    __shared__ PGemmShared sh;
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
#pragma unroll 8
            for (unsigned e = 0; e < 32; ++e) {
                const float4 w4 = *reinterpret_cast<const float4*>(&sh.w[e][PGEMM_TR * tr]);
                const float4 x4 = *reinterpret_cast<const float4*>(&sh.x[e][PGEMM_TC * tc]);
                const float wv[PGEMM_TR] = {w4.x, w4.y, w4.z, w4.w};
                const float xv[PGEMM_TC] = {x4.x, x4.y, x4.z, x4.w};
#pragma unroll
                for (unsigned i = 0; i < PGEMM_TR; ++i)
#pragma unroll
                    for (unsigned j = 0; j < PGEMM_TC; ++j) acc[i][j] = fmaf(wv[i], xv[j], acc[i][j]);
            }
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
