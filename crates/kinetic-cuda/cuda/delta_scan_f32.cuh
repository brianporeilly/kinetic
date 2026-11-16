#pragma once

#include "delta_chunk.cuh"

// The chunk scan of delta_scan.cuh in f32 on the CUDA cores, for devices
// without tensor cores: the same products of the same factors, summed in
// another order. A block owns 32 rows of one head's state and walks the
// chunks in order; thread t keeps row t / 4 in registers, columns
// 16 j + 4 (t % 4) .. + 3 for j = 0 .. 7, so that the four threads of a row
// read 64 consecutive bytes of a staged token. Per chunk the block stages
// W and Q by token, each thread forms its row's partial dot products with
// every token over its 32 columns and the four threads of the row add
// theirs; then U = T diag(b) V - W S^T, O = diag(e^G) Q S^T + P U and
// S = e^(G_C) S + U^T K' follow, K' staged where W was.

constexpr unsigned DSCAN32_THREADS = 128;

struct DeltaScan32Shared {
    // W, then K', by token.
    float a[DCHUNK][DSCAN_DIM + 4];
    float q[DCHUNK][DSCAN_DIM + 4];
    // U by token and row, and the chunk scores P.
    float u[DCHUNK][DSCAN_ROWS + 1];
    float p[DCHUNK][DCHUNK + 1];
    float g[DCHUNK];
};

// Stages rows of `rows` tokens, `stride` floats apart from `src`, times
// `scale(token)`, into `dst`; tokens past `rows` are zero.
template <class F>
__device__ __forceinline__ void scan32_stage(float (&dst)[DCHUNK][DSCAN_DIM + 4], const float* src,
                                             size_t stride, unsigned rows, F scale) {
    // thread t: token t / 4, columns 32 (t % 4) .. + 31
    const unsigned i = threadIdx.x / 4, c = 32 * (threadIdx.x % 4);
    const float f = i < rows ? scale(i) : 0.0f;
    const float* s = src + static_cast<size_t>(min(i, rows - 1)) * stride + c;
#pragma unroll
    for (unsigned q = 0; q < 32; q += 4) {
        float4 v = __ldg(reinterpret_cast<const float4*>(s + q));
        v.x *= f;
        v.y *= f;
        v.z *= f;
        v.w *= f;
        *reinterpret_cast<float4*>(&dst[i][c + q]) = v;
    }
}

// grid (n_v_heads, head_v_dim / DSCAN_ROWS), block DSCAN32_THREADS;
// head_k_dim = head_v_dim = DSCAN_DIM.
extern "C" __global__ void __launch_bounds__(DSCAN32_THREADS) delta_chunk_scan_f32(DeltaArgs a) {
    __shared__ __align__(16) DeltaScan32Shared sh;
    const unsigned vh = blockIdx.x, kh = vh % a.n_k_heads, tid = threadIdx.x;
    const unsigned r = tid / 4, cg = tid % 4, row = blockIdx.y * DSCAN_ROWS + r;
    const unsigned key_dim = a.n_k_heads * DSCAN_DIM, vd = a.n_v_heads * DSCAN_DIM;
    const size_t head = static_cast<size_t>(vh) * DSCAN_DIM * DSCAN_DIM;
    const DeltaBatch db = delta_batch(a);
    const auto col = [cg](unsigned j) { return 16 * j + 4 * cg; };

    float s[8][4];
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
        const float4 v = *reinterpret_cast<const float4*>(a.s_in + head + static_cast<size_t>(row) * DSCAN_DIM + col(j));
        s[j][0] = v.x;
        s[j][1] = v.y;
        s[j][2] = v.z;
        s[j][3] = v.w;
    }

    for (unsigned c0 = 0; c0 < a.n; c0 += DCHUNK) {
        const unsigned rows = min(DCHUNK, a.n - c0);
        const size_t tok = static_cast<size_t>(vh) * a.n + c0;
        const float* conv = db.conv + static_cast<size_t>(c0) * a.channels;
        if (tid < DCHUNK) sh.g[tid] = tid < rows ? db.gamma[tok + tid] : 0.0f;
        const auto one = [](unsigned) { return 1.0f; };
        scan32_stage(sh.a, db.w + tok * DSCAN_DIM, DSCAN_DIM, rows, one);
        scan32_stage(sh.q, conv + kh * DSCAN_DIM, a.channels, rows, one);
        __syncthreads();

        // W S^T and Q S^T of this row, over all the row's columns.
        float wu[DCHUNK], o[DCHUNK];
#pragma unroll
        for (unsigned i = 0; i < DCHUNK; ++i) {
            float x = 0.0f, y = 0.0f;
#pragma unroll
            for (unsigned j = 0; j < 8; ++j) {
                const float4 w = *reinterpret_cast<const float4*>(&sh.a[i][col(j)]);
                const float4 qv = *reinterpret_cast<const float4*>(&sh.q[i][col(j)]);
                x = fmaf(w.x, s[j][0], fmaf(w.y, s[j][1], fmaf(w.z, s[j][2], fmaf(w.w, s[j][3], x))));
                y = fmaf(qv.x, s[j][0], fmaf(qv.y, s[j][1], fmaf(qv.z, s[j][2], fmaf(qv.w, s[j][3], y))));
            }
            x += __shfl_xor_sync(FULL_MASK, x, 1);
            x += __shfl_xor_sync(FULL_MASK, x, 2);
            y += __shfl_xor_sync(FULL_MASK, y, 1);
            y += __shfl_xor_sync(FULL_MASK, y, 2);
            wu[i] = x;
            o[i] = y;
        }
        const float last = sh.g[rows - 1];
        __syncthreads();  // W and Q are read

        // U = T diag(b) V - W S^T: the row's threads take 8 tokens each.
#pragma unroll
        for (unsigned i = 0; i < DCHUNK; ++i)
            if (i / 8 == cg)
                sh.u[i][r] = i < rows ? db.u[(tok + i) * DSCAN_DIM + row] - wu[i] : 0.0f;
        const auto decay = [&](unsigned i) { return expf(last - sh.g[i]); };
        scan32_stage(sh.a, conv + key_dim + kh * DSCAN_DIM, a.channels, rows, decay);
        for (unsigned at = tid; at < DCHUNK * DCHUNK; at += DSCAN32_THREADS) {
            const unsigned i = at / DCHUNK, j = at % DCHUNK;
            sh.p[i][j] = i < rows ? db.p[(tok + i) * DCHUNK + j] : 0.0f;
        }
        __syncthreads();

        // O = diag(e^G) Q S^T + P U, the same 8 tokens per thread.
#pragma unroll
        for (unsigned i = 0; i < DCHUNK; ++i) {
            if (i / 8 != cg || i >= rows) continue;
            float v = expf(sh.g[i]) * o[i];
#pragma unroll 8
            for (unsigned j = 0; j < DCHUNK; ++j) v = fmaf(sh.p[i][j], sh.u[j][r], v);
            a.out[static_cast<size_t>(c0 + i) * vd + vh * DSCAN_DIM + row] = v;
        }

        // S = e^(G_C) S + U^T K'.
        const float keep = expf(last);
#pragma unroll
        for (unsigned j = 0; j < 8; ++j)
#pragma unroll
            for (unsigned e = 0; e < 4; ++e) s[j][e] *= keep;
#pragma unroll 4
        for (unsigned i = 0; i < DCHUNK; ++i) {
            const float ui = sh.u[i][r];
#pragma unroll
            for (unsigned j = 0; j < 8; ++j) {
                const float4 k = *reinterpret_cast<const float4*>(&sh.a[i][col(j)]);
                s[j][0] = fmaf(ui, k.x, s[j][0]);
                s[j][1] = fmaf(ui, k.y, s[j][1]);
                s[j][2] = fmaf(ui, k.z, s[j][2]);
                s[j][3] = fmaf(ui, k.w, s[j][3]);
            }
        }
        __syncthreads();  // the next chunk overwrites the staged tokens
    }

#pragma unroll
    for (unsigned j = 0; j < 8; ++j)
        *reinterpret_cast<float4*>(a.s_out + head + static_cast<size_t>(row) * DSCAN_DIM + col(j)) =
            make_float4(s[j][0], s[j][1], s[j][2], s[j][3]);
}
