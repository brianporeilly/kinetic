enable f16;
enable wgpu_cooperative_matrix;

// Y = W X for a pass of many tokens with Q8_0 weights, on the matrix units
// of devices with cooperative 16x16x16 f16 products into f32. A workgroup
// of four subgroups takes 64 rows of W by 64 tokens; per block of 32
// columns it expands the weights to f16 in workgroup memory, and the f32
// inputs to f16 beside them, and each subgroup multiplies a 32 by 32
// quarter of the tile as two by two products of 16. The sums are f32, of
// f16 terms: close to the int8 GEMV's, not equal, and each token's outputs
// depend only on that token, so how a prompt is cut changes no bit.
//
// Bindings and push constants are gemv.wgsl's, so that the host dispatches
// both alike: the batch picks the tokens, output token t starts at
// `t * stride + offset`. The grid is (row tiles, token tiles of the batch's
// capacity); tiles past the pass's tokens return at once.

struct Shape {
    rows: u32,
    blocks_per_row: u32,
    stride: u32,
    offset: u32,
    head: u32,
}

var<immediate> shape: Shape;

@group(0) @binding(0) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> xq: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> xd: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read> batch: array<u32>;

const TILE: u32 = 64u;
// f16 per row of a staged block: 32 and padding against bank conflicts
const LD: u32 = 40u;

// weights [row][k] and inputs [token][k] of one block of 32 columns
var<workgroup> wa: array<f16, 2560>;
var<workgroup> xb: array<f16, 2560>;
// the finished tile, [token][row]
var<workgroup> out: array<f32, 4096>;
// the batch's first token and token count
var<workgroup> span: vec2<u32>;

// Element `i` of the f16 plane of scales, which starts at word `base`.
fn scale_at(base: u32, i: u32) -> f32 {
    let pair = unpack2x16float(w[base + i / 8u][(i / 2u) % 4u]);
    return select(pair.x, pair.y, (i & 1u) == 1u);
}

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
    @builtin(subgroup_id) slot: u32,
) {
    if tid == 0u {
        var first = 0u;
        var tokens = batch[0];
        if shape.head != 0u {
            first = batch[2];
            tokens = batch[3];
        }
        span = vec2<u32>(first, tokens);
    }
    // the same for the whole workgroup, as the matrix products want to know
    let picked = workgroupUniformLoad(&span);
    let start = picked.x;
    let count = picked.y;
    let row0 = group.x * TILE;
    let tok0 = group.y * TILE;
    if tok0 >= count {
        return;
    }
    let n = shape.blocks_per_row;
    let total = shape.rows * n;
    let width4 = n * 8u;

    // staging: invocation tid takes half `half` of row (and token) tid / 2
    let r = tid / 2u;
    let half = tid % 2u;
    let wrow = min(row0 + r, shape.rows - 1u);
    let xrow = start + min(tok0 + r, count - 1u);

    // this subgroup's quarter: rows qr.., tokens qt..
    let qr = (slot / 2u) * 32u;
    let qt = (slot % 2u) * 32u;
    var c00 = coop_mat16x16<f32, C>();
    var c01 = coop_mat16x16<f32, C>();
    var c10 = coop_mat16x16<f32, C>();
    var c11 = coop_mat16x16<f32, C>();

    for (var kb = 0u; kb < n; kb++) {
        let blk = wrow * n + kb;
        let q = w[blk * 2u + half];
        let d = f16(scale_at(total * 2u, blk));
        let words = array<u32, 4>(q.x, q.y, q.z, q.w);
        for (var j = 0u; j < 4u; j++) {
            let v = unpack4xI8(words[j]);
            let at = r * LD + half * 16u + j * 4u;
            wa[at] = f16(v.x) * d;
            wa[at + 1u] = f16(v.y) * d;
            wa[at + 2u] = f16(v.z) * d;
            wa[at + 3u] = f16(v.w) * d;
        }
        let xi = xrow * width4 + kb * 8u + half * 4u;
        for (var j = 0u; j < 4u; j++) {
            let v = x[xi + j];
            let at = r * LD + half * 16u + j * 4u;
            xb[at] = f16(v.x);
            xb[at + 1u] = f16(v.y);
            xb[at + 2u] = f16(v.z);
            xb[at + 3u] = f16(v.w);
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 32u; kk += 16u) {
            let a0 = coopLoadT<coop_mat16x16<f16, A>>(&wa[qr * LD + kk], LD);
            let a1 = coopLoadT<coop_mat16x16<f16, A>>(&wa[(qr + 16u) * LD + kk], LD);
            let b0 = coopLoad<coop_mat16x16<f16, B>>(&xb[qt * LD + kk], LD);
            let b1 = coopLoad<coop_mat16x16<f16, B>>(&xb[(qt + 16u) * LD + kk], LD);
            c00 = coopMultiplyAdd(a0, b0, c00);
            c01 = coopMultiplyAdd(a0, b1, c01);
            c10 = coopMultiplyAdd(a1, b0, c10);
            c11 = coopMultiplyAdd(a1, b1, c11);
        }
        workgroupBarrier();
    }

    // out[token][row]: column-major stores of the row-by-token products
    coopStore(c00, &out[qt * TILE + qr], TILE);
    coopStore(c01, &out[(qt + 16u) * TILE + qr], TILE);
    coopStore(c10, &out[qt * TILE + qr + 16u], TILE);
    coopStore(c11, &out[(qt + 16u) * TILE + qr + 16u], TILE);
    workgroupBarrier();
    for (var i = tid; i < TILE * TILE; i += 128u) {
        let row = row0 + i % TILE;
        let t = tok0 + i / TILE;
        if row < shape.rows && t < count {
            y[t * shape.stride + shape.offset + row] = out[i];
        }
    }
}
