use std::ffi::CString;
use std::ops::Range;
use std::rc::Rc;

use kinetic_quant::TensorType;

use crate::error::{Error, Result};
use crate::gemv::{weight_index, Act, DeviceMatrix, GemvInput, MAX_BATCH, WEIGHT_TYPES};
use crate::kernels::{arg, Kernels};
use crate::memory::DeviceBuffer;
use crate::module::{Function, LaunchConfig, Module};
use crate::stream::Stream;

/// The tiles of the quantized kernels, narrowest first, which gemm.cuh
/// names `<format kernel>_<inputs>`, and of the float kernels; must match
/// gemm.cuh and gemm_float.cuh.
const QUANT_TILES: [Tile; 3] = [
    Tile {
        rows: 64,
        inputs: 32,
        threads: 256,
        shared: 33_792,
    },
    Tile {
        rows: 64,
        inputs: 64,
        threads: 256,
        shared: 45_056,
    },
    Tile {
        rows: 128,
        inputs: 128,
        threads: 512,
        shared: 90_112,
    },
];
/// f16 weights on tensor cores; must match GEMMH_* in gemm_float.cuh.
const HALF_TILES: [Tile; 1] = [Tile {
    rows: 64,
    inputs: 128,
    threads: 128,
    shared: 0,
}];
/// Quantized weights without tensor cores, narrow and wide: the kernels
/// `p<format kernel>` and `p<format kernel>_wide` of gemm_pascal.cuh, whose
/// shared memory is static. The wide tile takes 128 inputs where the
/// products run in f16 pairs and 64 in f32; must match pgemm's tiles.
const PASCAL_TILES: [Tile; 2] = [
    Tile {
        rows: 64,
        inputs: 64,
        threads: 256,
        shared: 0,
    },
    Tile {
        rows: 128,
        inputs: 64,
        threads: 256,
        shared: 0,
    },
];
const PASCAL_HALF_TILES: [Tile; 2] = [
    PASCAL_TILES[0],
    Tile {
        rows: 128,
        inputs: 128,
        threads: 256,
        shared: 0,
    },
];
const FLOAT_TILES: [Tile; 1] = [Tile {
    rows: 32,
    inputs: 8,
    threads: 256,
    shared: 0,
}];
/// Elements per stage of the k loop; every row length must be a multiple.
const STAGE: usize = 128;
/// Without tensor cores, in f16 pairs: batches of at least this many inputs
/// expand their weights to f16 once and multiply those, with tiles of
/// `PF16_ROWS` rows and `PF16_COLS` inputs and rows unpacked
/// `PUNPACK_ROWS` to a block; must match gemm_pascal.cuh.
const UNPACK_INPUTS: usize = 256;
const PF16_ROWS: u32 = 128;
const PF16_COLS: u32 = 64;
const PUNPACK_ROWS: u32 = 64;
/// Inputs the split of the k axis is chosen for: a full prompt batch.
const SPLIT_INPUTS: u32 = 512;

/// Rows, inputs and threads per block, and dynamic shared bytes.
#[derive(Clone, Copy)]
struct Tile {
    rows: u32,
    inputs: u32,
    threads: u32,
    shared: u32,
}

/// The tiled kernel of each weight type that has one.
fn gemm_name(ty: TensorType) -> Option<&'static str> {
    Some(match ty {
        TensorType::F32 => "gemm_f32",
        TensorType::F16 => "gemm_f16",
        TensorType::Q8_0 => "gemm_q8_0",
        TensorType::Q4_0 => "gemm_q4_0",
        TensorType::Q4K => "gemm_q4_k",
        TensorType::Q5K => "gemm_q5_k",
        TensorType::Q6K => "gemm_q6_k",
        TensorType::PQ2_0 => "gemm_pq2_0",
        _ => return None,
    })
}

fn is_float(ty: TensorType) -> bool {
    matches!(ty, TensorType::F32 | TensorType::F16)
}

fn tiles(ty: TensorType, tensor_cores: bool, half: bool) -> &'static [Tile] {
    if !tensor_cores {
        if half {
            &PASCAL_HALF_TILES
        } else {
            &PASCAL_TILES
        }
    } else if ty == TensorType::F16 {
        &HALF_TILES
    } else if is_float(ty) {
        &FLOAT_TILES
    } else {
        &QUANT_TILES
    }
}

/// The matrix-product kernels of `module`, by index into `WEIGHT_TYPES` and
/// then by width, allowed the shared memory they take; none for a type
/// without a tiled kernel or where the device has less shared memory
/// (Turing), so those batches run as GEMVs. With `tensor_cores` they are
/// gemm.cuh's and gemm_float.cuh's; without, gemm_pascal.cuh's narrow and
/// wide kernels of the quantized types.
pub(crate) fn gemm_functions(
    module: &Rc<Module>,
    tensor_cores: bool,
) -> Result<Vec<Vec<Function>>> {
    WEIGHT_TYPES
        .iter()
        .map(|&ty| {
            let Some(name) = gemm_name(ty) else {
                return Ok(Vec::new());
            };
            if !tensor_cores {
                if is_float(ty) {
                    return Ok(Vec::new());
                }
                let named = |n: String| {
                    module.function(&CString::new(n).expect("kernel names have no NUL"))
                };
                return Ok(vec![
                    named(format!("p{name}"))?,
                    named(format!("p{name}_wide"))?,
                    named(format!("p{name}_unpack"))?,
                ]);
            }
            let mut fs = Vec::new();
            for tile in tiles(ty, true, false) {
                let name = if is_float(ty) {
                    name.to_owned()
                } else {
                    format!("{name}_{}", tile.inputs)
                };
                let f = module.function(&CString::new(name).expect("kernel names have no NUL"))?;
                if f.allow_shared(tile.shared).is_err() {
                    return Ok(Vec::new());
                }
                fs.push(f);
            }
            Ok(fs)
        })
        .collect()
}

/// One launch: `w[..rows]`, and for a gated product `up`, times inputs
/// `inputs` of `x`.
struct Product<'a> {
    w: &'a DeviceMatrix,
    up: Option<&'a DeviceMatrix>,
    rows: usize,
    x: &'a GemvInput,
    inputs: Range<usize>,
    tile: &'static Tile,
    /// GELU on a gated product's gate in place of SiLU.
    gelu: bool,
}

impl Kernels {
    /// Queues `y = w[..rows] x[inputs]` as a tiled matrix product, or
    /// returns `false` if the type and shape of `w` have no such kernel.
    /// Shapes are checked by the caller.
    pub(crate) fn gemm(
        &self,
        w: &DeviceMatrix,
        rows: usize,
        x: &GemvInput,
        inputs: Range<usize>,
        y: &mut DeviceBuffer<f32>,
        stream: &Stream,
    ) -> Result<bool> {
        let Some((kernel, tile)) = self.gemm_kernel(w, inputs.len()) else {
            return Ok(false);
        };
        if self.half
            && inputs.len() >= UNPACK_INPUTS
            && self.gemm_unpacked(w, rows, x, &inputs, y, stream)?
        {
            return Ok(true);
        }
        let p = Product {
            w,
            up: None,
            rows,
            x,
            inputs,
            tile,
            gelu: false,
        };
        let blocks = self.gemm_grid(&p);
        let splits = if is_float(w.ty()) || !self.tensor_cores {
            1
        } else {
            self.gemm_splits(rows, w.cols() / STAGE)
        };
        if splits > 1 {
            y.zero_prefix(p.inputs.len() * rows, stream)?;
        }
        let out = Act {
            x: y.device_ptr(),
            qs: 0,
            ds: 0,
        };
        self.launch_gemm(kernel, &p, out, [blocks[0], blocks[1], splits], stream)?;
        Ok(true)
    }

    /// Queues `out = silu(gate x) * (up x)`, or with `gelu` `gelu(gate x) *
    /// (up x)`, for inputs `inputs` of `x`, into
    /// the first `inputs.len()` inputs of `out`, as one matrix product that
    /// applies the activation to its sums and quantizes the result. Returns
    /// `false`, queuing nothing, for batches of up to [`MAX_BATCH`] inputs,
    /// which keep to the GEMV kernels, and for matrices without a quantized
    /// product kernel or of different types; the separate products and
    /// [`Kernels::swiglu`] then do the same.
    #[allow(clippy::too_many_arguments)]
    pub fn glu_matmul(
        &self,
        gate: &DeviceMatrix,
        up: &DeviceMatrix,
        gelu: bool,
        x: &GemvInput,
        inputs: Range<usize>,
        out: &mut GemvInput,
        stream: &Stream,
    ) -> Result<bool> {
        let shape_ok = up.rows() == gate.rows()
            && up.cols() == gate.cols()
            && gate.cols() == x.width()
            && out.width() == gate.rows()
            && inputs.end <= x.capacity();
        if !shape_ok {
            return Err(Error::Shape(format!(
                "gated product of {}x{} and {}x{} matrices from inputs of {} into {}",
                gate.rows(),
                gate.cols(),
                up.rows(),
                up.cols(),
                x.width(),
                out.width()
            )));
        }
        out.check_rows(inputs.len())?;
        let (kernel, tile) = match self.gemm_kernel(gate, inputs.len()) {
            Some(k)
                if inputs.len() > MAX_BATCH
                    && up.ty() == gate.ty()
                    && !is_float(gate.ty())
                    && self.tensor_cores =>
            {
                k
            }
            _ => return Ok(false),
        };
        let p = Product {
            w: gate,
            up: Some(up),
            rows: gate.rows(),
            x,
            inputs,
            tile,
            gelu,
        };
        let [bx, by] = self.gemm_grid(&p);
        self.launch_gemm(kernel, &p, out.act(), [bx, by, 1], stream)?;
        Ok(true)
    }

    /// The kernel for `n` inputs of `w` and its tile: the narrowest that
    /// takes them all, or the widest; without tensor cores, the wide tile
    /// where its blocks fill every multiprocessor, one each, unless the
    /// products run in f16 pairs.
    fn gemm_kernel(&self, w: &DeviceMatrix, n: usize) -> Option<(&Function, &'static Tile)> {
        // Formats without a tiled kernel run their prompts through the GEMV.
        let i = weight_index(w.ty())?;
        if !w.cols().is_multiple_of(STAGE) {
            return None;
        }
        let tiles = tiles(w.ty(), self.tensor_cores, self.half);
        let k = if self.tensor_cores {
            tiles
                .iter()
                .position(|t| n <= t.inputs as usize)
                .unwrap_or(tiles.len() - 1)
        } else {
            let wide = &tiles[1];
            let blocks = (n as u32).div_ceil(wide.inputs) * (w.rows() as u32).div_ceil(wide.rows);
            // perf: on a Tesla P100 the wide f16 tile ran at 6.4-8.3 TFLOPS
            // against the narrow one's 8.5-8.9; its stage leaves room for one
            // block per multiprocessor, too few warps to hide the latency
            usize::from(!self.half && blocks >= self.multiprocessors)
        };
        Some((self.gemm.get(i)?.get(k)?, &tiles[k]))
    }

    /// Blocks along the inputs and the rows; a gated product's block covers
    /// half as many rows of each matrix.
    fn gemm_grid(&self, p: &Product) -> [u32; 2] {
        let tile = p.tile;
        let rows = if p.up.is_some() {
            tile.rows / 2
        } else {
            tile.rows
        };
        [
            (p.inputs.len() as u32).div_ceil(tile.inputs),
            (p.rows as u32).div_ceil(rows),
        ]
    }

    /// Queues `y = w[..rows] x[inputs]` in f16 pairs as two launches per
    /// piece of the rows: the rows expanded to f16, into this stream's
    /// buffer of them, then the product of those, with the bits of the
    /// product in one launch. The pieces are as many rows as the buffer
    /// holds, all of them unless memory is short. Returns `false`, queuing
    /// nothing, where the type has no such kernels or no buffer fits.
    fn gemm_unpacked(
        &self,
        w: &DeviceMatrix,
        rows: usize,
        x: &GemvInput,
        inputs: &Range<usize>,
        y: &mut DeviceBuffer<f32>,
        stream: &Stream,
    ) -> Result<bool> {
        let (Some(product), Some(unpack)) = (
            &self.pgemm_f16w,
            weight_index(w.ty()).and_then(|i| self.gemm.get(i)?.get(2)),
        ) else {
            return Ok(false);
        };
        let cols = w.cols();
        let mut unpacked = self.unpacked.borrow_mut();
        let key = stream.raw() as usize;
        let at = match unpacked.iter().position(|u| u.stream == key) {
            Some(at) => at,
            None => {
                unpacked.push(Unpacked {
                    stream: key,
                    buf: None,
                    tried: 0,
                });
                unpacked.len() - 1
            }
        };
        let u = &mut unpacked[at];
        let want = (rows * cols).min(self.unpack_limit.get());
        if want > u.tried {
            u.tried = want;
            u.grow(stream, want, PF16_ROWS as usize * cols);
        }
        let Some(buf) = &u.buf else {
            // perf: a model that fits only without this buffer keeps to the
            // kernel that expands its weights in every tile
            return Ok(false);
        };
        let piece = buf.len() / cols / PF16_ROWS as usize * PF16_ROWS as usize;
        if piece == 0 {
            return Ok(false);
        }
        let (wp, out) = (w.data().device_ptr(), buf.device_ptr());
        let (c, total, stride) = (cols as u32, w.total_blocks(), rows as u32);
        let at = inputs.start * x.width();
        let (xp, dp) = (x.quantized().ptr_at(at), x.block_scales().ptr_at(at / 16));
        let n = inputs.len() as u32;
        for first in (0..rows).step_by(piece) {
            let (f, r) = (first as u32, piece.min(rows - first) as u32);
            let config = LaunchConfig::linear(r.div_ceil(PUNPACK_ROWS), 256);
            // SAFETY: matches `p<format>_unpack(const uint8_t*, __half*,
            // unsigned, unsigned, unsigned, size_t)`; `w` holds rows `first`
            // to `first + r` of `cols`, a multiple of the stage, and the
            // buffer `r * cols` values.
            unsafe {
                unpack.launch(
                    config,
                    stream,
                    &mut [arg(&wp), arg(&out), arg(&f), arg(&r), arg(&c), arg(&total)],
                )?;
            }
            let yp = y.device_ptr() + 4 * first as u64;
            let config = LaunchConfig {
                grid: [n.div_ceil(PF16_COLS), r.div_ceil(PF16_ROWS), 1],
                block: [256, 1, 1],
                shared_bytes: 0,
            };
            // SAFETY: matches `pgemm_f16w(const __half*, const int8_t*, const
            // float2*, float*, unsigned, unsigned, unsigned, unsigned)`; the
            // buffer holds `r` rows of `cols` values, `x` the inputs, as the
            // caller checked, and `y` their `n * rows` outputs, of which the
            // piece writes rows `first` to `first + r`.
            unsafe {
                product.launch(
                    config,
                    stream,
                    &mut [
                        arg(&out),
                        arg(&xp),
                        arg(&dp),
                        arg(&yp),
                        arg(&r),
                        arg(&c),
                        arg(&n),
                        arg(&stride),
                    ],
                )
            }?;
        }
        Ok(true)
    }

    fn launch_gemm(
        &self,
        kernel: &Function,
        p: &Product,
        out: Act,
        grid: [u32; 3],
        stream: &Stream,
    ) -> Result<()> {
        let first = p.inputs.start * p.x.width();
        let (xp, dp) = if is_float(p.w.ty()) {
            (p.x.values().ptr_at(first), 0)
        } else {
            (
                p.x.quantized().ptr_at(first),
                p.x.block_scales().ptr_at(first / 16),
            )
        };
        let wp = p.w.data().device_ptr();
        let up = p.up.map_or(0, |m| m.data().device_ptr());
        let (rows, cols, n) = (p.rows as u32, p.w.cols() as u32, p.inputs.len() as u32);
        let total = p.w.total_blocks();
        let gelu = u32::from(p.gelu);
        let mut args = [
            arg(&wp),
            arg(&up),
            arg(&xp),
            arg(&dp),
            arg(&out.x),
            arg(&out.qs),
            arg(&out.ds),
            arg(&rows),
            arg(&cols),
            arg(&total),
            arg(&n),
            arg(&gelu),
        ];
        let tile = p.tile;
        let config = LaunchConfig {
            grid,
            block: [tile.threads, 1, 1],
            shared_bytes: tile.shared,
        };
        // SAFETY: matches `gemm_*(const uint8_t*, const uint8_t*, const
        // void*, const float2*, float*, int8_t*, float2*, unsigned, unsigned,
        // size_t, unsigned, unsigned)`. The callers checked that `x` holds the inputs,
        // `out` their `n * rows` outputs (and for a gated product their int8
        // copy and scales) and `w` (and `up`, of the same type and shape) at
        // least `rows` rows of `cols`, a multiple of the kernel's stage; the
        // kernel reads only those and writes only `out`.
        unsafe { kernel.launch(config, stream, &mut args) }
    }

    /// Splits of the k axis for `rows` rows over `stages` stages: two when
    /// that fills the last round of the widest tiles over [`SPLIT_INPUTS`]
    /// inputs, one per multiprocessor, markedly better. Every batch and tile
    /// splits the same, so that an input's outputs have the same bits
    /// whatever else shares the product.
    fn gemm_splits(&self, rows: usize, stages: usize) -> u32 {
        let wide = &QUANT_TILES[QUANT_TILES.len() - 1];
        let blocks = SPLIT_INPUTS.div_ceil(wide.inputs) * (rows as u32).div_ceil(wide.rows);
        let sms = self.multiprocessors;
        let fill = |b: u32| b as f32 / (b.div_ceil(sms) * sms) as f32;
        if stages >= 2 && fill(2 * blocks) > fill(blocks) + 0.05 {
            2
        } else {
            1
        }
    }
}

/// A stream's buffer of weights expanded to f16, and the most values it has
/// tried to hold, so that a size that did not fit is not tried again.
pub(crate) struct Unpacked {
    stream: usize,
    buf: Option<DeviceBuffer<u16>>,
    tried: usize,
}

impl Unpacked {
    /// Grows the buffer to `want` values or, where that does not fit, to
    /// the most of halves of it down to `least`, keeping a buffer at least
    /// as large as the one it has.
    fn grow(&mut self, stream: &Stream, want: usize, least: usize) {
        let have = self.buf.as_ref().map_or(0, |b| b.len());
        let mut len = want;
        while len > have && len >= least {
            if let Ok(buf) = DeviceBuffer::new(stream.context(), len) {
                self.buf = Some(buf);
                return;
            }
            len /= 2;
        }
    }
}
