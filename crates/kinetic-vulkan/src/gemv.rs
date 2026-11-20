use std::rc::Rc;

use kinetic_quant::{Matrix, TensorType};

use crate::batch::Batch;
use crate::commands::Commands;
use crate::device::Device;
use crate::error::{Error, Result};
use crate::kernels::{Kernels, MAX_GROUPS};
use crate::layout::device_layout;
use crate::memory::{Allocation, DeviceBuffer};
use crate::pipeline::Kernel;

/// Rows per workgroup; each row gets one subgroup of 32.
#[cfg(not(metal_driver))]
const ROWS_PER_GROUP: usize = 4;
/// Rows per workgroup of the GEMV kernels in `shaders/metal`: four
/// subgroups of 32, each taking a tile of four rows.
#[cfg(metal_driver)]
const ROWS_PER_GROUP: usize = 16;
/// Rows per workgroup of the kernels for many tokens in `shaders/metal`:
/// four subgroups of two rows.
#[cfg(metal_driver)]
const MANY_ROWS_PER_GROUP: usize = 8;
/// Elements per scale in the int8 copy of the input.
const Q8_BLOCK: usize = 32;
/// Invocations per workgroup of quantize_q8.wgsl, one per block of 32.
const QUANTIZE_THREADS: usize = 64;

/// Elements per kernel block for each type with a GEMV kernel.
fn block_len(ty: TensorType) -> Option<usize> {
    Some(match ty {
        TensorType::F32 => 4,
        TensorType::F16 => 8,
        TensorType::Q8_0 => 32,
        TensorType::Q4K | TensorType::Q5K | TensorType::Q6K => 256,
        _ => return None,
    })
}

/// A weight matrix in device memory, laid out for the GEMV kernels.
pub struct DeviceMatrix {
    ty: TensorType,
    rows: usize,
    cols: usize,
    bytes: usize,
    data: DeviceBuffer<u8>,
}

impl DeviceMatrix {
    /// Copies `m` to the device. Fails for types without a GEMV kernel and
    /// for row lengths that are not a multiple of 32.
    pub fn upload(device: &Rc<Device>, m: &Matrix) -> Result<DeviceMatrix> {
        let Some(block) = block_len(m.ty) else {
            return Err(Error::Unsupported(m.ty));
        };
        let blocks = m.rows.checked_mul(m.cols / block);
        if !m.cols.is_multiple_of(Q8_BLOCK) || blocks.is_none_or(|b| u32::try_from(b).is_err()) {
            return Err(Error::Shape(format!(
                "{}x{} matrix: columns must be a multiple of {Q8_BLOCK} and blocks fit in 32 bits",
                m.rows, m.cols
            )));
        }
        let mut data = device_layout(m).into_owned();
        // Kernels read whole 16-byte words, also of the last plane.
        data.resize(data.len().next_multiple_of(16), 0);
        Ok(DeviceMatrix {
            ty: m.ty,
            rows: m.rows,
            cols: m.cols,
            bytes: m.data.len(),
            data: DeviceBuffer::from_slice(device, &data)?,
        })
    }

    pub fn ty(&self) -> TensorType {
        self.ty
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Bytes of weights, which a GEMV reads once.
    pub fn size_bytes(&self) -> usize {
        self.bytes
    }
}

/// Rows of GEMV input vectors, kept both as f32 for float weights and
/// quantized to int8 in blocks of 32 for quantized weights.
pub struct GemvInput {
    width: usize,
    x: DeviceBuffer<f32>,
    qs: DeviceBuffer<i8>,
    ds: DeviceBuffer<f32>,
}

impl GemvInput {
    /// One input of `len` elements; `len` must be a multiple of 32.
    pub fn new(device: &Rc<Device>, len: usize) -> Result<GemvInput> {
        GemvInput::batch(device, len, 1)
    }

    /// `rows` inputs of `width` elements each; `width` must be a multiple
    /// of 32.
    pub fn batch(device: &Rc<Device>, width: usize, rows: usize) -> Result<GemvInput> {
        let len = width
            .checked_mul(rows)
            .filter(|&l| u32::try_from(l).is_ok());
        let Some(len) = len.filter(|_| width.is_multiple_of(Q8_BLOCK)) else {
            return Err(Error::Shape(format!(
                "GEMV input of {rows} rows of {width} is not a 32-bit multiple of {Q8_BLOCK}"
            )));
        };
        Ok(GemvInput {
            width,
            x: DeviceBuffer::new(device, len)?,
            qs: DeviceBuffer::new(device, len)?,
            ds: DeviceBuffer::new(device, 2 * len / Q8_BLOCK)?,
        })
    }

    /// Elements of one input.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Inputs held.
    pub fn rows(&self) -> usize {
        self.x.len() / self.width.max(1)
    }

    /// All elements of all rows.
    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// The f32 values, row after row.
    pub fn values(&self) -> &DeviceBuffer<f32> {
        &self.x
    }

    /// The three buffers a kernel writes the input to, at consecutive
    /// bindings in this order.
    pub(crate) fn outputs(&self) -> [&Rc<Allocation>; 3] {
        [
            self.x.allocation(),
            self.qs.allocation(),
            self.ds.allocation(),
        ]
    }

    /// Copies `x` to the device now and records its quantization in
    /// `commands`, which runs at submit: a second upload before then
    /// replaces the first.
    pub fn upload(&mut self, kernels: &Kernels, x: &[f32], commands: &mut Commands) -> Result<()> {
        self.x.copy_from(x)?;
        self.record_quantize(kernels, commands)
    }

    /// Records the quantization of every row from the f32 values.
    pub fn record_quantize(&self, kernels: &Kernels, commands: &mut Commands) -> Result<()> {
        let blocks = self.x.len() / Q8_BLOCK;
        let groups = blocks.div_ceil(QUANTIZE_THREADS);
        if groups > MAX_GROUPS as usize {
            return Err(Error::Shape(format!(
                "GEMV input of {} elements is too long to quantize",
                self.x.len()
            )));
        }
        // SAFETY: matches quantize_q8.wgsl: x, qs and ds at bindings 0..2
        // and the block count; it touches `blocks` blocks of each.
        unsafe {
            commands.dispatch(
                &kernels.quantize_q8,
                &[
                    self.x.allocation(),
                    self.qs.allocation(),
                    self.ds.allocation(),
                ],
                &[blocks as u32],
                [groups as u32, 1, 1],
            )
        };
        Ok(())
    }

    /// The int8 values: element `i` is `quantized[i] * d` of its block.
    pub fn quantized(&self) -> &DeviceBuffer<i8> {
        &self.qs
    }

    /// Per block of 32, interleaved: the scale `d` and `d` times the sum of
    /// the block's int8 values.
    pub fn block_scales(&self) -> &DeviceBuffer<f32> {
        &self.ds
    }
}

/// Which input rows a batched GEMV multiplies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rows {
    /// Every token of the batch.
    Tokens,
    /// The rows the output head runs on; see [`Batch::set`].
    Head,
}

/// Where a batched GEMV writes: output row `t` starts at element
/// `t * stride + offset` of the output buffer.
#[derive(Clone, Copy, Debug)]
pub struct Out {
    pub stride: usize,
    pub offset: usize,
}

impl Kernels {
    /// Records `y = w x` in `commands`.
    pub fn gemv(
        &self,
        w: &DeviceMatrix,
        x: &GemvInput,
        y: &DeviceBuffer<f32>,
        commands: &mut Commands,
    ) -> Result<()> {
        if x.len() != w.cols || y.len() != w.rows {
            return Err(Error::Shape(format!(
                "GEMV of a {}x{} matrix with {} inputs into {} outputs",
                w.rows,
                w.cols,
                x.len(),
                y.len()
            )));
        }
        let out = Out {
            stride: w.rows,
            offset: 0,
        };
        // SAFETY: the shapes were checked above, and the batch of one
        // input selects its only row either way.
        unsafe { self.record_gemv(w, x, y, out, false, false, &self.single, commands) }
    }

    /// Records `y = w x` for the rows of `x` that `batch` and `rows`
    /// choose, written as `out` says.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_rows(
        &self,
        w: &DeviceMatrix,
        x: &GemvInput,
        y: &DeviceBuffer<f32>,
        out: Out,
        rows: Rows,
        batch: &Batch,
        commands: &mut Commands,
    ) -> Result<()> {
        let capacity = match rows {
            Rows::Tokens => batch.rows(),
            Rows::Head => batch.head_rows(),
        };
        let needed = (capacity.max(1) - 1)
            .checked_mul(out.stride)
            .and_then(|end| end.checked_add(out.offset + w.rows));
        if x.width() != w.cols
            || x.rows() < batch.rows()
            || needed.is_none_or(|n| n > y.len() || u32::try_from(n).is_err())
            || (capacity > 1 && out.stride < out.offset + w.rows)
        {
            return Err(Error::Shape(format!(
                "GEMV of a {}x{} matrix on {} rows of {} into {} outputs at stride {} offset {}",
                w.rows,
                w.cols,
                x.rows(),
                x.width(),
                y.len(),
                out.stride,
                out.offset
            )));
        }
        // A pass of many prompt tokens runs every row of the batch.
        let many = rows == Rows::Tokens && commands.prompt();
        #[cfg(not(metal_driver))]
        if many {
            if let Some(kernel) = self.matrix_kernel(w.ty) {
                // SAFETY: as below; the kernel takes the same bindings.
                unsafe { self.record_matrix(kernel, w, x, y, out, capacity, batch, commands) };
                return Ok(());
            }
        }
        // SAFETY: checked above: `x` holds every row the batch can name and
        // `y` every output row it can ask for.
        unsafe {
            self.record_gemv(
                w,
                x,
                y,
                out,
                rows == Rows::Head,
                many,
                batch.device(),
                commands,
            )
        }
    }

    /// # Safety
    ///
    /// `x` must hold every input row and `y` every output row that the
    /// words in `batch` select.
    #[allow(clippy::too_many_arguments)]
    unsafe fn record_gemv(
        &self,
        w: &DeviceMatrix,
        x: &GemvInput,
        y: &DeviceBuffer<f32>,
        out: Out,
        head: bool,
        many: bool,
        batch: &DeviceBuffer<u32>,
        commands: &mut Commands,
    ) -> Result<()> {
        let block = block_len(w.ty).ok_or(Error::Unsupported(w.ty))?;
        let (kernel, rows_per_group) = self.gemv_kernel(w.ty, many);
        let groups = w.rows.div_ceil(rows_per_group);
        let wide = groups.min(MAX_GROUPS as usize);
        let tall = groups.div_ceil(wide.max(1));
        // SAFETY: matches gemv.wgsl: weights, x, qs, ds, y and the batch at
        // bindings 0..5, then rows, blocks per row, output stride and
        // offset, and the choice of rows. The kernel reads `rows * cols`
        // weights and the input rows the batch selects and writes the
        // output rows it selects, which the caller vouches for; workgroups
        // past the last row return early.
        unsafe {
            commands.dispatch(
                kernel,
                &[
                    w.data.allocation(),
                    x.x.allocation(),
                    x.qs.allocation(),
                    x.ds.allocation(),
                    y.allocation(),
                    batch.allocation(),
                ],
                &[
                    w.rows as u32,
                    (w.cols / block) as u32,
                    out.stride as u32,
                    out.offset as u32,
                    head as u32,
                ],
                [wide as u32, tall as u32, 1],
            )
        };
        Ok(())
    }

    /// Whether prompt passes with weights of type `ty` give every token the
    /// bits of the GEMV, which they don't where they run on the matrix units.
    pub fn exact_prompt(&self, ty: TensorType) -> bool {
        #[cfg(not(metal_driver))]
        {
            self.matrix_kernel(ty).is_none()
        }
        #[cfg(metal_driver)]
        {
            let _ = ty;
            true
        }
    }

    /// The kernel on the matrix units for weights of type `ty`, where the
    /// device has one.
    #[cfg(not(metal_driver))]
    fn matrix_kernel(&self, ty: TensorType) -> Option<&Rc<Kernel>> {
        match ty {
            TensorType::Q8_0 => self.mm_q8_0.as_ref(),
            _ => None,
        }
    }

    /// Records a product on the matrix units: tiles of 64 rows by 64 of
    /// the batch's `capacity` tokens.
    ///
    /// # Safety
    ///
    /// As [`Self::record_gemv`].
    #[cfg(not(metal_driver))]
    #[allow(clippy::too_many_arguments)]
    unsafe fn record_matrix(
        &self,
        kernel: &Rc<Kernel>,
        w: &DeviceMatrix,
        x: &GemvInput,
        y: &DeviceBuffer<f32>,
        out: Out,
        capacity: usize,
        batch: &Batch,
        commands: &mut Commands,
    ) {
        const TILE: usize = 64;
        let block = block_len(w.ty).expect("matrix kernels take types with GEMV kernels");
        // SAFETY: matches mm_*.wgsl, which takes gemv.wgsl's bindings and
        // push constants; it reads the rows and writes the outputs of the
        // tokens the batch selects, which the caller vouches for, and
        // tiles past them return early.
        unsafe {
            commands.dispatch(
                kernel,
                &[
                    w.data.allocation(),
                    x.x.allocation(),
                    x.qs.allocation(),
                    x.ds.allocation(),
                    y.allocation(),
                    batch.device().allocation(),
                ],
                &[
                    w.rows as u32,
                    (w.cols / block) as u32,
                    out.stride as u32,
                    out.offset as u32,
                    0,
                ],
                [
                    w.rows.div_ceil(TILE) as u32,
                    capacity.max(1).div_ceil(TILE) as u32,
                    1,
                ],
            )
        };
    }

    /// The kernel for weights of type `ty` and the rows one of its
    /// workgroups takes; with `many`, the one written for a pass of many
    /// tokens, where the driver has it.
    fn gemv_kernel(&self, ty: TensorType, many: bool) -> (&Rc<Kernel>, usize) {
        #[cfg(metal_driver)]
        if many {
            match ty {
                TensorType::Q4K => return (&self.gemv_q4_k_many, MANY_ROWS_PER_GROUP),
                TensorType::Q5K => return (&self.gemv_q5_k_many, MANY_ROWS_PER_GROUP),
                TensorType::Q6K => return (&self.gemv_q6_k_many, MANY_ROWS_PER_GROUP),
                _ => {}
            }
        }
        #[cfg(not(metal_driver))]
        let _ = many;
        let kernel = match ty {
            TensorType::F32 => &self.gemv_f32,
            TensorType::F16 => &self.gemv_f16,
            TensorType::Q8_0 => &self.gemv_q8_0,
            TensorType::Q4K => &self.gemv_q4_k,
            TensorType::Q5K => &self.gemv_q5_k,
            TensorType::Q6K => &self.gemv_q6_k,
            _ => unreachable!("block_len admits only types with kernels"),
        };
        (kernel, ROWS_PER_GROUP)
    }
}
