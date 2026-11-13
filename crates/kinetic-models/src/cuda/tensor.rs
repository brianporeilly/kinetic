//! Tensor parallelism: every layer split over several devices.
//!
//! Each device holds a share of every layer: of the attention heads (and
//! their keys and values), of the delta-rule heads and of the feed-forward
//! rows. It projects the whole normalized stream onto its heads, runs them,
//! and its share of the output projection gives a partial output; the
//! devices swap partials and add them up, so that each holds the same
//! stream, then run the feed-forward the same way. The norms, the residual
//! stream and its bits are the same on every device, and the main one also
//! runs the embeddings and the head.
//!
//! The partial sums run in another order than one device's products, so the
//! results differ slightly from a single device's.

use std::ops::Range;

use kinetic_cuda::{DeviceMatrix, Flow};
use kinetic_quant::Matrix;

use super::scratch::Work;
use super::{cuda_error, CudaModel, Gpu};
use crate::LoadError;

/// Which share of a layer a device holds: `rank` of `ranks` equal ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Part {
    pub rank: usize,
    pub ranks: usize,
}

impl Part {
    /// The whole layer, on one device.
    pub const WHOLE: Part = Part { rank: 0, ranks: 1 };

    pub fn is_whole(&self) -> bool {
        self.ranks == 1
    }

    /// This part's run of `n` units, as heads or rows, which the parts share
    /// equally.
    pub fn share(&self, n: usize, what: &str) -> Result<Range<usize>, LoadError> {
        if !n.is_multiple_of(self.ranks) {
            return Err(LoadError(format!(
                "{n} {what} do not split evenly over {} GPUs",
                self.ranks
            )));
        }
        let each = n / self.ranks;
        Ok(self.rank * each..(self.rank + 1) * each)
    }
}

/// `runs` scaled by `by`: runs of heads as runs of their values.
pub(super) fn scaled(runs: &[Range<usize>], by: usize) -> Vec<Range<usize>> {
    runs.iter().map(|r| r.start * by..r.end * by).collect()
}

/// `runs` moved on by `by`.
pub(super) fn shifted(runs: &[Range<usize>], by: usize) -> Vec<Range<usize>> {
    runs.iter().map(|r| r.start + by..r.end + by).collect()
}

fn total(runs: &[Range<usize>]) -> usize {
    runs.iter().map(|r| r.len()).sum()
}

/// The values of `v` in `runs`, in order.
pub(super) fn pick(v: &[f32], runs: &[Range<usize>]) -> Vec<f32> {
    runs.iter().flat_map(|r| &v[r.clone()]).copied().collect()
}

/// Uploads the rows of `m` in `runs`, in order.
pub(super) fn upload_rows(
    gpu: &Gpu,
    m: &Matrix,
    runs: &[Range<usize>],
) -> Result<DeviceMatrix, LoadError> {
    let row = m.row_bytes();
    let data: Vec<u8> = runs
        .iter()
        .flat_map(|r| &m.data[r.start * row..r.end * row])
        .copied()
        .collect();
    let part = Matrix::new(m.ty, &data, total(runs), m.cols);
    DeviceMatrix::upload(&gpu.ctx, &part).map_err(cuda_error)
}

/// Uploads the columns of `m` in `runs`, in order, each row's; each run
/// must start and end on the type's blocks.
pub(super) fn upload_cols(
    gpu: &Gpu,
    m: &Matrix,
    runs: &[Range<usize>],
) -> Result<DeviceMatrix, LoadError> {
    let (block, bytes) = (m.ty.block_len(), m.ty.block_bytes());
    if let Some(r) = runs
        .iter()
        .find(|r| !r.start.is_multiple_of(block) || !r.end.is_multiple_of(block))
    {
        return Err(LoadError(format!(
            "columns {r:?} of a {} matrix do not fall on its blocks of {block}, so it cannot be split over GPUs",
            m.ty
        )));
    }
    let row = m.row_bytes();
    let mut data = Vec::with_capacity(m.rows * total(runs) / block * bytes);
    for r in 0..m.rows {
        let src = &m.data[r * row..][..row];
        for run in runs {
            data.extend_from_slice(&src[run.start / block * bytes..run.end / block * bytes]);
        }
    }
    let part = Matrix::new(m.ty, &data, m.rows, total(runs));
    DeviceMatrix::upload(&gpu.ctx, &part).map_err(cuda_error)
}

/// Where a part's values sit in its layer's host buffer, rows of `width`
/// values: in each row the part holds the values in `runs`, in order.
#[derive(Clone, Debug)]
pub(super) struct Held {
    pub runs: Vec<Range<usize>>,
    pub width: usize,
}

impl Held {
    pub fn new(runs: Vec<Range<usize>>, width: usize) -> Held {
        Held { runs, width }
    }
}

/// The host buffer that `parts`, held as `held` says, make up together.
pub(super) fn join(parts: &[Vec<f32>], held: &[&Held]) -> Vec<f32> {
    let each = total(&held[0].runs);
    let rows = parts[0].len() / each.max(1);
    let mut whole = vec![0.0; rows * held[0].width];
    for (part, h) in parts.iter().zip(held) {
        let mut src = part.chunks_exact(total(&h.runs));
        for row in whole.chunks_exact_mut(h.width) {
            let mut src = src.next().expect("parts have the same rows").iter();
            for r in &h.runs {
                for (dst, s) in row[r.clone()].iter_mut().zip(src.by_ref()) {
                    *dst = *s;
                }
            }
        }
    }
    whole
}

/// The values a part held as `held` takes of the host buffer `whole`.
pub(super) fn part_of(whole: &[f32], held: &Held) -> Vec<f32> {
    whole
        .chunks_exact(held.width)
        .flat_map(|row| pick(row, &held.runs))
        .collect()
}

impl CudaModel<'_> {
    /// Queues every layer over the batch in `work`, whose embeddings are in
    /// the main device's scratch, split over the devices; leaves the
    /// normalized stream in the main device's `h`, for the head.
    pub(super) fn queue_tensor(&self, work: &mut Work, flow: Flow) -> kinetic_cuda::Result<()> {
        let mut shards = std::mem::take(&mut work.shards);
        let out = self.queue_parts(work, &mut shards, flow);
        work.shards = shards;
        out
    }

    fn queue_parts(
        &self,
        main: &mut Work,
        shards: &mut [Work],
        flow: Flow,
    ) -> kinetic_cuda::Result<()> {
        let n = main.batch.len();
        let len = n * self.n_embd;
        let ([peer], [other]) = (self.shards.as_slice(), shards) else {
            unreachable!("tensor splits run over two devices")
        };
        let (a, b) = (&self.gpu, &peer.gpu);
        // The batch and the embeddings go over to the other device.
        a.handoff.record(&a.stream)?;
        b.stream.wait(&a.handoff)?;
        other.batch.copy_from(&main.batch, &b.stream)?;
        other
            .scratch
            .x
            .copy_from_device(0, &main.scratch.x, 0..len, &b.stream)?;
        b.copied.record(&b.stream)?;
        a.stream.wait(&b.copied)?;

        let ranks = [
            (a, &self.layers, &self.output_norm),
            (b, &peer.layers, &peer.output_norm),
        ];
        let mut works = [main, other];
        for ((gpu, layers, _), w) in ranks.iter().zip(works.iter_mut()) {
            let s = &mut w.scratch;
            layers[0].input(gpu, &s.x, &mut s.h, n)?;
        }
        for i in 0..self.layers.len() {
            for ((gpu, layers, _), w) in ranks.iter().zip(works.iter_mut()) {
                let w = &mut **w;
                layers[i]
                    .mixer
                    .forward(gpu, &mut w.scratch, &mut w.layers[i], flow, &w.batch)?;
            }
            sum_partials(a, b, &mut works, len)?;
            for ((gpu, layers, _), w) in ranks.iter().zip(works.iter_mut()) {
                layers[i].feed(gpu, &mut w.scratch, n)?;
            }
            sum_partials(a, b, &mut works, len)?;
            for ((gpu, layers, norm), w) in ranks.iter().zip(works.iter_mut()) {
                let next = match layers.get(i + 1) {
                    Some(l) => l.mixer_norm.as_ref(),
                    None => Some(*norm),
                };
                layers[i].back(gpu, &mut w.scratch, n, next)?;
            }
        }
        Ok(())
    }
}

/// Adds up the partial outputs `y` of the first `len` values on devices
/// `a` and `b`, whose works are `works`, so that both hold the sum: each
/// copies the other's into its `sum` and adds it. Both add the same two
/// values, so the sums have the same bits.
fn sum_partials(
    a: &Gpu,
    b: &Gpu,
    works: &mut [&mut Work; 2],
    len: usize,
) -> kinetic_cuda::Result<()> {
    let [wa, wb] = works;
    let (sa, sb) = (&mut wa.scratch, &mut wb.scratch);
    a.handoff.record(&a.stream)?;
    b.handoff.record(&b.stream)?;
    a.stream.wait(&b.handoff)?;
    sa.sum.copy_from_device(0, &sb.y, 0..len, &a.stream)?;
    a.copied.record(&a.stream)?;
    b.stream.wait(&a.handoff)?;
    sb.sum.copy_from_device(0, &sa.y, 0..len, &b.stream)?;
    b.copied.record(&b.stream)?;
    // neither overwrites its partial before the other has copied it
    a.stream.wait(&b.copied)?;
    a.kernels.add(&mut sa.y, &sa.sum, len, &a.stream)?;
    b.stream.wait(&a.copied)?;
    b.kernels.add(&mut sb.y, &sb.sum, len, &b.stream)
}
