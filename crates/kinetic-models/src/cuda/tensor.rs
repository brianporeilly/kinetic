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

use kinetic_cuda::{DeviceMatrix, Exchange, Flow, Graph, EXCHANGE_BLOCKS};
use kinetic_quant::Matrix;

use super::scratch::{Scratch, Work};
use super::{cuda_error, CudaModel, Gpu, Step};
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
    /// Blocks each exchange runs on: one when both parts share a device, so
    /// that a waiting exchange leaves room for the other part's kernels.
    pub(super) fn exchange_blocks(&self) -> usize {
        let same = self
            .shards
            .iter()
            .any(|s| s.gpu.ctx.device() == self.gpu.ctx.device());
        if same {
            1
        } else {
            EXCHANGE_BLOCKS
        }
    }

    /// Queues every layer over the batch in `work`, whose embeddings are in
    /// the main device's scratch, split over the devices; leaves the
    /// normalized stream in the main device's `h`, for the head.
    pub(super) fn queue_tensor(&self, work: &mut Work, flow: Flow) -> kinetic_cuda::Result<()> {
        let mut shards = std::mem::take(&mut work.shards);
        let out = self.queue_parts(work, &mut shards, flow);
        work.shards = shards;
        out
    }

    /// Queues both devices' shares, a stretch between exchanges at a time:
    /// a pass may wait on the host for its device, which must not wait for
    /// an exchange whose other end is not queued yet.
    fn queue_parts(
        &self,
        main: &mut Work,
        shards: &mut [Work],
        flow: Flow,
    ) -> kinetic_cuda::Result<()> {
        self.hand_batch(main, shards)?;
        for stretch in 0..self.stretches() {
            self.queue_stretch(0, stretch, main, &shards[0], flow)?;
            self.queue_stretch(1, stretch, &mut shards[0], main, flow)?;
        }
        Ok(())
    }

    /// Queues the copy of the main device's batch to the other's, once the
    /// main device has written it; the main device then waits for the copy
    /// before it writes the next.
    fn hand_batch(&self, main: &Work, shards: &mut [Work]) -> kinetic_cuda::Result<()> {
        let (a, b) = (&self.gpu, &self.shards[0].gpu);
        a.handoff.record(&a.stream)?;
        b.stream.wait(&a.handoff)?;
        shards[0].batch.copy_from(&main.batch, &b.stream)?;
        b.copied.record(&b.stream)?;
        a.stream.wait(&b.copied)
    }

    /// Stretches of a pass that each end with an exchange: taking the
    /// embeddings, then each layer's mixer and its feed-forward.
    fn stretches(&self) -> usize {
        1 + 2 * self.layers.len()
    }

    /// Queues the batch in `w`, which holds the main device's embeddings
    /// there, through device `rank`'s share of every layer, exchanging
    /// partial outputs with the device of `peer`; the main device is left
    /// with the normalized stream, for the head.
    fn queue_part(
        &self,
        rank: usize,
        w: &mut Work,
        peer: &Work,
        flow: Flow,
    ) -> kinetic_cuda::Result<()> {
        (0..self.stretches())
            .try_for_each(|stretch| self.queue_stretch(rank, stretch, w, peer, flow))
    }

    /// Queues stretch `stretch` of [`CudaModel::queue_part`].
    fn queue_stretch(
        &self,
        rank: usize,
        stretch: usize,
        w: &mut Work,
        peer: &Work,
        flow: Flow,
    ) -> kinetic_cuda::Result<()> {
        let (gpu, layers, norm) = match rank {
            0 => (&self.gpu, &self.layers, &self.output_norm),
            r => {
                let s = &self.shards[r - 1];
                (&s.gpu, &s.layers, &s.output_norm)
            }
        };
        let n = w.batch.len();
        let len = n * self.n_embd;
        let s = &mut w.scratch;
        if stretch == 0 {
            let (Some(own), Some(other)) = (&s.mailbox, &peer.scratch.mailbox) else {
                unreachable!("works split by tensors have mailboxes")
            };
            let embed = match rank {
                0 => Exchange::Keep,
                _ => Exchange::Take,
            };
            gpu.kernels
                .exchange(&mut s.x, own, other, embed, len, &gpu.stream)?;
            return layers[0].input(gpu, &s.x, &mut s.h, n);
        }
        let i = (stretch - 1) / 2;
        let layer = &layers[i];
        if stretch % 2 == 1 {
            layer
                .mixer
                .forward(gpu, s, &mut w.layers[i], flow, &w.batch)?;
            return sum(gpu, s, peer, len);
        }
        layer.feed(gpu, s, n)?;
        sum(gpu, s, peer, len)?;
        let next = match layers.get(i + 1) {
            Some(l) => l.mixer_norm.as_ref(),
            None => Some(norm),
        };
        layer.back(gpu, s, n, next)
    }

    /// Records a one-token step over the batch in `work`, which holds
    /// `token`: a graph per device, the main one's ending with `head`.
    pub(super) fn record_parts(
        &self,
        main: &mut Work,
        shards: &mut [Work],
        token: &[u32],
        embed_on_device: bool,
        head: &mut dyn FnMut(&mut Work) -> kinetic_cuda::Result<()>,
    ) -> kinetic_cuda::Result<Vec<Graph>> {
        let flow = Flow::Commit;
        self.hand_batch(main, shards)?;
        let a = self.gpu.stream.capture(|| {
            if embed_on_device {
                self.embed(main, token)?;
            }
            self.queue_part(0, main, &shards[0], flow)?;
            head(main)
        })?;
        let b = self.shards[0]
            .gpu
            .stream
            .capture(|| self.queue_part(1, &mut shards[0], main, flow))?;
        Ok(vec![a, b])
    }

    /// Replays `step`, recorded by [`CudaModel::record_parts`], on the batch
    /// in `work`.
    pub(super) fn launch_parts(&self, step: &Step, work: &mut Work) -> kinetic_cuda::Result<()> {
        let [a, b] = step.graphs.as_slice() else {
            unreachable!("a graph per device")
        };
        let mut shards = std::mem::take(&mut work.shards);
        let handed = self.hand_batch(work, &mut shards);
        work.shards = shards;
        handed?;
        a.launch(&self.gpu.stream)?;
        b.launch(&self.shards[0].gpu.stream)
    }
}

/// Queues the exchange that adds the peer's partial outputs to those in
/// `s.y`, the first `len` values.
fn sum(gpu: &Gpu, s: &mut Scratch, peer: &Work, len: usize) -> kinetic_cuda::Result<()> {
    let (Some(own), Some(other)) = (&s.mailbox, &peer.scratch.mailbox) else {
        unreachable!("works split by tensors have mailboxes")
    };
    gpu.kernels
        .exchange(&mut s.y, own, other, Exchange::Add, len, &gpu.stream)
}
