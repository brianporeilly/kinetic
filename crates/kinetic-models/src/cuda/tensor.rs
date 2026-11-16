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

use std::cell::RefCell;
use std::ops::Range;
use std::rc::Rc;

use kinetic_cuda::{Courier, DeviceBuffer, DeviceMatrix, Exchange, Flow, Graph, EXCHANGE_BLOCKS};
use kinetic_quant::Matrix;

use super::layer::CudaLayer;
use super::scratch::{Scratch, Spare, Work};
use super::{cuda_error, CudaModel, Gpu, Step};
use crate::LoadError;

/// Which of two prompt passes in flight a stretch belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Flight {
    /// The only pass.
    Alone,
    First,
    Second,
}

/// A second set of full-pass buffers and streams on every device, which
/// lets two prompt passes run at once, the second a stretch behind the
/// first: while one pass's partial outputs cross between the devices, the
/// other's products run. The second pass runs each mixer after the first
/// has, so the caches and recurrent states, and every result, are those of
/// the two passes one after the other.
pub(super) struct Overlap {
    pub spare: RefCell<Spare>,
    /// The second pass's streams and events, a set per device in rank
    /// order.
    gpus: Vec<Gpu>,
}

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

    /// The buffers and streams of a second pass in flight, when every layer
    /// is split over distinct devices.
    pub(super) fn new_overlap(&self, one_pass: bool) -> kinetic_cuda::Result<Option<Overlap>> {
        // prototype: KINETIC_ONE_PASS=1 turns it off for comparisons
        let off = one_pass || std::env::var("KINETIC_ONE_PASS").is_ok_and(|v| v == "1");
        if !self.tensor || self.exchange_blocks() == 1 || off {
            return Ok(None);
        }
        let work = self.new_work(self.batch_rows())?;
        let mut gpus = Vec::with_capacity(1 + self.shards.len());
        for gpu in std::iter::once(&self.gpu).chain(self.shards.iter().map(|s| &s.gpu)) {
            let mut twin = Gpu::new(Rc::clone(&gpu.ctx))?;
            twin.cache = gpu.cache;
            gpus.push(twin);
        }
        Ok(Some(Overlap {
            spare: RefCell::new(Spare::new(work)),
            gpus,
        }))
    }

    /// The streams that device `rank` runs `flight` on.
    fn flight_gpu(&self, rank: usize, flight: Flight) -> &Gpu {
        match (flight, &self.overlap) {
            (Flight::Second, Some(o)) => &o.gpus[rank],
            _ => match rank {
                0 => &self.gpu,
                r => &self.shards[r - 1].gpu,
            },
        }
    }

    /// Queues the passes over the batches of `work` and of `spare`, the
    /// overlap's, which both hold their embeddings: the second a stretch
    /// behind the first, on its own streams.
    pub(super) fn queue_pair(
        &self,
        work: &mut Work,
        spare: &mut Spare,
        flow: Flow,
    ) -> kinetic_cuda::Result<()> {
        // the second pass's embeddings were written on the main stream
        let first = &self.gpu;
        first.mixed.record(&first.stream)?;
        self.flight_gpu(0, Flight::Second)
            .stream
            .wait(&first.mixed)?;
        self.with_shards(work, |main, shards| {
            self.hand_batch(Flight::First, main, shards)
        })?;
        spare.swap(work);
        let handed = self.with_shards(work, |main, shards| {
            self.hand_batch(Flight::Second, main, shards)
        });
        spare.swap(work);
        handed?;
        for stretch in 0..self.stretches() {
            self.queue_both(stretch, Flight::First, work, flow)?;
            spare.swap(work);
            let queued = self.queue_both(stretch, Flight::Second, work, flow);
            spare.swap(work);
            queued?;
        }
        Ok(())
    }

    /// Runs `f` on `work` and its shards' works apart.
    fn with_shards<T>(
        &self,
        work: &mut Work,
        f: impl FnOnce(&mut Work, &mut [Work]) -> kinetic_cuda::Result<T>,
    ) -> kinetic_cuda::Result<T> {
        let mut shards = std::mem::take(&mut work.shards);
        let out = f(work, &mut shards);
        work.shards = shards;
        out
    }

    /// Queues stretch `stretch` of `flight` on both devices.
    fn queue_both(
        &self,
        stretch: usize,
        flight: Flight,
        work: &mut Work,
        flow: Flow,
    ) -> kinetic_cuda::Result<()> {
        let carry = Carry::Copies;
        self.with_shards(work, |main, shards| {
            self.stretch_send(0, stretch, flight, carry, main, &shards[0], flow)?;
            self.stretch_send(1, stretch, flight, carry, &mut shards[0], main, flow)?;
            self.stretch_receive(0, stretch, flight, carry, main, &shards[0])?;
            self.stretch_receive(1, stretch, flight, carry, &mut shards[0], main)
        })
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
        self.hand_batch(Flight::Alone, main, shards)?;
        let carry = match main.scratch.courier {
            Some(_) => Carry::Copies,
            None => Carry::Kernel,
        };
        for stretch in 0..self.stretches() {
            let alone = Flight::Alone;
            self.stretch_send(0, stretch, alone, carry, main, &shards[0], flow)?;
            self.stretch_send(1, stretch, alone, carry, &mut shards[0], main, flow)?;
            self.stretch_receive(0, stretch, alone, carry, main, &shards[0])?;
            self.stretch_receive(1, stretch, alone, carry, &mut shards[0], main)?;
        }
        Ok(())
    }

    /// Queues the copy of the main device's batch to the other's, once the
    /// main device has written it; the main device then waits for the copy
    /// before it writes the next.
    fn hand_batch(
        &self,
        flight: Flight,
        main: &Work,
        shards: &mut [Work],
    ) -> kinetic_cuda::Result<()> {
        let (a, b) = (self.flight_gpu(0, flight), self.flight_gpu(1, flight));
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
            .try_for_each(|stretch| self.queue_stretch(rank, stretch, Flight::Alone, w, peer, flow))
    }

    /// Queues stretch `stretch` of [`CudaModel::queue_part`], for `flight`,
    /// its exchange run by kernels.
    fn queue_stretch(
        &self,
        rank: usize,
        stretch: usize,
        flight: Flight,
        w: &mut Work,
        peer: &Work,
        flow: Flow,
    ) -> kinetic_cuda::Result<()> {
        self.stretch_send(rank, stretch, flight, Carry::Kernel, w, peer, flow)?;
        self.stretch_receive(rank, stretch, flight, Carry::Kernel, w, peer)
    }

    /// The layers and output norm of device `rank`.
    fn rank_layers(&self, rank: usize) -> (&[CudaLayer], &DeviceBuffer<f32>) {
        match rank {
            0 => (&self.layers, &self.output_norm),
            r => {
                let s = &self.shards[r - 1];
                (&s.layers, &s.output_norm)
            }
        }
    }

    /// Queues the first step of stretch `stretch` on device `rank`: its
    /// work up to the exchange that ends it, and that exchange's sending.
    #[allow(clippy::too_many_arguments)]
    fn stretch_send(
        &self,
        rank: usize,
        stretch: usize,
        flight: Flight,
        carry: Carry,
        w: &mut Work,
        peer: &Work,
        flow: Flow,
    ) -> kinetic_cuda::Result<()> {
        let gpu = self.flight_gpu(rank, flight);
        let len = w.batch.len() * self.n_embd;
        if stretch == 0 {
            return give(
                gpu,
                carry,
                &mut w.scratch,
                Values::X,
                embed_how(rank),
                len,
                peer,
            );
        }
        let i = (stretch - 1) / 2;
        let layer = &self.rank_layers(rank).0[i];
        if stretch % 2 == 1 {
            // the caches and states take the first pass's tokens first
            if flight == Flight::Second {
                gpu.stream
                    .wait(&self.flight_gpu(rank, Flight::First).mixed)?;
            }
            layer
                .mixer
                .forward(gpu, &mut w.scratch, &mut w.layers[i], flow, &w.batch)?;
            if flight == Flight::First {
                gpu.mixed.record(&gpu.stream)?;
            }
        } else {
            layer.feed(gpu, &mut w.scratch, w.batch.len())?;
        }
        give(
            gpu,
            carry,
            &mut w.scratch,
            Values::Y,
            Exchange::Add,
            len,
            peer,
        )
    }

    /// Queues the second step of stretch `stretch` on device `rank`: the
    /// exchange's receiving, then what follows it in the stretch.
    fn stretch_receive(
        &self,
        rank: usize,
        stretch: usize,
        flight: Flight,
        carry: Carry,
        w: &mut Work,
        peer: &Work,
    ) -> kinetic_cuda::Result<()> {
        let gpu = self.flight_gpu(rank, flight);
        let (layers, norm) = self.rank_layers(rank);
        let n = w.batch.len();
        let len = n * self.n_embd;
        let s = &mut w.scratch;
        if stretch == 0 {
            take(gpu, carry, s, Values::X, embed_how(rank), len, peer)?;
            return layers[0].input(gpu, &s.x, &mut s.h, n);
        }
        take(gpu, carry, s, Values::Y, Exchange::Add, len, peer)?;
        if stretch % 2 == 1 {
            return Ok(());
        }
        let i = (stretch - 1) / 2;
        let next = match layers.get(i + 1) {
            Some(l) => l.mixer_norm.as_ref(),
            None => Some(norm),
        };
        layers[i].back(gpu, s, n, next)
    }

    /// Records a pass of `flow` over the batch in `main`, which holds
    /// `tokens`: a graph per device, the main one's ending with `head`.
    pub(super) fn record_parts(
        &self,
        main: &mut Work,
        shards: &mut [Work],
        tokens: &[u32],
        flow: Flow,
        embed_on_device: bool,
        head: &mut dyn FnMut(&mut Work) -> kinetic_cuda::Result<()>,
    ) -> kinetic_cuda::Result<Vec<Graph>> {
        self.hand_batch(Flight::Alone, main, shards)?;
        let a = self.gpu.stream.capture(|| {
            if embed_on_device {
                self.embed(main, tokens)?;
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
        let handed = self.hand_batch(Flight::Alone, work, &mut shards);
        work.shards = shards;
        handed?;
        a.launch(&self.gpu.stream)?;
        b.launch(&self.shards[0].gpu.stream)
    }
}

/// How an exchange's values cross between the devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Carry {
    /// A kernel on each device writes into the other's mailbox and waits
    /// for the other's values; a graph can hold it.
    Kernel,
    /// The copy engines, beside the devices' other work; see [`Courier`].
    Copies,
}

/// Which values of a scratch an exchange carries.
#[derive(Clone, Copy)]
enum Values {
    /// The residual stream: the embeddings, which the main device gives.
    X,
    /// The block's partial output.
    Y,
}

/// What device `rank` does with the embeddings: the main one gives them.
fn embed_how(rank: usize) -> Exchange {
    match rank {
        0 => Exchange::Keep,
        _ => Exchange::Take,
    }
}

/// Queues the sending of exchange `how` of the first `len` values of `part`
/// with the device of `peer`: with kernels, the whole exchange.
fn give(
    gpu: &Gpu,
    carry: Carry,
    s: &mut Scratch,
    part: Values,
    how: Exchange,
    len: usize,
    peer: &Work,
) -> kinetic_cuda::Result<()> {
    let y = match part {
        Values::X => &mut s.x,
        Values::Y => &mut s.y,
    };
    match carry {
        Carry::Kernel => {
            let (Some(own), Some(other)) = (&s.mailbox, &peer.scratch.mailbox) else {
                unreachable!("works split by tensors have mailboxes")
            };
            gpu.kernels.exchange(y, own, other, how, len, &gpu.stream)
        }
        Carry::Copies => {
            let (own, other) = couriers(s.courier.as_ref(), peer);
            gpu.kernels.send(y, how, len, own, other, &gpu.stream)
        }
    }
}

/// Queues the receiving of the exchange [`give`] sent; nothing with kernels,
/// whose exchange is done.
fn take(
    gpu: &Gpu,
    carry: Carry,
    s: &mut Scratch,
    part: Values,
    how: Exchange,
    len: usize,
    peer: &Work,
) -> kinetic_cuda::Result<()> {
    if carry == Carry::Kernel {
        return Ok(());
    }
    let (own, other) = couriers(s.courier.as_ref(), peer);
    let y = match part {
        Values::X => &mut s.x,
        Values::Y => &mut s.y,
    };
    gpu.kernels.receive(y, how, len, own, other, &gpu.stream)
}

fn couriers<'a>(own: Option<&'a Courier>, peer: &'a Work) -> (&'a Courier, &'a Courier) {
    match (own, peer.scratch.courier.as_ref()) {
        (Some(own), Some(other)) => (own, other),
        _ => unreachable!("works over distinct devices have couriers"),
    }
}
