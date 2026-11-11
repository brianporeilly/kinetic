//! Models larger than the device: the last layers that fit run on the
//! device, the first ones on the host, exactly as the CPU path runs them.
//! A token's embedding goes through the host layers, and its hidden state
//! is copied over to the device for the rest, so only `n_embd` values per
//! token cross between them.
//!
//! With several devices, the layers after the main device's run on the
//! others in turn, see [`Shard`](super::Shard); their stream and its input
//! norm go over to each next device, and back to the main one for the head.

use kinetic_cuda::KvType;

use super::experts::DEVICE_RESERVE;
use super::layer::CudaLayer;
use super::scratch::Work;
use super::{cuda_error, CudaModel, Gpu};
use crate::blocks::MixerState;
use crate::model::Layer;
use crate::{LoadError, Model};

/// Limits on what of a model goes to the device, and how it runs there.
/// The default puts as much there as fits and records decode steps.
#[derive(Clone, Debug, Default)]
pub struct Placement {
    /// Most layers on the device, as llama.cpp's `-ngl` counts them; over
    /// all devices when there are several.
    pub gpu_layers: Option<usize>,
    /// Most device bytes the model may take, weights and one state, on
    /// each device.
    pub device_bytes: Option<usize>,
    /// Launch each kernel of a decode step on its own rather than replaying
    /// the recorded step: Torque off.
    pub eager: bool,
    /// How attention caches store keys and values; anything but f16 rounds
    /// them and so changes outputs.
    pub cache_k: KvType,
    pub cache_v: KvType,
    /// The CUDA devices to spread the layers over, by ordinal, in the order
    /// the layers run; the first also holds the embeddings and the output
    /// head. Empty means device 0 alone. A device may be named twice, which
    /// splits the layers on it as if over two, for testing the split.
    pub devices: Vec<usize>,
}

/// What [`upload_layers`] fits a run of layers beside.
pub(super) struct Fit<'a> {
    pub model: &'a Model<'a>,
    pub context: usize,
    pub rows: usize,
    pub cap: usize,
}

/// Where the layers went: how many stay on the host, in front, then those
/// on the main device and those on each further device, in order.
pub(super) struct Split {
    pub front: usize,
    pub main: Vec<CudaLayer>,
    pub shards: Vec<Vec<CudaLayer>>,
}

/// Uploads the model's layers over `gpu`, the main device, which already
/// holds the embeddings and head, and `others`, each with the free memory it
/// had before the model took any: the last layers to the last device, each
/// device taking a share of them as large as its share of the free memory,
/// up to what fits; the main device the run before theirs, and at least
/// one; the host what is left.
pub(super) fn upload_split(
    gpu: &Gpu,
    start: usize,
    others: &[(&Gpu, usize)],
    fit: &Fit,
    vocab: usize,
    placement: &Placement,
) -> Result<Split, LoadError> {
    let n = fit.model.layers.len();
    let mut budget = placement.gpu_layers.unwrap_or(n).min(n);
    let free = |g: &Gpu| g.ctx.memory_info().map(|(f, _)| f).map_err(cuda_error);
    let mut frees = vec![free(gpu)?];
    for (g, _) in others {
        frees.push(free(g)?);
    }
    let total: usize = frees.iter().sum::<usize>().max(1);
    let mut end = n;
    let mut shards = Vec::new();
    for (k, &(g, start)) in others.iter().enumerate().rev() {
        let share = (budget as f64 * frees[k + 1] as f64 / total as f64).round() as usize;
        // leave the main device a layer
        let most = share.max(1).min(budget).min(end.saturating_sub(1));
        let (rest, layers) = upload_layers(g, &fit.model.layers[..end], fit, most, 0, start)?;
        if layers.is_empty() {
            return Err(LoadError(format!("no layer fits on CUDA device {}", k + 2)));
        }
        budget -= layers.len();
        end = rest;
        shards.push(layers);
    }
    shards.reverse();
    let (front, main) = upload_layers(gpu, &fit.model.layers[..end], fit, budget, vocab, start)?;
    if main.is_empty() && n > 0 {
        return Err(LoadError("no layer fits on the device".into()));
    }
    Ok(Split {
        front,
        main,
        shards,
    })
}

/// Uploads the longest run of the last of `layers` that fits in the device
/// beside one state of `fit.context` positions in batches of `fit.rows`
/// with `vocab` logits a row, at most `most` of them; `start` is the free
/// device memory before the model took any. Returns how many of `layers`
/// stay before them and the uploaded ones in order.
fn upload_layers(
    gpu: &Gpu,
    layers: &[Layer],
    fit: &Fit,
    most: usize,
    vocab: usize,
    start: usize,
) -> Result<(usize, Vec<CudaLayer>), LoadError> {
    let model = fit.model;
    let n = layers.len();
    let most = most.min(n);
    let free = || gpu.ctx.memory_info().map(|(f, _)| f).map_err(cuda_error);
    let cap = fit.cap;
    let mut uploaded = Vec::new();
    let mut largest = 0;
    for l in layers.iter().rev().take(most) {
        let before = free()?;
        if before < DEVICE_RESERVE + largest || start.saturating_sub(before) + largest > cap {
            break;
        }
        match CudaLayer::upload(gpu, l, None, model.eps) {
            Ok(layer) => uploaded.push(layer),
            // A failure past the first layer is the device running out.
            Err(_) if !uploaded.is_empty() => break,
            Err(e) => return Err(e),
        }
        largest = largest.max(before.saturating_sub(free()?));
    }
    uploaded.reverse();
    // Give up layers from the front until one state fits beside them.
    while !uploaded.is_empty() {
        let refs: Vec<&CudaLayer> = uploaded.iter().collect();
        let fits = match Work::new(
            gpu,
            &refs,
            model.n_embd,
            fit.context,
            fit.rows,
            vocab,
            0,
            model.folded,
        ) {
            Ok(work) => {
                let used = start.saturating_sub(free()?);
                drop(work);
                used <= cap && free()? >= DEVICE_RESERVE
            }
            Err(_) => false,
        };
        if fits {
            break;
        }
        uploaded.remove(0);
    }
    Ok((n - uploaded.len(), uploaded))
}

impl CudaModel<'_> {
    /// Layers that run on the host, before the device's.
    pub fn host_layers(&self) -> usize {
        self.front
    }

    /// Fresh host states for the layers that run there.
    pub(super) fn new_front(&self) -> Vec<MixerState> {
        self.host.layers[..self.front]
            .iter()
            .map(|l| l.mixer.new_state())
            .collect()
    }

    /// Writes the embeddings of `tokens` into the scratch of `work`, after
    /// the host layers, which run from the batch's position.
    pub(super) fn embed_front(&self, work: &mut Work, tokens: &[u32]) -> kinetic_cuda::Result<()> {
        let mut x = self.host.embed(tokens);
        let pos = work.batch.pos();
        for (layer, state) in self.host.layers.iter().zip(&mut work.front) {
            layer.forward(&mut x, self.eps, pos, None, None, state);
        }
        // Ordered after the work queued so far, like every synchronous copy.
        work.scratch.x.copy_range_from(0, &x)
    }

    pub(super) fn checkpoint_front(&self, work: &mut Work) {
        work.front.iter_mut().for_each(MixerState::checkpoint);
    }

    /// Keeps the first `keep` tokens of the host layers since their
    /// checkpoint.
    pub(super) fn rollback_front(&self, work: &mut Work, keep: usize) {
        let rows: Vec<usize> = (0..keep).collect();
        for (layer, state) in self.host.layers.iter().zip(&mut work.front) {
            layer.mixer.rollback(state, &rows);
        }
    }
}
