//! The hidden rows of prompt passes over several devices, on their way to
//! the host while the devices go on.
//!
//! A pass whose logits nobody reads need not come back to the main device:
//! the last shard copies its normalized stream into one of two page-locked
//! buffers, and the main device starts the next pass as soon as the shards
//! have copied their inputs. With two devices, the first then runs pass
//! k + 1 while the second runs pass k.

use std::ops::Range;
use std::rc::Rc;

use kinetic_cuda::{Context, DeviceBuffer, Event, PinnedBuffer, Stream};

/// Two page-locked buffers of a device's rows, each with the event that
/// marks its copy done and the range of the output it holds until read.
pub(super) struct Staging {
    bufs: [PinnedBuffer<f32>; 2],
    copied: [Event; 2],
    pending: [Option<Range<usize>>; 2],
    next: usize,
}

impl Staging {
    /// Buffers of `len` values each, page-locked for `ctx`, the context of
    /// the device whose rows they take.
    pub fn new(ctx: &Rc<Context>, len: usize) -> kinetic_cuda::Result<Staging> {
        Ok(Staging {
            bufs: [PinnedBuffer::new(ctx, len)?, PinnedBuffer::new(ctx, len)?],
            copied: [Event::new(ctx)?, Event::new(ctx)?],
            pending: [None, None],
            next: 0,
        })
    }

    /// Values a buffer holds.
    pub fn len(&self) -> usize {
        self.bufs[0].len()
    }

    /// Queues on `stream` the copy of the first `range.len()` values of `src`,
    /// for `range` of `out`, into the next buffer; first writes to `out`
    /// what that buffer still holds.
    pub fn stage(
        &mut self,
        src: &DeviceBuffer<f32>,
        range: Range<usize>,
        out: &mut [f32],
        stream: &Stream,
    ) -> kinetic_cuda::Result<()> {
        let i = self.next;
        self.read(i, out)?;
        // SAFETY: the buffer is read again only in `read`, after the stream
        // has passed the copy, and lives as long as `self`.
        unsafe { self.bufs[i].download(0, src, 0..range.len(), stream)? };
        self.copied[i].record(stream)?;
        self.pending[i] = Some(range);
        self.next = 1 - i;
        Ok(())
    }

    /// Waits for and forgets what a run left staged, if it stopped early.
    pub fn reset(&mut self) -> kinetic_cuda::Result<()> {
        for i in 0..2 {
            if self.pending[i].take().is_some() {
                self.copied[i].synchronize()?;
            }
        }
        Ok(())
    }

    /// Writes every staged range to `out`, waiting for the copies.
    pub fn drain(&mut self, out: &mut [f32]) -> kinetic_cuda::Result<()> {
        // oldest first, though each writes only its own range
        let first = self.next;
        self.read(first, out)?;
        self.read(1 - first, out)
    }

    fn read(&mut self, i: usize, out: &mut [f32]) -> kinetic_cuda::Result<()> {
        if let Some(range) = self.pending[i].take() {
            self.copied[i].synchronize()?;
            let len = range.len();
            out[range].copy_from_slice(&self.bufs[i].as_slice()[..len]);
        }
        Ok(())
    }
}
