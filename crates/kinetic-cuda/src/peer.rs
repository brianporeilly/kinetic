//! Timing copies between two devices, for `kinetic doctor`: a model spread
//! over several GPUs copies a token's hidden state from one to the next, so
//! a slow path between them slows every token.

use std::rc::Rc;
use std::time::Duration;

use crate::context::Context;
use crate::device::Device;
use crate::error::Result;
use crate::memory::DeviceBuffer;
use crate::stream::{Event, Stream};

/// About the size of what crosses per token: a hidden state of 5120 f32
/// values and its int8 copy.
pub const SMALL_COPY: usize = 32 << 10;
/// A copy large enough to show the bandwidth.
pub const LARGE_COPY: usize = 64 << 20;
const SMALL_REPEATS: usize = 64;
/// How long one measurement may take before it counts as stuck.
const TIMEOUT: Duration = Duration::from_secs(5);

/// One way of copying between the devices, measured; `None` where the
/// copies did not finish within five seconds.
#[derive(Clone, Copy, Debug)]
pub struct CopyTiming {
    /// Microseconds per copy of [`SMALL_COPY`] bytes.
    pub small_us: Option<f64>,
    /// Bytes per second of one copy of [`LARGE_COPY`] bytes.
    pub large_bytes_per_s: Option<f64>,
}

/// Copies from one device to another, through host memory and, where the
/// driver offers it, directly.
#[derive(Clone, Copy, Debug)]
pub struct PeerCopies {
    /// Whether the driver offers direct access between the two.
    pub direct_offered: bool,
    pub through_host: CopyTiming,
    /// Measured even when `KINETIC_CUDA_P2P` turns direct access off, to
    /// show what it would give.
    pub direct: Option<CopyTiming>,
    /// Whether some copy did not finish: the device may still be busy with
    /// it, so its buffers are left allocated rather than freed.
    pub stuck: bool,
}

/// Times copies from `from` to `to`, two different devices.
pub fn time_copies(from: Device, to: Device) -> Result<PeerCopies> {
    let (a, b) = (Context::new(from)?, Context::new(to)?);
    let src = DeviceBuffer::<u8>::new(&a, LARGE_COPY)?;
    let mut dst = DeviceBuffer::<u8>::new(&b, LARGE_COPY)?;
    let stream = Stream::new(&b)?;
    let through_host = time_one(&b, &src, &mut dst, &stream)?;
    let mut stuck = !through_host.finished();
    let mut direct = None;
    let direct_offered = !stuck && b.enable_peer_access_anyway(&a)?;
    if direct_offered {
        let timing = time_one(&b, &src, &mut dst, &stream)?;
        stuck = !timing.finished();
        direct = Some(timing);
        if !stuck {
            b.disable_peer_access(&a)?;
        }
    }
    if stuck {
        // freeing memory a copy may still use could wait for it forever
        std::mem::forget((src, dst, stream, a, b));
    }
    Ok(PeerCopies {
        direct_offered,
        through_host,
        direct,
        stuck,
    })
}

impl CopyTiming {
    fn finished(&self) -> bool {
        self.small_us.is_some() && self.large_bytes_per_s.is_some()
    }
}

fn time_one(
    ctx: &Rc<Context>,
    src: &DeviceBuffer<u8>,
    dst: &mut DeviceBuffer<u8>,
    stream: &Stream,
) -> Result<CopyTiming> {
    let (start, end) = (Event::new(ctx)?, Event::new(ctx)?);
    // the first copy sets up the path; time the ones after
    dst.copy_from_device(0, src, 0..SMALL_COPY, stream)?;
    start.record(stream)?;
    for _ in 0..SMALL_REPEATS {
        dst.copy_from_device(0, src, 0..SMALL_COPY, stream)?;
    }
    end.record(stream)?;
    if !end.wait_for(TIMEOUT)? {
        return Ok(CopyTiming {
            small_us: None,
            large_bytes_per_s: None,
        });
    }
    let small_us = 1e3 * start.elapsed_ms(&end)? as f64 / SMALL_REPEATS as f64;
    start.record(stream)?;
    dst.copy_from_device(0, src, 0..LARGE_COPY, stream)?;
    end.record(stream)?;
    if !end.wait_for(TIMEOUT)? {
        return Ok(CopyTiming {
            small_us: Some(small_us),
            large_bytes_per_s: None,
        });
    }
    let seconds = start.elapsed_ms(&end)? as f64 / 1e3;
    Ok(CopyTiming {
        small_us: Some(small_us),
        large_bytes_per_s: Some(LARGE_COPY as f64 / seconds),
    })
}
