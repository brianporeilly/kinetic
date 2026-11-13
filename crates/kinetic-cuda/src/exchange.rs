//! Sums between two devices that each hold a share of every layer, run by a
//! kernel on each that writes straight into the other's memory and waits
//! for the other's values, so that a CUDA graph can hold them; see
//! `exchange.cuh`.

use std::rc::Rc;

use crate::context::Context;
use crate::error::Result;
use crate::kernels::{launch_struct, Kernels};
use crate::memory::DeviceBuffer;
use crate::module::LaunchConfig;
use crate::stream::Stream;

/// Most blocks an exchange runs on.
pub const EXCHANGE_BLOCKS: usize = 32;
const THREADS: u32 = 256;

/// One device's end of the exchanges: where the peer's values and flags
/// land, and how many exchanges it has run.
pub struct Mailbox {
    values: DeviceBuffer<f32>,
    flags: DeviceBuffer<u32>,
    count: DeviceBuffer<u32>,
    cap: usize,
    blocks: u32,
}

impl Mailbox {
    /// A mailbox for exchanges of up to `cap` values, run on `blocks`
    /// blocks, at most [`EXCHANGE_BLOCKS`], which the peer's must match.
    /// Two ends on one device take one block: a waiting exchange holds its
    /// blocks' room, which the other end's kernels may need whole.
    pub fn new(ctx: &Rc<Context>, cap: usize, blocks: usize) -> Result<Mailbox> {
        let blocks = blocks.clamp(1, EXCHANGE_BLOCKS);
        Ok(Mailbox {
            values: DeviceBuffer::new(ctx, 2 * cap)?,
            flags: DeviceBuffer::from_slice(ctx, &[0; EXCHANGE_BLOCKS])?,
            count: DeviceBuffer::from_slice(ctx, &[0; 2])?,
            cap,
            blocks: blocks as u32,
        })
    }

    pub fn size_bytes(&self) -> usize {
        4 * (self.values.len() + self.flags.len() + self.count.len())
    }
}

/// What a device does with the values its peer gives it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exchange {
    /// Adds them to its own.
    Add = 0,
    /// Replaces its own with them.
    Take = 1,
    /// Keeps its own: it only gives.
    Keep = 2,
}

#[repr(C)]
struct ExchangeArgs {
    y: u64,
    peer_box: u64,
    peer_flags: u64,
    own_box: u64,
    own_flags: u64,
    count: u64,
    cap: u32,
    n: u32,
    mode: u32,
}

impl Kernels {
    /// Queues one exchange of the first `len` values of `y` with the peer
    /// whose mailbox is `peer`, through this device's `own`: the peer must
    /// queue the matching exchange, of the same length, on its device, and
    /// both run the same exchanges in the same order. Both devices must be
    /// able to reach each other's memory.
    pub fn exchange(
        &self,
        y: &mut DeviceBuffer<f32>,
        own: &Mailbox,
        peer: &Mailbox,
        how: Exchange,
        len: usize,
        stream: &Stream,
    ) -> Result<()> {
        y.check_holds(len, "exchanged values")?;
        if own.blocks != peer.blocks {
            return Err(crate::Error::Shape(
                "exchanging through mailboxes of different blocks".into(),
            ));
        }
        if len > own.cap || len > peer.cap {
            return Err(crate::Error::Shape(format!(
                "an exchange of {len} values through mailboxes of {}",
                own.cap.min(peer.cap)
            )));
        }
        let args = ExchangeArgs {
            y: y.device_ptr(),
            peer_box: peer.values.device_ptr(),
            peer_flags: peer.flags.device_ptr(),
            own_box: own.values.device_ptr(),
            own_flags: own.flags.device_ptr(),
            count: own.count.device_ptr(),
            cap: own.cap as u32,
            n: len as u32,
            mode: how as u32,
        };
        let config = LaunchConfig {
            grid: [own.blocks, 1, 1],
            block: [THREADS, 1, 1],
            shared_bytes: 0,
        };
        // SAFETY: matches `ExchangeArgs`; `y` holds `len` values and both
        // mailboxes room for them in each half, as checked. The peer's
        // buffers are written from this device, which the caller lets
        // reach them.
        unsafe { launch_struct(&self.exchange, config, stream, &args) }
    }
}
