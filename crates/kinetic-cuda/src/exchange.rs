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
use crate::stream::{Event, Stream};

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

/// One device's end of exchanges that the copy engines carry, for passes
/// that are not recorded into graphs: the values cross on a copy stream of
/// their own while the device runs other work, and events order the steps,
/// so no kernel waits on the peer. Each exchange is queued in two steps on
/// both devices, [`Kernels::send`] on both before [`Kernels::receive`] on
/// either, since a stream can only wait for an event already queued.
pub struct Courier {
    copy: Stream,
    /// Where the peer's values land.
    inbox: DeviceBuffer<f32>,
    /// Recorded once the values to send are written.
    ready: Event,
    /// Recorded once they are in the peer's inbox.
    landed: Event,
    /// Recorded once this device has read its inbox, which the peer waits
    /// for before it writes there again.
    taken: Event,
}

impl Courier {
    /// A courier for exchanges of up to `cap` values.
    pub fn new(ctx: &Rc<Context>, cap: usize) -> Result<Courier> {
        Ok(Courier {
            copy: Stream::new(ctx)?,
            inbox: DeviceBuffer::new(ctx, cap.max(1))?,
            ready: Event::new(ctx)?,
            landed: Event::new(ctx)?,
            taken: Event::new(ctx)?,
        })
    }

    pub fn size_bytes(&self) -> usize {
        4 * self.inbox.len()
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

    /// Queues the first step of an exchange through the copy engines: the
    /// first `len` values of `y`, once `stream` has written them, go to
    /// the peer's inbox on `own`'s copy stream, after the peer has read
    /// what it was last sent. A device that takes the peer's values without
    /// giving any ([`Exchange::Take`]) sends nothing.
    pub fn send(
        &self,
        y: &DeviceBuffer<f32>,
        how: Exchange,
        len: usize,
        own: &Courier,
        peer: &Courier,
        stream: &Stream,
    ) -> Result<()> {
        y.check_holds(len, "exchanged values")?;
        peer.inbox.check_holds(len, "peer's inbox")?;
        if how == Exchange::Take || len == 0 {
            return Ok(());
        }
        own.ready.record(stream)?;
        own.copy.wait(&own.ready)?;
        own.copy.wait(&peer.taken)?;
        let api = own.copy.context().enter()?;
        // SAFETY: both ranges hold `len` values, as checked; unified
        // addressing tells the driver the inbox is on the peer, which this
        // device may reach. The peer reads it only after `landed`.
        api.check("cuMemcpyAsync", unsafe {
            (api.memcpy_async)(
                peer.inbox.device_ptr(),
                y.device_ptr(),
                4 * len,
                own.copy.raw(),
            )
        })?;
        own.landed.record(&own.copy)
    }

    /// Queues the second step of the exchange [`Kernels::send`] began: once
    /// this device's values have left and the peer's are in `own`'s inbox,
    /// `stream` adds them to the first `len` values of `y` or takes them,
    /// as `how` says. Nothing queued after it on `stream` changes the values
    /// before they have left.
    pub fn receive(
        &self,
        y: &mut DeviceBuffer<f32>,
        how: Exchange,
        len: usize,
        own: &Courier,
        peer: &Courier,
        stream: &Stream,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        stream.wait(&own.landed)?;
        if how == Exchange::Keep {
            return Ok(());
        }
        stream.wait(&peer.landed)?;
        match how {
            Exchange::Add => self.add(y, &own.inbox, len, stream)?,
            _ => y.copy_from_device(0, &own.inbox, 0..len, stream)?,
        }
        own.taken.record(stream)
    }
}
