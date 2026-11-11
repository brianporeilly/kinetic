use std::ptr;
use std::rc::Rc;

use crate::device::Device;
use crate::driver::{self, Api, CuContext};
use crate::error::Result;

/// `CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED`: another context of this
/// process asked first, which is as good.
const PEER_ACCESS_ALREADY_ENABLED: i32 = 704;

/// A device's primary context.
///
/// Every resource holds an `Rc` to the context it was made in, so the
/// context outlives them, and none of them is `Send`. Each call makes its
/// context current on the calling thread first, so contexts of several
/// devices can be used from one thread.
pub struct Context {
    raw: CuContext,
    device: Device,
    api: &'static Api,
}

impl Context {
    pub fn new(device: Device) -> Result<Rc<Context>> {
        let api = driver::api()?;
        let mut raw = ptr::null_mut();
        // SAFETY: `raw` is a valid out-pointer and `device` came from cuDeviceGet.
        api.check("cuDevicePrimaryCtxRetain", unsafe {
            (api.primary_ctx_retain)(&mut raw, device.raw)
        })?;
        let ctx = Context { raw, device, api };
        ctx.enter()?;
        Ok(Rc::new(ctx))
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Lets this context read and write `peer`'s memory directly, over
    /// NVLink or PCIe, where the two devices can; returns whether they can.
    /// Copies between contexts work either way, those between devices
    /// without access going through host memory, which the driver handles.
    pub fn enable_peer_access(&self, peer: &Context) -> Result<bool> {
        if self.device == peer.device {
            return Ok(true);
        }
        let api = self.enter()?;
        let mut can = 0;
        // SAFETY: `can` is a valid out-pointer; both devices are valid.
        api.check("cuDeviceCanAccessPeer", unsafe {
            (api.device_can_access_peer)(&mut can, self.device.raw, peer.device.raw)
        })?;
        if can == 0 {
            return Ok(false);
        }
        // SAFETY: `peer.raw` is a retained context; flags must be 0.
        let status = unsafe { (api.ctx_enable_peer_access)(peer.raw, 0) };
        match status {
            PEER_ACCESS_ALREADY_ENABLED => Ok(true),
            _ => api.check("cuCtxEnablePeerAccess", status).map(|()| true),
        }
    }

    /// Makes this context current on the calling thread.
    pub(crate) fn enter(&self) -> Result<&'static Api> {
        // SAFETY: `raw` is a retained context.
        self.api.check("cuCtxSetCurrent", unsafe {
            (self.api.ctx_set_current)(self.raw)
        })?;
        Ok(self.api)
    }

    /// Waits for all work in this context.
    pub fn synchronize(&self) -> Result<()> {
        let api = self.enter()?;
        // SAFETY: no arguments; this context is current.
        api.check("cuCtxSynchronize", unsafe { (api.ctx_synchronize)() })
    }

    /// Free and total device memory in bytes.
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        let api = self.enter()?;
        let (mut free, mut total) = (0, 0);
        // SAFETY: both are valid out-pointers.
        api.check("cuMemGetInfo", unsafe {
            (api.mem_get_info)(&mut free, &mut total)
        })?;
        Ok((free, total))
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: balances the retain in `new`; every resource made in this
        // context has been dropped because each holds an `Rc` to it.
        unsafe { (self.api.primary_ctx_release)(self.device.raw) };
    }
}
