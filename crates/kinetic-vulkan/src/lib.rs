//! Vulkan backend for AMD, Intel and NVIDIA GPUs.
//!
//! The Vulkan loader is opened at runtime, so this crate builds and links
//! on systems without Vulkan; [`Instance::new`] then fails with
//! [`Error::LoaderNotFound`]. Kernels are written in WGSL and compiled to
//! SPIR-V by naga at build time, so no Vulkan SDK is needed either.
//!
//! The kernels need Vulkan 1.3 with subgroup arithmetic, subgroups of 32,
//! packed int8 dot products and push descriptors, which current drivers for
//! all three vendors provide; [`DeviceInfo::missing`] says what a device
//! lacks. Integrated GPUs whose device-local heap is small keep buffers in
//! system memory instead.
//!
//! With the `metal` feature on macOS the same kernels, written out as Metal
//! shaders at build time, run through Metal itself: the six driver modules
//! are replaced by those in `src/metal`, and nothing above them changes.
//! Its GEMV kernels are built from `shaders/metal`, where an invocation
//! takes a tile of four rows: faster on a Mac's GPU, slower on the others.
//! For passes of many prompt tokens it has kernels of its own; see
//! [`Commands::set_prompt`].
//!
//! The decode-path GEMV kernels compute `y = W x` for one f32 vector. Float
//! weights read `x` as f32; quantized weights read an int8 copy of `x`, in
//! blocks of 32 with one scale each, made by [`GemvInput::upload`], and use
//! integer dot products.
//!
//! ```no_run
//! # fn main() -> kinetic_vulkan::Result<()> {
//! use kinetic_vulkan::{Commands, Device, DeviceBuffer, DeviceMatrix, GemvInput, Instance, Kernels};
//! # let (weights, x): (kinetic_quant::Matrix, Vec<f32>) = unimplemented!();
//! let physical = Instance::new()?.physical_devices()?.remove(0);
//! let device = Device::new(&physical)?;
//! let kernels = Kernels::load(&device)?;
//! let w = DeviceMatrix::upload(&device, &weights)?;
//! let mut input = GemvInput::new(&device, w.cols())?;
//! let y = DeviceBuffer::new(&device, w.rows())?;
//! let mut commands = Commands::new(&device)?;
//! input.upload(&kernels, &x, &mut commands)?;
//! kernels.gemv(&w, &input, &y, &mut commands)?;
//! commands.submit()?;
//! let y = y.to_vec()?;
//! # Ok(()) }
//! ```

mod argmax;
mod attention;
mod batch;
mod delta_net;
mod error;
mod ffn;
mod gemv;
mod info;
mod kernels;
mod layout;
mod norm;
mod short_conv;

// The driver: what opens the device, holds its memory, builds its kernels
// and runs them. Everything above is written against these six modules.
#[cfg(not(metal_driver))]
mod commands;
#[cfg(not(metal_driver))]
mod device;
#[cfg(not(metal_driver))]
mod instance;
#[cfg(not(metal_driver))]
mod memory;
#[cfg(not(metal_driver))]
mod physical;
#[cfg(not(metal_driver))]
mod pipeline;

#[cfg(metal_driver)]
#[path = "metal/commands.rs"]
mod commands;
#[cfg(metal_driver)]
#[path = "metal/device.rs"]
mod device;
#[cfg(metal_driver)]
#[path = "metal/instance.rs"]
mod instance;
#[cfg(metal_driver)]
#[path = "metal/memory.rs"]
mod memory;
#[cfg(metal_driver)]
#[path = "metal/physical.rs"]
mod physical;
#[cfg(metal_driver)]
#[path = "metal/pipeline.rs"]
mod pipeline;

/// What drives the GPU: Vulkan, or Metal with the `metal` feature on macOS.
#[cfg(not(metal_driver))]
pub const API: &str = "Vulkan";
#[cfg(metal_driver)]
pub const API: &str = "Metal";

pub use attention::{AttentionGate, AttentionShape, KvCache, QkvConsts};
pub use batch::{Batch, BATCH_HEADER};
pub use commands::{Commands, Graph, Timestamp, Timings};
pub use delta_net::{DeltaShape, DeltaState, DeltaWeights};
pub use device::Device;
pub use error::{Error, Result};
pub use gemv::{DeviceMatrix, GemvInput, Out, Rows};
pub use info::{DeviceInfo, DeviceKind, Features};
pub use instance::Instance;
pub use kernels::Kernels;
pub use memory::{DeviceBuffer, DeviceCopy, HostBuffer};
pub use norm::Residual;
pub use physical::PhysicalDevice;
pub use short_conv::ConvRing;
