use std::rc::Rc;

use ash::vk;

use crate::error::{Check, Error, Result};
use crate::physical::PhysicalDevice;

/// The Vulkan version this crate asks for; its kernels need 1.3 devices.
pub(crate) const API_VERSION: u32 = vk::API_VERSION_1_3;

/// Where Homebrew installs the Vulkan loader on Apple Silicon and on Intel.
const HOMEBREW_LOADERS: [&str; 2] = [
    "/opt/homebrew/lib/libvulkan.dylib",
    "/usr/local/lib/libvulkan.dylib",
];

/// Where Homebrew's Mesa puts the manifest of KosmicKrisp, which its loader
/// does not search by default.
const KOSMICKRISP_MANIFESTS: [&str; 2] = [
    "/opt/homebrew/share/vulkan/icd.d/kosmickrisp_mesa_icd.aarch64.json",
    "/usr/local/share/vulkan/icd.d/kosmickrisp_mesa_icd.aarch64.json",
];

/// The Vulkan loader, opened at runtime, and one instance of it.
pub struct Instance {
    /// Keeps the loader library open while the instance's functions are used.
    pub(crate) entry: ash::Entry,
    pub(crate) raw: ash::Instance,
}

impl Instance {
    /// Opens the system's Vulkan loader (`vulkan-1.dll`, `libvulkan.so.1`,
    /// `libvulkan.dylib`) and creates an instance. Fails with [`Error::LoaderNotFound`] when no
    /// Vulkan driver is installed.
    pub fn new() -> Result<Rc<Instance>> {
        // The loader reads its environment when an instance is created, so
        // KosmicKrisp is added before that, unless the user chose drivers.
        if cfg!(target_os = "macos")
            && std::env::var_os("VK_DRIVER_FILES").is_none()
            && std::env::var_os("VK_ADD_DRIVER_FILES").is_none()
        {
            if let Some(manifest) = KOSMICKRISP_MANIFESTS
                .iter()
                .find(|m| std::path::Path::new(m).exists())
            {
                std::env::set_var("VK_ADD_DRIVER_FILES", manifest);
            }
        }
        // SAFETY: loading the loader runs its initialisation, which has no
        // preconditions; the library stays loaded while `entry` lives.
        let entry = unsafe { ash::Entry::load() }
            .or_else(|e| {
                // Homebrew's loader is not on the dynamic linker's path.
                HOMEBREW_LOADERS
                    .iter()
                    .filter(|_| cfg!(target_os = "macos"))
                    // SAFETY: as above.
                    .find_map(|path| unsafe { ash::Entry::load_from(path) }.ok())
                    .ok_or(e)
            })
            .map_err(|e| Error::LoaderNotFound(e.to_string()))?;
        let app = vk::ApplicationInfo::default()
            .application_name(c"kinetic")
            .engine_name(c"kinetic")
            .api_version(API_VERSION);
        let mut info = vk::InstanceCreateInfo::default().application_info(&app);
        // macOS has no native Vulkan driver: MoltenVK implements it on Metal
        // as a portability driver, which the loader lists only when asked.
        let portability = [ash::khr::portability_enumeration::NAME.as_ptr()];
        if cfg!(target_os = "macos") {
            info = info
                .enabled_extension_names(&portability)
                .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
        }
        // SAFETY: `info` and what it points to are valid for the call.
        let raw = unsafe { entry.create_instance(&info, None) }.check("vkCreateInstance")?;
        Ok(Rc::new(Instance { entry, raw }))
    }

    /// Every physical device, discrete GPUs first, then integrated ones, then
    /// the rest, each group in driver order.
    pub fn physical_devices(self: &Rc<Self>) -> Result<Vec<PhysicalDevice>> {
        // SAFETY: the instance is valid for the life of `self`.
        let raw =
            unsafe { self.raw.enumerate_physical_devices() }.check("vkEnumeratePhysicalDevices")?;
        let mut devices = raw
            .into_iter()
            .map(|pd| PhysicalDevice::query(self, pd))
            .collect::<Result<Vec<_>>>()?;
        // On a Mac one GPU appears once per driver; KosmicKrisp is the
        // faster of the two on Metal.
        devices.sort_by_key(|d| {
            let info = d.info();
            (
                info.kind.preference(),
                !info.driver_name.contains("KosmicKrisp"),
            )
        });
        Ok(devices)
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: every object made from the instance holds an `Rc` to it and
        // is gone by now.
        unsafe { self.raw.destroy_instance(None) };
    }
}
