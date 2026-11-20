//! What a device is and offers, whichever driver reports it.

use crate::physical::SUBGROUP;

/// What kind of processor a device is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    Discrete,
    Integrated,
    Virtual,
    Cpu,
    Other,
}

impl DeviceKind {
    /// Sort key: lower is preferred.
    #[cfg_attr(metal_driver, allow(dead_code))]
    pub(crate) fn preference(self) -> u8 {
        match self {
            DeviceKind::Discrete => 0,
            DeviceKind::Integrated => 1,
            DeviceKind::Virtual => 2,
            DeviceKind::Other => 3,
            DeviceKind::Cpu => 4,
        }
    }
}

/// Optional capabilities found on a device. Those the kernels rely on are
/// listed by [`DeviceInfo::missing`] when absent.
#[derive(Clone, Debug, Default)]
pub struct Features {
    /// The subgroup size compute shaders get by default.
    pub subgroup_size: u32,
    /// The smallest and largest subgroup sizes a pipeline can ask for.
    pub subgroup_size_range: (u32, u32),
    /// Subgroup reductions such as `subgroupAdd` in compute shaders.
    pub subgroup_arithmetic: bool,
    /// Compute pipelines run with full subgroups of 32: either they can
    /// require that size, or it is the only size the device has.
    pub subgroup_size_32: bool,
    /// Compute pipelines can require a subgroup size. Without it (the
    /// Vulkan drivers on Metal) the size is the device's own.
    pub subgroup_size_required: bool,
    /// Packed int8 dot products (`VK_KHR_shader_integer_dot_product`).
    pub integer_dot_product: bool,
    /// The driver reports packed signed int8 dot products as hardware-accelerated.
    pub integer_dot_product_accelerated: bool,
    /// 16-bit types in storage buffers.
    pub storage_16bit: bool,
    /// 8-bit types in storage buffers.
    pub storage_8bit: bool,
    /// Cooperative matrix operations (`VK_KHR_cooperative_matrix`).
    pub cooperative_matrix: bool,
    /// What the f16 matrix kernels for prompt passes need: cooperative
    /// 16x16x16 f16 products into f32 in a subgroup, f16 arithmetic, 16-bit
    /// storage and the Vulkan memory model.
    pub matrix_f16: bool,
    /// Descriptors pushed straight into command buffers (`VK_KHR_push_descriptor`).
    pub push_descriptor: bool,
    /// The compute queue can write timestamps.
    pub timestamps: bool,
}

/// Properties of a physical device.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub kind: DeviceKind,
    pub vendor_id: u32,
    pub device_id: u32,
    /// The Vulkan version the driver supports, as (major, minor, patch).
    pub api_version: (u32, u32, u32),
    pub driver_name: String,
    /// The driver's own version string.
    pub driver_info: String,
    /// Size of the largest device-local memory heap.
    pub device_local_bytes: u64,
    /// The largest storage buffer a kernel can bind.
    pub max_buffer_bytes: u64,
    /// Seconds per timestamp tick.
    pub timestamp_period: f64,
    pub features: Features,
}

impl DeviceInfo {
    /// Why this crate's kernels cannot run on the device, or `None` when they can.
    pub fn missing(&self) -> Option<String> {
        let f = &self.features;
        let (major, minor, _) = self.api_version;
        let mut lacks = Vec::new();
        if (major, minor) < (1, 3) {
            lacks.push(format!("Vulkan 1.3 (driver offers {major}.{minor})"));
        }
        let subgroups = match (self.subgroups_too_wide(), f.subgroup_size_range) {
            (true, (min, max)) if min == max => {
                format!("subgroups of 32 (this GPU only runs {min})")
            }
            (true, (min, max)) => format!("subgroups of 32 (this GPU only runs {min} to {max})"),
            (false, _) => "subgroups of 32".into(),
        };
        for (have, what) in [
            (f.subgroup_arithmetic, "subgroup arithmetic"),
            (f.subgroup_size_32, subgroups.as_str()),
            (f.integer_dot_product, "integer dot products"),
            (f.push_descriptor, "push descriptors"),
        ] {
            if !have {
                lacks.push(what.to_string());
            }
        }
        (!lacks.is_empty()).then(|| format!("needs {}", lacks.join(", ")))
    }

    /// The device only runs subgroups wider than the 32 the kernels are
    /// written for, which is the hardware and not its driver.
    pub fn subgroups_too_wide(&self) -> bool {
        let (min, _) = self.features.subgroup_size_range;
        !self.features.subgroup_size_32 && min > SUBGROUP
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(api_version: (u32, u32, u32), features: Features) -> DeviceInfo {
        DeviceInfo {
            name: "test".into(),
            kind: DeviceKind::Integrated,
            vendor_id: 0x1002,
            device_id: 0,
            api_version,
            driver_name: String::new(),
            driver_info: String::new(),
            device_local_bytes: 0,
            max_buffer_bytes: 0,
            timestamp_period: 1.0,
            features,
        }
    }

    fn capable() -> Features {
        Features {
            subgroup_arithmetic: true,
            integer_dot_product: true,
            push_descriptor: true,
            ..Features::default()
        }
    }

    #[test]
    fn a_wave64_gpu_is_too_wide() {
        let gcn = device(
            (1, 3, 260),
            Features {
                subgroup_size: 64,
                subgroup_size_range: (64, 64),
                ..capable()
            },
        );
        assert!(gcn.subgroups_too_wide());
        assert_eq!(
            gcn.missing().as_deref(),
            Some("needs subgroups of 32 (this GPU only runs 64)")
        );
    }

    #[test]
    fn an_old_driver_is_not_too_wide() {
        let old = device((1, 2, 0), capable());
        assert!(!old.subgroups_too_wide());
        assert_eq!(
            old.missing().as_deref(),
            Some("needs Vulkan 1.3 (driver offers 1.2), subgroups of 32")
        );
    }

    #[test]
    fn a_gpu_with_subgroups_of_32_runs() {
        let rdna = device(
            (1, 3, 260),
            Features {
                subgroup_size: 64,
                subgroup_size_range: (32, 64),
                subgroup_size_32: true,
                ..capable()
            },
        );
        assert!(!rdna.subgroups_too_wide());
        assert_eq!(rdna.missing(), None);
    }
}
