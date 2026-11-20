use std::ffi::CStr;
use std::rc::Rc;

use ash::vk;

use crate::error::{Check, Result};
use crate::info::{DeviceInfo, DeviceKind, Features};
use crate::instance::{Instance, API_VERSION};

/// The subgroup size every kernel is written for.
pub(crate) const SUBGROUP: u32 = 32;

fn kind(ty: vk::PhysicalDeviceType) -> DeviceKind {
    match ty {
        vk::PhysicalDeviceType::DISCRETE_GPU => DeviceKind::Discrete,
        vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceKind::Integrated,
        vk::PhysicalDeviceType::VIRTUAL_GPU => DeviceKind::Virtual,
        vk::PhysicalDeviceType::CPU => DeviceKind::Cpu,
        _ => DeviceKind::Other,
    }
}

/// A Vulkan device as the instance reports it, before it is opened.
#[derive(Clone)]
pub struct PhysicalDevice {
    pub(crate) instance: Rc<Instance>,
    pub(crate) raw: vk::PhysicalDevice,
    pub(crate) compute_family: Option<u32>,
    pub(crate) memory: vk::PhysicalDeviceMemoryProperties,
    info: DeviceInfo,
}

impl PhysicalDevice {
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    pub(crate) fn query(instance: &Rc<Instance>, raw: vk::PhysicalDevice) -> Result<Self> {
        let vk_instance = &instance.raw;
        // SAFETY: `raw` came from this instance; every out-structure below is
        // owned here and its `p_next` chain only links structures that
        // outlive the call.
        let (props, memory, families, extensions) = unsafe {
            (
                vk_instance.get_physical_device_properties(raw),
                vk_instance.get_physical_device_memory_properties(raw),
                vk_instance.get_physical_device_queue_family_properties(raw),
                vk_instance
                    .enumerate_device_extension_properties(raw)
                    .check("vkEnumerateDeviceExtensionProperties")?,
            )
        };
        let has_extension = |name: &CStr| {
            extensions
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(name))
        };

        let compute_family = families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|i| i as u32);
        let mut features = Features {
            push_descriptor: has_extension(ash::khr::push_descriptor::NAME),
            timestamps: compute_family
                .is_some_and(|i| families[i as usize].timestamp_valid_bits > 0),
            ..Features::default()
        };
        let mut driver_name = String::new();
        let mut driver_info = String::new();
        if props.api_version >= API_VERSION {
            let mut p11 = vk::PhysicalDeviceVulkan11Properties::default();
            let mut p12 = vk::PhysicalDeviceVulkan12Properties::default();
            let mut p13 = vk::PhysicalDeviceVulkan13Properties::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut p11)
                .push_next(&mut p12)
                .push_next(&mut p13);
            let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut coop = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
            let has_coop = has_extension(ash::khr::cooperative_matrix::NAME);
            let mut f2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut f11)
                .push_next(&mut f12)
                .push_next(&mut f13);
            if has_coop {
                f2 = f2.push_next(&mut coop);
            }
            // SAFETY: as above; the chains hold only core 1.3 structures and,
            // when the extension exists, its own feature structure.
            unsafe {
                vk_instance.get_physical_device_properties2(raw, &mut p2);
                vk_instance.get_physical_device_features2(raw, &mut f2);
            }
            features.subgroup_size = p11.subgroup_size;
            features.subgroup_size_range = (p13.min_subgroup_size, p13.max_subgroup_size);
            features.subgroup_arithmetic = p11
                .subgroup_supported_stages
                .contains(vk::ShaderStageFlags::COMPUTE)
                && p11
                    .subgroup_supported_operations
                    .contains(vk::SubgroupFeatureFlags::ARITHMETIC);
            features.subgroup_size_required = p13
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE)
                && (p13.min_subgroup_size..=p13.max_subgroup_size).contains(&SUBGROUP);
            // A pipeline that neither requires a size nor allows a varying
            // one runs at the device's default size.
            features.subgroup_size_32 = f13.subgroup_size_control == vk::TRUE
                && f13.compute_full_subgroups == vk::TRUE
                && (features.subgroup_size_required || p11.subgroup_size == SUBGROUP);
            features.integer_dot_product = f13.shader_integer_dot_product == vk::TRUE;
            features.integer_dot_product_accelerated =
                p13.integer_dot_product4x8_bit_packed_signed_accelerated == vk::TRUE;
            features.storage_16bit = f11.storage_buffer16_bit_access == vk::TRUE;
            features.storage_8bit = f12.storage_buffer8_bit_access == vk::TRUE;
            features.cooperative_matrix = has_coop && coop.cooperative_matrix == vk::TRUE;
            features.matrix_f16 = features.cooperative_matrix
                && f12.shader_float16 == vk::TRUE
                && f12.vulkan_memory_model == vk::TRUE
                && f12.vulkan_memory_model_device_scope == vk::TRUE
                && f11.storage_buffer16_bit_access == vk::TRUE
                && f11.uniform_and_storage_buffer16_bit_access == vk::TRUE
                && has_matrix_f16(instance, raw);
            driver_name = c_text(p12.driver_name_as_c_str());
            driver_info = c_text(p12.driver_info_as_c_str());
        }

        let heaps = &memory.memory_heaps[..memory.memory_heap_count as usize];
        let device_local_bytes = heaps
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .max()
            .unwrap_or(0);
        let info = DeviceInfo {
            name: c_text(props.device_name_as_c_str()),
            kind: kind(props.device_type),
            vendor_id: props.vendor_id,
            device_id: props.device_id,
            api_version: (
                vk::api_version_major(props.api_version),
                vk::api_version_minor(props.api_version),
                vk::api_version_patch(props.api_version),
            ),
            driver_name,
            driver_info,
            device_local_bytes,
            max_buffer_bytes: props.limits.max_storage_buffer_range.into(),
            timestamp_period: f64::from(props.limits.timestamp_period) * 1e-9,
            features,
        };
        Ok(PhysicalDevice {
            instance: Rc::clone(instance),
            raw,
            compute_family,
            memory,
            info,
        })
    }
}

fn c_text<E>(s: std::result::Result<&CStr, E>) -> String {
    s.map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Whether a subgroup of `raw` multiplies 16x16 f16 matrices into a 16x16
/// f32 one, the only shape the matrix kernels take.
fn has_matrix_f16(instance: &Instance, raw: vk::PhysicalDevice) -> bool {
    let coop = ash::khr::cooperative_matrix::Instance::new(&instance.entry, &instance.raw);
    // SAFETY: `raw` came from this instance, which has the extension's
    // entry point when the device lists the extension, as the caller checked.
    let shapes = unsafe { coop.get_physical_device_cooperative_matrix_properties(raw) };
    shapes.unwrap_or_default().iter().any(|p| {
        p.m_size == 16
            && p.n_size == 16
            && p.k_size == 16
            && p.a_type == vk::ComponentTypeKHR::FLOAT16
            && p.b_type == vk::ComponentTypeKHR::FLOAT16
            && p.c_type == vk::ComponentTypeKHR::FLOAT32
            && p.result_type == vk::ComponentTypeKHR::FLOAT32
            && p.scope == vk::ScopeKHR::SUBGROUP
    })
}
