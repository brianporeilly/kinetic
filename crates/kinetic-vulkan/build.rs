//! Compiles the WGSL kernels in `shaders/` to SPIR-V with naga, so building
//! needs no Vulkan SDK. Each kernel is its constants followed by the
//! concatenation of its source parts; `src/kernels.rs` embeds the resulting
//! `<name>.spv` files.

use std::env;
use std::fs;
use std::path::PathBuf;

use naga::back::spv;
use naga::valid::{Capabilities, ValidationFlags, Validator};

/// (kernel, constants, source parts in order).
const KERNELS: &[(&str, &str, &[&str])] = &[
    ("gemv_f32", "", &["gemv", "gemv_f32"]),
    ("gemv_f16", "", &["gemv", "gemv_f16"]),
    ("gemv_q8_0", "", &["gemv", "gemv_q8_0"]),
    ("gemv_q4_k", "", &["gemv", "k4", "gemv_q4_k"]),
    ("gemv_q5_k", "", &["gemv", "k4", "gemv_q5_k"]),
    ("gemv_q6_k", "", &["gemv", "gemv_q6_k"]),
    ("quantize_q8", "", &["quantize_q8"]),
    ("read_sum", "", &["read_sum"]),
    ("norm", "", &["norm", "put_q8"]),
    ("swiglu", "", &["swiglu", "put_q8"]),
    ("argmax", "", &["argmax"]),
    ("rope", "", &["rope"]),
    ("short_conv", "", &["short_conv", "put_q8"]),
    ("delta_conv", "", &["delta_conv"]),
    ("delta_rule", "", &["delta_rule"]),
    ("gated_norm", "", &["gated_norm", "put_q8"]),
    ("attend_64", "const PAIRS: u32 = 1u;", &["attend", "put_q8"]),
    (
        "attend_128",
        "const PAIRS: u32 = 2u;",
        &["attend", "put_q8"],
    ),
    (
        "attend_256",
        "const PAIRS: u32 = 4u;",
        &["attend", "put_q8"],
    ),
];

/// Kernels on the matrix units, for devices with cooperative f16 matrices
/// (`Features::matrix_f16`): f16 arithmetic and cooperative matrices
/// validate, and their modules take the Vulkan memory model. Not built for
/// Metal, whose simdgroup matrices are 8x8.
const MATRIX_KERNELS: &[(&str, &str, &[&str])] = &[("mm_q8_0", "", &["mm_q8_0"])];

/// Kernels that take one part more on Metal, whose GEMV kernels are
/// written differently: there an invocation takes a tile of four rows and
/// the block sums are floats, which is faster on a Mac's GPU and slower on
/// the others. A part with a file in `shaders/metal` is read from there.
#[cfg(feature = "metal")]
const METAL_PARTS: &[(&str, &str)] = &[
    ("gemv_f32", "gemv_dots"),
    ("gemv_f16", "gemv_dots"),
    ("gemv_q8_0", "gemv_dots"),
    ("gemv_q5_k", "gemv_dots"),
];

/// The source of a kernel: its constants, then its parts from `dirs`, the
/// first that has each.
fn source(constants: &str, parts: &[&str], dirs: &[&str]) -> String {
    std::iter::once(constants.to_string())
        .chain(parts.iter().map(|part| {
            let path = dirs
                .iter()
                .map(|dir| format!("{dir}/{part}.wgsl"))
                .find(|path| fs::exists(path).unwrap_or(false))
                .unwrap_or_else(|| panic!("no shader part {part}"));
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"))
        }))
        .collect::<Vec<_>>()
        .join("\n")
}

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rustc-check-cfg=cfg(metal_driver)");
    let metal =
        cfg!(feature = "metal") && env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "macos");
    if metal {
        println!("cargo:rustc-cfg=metal_driver");
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"));
    for &(name, constants, parts) in MATRIX_KERNELS {
        let source = source(constants, parts, &["shaders"]);
        let (module, info) = parse_with(name, &source, MATRIX_CAPABILITIES);
        fs::write(out.join(format!("{name}.spv")), spirv(name, &module, &info))
            .expect("write SPIR-V");
    }
    for &(name, constants, parts) in KERNELS {
        let (module, info) = parse(name, &source(constants, parts, &["shaders"]));
        fs::write(out.join(format!("{name}.spv")), spirv(name, &module, &info))
            .expect("write SPIR-V");
        #[cfg(feature = "metal")]
        if metal {
            let mut parts = parts.to_vec();
            parts.extend(METAL_PARTS.iter().filter(|p| p.0 == name).map(|p| p.1));
            let source = source(constants, &parts, &["shaders/metal", "shaders"]);
            let (module, info) = parse(name, &source);
            fs::write(out.join(format!("{name}.metal")), msl(name, &module, &info))
                .expect("write Metal source");
        }
    }
}

/// What every kernel may use.
const CAPABILITIES: Capabilities = Capabilities::IMMEDIATES
    .union(Capabilities::SUBGROUP)
    .union(Capabilities::SHADER_FLOAT16_IN_FLOAT32);

/// What the matrix kernels may use besides.
const MATRIX_CAPABILITIES: Capabilities = CAPABILITIES
    .union(Capabilities::SHADER_FLOAT16)
    .union(Capabilities::COOPERATIVE_MATRIX);

fn parse(name: &str, source: &str) -> (naga::Module, naga::valid::ModuleInfo) {
    parse_with(name, source, CAPABILITIES)
}

fn parse_with(
    name: &str,
    source: &str,
    capabilities: Capabilities,
) -> (naga::Module, naga::valid::ModuleInfo) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{name}:\n{}", e.emit_to_string(source)));
    let info = Validator::new(ValidationFlags::all(), capabilities)
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}:\n{}", e.emit_to_string(source)));
    (module, info)
}

/// The kernel as a Metal shader. Storage buffer `i` of set 0 is Metal
/// buffer `i`; the push constants follow them, then the sizes naga wants
/// for runtime arrays, which `src/metal/commands.rs` binds in that order.
///
/// Metal wants the size of each block of workgroup memory when a kernel is
/// dispatched, so the first line lists them, in the order the entry point
/// takes them, for `src/metal/pipeline.rs` to read back.
#[cfg(feature = "metal")]
fn msl(name: &str, module: &naga::Module, info: &naga::valid::ModuleInfo) -> String {
    use naga::back::msl;
    let bindings = module
        .global_variables
        .iter()
        .filter_map(|(_, v)| v.binding.as_ref())
        .map(|b| b.binding + 1)
        .max()
        .unwrap_or(0);
    let mut resources = msl::EntryPointResources::default();
    for binding in 0..bindings {
        resources.resources.insert(
            naga::ResourceBinding { group: 0, binding },
            msl::BindTarget {
                buffer: Some(binding as u8),
                mutable: true,
                ..Default::default()
            },
        );
    }
    resources.immediates_buffer = Some(bindings as u8);
    resources.sizes_buffer = Some(bindings as u8 + 1);
    let mut options = msl::Options {
        lang_version: (3, 0),
        ..Default::default()
    };
    options.per_entry_point_map.insert("main".into(), resources);
    let source = msl::write_string(module, info, &options, &msl::PipelineOptions::default())
        .unwrap_or_else(|e| panic!("{name}: {e}"))
        .0;
    let mut layouts = naga::proc::Layouter::default();
    layouts
        .update(module.to_ctx())
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let workgroup: Vec<String> = module
        .global_variables
        .iter()
        .filter(|(_, v)| v.space == naga::AddressSpace::WorkGroup)
        // Metal takes the lengths in multiples of 16 bytes.
        .map(|(_, v)| layouts[v.ty].size.next_multiple_of(16).to_string())
        .collect();
    format!("{THREADGROUP_BYTES}{}\n{source}", workgroup.join(" "))
}

/// What starts the line of workgroup memory sizes in a Metal shader.
#[cfg(feature = "metal")]
const THREADGROUP_BYTES: &str = "// threadgroup bytes: ";

fn spirv(name: &str, module: &naga::Module, info: &naga::valid::ModuleInfo) -> Vec<u8> {
    // Vulkan 1.3 consumes SPIR-V 1.6, which has packed int8 dot products in core.
    let options = spv::Options {
        lang_version: (1, 6),
        flags: spv::WriterFlags::empty(),
        emit_int_div_checks: false,
        force_loop_bounding: false,
        // f16 never crosses a stage boundary in a compute kernel
        use_storage_input_output_16: false,
        ..spv::Options::default()
    };
    let words =
        spv::write_vec(module, info, &options, None).unwrap_or_else(|e| panic!("{name}: {e}"));
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}
