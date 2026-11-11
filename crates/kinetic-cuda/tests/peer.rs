//! Timing copies between devices, which `kinetic doctor` reports.

use kinetic_cuda::peer::time_copies;
use kinetic_cuda::Device;

#[test]
fn copies_between_contexts_are_timed() {
    let Ok(device) = Device::get(0) else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    // device 0 to itself still copies between two contexts' buffers
    let p = time_copies(device, device).unwrap();
    assert!(!p.stuck);
    let t = p.through_host;
    assert!(t.small_us.is_some_and(|us| us > 0.0), "{p:?}");
    assert!(t.large_bytes_per_s.is_some_and(|b| b > 1e8), "{p:?}");
}
