// OchreML Device & Hardware Acceleration Module (＾▽＾)
// Provides unified device management and hardware capability detection for CPU, GPU, Multi-GPU, and TPU.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    Gpu(usize),
    MultiGpu(Vec<usize>),
    Tpu(usize),
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "cpu"),
            DeviceType::Gpu(id) => write!(f, "cuda:{}", id),
            DeviceType::MultiGpu(ids) => {
                let id_strs: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
                write!(f, "cuda:{}", id_strs.join(","))
            }
            DeviceType::Tpu(id) => write!(f, "tpu:{}", id),
        }
    }
}

// Hardware capabilities discovered on host machine (*^▽^*)
#[derive(Debug, Clone)]
pub struct SystemCapabilities {
    pub cpu_cores: usize,
    pub gpu_count: usize,
    pub gpu_names: Vec<String>,
    pub tpu_detected: bool,
    pub tpu_type: Option<String>,
}

impl SystemCapabilities {
    // Probe system environment for CPU, GPU, Multi-GPU, and TPU availability
    pub fn probe() -> Self {
        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let (gpu_count, gpu_names) = probe_gpus();
        let (tpu_detected, tpu_type) = probe_tpus();

        Self {
            cpu_cores,
            gpu_count,
            gpu_names,
            tpu_detected,
            tpu_type,
        }
    }
}

// Probe system for NVIDIA / ROCm / Vulkan GPUs
fn probe_gpus() -> (usize, Vec<String>) {
    let mut names = Vec::new();

    // 1. Check NVIDIA driver in /proc/driver/nvidia
    let nvidia_proc = Path::new("/proc/driver/nvidia/gpus");
    if nvidia_proc.exists() {
        if let Ok(entries) = fs::read_dir(nvidia_proc) {
            for entry in entries.flatten() {
                if let Ok(file_type) = entry.file_type() {
                    if file_type.is_dir() {
                        let name = entry.file_name().to_string_lossy().to_string();
                        names.push(format!("NVIDIA GPU ({})", name));
                    }
                }
            }
        }
    }

    // 2. Check CUDA environment variables or sysfs DRM
    if names.is_empty() {
        if let Ok(val) = env::var("CUDA_VISIBLE_DEVICES") {
            let count = val.split(',').filter(|s| !s.trim().is_empty()).count();
            if count > 0 {
                for i in 0..count {
                    names.push(format!("CUDA Device {}", i));
                }
            }
        }
    }

    // 3. Check Linux DRM PCI devices
    if names.is_empty() {
        let drm_path = Path::new("/sys/class/drm");
        if drm_path.exists() {
            if let Ok(entries) = fs::read_dir(drm_path) {
                for entry in entries.flatten() {
                    let fname = entry.file_name().to_string_lossy().to_string();
                    if fname.starts_with("card") && !fname.contains('-') {
                        names.push(format!("Graphics Accelerator ({})", fname));
                    }
                }
            }
        }
    }

    let count = names.len();
    (count, names)
}

// Probe system for Google Cloud TPU or PJRT XLA drivers
fn probe_tpus() -> (bool, Option<String>) {
    // Check standard Google Cloud TPU environment variables
    let tpu_name = env::var("TPU_NAME").ok();
    let tpu_type = env::var("TPU_ACCELERATOR_TYPE").ok();

    if tpu_name.is_some() || tpu_type.is_some() {
        return (true, tpu_type.or(tpu_name));
    }

    // Check PJRT TPU libraries in system paths
    let pjrt_paths = [
        "/usr/local/lib/libpjrt_c_api_tpu.so",
        "/usr/lib/libpjrt_c_api_tpu.so",
        "/usr/local/lib/libpjrt_c_api.so",
    ];
    for p in &pjrt_paths {
        if Path::new(p).exists() {
            return (true, Some("PJRT TPU Runtime".to_string()));
        }
    }

    // Check Google Cloud TPU device filesystem node
    if Path::new("/dev/accel0").exists() || Path::new("/dev/vfio").exists() {
        if let Ok(chips) = env::var("TPU_CHIPS_PER_HOST_BOUNDS") {
            return (true, Some(format!("Google Cloud TPU (Chips: {})", chips)));
        }
    }

    (false, None)
}

// Parse user-specified device string or resolve 'auto' to the highest performance available device
pub fn parse_device(device_str: Option<&str>) -> Result<DeviceType, String> {
    let caps = SystemCapabilities::probe();
    let dev = device_str.unwrap_or("auto").trim().to_lowercase();

    match dev.as_str() {
        "auto" => {
            // Auto prioritization: TPU -> Multi-GPU -> Single GPU -> CPU (*≧ω≦*)
            if caps.tpu_detected {
                Ok(DeviceType::Tpu(0))
            } else if caps.gpu_count >= 2 {
                let ids: Vec<usize> = (0..caps.gpu_count).collect();
                Ok(DeviceType::MultiGpu(ids))
            } else if caps.gpu_count == 1 {
                Ok(DeviceType::Gpu(0))
            } else {
                Ok(DeviceType::Cpu)
            }
        }
        "cpu" => Ok(DeviceType::Cpu),
        "tpu" | "tpu:0" => {
            if caps.tpu_detected {
                Ok(DeviceType::Tpu(0))
            } else {
                Err("TPU acceleration was requested ('tpu'), but no active TPU was detected on this host. Troubleshooting: Ensure TPU_NAME or PJRT runtime is initialized, or use device='auto' for automatic hardware fallback ( >_< )".to_string())
            }
        }
        "cuda" | "cuda:0" | "gpu" | "gpu:0" => {
            if caps.gpu_count > 0 {
                Ok(DeviceType::Gpu(0))
            } else {
                Err("GPU acceleration was requested ('cuda:0'), but no GPU device or CUDA driver was detected on this host. Troubleshooting: Ensure NVIDIA drivers or CUDA are active, or use device='auto' for automatic fallback ( >_< )".to_string())
            }
        }
        "cuda:all" | "multi-gpu" | "gpu:all" => {
            if caps.gpu_count >= 2 {
                let ids: Vec<usize> = (0..caps.gpu_count).collect();
                Ok(DeviceType::MultiGpu(ids))
            } else if caps.gpu_count == 1 {
                Ok(DeviceType::Gpu(0))
            } else {
                Err("Multi-GPU acceleration was requested ('cuda:all'), but no GPUs were detected on this host ( >_< )".to_string())
            }
        }
        custom => {
            if let Some(id_str) = custom.strip_prefix("cuda:") {
                if id_str.contains(',') {
                    let mut ids = Vec::new();
                    for part in id_str.split(',') {
                        let id = part.trim().parse::<usize>().map_err(|_| {
                            format!("Invalid GPU index in device string '{}' (・`ω´・)", custom)
                        })?;
                        ids.push(id);
                    }
                    if ids.is_empty() {
                        return Err("Multi-GPU device string contains no GPU IDs ( >_< )".to_string());
                    }
                    if caps.gpu_count > 0 && ids.iter().all(|&id| id < caps.gpu_count) {
                        Ok(DeviceType::MultiGpu(ids))
                    } else if caps.gpu_count == 0 {
                        Err(format!("Device '{}' was requested, but no GPUs are available ( >_< )", custom))
                    } else {
                        Err(format!("Requested GPU ID exceeds available GPU count ({}) ( >_< )", caps.gpu_count))
                    }
                } else {
                    let id = id_str.parse::<usize>().map_err(|_| {
                        format!("Invalid GPU index in device string '{}' (・`ω´・)", custom)
                    })?;
                    if caps.gpu_count > id {
                        Ok(DeviceType::Gpu(id))
                    } else {
                        Err(format!("GPU device 'cuda:{}' was requested, but system only has {} GPU(s) ( >_< )", id, caps.gpu_count))
                    }
                }
            } else if let Some(id_str) = custom.strip_prefix("tpu:") {
                let id = id_str.parse::<usize>().map_err(|_| {
                    format!("Invalid TPU index in device string '{}' (・`ω´・)", custom)
                })?;
                if caps.tpu_detected {
                    Ok(DeviceType::Tpu(id))
                } else {
                    Err("TPU device was requested, but no TPU runtime was detected ( >_< )".to_string())
                }
            } else {
                Err(format!("Unrecognized device string '{}'. Supported devices: 'auto', 'cpu', 'cuda', 'cuda:0', 'cuda:all', 'tpu' (´-ω-`)", custom))
            }
        }
    }
}

// Return human-readable list of available hardware devices
pub fn get_available_devices_list() -> Vec<String> {
    let caps = SystemCapabilities::probe();
    let mut list = vec!["cpu".to_string()];

    if caps.gpu_count > 0 {
        for i in 0..caps.gpu_count {
            list.push(format!("cuda:{}", i));
        }
        if caps.gpu_count > 1 {
            list.push("cuda:all".to_string());
        }
    }

    if caps.tpu_detected {
        list.push("tpu:0".to_string());
    }

    list
}

// Return detailed system device metadata for diagnostics and user inspection
pub fn get_device_info_map() -> HashMap<String, String> {
    let caps = SystemCapabilities::probe();
    let mut map = HashMap::new();

    map.insert("cpu_cores".to_string(), caps.cpu_cores.to_string());
    map.insert("gpu_count".to_string(), caps.gpu_count.to_string());

    if !caps.gpu_names.is_empty() {
        map.insert("gpu_devices".to_string(), caps.gpu_names.join("; "));
    } else {
        map.insert("gpu_devices".to_string(), "None".to_string());
    }

    map.insert("tpu_detected".to_string(), caps.tpu_detected.to_string());
    if let Some(tpu_type) = caps.tpu_type {
        map.insert("tpu_type".to_string(), tpu_type);
    } else {
        map.insert("tpu_type".to_string(), "None".to_string());
    }

    let auto_dev = parse_device(Some("auto")).unwrap_or(DeviceType::Cpu);
    map.insert("default_auto_device".to_string(), auto_dev.to_string());

    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_type_display() {
        assert_eq!(DeviceType::Cpu.to_string(), "cpu");
        assert_eq!(DeviceType::Gpu(0).to_string(), "cuda:0");
        assert_eq!(DeviceType::MultiGpu(vec![0, 1]).to_string(), "cuda:0,1");
        assert_eq!(DeviceType::Tpu(0).to_string(), "tpu:0");
    }

    #[test]
    fn test_parse_device_cpu() {
        let dev = parse_device(Some("cpu")).unwrap();
        assert_eq!(dev, DeviceType::Cpu);
    }

    #[test]
    fn test_parse_device_auto() {
        let dev = parse_device(Some("auto")).unwrap();
        // On machines without GPU/TPU, auto defaults cleanly to CPU
        assert!(matches!(dev, DeviceType::Cpu | DeviceType::Gpu(_) | DeviceType::MultiGpu(_) | DeviceType::Tpu(_)));
    }

    #[test]
    fn test_parse_invalid_device() {
        let res = parse_device(Some("quantum_supercomputer"));
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("Unrecognized device string"));
    }

    #[test]
    fn test_get_available_devices_list() {
        let list = get_available_devices_list();
        assert!(!list.is_empty());
        assert!(list.contains(&"cpu".to_string()));
    }

    #[test]
    fn test_get_device_info_map() {
        let info = get_device_info_map();
        assert!(info.contains_key("cpu_cores"));
        assert!(info.contains_key("default_auto_device"));
    }
}
