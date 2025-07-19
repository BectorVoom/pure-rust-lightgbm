//! Device configuration and capabilities detection for Pure Rust LightGBM.
//!
//! This module provides device configuration management including CPU and GPU
//! settings, capability detection, and device-specific optimization parameters.

use crate::core::types::*;
use crate::core::error::{Result, LightGBMError};
use crate::core::constants::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Device configuration structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Primary device type (CPU or GPU)
    pub device_type: DeviceType,
    /// Number of threads for CPU operations
    pub num_threads: usize,
    /// GPU platform ID (OpenCL)
    pub gpu_platform_id: i32,
    /// GPU device ID
    pub gpu_device_id: i32,
    /// Use double precision on GPU
    pub gpu_use_dp: bool,
    /// GPU memory allocation strategy
    pub gpu_memory_strategy: GpuMemoryStrategy,
    /// GPU work group size
    pub gpu_work_group_size: usize,
    /// GPU max local memory usage
    pub gpu_max_local_memory: usize,
    /// GPU batch size for histogram construction
    pub gpu_batch_size: usize,
    /// GPU timeout in milliseconds
    pub gpu_timeout_ms: u64,
    /// Force column-wise histogram construction
    pub force_col_wise: bool,
    /// Force row-wise histogram construction
    pub force_row_wise: bool,
    /// CPU instruction set extensions
    pub cpu_extensions: CpuExtensions,
    /// Memory alignment preference
    pub memory_alignment: usize,
    /// Enable prefetching
    pub enable_prefetching: bool,
    /// NUMA node affinity
    pub numa_node: Option<usize>,
    /// Thread affinity mask
    pub thread_affinity: Option<Vec<usize>>,
}

/// GPU memory allocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuMemoryStrategy {
    /// Allocate all required memory upfront
    Preallocate,
    /// Allocate memory on-demand
    OnDemand,
    /// Use memory pools for efficiency
    Pooled,
}

impl Default for GpuMemoryStrategy {
    fn default() -> Self {
        GpuMemoryStrategy::Pooled
    }
}

/// CPU instruction set extensions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuExtensions {
    /// SSE support
    pub sse: bool,
    /// SSE2 support
    pub sse2: bool,
    /// SSE3 support
    pub sse3: bool,
    /// SSSE3 support
    pub ssse3: bool,
    /// SSE4.1 support
    pub sse4_1: bool,
    /// SSE4.2 support
    pub sse4_2: bool,
    /// AVX support
    pub avx: bool,
    /// AVX2 support
    pub avx2: bool,
    /// AVX-512 support
    pub avx512: bool,
    /// FMA support
    pub fma: bool,
    /// BMI1 support
    pub bmi1: bool,
    /// BMI2 support
    pub bmi2: bool,
}

impl Default for CpuExtensions {
    fn default() -> Self {
        CpuExtensions {
            sse: false,
            sse2: false,
            sse3: false,
            ssse3: false,
            sse4_1: false,
            sse4_2: false,
            avx: false,
            avx2: false,
            avx512: false,
            fma: false,
            bmi1: false,
            bmi2: false,
        }
    }
}

impl Default for DeviceConfig {
    fn default() -> Self {
        DeviceConfig {
            device_type: DeviceType::CPU,
            num_threads: DEFAULT_NUM_THREADS,
            gpu_platform_id: -1,
            gpu_device_id: -1,
            gpu_use_dp: false,
            gpu_memory_strategy: GpuMemoryStrategy::default(),
            gpu_work_group_size: MAX_GPU_WORK_GROUP_SIZE,
            gpu_max_local_memory: 64 * 1024, // 64KB
            gpu_batch_size: DEFAULT_GPU_BATCH_SIZE,
            gpu_timeout_ms: DEFAULT_GPU_TIMEOUT_MS,
            force_col_wise: false,
            force_row_wise: false,
            cpu_extensions: CpuExtensions::default(),
            memory_alignment: ALIGNED_SIZE,
            enable_prefetching: true,
            numa_node: None,
            thread_affinity: None,
        }
    }
}

impl DeviceConfig {
    /// Create a new device configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Create CPU-specific configuration
    pub fn cpu() -> Self {
        let mut config = DeviceConfig::default();
        config.device_type = DeviceType::CPU;
        config.detect_cpu_capabilities();
        config
    }

    /// Create GPU-specific configuration
    pub fn gpu() -> Self {
        let mut config = DeviceConfig::default();
        config.device_type = DeviceType::GPU;
        config.gpu_platform_id = 0;
        config.gpu_device_id = 0;
        config
    }

    /// Set the number of threads
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Set GPU platform and device IDs
    pub fn with_gpu_ids(mut self, platform_id: i32, device_id: i32) -> Self {
        self.gpu_platform_id = platform_id;
        self.gpu_device_id = device_id;
        self
    }

    /// Set GPU memory strategy
    pub fn with_gpu_memory_strategy(mut self, strategy: GpuMemoryStrategy) -> Self {
        self.gpu_memory_strategy = strategy;
        self
    }

    /// Set GPU work group size
    pub fn with_gpu_work_group_size(mut self, size: usize) -> Self {
        self.gpu_work_group_size = size;
        self
    }

    /// Set GPU batch size
    pub fn with_gpu_batch_size(mut self, batch_size: usize) -> Self {
        self.gpu_batch_size = batch_size;
        self
    }

    /// Enable column-wise histogram construction
    pub fn with_col_wise(mut self) -> Self {
        self.force_col_wise = true;
        self.force_row_wise = false;
        self
    }

    /// Enable row-wise histogram construction
    pub fn with_row_wise(mut self) -> Self {
        self.force_row_wise = true;
        self.force_col_wise = false;
        self
    }

    /// Set NUMA node affinity
    pub fn with_numa_node(mut self, node: usize) -> Self {
        self.numa_node = Some(node);
        self
    }

    /// Set thread affinity
    pub fn with_thread_affinity(mut self, affinity: Vec<usize>) -> Self {
        self.thread_affinity = Some(affinity);
        self
    }

    /// Detect CPU capabilities automatically
    pub fn detect_cpu_capabilities(&mut self) {
        // Basic CPU feature detection
        self.cpu_extensions = detect_cpu_extensions();
        
        // Set optimal number of threads if not specified
        if self.num_threads == 0 {
            self.num_threads = num_cpus::get();
        }
        
        // Set optimal memory alignment based on available extensions
        if self.cpu_extensions.avx512 {
            self.memory_alignment = 64;
        } else if self.cpu_extensions.avx || self.cpu_extensions.avx2 {
            self.memory_alignment = 32;
        } else if self.cpu_extensions.sse2 {
            self.memory_alignment = 16;
        } else {
            self.memory_alignment = 8;
        }
    }

    /// Validate device configuration
    pub fn validate(&self) -> Result<()> {
        // Validate thread count
        if self.num_threads > 0 && self.num_threads > num_cpus::get() * 4 {
            log::warn!("num_threads ({}) is much larger than available CPU cores ({})", 
                      self.num_threads, num_cpus::get());
        }

        // Validate GPU parameters
        if self.device_type == DeviceType::GPU {
            if self.gpu_platform_id < -1 {
                return Err(LightGBMError::invalid_parameter(
                    "gpu_platform_id",
                    self.gpu_platform_id.to_string(),
                    "must be >= -1",
                ));
            }

            if self.gpu_device_id < -1 {
                return Err(LightGBMError::invalid_parameter(
                    "gpu_device_id",
                    self.gpu_device_id.to_string(),
                    "must be >= -1",
                ));
            }

            if self.gpu_work_group_size == 0 || self.gpu_work_group_size > MAX_GPU_WORK_GROUP_SIZE {
                return Err(LightGBMError::invalid_parameter(
                    "gpu_work_group_size",
                    self.gpu_work_group_size.to_string(),
                    &format!("must be in range [1, {}]", MAX_GPU_WORK_GROUP_SIZE),
                ));
            }

            if self.gpu_batch_size == 0 {
                return Err(LightGBMError::invalid_parameter(
                    "gpu_batch_size",
                    self.gpu_batch_size.to_string(),
                    "must be positive",
                ));
            }

            if self.gpu_timeout_ms == 0 {
                return Err(LightGBMError::invalid_parameter(
                    "gpu_timeout_ms",
                    self.gpu_timeout_ms.to_string(),
                    "must be positive",
                ));
            }
        }

        // Validate memory alignment
        if !self.memory_alignment.is_power_of_two() {
            return Err(LightGBMError::invalid_parameter(
                "memory_alignment",
                self.memory_alignment.to_string(),
                "must be a power of 2",
            ));
        }

        // Validate conflicting histogram construction modes
        if self.force_col_wise && self.force_row_wise {
            return Err(LightGBMError::config(
                "Cannot force both column-wise and row-wise histogram construction"
            ));
        }

        // Validate thread affinity
        if let Some(ref affinity) = self.thread_affinity {
            if affinity.len() != self.num_threads {
                return Err(LightGBMError::invalid_parameter(
                    "thread_affinity",
                    format!("length {}", affinity.len()),
                    &format!("must match num_threads ({})", self.num_threads),
                ));
            }
            
            for &core in affinity {
                if core >= num_cpus::get() {
                    return Err(LightGBMError::invalid_parameter(
                        "thread_affinity",
                        format!("core {}", core),
                        &format!("must be < num_cpus ({})", num_cpus::get()),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Get the effective number of threads
    pub fn effective_num_threads(&self) -> usize {
        if self.num_threads == 0 {
            num_cpus::get()
        } else {
            self.num_threads
        }
    }

    /// Check if GPU is available and configured
    pub fn is_gpu_available(&self) -> bool {
        self.device_type == DeviceType::GPU && self.gpu_platform_id >= 0 && self.gpu_device_id >= 0
    }

    /// Get optimal histogram construction mode
    pub fn histogram_construction_mode(&self) -> HistogramConstructionMode {
        if self.force_col_wise {
            HistogramConstructionMode::ColumnWise
        } else if self.force_row_wise {
            HistogramConstructionMode::RowWise
        } else {
            // Auto-detect based on device capabilities
            if self.device_type == DeviceType::GPU {
                HistogramConstructionMode::ColumnWise
            } else if self.cpu_extensions.avx2 || self.cpu_extensions.avx512 {
                HistogramConstructionMode::ColumnWise
            } else {
                HistogramConstructionMode::RowWise
            }
        }
    }

    /// Get optimal SIMD width for current configuration
    pub fn simd_width(&self) -> usize {
        if self.cpu_extensions.avx512 {
            16  // 512 bits / 32 bits per f32
        } else if self.cpu_extensions.avx2 || self.cpu_extensions.avx {
            8   // 256 bits / 32 bits per f32
        } else if self.cpu_extensions.sse2 {
            4   // 128 bits / 32 bits per f32
        } else {
            1   // Scalar
        }
    }

    /// Get device configuration as parameter map
    pub fn as_parameter_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        
        map.insert("device_type".to_string(), self.device_type.to_string());
        map.insert("num_threads".to_string(), self.num_threads.to_string());
        map.insert("gpu_platform_id".to_string(), self.gpu_platform_id.to_string());
        map.insert("gpu_device_id".to_string(), self.gpu_device_id.to_string());
        map.insert("gpu_use_dp".to_string(), self.gpu_use_dp.to_string());
        map.insert("gpu_work_group_size".to_string(), self.gpu_work_group_size.to_string());
        map.insert("gpu_batch_size".to_string(), self.gpu_batch_size.to_string());
        map.insert("gpu_timeout_ms".to_string(), self.gpu_timeout_ms.to_string());
        map.insert("force_col_wise".to_string(), self.force_col_wise.to_string());
        map.insert("force_row_wise".to_string(), self.force_row_wise.to_string());
        map.insert("memory_alignment".to_string(), self.memory_alignment.to_string());
        map.insert("enable_prefetching".to_string(), self.enable_prefetching.to_string());
        
        if let Some(numa_node) = self.numa_node {
            map.insert("numa_node".to_string(), numa_node.to_string());
        }
        
        map
    }
}

/// Histogram construction mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HistogramConstructionMode {
    /// Column-wise histogram construction (better for wide datasets)
    ColumnWise,
    /// Row-wise histogram construction (better for tall datasets)
    RowWise,
}

/// Device capabilities detection
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// CPU information
    pub cpu_info: CpuInfo,
    /// GPU information
    pub gpu_info: Option<GpuInfo>,
    /// Memory information
    pub memory_info: MemoryInfo,
    /// Performance characteristics
    pub performance_info: PerformanceInfo,
}

/// CPU information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuInfo {
    /// Number of physical cores
    pub num_physical_cores: usize,
    /// Number of logical cores
    pub num_logical_cores: usize,
    /// CPU vendor
    pub vendor: String,
    /// CPU model name
    pub model_name: String,
    /// CPU frequency in MHz
    pub frequency_mhz: u32,
    /// Cache sizes in KB
    pub cache_sizes: CacheSizes,
    /// Supported instruction sets
    pub extensions: CpuExtensions,
}

/// CPU cache sizes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheSizes {
    /// L1 data cache size in KB
    pub l1_data: usize,
    /// L1 instruction cache size in KB
    pub l1_instruction: usize,
    /// L2 cache size in KB
    pub l2: usize,
    /// L3 cache size in KB
    pub l3: usize,
}

/// GPU information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU vendor
    pub vendor: String,
    /// GPU model name
    pub model_name: String,
    /// GPU driver version
    pub driver_version: String,
    /// GPU memory in MB
    pub memory_mb: usize,
    /// Maximum work group size
    pub max_work_group_size: usize,
    /// Maximum local memory in KB
    pub max_local_memory_kb: usize,
    /// Supports double precision
    pub supports_double_precision: bool,
    /// OpenCL version
    pub opencl_version: String,
    /// CUDA version (if available)
    pub cuda_version: Option<String>,
}

/// Memory information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// Total system memory in MB
    pub total_memory_mb: usize,
    /// Available memory in MB
    pub available_memory_mb: usize,
    /// Number of NUMA nodes
    pub num_numa_nodes: usize,
    /// Page size in bytes
    pub page_size_bytes: usize,
    /// Supports large pages
    pub supports_large_pages: bool,
}

/// Performance characteristics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceInfo {
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f64,
    /// Peak FLOPS (single precision)
    pub peak_flops_sp: f64,
    /// Peak FLOPS (double precision)
    pub peak_flops_dp: f64,
    /// Optimal histogram construction mode
    pub optimal_histogram_mode: HistogramConstructionMode,
    /// Recommended number of threads
    pub recommended_num_threads: usize,
}

/// Detect CPU instruction set extensions
fn detect_cpu_extensions() -> CpuExtensions {
    let extensions = CpuExtensions::default();
    
    // Use std::arch to detect CPU features
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        
        // Check if cpuid is available
        if std::arch::is_x86_feature_detected!("sse") {
            extensions.sse = true;
        }
        if std::arch::is_x86_feature_detected!("sse2") {
            extensions.sse2 = true;
        }
        if std::arch::is_x86_feature_detected!("sse3") {
            extensions.sse3 = true;
        }
        if std::arch::is_x86_feature_detected!("ssse3") {
            extensions.ssse3 = true;
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            extensions.sse4_1 = true;
        }
        if std::arch::is_x86_feature_detected!("sse4.2") {
            extensions.sse4_2 = true;
        }
        if std::arch::is_x86_feature_detected!("avx") {
            extensions.avx = true;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            extensions.avx2 = true;
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            extensions.avx512 = true;
        }
        if std::arch::is_x86_feature_detected!("fma") {
            extensions.fma = true;
        }
        if std::arch::is_x86_feature_detected!("bmi1") {
            extensions.bmi1 = true;
        }
        if std::arch::is_x86_feature_detected!("bmi2") {
            extensions.bmi2 = true;
        }
    }
    
    extensions
}

/// Detect system capabilities
pub fn detect_capabilities() -> DeviceCapabilities {
    let cpu_info = detect_cpu_info();
    let gpu_info = detect_gpu_info();
    let memory_info = detect_memory_info();
    let performance_info = detect_performance_info(&cpu_info, &gpu_info);
    
    DeviceCapabilities {
        cpu_info,
        gpu_info,
        memory_info,
        performance_info,
    }
}

/// Detect CPU information
fn detect_cpu_info() -> CpuInfo {
    let num_logical_cores = num_cpus::get();
    let num_physical_cores = num_cpus::get_physical();
    
    CpuInfo {
        num_physical_cores,
        num_logical_cores,
        vendor: "Unknown".to_string(),
        model_name: "Unknown".to_string(),
        frequency_mhz: 0,
        cache_sizes: CacheSizes {
            l1_data: 32,
            l1_instruction: 32,
            l2: 256,
            l3: 8192,
        },
        extensions: detect_cpu_extensions(),
    }
}

/// Detect GPU information
fn detect_gpu_info() -> Option<GpuInfo> {
    // GPU detection would require OpenCL or CUDA libraries
    // For now, return None
    None
}

/// Detect memory information
fn detect_memory_info() -> MemoryInfo {
    MemoryInfo {
        total_memory_mb: 8192,  // Placeholder
        available_memory_mb: 4096,  // Placeholder
        num_numa_nodes: 1,
        page_size_bytes: 4096,
        supports_large_pages: false,
    }
}

/// Detect performance characteristics
fn detect_performance_info(cpu_info: &CpuInfo, _gpu_info: &Option<GpuInfo>) -> PerformanceInfo {
    let optimal_histogram_mode = if cpu_info.extensions.avx2 || cpu_info.extensions.avx512 {
        HistogramConstructionMode::ColumnWise
    } else {
        HistogramConstructionMode::RowWise
    };
    
    PerformanceInfo {
        memory_bandwidth_gbps: 50.0,  // Placeholder
        peak_flops_sp: 100.0,  // Placeholder
        peak_flops_dp: 50.0,   // Placeholder
        optimal_histogram_mode,
        recommended_num_threads: cpu_info.num_logical_cores,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_config_default() {
        let config = DeviceConfig::default();
        assert_eq!(config.device_type, DeviceType::CPU);
        assert_eq!(config.num_threads, DEFAULT_NUM_THREADS);
        assert_eq!(config.gpu_platform_id, -1);
        assert_eq!(config.gpu_device_id, -1);
        assert!(!config.gpu_use_dp);
    }

    #[test]
    fn test_device_config_cpu() {
        let config = DeviceConfig::cpu();
        assert_eq!(config.device_type, DeviceType::CPU);
        assert!(config.memory_alignment.is_power_of_two());
    }

    #[test]
    fn test_device_config_gpu() {
        let config = DeviceConfig::gpu();
        assert_eq!(config.device_type, DeviceType::GPU);
        assert_eq!(config.gpu_platform_id, 0);
        assert_eq!(config.gpu_device_id, 0);
    }

    #[test]
    fn test_device_config_builders() {
        let config = DeviceConfig::cpu()
            .with_num_threads(8)
            .with_col_wise()
            .with_numa_node(0);
        
        assert_eq!(config.num_threads, 8);
        assert!(config.force_col_wise);
        assert!(!config.force_row_wise);
        assert_eq!(config.numa_node, Some(0));
    }

    #[test]
    fn test_device_config_validation() {
        let mut config = DeviceConfig::default();
        assert!(config.validate().is_ok());
        
        config.gpu_platform_id = -2;
        config.device_type = DeviceType::GPU;
        assert!(config.validate().is_err());
        
        config.gpu_platform_id = 0;
        config.gpu_work_group_size = 0;
        assert!(config.validate().is_err());
        
        config.gpu_work_group_size = 256;
        config.force_col_wise = true;
        config.force_row_wise = true;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cpu_extensions_detection() {
        let extensions = detect_cpu_extensions();
        // At least SSE2 should be available on x86_64
        #[cfg(target_arch = "x86_64")]
        assert!(extensions.sse2);
    }

    #[test]
    fn test_histogram_construction_mode() {
        let config = DeviceConfig::cpu();
        let mode = config.histogram_construction_mode();
        assert!(matches!(mode, HistogramConstructionMode::ColumnWise | HistogramConstructionMode::RowWise));
    }

    #[test]
    fn test_simd_width() {
        let config = DeviceConfig::cpu();
        let width = config.simd_width();
        assert!(width >= 1);
        assert!(width <= 16);
    }

    #[test]
    fn test_capabilities_detection() {
        let capabilities = detect_capabilities();
        assert!(capabilities.cpu_info.num_logical_cores > 0);
        assert!(capabilities.memory_info.total_memory_mb > 0);
        assert!(capabilities.performance_info.recommended_num_threads > 0);
    }

    #[test]
    fn test_device_config_parameter_map() {
        let config = DeviceConfig::default();
        let map = config.as_parameter_map();
        
        assert!(map.contains_key("device_type"));
        assert!(map.contains_key("num_threads"));
        assert!(map.contains_key("gpu_platform_id"));
        assert_eq!(map.get("device_type").unwrap(), "cpu");
    }
}