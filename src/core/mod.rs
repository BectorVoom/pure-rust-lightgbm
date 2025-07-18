//! Core infrastructure module for Pure Rust LightGBM.
//!
//! This module provides the foundational components that form the backbone
//! of the entire LightGBM implementation, including fundamental data types,
//! error handling, memory management, and core trait abstractions.
//!
//! # Organization
//!
//! The core module is organized into several key components:
//!
//! - [`types`]: Fundamental data types and enumerations
//! - [`constants`]: System constants and configuration defaults
//! - [`error`]: Comprehensive error handling and error types
//! - [`memory`]: Aligned memory allocation and management utilities
//! - [`traits`]: Core trait abstractions for polymorphism and extensibility
//!
//! # Usage
//!
//! Most users will interact with the core module through the main library interface,
//! but these components can also be used directly for advanced use cases:
//!
//! ```rust
//! use lightgbm_rust::core::{
//!     types::{DataSize, Score, Label, Hist, ObjectiveType},
//!     constants::DEFAULT_LEARNING_RATE,
//!     error::{Result, LightGBMError},
//!     memory::AlignedBuffer,
//!     traits::ObjectiveFunction,
//! };
//! 
//! // Create an aligned buffer for SIMD operations
//! let mut buffer: AlignedBuffer<f32> = AlignedBuffer::new(1000)?;
//! 
//! // Work with fundamental types
//! let learning_rate = DEFAULT_LEARNING_RATE;
//! let objective = ObjectiveType::Regression;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// Public module declarations
pub mod constants;
pub mod error;
pub mod memory;
pub mod traits;
pub mod types;

// Re-export commonly used items for convenience
pub use constants::*;
pub use error::{LightGBMError, Result};
pub use memory::{AlignedBuffer, MemoryPool, MemoryStats};
pub use traits::*;
pub use types::*;

/// Version information for the core module
pub const CORE_MODULE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Core module features and capabilities
#[derive(Debug, Clone)]
pub struct CoreCapabilities {
    /// SIMD alignment support
    pub simd_aligned_memory: bool,
    /// Thread-safe memory management
    pub thread_safe_memory: bool,
    /// Comprehensive error handling
    pub rich_error_types: bool,
    /// Trait-based polymorphism
    pub trait_abstractions: bool,
    /// Serialization support
    pub serialization: bool,
    /// GPU acceleration ready
    pub gpu_ready: bool,
}

impl Default for CoreCapabilities {
    fn default() -> Self {
        CoreCapabilities {
            simd_aligned_memory: true,
            thread_safe_memory: true,
            rich_error_types: true,
            trait_abstractions: true,
            serialization: true,
            gpu_ready: cfg!(feature = "gpu"),
        }
    }
}

impl CoreCapabilities {
    /// Get current core capabilities
    pub fn current() -> Self {
        Self::default()
    }

    /// Check if all capabilities are available
    pub fn all_available(&self) -> bool {
        self.simd_aligned_memory
            && self.thread_safe_memory
            && self.rich_error_types
            && self.trait_abstractions
            && self.serialization
    }

    /// Get a summary of available capabilities
    pub fn summary(&self) -> String {
        let mut features = Vec::new();
        
        if self.simd_aligned_memory {
            features.push("SIMD Memory");
        }
        if self.thread_safe_memory {
            features.push("Thread Safety");
        }
        if self.rich_error_types {
            features.push("Error Handling");
        }
        if self.trait_abstractions {
            features.push("Traits");
        }
        if self.serialization {
            features.push("Serialization");
        }
        if self.gpu_ready {
            features.push("GPU Ready");
        }

        format!("Core capabilities: {}", features.join(", "))
    }
}

/// Core module initialization and configuration
pub struct CoreModule {
    capabilities: CoreCapabilities,
    initialized: bool,
}

impl CoreModule {
    /// Create a new core module instance
    pub fn new() -> Self {
        CoreModule {
            capabilities: CoreCapabilities::current(),
            initialized: false,
        }
    }

    /// Initialize the core module
    pub fn initialize(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        // Verify memory alignment capabilities
        if !self.verify_memory_alignment() {
            return Err(LightGBMError::internal(
                "SIMD memory alignment verification failed"
            ));
        }

        // Initialize logging if not already done
        self.initialize_logging();

        // Verify thread safety
        if !self.verify_thread_safety() {
            return Err(LightGBMError::internal(
                "Thread safety verification failed"
            ));
        }

        self.initialized = true;
        log::info!("Core module initialized successfully");
        log::debug!("{}", self.capabilities.summary());

        Ok(())
    }

    /// Check if the core module is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get core capabilities
    pub fn capabilities(&self) -> &CoreCapabilities {
        &self.capabilities
    }

    /// Verify memory alignment functionality
    fn verify_memory_alignment(&self) -> bool {
        // Test aligned buffer creation
        if let Ok(buffer) = AlignedBuffer::<f32>::new(100) {
            buffer.is_aligned()
        } else {
            false
        }
    }

    /// Initialize logging subsystem
    fn initialize_logging(&self) {
        // Only initialize if not already done
        if std::env::var("RUST_LOG").is_err() {
            std::env::set_var("RUST_LOG", "info");
        }
        
        // Try to initialize env_logger, ignore if already initialized
        let _ = env_logger::try_init();
    }

    /// Verify thread safety functionality
    fn verify_thread_safety(&self) -> bool {
        // Basic thread safety verification
        use std::sync::{Arc, Mutex};
        use std::thread;

        let counter = Arc::new(Mutex::new(0));
        let mut handles = vec![];

        // Spawn a few threads to test basic thread safety
        for _ in 0..4 {
            let counter = Arc::clone(&counter);
            let handle = thread::spawn(move || {
                if let Ok(mut num) = counter.lock() {
                    *num += 1;
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            if handle.join().is_err() {
                return false;
            }
        }

        // Check final value
        let result = match counter.lock() {
            Ok(final_count) => *final_count == 4,
            Err(_) => false,
        };
        result
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let allocated = memory::total_allocated_memory();
        MemoryStats {
            allocated_bytes: allocated,
            used_bytes: allocated, // Simplified for now
            alignment: ALIGNED_SIZE,
            capacity_elements: 0,
            length_elements: 0,
            free_bytes: 0,
            num_allocations: 1, // Simplified for now
        }
    }

    /// Perform core module self-test
    pub fn self_test(&self) -> Result<()> {
        log::info!("Running core module self-test...");

        // Test memory allocation
        let _buffer: AlignedBuffer<f64> = AlignedBuffer::new(1000)?;
        log::debug!("Memory allocation test passed");

        // Test error handling
        let _error = LightGBMError::config("test error");
        log::debug!("Error handling test passed");

        // Test type system
        let _data_size: DataSize = 42;
        let _score: Score = 3.14;
        let _label: Label = 1.0;
        let _hist: Hist = 2.718;
        log::debug!("Type system test passed");

        // Test constants
        assert!(ALIGNED_SIZE > 0);
        assert!(DEFAULT_LEARNING_RATE > 0.0);
        log::debug!("Constants test passed");

        log::info!("Core module self-test completed successfully");
        Ok(())
    }
}

impl Default for CoreModule {
    fn default() -> Self {
        Self::new()
    }
}

/// Global core module instance
static mut CORE_MODULE: Option<CoreModule> = None;
static CORE_INIT: std::sync::Once = std::sync::Once::new();

/// Initialize the global core module
pub fn initialize_core() -> Result<()> {
    CORE_INIT.call_once(|| {
        unsafe {
            CORE_MODULE = Some(CoreModule::new());
        }
    });

    unsafe {
        if let Some(ref mut core) = CORE_MODULE {
            core.initialize()
        } else {
            Err(LightGBMError::internal("Failed to initialize core module"))
        }
    }
}

/// Check if the core module is initialized
pub fn is_core_initialized() -> bool {
    unsafe {
        CORE_MODULE.as_ref()
            .map(|core| core.is_initialized())
            .unwrap_or(false)
    }
}

/// Get current core capabilities
pub fn core_capabilities() -> CoreCapabilities {
    unsafe {
        CORE_MODULE.as_ref()
            .map(|core| core.capabilities().clone())
            .unwrap_or_default()
    }
}

/// Utility macro for ensuring core module is initialized
#[macro_export]
macro_rules! ensure_core_initialized {
    () => {
        if !$crate::core::is_core_initialized() {
            $crate::core::initialize_core()?;
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_module_creation() {
        let core = CoreModule::new();
        assert!(!core.is_initialized());
        assert!(core.capabilities().all_available() || !core.capabilities().gpu_ready);
    }

    #[test]
    fn test_core_module_initialization() {
        let mut core = CoreModule::new();
        assert!(core.initialize().is_ok());
        assert!(core.is_initialized());
    }

    #[test]
    fn test_core_capabilities() {
        let caps = CoreCapabilities::current();
        assert!(caps.simd_aligned_memory);
        assert!(caps.thread_safe_memory);
        assert!(caps.rich_error_types);
        assert!(caps.trait_abstractions);
        assert!(caps.serialization);
        
        let summary = caps.summary();
        assert!(summary.contains("Core capabilities"));
    }

    #[test]
    fn test_global_initialization() {
        // Test global initialization
        assert!(initialize_core().is_ok());
        assert!(is_core_initialized());
        
        let caps = core_capabilities();
        assert!(caps.simd_aligned_memory);
    }

    #[test]
    fn test_version_constants() {
        assert!(!CORE_MODULE_VERSION.is_empty());
    }

    #[test]
    fn test_memory_alignment_verification() {
        let core = CoreModule::new();
        assert!(core.verify_memory_alignment());
    }

    #[test]
    fn test_thread_safety_verification() {
        let core = CoreModule::new();
        assert!(core.verify_thread_safety());
    }

    #[test]
    fn test_self_test() {
        let mut core = CoreModule::new();
        core.initialize().unwrap();
        assert!(core.self_test().is_ok());
    }

    #[test]
    fn test_module_reexports() {
        // Test that key items are properly re-exported
        let _error: LightGBMError = LightGBMError::config("test");
        let _buffer: AlignedBuffer<i32> = AlignedBuffer::new(10).unwrap();
        let _size: DataSize = 42;
        let _score: Score = 3.14;
        let _constant = DEFAULT_LEARNING_RATE;
    }
}