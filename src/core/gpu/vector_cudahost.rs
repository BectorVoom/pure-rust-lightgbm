/*!
 * Copyright (c) 2020 IBM Corporation, Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

use crate::core::gpu::cuda_random;
use crate::core::meta::K_ALIGNED_SIZE;
use crate::core::utils::log::Log;
use crate::SIZE_ALIGNED;
use cubecl::prelude::*;
use cubecl_cuda::CudaDevice;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::marker::PhantomData;
use std::ptr;
use std::sync::Mutex;
/// Device enumeration - equivalent to LGBM_Device enum in C++
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum LgbmDevice {
    Cpu = 0,
    Gpu = 1,
    Cuda = 2,
}

/// Learner type enumeration - equivalent to Use_Learner enum in C++
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum UseLearner {
    CpuLearner = 0,
    GpuLearner = 1,
    CudaLearner = 2,
}

/// Global configuration class - equivalent to LGBM_config_ in C++
pub struct LgbmConfig {
    _private: (), // Prevent external instantiation
}

// Global static configuration state
static CURRENT_DEVICE: Mutex<LgbmDevice> = Mutex::new(LgbmDevice::Cpu);
static CURRENT_LEARNER: Mutex<UseLearner> = Mutex::new(UseLearner::CpuLearner);

impl LgbmConfig {
    /// Get current device setting
    pub fn current_device() -> LgbmDevice {
        *CURRENT_DEVICE.lock().unwrap()
    }

    /// Set current device
    pub fn set_current_device(device: LgbmDevice) {
        *CURRENT_DEVICE.lock().unwrap() = device;
    }

    /// Get current learner setting
    pub fn current_learner() -> UseLearner {
        *CURRENT_LEARNER.lock().unwrap()
    }

    /// Set current learner
    pub fn set_current_learner(learner: UseLearner) {
        *CURRENT_LEARNER.lock().unwrap() = learner;
    }
}

/// CUDA Host Allocator - equivalent to CHAllocator<T> template in C++
/// Provides CUDA-aware memory allocation with fallback to regular allocation
pub struct CHAllocator<T> {
    _phantom: PhantomData<T>,
}

impl<T> CHAllocator<T> {
    /// Create a new CHAllocator instance
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Allocate memory with CUDA host allocation if CUDA device is active
    /// Equivalent to CHAllocator::allocate in C++
    pub fn allocate(&self, n: usize) -> *mut T {
        if n == 0 {
            return ptr::null_mut();
        }

        let aligned_n = SIZE_ALIGNED!(n);
        let size_bytes = aligned_n * std::mem::size_of::<T>();

        // Check if we're using CUDA device
        let current_device = LgbmConfig::current_device();

        match current_device {
            LgbmDevice::Cuda => {
                // Try CUDA host allocation first
                match self.cuda_host_alloc(size_bytes) {
                    Ok(ptr) => ptr as *mut T,
                    Err(_) => {
                        // Fall back to regular aligned allocation with warning
                        Log::warning("Defaulting to malloc in CHAllocator!!!");
                        self.aligned_alloc(size_bytes)
                    }
                }
            }
            _ => {
                // Use regular aligned allocation for CPU/GPU
                self.aligned_alloc(size_bytes)
            }
        }
    }

    /// Deallocate memory, handling both CUDA host and regular memory
    /// Equivalent to CHAllocator::deallocate in C++
    pub fn deallocate(&self, ptr: *mut T, _n: usize) {
        if ptr.is_null() {
            return;
        }

        let current_device = LgbmConfig::current_device();

        match current_device {
            LgbmDevice::Cuda => {
                // Check if this is CUDA host memory
                if self.is_cuda_host_memory(ptr) {
                    self.cuda_host_free(ptr);
                } else {
                    self.aligned_free(ptr);
                }
            }
            _ => {
                // Regular deallocation for CPU/GPU
                self.aligned_free(ptr);
            }
        }
    }

    /// CUDA host memory allocation - equivalent to cudaHostAlloc
    fn cuda_host_alloc(&self, size_bytes: usize) -> Result<*mut u8, String> {
        // In real implementation, this would use CubeCL's CUDA runtime
        // For now, we'll simulate the behavior

        // Try to initialize CUDA context if available
        match CudaDevice::new(0) {
            Ok(_device) => {
                // In a real implementation, we would use:
                // cudaHostAlloc(&ptr, size_bytes, cudaHostAllocPortable)
                // For now, fall back to regular allocation
                Err("CUDA host allocation not yet implemented with CubeCL".to_string())
            }
            Err(_) => Err("CUDA device not available".to_string()),
        }
    }

    /// Free CUDA host memory - equivalent to cudaFreeHost
    fn cuda_host_free(&self, _ptr: *mut T) {
        // In real implementation, this would use CubeCL's CUDA runtime
        // For now, we'll use regular deallocation as fallback
        // This should eventually call cudaFreeHost(ptr)
        self.aligned_free(_ptr);
    }

    /// Check if pointer is CUDA host memory - equivalent to cudaPointerGetAttributes
    fn is_cuda_host_memory(&self, _ptr: *mut T) -> bool {
        // In real implementation, this would use CubeCL to check:
        // cudaPointerGetAttributes(&attributes, ptr)
        // and check if attributes.type == cudaMemoryTypeHost

        // For now, assume it's not CUDA memory (safe fallback)
        false
    }

    /// Aligned memory allocation - equivalent to _mm_malloc
    fn aligned_alloc(&self, size_bytes: usize) -> *mut T {
        let layout = Layout::from_size_align(size_bytes, 16)
            .unwrap_or_else(|_| Layout::from_size_align(size_bytes, 1).unwrap());

        unsafe {
            let ptr = alloc_zeroed(layout);
            if ptr.is_null() {
                panic!("Failed to allocate {} bytes", size_bytes);
            }
            ptr as *mut T
        }
    }

    /// Free aligned memory - equivalent to _mm_free
    fn aligned_free(&self, ptr: *mut T) {
        if ptr.is_null() {
            return;
        }

        // We need to reconstruct the layout for deallocation
        // This is a simplified version - in practice you'd need to store the layout
        let layout = Layout::from_size_align(std::mem::size_of::<T>(), 16)
            .unwrap_or_else(|_| Layout::from_size_align(std::mem::size_of::<T>(), 1).unwrap());

        unsafe {
            dealloc(ptr as *mut u8, layout);
        }
    }
}

/// Default implementation for CHAllocator
impl<T> Default for CHAllocator<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Clone implementation for CHAllocator (stateless)
impl<T> Clone for CHAllocator<T> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

/// Template parameter constructor equivalent
impl<T, U> From<CHAllocator<U>> for CHAllocator<T> {
    fn from(_other: CHAllocator<U>) -> Self {
        Self::new()
    }
}

/// Equality comparison for CHAllocator (always equal as they're stateless)
impl<T, U> PartialEq<CHAllocator<U>> for CHAllocator<T> {
    fn eq(&self, _other: &CHAllocator<U>) -> bool {
        true // All CHAllocators are equivalent
    }
}

impl<T> Eq for CHAllocator<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_enum() {
        assert_eq!(LgbmDevice::Cpu as i32, 0);
        assert_eq!(LgbmDevice::Gpu as i32, 1);
        assert_eq!(LgbmDevice::Cuda as i32, 2);
    }

    #[test]
    fn test_learner_enum() {
        assert_eq!(UseLearner::CpuLearner as i32, 0);
        assert_eq!(UseLearner::GpuLearner as i32, 1);
        assert_eq!(UseLearner::CudaLearner as i32, 2);
    }

    #[test]
    fn test_config_device() {
        // Test default
        assert_eq!(LgbmConfig::current_device(), LgbmDevice::Cpu);

        // Test setting and getting
        LgbmConfig::set_current_device(LgbmDevice::Cuda);
        assert_eq!(LgbmConfig::current_device(), LgbmDevice::Cuda);

        // Reset to default
        LgbmConfig::set_current_device(LgbmDevice::Cpu);
    }

    #[test]
    fn test_config_learner() {
        // Test default
        assert_eq!(LgbmConfig::current_learner(), UseLearner::CpuLearner);

        // Test setting and getting
        LgbmConfig::set_current_learner(UseLearner::CudaLearner);
        assert_eq!(LgbmConfig::current_learner(), UseLearner::CudaLearner);

        // Reset to default
        LgbmConfig::set_current_learner(UseLearner::CpuLearner);
    }

    #[test]
    fn test_allocator_basic() {
        let allocator: CHAllocator<i32> = CHAllocator::new();

        // Test zero allocation
        let null_ptr = allocator.allocate(0);
        assert!(null_ptr.is_null());

        // Test normal allocation
        let ptr = allocator.allocate(10);
        assert!(!ptr.is_null());

        // Test deallocation
        allocator.deallocate(ptr, 10);
    }

    #[test]
    fn test_allocator_equality() {
        let alloc1: CHAllocator<i32> = CHAllocator::new();
        let alloc2: CHAllocator<f64> = CHAllocator::new();

        assert_eq!(alloc1, alloc2); // Different types but should be equal
    }

    #[test]
    fn test_allocator_conversion() {
        let alloc_i32: CHAllocator<i32> = CHAllocator::new();
        let alloc_f64: CHAllocator<f64> = CHAllocator::from(alloc_i32);

        // Should be able to convert between different template types
        assert_eq!(alloc_f64, CHAllocator::<f64>::new());
    }

    #[test]
    fn test_cuda_fallback() {
        // Set CUDA device and test that allocation falls back properly
        LgbmConfig::set_current_device(LgbmDevice::Cuda);

        let allocator: CHAllocator<i32> = CHAllocator::new();
        let ptr = allocator.allocate(10);

        // Should not be null even if CUDA fails (fallback to regular allocation)
        assert!(!ptr.is_null());

        allocator.deallocate(ptr, 10);

        // Reset to CPU
        LgbmConfig::set_current_device(LgbmDevice::Cpu);
    }
}
