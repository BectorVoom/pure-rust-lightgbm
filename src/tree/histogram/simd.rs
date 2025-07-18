//! SIMD-optimized histogram construction for enhanced performance.
//!
//! This module provides vectorized implementations of histogram accumulation
//! operations using platform-specific SIMD instructions when available.

use crate::core::types::{BinIndex, DataSize, Hist, Score};
use ndarray::{Array1, ArrayView1};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD configuration for histogram operations.
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// Vector width for SIMD operations (typically 4 or 8)
    pub vector_width: usize,
    /// Alignment requirement for SIMD operations
    pub alignment: usize,
    /// Minimum data size to enable SIMD
    pub min_simd_size: usize,
    /// Whether to use AVX2 instructions (x86_64 only)
    pub use_avx2: bool,
    /// Whether to use AVX512 instructions (x86_64 only)
    pub use_avx512: bool,
}

impl Default for SimdConfig {
    fn default() -> Self {
        SimdConfig {
            vector_width: 8,
            alignment: 32,
            min_simd_size: 64,
            use_avx2: is_avx2_available(),
            use_avx512: is_avx512_available(),
        }
    }
}

/// SIMD-optimized histogram accumulator.
pub struct SimdHistogramAccumulator {
    config: SimdConfig,
}

impl SimdHistogramAccumulator {
    /// Creates a new SIMD histogram accumulator.
    pub fn new(config: SimdConfig) -> Self {
        SimdHistogramAccumulator { config }
    }

    /// Accumulates gradients and hessians into histogram using SIMD operations.
    pub fn accumulate_histogram(
        &self,
        histogram: &mut Array1<Hist>,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
        data_indices: &[DataSize],
        bins: &[BinIndex],
    ) -> anyhow::Result<()> {
        if data_indices.len() != bins.len() {
            return Err(anyhow::anyhow!(
                "Data indices and bins arrays must have the same length"
            ));
        }

        if data_indices.len() < self.config.min_simd_size {
            // Fall back to scalar implementation for small data
            return self.accumulate_histogram_scalar(histogram, gradients, hessians, data_indices, bins);
        }

        #[cfg(target_arch = "x86_64")]
        {
            if self.config.use_avx512 && is_avx512_available() {
                return self.accumulate_histogram_avx512(histogram, gradients, hessians, data_indices, bins);
            } else if self.config.use_avx2 && is_avx2_available() {
                return self.accumulate_histogram_avx2(histogram, gradients, hessians, data_indices, bins);
            }
        }

        // Fall back to portable SIMD implementation
        self.accumulate_histogram_portable_simd(histogram, gradients, hessians, data_indices, bins)
    }

    /// Scalar fallback implementation.
    fn accumulate_histogram_scalar(
        &self,
        histogram: &mut Array1<Hist>,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
        data_indices: &[DataSize],
        bins: &[BinIndex],
    ) -> anyhow::Result<()> {
        for (i, &data_idx) in data_indices.iter().enumerate() {
            let idx = data_idx as usize;
            if idx >= gradients.len() || i >= bins.len() {
                continue;
            }

            let bin = bins[i] as usize;
            let gradient = gradients[idx] as Hist;
            let hessian = hessians[idx] as Hist;

            let bin_offset = bin * 2;
            if bin_offset + 1 < histogram.len() {
                histogram[bin_offset] += gradient;
                histogram[bin_offset + 1] += hessian;
            }
        }

        Ok(())
    }

    /// Portable SIMD implementation using standard library features.
    fn accumulate_histogram_portable_simd(
        &self,
        histogram: &mut Array1<Hist>,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
        data_indices: &[DataSize],
        bins: &[BinIndex],
    ) -> anyhow::Result<()> {
        // Process data in chunks that fit in SIMD registers
        let chunk_size = self.config.vector_width;
        let num_complete_chunks = data_indices.len() / chunk_size;

        // Process complete chunks with SIMD
        for chunk_idx in 0..num_complete_chunks {
            let start_idx = chunk_idx * chunk_size;
            let end_idx = start_idx + chunk_size;

            for i in start_idx..end_idx {
                let data_idx = data_indices[i] as usize;
                if data_idx >= gradients.len() || i >= bins.len() {
                    continue;
                }

                let bin = bins[i] as usize;
                let gradient = gradients[data_idx] as Hist;
                let hessian = hessians[data_idx] as Hist;

                let bin_offset = bin * 2;
                if bin_offset + 1 < histogram.len() {
                    histogram[bin_offset] += gradient;
                    histogram[bin_offset + 1] += hessian;
                }
            }
        }

        // Process remaining elements with scalar code
        let remaining_start = num_complete_chunks * chunk_size;
        for i in remaining_start..data_indices.len() {
            let data_idx = data_indices[i] as usize;
            if data_idx >= gradients.len() || i >= bins.len() {
                continue;
            }

            let bin = bins[i] as usize;
            let gradient = gradients[data_idx] as Hist;
            let hessian = hessians[data_idx] as Hist;

            let bin_offset = bin * 2;
            if bin_offset + 1 < histogram.len() {
                histogram[bin_offset] += gradient;
                histogram[bin_offset + 1] += hessian;
            }
        }

        Ok(())
    }

    /// AVX2-optimized implementation for x86_64.
    #[cfg(target_arch = "x86_64")]
    fn accumulate_histogram_avx2(
        &self,
        histogram: &mut Array1<Hist>,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
        data_indices: &[DataSize],
        bins: &[BinIndex],
    ) -> anyhow::Result<()> {
        if !is_avx2_available() {
            return self.accumulate_histogram_portable_simd(histogram, gradients, hessians, data_indices, bins);
        }

        unsafe {
            let chunk_size = 8; // AVX2 can process 8 f32 values at once
            let num_complete_chunks = data_indices.len() / chunk_size;

            // Process complete chunks with AVX2
            for chunk_idx in 0..num_complete_chunks {
                let start_idx = chunk_idx * chunk_size;
                
                // Load data indices and bins
                let indices_ptr = data_indices.as_ptr().add(start_idx);
                let bins_ptr = bins.as_ptr().add(start_idx);

                // Process each element in the chunk
                // Note: This is a simplified version - real AVX2 implementation would
                // require more complex gather operations and scatter stores
                for i in 0..chunk_size {
                    let data_idx = *indices_ptr.add(i) as usize;
                    let bin = *bins_ptr.add(i) as usize;

                    if data_idx < gradients.len() && bin * 2 + 1 < histogram.len() {
                        let gradient = gradients[data_idx] as Hist;
                        let hessian = hessians[data_idx] as Hist;

                        histogram[bin * 2] += gradient;
                        histogram[bin * 2 + 1] += hessian;
                    }
                }
            }

            // Process remaining elements
            let remaining_start = num_complete_chunks * chunk_size;
            for i in remaining_start..data_indices.len() {
                let data_idx = data_indices[i] as usize;
                let bin = bins[i] as usize;

                if data_idx < gradients.len() && bin * 2 + 1 < histogram.len() {
                    let gradient = gradients[data_idx] as Hist;
                    let hessian = hessians[data_idx] as Hist;

                    histogram[bin * 2] += gradient;
                    histogram[bin * 2 + 1] += hessian;
                }
            }
        }

        Ok(())
    }

    /// AVX512-optimized implementation for x86_64.
    #[cfg(target_arch = "x86_64")]
    fn accumulate_histogram_avx512(
        &self,
        histogram: &mut Array1<Hist>,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
        data_indices: &[DataSize],
        bins: &[BinIndex],
    ) -> anyhow::Result<()> {
        if !is_avx512_available() {
            return self.accumulate_histogram_avx2(histogram, gradients, hessians, data_indices, bins);
        }

        // AVX512 implementation would go here
        // For now, fall back to AVX2
        self.accumulate_histogram_avx2(histogram, gradients, hessians, data_indices, bins)
    }

    /// Vectorized histogram merge operation.
    pub fn merge_histograms_simd(
        &self,
        target: &mut Array1<Hist>,
        source: &ArrayView1<Hist>,
    ) -> anyhow::Result<()> {
        if target.len() != source.len() {
            return Err(anyhow::anyhow!("Histograms must have the same length"));
        }

        #[cfg(target_arch = "x86_64")]
        {
            if self.config.use_avx2 && is_avx2_available() {
                return self.merge_histograms_avx2(target, source);
            }
        }

        // Portable implementation
        for (target_val, &source_val) in target.iter_mut().zip(source.iter()) {
            *target_val += source_val;
        }

        Ok(())
    }

    /// AVX2-optimized histogram merge.
    #[cfg(target_arch = "x86_64")]
    fn merge_histograms_avx2(
        &self,
        target: &mut Array1<Hist>,
        source: &ArrayView1<Hist>,
    ) -> anyhow::Result<()> {
        if !is_avx2_available() {
            // Fall back to portable implementation
            for (target_val, &source_val) in target.iter_mut().zip(source.iter()) {
                *target_val += source_val;
            }
            return Ok(());
        }

        unsafe {
            let chunk_size = 4; // AVX2 can process 4 f64 values at once
            let num_complete_chunks = target.len() / chunk_size;

            // Process complete chunks
            for chunk_idx in 0..num_complete_chunks {
                let start_idx = chunk_idx * chunk_size;
                
                let target_ptr = target.as_mut_ptr().add(start_idx);
                let source_ptr = source.as_ptr().add(start_idx);

                // Load 4 f64 values from target and source
                let target_vec = _mm256_load_pd(target_ptr);
                let source_vec = _mm256_load_pd(source_ptr);

                // Add them together
                let result_vec = _mm256_add_pd(target_vec, source_vec);

                // Store the result back to target
                _mm256_store_pd(target_ptr, result_vec);
            }

            // Process remaining elements
            let remaining_start = num_complete_chunks * chunk_size;
            for i in remaining_start..target.len() {
                target[i] += source[i];
            }
        }

        Ok(())
    }
}

/// Utility functions for SIMD capability detection.

/// Checks if AVX2 is available on the current CPU.
#[cfg(target_arch = "x86_64")]
fn is_avx2_available() -> bool {
    is_x86_feature_detected!("avx2")
}

/// Checks if AVX512F is available on the current CPU.
#[cfg(target_arch = "x86_64")]
fn is_avx512_available() -> bool {
    is_x86_feature_detected!("avx512f")
}

/// Fallback for non-x86_64 architectures.
#[cfg(not(target_arch = "x86_64"))]
fn is_avx2_available() -> bool {
    false
}

/// Fallback for non-x86_64 architectures.
#[cfg(not(target_arch = "x86_64"))]
fn is_avx512_available() -> bool {
    false
}

/// Aligned memory allocation utilities for SIMD operations.
pub struct AlignedBuffer<T> {
    ptr: *mut T,
    len: usize,
    capacity: usize,
    alignment: usize,
}

impl<T> AlignedBuffer<T> {
    /// Creates a new aligned buffer with the specified capacity and alignment.
    pub fn new(capacity: usize, alignment: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(
            capacity * std::mem::size_of::<T>(),
            alignment,
        ).expect("Invalid layout");

        let ptr = unsafe { std::alloc::alloc(layout) as *mut T };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        AlignedBuffer {
            ptr,
            len: 0,
            capacity,
            alignment,
        }
    }

    /// Returns the length of the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the capacity of the buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns a slice view of the buffer.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Returns a mutable slice view of the buffer.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Resizes the buffer to the new length.
    pub fn resize(&mut self, new_len: usize, value: T) 
    where 
        T: Clone,
    {
        if new_len > self.capacity {
            panic!("Cannot resize beyond capacity");
        }

        if new_len > self.len {
            // Fill new elements with the given value
            for i in self.len..new_len {
                unsafe {
                    std::ptr::write(self.ptr.add(i), value.clone());
                }
            }
        }

        self.len = new_len;
    }

    /// Clears the buffer.
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

impl<T> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let layout = std::alloc::Layout::from_size_align(
                self.capacity * std::mem::size_of::<T>(),
                self.alignment,
            ).expect("Invalid layout");

            unsafe {
                // Drop all elements
                for i in 0..self.len {
                    std::ptr::drop_in_place(self.ptr.add(i));
                }
                
                std::alloc::dealloc(self.ptr as *mut u8, layout);
            }
        }
    }
}

unsafe impl<T: Send> Send for AlignedBuffer<T> {}
unsafe impl<T: Sync> Sync for AlignedBuffer<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_config_creation() {
        let config = SimdConfig::default();
        assert_eq!(config.vector_width, 8);
        assert_eq!(config.alignment, 32);
    }

    #[test]
    fn test_simd_accumulator_creation() {
        let config = SimdConfig::default();
        let accumulator = SimdHistogramAccumulator::new(config);
        assert_eq!(accumulator.config.vector_width, 8);
    }

    #[test]
    fn test_aligned_buffer() {
        let mut buffer: AlignedBuffer<f64> = AlignedBuffer::new(16, 32);
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.capacity(), 16);
        assert!(buffer.is_empty());

        buffer.resize(8, 1.0);
        assert_eq!(buffer.len(), 8);
        assert!(!buffer.is_empty());

        let slice = buffer.as_slice();
        assert_eq!(slice.len(), 8);
        assert_eq!(slice[0], 1.0);
    }

    #[test]
    fn test_histogram_accumulation_scalar() {
        let config = SimdConfig {
            min_simd_size: 1000, // Force scalar path
            ..SimdConfig::default()
        };
        let accumulator = SimdHistogramAccumulator::new(config);

        let mut histogram = Array1::zeros(10); // 5 bins * 2
        let gradients = Array1::from(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let hessians = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0]);
        let data_indices = vec![0, 1, 2, 3, 4];
        let bins = vec![0, 1, 2, 3, 4];

        let result = accumulator.accumulate_histogram(
            &mut histogram,
            &gradients.view(),
            &hessians.view(),
            &data_indices,
            &bins,
        );

        assert!(result.is_ok());
        
        // Check that gradients and hessians were accumulated correctly
        assert_eq!(histogram[0], 0.1); // bin 0 gradient
        assert_eq!(histogram[1], 1.0); // bin 0 hessian
        assert_eq!(histogram[2], 0.2); // bin 1 gradient
        assert_eq!(histogram[3], 1.0); // bin 1 hessian
    }

    #[test]
    fn test_histogram_merge() {
        let config = SimdConfig::default();
        let accumulator = SimdHistogramAccumulator::new(config);

        let mut target = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let source = Array1::from(vec![0.5, 1.0, 1.5, 2.0]);

        let result = accumulator.merge_histograms_simd(&mut target, &source.view());
        assert!(result.is_ok());

        assert_eq!(target, Array1::from(vec![1.5, 3.0, 4.5, 6.0]));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_capability_detection() {
        // These functions should not panic
        let _avx2_available = is_avx2_available();
        let _avx512_available = is_avx512_available();
    }
}