/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

//! # LightGBM Meta Types and Utilities
//!
//! This module provides the core type definitions, constants, and utility functions
//! that form the foundation of the LightGBM framework. It includes:
//!
//! - Basic data types (`DataSizeT`, `ScoreT`, `LabelT`, `CommSizeT`)
//! - Mathematical constants and limits
//! - Memory alignment utilities
//! - SIMD-optimized operations for high-performance computing
//! - Function type aliases for callbacks and distributed computing
//!
//! ## Feature Flags
//!
//! - `ScoreT_use_double`: Use f64 instead of f32 for score calculations
//! - `LabelT_use_double`: Use f64 instead of f32 for label storage
//!
//! ## Architecture Support
//!
//! The module provides platform-specific optimizations including:
//! - x86/x86_64 SIMD instructions and prefetch hints
//! - 32-byte aligned memory allocation for optimal cache performance
//! - Vectorized operations using Rust's portable SIMD

use std::collections::HashMap;
use std::simd::num::SimdFloat;
use std::simd::{f32x8, i32x8};

/// Type of data size, it is better to use signed type
pub type DataSizeT = i32;

/// Type of communication size for distributed computing
pub type CommSizeT = i32;

/// Type of score and gradients - f32 by default, f64 with feature flag
#[cfg(feature = "score_t_use_double")]
pub type ScoreT = f64;
/// Type of score and gradients - f32 by default, f64 with feature flag
#[cfg(not(feature = "score_t_use_double"))]
pub type ScoreT = f32;

/// Type of metadata including weights and labels - f32 by default, f64 with feature flag
#[cfg(feature = "label_t_use_double")]
pub type LabelT = f64;
/// Type of metadata including weights and labels - f32 by default, f64 with feature flag
#[cfg(not(feature = "label_t_use_double"))]
pub type LabelT = f32;

/// SIMD vector type for score operations using 8-lane vectors
#[cfg(feature = "score_t_use_double")]
pub type ScoreSimdT = f64x8;
/// SIMD vector type for score operations using 8-lane vectors
#[cfg(not(feature = "score_t_use_double"))]
pub type ScoreSimdT = f32x8;

/// SIMD vector type for label operations using 8-lane vectors
#[cfg(feature = "label_t_use_double")]
pub type LabelSimdT = f64x8;
/// SIMD vector type for label operations using 8-lane vectors
#[cfg(not(feature = "label_t_use_double"))]
pub type LabelSimdT = f32x8;

/// SIMD vector type for data size operations using 8-lane vectors
pub type DataSizeSimdT = i32x8;

/// Constants matching C++ implementation

/// Minimum possible score value (negative infinity)
pub const K_MIN_SCORE: ScoreT = ScoreT::NEG_INFINITY;
/// Maximum possible score value (positive infinity)
pub const K_MAX_SCORE: ScoreT = ScoreT::INFINITY;
/// Small epsilon value for numerical comparisons
pub const K_EPSILON: ScoreT = 1e-15;
/// Threshold for determining zero values in numerical computations
pub const K_ZERO_THRESHOLD: f64 = 1e-35;
/// Constant indicating no specific value is set
pub const NO_SPECIFIC: i32 = -1;
/// Memory alignment size for optimal cache performance (32 bytes)
pub const K_ALIGNED_SIZE: usize = 32;

/// SIZE_ALIGNED macro equivalent
#[macro_export]
macro_rules! SIZE_ALIGNED {
    ($t:expr) => {
        (($t) + K_ALIGNED_SIZE - 1) / K_ALIGNED_SIZE * K_ALIGNED_SIZE
    };
}

/// Cache prefetch macro for performance optimization on x86_64 architecture.
/// Prefetches data into the L1 cache for faster access.
#[cfg(target_arch = "x86_64")]
#[macro_export]
macro_rules! PREFETCH_T0 {
    ($addr:expr) => {
        unsafe {
            use std::arch::x86_64::_mm_prefetch;
            use std::arch::x86_64::_MM_HINT_T0;
            _mm_prefetch($addr as *const i8, _MM_HINT_T0);
        }
    };
}

#[cfg(target_arch = "x86")]
#[macro_export]
macro_rules! PREFETCH_T0 {
    ($addr:expr) => {
        unsafe {
            use std::arch::x86::_mm_prefetch;
            use std::arch::x86::_MM_HINT_T0;
            _mm_prefetch($addr as *const i8, _MM_HINT_T0);
        }
    };
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[macro_export]
/// Cache prefetch macro for non-x86 architectures.
/// This is a no-op implementation for architectures that don't support prefetch instructions.
macro_rules! PREFETCH_T0 {
    ($addr:expr) => {
        // No-op for non-x86 architectures
    };
}

/// Function type aliases matching C++ implementation

/// Function type for dense prediction operations. Takes sparse feature vector and outputs dense predictions.
pub type PredictFunction = Box<dyn Fn(&Vec<(i32, f64)>, &mut [f64]) + Send + Sync>;

/// Function type for sparse prediction operations. Takes sparse features and outputs sparse predictions.
pub type PredictSparseFunction =
    Box<dyn Fn(&Vec<(i32, f64)>, &mut Vec<HashMap<i32, f64>>) + Send + Sync>;

/// Function type for reduction operations in distributed computing.
pub type ReduceFunction = Box<dyn Fn(&[u8], &mut [u8], usize, CommSizeT) + Send + Sync>;

/// Function type for reduce-scatter operations in distributed computing.
/// Combines reduction and scattering of data across multiple processes.
pub type ReduceScatterFunction = Box<
    dyn Fn(
            &mut [u8],
            CommSizeT,
            usize,
            &[CommSizeT],
            &[CommSizeT],
            usize,
            &mut [u8],
            CommSizeT,
            &ReduceFunction,
        ) + Send
        + Sync,
>;

/// Function type for all-gather operations in distributed computing.
/// Gathers data from all processes and distributes the complete result.
pub type AllgatherFunction = Box<
    dyn Fn(&mut [u8], CommSizeT, &[CommSizeT], &[CommSizeT], usize, &mut [u8], CommSizeT)
        + Send
        + Sync,
>;

/// SIMD-optimized utility functions for high-performance numerical operations.
///
/// This module provides vectorized implementations of common mathematical operations
/// using Rust's portable SIMD. Functions automatically handle both complete SIMD
/// chunks and remaining scalar elements for arrays of any size.
pub mod simd_ops {
    use super::*;

    /// Loads aligned data into SIMD vectors for batch processing.
    /// Converts a slice of scores into a vector of SIMD lanes for vectorized operations.
    pub fn load_aligned_scores(data: &[ScoreT]) -> Vec<ScoreSimdT> {
        let chunk_size = ScoreSimdT::LEN;
        data.chunks_exact(chunk_size)
            .map(|chunk| ScoreSimdT::from_slice(chunk))
            .collect()
    }

    /// Stores SIMD vectors back to aligned memory.
    /// Converts SIMD vectors back to a contiguous slice of scores.
    pub fn store_aligned_scores(vectors: &[ScoreSimdT], output: &mut [ScoreT]) {
        let chunk_size = ScoreSimdT::LEN;
        for (i, vector) in vectors.iter().enumerate() {
            let start = i * chunk_size;
            let end = start + chunk_size;
            if end <= output.len() {
                vector.copy_to_slice(&mut output[start..end]);
            }
        }
    }

    /// SIMD-accelerated element-wise vector addition.
    /// Computes result[i] = a[i] + b[i] for all elements using vectorized operations.
    pub fn simd_add_scores(a: &[ScoreT], b: &[ScoreT], result: &mut [ScoreT]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        let chunk_size = ScoreSimdT::LEN;
        let chunks = a.len() / chunk_size;

        // Process full SIMD chunks
        for i in 0..chunks {
            let start = i * chunk_size;
            let end = start + chunk_size;

            let va = ScoreSimdT::from_slice(&a[start..end]);
            let vb = ScoreSimdT::from_slice(&b[start..end]);
            let vr = va + vb;

            vr.copy_to_slice(&mut result[start..end]);
        }

        // Handle remaining elements
        let remaining_start = chunks * chunk_size;
        for i in remaining_start..a.len() {
            result[i] = a[i] + b[i];
        }
    }

    /// SIMD-accelerated element-wise vector subtraction.
    /// Computes result[i] = a[i] - b[i] for all elements using vectorized operations.
    pub fn simd_sub_scores(a: &[ScoreT], b: &[ScoreT], result: &mut [ScoreT]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        let chunk_size = ScoreSimdT::LEN;
        let chunks = a.len() / chunk_size;

        // Process full SIMD chunks
        for i in 0..chunks {
            let start = i * chunk_size;
            let end = start + chunk_size;

            let va = ScoreSimdT::from_slice(&a[start..end]);
            let vb = ScoreSimdT::from_slice(&b[start..end]);
            let vr = va - vb;

            vr.copy_to_slice(&mut result[start..end]);
        }

        // Handle remaining elements
        let remaining_start = chunks * chunk_size;
        for i in remaining_start..a.len() {
            result[i] = a[i] - b[i];
        }
    }

    /// SIMD-accelerated element-wise vector multiplication.
    /// Computes result[i] = a[i] * b[i] for all elements using vectorized operations.
    pub fn simd_mul_scores(a: &[ScoreT], b: &[ScoreT], result: &mut [ScoreT]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        let chunk_size = ScoreSimdT::LEN;
        let chunks = a.len() / chunk_size;

        // Process full SIMD chunks
        for i in 0..chunks {
            let start = i * chunk_size;
            let end = start + chunk_size;

            let va = ScoreSimdT::from_slice(&a[start..end]);
            let vb = ScoreSimdT::from_slice(&b[start..end]);
            let vr = va * vb;

            vr.copy_to_slice(&mut result[start..end]);
        }

        // Handle remaining elements
        let remaining_start = chunks * chunk_size;
        for i in remaining_start..a.len() {
            result[i] = a[i] * b[i];
        }
    }

    /// SIMD-accelerated dot product computation.
    /// Computes the sum of a[i] * b[i] for all elements using vectorized operations.
    pub fn simd_dot_product(a: &[ScoreT], b: &[ScoreT]) -> ScoreT {
        assert_eq!(a.len(), b.len());

        let chunk_size = ScoreSimdT::LEN;
        let chunks = a.len() / chunk_size;
        let mut sum_vector = ScoreSimdT::splat(0.0);

        // Process full SIMD chunks
        for i in 0..chunks {
            let start = i * chunk_size;
            let end = start + chunk_size;

            let va = ScoreSimdT::from_slice(&a[start..end]);
            let vb = ScoreSimdT::from_slice(&b[start..end]);
            sum_vector += va * vb;
        }

        // Sum the SIMD vector elements
        let mut result = sum_vector.reduce_sum();

        // Handle remaining elements
        let remaining_start = chunks * chunk_size;
        for i in remaining_start..a.len() {
            result += a[i] * b[i];
        }

        result
    }

    /// SIMD-accelerated vector sum computation.
    /// Computes the sum of all elements in the vector using vectorized operations.
    pub fn simd_sum(data: &[ScoreT]) -> ScoreT {
        let chunk_size = ScoreSimdT::LEN;
        let chunks = data.len() / chunk_size;
        let mut sum_vector = ScoreSimdT::splat(0.0);

        // Process full SIMD chunks
        for i in 0..chunks {
            let start = i * chunk_size;
            let end = start + chunk_size;

            let v = ScoreSimdT::from_slice(&data[start..end]);
            sum_vector += v;
        }

        // Sum the SIMD vector elements
        let mut result = sum_vector.reduce_sum();

        // Handle remaining elements
        let remaining_start = chunks * chunk_size;
        for i in remaining_start..data.len() {
            result += data[i];
        }

        result
    }

    /// SIMD-accelerated scalar multiplication.
    /// Computes result[i] = data[i] * scalar for all elements using vectorized operations.
    pub fn simd_scalar_mul(data: &[ScoreT], scalar: ScoreT, result: &mut [ScoreT]) {
        assert_eq!(data.len(), result.len());

        let chunk_size = ScoreSimdT::LEN;
        let chunks = data.len() / chunk_size;
        let scalar_vector = ScoreSimdT::splat(scalar);

        // Process full SIMD chunks
        for i in 0..chunks {
            let start = i * chunk_size;
            let end = start + chunk_size;

            let v = ScoreSimdT::from_slice(&data[start..end]);
            let vr = v * scalar_vector;

            vr.copy_to_slice(&mut result[start..end]);
        }

        // Handle remaining elements
        let remaining_start = chunks * chunk_size;
        for i in remaining_start..data.len() {
            result[i] = data[i] * scalar;
        }
    }
}

/// Memory-aligned vector for SIMD operations.
/// Provides 32-byte alignment for optimal cache performance and SIMD compatibility.
#[repr(align(32))]
#[derive(Debug)]
pub struct AlignedVector<T> {
    data: Vec<T>,
}

impl<T> AlignedVector<T> {
    /// Creates a new empty aligned vector.
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Creates a new aligned vector with the specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Appends an element to the back of the vector.
    pub fn push(&mut self, value: T) {
        self.data.push(value);
    }

    /// Returns a slice containing the entire vector.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Returns a mutable slice containing the entire vector.
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Returns the number of elements in the vector.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the vector contains no elements.
    pub fn empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Resizes the vector in-place so that len is equal to new_size.
    pub fn resize(&mut self, new_size: usize, value: T)
    where
        T: Clone,
    {
        self.data.resize(new_size, value);
    }

    /// Clears the vector, removing all values.
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

impl<T: Default> Default for AlignedVector<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Type aliases for common aligned vectors matching C++ style

/// Aligned vector specialized for score values.
pub type ScoreVector = AlignedVector<ScoreT>;
/// Aligned vector specialized for label values.
pub type LabelVector = AlignedVector<LabelT>;
/// Aligned vector specialized for data size values.
pub type DataSizeVector = AlignedVector<DataSizeT>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_add_scores() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut result = vec![0.0; 8];

        simd_ops::simd_add_scores(&a, &b, &mut result);

        for &val in &result {
            assert_eq!(val, 9.0);
        }
    }

    #[test]
    fn test_simd_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 3.0, 2.0, 1.0];

        let result = simd_ops::simd_dot_product(&a, &b);
        assert_eq!(result, 20.0); // 1*4 + 2*3 + 3*2 + 4*1 = 20
    }

    #[test]
    fn test_AlignedVector() {
        let mut vec = ScoreVector::new();
        vec.push(1.0);
        vec.push(2.0);

        assert_eq!(vec.size(), 2);
        assert_eq!(vec.data(), &[1.0, 2.0]);
    }

    #[test]
    fn test_constants() {
        assert_eq!(NO_SPECIFIC, -1);
        assert_eq!(K_ALIGNED_SIZE, 32);
        assert!(K_MIN_SCORE.is_infinite() && K_MIN_SCORE.is_sign_negative());
        assert!(K_MAX_SCORE.is_infinite() && K_MAX_SCORE.is_sign_positive());
    }

    #[test]
    fn test_size_aligned_macro() {
        assert_eq!(SIZE_ALIGNED!(10), 32);
        assert_eq!(SIZE_ALIGNED!(32), 32);
        assert_eq!(SIZE_ALIGNED!(33), 64);
    }
}
