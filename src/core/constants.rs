//! System constants and alignment specifications for Pure Rust LightGBM.
//!
//! This module defines critical constants that control memory alignment,
//! SIMD operations, and default configuration values throughout the system.

use crate::core::types::*;

/// Memory alignment size for SIMD operations (32-byte aligned like C++ version).
/// This ensures optimal performance for vectorized operations on modern CPUs.
pub const ALIGNED_SIZE: usize = 32;

/// Default maximum number of bins for feature discretization.
/// This balances memory usage with split finding quality.
pub const DEFAULT_MAX_BIN: usize = 255;

/// Default minimum number of data points required in a leaf.
/// Prevents overfitting by ensuring statistical significance.
pub const DEFAULT_MIN_DATA_IN_LEAF: DataSize = 20;

/// Default minimum sum of hessian values required in a leaf.
/// Provides numerical stability for leaf value calculation.
pub const DEFAULT_MIN_SUM_HESSIAN_IN_LEAF: f64 = 1e-3;

/// Default maximum tree depth.
/// Negative value means no limit.
pub const DEFAULT_MAX_DEPTH: i32 = -1;

/// Default number of leaves for each tree.
/// Controls model complexity and training speed.
pub const DEFAULT_NUM_LEAVES: usize = 31;

/// Default learning rate (shrinkage) for gradient boosting.
/// Controls the contribution of each tree to the ensemble.
pub const DEFAULT_LEARNING_RATE: f64 = 0.1;

/// Default L1 regularization parameter.
/// Promotes sparsity in the model.
pub const DEFAULT_LAMBDA_L1: f64 = 0.0;

/// Default L2 regularization parameter.
/// Prevents overfitting through weight decay.
pub const DEFAULT_LAMBDA_L2: f64 = 0.0;

/// Default feature fraction for subsampling features.
/// Controls the fraction of features used for each tree.
pub const DEFAULT_FEATURE_FRACTION: f64 = 1.0;

/// Default bagging fraction for subsampling data.
/// Controls the fraction of data used for each tree.
pub const DEFAULT_BAGGING_FRACTION: f64 = 1.0;

/// Default bagging frequency.
/// Controls how often to perform bagging (0 = no bagging).
pub const DEFAULT_BAGGING_FREQ: usize = 0;

/// Default number of boosting iterations.
/// Controls the number of trees in the ensemble.
pub const DEFAULT_NUM_ITERATIONS: usize = 100;

/// Default number of classes for multiclass classification.
/// Only relevant for multiclass objectives.
pub const DEFAULT_NUM_CLASS: usize = 1;

/// Default early stopping rounds.
/// Training stops if no improvement for this many rounds.
pub const DEFAULT_EARLY_STOPPING_ROUNDS: usize = 10;

/// Default early stopping tolerance.
/// Minimum improvement required to continue training.
pub const DEFAULT_EARLY_STOPPING_TOLERANCE: f64 = 1e-5;

/// Default number of threads for parallel processing.
/// 0 means use all available cores.
pub const DEFAULT_NUM_THREADS: usize = 0;

/// Default random seed for reproducibility.
pub const DEFAULT_RANDOM_SEED: u64 = 0;

/// Default verbosity level for logging.
pub const DEFAULT_VERBOSITY: VerbosityLevel = VerbosityLevel::Info;

/// Maximum allowed feature value for numerical features.
/// Used for input validation and binning.
pub const MAX_FEATURE_VALUE: f64 = 1e30;

/// Minimum allowed feature value for numerical features.
/// Used for input validation and binning.
pub const MIN_FEATURE_VALUE: f64 = -1e30;

/// Epsilon value for floating point comparisons.
/// Used throughout the system for numerical stability.
pub const EPSILON: f64 = 1e-15;

/// Small value used for avoiding division by zero.
pub const SMALL_VALUE: f64 = 1e-35;

/// Large value used for initialization of min/max searches.
pub const LARGE_VALUE: f64 = 1e30;

/// Maximum number of categorical features allowed.
/// Limits memory usage and processing complexity.
pub const MAX_CATEGORICAL_FEATURES: usize = 1000;

/// Maximum number of categories per categorical feature.
/// Prevents excessive memory usage for high-cardinality features.
pub const MAX_CATEGORIES_PER_FEATURE: usize = 65536;

/// Default batch size for GPU operations.
/// Balances memory usage with computational efficiency.
pub const DEFAULT_GPU_BATCH_SIZE: usize = 65536;

/// Maximum work group size for GPU kernels.
/// Hardware-dependent limit for OpenCL/CUDA kernels.
pub const MAX_GPU_WORK_GROUP_SIZE: usize = 1024;

/// Default number of histogram bins for GPU processing.
/// Optimized for GPU memory hierarchy.
pub const DEFAULT_GPU_MAX_BIN: usize = 63;

/// Chunk size for parallel processing operations.
/// Optimizes cache locality and load balancing.
pub const PARALLEL_CHUNK_SIZE: usize = 1024;

/// Default histogram pool size.
/// Controls memory allocation for histogram construction.
pub const DEFAULT_HISTOGRAM_POOL_SIZE: usize = 16;

/// Maximum string length for feature names.
/// Prevents excessive memory usage for metadata.
pub const MAX_FEATURE_NAME_LENGTH: usize = 256;

/// Default timeout for GPU operations (in milliseconds).
/// Prevents hanging on GPU operations.
pub const DEFAULT_GPU_TIMEOUT_MS: u64 = 30000;

/// SIMD vector width for f32 operations.
/// Hardware-dependent optimization parameter.
pub const SIMD_F32_WIDTH: usize = 8;

/// SIMD vector width for f64 operations.
/// Hardware-dependent optimization parameter.
pub const SIMD_F64_WIDTH: usize = 4;

/// Cache line size for memory alignment optimizations.
/// Prevents false sharing in parallel operations.
pub const CACHE_LINE_SIZE: usize = 64;

/// Default prediction buffer size.
/// Initial allocation size for prediction operations.
pub const DEFAULT_PREDICTION_BUFFER_SIZE: usize = 10000;

/// Version information.
pub const LIGHTGBM_RUST_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const LIGHTGBM_RUST_VERSION_MAJOR: &str = env!("CARGO_PKG_VERSION_MAJOR");
pub const LIGHTGBM_RUST_VERSION_MINOR: &str = env!("CARGO_PKG_VERSION_MINOR");
pub const LIGHTGBM_RUST_VERSION_PATCH: &str = env!("CARGO_PKG_VERSION_PATCH");

/// Compile-time assertions to verify type sizes and alignments.
#[cfg(test)]
mod compile_time_assertions {
    use super::*;
    use static_assertions::*;

    // Verify that our alignment constant is a power of 2
    const_assert!(ALIGNED_SIZE.is_power_of_two());
    
    // Verify that ALIGNED_SIZE is at least as large as the cache line size
    const_assert!(ALIGNED_SIZE >= CACHE_LINE_SIZE / 2);
    
    // Verify that SIMD widths are reasonable
    const_assert!(SIMD_F32_WIDTH <= 16);
    const_assert!(SIMD_F64_WIDTH <= 8);
    
    // Verify that default values are reasonable
    const_assert!(DEFAULT_MAX_BIN <= 65536);
    const_assert!(DEFAULT_NUM_LEAVES >= 2);
    const_assert!(DEFAULT_LEARNING_RATE > 0.0);
    const_assert!(DEFAULT_MIN_DATA_IN_LEAF > 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_properties() {
        // Verify ALIGNED_SIZE is a power of 2
        assert!(ALIGNED_SIZE.is_power_of_two());
        
        // Verify ALIGNED_SIZE is reasonable for SIMD operations
        assert!(ALIGNED_SIZE >= 16);
        assert!(ALIGNED_SIZE <= 128);
    }

    #[test]
    fn test_default_values() {
        // Test that default values are within reasonable ranges
        assert!(DEFAULT_MAX_BIN > 0);
        assert!(DEFAULT_MAX_BIN <= 65536);
        
        assert!(DEFAULT_MIN_DATA_IN_LEAF > 0);
        assert!(DEFAULT_MIN_SUM_HESSIAN_IN_LEAF > 0.0);
        
        assert!(DEFAULT_NUM_LEAVES >= 2);
        assert!(DEFAULT_LEARNING_RATE > 0.0);
        assert!(DEFAULT_LEARNING_RATE <= 1.0);
        
        assert!(DEFAULT_LAMBDA_L1 >= 0.0);
        assert!(DEFAULT_LAMBDA_L2 >= 0.0);
        
        assert!(DEFAULT_FEATURE_FRACTION > 0.0);
        assert!(DEFAULT_FEATURE_FRACTION <= 1.0);
        
        assert!(DEFAULT_BAGGING_FRACTION > 0.0);
        assert!(DEFAULT_BAGGING_FRACTION <= 1.0);
    }

    #[test]
    fn test_numerical_constants() {
        // Test that epsilon values are reasonable
        assert!(EPSILON > 0.0);
        assert!(EPSILON < 1e-10);
        
        assert!(SMALL_VALUE > 0.0);
        assert!(SMALL_VALUE < EPSILON);
        
        assert!(LARGE_VALUE > 1.0);
        
        // Test feature value bounds
        assert!(MIN_FEATURE_VALUE < 0.0);
        assert!(MAX_FEATURE_VALUE > 0.0);
        assert!(MAX_FEATURE_VALUE >= MIN_FEATURE_VALUE.abs());
    }

    #[test]
    fn test_simd_constants() {
        // Test SIMD width constants
        assert!(SIMD_F32_WIDTH > 0);
        assert!(SIMD_F64_WIDTH > 0);
        assert!(SIMD_F32_WIDTH >= SIMD_F64_WIDTH);
    }

    #[test]
    fn test_gpu_constants() {
        // Test GPU-related constants
        assert!(DEFAULT_GPU_BATCH_SIZE > 0);
        assert!(MAX_GPU_WORK_GROUP_SIZE > 0);
        assert!(DEFAULT_GPU_MAX_BIN > 0);
        assert!(DEFAULT_GPU_TIMEOUT_MS > 0);
    }

    #[test]
    fn test_parallel_constants() {
        // Test parallel processing constants
        assert!(PARALLEL_CHUNK_SIZE > 0);
        assert!(DEFAULT_HISTOGRAM_POOL_SIZE > 0);
        assert!(CACHE_LINE_SIZE > 0);
        assert!(CACHE_LINE_SIZE.is_power_of_two());
    }

    #[test]
    fn test_version_constants() {
        // Test that version constants are not empty
        assert!(!LIGHTGBM_RUST_VERSION.is_empty());
        assert!(!LIGHTGBM_RUST_VERSION_MAJOR.is_empty());
        assert!(!LIGHTGBM_RUST_VERSION_MINOR.is_empty());
        assert!(!LIGHTGBM_RUST_VERSION_PATCH.is_empty());
    }

    #[test]
    fn test_bounds_constants() {
        // Test maximum bounds are reasonable
        assert!(MAX_CATEGORICAL_FEATURES > 0);
        assert!(MAX_CATEGORIES_PER_FEATURE > 0);
        assert!(MAX_FEATURE_NAME_LENGTH > 0);
        assert!(DEFAULT_PREDICTION_BUFFER_SIZE > 0);
    }
}