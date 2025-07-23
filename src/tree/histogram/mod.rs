//! Histogram construction module for the Pure Rust LightGBM framework.
//!
//! This module provides efficient histogram construction algorithms with
//! memory pooling, SIMD optimizations, and parallel processing capabilities.

pub mod builder;
pub mod pool;
pub mod simd;

// Re-export key types and traits
pub use builder::{
    BinMapper, FeatureType, HistogramBuilder, HistogramBuilderConfig,
    FeatureHistogram, HistogramStatistics,
};
pub use pool::{
    HistogramPool, HistogramPoolConfig, HistogramPoolStatistics, HistogramBorrow,
};
pub use simd::{
    AlignedBuffer, SimdConfig, SimdHistogramAccumulator,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Hist;
    use ndarray::Array1;

    #[test]
    fn test_module_integration() {
        // Test that all histogram components work together
        let pool_config = HistogramPoolConfig {
            max_bin: 10,
            num_features: 3,
            max_pool_size: 20,
            initial_pool_size: 5,
            use_double_precision: true,
        };

        let builder_config = HistogramBuilderConfig {
            num_threads: 2,
            chunk_size: 512,
            use_simd: true,
            use_sparse: false,
            sparse_threshold: 0.5,
        };

        let simd_config = SimdConfig {
            vector_width: 4,
            alignment: 16,
            min_simd_size: 32,
            use_avx2: false, // Disable for testing
            use_avx512: false,
        };

        // Create components
        let pool = HistogramPool::new(pool_config);
        let builder = HistogramBuilder::new(builder_config, pool);
        let simd_accumulator = SimdHistogramAccumulator::new(simd_config);

        // Verify components are functional
        assert_eq!(builder.config().num_threads, 2);
        assert_eq!(simd_accumulator.config.vector_width, 4);

        let stats = builder.histogram_pool().lock().unwrap().statistics();
        assert_eq!(stats.feature_pool_size, 5);
    }

    #[test]
    fn test_histogram_workflow() {
        // Test a complete histogram construction workflow
        let pool_config = HistogramPoolConfig {
            max_bin: 5,
            num_features: 2,
            max_pool_size: 10,
            initial_pool_size: 3,
            use_double_precision: true,
        };

        let mut pool = HistogramPool::new(pool_config);

        // Get histogram from pool
        let mut histogram = pool.get_feature_histogram();
        assert_eq!(histogram.len(), 10); // 5 bins * 2

        // Simulate histogram accumulation
        histogram[0] = 1.5;  // bin 0 gradient
        histogram[1] = 2.0;  // bin 0 hessian
        histogram[2] = 0.8;  // bin 1 gradient
        histogram[3] = 1.2;  // bin 1 hessian

        // Return to pool
        pool.return_feature_histogram(histogram);

        // Verify pool statistics
        let stats = pool.statistics();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.pool_hits, 1);
    }

    #[test]
    fn test_bin_mapper_integration() {
        // Test bin mapper with histogram construction
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mapper = BinMapper::new_numerical(&values, 4);

        assert_eq!(mapper.num_bins(), 4);

        // Test value to bin mapping
        let test_values = vec![1.5, 3.5, 5.5, 7.5];
        let expected_bins = vec![0, 1, 2, 3];

        for (value, expected_bin) in test_values.iter().zip(expected_bins.iter()) {
            let actual_bin = mapper.value_to_bin(*value);
            assert_eq!(actual_bin as usize, *expected_bin);
        }
    }

    #[test]
    fn test_histogram_borrow_raii() {
        let pool_config = HistogramPoolConfig {
            max_bin: 3,
            num_features: 1,
            max_pool_size: 5,
            initial_pool_size: 2,
            use_double_precision: true,
        };

        let mut pool = HistogramPool::new(pool_config);
        let initial_size = pool.feature_pool_size();

        {
            // Borrow histogram using RAII wrapper
            let mut borrow = HistogramBorrow::feature_histogram(&mut pool);
            let histogram = borrow.histogram_mut();
            
            assert_eq!(histogram.len(), 6); // 3 bins * 2
            histogram[0] = 42.0;
            
            assert_eq!(pool.feature_pool_size(), initial_size - 1);
        } // Histogram should be automatically returned here

        // Verify histogram was returned to pool
        assert_eq!(pool.feature_pool_size(), initial_size);
    }

    #[test]
    fn test_aligned_buffer_usage() {
        let mut buffer: AlignedBuffer<Hist> = AlignedBuffer::new(16, 32);
        
        // Test basic operations
        assert!(buffer.is_empty());
        assert_eq!(buffer.capacity(), 16);

        buffer.resize(8, 0.0);
        assert_eq!(buffer.len(), 8);

        // Modify values
        let slice = buffer.as_mut_slice();
        slice[0] = 1.5;
        slice[1] = 2.5;

        // Verify values
        let read_slice = buffer.as_slice();
        assert_eq!(read_slice[0], 1.5);
        assert_eq!(read_slice[1], 2.5);

        buffer.clear();
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_simd_accumulator_basic() {
        let config = SimdConfig {
            min_simd_size: 1000, // Force scalar path for testing
            ..SimdConfig::default()
        };
        let accumulator = SimdHistogramAccumulator::new(config);

        let mut histogram = Array1::zeros(8); // 4 bins * 2
        let gradients = Array1::from(vec![0.1, 0.2, 0.3, 0.4]);
        let hessians = Array1::from(vec![1.0, 1.0, 1.0, 1.0]);
        let data_indices = vec![0, 1, 2, 3];
        let bins = vec![0, 1, 2, 3];

        let result = accumulator.accumulate_histogram(
            &mut histogram,
            &gradients.view(),
            &hessians.view(),
            &data_indices,
            &bins,
        );

        assert!(result.is_ok());

        // Verify accumulation
        assert_eq!(histogram[0], 0.1); // bin 0 gradient
        assert_eq!(histogram[1], 1.0); // bin 0 hessian
        assert_eq!(histogram[6], 0.4); // bin 3 gradient
        assert_eq!(histogram[7], 1.0); // bin 3 hessian
    }

    #[test]
    fn test_sparse_vs_dense_threshold() {
        let builder_config = HistogramBuilderConfig {
            use_sparse: true,
            sparse_threshold: 0.3,
            ..Default::default()
        };

        let pool_config = HistogramPoolConfig::default();
        let pool = HistogramPool::new(pool_config);
        let builder = HistogramBuilder::new(builder_config, pool);

        // Test threshold behavior
        assert!(builder.should_use_sparse_construction(25, 100)); // 25% < 30% threshold
        assert!(!builder.should_use_sparse_construction(35, 100)); // 35% > 30% threshold
    }

    #[test]
    fn test_memory_usage_tracking() {
        let pool_config = HistogramPoolConfig {
            max_bin: 10,
            num_features: 5,
            max_pool_size: 20,
            initial_pool_size: 5,
            use_double_precision: true,
        };

        let pool = HistogramPool::new(pool_config);
        let memory_usage = pool.memory_usage();

        // Should have non-zero memory usage from initial histograms
        assert!(memory_usage > 0);

        let stats = pool.statistics();
        
        // Feature histograms: 5 histograms * 10 bins * 2 * 8 bytes = 800 bytes
        // Full histograms: 5 histograms * 5 features * 10 bins * 2 * 8 bytes = 4000 bytes
        // Total: ~4800 bytes
        assert!(memory_usage >= 4000);
        assert!(memory_usage <= 6000); // Allow some overhead
    }
}