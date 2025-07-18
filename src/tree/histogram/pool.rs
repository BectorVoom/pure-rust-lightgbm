//! Histogram memory pool for efficient histogram construction.
//!
//! This module provides a memory pool for histogram arrays to avoid frequent
//! allocations and deallocations during tree construction.

use crate::core::types::{DataSize, FeatureIndex, Hist};
use ndarray::Array1;
use std::collections::VecDeque;

/// Configuration for histogram pool management.
#[derive(Debug, Clone)]
pub struct HistogramPoolConfig {
    /// Maximum number of bins per feature
    pub max_bin: usize,
    /// Number of features in the dataset
    pub num_features: usize,
    /// Maximum number of histograms to keep in the pool
    pub max_pool_size: usize,
    /// Initial pool size to pre-allocate
    pub initial_pool_size: usize,
    /// Whether to use double-precision histograms
    pub use_double_precision: bool,
}

impl Default for HistogramPoolConfig {
    fn default() -> Self {
        HistogramPoolConfig {
            max_bin: 255,
            num_features: 0,
            max_pool_size: 1000,
            initial_pool_size: 100,
            use_double_precision: true,
        }
    }
}

/// Memory pool for histogram arrays to reduce allocation overhead.
pub struct HistogramPool {
    /// Configuration for the pool
    config: HistogramPoolConfig,
    /// Pool of available histograms for single features
    feature_histogram_pool: VecDeque<Array1<Hist>>,
    /// Pool of available histograms for all features combined
    full_histogram_pool: VecDeque<Array1<Hist>>,
    /// Size of each feature histogram (bins * 2 for gradient and hessian)
    feature_histogram_size: usize,
    /// Size of full histogram (features * bins * 2)
    full_histogram_size: usize,
    /// Statistics for pool usage
    total_allocations: usize,
    pool_hits: usize,
    pool_misses: usize,
}

impl HistogramPool {
    /// Creates a new histogram pool with the given configuration.
    pub fn new(config: HistogramPoolConfig) -> Self {
        let feature_histogram_size = config.max_bin * 2; // gradient + hessian per bin
        let full_histogram_size = config.num_features * feature_histogram_size;

        let mut pool = HistogramPool {
            config: config.clone(),
            feature_histogram_pool: VecDeque::new(),
            full_histogram_pool: VecDeque::new(),
            feature_histogram_size,
            full_histogram_size,
            total_allocations: 0,
            pool_hits: 0,
            pool_misses: 0,
        };

        // Pre-allocate initial histograms
        pool.preallocate_histograms();
        pool
    }

    /// Pre-allocates initial histograms to warm up the pool.
    fn preallocate_histograms(&mut self) {
        for _ in 0..self.config.initial_pool_size {
            self.feature_histogram_pool
                .push_back(Array1::zeros(self.feature_histogram_size));
            if self.config.num_features > 0 {
                self.full_histogram_pool
                    .push_back(Array1::zeros(self.full_histogram_size));
            }
        }
    }

    /// Gets a histogram for a single feature from the pool.
    pub fn get_feature_histogram(&mut self) -> Array1<Hist> {
        self.total_allocations += 1;

        if let Some(mut histogram) = self.feature_histogram_pool.pop_front() {
            self.pool_hits += 1;
            histogram.fill(0.0); // Clear the histogram
            histogram
        } else {
            self.pool_misses += 1;
            Array1::zeros(self.feature_histogram_size)
        }
    }

    /// Gets a histogram for all features from the pool.
    pub fn get_full_histogram(&mut self) -> Array1<Hist> {
        self.total_allocations += 1;

        if let Some(mut histogram) = self.full_histogram_pool.pop_front() {
            self.pool_hits += 1;
            histogram.fill(0.0); // Clear the histogram
            histogram
        } else {
            self.pool_misses += 1;
            Array1::zeros(self.full_histogram_size)
        }
    }

    /// Returns a feature histogram to the pool.
    pub fn return_feature_histogram(&mut self, histogram: Array1<Hist>) {
        if self.feature_histogram_pool.len() < self.config.max_pool_size
            && histogram.len() == self.feature_histogram_size
        {
            self.feature_histogram_pool.push_back(histogram);
        }
        // If pool is full or histogram is wrong size, just let it drop
    }

    /// Returns a full histogram to the pool.
    pub fn return_full_histogram(&mut self, histogram: Array1<Hist>) {
        if self.full_histogram_pool.len() < self.config.max_pool_size
            && histogram.len() == self.full_histogram_size
        {
            self.full_histogram_pool.push_back(histogram);
        }
        // If pool is full or histogram is wrong size, just let it drop
    }

    /// Gets multiple feature histograms at once.
    pub fn get_feature_histograms(&mut self, count: usize) -> Vec<Array1<Hist>> {
        let mut histograms = Vec::with_capacity(count);
        for _ in 0..count {
            histograms.push(self.get_feature_histogram());
        }
        histograms
    }

    /// Returns multiple feature histograms to the pool.
    pub fn return_feature_histograms(&mut self, histograms: Vec<Array1<Hist>>) {
        for histogram in histograms {
            self.return_feature_histogram(histogram);
        }
    }

    /// Clears all histograms from the pool.
    pub fn clear(&mut self) {
        self.feature_histogram_pool.clear();
        self.full_histogram_pool.clear();
    }

    /// Resets the pool and statistics.
    pub fn reset(&mut self) {
        self.clear();
        self.total_allocations = 0;
        self.pool_hits = 0;
        self.pool_misses = 0;
        self.preallocate_histograms();
    }

    /// Returns the current pool configuration.
    pub fn config(&self) -> &HistogramPoolConfig {
        &self.config
    }

    /// Updates the pool configuration.
    pub fn update_config(&mut self, config: HistogramPoolConfig) {
        let old_feature_size = self.feature_histogram_size;
        let old_full_size = self.full_histogram_size;

        self.config = config;
        self.feature_histogram_size = self.config.max_bin * 2;
        self.full_histogram_size = self.config.num_features * self.feature_histogram_size;

        // If histogram sizes changed, clear the pool
        if old_feature_size != self.feature_histogram_size || old_full_size != self.full_histogram_size {
            self.clear();
            self.preallocate_histograms();
        }
    }

    /// Returns pool usage statistics.
    pub fn statistics(&self) -> HistogramPoolStatistics {
        HistogramPoolStatistics {
            total_allocations: self.total_allocations,
            pool_hits: self.pool_hits,
            pool_misses: self.pool_misses,
            hit_rate: if self.total_allocations > 0 {
                self.pool_hits as f64 / self.total_allocations as f64
            } else {
                0.0
            },
            feature_pool_size: self.feature_histogram_pool.len(),
            full_pool_size: self.full_histogram_pool.len(),
            feature_histogram_size: self.feature_histogram_size,
            full_histogram_size: self.full_histogram_size,
        }
    }

    /// Returns the size of a feature histogram.
    pub fn feature_histogram_size(&self) -> usize {
        self.feature_histogram_size
    }

    /// Returns the size of a full histogram.
    pub fn full_histogram_size(&self) -> usize {
        self.full_histogram_size
    }

    /// Returns the current number of feature histograms in the pool.
    pub fn feature_pool_size(&self) -> usize {
        self.feature_histogram_pool.len()
    }

    /// Returns the current number of full histograms in the pool.
    pub fn full_pool_size(&self) -> usize {
        self.full_histogram_pool.len()
    }

    /// Shrinks the pool to the target size.
    pub fn shrink_to_fit(&mut self, target_size: usize) {
        while self.feature_histogram_pool.len() > target_size {
            self.feature_histogram_pool.pop_back();
        }
        while self.full_histogram_pool.len() > target_size {
            self.full_histogram_pool.pop_back();
        }
    }

    /// Estimates memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let feature_memory = self.feature_histogram_pool.len() * self.feature_histogram_size * std::mem::size_of::<Hist>();
        let full_memory = self.full_histogram_pool.len() * self.full_histogram_size * std::mem::size_of::<Hist>();
        feature_memory + full_memory
    }
}

/// RAII wrapper for histogram borrowing.
pub struct HistogramBorrow<'a> {
    pool: &'a mut HistogramPool,
    histogram: Option<Array1<Hist>>,
    is_feature_histogram: bool,
}

impl<'a> HistogramBorrow<'a> {
    /// Creates a new histogram borrow for a feature histogram.
    pub fn feature_histogram(pool: &'a mut HistogramPool) -> Self {
        let histogram = pool.get_feature_histogram();
        HistogramBorrow {
            pool,
            histogram: Some(histogram),
            is_feature_histogram: true,
        }
    }

    /// Creates a new histogram borrow for a full histogram.
    pub fn full_histogram(pool: &'a mut HistogramPool) -> Self {
        let histogram = pool.get_full_histogram();
        HistogramBorrow {
            pool,
            histogram: Some(histogram),
            is_feature_histogram: false,
        }
    }

    /// Gets a reference to the borrowed histogram.
    pub fn histogram(&self) -> &Array1<Hist> {
        self.histogram.as_ref().expect("Histogram should be available")
    }

    /// Gets a mutable reference to the borrowed histogram.
    pub fn histogram_mut(&mut self) -> &mut Array1<Hist> {
        self.histogram.as_mut().expect("Histogram should be available")
    }

    /// Takes ownership of the histogram, consuming the borrow.
    pub fn take(mut self) -> Array1<Hist> {
        self.histogram.take().expect("Histogram should be available")
    }
}

impl<'a> Drop for HistogramBorrow<'a> {
    fn drop(&mut self) {
        if let Some(histogram) = self.histogram.take() {
            if self.is_feature_histogram {
                self.pool.return_feature_histogram(histogram);
            } else {
                self.pool.return_full_histogram(histogram);
            }
        }
    }
}

/// Statistics about histogram pool usage.
#[derive(Debug, Clone)]
pub struct HistogramPoolStatistics {
    /// Total number of histogram allocations requested
    pub total_allocations: usize,
    /// Number of allocations served from the pool
    pub pool_hits: usize,
    /// Number of allocations that required new allocations
    pub pool_misses: usize,
    /// Pool hit rate (hits / total_allocations)
    pub hit_rate: f64,
    /// Current number of feature histograms in pool
    pub feature_pool_size: usize,
    /// Current number of full histograms in pool
    pub full_pool_size: usize,
    /// Size of each feature histogram in elements
    pub feature_histogram_size: usize,
    /// Size of each full histogram in elements
    pub full_histogram_size: usize,
}

impl std::fmt::Display for HistogramPoolStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HistogramPool(allocations={}, hits={}, misses={}, hit_rate={:.2}, feature_pool={}, full_pool={})",
            self.total_allocations,
            self.pool_hits,
            self.pool_misses,
            self.hit_rate,
            self.feature_pool_size,
            self.full_pool_size
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_pool_creation() {
        let config = HistogramPoolConfig {
            max_bin: 10,
            num_features: 5,
            max_pool_size: 50,
            initial_pool_size: 10,
            use_double_precision: true,
        };

        let pool = HistogramPool::new(config);
        assert_eq!(pool.feature_histogram_size(), 20); // 10 bins * 2
        assert_eq!(pool.full_histogram_size(), 100);   // 5 features * 20
        assert_eq!(pool.feature_pool_size(), 10);
        assert_eq!(pool.full_pool_size(), 10);
    }

    #[test]
    fn test_histogram_borrowing() {
        let config = HistogramPoolConfig {
            max_bin: 5,
            num_features: 2,
            max_pool_size: 10,
            initial_pool_size: 2,
            use_double_precision: true,
        };

        let mut pool = HistogramPool::new(config);
        
        // Test feature histogram borrowing
        {
            let borrow = HistogramBorrow::feature_histogram(&mut pool);
            assert_eq!(borrow.histogram().len(), 10); // 5 bins * 2
        }

        // Pool should have one less histogram after borrow is returned
        let stats = pool.statistics();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.pool_hits, 1);
    }

    #[test]
    fn test_histogram_get_and_return() {
        let config = HistogramPoolConfig::default();
        let mut pool = HistogramPool::new(config);

        let initial_size = pool.feature_pool_size();
        
        // Get a histogram
        let histogram = pool.get_feature_histogram();
        assert_eq!(pool.feature_pool_size(), initial_size - 1);

        // Return the histogram
        pool.return_feature_histogram(histogram);
        assert_eq!(pool.feature_pool_size(), initial_size);
    }

    #[test]
    fn test_pool_statistics() {
        let config = HistogramPoolConfig {
            max_bin: 5,
            num_features: 1,
            max_pool_size: 5,
            initial_pool_size: 2,
            use_double_precision: true,
        };

        let mut pool = HistogramPool::new(config);

        // Use up the pool
        let _h1 = pool.get_feature_histogram();
        let _h2 = pool.get_feature_histogram();
        let _h3 = pool.get_feature_histogram(); // This should be a miss

        let stats = pool.statistics();
        assert_eq!(stats.total_allocations, 3);
        assert_eq!(stats.pool_hits, 2);
        assert_eq!(stats.pool_misses, 1);
        assert!((stats.hit_rate - 2.0/3.0).abs() < 1e-6);
    }

    #[test]
    fn test_pool_reset() {
        let config = HistogramPoolConfig::default();
        let mut pool = HistogramPool::new(config);

        // Use the pool
        let _h1 = pool.get_feature_histogram();
        let _h2 = pool.get_feature_histogram();

        let stats_before = pool.statistics();
        assert!(stats_before.total_allocations > 0);

        // Reset the pool
        pool.reset();

        let stats_after = pool.statistics();
        assert_eq!(stats_after.total_allocations, 0);
        assert_eq!(stats_after.pool_hits, 0);
        assert_eq!(stats_after.pool_misses, 0);
    }

    #[test]
    fn test_memory_usage_estimation() {
        let config = HistogramPoolConfig {
            max_bin: 10,
            num_features: 2,
            max_pool_size: 10,
            initial_pool_size: 5,
            use_double_precision: true,
        };

        let pool = HistogramPool::new(config);
        let memory_usage = pool.memory_usage();
        
        // Should have some memory usage from the initial histograms
        assert!(memory_usage > 0);
        
        // Rough calculation: 5 feature histograms (20 elements each) + 5 full histograms (40 elements each)
        // = 100 + 200 = 300 elements * 8 bytes per f64 = 2400 bytes
        let expected_min = 2000; // Allow some variance
        let expected_max = 3000;
        assert!(memory_usage >= expected_min && memory_usage <= expected_max);
    }
}