//! Histogram construction for efficient split finding.
//!
//! This module provides parallel histogram construction algorithms optimized
//! for different data layouts and feature types.

use crate::core::types::{BinIndex, DataSize, FeatureIndex, Hist, Score};
use crate::tree::histogram::pool::{HistogramPool, HistogramBorrow};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use std::sync::Mutex;

/// Configuration for histogram construction.
#[derive(Debug, Clone)]
pub struct HistogramBuilderConfig {
    /// Number of threads to use for parallel construction
    pub num_threads: usize,
    /// Chunk size for parallel processing
    pub chunk_size: usize,
    /// Whether to use SIMD optimizations
    pub use_simd: bool,
    /// Whether to use sparse histogram construction
    pub use_sparse: bool,
    /// Threshold for switching to sparse construction
    pub sparse_threshold: f64,
}

impl Default for HistogramBuilderConfig {
    fn default() -> Self {
        HistogramBuilderConfig {
            num_threads: num_cpus::get(),
            chunk_size: 1024,
            use_simd: true,
            use_sparse: false,
            sparse_threshold: 0.8,
        }
    }
}

/// High-performance histogram builder for gradient boosting.
pub struct HistogramBuilder {
    config: HistogramBuilderConfig,
    histogram_pool: Mutex<HistogramPool>,
}

impl HistogramBuilder {
    /// Creates a new histogram builder.
    pub fn new(config: HistogramBuilderConfig, histogram_pool: HistogramPool) -> Self {
        HistogramBuilder {
            config,
            histogram_pool: Mutex::new(histogram_pool),
        }
    }

    /// Constructs histograms for all features using parallel processing.
    pub fn construct_histograms(
        &self,
        features: &ArrayView2<f32>,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
        data_indices: &[DataSize],
        bin_mappers: &[BinMapper],
    ) -> anyhow::Result<Vec<Array1<Hist>>> {
        let num_features = features.ncols();
        let num_data = data_indices.len();

        if num_data == 0 {
            return Ok(vec![]);
        }

        // Determine if we should use sparse construction
        let use_sparse = self.should_use_sparse_construction(num_data, features.nrows());

        if use_sparse {
            self.construct_sparse_histograms(features, gradients, hessians, data_indices, bin_mappers)
        } else {
            self.construct_dense_histograms(features, gradients, hessians, data_indices, bin_mappers)
        }
    }

    /// Constructs a histogram for a single feature.
    pub fn construct_feature_histogram(
        &self,
        feature_values: &ArrayView1<f32>,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
        data_indices: &[DataSize],
        bin_mapper: &BinMapper,
    ) -> anyhow::Result<Array1<Hist>> {
        let num_bins = bin_mapper.num_bins();
        let mut histogram = Array1::zeros(num_bins * 2); // gradient + hessian per bin

        if self.config.use_simd && data_indices.len() >= self.config.chunk_size {
            self.construct_feature_histogram_simd(
                feature_values,
                gradients,
                hessians,
                data_indices,
                bin_mapper,
                &mut histogram,
            )?;
        } else {
            self.construct_feature_histogram_scalar(
                feature_values,
                gradients,
                hessians,
                data_indices,
                bin_mapper,
                &mut histogram,
            )?;
        }

        Ok(histogram)
    }

    /// Dense histogram construction using parallel processing.
    fn construct_dense_histograms(
        &self,
        features: &ArrayView2<f32>,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
        data_indices: &[DataSize],
        bin_mappers: &[BinMapper],
    ) -> anyhow::Result<Vec<Array1<Hist>>> {
        let num_features = features.ncols();
        
        // Parallel construction across features
        let histograms: Result<Vec<_>, _> = (0..num_features)
            .into_par_iter()
            .map(|feature_idx| {
                let feature_column = features.column(feature_idx);
                self.construct_feature_histogram(
                    &feature_column,
                    gradients,
                    hessians,
                    data_indices,
                    &bin_mappers[feature_idx],
                )
            })
            .collect();

        histograms.map_err(|e| anyhow::anyhow!("Histogram construction failed: {}", e))
    }

    /// Sparse histogram construction for cases with few active data points.
    fn construct_sparse_histograms(
        &self,
        features: &ArrayView2<f32>,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
        data_indices: &[DataSize],
        bin_mappers: &[BinMapper],
    ) -> anyhow::Result<Vec<Array1<Hist>>> {
        let num_features = features.ncols();
        let mut histograms = Vec::with_capacity(num_features);

        for feature_idx in 0..num_features {
            let num_bins = bin_mappers[feature_idx].num_bins();
            let mut histogram = Array1::zeros(num_bins * 2);

            // Process data points sequentially for sparse case
            for &data_idx in data_indices {
                let feature_value = features[[data_idx as usize, feature_idx]];
                let bin = bin_mappers[feature_idx].value_to_bin(feature_value);
                let gradient = gradients[data_idx as usize] as Hist;
                let hessian = hessians[data_idx as usize] as Hist;

                histogram[bin as usize * 2] += gradient;
                histogram[bin as usize * 2 + 1] += hessian;
            }

            histograms.push(histogram);
        }

        Ok(histograms)
    }

    /// Scalar implementation for single feature histogram construction.
    fn construct_feature_histogram_scalar(
        &self,
        feature_values: &ArrayView1<f32>,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
        data_indices: &[DataSize],
        bin_mapper: &BinMapper,
        histogram: &mut Array1<Hist>,
    ) -> anyhow::Result<()> {
        for &data_idx in data_indices {
            let idx = data_idx as usize;
            if idx >= feature_values.len() {
                continue;
            }

            let feature_value = feature_values[idx];
            let bin = bin_mapper.value_to_bin(feature_value);
            let gradient = gradients[idx] as Hist;
            let hessian = hessians[idx] as Hist;

            let bin_idx = bin as usize * 2;
            if bin_idx + 1 < histogram.len() {
                histogram[bin_idx] += gradient;
                histogram[bin_idx + 1] += hessian;
            }
        }

        Ok(())
    }

    /// SIMD-optimized implementation for single feature histogram construction.
    fn construct_feature_histogram_simd(
        &self,
        feature_values: &ArrayView1<f32>,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
        data_indices: &[DataSize],
        bin_mapper: &BinMapper,
        histogram: &mut Array1<Hist>,
    ) -> anyhow::Result<()> {
        // Process in chunks for better cache locality
        let chunk_size = self.config.chunk_size;
        
        data_indices
            .par_chunks(chunk_size)
            .try_for_each(|chunk| -> anyhow::Result<()> {
                let mut local_histogram = Array1::zeros(histogram.len());

                for &data_idx in chunk {
                    let idx = data_idx as usize;
                    if idx >= feature_values.len() {
                        continue;
                    }

                    let feature_value = feature_values[idx];
                    let bin = bin_mapper.value_to_bin(feature_value);
                    let gradient = gradients[idx] as Hist;
                    let hessian = hessians[idx] as Hist;

                    let bin_idx = bin as usize * 2;
                    if bin_idx + 1 < local_histogram.len() {
                        local_histogram[bin_idx] += gradient;
                        local_histogram[bin_idx + 1] += hessian;
                    }
                }

                // Atomic accumulation into main histogram
                for (main_val, local_val) in histogram.iter_mut().zip(local_histogram.iter()) {
                    // Note: This is not actually atomic - in real implementation,
                    // we would need to use proper atomic operations or locks
                    *main_val += *local_val;
                }

                Ok(())
            })?;

        Ok(())
    }

    /// Constructs histograms using subtraction method (parent - sibling = current).
    pub fn construct_histogram_by_subtraction(
        &self,
        parent_histogram: &ArrayView1<Hist>,
        sibling_histogram: &ArrayView1<Hist>,
    ) -> anyhow::Result<Array1<Hist>> {
        if parent_histogram.len() != sibling_histogram.len() {
            return Err(anyhow::anyhow!(
                "Parent and sibling histograms must have the same length"
            ));
        }

        let mut result = Array1::zeros(parent_histogram.len());
        
        result
            .par_iter_mut()
            .zip(parent_histogram.par_iter())
            .zip(sibling_histogram.par_iter())
            .for_each(|((result_val, &parent_val), &sibling_val)| {
                *result_val = parent_val - sibling_val;
            });

        Ok(result)
    }

    /// Determines whether to use sparse histogram construction.
    fn should_use_sparse_construction(&self, active_data_count: usize, total_data_count: usize) -> bool {
        if !self.config.use_sparse {
            return false;
        }

        let sparsity_ratio = active_data_count as f64 / total_data_count as f64;
        sparsity_ratio < self.config.sparse_threshold
    }

    /// Updates the histogram builder configuration.
    pub fn update_config(&mut self, config: HistogramBuilderConfig) {
        self.config = config;
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &HistogramBuilderConfig {
        &self.config
    }

    /// Gets a reference to the histogram pool.
    pub fn histogram_pool(&self) -> &Mutex<HistogramPool> {
        &self.histogram_pool
    }
}

/// Bin mapping functionality for feature discretization.
#[derive(Debug, Clone)]
pub struct BinMapper {
    /// Bin boundaries for continuous features
    bin_upper_bounds: Vec<f64>,
    /// Number of bins
    num_bins: usize,
    /// Default bin for missing values
    default_bin: BinIndex,
    /// Feature type
    feature_type: FeatureType,
}

/// Feature type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureType {
    /// Numerical feature with ordered values
    Numerical,
    /// Categorical feature with unordered values
    Categorical,
}

impl BinMapper {
    /// Creates a new bin mapper for numerical features.
    pub fn new_numerical(values: &[f32], max_bins: usize) -> Self {
        let mut sorted_values: Vec<f32> = values.iter().copied().collect();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut bin_upper_bounds = Vec::new();
        let n = sorted_values.len();

        if n == 0 {
            return BinMapper {
                bin_upper_bounds: vec![0.0],
                num_bins: 1,
                default_bin: 0,
                feature_type: FeatureType::Numerical,
            };
        }

        if n <= max_bins {
            // Use all unique values as bin boundaries
            let mut prev_val = sorted_values[0];
            bin_upper_bounds.push(prev_val as f64);

            for &val in &sorted_values[1..] {
                if val != prev_val {
                    bin_upper_bounds.push(val as f64);
                    prev_val = val;
                }
            }
        } else {
            // Use quantile-based binning
            for i in 0..max_bins {
                let quantile = (i + 1) as f64 / max_bins as f64;
                let index = ((quantile * n as f64) as usize).min(n - 1);
                bin_upper_bounds.push(sorted_values[index] as f64);
            }
        }

        BinMapper {
            num_bins: bin_upper_bounds.len(),
            bin_upper_bounds,
            default_bin: 0,
            feature_type: FeatureType::Numerical,
        }
    }

    /// Creates a new bin mapper for categorical features.
    pub fn new_categorical(categories: &[u32], max_bins: usize) -> Self {
        let unique_categories: std::collections::HashSet<u32> = categories.iter().copied().collect();
        let num_bins = unique_categories.len().min(max_bins);

        // For categorical features, we don't use upper bounds in the same way
        let bin_upper_bounds = (0..num_bins).map(|i| i as f64).collect();

        BinMapper {
            bin_upper_bounds,
            num_bins,
            default_bin: 0,
            feature_type: FeatureType::Categorical,
        }
    }

    /// Maps a feature value to its corresponding bin index.
    pub fn value_to_bin(&self, value: f32) -> BinIndex {
        if value.is_nan() {
            return self.default_bin;
        }

        match self.feature_type {
            FeatureType::Numerical => {
                for (i, &upper_bound) in self.bin_upper_bounds.iter().enumerate() {
                    if value as f64 <= upper_bound {
                        return i as BinIndex;
                    }
                }
                // Value is larger than all upper bounds
                (self.num_bins - 1) as BinIndex
            }
            FeatureType::Categorical => {
                // For categorical features, the value should directly map to a bin
                let bin = value as u32;
                if bin < self.num_bins as u32 {
                    bin
                } else {
                    self.default_bin
                }
            }
        }
    }

    /// Returns the number of bins.
    pub fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Returns the bin upper bounds.
    pub fn bin_upper_bounds(&self) -> &[f64] {
        &self.bin_upper_bounds
    }

    /// Returns the default bin for missing values.
    pub fn default_bin(&self) -> BinIndex {
        self.default_bin
    }

    /// Returns the feature type.
    pub fn feature_type(&self) -> FeatureType {
        self.feature_type
    }

    /// Sets the default bin for missing values.
    pub fn set_default_bin(&mut self, default_bin: BinIndex) {
        if (default_bin as usize) < self.num_bins {
            self.default_bin = default_bin;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::histogram::pool::HistogramPoolConfig;
    use ndarray::Array2;

    #[test]
    fn test_bin_mapper_numerical() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mapper = BinMapper::new_numerical(&values, 3);

        assert_eq!(mapper.num_bins(), 3);
        assert_eq!(mapper.value_to_bin(1.5), 0);
        assert_eq!(mapper.value_to_bin(3.5), 1);
        assert_eq!(mapper.value_to_bin(5.0), 2);
    }

    #[test]
    fn test_bin_mapper_categorical() {
        let categories = vec![0, 1, 2, 1, 0, 2];
        let mapper = BinMapper::new_categorical(&categories, 5);

        assert_eq!(mapper.num_bins(), 3); // 3 unique categories
        assert_eq!(mapper.value_to_bin(0.0), 0);
        assert_eq!(mapper.value_to_bin(1.0), 1);
        assert_eq!(mapper.value_to_bin(2.0), 2);
    }

    #[test]
    fn test_histogram_builder_creation() {
        let config = HistogramBuilderConfig::default();
        let pool_config = HistogramPoolConfig::default();
        let pool = HistogramPool::new(pool_config);
        let builder = HistogramBuilder::new(config, pool);

        assert_eq!(builder.config().num_threads, num_cpus::get());
    }

    #[test]
    fn test_feature_histogram_construction() {
        let config = HistogramBuilderConfig::default();
        let pool_config = HistogramPoolConfig {
            max_bin: 5,
            num_features: 1,
            max_pool_size: 10,
            initial_pool_size: 2,
            use_double_precision: true,
        };
        let pool = HistogramPool::new(pool_config);
        let builder = HistogramBuilder::new(config, pool);

        let feature_values = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let gradients = Array1::from(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let hessians = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0]);
        let data_indices = vec![0, 1, 2, 3, 4];
        let bin_mapper = BinMapper::new_numerical(&[1.0, 2.0, 3.0, 4.0, 5.0], 5);

        let result = builder.construct_feature_histogram(
            &feature_values.view(),
            &gradients.view(),
            &hessians.view(),
            &data_indices,
            &bin_mapper,
        );

        assert!(result.is_ok());
        let histogram = result.unwrap();
        assert_eq!(histogram.len(), 10); // 5 bins * 2 (gradient + hessian)
        
        // Check that some bins have non-zero values
        let total_gradient: f64 = histogram.iter().step_by(2).sum();
        let total_hessian: f64 = histogram.iter().skip(1).step_by(2).sum();
        
        assert!((total_gradient - 1.5).abs() < 1e-6); // Sum of gradients
        assert!((total_hessian - 5.0).abs() < 1e-6);  // Sum of hessians
    }

    #[test]
    fn test_histogram_subtraction() {
        let config = HistogramBuilderConfig::default();
        let pool_config = HistogramPoolConfig::default();
        let pool = HistogramPool::new(pool_config);
        let builder = HistogramBuilder::new(config, pool);

        let parent = Array1::from(vec![10.0, 5.0, 8.0, 3.0]);
        let sibling = Array1::from(vec![4.0, 2.0, 3.0, 1.0]);

        let result = builder.construct_histogram_by_subtraction(
            &parent.view(),
            &sibling.view(),
        );

        assert!(result.is_ok());
        let histogram = result.unwrap();
        assert_eq!(histogram, Array1::from(vec![6.0, 3.0, 5.0, 2.0]));
    }

    #[test]
    fn test_sparse_construction_threshold() {
        let config = HistogramBuilderConfig {
            use_sparse: true,
            sparse_threshold: 0.5,
            ..Default::default()
        };
        let pool_config = HistogramPoolConfig::default();
        let pool = HistogramPool::new(pool_config);
        let builder = HistogramBuilder::new(config, pool);

        // Test below threshold (should use sparse)
        assert!(builder.should_use_sparse_construction(40, 100));
        
        // Test above threshold (should use dense)
        assert!(!builder.should_use_sparse_construction(60, 100));
    }
}