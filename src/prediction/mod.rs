//! Prediction pipeline module for Pure Rust LightGBM.
//!
//! This module provides comprehensive prediction functionality including:
//! - Core prediction engine and configuration
//! - SHAP value computation for feature importance
//! - Feature importance calculation
//! - Early stopping mechanisms for prediction
//! - Leaf index prediction for advanced analysis

use crate::core::error::{LightGBMError, Result};
use crate::core::types::*;
use ndarray::ArrayView1;

pub mod predictor;
pub mod shap;
pub mod feature_importance;
pub mod early_stopping;
pub mod leaf_index;

// Re-export main prediction functionality
pub use predictor::{PredictionConfig, Predictor, PredictorTrait};
pub use shap::{SHAPCalculator, SHAPConfig};
pub use feature_importance::{FeatureImportanceCalculator, ImportanceType};

/// Histogram pool for managing histogram allocation and reuse
#[derive(Debug)]
pub struct HistogramPool {
    /// Pool configuration
    // TODO: implement config usage in histogram pool operations
    config: crate::config::Config,
    /// Pool state
    // TODO: implement proper initialization state management
    initialized: bool,
    /// Available histogram indices
    available_indices: Vec<usize>,
    /// Next histogram index to allocate
    next_index: usize,
    /// Maximum number of histograms
    max_histograms: usize,
    /// Storage for histogram data (indexed by histogram_index)
    histograms: std::collections::HashMap<usize, Vec<f64>>,
}

impl HistogramPool {
    /// Create a new histogram pool
    pub fn new(config: &crate::config::Config) -> Result<Self> {
        let max_histograms = std::cmp::max(config.num_leaves * 2, 64); // reasonable default
        Ok(Self {
            config: config.clone(),
            initialized: true,
            available_indices: Vec::new(),
            next_index: 0,
            max_histograms,
            histograms: std::collections::HashMap::new(),
        })
    }

    /// Get histogram from pool
    pub fn get_histogram(&mut self) -> Result<usize> {
        // Try to reuse an available histogram first
        if let Some(index) = self.available_indices.pop() {
            return Ok(index);
        }

        // Allocate a new histogram if under the limit
        if self.next_index < self.max_histograms {
            let index = self.next_index;
            self.next_index += 1;
            Ok(index)
        } else {
            Err(LightGBMError::memory(
                "Histogram pool exhausted - no more histograms available",
            ))
        }
    }

    /// Release histogram to pool
    pub fn release_histogram(&mut self, index: usize) {
        // Return the histogram index to the available pool for reuse
        if index < self.next_index && !self.available_indices.contains(&index) {
            self.available_indices.push(index);
            // Clear the histogram data to free memory
            self.histograms.remove(&index);
        }
    }

    /// Get histogram data from pool
    pub fn get_histogram_data(&self, histogram_index: usize) -> Option<&Vec<f64>> {
        self.histograms.get(&histogram_index)
    }

    /// Construct histogram
    pub fn construct_histogram(
        &mut self,
        histogram_index: usize,
        dataset: &crate::dataset::Dataset,
        gradients: &ArrayView1<'_, Score>,
        hessians: &ArrayView1<'_, Score>,
        data_indices: &[i32],
        feature_index: usize,
    ) -> Result<()> {
        // Validate histogram index
        if histogram_index >= self.next_index {
            return Err(LightGBMError::invalid_parameter(
                "histogram_index",
                histogram_index.to_string(),
                "Histogram index must be less than next_index",
            ));
        }

        // Validate feature index
        if feature_index >= dataset.num_features() {
            return Err(LightGBMError::invalid_parameter(
                "feature_index",
                feature_index.to_string(),
                "Feature index must be less than num_features",
            ));
        }

        // Get the bin mapper for this feature
        let bin_mapper = dataset.bin_mapper(feature_index).ok_or_else(|| {
            LightGBMError::invalid_parameter(
                "feature_index",
                feature_index.to_string(),
                "No bin mapper found for feature index",
            )
        })?;
        let num_bins = bin_mapper.num_bins;
        
        // Create histogram array: gradient and hessian per bin
        let histogram_size = num_bins * 2;
        let mut histogram = vec![0.0; histogram_size];
        
        // Get feature matrix to access feature values
        let features = dataset.features();
        
        // Construct histogram by accumulating gradients and hessians for each bin
        for &data_idx in data_indices {
            let idx = data_idx as usize;
            
            // Validate data index
            if idx >= dataset.num_data() {
                continue; // Skip invalid indices
            }
            
            // Get feature value for this data point
            let feature_value = features[[idx, feature_index]];
            
            // Convert feature value to bin index
            let bin = bin_mapper.value_to_bin(feature_value);
            let bin_idx = bin as usize;
            
            // Validate bin index
            if bin_idx >= num_bins {
                continue; // Skip invalid bins
            }
            
            // Get gradient and hessian for this data point
            let gradient = gradients[idx] as crate::core::types::Hist;
            let hessian = hessians[idx] as crate::core::types::Hist;
            
            // Accumulate into histogram (gradient at even indices, hessian at odd indices)
            histogram[bin_idx * 2] += gradient;
            histogram[bin_idx * 2 + 1] += hessian;
        }
        
        // Store the histogram data in the pool for later use in split finding
        self.histograms.insert(histogram_index, histogram);
        
        Ok(())
    }
}

/// Split finder placeholder type
#[derive(Debug)]
pub struct SplitFinder {
    /// Configuration
    config: crate::config::Config,
}

impl SplitFinder {
    /// Create a new split finder
    pub fn new(config: &crate::config::Config) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    /// Find best split using histogram-based evaluation
    pub fn find_best_split(
        &self,
        histogram_index: usize,
        histogram_pool: &mut HistogramPool,
        dataset: &crate::dataset::Dataset,
        feature_index: usize,
        data_indices: &[i32],
        gradients: &ArrayView1<'_, Score>,
        hessians: &ArrayView1<'_, Score>,
    ) -> Result<SplitInfo> {
        // Validate inputs
        if data_indices.is_empty() {
            return Err(LightGBMError::invalid_parameter(
                "data_indices",
                "empty".to_string(),
                "Data indices cannot be empty",
            ));
        }

        // Calculate total gradient and hessian for this node
        let mut total_gradient = 0.0;
        let mut total_hessian = 0.0;
        
        for &data_idx in data_indices {
            let idx = data_idx as usize;
            if idx < gradients.len() && idx < hessians.len() {
                total_gradient += gradients[idx] as f64;
                total_hessian += hessians[idx] as f64;
            }
        }

        // Check minimum hessian constraint
        let min_sum_hessian = self.config.min_sum_hessian_in_leaf;
        if total_hessian < min_sum_hessian {
            return Ok(SplitInfo::new()); // Return invalid split
        }

        // Use histogram pool for efficient memory management
        histogram_pool.construct_histogram(
            histogram_index,
            dataset,
            gradients,
            hessians,
            data_indices,
            feature_index,
        )?;

        // Get bin mapper for this feature to get proper histogram data
        let bin_mapper = dataset.bin_mapper(feature_index).ok_or_else(|| {
            LightGBMError::invalid_parameter(
                "feature_index",
                feature_index.to_string(),
                "No bin mapper found for feature index",
            )
        })?;
        let max_bins = bin_mapper.num_bins;

        // Get histogram data from pool
        let histogram = histogram_pool.get_histogram_data(histogram_index).ok_or_else(|| {
            LightGBMError::invalid_parameter(
                "histogram_index",
                histogram_index.to_string(),
                "Histogram not found in pool",
            )
        })?;

        // Find the best split by evaluating all possible split points
        let mut best_split = SplitInfo::new();
        let lambda_l1 = self.config.lambda_l1;
        let lambda_l2 = self.config.lambda_l2;
        let min_data_in_leaf = self.config.min_data_in_leaf as usize;

        // Evaluate each possible split point between bins
        for split_bin in 1..max_bins {
            let mut left_gradient = 0.0;
            let mut left_hessian = 0.0;
            let mut left_count = 0;

            // Calculate left side statistics
            for bin in 0..split_bin {
                if bin * 2 + 1 < histogram.len() {
                    left_gradient += histogram[bin * 2];
                    left_hessian += histogram[bin * 2 + 1];
                    // Approximate count based on hessian (assuming hessian ~= count for simple cases)
                    left_count += histogram[bin * 2 + 1].round() as usize;
                }
            }

            // Calculate right side statistics  
            let right_gradient = total_gradient - left_gradient;
            let right_hessian = total_hessian - left_hessian;
            let right_count = data_indices.len() - left_count;

            // Check constraints
            if left_count < min_data_in_leaf || right_count < min_data_in_leaf {
                continue;
            }
            if left_hessian < min_sum_hessian || right_hessian < min_sum_hessian {
                continue;
            }

            // Calculate gain using LightGBM formula with regularization
            let left_gain = Self::calculate_leaf_gain(left_gradient, left_hessian, lambda_l1, lambda_l2);
            let right_gain = Self::calculate_leaf_gain(right_gradient, right_hessian, lambda_l1, lambda_l2);
            let parent_gain = Self::calculate_leaf_gain(total_gradient, total_hessian, lambda_l1, lambda_l2);
            
            let split_gain = left_gain + right_gain - parent_gain;

            // Update best split if this is better
            if split_gain > best_split.gain {
                best_split = SplitInfo {
                    feature: feature_index,
                    threshold: split_bin as f64 / max_bins as f64, // Normalized threshold
                    gain: split_gain,
                    left_count,
                    right_count,
                };
            }
        }

        Ok(best_split)
    }

    /// Calculate leaf gain with L1/L2 regularization
    fn calculate_leaf_gain(gradient: f64, hessian: f64, lambda_l1: f64, lambda_l2: f64) -> f64 {
        if hessian <= 0.0 {
            return 0.0;
        }
        
        // Apply L1 regularization (soft thresholding)
        let gradient_abs = gradient.abs();
        if gradient_abs <= lambda_l1 {
            return 0.0;
        }
        
        let regularized_gradient = if gradient > 0.0 {
            gradient - lambda_l1
        } else {
            gradient + lambda_l1
        };
        
        // Apply L2 regularization
        let regularized_hessian = hessian + lambda_l2;
        
        // Calculate gain: -0.5 * gradient^2 / hessian (negative because we minimize loss)
        0.5 * (regularized_gradient * regularized_gradient) / regularized_hessian
    }
}

/// Split information
#[derive(Debug, Clone)]
pub struct SplitInfo {
    /// Feature index
    pub feature: usize,
    /// Split threshold
    pub threshold: f64,
    /// Split gain
    pub gain: f64,
    /// Left child count
    pub left_count: usize,
    /// Right child count
    pub right_count: usize,
}

impl SplitInfo {
    /// Create a new split info
    pub fn new() -> Self {
        Self {
            feature: 0,
            threshold: 0.0,
            gain: 0.0,
            left_count: 0,
            right_count: 0,
        }
    }

    /// Check if split is valid
    pub fn is_valid(&self) -> bool {
        self.gain > 0.0 && self.left_count > 0 && self.right_count > 0
    }
}

impl Default for SplitInfo {
    fn default() -> Self {
        Self::new()
    }
}