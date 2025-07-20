//! Prediction module for Pure Rust LightGBM.
//!
//! This module provides prediction configuration, predictor implementations,
//! and prediction pipeline functionality. It now includes the new modular
//! prediction system with SHAP and feature importance calculations.

// Re-export the new prediction module structure
pub mod predictor;
pub mod shap;
pub mod feature_importance;
pub mod early_stopping;
pub mod leaf_index;

// Re-export main types for backward compatibility
pub use predictor::{PredictionConfig, Predictor, PredictorTrait};
pub use shap::{SHAPCalculator, SHAPConfig};
pub use feature_importance::{FeatureImportanceCalculator, ImportanceType};

// Keep the original functionality for components that haven't been moved
use crate::core::error::{LightGBMError, Result};
use crate::core::types::*;
use ndarray::ArrayView1;
use serde::{Deserialize, Serialize};

/// Histogram pool for managing histogram allocation and reuse
#[derive(Debug)]
pub struct HistogramPool {
    /// Pool configuration
    config: crate::config::Config,
    /// Pool state
    initialized: bool,
    /// Available histogram indices
    available_indices: Vec<usize>,
    /// Next histogram index to allocate
    next_index: usize,
    /// Maximum number of histograms
    max_histograms: usize,
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
        }
    }

    /// Construct histogram
    pub fn construct_histogram(
        &mut self,
        histogram_index: usize,
        dataset: &crate::dataset::Dataset,
        gradients: &ndarray::ArrayView1<'_, Score>,
        hessians: &ndarray::ArrayView1<'_, Score>,
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

        // For now, this is a placeholder that simulates successful histogram construction
        // In a complete implementation, this would:
        // 1. Extract feature values for the given data indices
        // 2. Bin the values according to the dataset's binning scheme
        // 3. Accumulate gradients and hessians for each bin
        // 4. Store the histogram data for later use in split finding

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

    /// Find best split
    pub fn find_best_split(
        &self,
        histogram_index: usize,

        histogram_pool: &HistogramPool,
        feature_index: usize,
        data_indices: &[i32],
        gradients: &ndarray::ArrayView1<'_, Score>,
        hessians: &ndarray::ArrayView1<'_, Score>,
    ) -> Result<SplitInfo> {
        // For now, return a dummy split that simulates finding a reasonable split
        // In a complete implementation, this would:
        // 1. Use the histogram to evaluate all possible split points
        // 2. Calculate gain for each split using gradient and hessian information
        // 3. Return the split with the highest gain

        let split_threshold = 0.5; // Dummy threshold
        let split_gain = 1.0; // Dummy gain
        let left_count = data_indices.len() / 2;
        let right_count = data_indices.len() - left_count;

        Ok(SplitInfo {
            feature: feature_index,
            threshold: split_threshold,
            gain: split_gain,
            left_count,
            right_count,
        })
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
