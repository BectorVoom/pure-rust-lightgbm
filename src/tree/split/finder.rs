//! Split finding implementation for the Pure Rust LightGBM framework.
//!
//! This module provides efficient split finding algorithms using histogram-based
//! methods to identify optimal feature thresholds for tree node splitting.

use crate::core::types::{BinIndex, DataSize, FeatureIndex, Hist, Score};
use ndarray::{Array1, ArrayView1};
use std::cmp::Ordering;

/// Information about a potential split point.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitInfo {
    /// Feature index for the split
    pub feature: FeatureIndex,
    /// Bin threshold for the split
    pub threshold_bin: BinIndex,
    /// Actual threshold value
    pub threshold_value: f64,
    /// Split gain (improvement in loss function)
    pub gain: f64,
    /// Left child statistics
    pub left_sum_gradient: f64,
    pub left_sum_hessian: f64,
    pub left_count: DataSize,
    /// Right child statistics
    pub right_sum_gradient: f64,
    pub right_sum_hessian: f64,
    pub right_count: DataSize,
    /// Output values for children
    pub left_output: Score,
    pub right_output: Score,
    /// Default direction for missing values (true = left, false = right)
    pub default_left: bool,
}

impl SplitInfo {
    /// Creates a new empty split info.
    pub fn new() -> Self {
        SplitInfo {
            feature: 0,
            threshold_bin: 0,
            threshold_value: 0.0,
            gain: 0.0,
            left_sum_gradient: 0.0,
            left_sum_hessian: 0.0,
            left_count: 0,
            right_sum_gradient: 0.0,
            right_sum_hessian: 0.0,
            right_count: 0,
            left_output: 0.0,
            right_output: 0.0,
            default_left: false,
        }
    }

    /// Returns true if this split is valid and beneficial.
    pub fn is_valid(&self) -> bool {
        self.gain > 0.0 && self.left_count > 0 && self.right_count > 0
    }

    /// Calculates the leaf output values using regularization parameters.
    pub fn calculate_outputs(&mut self, lambda_l1: f64, lambda_l2: f64) {
        self.left_output = Self::calculate_leaf_output(
            self.left_sum_gradient,
            self.left_sum_hessian,
            lambda_l1,
            lambda_l2,
        );
        self.right_output = Self::calculate_leaf_output(
            self.right_sum_gradient,
            self.right_sum_hessian,
            lambda_l1,
            lambda_l2,
        );
    }

    /// Calculates optimal leaf output for given statistics.
    fn calculate_leaf_output(
        sum_gradient: f64,
        sum_hessian: f64,
        lambda_l1: f64,
        lambda_l2: f64,
    ) -> Score {
        if sum_hessian <= 0.0 {
            return 0.0;
        }

        let numerator = if lambda_l1 > 0.0 {
            if sum_gradient > lambda_l1 {
                sum_gradient - lambda_l1
            } else if sum_gradient < -lambda_l1 {
                sum_gradient + lambda_l1
            } else {
                0.0
            }
        } else {
            sum_gradient
        };

        (-numerator / (sum_hessian + lambda_l2)) as Score
    }
}

impl Default for SplitInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for split finding operations.
#[derive(Debug, Clone)]
pub struct SplitFinderConfig {
    /// Minimum number of data points required in each leaf
    pub min_data_in_leaf: DataSize,
    /// Minimum sum of hessians required in each leaf
    pub min_sum_hessian_in_leaf: f64,
    /// L1 regularization parameter
    pub lambda_l1: f64,
    /// L2 regularization parameter
    pub lambda_l2: f64,
    /// Minimum gain required for a split
    pub min_split_gain: f64,
    /// Maximum number of bins for histogram construction
    pub max_bin: usize,
}

impl Default for SplitFinderConfig {
    fn default() -> Self {
        SplitFinderConfig {
            min_data_in_leaf: 20,
            min_sum_hessian_in_leaf: 1e-3,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_split_gain: 0.0,
            max_bin: 255,
        }
    }
}

/// Split finder for identifying optimal split points using histogram data.
pub struct SplitFinder {
    config: SplitFinderConfig,
}

impl SplitFinder {
    /// Creates a new split finder with the given configuration.
    pub fn new(config: SplitFinderConfig) -> Self {
        SplitFinder { config }
    }

    /// Finds the best split for a feature using histogram data.
    ///
    /// The histogram should contain gradient and hessian pairs for each bin:
    /// [grad_bin0, hess_bin0, grad_bin1, hess_bin1, ...]
    pub fn find_best_split_for_feature(
        &self,
        feature_index: FeatureIndex,
        histogram: &ArrayView1<Hist>,
        total_sum_gradient: f64,
        total_sum_hessian: f64,
        total_count: DataSize,
        bin_boundaries: &[f64],
    ) -> Option<SplitInfo> {
        let num_bins = histogram.len() / 2; // Each bin has gradient and hessian
        if num_bins < 2 {
            return None;
        }

        let mut best_split = SplitInfo::new();
        best_split.feature = feature_index;

        let mut left_sum_gradient = 0.0;
        let mut left_sum_hessian = 0.0;
        let mut left_count = 0;

        // Try each possible split point
        for bin in 0..num_bins - 1 {
            // Accumulate left side statistics
            let grad_idx = bin * 2;
            let hess_idx = bin * 2 + 1;
            
            left_sum_gradient += histogram[grad_idx];
            left_sum_hessian += histogram[hess_idx];
            left_count += (histogram[hess_idx] / (histogram[hess_idx] / histogram[grad_idx].abs().max(1e-6))).round() as DataSize;

            // Calculate right side statistics
            let right_sum_gradient = total_sum_gradient - left_sum_gradient;
            let right_sum_hessian = total_sum_hessian - left_sum_hessian;
            let right_count = total_count - left_count;

            // Check minimum data constraints
            if left_count < self.config.min_data_in_leaf
                || right_count < self.config.min_data_in_leaf
                || left_sum_hessian < self.config.min_sum_hessian_in_leaf
                || right_sum_hessian < self.config.min_sum_hessian_in_leaf
            {
                continue;
            }

            // Calculate split gain
            let gain = self.calculate_split_gain(
                total_sum_gradient,
                total_sum_hessian,
                left_sum_gradient,
                left_sum_hessian,
                right_sum_gradient,
                right_sum_hessian,
            );

            // Check if this split is better
            if gain > best_split.gain && gain > self.config.min_split_gain {
                best_split.threshold_bin = bin as BinIndex;
                best_split.threshold_value = if bin < bin_boundaries.len() {
                    bin_boundaries[bin]
                } else {
                    bin as f64
                };
                best_split.gain = gain;
                best_split.left_sum_gradient = left_sum_gradient;
                best_split.left_sum_hessian = left_sum_hessian;
                best_split.left_count = left_count;
                best_split.right_sum_gradient = right_sum_gradient;
                best_split.right_sum_hessian = right_sum_hessian;
                best_split.right_count = right_count;

                // Determine default direction for missing values
                // Go to the side with larger hessian sum (more confident)
                best_split.default_left = left_sum_hessian >= right_sum_hessian;
            }
        }

        if best_split.is_valid() {
            best_split.calculate_outputs(self.config.lambda_l1, self.config.lambda_l2);
            Some(best_split)
        } else {
            None
        }
    }

    /// Finds the best split among multiple features.
    pub fn find_best_split(
        &self,
        histograms: &[ArrayView1<Hist>],
        feature_indices: &[FeatureIndex],
        total_sum_gradient: f64,
        total_sum_hessian: f64,
        total_count: DataSize,
        bin_boundaries: &[Vec<f64>],
    ) -> Option<SplitInfo> {
        let mut best_split: Option<SplitInfo> = None;

        for (i, &feature_idx) in feature_indices.iter().enumerate() {
            if i >= histograms.len() || feature_idx >= bin_boundaries.len() {
                continue;
            }

            if let Some(split) = self.find_best_split_for_feature(
                feature_idx,
                &histograms[i],
                total_sum_gradient,
                total_sum_hessian,
                total_count,
                &bin_boundaries[feature_idx],
            ) {
                match &best_split {
                    None => best_split = Some(split),
                    Some(current_best) => {
                        if split.gain > current_best.gain {
                            best_split = Some(split);
                        }
                    }
                }
            }
        }

        best_split
    }

    /// Calculates the gain for a potential split.
    fn calculate_split_gain(
        &self,
        total_sum_gradient: f64,
        total_sum_hessian: f64,
        left_sum_gradient: f64,
        left_sum_hessian: f64,
        right_sum_gradient: f64,
        right_sum_hessian: f64,
    ) -> f64 {
        let parent_gain = self.calculate_leaf_gain(total_sum_gradient, total_sum_hessian);
        let left_gain = self.calculate_leaf_gain(left_sum_gradient, left_sum_hessian);
        let right_gain = self.calculate_leaf_gain(right_sum_gradient, right_sum_hessian);

        left_gain + right_gain - parent_gain
    }

    /// Calculates the gain for a leaf node.
    fn calculate_leaf_gain(&self, sum_gradient: f64, sum_hessian: f64) -> f64 {
        if sum_hessian <= 0.0 {
            return 0.0;
        }

        let numerator = if self.config.lambda_l1 > 0.0 {
            if sum_gradient > self.config.lambda_l1 {
                (sum_gradient - self.config.lambda_l1).powi(2)
            } else if sum_gradient < -self.config.lambda_l1 {
                (sum_gradient + self.config.lambda_l1).powi(2)
            } else {
                0.0
            }
        } else {
            sum_gradient.powi(2)
        };

        numerator / (2.0 * (sum_hessian + self.config.lambda_l2))
    }

    /// Updates the configuration.
    pub fn update_config(&mut self, config: SplitFinderConfig) {
        self.config = config;
    }

    /// Returns a reference to the current configuration.
    pub fn config(&self) -> &SplitFinderConfig {
        &self.config
    }
}

impl PartialOrd for SplitInfo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.gain.partial_cmp(&other.gain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_split_info_creation() {
        let split = SplitInfo::new();
        assert_eq!(split.gain, 0.0);
        assert!(!split.is_valid());
    }

    #[test]
    fn test_split_info_validity() {
        let mut split = SplitInfo::new();
        split.gain = 1.0;
        split.left_count = 10;
        split.right_count = 10;
        assert!(split.is_valid());
    }

    #[test]
    fn test_calculate_outputs() {
        let mut split = SplitInfo::new();
        split.left_sum_gradient = -10.0;
        split.left_sum_hessian = 5.0;
        split.right_sum_gradient = 5.0;
        split.right_sum_hessian = 3.0;

        split.calculate_outputs(0.0, 0.1);

        assert!((split.left_output - 2.0).abs() < 1e-6);
        assert!((split.right_output - (-1.6129)).abs() < 1e-3);
    }

    #[test]
    fn test_split_finder_creation() {
        let config = SplitFinderConfig::default();
        let finder = SplitFinder::new(config);
        assert_eq!(finder.config.min_data_in_leaf, 20);
    }

    #[test]
    fn test_find_best_split_simple() {
        let config = SplitFinderConfig {
            min_data_in_leaf: 1,
            min_sum_hessian_in_leaf: 0.1,
            lambda_l1: 0.0,
            lambda_l2: 0.1,
            min_split_gain: 0.0,
            max_bin: 10,
        };

        let finder = SplitFinder::new(config);

        // Create a simple histogram: [grad0, hess0, grad1, hess1, ...]
        let histogram = Array1::from(vec![
            -10.0, 5.0,  // bin 0: gradient=-10, hessian=5
            -5.0, 3.0,   // bin 1: gradient=-5, hessian=3
            5.0, 2.0,    // bin 2: gradient=5, hessian=2
            10.0, 4.0,   // bin 3: gradient=10, hessian=4
        ]);

        let bin_boundaries = vec![1.0, 2.0, 3.0, 4.0];
        let total_sum_gradient = 0.0;
        let total_sum_hessian = 14.0;
        let total_count = 100;

        let split = finder.find_best_split_for_feature(
            0,
            &histogram.view(),
            total_sum_gradient,
            total_sum_hessian,
            total_count,
            &bin_boundaries,
        );

        assert!(split.is_some());
        let split = split.unwrap();
        assert!(split.is_valid());
        assert!(split.gain > 0.0);
    }

    #[test]
    fn test_calculate_leaf_gain() {
        let config = SplitFinderConfig::default();
        let finder = SplitFinder::new(config);

        let gain = finder.calculate_leaf_gain(10.0, 5.0);
        assert_eq!(gain, 10.0); // 10^2 / (2 * 5) = 100 / 10 = 10
    }

    #[test]
    fn test_calculate_split_gain() {
        let config = SplitFinderConfig::default();
        let finder = SplitFinder::new(config);

        let gain = finder.calculate_split_gain(
            0.0, 10.0,  // parent: grad=0, hess=10
            -5.0, 5.0,  // left: grad=-5, hess=5
            5.0, 5.0,   // right: grad=5, hess=5
        );

        // left_gain = 25/10 = 2.5, right_gain = 25/10 = 2.5, parent_gain = 0
        // total_gain = 2.5 + 2.5 - 0 = 5.0
        assert_eq!(gain, 5.0);
    }
}