//! Split evaluation utilities for the Pure Rust LightGBM framework.
//!
//! This module provides advanced split evaluation methods including
//! monotonic constraints, categorical feature handling, and split validation.

use crate::core::types::{BinIndex, DataSize, FeatureIndex, Score};
use crate::tree::split::finder::{SplitInfo, SplitFinderConfig};
use std::collections::HashMap;

/// Monotonic constraint types for features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonotonicConstraint {
    /// No constraint
    None,
    /// Increasing constraint (split threshold increases with feature value)
    Increasing,
    /// Decreasing constraint (split threshold decreases with feature value)
    Decreasing,
}

impl Default for MonotonicConstraint {
    fn default() -> Self {
        MonotonicConstraint::None
    }
}

/// Configuration for split evaluation with advanced features.
#[derive(Debug, Clone)]
pub struct SplitEvaluatorConfig {
    /// Base split finder configuration
    pub base_config: SplitFinderConfig,
    /// Monotonic constraints for features
    pub monotonic_constraints: HashMap<FeatureIndex, MonotonicConstraint>,
    /// Maximum number of categorical bins
    pub max_cat_to_onehot: usize,
    /// Feature interaction constraints
    pub feature_groups: Vec<Vec<FeatureIndex>>,
    /// Penalty for tree complexity
    pub complexity_penalty: f64,
}

impl Default for SplitEvaluatorConfig {
    fn default() -> Self {
        SplitEvaluatorConfig {
            base_config: SplitFinderConfig::default(),
            monotonic_constraints: HashMap::new(),
            max_cat_to_onehot: 4,
            feature_groups: Vec::new(),
            complexity_penalty: 0.0,
        }
    }
}

/// Advanced split evaluator with support for constraints and categorical features.
pub struct SplitEvaluator {
    config: SplitEvaluatorConfig,
}

impl SplitEvaluator {
    /// Creates a new split evaluator with the given configuration.
    pub fn new(config: SplitEvaluatorConfig) -> Self {
        SplitEvaluator { config }
    }

    /// Evaluates and validates a potential split.
    pub fn evaluate_split(&self, split: &SplitInfo, parent_output: Score) -> SplitEvaluationResult {
        let mut result = SplitEvaluationResult::new();
        
        // Basic validity checks
        if !split.is_valid() {
            result.is_valid = false;
            result.rejection_reason = Some("Split is not valid".to_string());
            return result;
        }

        // Check minimum data constraints
        if split.left_count < self.config.base_config.min_data_in_leaf
            || split.right_count < self.config.base_config.min_data_in_leaf
        {
            result.is_valid = false;
            result.rejection_reason = Some("Insufficient data in leaf nodes".to_string());
            return result;
        }

        // Check minimum hessian constraints
        if split.left_sum_hessian < self.config.base_config.min_sum_hessian_in_leaf
            || split.right_sum_hessian < self.config.base_config.min_sum_hessian_in_leaf
        {
            result.is_valid = false;
            result.rejection_reason = Some("Insufficient hessian sum in leaf nodes".to_string());
            return result;
        }

        // Check minimum gain requirement
        if split.gain < self.config.base_config.min_split_gain {
            result.is_valid = false;
            result.rejection_reason = Some("Split gain below threshold".to_string());
            return result;
        }

        // Check monotonic constraints
        if !self.check_monotonic_constraint(split, parent_output) {
            result.is_valid = false;
            result.rejection_reason = Some("Violates monotonic constraint".to_string());
            return result;
        }

        // Apply complexity penalty
        let penalized_gain = split.gain - self.config.complexity_penalty;
        if penalized_gain <= 0.0 {
            result.is_valid = false;
            result.rejection_reason = Some("Gain too low after complexity penalty".to_string());
            return result;
        }

        // Calculate additional metrics
        result.is_valid = true;
        result.adjusted_gain = penalized_gain;
        result.confidence_score = self.calculate_confidence_score(split);
        result.balance_score = self.calculate_balance_score(split);
        result.improvement_ratio = split.gain / (split.gain + 1.0); // Normalize to [0,1)

        result
    }

    /// Checks if a split satisfies monotonic constraints.
    fn check_monotonic_constraint(&self, split: &SplitInfo, parent_output: Score) -> bool {
        if let Some(&constraint) = self.config.monotonic_constraints.get(&split.feature) {
            match constraint {
                MonotonicConstraint::None => true,
                MonotonicConstraint::Increasing => {
                    // Left child (lower values) should have lower or equal output
                    split.left_output <= split.right_output
                }
                MonotonicConstraint::Decreasing => {
                    // Left child (lower values) should have higher or equal output
                    split.left_output >= split.right_output
                }
            }
        } else {
            true // No constraint specified
        }
    }

    /// Calculates a confidence score for the split based on data distribution.
    fn calculate_confidence_score(&self, split: &SplitInfo) -> f64 {
        let total_count = split.left_count + split.right_count;
        if total_count == 0 {
            return 0.0;
        }

        let left_ratio = split.left_count as f64 / total_count as f64;
        let right_ratio = split.right_count as f64 / total_count as f64;

        // Higher confidence when data is more evenly distributed
        let balance_factor = 1.0 - (left_ratio - right_ratio).abs();
        
        // Higher confidence with more total data points
        let data_factor = (total_count as f64).ln() / 10.0; // Logarithmic scaling
        
        // Higher confidence with higher hessian sums (more certainty)
        let hessian_factor = (split.left_sum_hessian + split.right_sum_hessian) / 100.0;

        (balance_factor * data_factor * hessian_factor).min(1.0).max(0.0)
    }

    /// Calculates how balanced the split is.
    fn calculate_balance_score(&self, split: &SplitInfo) -> f64 {
        let total_count = split.left_count + split.right_count;
        if total_count == 0 {
            return 0.0;
        }

        let left_ratio = split.left_count as f64 / total_count as f64;
        let right_ratio = split.right_count as f64 / total_count as f64;

        // Perfect balance (0.5, 0.5) gives score of 1.0
        // Extreme imbalance (1.0, 0.0) or (0.0, 1.0) gives score of 0.0
        1.0 - 2.0 * (left_ratio - 0.5).abs()
    }

    /// Compares two splits and returns the better one.
    pub fn compare_splits(&self, split_a: &SplitInfo, split_b: &SplitInfo) -> std::cmp::Ordering {
        // Primary comparison: gain
        let gain_cmp = split_a.gain.partial_cmp(&split_b.gain)
            .unwrap_or(std::cmp::Ordering::Equal);

        if gain_cmp != std::cmp::Ordering::Equal {
            return gain_cmp;
        }

        // Secondary comparison: balance (prefer more balanced splits)
        let balance_a = self.calculate_balance_score(split_a);
        let balance_b = self.calculate_balance_score(split_b);
        let balance_cmp = balance_a.partial_cmp(&balance_b)
            .unwrap_or(std::cmp::Ordering::Equal);

        if balance_cmp != std::cmp::Ordering::Equal {
            return balance_cmp;
        }

        // Tertiary comparison: confidence
        let conf_a = self.calculate_confidence_score(split_a);
        let conf_b = self.calculate_confidence_score(split_b);
        conf_a.partial_cmp(&conf_b).unwrap_or(std::cmp::Ordering::Equal)
    }

    /// Validates that a feature can be used for splitting.
    pub fn can_split_feature(&self, feature_index: FeatureIndex, used_features: &[FeatureIndex]) -> bool {
        // Check feature interaction constraints
        for group in &self.config.feature_groups {
            if group.contains(&feature_index) {
                // Check if any other feature from the same group is already used
                for &used_feature in used_features {
                    if group.contains(&used_feature) && used_feature != feature_index {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Filters splits based on evaluation criteria.
    pub fn filter_splits(&self, splits: Vec<SplitInfo>, parent_output: Score) -> Vec<SplitInfo> {
        splits
            .into_iter()
            .filter_map(|split| {
                let evaluation = self.evaluate_split(&split, parent_output);
                if evaluation.is_valid {
                    Some(split)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &SplitEvaluatorConfig {
        &self.config
    }

    /// Updates the configuration.
    pub fn update_config(&mut self, config: SplitEvaluatorConfig) {
        self.config = config;
    }
}

/// Result of split evaluation with detailed metrics.
#[derive(Debug, Clone)]
pub struct SplitEvaluationResult {
    /// Whether the split is valid and should be considered
    pub is_valid: bool,
    /// Reason for rejection if invalid
    pub rejection_reason: Option<String>,
    /// Gain adjusted for complexity penalty
    pub adjusted_gain: f64,
    /// Confidence score in the split quality
    pub confidence_score: f64,
    /// How balanced the split is (0.0 to 1.0)
    pub balance_score: f64,
    /// Improvement ratio relative to no split
    pub improvement_ratio: f64,
}

impl SplitEvaluationResult {
    fn new() -> Self {
        SplitEvaluationResult {
            is_valid: false,
            rejection_reason: None,
            adjusted_gain: 0.0,
            confidence_score: 0.0,
            balance_score: 0.0,
            improvement_ratio: 0.0,
        }
    }
}

/// Utility functions for categorical feature handling.
pub struct CategoricalSplitHandler;

impl CategoricalSplitHandler {
    /// Finds the best binary split for a categorical feature.
    pub fn find_best_categorical_split(
        gradients: &[f64],
        hessians: &[f64],
        categories: &[u32],
        config: &SplitFinderConfig,
    ) -> Option<SplitInfo> {
        let unique_categories: std::collections::HashSet<u32> = categories.iter().copied().collect();
        
        if unique_categories.len() <= 1 {
            return None;
        }

        // For small number of categories, try all possible binary splits
        if unique_categories.len() <= config.max_bin {
            Self::exhaustive_categorical_split(gradients, hessians, categories, config)
        } else {
            // For large number of categories, use heuristic approach
            Self::heuristic_categorical_split(gradients, hessians, categories, config)
        }
    }

    fn exhaustive_categorical_split(
        gradients: &[f64],
        hessians: &[f64],
        categories: &[u32],
        config: &SplitFinderConfig,
    ) -> Option<SplitInfo> {
        let unique_categories: Vec<u32> = categories.iter().copied().collect::<std::collections::HashSet<_>>().into_iter().collect();
        let n_categories = unique_categories.len();
        
        if n_categories >= 64 {
            return None; // Too many combinations to try
        }

        let mut best_split = SplitInfo::new();

        // Try all possible binary partitions (2^n - 2 combinations, excluding empty sets)
        for mask in 1..(1 << n_categories) - 1 {
            let mut left_gradient = 0.0;
            let mut left_hessian = 0.0;
            let mut left_count = 0;

            for (i, &category) in unique_categories.iter().enumerate() {
                if (mask >> i) & 1 == 1 {
                    // Category goes to left child
                    for (j, &cat) in categories.iter().enumerate() {
                        if cat == category {
                            left_gradient += gradients[j];
                            left_hessian += hessians[j];
                            left_count += 1;
                        }
                    }
                }
            }

            let total_gradient: f64 = gradients.iter().sum();
            let total_hessian: f64 = hessians.iter().sum();
            let total_count = categories.len() as DataSize;

            let right_gradient = total_gradient - left_gradient;
            let right_hessian = total_hessian - left_hessian;
            let right_count = total_count - left_count;

            // Check constraints
            if left_count < config.min_data_in_leaf
                || right_count < config.min_data_in_leaf
                || left_hessian < config.min_sum_hessian_in_leaf
                || right_hessian < config.min_sum_hessian_in_leaf
            {
                continue;
            }

            // Calculate gain
            let gain = Self::calculate_categorical_gain(
                total_gradient, total_hessian,
                left_gradient, left_hessian,
                right_gradient, right_hessian,
                config,
            );

            if gain > best_split.gain {
                best_split.gain = gain;
                best_split.left_sum_gradient = left_gradient;
                best_split.left_sum_hessian = left_hessian;
                best_split.left_count = left_count;
                best_split.right_sum_gradient = right_gradient;
                best_split.right_sum_hessian = right_hessian;
                best_split.right_count = right_count;
                best_split.threshold_bin = mask as BinIndex; // Store the mask as threshold
            }
        }

        if best_split.is_valid() {
            Some(best_split)
        } else {
            None
        }
    }

    fn heuristic_categorical_split(
        gradients: &[f64],
        hessians: &[f64],
        categories: &[u32],
        config: &SplitFinderConfig,
    ) -> Option<SplitInfo> {
        // Use gradient/hessian ratio to order categories
        let mut category_stats: HashMap<u32, (f64, f64, usize)> = HashMap::new();

        for (i, &category) in categories.iter().enumerate() {
            let entry = category_stats.entry(category).or_insert((0.0, 0.0, 0));
            entry.0 += gradients[i];
            entry.1 += hessians[i];
            entry.2 += 1;
        }

        // Sort categories by gradient/hessian ratio
        let mut sorted_categories: Vec<_> = category_stats.into_iter().collect();
        sorted_categories.sort_by(|a, b| {
            let ratio_a = if a.1.1 > 0.0 { a.1.0 / a.1.1 } else { 0.0 };
            let ratio_b = if b.1.1 > 0.0 { b.1.0 / b.1.1 } else { 0.0 };
            ratio_a.partial_cmp(&ratio_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Try different split points in the sorted order
        let mut best_split = SplitInfo::new();
        let mut left_gradient = 0.0;
        let mut left_hessian = 0.0;
        let mut left_count = 0;

        for i in 0..sorted_categories.len() - 1 {
            let (_, (grad, hess, count)) = &sorted_categories[i];
            left_gradient += grad;
            left_hessian += hess;
            left_count += *count as DataSize;

            let total_gradient: f64 = gradients.iter().sum();
            let total_hessian: f64 = hessians.iter().sum();
            let total_count = categories.len() as DataSize;

            let right_gradient = total_gradient - left_gradient;
            let right_hessian = total_hessian - left_hessian;
            let right_count = total_count - left_count;

            // Check constraints
            if left_count < config.min_data_in_leaf
                || right_count < config.min_data_in_leaf
                || left_hessian < config.min_sum_hessian_in_leaf
                || right_hessian < config.min_sum_hessian_in_leaf
            {
                continue;
            }

            let gain = Self::calculate_categorical_gain(
                total_gradient, total_hessian,
                left_gradient, left_hessian,
                right_gradient, right_hessian,
                config,
            );

            if gain > best_split.gain {
                best_split.gain = gain;
                best_split.left_sum_gradient = left_gradient;
                best_split.left_sum_hessian = left_hessian;
                best_split.left_count = left_count;
                best_split.right_sum_gradient = right_gradient;
                best_split.right_sum_hessian = right_hessian;
                best_split.right_count = right_count;
                best_split.threshold_bin = i as BinIndex; // Store split point index
            }
        }

        if best_split.is_valid() {
            Some(best_split)
        } else {
            None
        }
    }

    fn calculate_categorical_gain(
        total_gradient: f64,
        total_hessian: f64,
        left_gradient: f64,
        left_hessian: f64,
        right_gradient: f64,
        right_hessian: f64,
        config: &SplitFinderConfig,
    ) -> f64 {
        let parent_gain = Self::calculate_leaf_gain(total_gradient, total_hessian, config);
        let left_gain = Self::calculate_leaf_gain(left_gradient, left_hessian, config);
        let right_gain = Self::calculate_leaf_gain(right_gradient, right_hessian, config);

        left_gain + right_gain - parent_gain
    }

    fn calculate_leaf_gain(sum_gradient: f64, sum_hessian: f64, config: &SplitFinderConfig) -> f64 {
        if sum_hessian <= 0.0 {
            return 0.0;
        }

        let numerator = if config.lambda_l1 > 0.0 {
            if sum_gradient > config.lambda_l1 {
                (sum_gradient - config.lambda_l1).powi(2)
            } else if sum_gradient < -config.lambda_l1 {
                (sum_gradient + config.lambda_l1).powi(2)
            } else {
                0.0
            }
        } else {
            sum_gradient.powi(2)
        };

        numerator / (2.0 * (sum_hessian + config.lambda_l2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_evaluator_creation() {
        let config = SplitEvaluatorConfig::default();
        let evaluator = SplitEvaluator::new(config);
        assert_eq!(evaluator.config.max_cat_to_onehot, 4);
    }

    #[test]
    fn test_monotonic_constraint_check() {
        let mut config = SplitEvaluatorConfig::default();
        config.monotonic_constraints.insert(0, MonotonicConstraint::Increasing);
        
        let evaluator = SplitEvaluator::new(config);
        
        let mut split = SplitInfo::new();
        split.feature = 0;
        split.left_output = 1.0;
        split.right_output = 2.0;
        split.gain = 1.0;
        split.left_count = 10;
        split.right_count = 10;
        split.left_sum_hessian = 5.0;
        split.right_sum_hessian = 5.0;

        assert!(evaluator.check_monotonic_constraint(&split, 0.0));

        split.left_output = 2.0;
        split.right_output = 1.0;
        assert!(!evaluator.check_monotonic_constraint(&split, 0.0));
    }

    #[test]
    fn test_balance_score_calculation() {
        let config = SplitEvaluatorConfig::default();
        let evaluator = SplitEvaluator::new(config);

        let mut split = SplitInfo::new();
        split.left_count = 50;
        split.right_count = 50;
        let balance = evaluator.calculate_balance_score(&split);
        assert_eq!(balance, 1.0); // Perfect balance

        split.left_count = 90;
        split.right_count = 10;
        let balance = evaluator.calculate_balance_score(&split);
        assert_eq!(balance, 0.2); // Imbalanced
    }

    #[test]
    fn test_categorical_split_simple() {
        let gradients = vec![-1.0, -1.0, 1.0, 1.0];
        let hessians = vec![1.0, 1.0, 1.0, 1.0];
        let categories = vec![0, 0, 1, 1];
        let config = SplitFinderConfig {
            min_data_in_leaf: 1,
            min_sum_hessian_in_leaf: 0.1,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_split_gain: 0.0,
            max_bin: 255,
        };

        let split = CategoricalSplitHandler::find_best_categorical_split(
            &gradients, &hessians, &categories, &config
        );

        assert!(split.is_some());
        let split = split.unwrap();
        assert!(split.is_valid());
        assert!(split.gain > 0.0);
    }
}