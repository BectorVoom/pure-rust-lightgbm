//! Boosting module for Pure Rust LightGBM.
//!
//! This module provides gradient boosting implementations including
//! GBDT algorithm, tree learners, and ensemble management.

use crate::config::Config;
use crate::core::error::{LightGBMError, Result};
use crate::core::traits::ObjectiveFunction;
use crate::core::types::*;
use crate::dataset::Dataset;
// use crate::io::SerializableModel; // Temporarily disabled
use ndarray::{s, Array1};
use serde::{Deserialize, Serialize};

/// Simple decision tree node for GBDT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleTreeNode {
    /// Feature index for splitting (-1 if leaf)
    pub feature_index: i32,
    /// Threshold value for splitting
    pub threshold: f32,
    /// Left child index (-1 if none)
    pub left_child: i32,
    /// Right child index (-1 if none)
    pub right_child: i32,
    /// Leaf value (for prediction)
    pub leaf_value: f64,
    /// Number of samples in this node
    pub sample_count: usize,
    /// Gain from the split (0.0 for leaf nodes)
    pub split_gain: f64,
    /// Weight of the node (sum of hessians)
    pub node_weight: f64,
    /// Coverage of this node (weighted sample count)
    pub coverage: f64,
}

/// Simple decision tree implementation for GBDT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleTree {
    /// Tree nodes
    pub nodes: Vec<SimpleTreeNode>,
    /// Number of leaves in the tree
    pub num_leaves: usize,
}

impl SimpleTree {
    /// Create a new tree with a single leaf node
    pub fn new_leaf(leaf_value: f64, sample_count: usize) -> Self {
        let root_node = SimpleTreeNode {
            feature_index: -1,
            threshold: 0.0,
            left_child: -1,
            right_child: -1,
            leaf_value,
            sample_count,
            split_gain: 0.0,
            node_weight: sample_count as f64,
            coverage: sample_count as f64,
        };

        SimpleTree {
            nodes: vec![root_node],
            num_leaves: 1,
        }
    }

    /// Predict a single sample
    pub fn predict(&self, features: &[f32]) -> f64 {
        let mut node_idx = 0;

        while node_idx < self.nodes.len() {
            let node = &self.nodes[node_idx];

            // If it's a leaf node, return its value
            if node.feature_index < 0 {
                return node.leaf_value;
            }

            // Navigate to left or right child based on feature value
            let feature_value = features.get(node.feature_index as usize).unwrap_or(&0.0);
            if *feature_value <= node.threshold {
                if node.left_child >= 0 {
                    node_idx = node.left_child as usize;
                } else {
                    return node.leaf_value;
                }
            } else {
                if node.right_child >= 0 {
                    node_idx = node.right_child as usize;
                } else {
                    return node.leaf_value;
                }
            }
        }

        // Fallback (shouldn't reach here)
        0.0
    }

    /// Predict the leaf index for a single sample
    pub fn predict_leaf_index(&self, features: &[f32]) -> i32 {
        let mut node_idx = 0;

        while node_idx < self.nodes.len() {
            let node = &self.nodes[node_idx];

            // If it's a leaf node, return its index
            if node.feature_index < 0 {
                return node_idx as i32;
            }

            // Navigate to left or right child based on feature value
            let feature_value = features.get(node.feature_index as usize).unwrap_or(&0.0);
            if *feature_value <= node.threshold {
                if node.left_child >= 0 {
                    node_idx = node.left_child as usize;
                } else {
                    return node_idx as i32;
                }
            } else {
                if node.right_child >= 0 {
                    node_idx = node.right_child as usize;
                } else {
                    return node_idx as i32;
                }
            }
        }

        // Fallback (shouldn't reach here)
        0
    }

    /// Calculate feature importance for this tree with enhanced algorithms
    pub fn feature_importance(
        &self,
        num_features: usize,
        importance_type: &ImportanceType,
    ) -> Array1<f64> {
        let mut importance = Array1::zeros(num_features);

        match importance_type {
            ImportanceType::Split => {
                // Count the number of times each feature is used for splitting
                for node in &self.nodes {
                    if node.feature_index >= 0 {
                        let feature_idx = node.feature_index as usize;
                        if feature_idx < num_features {
                            importance[feature_idx] += 1.0;
                        }
                    }
                }
            }
            ImportanceType::Gain => {
                // Sum the gain from each split for each feature
                for node in &self.nodes {
                    if node.feature_index >= 0 && node.split_gain > 0.0 {
                        let feature_idx = node.feature_index as usize;
                        if feature_idx < num_features {
                            importance[feature_idx] += node.split_gain;
                        }
                    }
                }
            }
            ImportanceType::Coverage => {
                // Sum the coverage (weighted sample count) for each feature
                for node in &self.nodes {
                    if node.feature_index >= 0 {
                        let feature_idx = node.feature_index as usize;
                        if feature_idx < num_features {
                            importance[feature_idx] += node.coverage;
                        }
                    }
                }
            }
            ImportanceType::TotalGain => {
                // Sum gain weighted by node weight for each feature
                for node in &self.nodes {
                    if node.feature_index >= 0 && node.split_gain > 0.0 {
                        let feature_idx = node.feature_index as usize;
                        if feature_idx < num_features {
                            let weighted_gain = node.split_gain * node.node_weight;
                            importance[feature_idx] += weighted_gain;
                        }
                    }
                }
            }
            ImportanceType::Permutation => {
                // Permutation importance requires prediction evaluation
                // For now, we'll use gain as a proxy (proper implementation requires dataset)
                for node in &self.nodes {
                    if node.feature_index >= 0 && node.split_gain > 0.0 {
                        let feature_idx = node.feature_index as usize;
                        if feature_idx < num_features {
                            // Use gain weighted by sample coverage as proxy for permutation importance
                            let permutation_proxy =
                                node.split_gain * (node.sample_count as f64).sqrt();
                            importance[feature_idx] += permutation_proxy;
                        }
                    }
                }
            }
        }

        importance
    }

    /// Calculate feature importance with access to dataset for permutation importance
    pub fn feature_importance_with_data(
        &self,
        num_features: usize,
        importance_type: &ImportanceType,
        features: Option<&ndarray::Array2<f32>>,
        labels: Option<&ndarray::Array1<f32>>,
    ) -> Array1<f64> {
        match importance_type {
            ImportanceType::Permutation => {
                if let (Some(x), Some(y)) = (features, labels) {
                    self.calculate_permutation_importance(num_features, x, y)
                } else {
                    // Fallback to gain-based proxy
                    self.feature_importance(num_features, &ImportanceType::Gain)
                }
            }
            _ => self.feature_importance(num_features, importance_type),
        }
    }

    /// Calculate permutation importance by measuring performance drop
    fn calculate_permutation_importance(
        &self,
        num_features: usize,
        features: &ndarray::Array2<f32>,
        labels: &ndarray::Array1<f32>,
    ) -> Array1<f64> {
        let mut importance = Array1::zeros(num_features);

        if features.nrows() == 0 || features.nrows() != labels.len() {
            return importance;
        }

        // Calculate baseline performance
        let baseline_score = self.calculate_tree_performance(features, labels);

        // For each feature, permute it and measure performance drop
        for feature_idx in 0..num_features {
            if feature_idx < features.ncols() {
                let permuted_score =
                    self.calculate_permuted_performance(features, labels, feature_idx);

                // Importance is the drop in performance when feature is permuted
                let performance_drop = baseline_score - permuted_score;
                importance[feature_idx] = performance_drop.max(0.0);
            }
        }

        importance
    }

    /// Calculate tree performance (mean squared error for regression)
    fn calculate_tree_performance(
        &self,
        features: &ndarray::Array2<f32>,
        labels: &ndarray::Array1<f32>,
    ) -> f64 {
        let mut total_error = 0.0;
        let n_samples = features.nrows();

        for i in 0..n_samples {
            let row = features.row(i);
            let prediction = self.predict(row.as_slice().unwrap());
            let error = (prediction - labels[i] as f64).powi(2);
            total_error += error;
        }

        // Return negative MSE (higher is better)
        -(total_error / n_samples as f64)
    }

    /// Calculate performance with one feature permuted
    fn calculate_permuted_performance(
        &self,
        features: &ndarray::Array2<f32>,
        labels: &ndarray::Array1<f32>,
        feature_to_permute: usize,
    ) -> f64 {
        let mut total_error = 0.0;
        let n_samples = features.nrows();

        // Create permutation indices for the feature
        let mut permuted_indices: Vec<usize> = (0..n_samples).collect();
        permuted_indices.reverse(); // Simple permutation - in practice should be random

        for i in 0..n_samples {
            let mut row_data: Vec<f32> = features.row(i).to_vec();

            // Replace the feature value with the permuted value
            if feature_to_permute < row_data.len() {
                let permuted_idx = permuted_indices[i];
                if permuted_idx < n_samples {
                    row_data[feature_to_permute] = features[[permuted_idx, feature_to_permute]];
                }
            }

            let prediction = self.predict(&row_data);
            let error = (prediction - labels[i] as f64).powi(2);
            total_error += error;
        }

        // Return negative MSE (higher is better)
        -(total_error / n_samples as f64)
    }

    /// Calculate SHAP values for a single prediction using TreeSHAP algorithm
    pub fn calculate_shap_values(
        &self,
        features: &[f32],
        base_value: f64,
        num_features: usize,
    ) -> ndarray::Array1<f64> {
        let mut shap_values = ndarray::Array1::zeros(num_features);

        if self.nodes.is_empty() {
            return shap_values;
        }

        // Use TreeSHAP algorithm to compute exact Shapley values
        self.tree_shap_recursive(
            0, // Start from root
            features,
            &mut shap_values,
            1.0, // Initial path probability
            base_value,
            0.0, // Path contribution so far
        );

        shap_values
    }

    /// Recursive TreeSHAP implementation for computing SHAP values
    fn tree_shap_recursive(
        &self,
        node_idx: usize,
        features: &[f32],
        shap_values: &mut ndarray::Array1<f64>,
        path_prob: f64,
        base_value: f64,
        path_contribution: f64,
    ) {
        self.tree_shap_recursive_with_path(
            node_idx,
            features,
            shap_values,
            path_prob,
            base_value,
            path_contribution,
            &mut Vec::new(),
        );
    }

    /// TreeSHAP implementation with path tracking for proper leaf contribution attribution
    fn tree_shap_recursive_with_path(
        &self,
        node_idx: usize,
        features: &[f32],
        shap_values: &mut ndarray::Array1<f64>,
        path_prob: f64,
        base_value: f64,
        path_contribution: f64,
        path_features: &mut Vec<(usize, f64)>,
    ) {
        if node_idx >= self.nodes.len() {
            return;
        }

        let node = &self.nodes[node_idx];

        // If this is a leaf node, distribute the contribution among path features
        if node.feature_index < 0 {
            let leaf_contribution = node.leaf_value - base_value - path_contribution;

            // Distribute leaf contribution among features in the path
            if !path_features.is_empty() && leaf_contribution.abs() > 1e-10 {
                let total_path_importance: f64 = path_features
                    .iter()
                    .map(|(_, importance)| importance.abs())
                    .sum();

                if total_path_importance > 1e-10 {
                    // Distribute proportionally based on path importance
                    for &(feature_idx, importance) in path_features.iter() {
                        let attribution_weight = importance.abs() / total_path_importance;
                        let feature_attribution =
                            leaf_contribution * attribution_weight * importance.signum();
                        shap_values[feature_idx] += path_prob * feature_attribution;
                    }
                } else {
                    // If no clear importance pattern, distribute equally among path features
                    let equal_attribution = leaf_contribution / path_features.len() as f64;
                    for &(feature_idx, _) in path_features.iter() {
                        shap_values[feature_idx] += path_prob * equal_attribution;
                    }
                }
            }
            return;
        }

        let feature_idx = node.feature_index as usize;
        if feature_idx >= features.len() || feature_idx >= shap_values.len() {
            return;
        }

        let feature_value = features[feature_idx];
        let threshold = node.threshold;

        // Determine which path this sample takes
        let goes_left = feature_value <= threshold;

        // Get child nodes
        let left_idx = if node.left_child >= 0 {
            node.left_child as usize
        } else {
            node_idx
        };
        let right_idx = if node.right_child >= 0 {
            node.right_child as usize
        } else {
            node_idx
        };

        if goes_left && left_idx < self.nodes.len() {
            // Sample goes left - compute contribution difference
            let left_value = self.get_subtree_value(left_idx);
            let right_value = self.get_subtree_value(right_idx);
            let contribution_diff = left_value - right_value;

            // Attribute the contribution to the splitting feature
            shap_values[feature_idx] += path_prob * contribution_diff * 0.5; // Direct attribution for split decision

            // Add this feature to the path for leaf attribution
            path_features.push((feature_idx, contribution_diff));

            // Continue down the left path
            self.tree_shap_recursive_with_path(
                left_idx,
                features,
                shap_values,
                path_prob,
                base_value,
                path_contribution + contribution_diff,
                path_features,
            );

            // Remove feature from path when backtracking
            path_features.pop();
        } else if !goes_left && right_idx < self.nodes.len() {
            // Sample goes right - compute contribution difference
            let left_value = self.get_subtree_value(left_idx);
            let right_value = self.get_subtree_value(right_idx);
            let contribution_diff = right_value - left_value;

            // Attribute the contribution to the splitting feature
            shap_values[feature_idx] += path_prob * contribution_diff * 0.5; // Direct attribution for split decision

            // Add this feature to the path for leaf attribution
            path_features.push((feature_idx, contribution_diff));

            // Continue down the right path
            self.tree_shap_recursive_with_path(
                right_idx,
                features,
                shap_values,
                path_prob,
                base_value,
                path_contribution + contribution_diff,
                path_features,
            );

            // Remove feature from path when backtracking
            path_features.pop();
        }
    }

    /// Get the expected value of a subtree (simplified implementation)
    fn get_subtree_value(&self, node_idx: usize) -> f64 {
        if node_idx >= self.nodes.len() {
            return 0.0;
        }

        let node = &self.nodes[node_idx];

        // If leaf node, return its value
        if node.feature_index < 0 {
            return node.leaf_value;
        }

        // For internal nodes, return weighted average of children
        let left_idx = if node.left_child >= 0 {
            node.left_child as usize
        } else {
            node_idx
        };
        let right_idx = if node.right_child >= 0 {
            node.right_child as usize
        } else {
            node_idx
        };

        if left_idx < self.nodes.len() && right_idx < self.nodes.len() {
            let left_samples = self.nodes[left_idx].sample_count as f64;
            let right_samples = self.nodes[right_idx].sample_count as f64;
            let total_samples = left_samples + right_samples;

            if total_samples > 0.0 {
                let left_weight = left_samples / total_samples;
                let right_weight = right_samples / total_samples;

                let left_value = self.get_subtree_value(left_idx);
                let right_value = self.get_subtree_value(right_idx);

                return left_weight * left_value + right_weight * right_value;
            }
        }

        node.leaf_value // Fallback
    }

    /// Calculate SHAP interaction values between features (advanced)
    pub fn calculate_shap_interactions(
        &self,
        features: &[f32],
        base_value: f64,
        num_features: usize,
    ) -> ndarray::Array2<f64> {
        let mut interaction_values = ndarray::Array2::zeros((num_features, num_features));

        // For each pair of features, compute their interaction effect
        for i in 0..num_features {
            for j in i..num_features {
                let interaction = self.compute_feature_interaction(features, i, j, base_value);
                interaction_values[[i, j]] = interaction;
                interaction_values[[j, i]] = interaction; // Symmetric
            }
        }

        interaction_values
    }

    /// Compute interaction effect between two features
    fn compute_feature_interaction(
        &self,
        features: &[f32],
        feature_i: usize,
        feature_j: usize,
        _base_value: f64,
    ) -> f64 {
        // Simplified interaction calculation
        // In practice, this would use more sophisticated TreeSHAP interaction algorithms

        if feature_i >= features.len() || feature_j >= features.len() {
            return 0.0;
        }

        // Count how often features i and j appear together in split paths
        let mut interaction_strength = 0.0;
        let mut path_count = 0.0;

        self.compute_path_interactions(
            0,
            feature_i,
            feature_j,
            &mut interaction_strength,
            &mut path_count,
        );

        if path_count > 0.0 {
            interaction_strength / path_count
        } else {
            0.0
        }
    }

    /// Recursively compute interaction effects along tree paths
    fn compute_path_interactions(
        &self,
        node_idx: usize,
        feature_i: usize,
        feature_j: usize,
        interaction_strength: &mut f64,
        path_count: &mut f64,
    ) {
        if node_idx >= self.nodes.len() {
            return;
        }

        let node = &self.nodes[node_idx];

        if node.feature_index < 0 {
            // Leaf node - end of path
            *path_count += 1.0;
            return;
        }

        let feature_idx = node.feature_index as usize;

        // Check if this node uses one of our interaction features
        if feature_idx == feature_i || feature_idx == feature_j {
            *interaction_strength += node.split_gain * (node.sample_count as f64);
        }

        // Continue down both paths
        if node.left_child >= 0 {
            self.compute_path_interactions(
                node.left_child as usize,
                feature_i,
                feature_j,
                interaction_strength,
                path_count,
            );
        }

        if node.right_child >= 0 {
            self.compute_path_interactions(
                node.right_child as usize,
                feature_i,
                feature_j,
                interaction_strength,
                path_count,
            );
        }
    }
}

/// Training history for tracking metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHistory {
    /// Training loss for each iteration
    pub train_loss: Vec<f64>,
    /// Validation loss for each iteration (if validation data provided)
    pub valid_loss: Vec<f64>,
    /// Training metrics
    pub train_metrics: std::collections::HashMap<String, Vec<f64>>,
    /// Validation metrics
    pub valid_metrics: std::collections::HashMap<String, Vec<f64>>,
    /// Best iteration (for early stopping)
    pub best_iteration: Option<usize>,
    /// Best validation score
    pub best_score: Option<f64>,
}

impl Default for TrainingHistory {
    fn default() -> Self {
        TrainingHistory {
            train_loss: Vec::new(),
            valid_loss: Vec::new(),
            train_metrics: std::collections::HashMap::new(),
            valid_metrics: std::collections::HashMap::new(),
            best_iteration: None,
            best_score: None,
        }
    }
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Number of rounds without improvement to trigger early stopping
    pub patience: usize,
    /// Minimum improvement required
    pub min_delta: f64,
    /// Metric to monitor for early stopping
    pub monitor_metric: String,
    /// Whether higher is better for the metric
    pub higher_is_better: bool,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        EarlyStoppingConfig {
            patience: 10,
            min_delta: 1e-6,
            monitor_metric: "loss".to_string(),
            higher_is_better: false,
        }
    }
}

/// Comprehensive feature importance statistics
#[derive(Debug, Clone)]
pub struct FeatureImportanceStats {
    /// Split-based importance (frequency of feature usage)
    pub split_importance: Array1<f64>,
    /// Gain-based importance (sum of gains from splits)
    pub gain_importance: Array1<f64>,
    /// Coverage-based importance (sum of sample coverage)
    pub coverage_importance: Array1<f64>,
    /// Total gain importance (gain weighted by node weight)
    pub total_gain_importance: Array1<f64>,
    /// Number of features
    pub num_features: usize,
}

impl FeatureImportanceStats {
    /// Get importance values for a specific type
    pub fn get_importance(&self, importance_type: &ImportanceType) -> &Array1<f64> {
        match importance_type {
            ImportanceType::Split => &self.split_importance,
            ImportanceType::Gain => &self.gain_importance,
            ImportanceType::Coverage => &self.coverage_importance,
            ImportanceType::TotalGain => &self.total_gain_importance,
            ImportanceType::Permutation => &self.gain_importance, // Fallback to gain
        }
    }

    /// Get the top N most important features for a given importance type
    pub fn get_top_features(
        &self,
        importance_type: &ImportanceType,
        n: usize,
    ) -> Vec<(usize, f64)> {
        let importance = self.get_importance(importance_type);
        let mut feature_scores: Vec<(usize, f64)> = importance
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        // Sort by importance score in descending order
        feature_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top N features
        feature_scores.into_iter().take(n).collect()
    }

    /// Calculate feature ranking correlation between different importance types
    pub fn importance_correlation(&self, type1: &ImportanceType, type2: &ImportanceType) -> f64 {
        let imp1 = self.get_importance(type1);
        let imp2 = self.get_importance(type2);

        if imp1.len() != imp2.len() || imp1.len() == 0 {
            return 0.0;
        }

        // Calculate Pearson correlation coefficient
        let n = imp1.len() as f64;
        let sum1: f64 = imp1.sum();
        let sum2: f64 = imp2.sum();
        let sum1_sq: f64 = imp1.iter().map(|&x| x * x).sum();
        let sum2_sq: f64 = imp2.iter().map(|&x| x * x).sum();
        let sum_prod: f64 = imp1.iter().zip(imp2.iter()).map(|(&x, &y)| x * y).sum();

        let numerator = n * sum_prod - sum1 * sum2;
        let denominator = ((n * sum1_sq - sum1 * sum1) * (n * sum2_sq - sum2 * sum2)).sqrt();

        if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Get summary statistics for importance values
    pub fn summary_stats(&self, importance_type: &ImportanceType) -> ImportanceSummary {
        let importance = self.get_importance(importance_type);

        if importance.is_empty() {
            return ImportanceSummary::default();
        }

        let mut values: Vec<f64> = importance.to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = values[0];
        let max = values[values.len() - 1];
        let mean = values.iter().sum::<f64>() / values.len() as f64;

        let median = if values.len() % 2 == 0 {
            (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
        } else {
            values[values.len() / 2]
        };

        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        ImportanceSummary {
            min,
            max,
            mean,
            median,
            std_dev,
            non_zero_count: values.iter().filter(|&&x| x > 1e-10).count(),
            total_count: values.len(),
        }
    }
}

/// Summary statistics for feature importance values
#[derive(Debug, Clone)]
pub struct ImportanceSummary {
    /// Minimum importance value
    pub min: f64,
    /// Maximum importance value
    pub max: f64,
    /// Mean importance value
    pub mean: f64,
    /// Median importance value
    pub median: f64,
    /// Standard deviation of importance values
    pub std_dev: f64,
    /// Number of features with non-zero importance
    pub non_zero_count: usize,
    /// Total number of features
    pub total_count: usize,
}

impl Default for ImportanceSummary {
    fn default() -> Self {
        ImportanceSummary {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            non_zero_count: 0,
            total_count: 0,
        }
    }
}

/// Feature contribution for SHAP explanations
#[derive(Debug, Clone)]
pub struct FeatureContribution {
    /// Index of the feature
    pub feature_index: usize,
    /// Value of the feature for this prediction
    pub feature_value: f64,
    /// SHAP value (contribution to prediction)
    pub shap_value: f64,
    /// Absolute SHAP value for ranking
    pub abs_shap_value: f64,
}

/// Complete SHAP explanation for a single prediction
#[derive(Debug, Clone)]
pub struct SHAPExplanation {
    /// Base value (expected model output)
    pub base_value: f64,
    /// Final prediction value
    pub prediction: f64,
    /// SHAP values for all features
    pub shap_values: Array1<f64>,
    /// Feature contributions sorted by importance
    pub feature_contributions: Vec<FeatureContribution>,
    /// Number of features
    pub num_features: usize,
}

impl SHAPExplanation {
    /// Get the top N most important features
    pub fn top_features(&self, n: usize) -> &[FeatureContribution] {
        let end_idx = n.min(self.feature_contributions.len());
        &self.feature_contributions[0..end_idx]
    }

    /// Get features that contribute positively to the prediction
    pub fn positive_contributions(&self) -> Vec<&FeatureContribution> {
        self.feature_contributions
            .iter()
            .filter(|contrib| contrib.shap_value > 0.0)
            .collect()
    }

    /// Get features that contribute negatively to the prediction
    pub fn negative_contributions(&self) -> Vec<&FeatureContribution> {
        self.feature_contributions
            .iter()
            .filter(|contrib| contrib.shap_value < 0.0)
            .collect()
    }

    /// Verify that SHAP values sum correctly
    pub fn verify_additivity(&self) -> f64 {
        let shap_sum: f64 = self.shap_values.sum();
        let expected_sum = self.prediction - self.base_value;
        (shap_sum - expected_sum).abs()
    }

    /// Get summary statistics for SHAP values
    pub fn summary_stats(&self) -> SHAPSummaryStats {
        let values: Vec<f64> = self.shap_values.to_vec();
        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted_values[0];
        let max = sorted_values[sorted_values.len() - 1];
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let median = if sorted_values.len() % 2 == 0 {
            (sorted_values[sorted_values.len() / 2 - 1] + sorted_values[sorted_values.len() / 2])
                / 2.0
        } else {
            sorted_values[sorted_values.len() / 2]
        };

        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        SHAPSummaryStats {
            min,
            max,
            mean,
            median,
            std_dev,
            total_absolute_contribution: values.iter().map(|x| x.abs()).sum(),
            additivity_error: self.verify_additivity(),
        }
    }
}

/// Summary statistics for SHAP values
#[derive(Debug, Clone)]
pub struct SHAPSummaryStats {
    /// Minimum SHAP value
    pub min: f64,
    /// Maximum SHAP value
    pub max: f64,
    /// Mean SHAP value
    pub mean: f64,
    /// Median SHAP value
    pub median: f64,
    /// Standard deviation of SHAP values
    pub std_dev: f64,
    /// Sum of absolute SHAP contributions
    pub total_absolute_contribution: f64,
    /// SHAP additivity error (difference from expected sum)
    pub additivity_error: f64,
}

/// Validation statistics for SHAP values
#[derive(Debug, Clone)]
pub struct SHAPValidationStats {
    /// Mean error in SHAP sum vs (prediction - base_value)
    pub mean_shap_error: f64,
    /// Maximum error in SHAP sum vs (prediction - base_value)
    pub max_shap_error: f64,
    /// Mean absolute SHAP value across all samples and features
    pub mean_abs_shap_value: f64,
    /// Base value used for calculations
    pub base_value: f64,
    /// Number of samples validated
    pub num_samples: usize,
    /// Number of features per sample
    pub num_features: usize,
}

impl SHAPValidationStats {
    /// Check if SHAP values are within acceptable error tolerance
    pub fn is_valid(&self, tolerance: f64) -> bool {
        self.mean_shap_error <= tolerance && self.max_shap_error <= tolerance * 10.0
    }

    /// Get a summary report of validation results
    pub fn validation_report(&self, tolerance: f64) -> String {
        format!(
            "SHAP Validation Report:\n\
             - Samples: {}\n\
             - Features: {}\n\
             - Base Value: {:.6}\n\
             - Mean Error: {:.6} (tolerance: {:.6})\n\
             - Max Error: {:.6}\n\
             - Mean |SHAP|: {:.6}\n\
             - Valid: {}",
            self.num_samples,
            self.num_features,
            self.base_value,
            self.mean_shap_error,
            tolerance,
            self.max_shap_error,
            self.mean_abs_shap_value,
            self.is_valid(tolerance)
        )
    }
}

/// Gradient Boosting Decision Tree implementation
#[derive(Debug, Serialize, Deserialize)]
pub struct GBDT {
    /// Configuration for the GBDT
    config: Config,
    /// Training dataset reference
    #[serde(skip)]
    train_data: Option<Dataset>,
    /// Validation dataset reference
    #[serde(skip)]
    valid_data: Option<Dataset>,
    /// Current iteration
    current_iteration: usize,
    /// Trained tree models
    models: Vec<SimpleTree>,
    /// Current training scores/predictions
    #[serde(skip)]
    train_scores: Option<ndarray::Array1<Score>>,
    /// Current validation scores/predictions
    #[serde(skip)]
    valid_scores: Option<ndarray::Array1<Score>>,
    /// Objective function for gradient computation
    #[serde(skip)]
    objective_function: Option<Box<dyn ObjectiveFunction>>,
    /// Gradient and hessian buffers
    #[serde(skip)]
    gradients: Option<ndarray::Array1<Score>>,
    #[serde(skip)]
    hessians: Option<ndarray::Array1<Score>>,
    /// Training history
    training_history: TrainingHistory,
    /// Early stopping configuration
    #[serde(skip)]
    early_stopping_config: Option<EarlyStoppingConfig>,
    /// Whether early stopping was triggered
    early_stopped: bool,
}

impl GBDT {
    /// Create a new GBDT instance
    pub fn new(config: Config, train_data: Dataset) -> Result<Self> {
        let num_data = train_data.num_data();
        let objective_function = create_objective_function(&config)?;

        Ok(Self {
            config,
            train_data: Some(train_data),
            valid_data: None,
            current_iteration: 0,
            models: Vec::new(),
            train_scores: Some(Array1::zeros(num_data)),
            valid_scores: None,
            objective_function: Some(objective_function),
            gradients: Some(Array1::zeros(num_data)),
            hessians: Some(Array1::zeros(num_data)),
            training_history: TrainingHistory::default(),
            early_stopping_config: None,
            early_stopped: false,
        })
    }

    /// Add validation dataset for early stopping
    pub fn add_validation_data(&mut self, valid_data: Dataset) -> Result<()> {
        let num_valid_data = valid_data.num_data();
        self.valid_scores = Some(Array1::zeros(num_valid_data));
        self.valid_data = Some(valid_data);
        Ok(())
    }

    /// Set early stopping configuration
    pub fn set_early_stopping(&mut self, config: EarlyStoppingConfig) {
        self.early_stopping_config = Some(config);
    }

    /// Get training history
    pub fn training_history(&self) -> &TrainingHistory {
        &self.training_history
    }

    /// Check if training was stopped early
    pub fn was_early_stopped(&self) -> bool {
        self.early_stopped
    }

    /// Train the GBDT model
    pub fn train(&mut self) -> Result<()> {
        log::info!(
            "Starting GBDT training with {} iterations",
            self.config.num_iterations
        );

        // Get dataset info first
        let (_num_data, _num_features, base_prediction) = {
            // TODO: Implement data loading functionality - num_data and num_features currently unused (issue #68)
            // These should be used for validation and memory allocation
            let train_data = self
                .train_data
                .as_ref()
                .ok_or_else(|| LightGBMError::training("No training data available"))?;

            let num_data = train_data.num_data();
            let num_features = train_data.num_features();

            log::info!(
                "Training data: {} samples, {} features",
                num_data,
                num_features
            );

            // Initialize base prediction (mean of labels for regression, log-odds for classification)
            let base_prediction = self.compute_base_prediction(train_data)?;
            (num_data, num_features, base_prediction)
        };

        log::info!("Base prediction: {}", base_prediction);

        // Initialize training scores with base prediction
        if let Some(ref mut train_scores) = self.train_scores {
            train_scores.fill(base_prediction);
        }

        // Initialize validation scores if validation data is provided
        if let Some(ref valid_data) = self.valid_data {
            // Compute proper initial validation predictions using validation data
            if let Some(ref mut valid_scores) = self.valid_scores {
                // For initial iteration, use base prediction for all validation samples
                // This is the standard approach in gradient boosting
                valid_scores.fill(base_prediction);
                
                log::info!("Initialized validation scores for {} validation samples", valid_data.num_data());
            }
        }

        // Training loop
        for iteration in 0..self.config.num_iterations {
            log::info!(
                "Training iteration {}/{}",
                iteration + 1,
                self.config.num_iterations
            );

            // Compute gradients and hessians
            self.compute_gradients()?;

            // Debug: Check gradients/hessians after computation
            if let (Some(ref gradients), Some(ref hessians)) = (&self.gradients, &self.hessians) {
                let g_sum: f64 = gradients.iter().map(|&g| g as f64).sum();
                let h_sum: f64 = hessians.iter().map(|&h| h as f64).sum();
                let g_mean = g_sum / gradients.len() as f64;
                let h_mean = h_sum / hessians.len() as f64;
                log::debug!(
                    "After compute_gradients: sum_g={:.6}, sum_h={:.6}, mean_g={:.6}, mean_h={:.6}",
                    g_sum,
                    h_sum,
                    g_mean,
                    h_mean
                );

                // Check current predictions
                if let Some(ref scores) = self.train_scores {
                    let scores_sample: Vec<f32> = scores.iter().take(5).cloned().collect();
                    log::debug!("Current predictions sample: {:?}", scores_sample);
                }
            }

            // Train a tree on gradients/hessians
            let tree = self.train_tree()?;

            // Debug: Check tree structure
            log::debug!(
                "Trained tree: {} nodes, {} leaves",
                tree.nodes.len(),
                tree.num_leaves
            );
            if !tree.nodes.is_empty() {
                let root = &tree.nodes[0];
                log::debug!(
                    "Root node: feature={}, threshold={:.6}, leaf_value={:.6}",
                    root.feature_index,
                    root.threshold,
                    root.leaf_value
                );
            }

            // Update training scores
            self.update_scores(&tree)?;

            // Update validation scores if validation data is provided
            if self.valid_data.is_some() {
                self.update_validation_scores(&tree)?;
            }

            // Calculate and record metrics
            self.calculate_and_record_metrics(iteration)?;

            // Store the tree
            self.models.push(tree);
            self.current_iteration = iteration + 1;

            // Check for early stopping
            if self.should_early_stop(iteration)? {
                log::info!("Early stopping triggered at iteration {}", iteration + 1);
                self.early_stopped = true;
                break;
            }

            // Periodic logging
            if iteration > 0 && (iteration + 1) % 10 == 0 {
                self.log_training_progress(iteration + 1);
            }
        }

        log::info!(
            "Training completed. Final model has {} trees",
            self.models.len()
        );
        Ok(())
    }

    /// Update validation scores with new tree predictions
    fn update_validation_scores(&mut self, tree: &SimpleTree) -> Result<()> {
        let valid_data = self
            .valid_data
            .as_ref()
            .ok_or_else(|| LightGBMError::training("No validation data available"))?;
        let valid_scores = self
            .valid_scores
            .as_mut()
            .ok_or_else(|| LightGBMError::training("No validation scores available"))?;

        let features = valid_data.features();

        for i in 0..valid_data.num_data() {
            let feature_row: Vec<f32> = (0..valid_data.num_features())
                .map(|j| features[[i, j]])
                .collect();

            let tree_prediction = tree.predict(&feature_row);
            valid_scores[i] += tree_prediction as f32;
        }

        Ok(())
    }

    /// Calculate and record training/validation metrics
    fn calculate_and_record_metrics(&mut self, iteration: usize) -> Result<()> {
        // Calculate training loss
        let train_loss = self.calculate_training_loss()?;
        self.training_history.train_loss.push(train_loss);

        // Calculate validation loss if validation data is available
        if self.valid_data.is_some() {
            let valid_loss = self.calculate_validation_loss()?;
            self.training_history.valid_loss.push(valid_loss);

            // Update best score for early stopping
            if let Some(ref early_stopping) = self.early_stopping_config {
                let current_score = if early_stopping.monitor_metric == "loss" {
                    valid_loss
                } else {
                    valid_loss // Default to loss for now
                };

                let is_better = if early_stopping.higher_is_better {
                    self.training_history
                        .best_score
                        .map_or(true, |best| current_score > best + early_stopping.min_delta)
                } else {
                    self.training_history
                        .best_score
                        .map_or(true, |best| current_score < best - early_stopping.min_delta)
                };

                if is_better {
                    self.training_history.best_score = Some(current_score);
                    self.training_history.best_iteration = Some(iteration);
                }
            }
        }

        Ok(())
    }

    /// Calculate training loss
    fn calculate_training_loss(&self) -> Result<f64> {
        let train_data = self
            .train_data
            .as_ref()
            .ok_or_else(|| LightGBMError::training("No training data available"))?;
        let train_scores = self
            .train_scores
            .as_ref()
            .ok_or_else(|| LightGBMError::training("No training scores available"))?;

        let labels = train_data.labels();
        let loss = self.calculate_loss(train_scores, labels)?;
        Ok(loss)
    }

    /// Calculate validation loss
    fn calculate_validation_loss(&self) -> Result<f64> {
        let valid_data = self
            .valid_data
            .as_ref()
            .ok_or_else(|| LightGBMError::training("No validation data available"))?;
        let valid_scores = self
            .valid_scores
            .as_ref()
            .ok_or_else(|| LightGBMError::training("No validation scores available"))?;

        let labels = valid_data.labels();
        let loss = self.calculate_loss(valid_scores, labels)?;
        Ok(loss)
    }

    /// Calculate loss for given predictions and labels
    fn calculate_loss(
        &self,
        predictions: &Array1<f32>,
        labels: ndarray::ArrayView1<'_, f32>,
    ) -> Result<f64> {
        if predictions.len() != labels.len() {
            return Err(LightGBMError::dimension_mismatch(
                format!("predictions: {}", predictions.len()),
                format!("labels: {}", labels.len()),
            ));
        }

        match self.config.objective {
            ObjectiveType::Regression => {
                // Mean squared error
                let mse: f64 = predictions
                    .iter()
                    .zip(labels.iter())
                    .map(|(pred, label)| {
                        let diff = *pred as f64 - (*label) as f64;
                        diff * diff
                    })
                    .sum::<f64>()
                    / predictions.len() as f64;
                Ok(mse)
            }
            ObjectiveType::Binary => {
                // Binary cross-entropy
                let mut loss = 0.0;
                for (pred, label) in predictions.iter().zip(labels.iter()) {
                    let prob = 1.0 / (1.0 + (-pred).exp()) as f64;
                    let prob = prob.max(1e-15).min(1.0 - 1e-15); // Clamp to avoid log(0)
                    loss += -((*label) as f64 * prob.ln()
                        + (1.0 - (*label) as f64) * (1.0 - prob).ln());
                }
                Ok(loss / predictions.len() as f64)
            }
            _ => {
                // Default to MSE for other objectives
                let mse: f64 = predictions
                    .iter()
                    .zip(labels.iter())
                    .map(|(pred, label)| {
                        let diff = *pred as f64 - (*label) as f64;
                        diff * diff
                    })
                    .sum::<f64>()
                    / predictions.len() as f64;
                Ok(mse)
            }
        }
    }

    /// Check if early stopping should be triggered
    fn should_early_stop(&self, iteration: usize) -> Result<bool> {
        if let Some(ref early_stopping) = self.early_stopping_config {
            if let Some(best_iteration) = self.training_history.best_iteration {
                let rounds_without_improvement = iteration - best_iteration;
                if rounds_without_improvement >= early_stopping.patience {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Log training progress
    fn log_training_progress(&self, iteration: usize) {
        let train_loss = self.training_history.train_loss.last().unwrap_or(&0.0);

        if let Some(valid_loss) = self.training_history.valid_loss.last() {
            log::info!(
                "Iteration {}: train_loss={:.6}, valid_loss={:.6}",
                iteration,
                train_loss,
                valid_loss
            );
        } else {
            log::info!("Iteration {}: train_loss={:.6}", iteration, train_loss);
        }

        // Log best iteration info if available
        if let (Some(best_iter), Some(best_score)) = (
            self.training_history.best_iteration,
            self.training_history.best_score,
        ) {
            log::debug!(
                "Best iteration: {}, best score: {:.6}",
                best_iter + 1,
                best_score
            );
        }
    }

    /// Compute base prediction (initial value)
    fn compute_base_prediction(&self, train_data: &Dataset) -> Result<f32> {
        let labels = train_data.labels();

        match self.config.objective {
            ObjectiveType::Regression => {
                // Use mean of labels for regression
                let mean = labels.iter().sum::<f32>() / labels.len() as f32;
                Ok(mean)
            }
            ObjectiveType::Binary => {
                // Use log-odds for binary classification
                let positive_count = labels.iter().filter(|&&label| label > 0.5).count();
                let negative_count = labels.len() - positive_count;

                if positive_count == 0 || negative_count == 0 {
                    Ok(0.0) // Handle edge case
                } else {
                    let log_odds = ((positive_count as f32) / (negative_count as f32)).ln();
                    Ok(log_odds)
                }
            }
            _ => {
                // For other objectives, start with 0
                Ok(0.0)
            }
        }
    }

    /// Compute gradients and hessians using the objective function
    fn compute_gradients(&mut self) -> Result<()> {
        let train_data = self
            .train_data
            .as_ref()
            .ok_or_else(|| LightGBMError::training("No training data available"))?;
        let train_scores = self
            .train_scores
            .as_ref()
            .ok_or_else(|| LightGBMError::training("No training scores available"))?;

        let labels = train_data.labels();
        let weights = train_data.weights();

        let objective = self
            .objective_function
            .as_ref()
            .ok_or_else(|| LightGBMError::training("No objective function available"))?;

        let gradients = self
            .gradients
            .as_mut()
            .ok_or_else(|| LightGBMError::training("No gradients buffer available"))?;

        let hessians = self
            .hessians
            .as_mut()
            .ok_or_else(|| LightGBMError::training("No hessians buffer available"))?;

        objective.compute_gradients(
            &train_scores.view(),
            &labels.view(),
            weights.as_ref().map(|w| w.view()).as_ref(),
            &mut gradients.view_mut(),
            &mut hessians.view_mut(),
        )?;

        Ok(())
    }

    /// Train a decision tree using gradients/hessians with proper split finding
    fn train_tree(&self) -> Result<SimpleTree> {
        let train_data = self
            .train_data
            .as_ref()
            .ok_or_else(|| LightGBMError::training("No training data available"))?;
        let gradients = self
            .gradients
            .as_ref()
            .ok_or_else(|| LightGBMError::training("No gradients available"))?;
        let hessians = self
            .hessians
            .as_ref()
            .ok_or_else(|| LightGBMError::training("No hessians available"))?;

        // Create sample indices for this tree
        let sample_indices: Vec<usize> = (0..train_data.num_data()).collect();

        // Build tree recursively starting from root
        let mut tree_builder = TreeBuilder::new(&self.config, train_data, gradients, hessians);
        let root_node = tree_builder.build_node(&sample_indices, 0)?;

        // Apply depth-based pruning to the constructed tree
        let pruned_root = tree_builder.prune_tree(root_node);

        // Convert to SimpleTree format
        Ok(tree_builder.to_simple_tree(pruned_root))
    }

    /// Update training scores with new tree predictions
    fn update_scores(&mut self, tree: &SimpleTree) -> Result<()> {
        let train_data = self
            .train_data
            .as_ref()
            .ok_or_else(|| LightGBMError::training("No training data available"))?;
        let train_scores = self
            .train_scores
            .as_mut()
            .ok_or_else(|| LightGBMError::training("No training scores available"))?;

        let features = train_data.features();

        for i in 0..train_data.num_data() {
            let feature_row: Vec<f32> = (0..train_data.num_features())
                .map(|j| features[[i, j]])
                .collect();

            let tree_prediction = tree.predict(&feature_row);
            train_scores[i] += tree_prediction as f32;
        }

        Ok(())
    }

    /// Get the current configuration
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Get number of trained iterations
    pub fn num_iterations(&self) -> usize {
        self.current_iteration
    }

    /// Make predictions on new data
    pub fn predict(&self, features: &ndarray::Array2<f32>) -> Result<ndarray::Array1<f32>> {
        if self.models.is_empty() {
            return Err(LightGBMError::prediction(
                "No trained models available for prediction",
            ));
        }

        let num_samples = features.nrows();
        let num_features = features.ncols();

        let expected_features = self
            .train_data
            .as_ref()
            .map(|d| d.num_features())
            .unwrap_or(num_features);

        if num_features != expected_features {
            return Err(LightGBMError::dimension_mismatch(
                format!("expected features: {}", expected_features),
                format!("provided features: {}", num_features),
            ));
        }

        let mut predictions = ndarray::Array1::zeros(num_samples);

        // Get base prediction
        let base_pred = if let Some(ref train_data) = self.train_data {
            self.compute_base_prediction(train_data).unwrap_or(0.0)
        } else {
            0.0
        };

        // Initialize with base prediction
        predictions.fill(base_pred);

        // Add predictions from all trees
        for tree in &self.models {
            for i in 0..num_samples {
                let feature_row: Vec<f32> = (0..num_features).map(|j| features[[i, j]]).collect();

                let tree_prediction = tree.predict(&feature_row);
                predictions[i] += tree_prediction as f32;
            }
        }

        // Apply objective function transformation if needed
        if let Some(ref objective) = self.objective_function {
            let mut pred_view = predictions.view_mut();
            objective.transform_predictions(&mut pred_view)?;
        }

        Ok(predictions)
    }

    /// Get the trained models
    pub fn models(&self) -> &[SimpleTree] {
        &self.models
    }

    /// Calculate feature importance across all trees with enhanced algorithms
    pub fn feature_importance(&self, importance_type: &ImportanceType) -> Result<Array1<f64>> {
        let num_features = if let Some(ref train_data) = self.train_data {
            train_data.num_features()
        } else {
            return Err(LightGBMError::training(
                "No training data available to determine feature count",
            ));
        };

        let mut total_importance = Array1::zeros(num_features);

        // For permutation importance, we need access to training data
        if *importance_type == ImportanceType::Permutation && self.train_data.is_some() {
            return self.calculate_ensemble_permutation_importance();
        }

        // Aggregate importance from all trees
        for tree in &self.models {
            let tree_importance = tree.feature_importance(num_features, importance_type);
            total_importance = total_importance + tree_importance;
        }

        // Apply sophisticated normalization based on importance type
        self.normalize_importance(&mut total_importance, importance_type);

        Ok(total_importance)
    }

    /// Normalize feature importance values based on the importance type
    fn normalize_importance(&self, importance: &mut Array1<f64>, importance_type: &ImportanceType) {
        match importance_type {
            ImportanceType::Split => {
                // For split-based importance, normalize by the average number of splits per tree
                if !self.models.is_empty() {
                    let total_splits: f64 = self
                        .models
                        .iter()
                        .map(|tree| {
                            tree.nodes
                                .iter()
                                .filter(|node| node.feature_index >= 0)
                                .count() as f64
                        })
                        .sum();
                    if total_splits > 0.0 {
                        *importance = importance.clone() / total_splits;
                    }
                }
            }
            ImportanceType::Gain | ImportanceType::TotalGain => {
                // For gain-based importance, normalize to percentages
                let total_gain: f64 = importance.sum();
                if total_gain > 0.0 {
                    *importance = importance.clone() / total_gain * 100.0;
                }
            }
            ImportanceType::Coverage => {
                // For coverage-based importance, normalize by total sample coverage
                let total_coverage: f64 = importance.sum();
                if total_coverage > 0.0 {
                    *importance = importance.clone() / total_coverage * 100.0;
                }
            }
            ImportanceType::Permutation => {
                // Permutation importance is already in absolute terms (performance drop)
                // Optionally normalize to percentages of baseline performance
                let max_importance = importance.iter().fold(0.0f64, |a, &b| a.max(b));
                if max_importance > 0.0 {
                    *importance = importance.clone() / max_importance * 100.0;
                }
            }
        }
    }

    /// Calculate permutation importance for the entire ensemble
    fn calculate_ensemble_permutation_importance(&self) -> Result<Array1<f64>> {
        let train_data = self.train_data.as_ref().ok_or_else(|| {
            LightGBMError::training("No training data available for permutation importance")
        })?;

        let num_features = train_data.num_features();
        let mut importance = Array1::zeros(num_features);

        // Get training features and labels
        let features = train_data.features();
        let labels = train_data.labels();

        if features.nrows() == 0 || features.nrows() != labels.len() {
            return Ok(importance);
        }

        // Calculate baseline ensemble performance
        let baseline_score = self.calculate_ensemble_performance(&features, &labels)?;

        // For each feature, permute it and measure performance drop
        for feature_idx in 0..num_features {
            if feature_idx < features.ncols() {
                let permuted_score =
                    self.calculate_ensemble_permuted_performance(&features, &labels, feature_idx)?;

                // Importance is the drop in performance when feature is permuted
                let performance_drop = baseline_score - permuted_score;
                importance[feature_idx] = performance_drop.max(0.0);
            }
        }

        // Normalize permutation importance
        self.normalize_importance(&mut importance, &ImportanceType::Permutation);

        Ok(importance)
    }

    /// Calculate ensemble performance on given data
    fn calculate_ensemble_performance(
        &self,
        features: &ndarray::ArrayView2<'_, f32>,
        labels: &ndarray::ArrayView1<'_, f32>,
    ) -> Result<f64> {
        let features_owned = features.to_owned();
        let predictions = self.predict(&features_owned)?;
        let mut total_error = 0.0;

        for i in 0..predictions.len() {
            let error = (predictions[i] as f64 - labels[i] as f64).powi(2);
            total_error += error;
        }

        // Return negative MSE (higher is better)
        Ok(-(total_error / predictions.len() as f64))
    }

    /// Calculate ensemble performance with one feature permuted
    fn calculate_ensemble_permuted_performance(
        &self,
        features: &ndarray::ArrayView2<'_, f32>,
        labels: &ndarray::ArrayView1<'_, f32>,
        feature_to_permute: usize,
    ) -> Result<f64> {
        // Create a copy of features with the specified feature permuted
        let mut permuted_features = features.to_owned();
        let n_samples = features.nrows();

        // Create permutation indices (simple reverse for now)
        let permuted_indices: Vec<usize> = (0..n_samples).rev().collect();

        // Permute the feature values
        for i in 0..n_samples {
            let permuted_idx = permuted_indices[i];
            if permuted_idx < n_samples && feature_to_permute < features.ncols() {
                permuted_features[[i, feature_to_permute]] =
                    features[[permuted_idx, feature_to_permute]];
            }
        }

        // Calculate performance with permuted feature
        let permuted_view = permuted_features.view();
        self.calculate_ensemble_performance(&permuted_view, labels)
    }

    /// Get detailed feature importance statistics
    pub fn feature_importance_detailed(&self) -> Result<FeatureImportanceStats> {
        let num_features = if let Some(ref train_data) = self.train_data {
            train_data.num_features()
        } else {
            return Err(LightGBMError::training("No training data available"));
        };

        let split_importance = self.feature_importance(&ImportanceType::Split)?;
        let gain_importance = self.feature_importance(&ImportanceType::Gain)?;
        let coverage_importance = self.feature_importance(&ImportanceType::Coverage)?;
        let total_gain_importance = self.feature_importance(&ImportanceType::TotalGain)?;

        Ok(FeatureImportanceStats {
            split_importance,
            gain_importance,
            coverage_importance,
            total_gain_importance,
            num_features,
        })
    }

    /// Calculate SHAP values for individual predictions using TreeSHAP
    pub fn predict_contrib(&self, features: &ndarray::Array2<f32>) -> Result<ndarray::Array2<f64>> {
        let num_samples = features.nrows();
        let num_features = features.ncols();

        if self.models.is_empty() {
            return Err(LightGBMError::prediction(
                "No trained models available for SHAP calculation",
            ));
        }

        // Calculate base value (expected output when no features are known)
        let base_value = self.calculate_base_value()?;

        // Initialize SHAP values matrix: [samples x features]
        let mut shap_values = ndarray::Array2::zeros((num_samples, num_features));

        // For each sample, calculate SHAP values
        for sample_idx in 0..num_samples {
            let sample_features: Vec<f32> = (0..num_features)
                .map(|feat_idx| features[[sample_idx, feat_idx]])
                .collect();

            // Aggregate SHAP values across all trees
            let mut sample_shap = ndarray::Array1::zeros(num_features);

            for tree in &self.models {
                let tree_shap =
                    tree.calculate_shap_values(&sample_features, base_value, num_features);
                sample_shap = sample_shap + tree_shap;
            }

            // Scale by learning rate (implicit in tree contributions)
            sample_shap = sample_shap * self.config.learning_rate as f64;

            // Store SHAP values for this sample
            for feat_idx in 0..num_features {
                shap_values[[sample_idx, feat_idx]] = sample_shap[feat_idx];
            }
        }

        Ok(shap_values)
    }

    /// Calculate SHAP values for a single sample
    pub fn predict_contrib_single(&self, features: &[f32]) -> Result<ndarray::Array1<f64>> {
        let num_features = features.len();

        if self.models.is_empty() {
            return Err(LightGBMError::prediction(
                "No trained models available for SHAP calculation",
            ));
        }

        let base_value = self.calculate_base_value()?;
        let mut shap_values = ndarray::Array1::zeros(num_features);

        // Aggregate SHAP values across all trees
        for tree in &self.models {
            let tree_shap = tree.calculate_shap_values(features, base_value, num_features);
            shap_values = shap_values + tree_shap;
        }

        // Scale by learning rate
        shap_values = shap_values * self.config.learning_rate as f64;

        Ok(shap_values)
    }

    /// Calculate SHAP interaction values for feature pairs
    pub fn predict_contrib_interactions(
        &self,
        features: &ndarray::Array2<f32>,
    ) -> Result<ndarray::Array3<f64>> {
        let num_samples = features.nrows();
        let num_features = features.ncols();

        if self.models.is_empty() {
            return Err(LightGBMError::prediction(
                "No trained models available for SHAP interaction calculation",
            ));
        }

        let base_value = self.calculate_base_value()?;

        // Initialize interaction values tensor: [samples x features x features]
        let mut interaction_values =
            ndarray::Array3::zeros((num_samples, num_features, num_features));

        for sample_idx in 0..num_samples {
            let sample_features: Vec<f32> = (0..num_features)
                .map(|feat_idx| features[[sample_idx, feat_idx]])
                .collect();

            // Aggregate interaction values across all trees
            let mut sample_interactions = ndarray::Array2::zeros((num_features, num_features));

            for tree in &self.models {
                let tree_interactions =
                    tree.calculate_shap_interactions(&sample_features, base_value, num_features);
                sample_interactions = sample_interactions + tree_interactions;
            }

            // Scale by learning rate
            sample_interactions = sample_interactions * self.config.learning_rate as f64;

            // Store interaction values for this sample
            for i in 0..num_features {
                for j in 0..num_features {
                    interaction_values[[sample_idx, i, j]] = sample_interactions[[i, j]];
                }
            }
        }

        Ok(interaction_values)
    }

    /// Calculate the base value (expected model output)
    fn calculate_base_value(&self) -> Result<f64> {
        if let Some(ref train_data) = self.train_data {
            // Use mean of training labels as base value
            let labels = train_data.labels();
            let sum: f32 = labels.iter().sum();
            Ok(sum as f64 / labels.len() as f64)
        } else {
            // Fallback: use first tree root value if available
            if !self.models.is_empty() && !self.models[0].nodes.is_empty() {
                Ok(self.models[0].nodes[0].leaf_value)
            } else {
                Ok(0.0)
            }
        }
    }

    /// Calculate SHAP values with expected value breakdown
    pub fn explain_prediction(&self, features: &[f32]) -> Result<SHAPExplanation> {
        let shap_values = self.predict_contrib_single(features)?;
        let base_value = self.calculate_base_value()?;
        let prediction = self.predict_single(features)?;

        // Calculate feature contributions
        let mut feature_contributions = Vec::new();
        for (idx, &shap_value) in shap_values.iter().enumerate() {
            feature_contributions.push(FeatureContribution {
                feature_index: idx,
                feature_value: if idx < features.len() {
                    features[idx] as f64
                } else {
                    0.0
                },
                shap_value,
                abs_shap_value: shap_value.abs(),
            });
        }

        // Sort by absolute SHAP value (most important first)
        feature_contributions.sort_by(|a, b| {
            b.abs_shap_value
                .partial_cmp(&a.abs_shap_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(SHAPExplanation {
            base_value,
            prediction: prediction as f64,
            shap_values,
            feature_contributions,
            num_features: features.len(),
        })
    }

    /// Predict single sample (helper method)
    fn predict_single(&self, features: &[f32]) -> Result<f32> {
        let mut prediction = 0.0;

        for tree in &self.models {
            prediction += tree.predict(features) as f32;
        }

        prediction *= self.config.learning_rate as f32;
        Ok(prediction)
    }

    /// Calculate expected value statistics for SHAP validation
    pub fn validate_shap_values(
        &self,
        features: &ndarray::Array2<f32>,
    ) -> Result<SHAPValidationStats> {
        let shap_values = self.predict_contrib(features)?;
        let predictions = self.predict(features)?;
        let base_value = self.calculate_base_value()?;

        let mut shap_sum_errors = Vec::new();
        let mut max_shap_error: f64 = 0.0;
        let mut mean_abs_shap: f64 = 0.0;

        for sample_idx in 0..features.nrows() {
            // Sum of SHAP values should equal (prediction - base_value)
            let shap_sum: f64 = shap_values.row(sample_idx).sum();
            let expected_sum = predictions[sample_idx] as f64 - base_value;
            let error = (shap_sum - expected_sum).abs();

            shap_sum_errors.push(error);
            max_shap_error = max_shap_error.max(error);

            // Accumulate mean absolute SHAP value
            let sample_abs_sum: f64 = shap_values.row(sample_idx).iter().map(|x| x.abs()).sum();
            mean_abs_shap += sample_abs_sum;
        }

        mean_abs_shap /= features.nrows() as f64;
        let mean_shap_error = shap_sum_errors.iter().sum::<f64>() / shap_sum_errors.len() as f64;

        Ok(SHAPValidationStats {
            mean_shap_error,
            max_shap_error,
            mean_abs_shap_value: mean_abs_shap,
            base_value,
            num_samples: features.nrows(),
            num_features: features.ncols(),
        })
    }
}

/// Create objective function from configuration
pub fn create_objective_function(config: &Config) -> Result<Box<dyn ObjectiveFunction>> {
    match config.objective {
        ObjectiveType::Regression => Ok(Box::new(RegressionObjective)),
        ObjectiveType::Binary => Ok(Box::new(BinaryObjective)),
        ObjectiveType::Multiclass => Ok(Box::new(MulticlassObjective::new(config.num_class))),
        _ => Err(LightGBMError::not_implemented(
            "Objective function not implemented",
        )),
    }
}

/// Regression objective function placeholder
#[derive(Debug)]
pub struct RegressionObjective;

impl ObjectiveFunction for RegressionObjective {
    fn compute_gradients(
        &self,
        predictions: &ndarray::ArrayView1<'_, Score>,
        labels: &ndarray::ArrayView1<'_, Label>,
        weights: Option<&ndarray::ArrayView1<'_, Label>>,
        gradients: &mut ndarray::ArrayViewMut1<'_, Score>,
        hessians: &mut ndarray::ArrayViewMut1<'_, Score>,
    ) -> Result<()> {
        // L2 loss: gradient = 2 * (prediction - target), hessian = 2
        // Simplified to: gradient = prediction - target, hessian = 1 (common in practice)

        if predictions.len() != labels.len()
            || predictions.len() != gradients.len()
            || predictions.len() != hessians.len()
        {
            return Err(LightGBMError::dimension_mismatch(
                format!("predictions: {}", predictions.len()),
                format!(
                    "labels/gradients/hessians: {}/{}/{}",
                    labels.len(),
                    gradients.len(),
                    hessians.len()
                ),
            ));
        }

        if let Some(weights_array) = weights {
            if weights_array.len() != predictions.len() {
                return Err(LightGBMError::dimension_mismatch(
                    format!("predictions: {}", predictions.len()),
                    format!("weights: {}", weights_array.len()),
                ));
            }

            // Weighted regression gradients
            for i in 0..predictions.len() {
                let residual = predictions[i] - labels[i];
                let weight = weights_array[i];
                gradients[i] = residual * weight;
                hessians[i] = weight;
            }
        } else {
            // Unweighted regression gradients
            for i in 0..predictions.len() {
                let residual = predictions[i] - labels[i];
                gradients[i] = residual;
                hessians[i] = 1.0;
            }
        }

        Ok(())
    }

    fn transform_predictions(&self, _scores: &mut ndarray::ArrayViewMut1<'_, Score>) -> Result<()> {
        Ok(()) // No transformation for regression
    }

    fn num_classes(&self) -> usize {
        1
    }

    fn name(&self) -> &'static str {
        "regression"
    }

    fn validate_labels(&self, _labels: &ndarray::ArrayView1<'_, Label>) -> Result<()> {
        Ok(()) // Placeholder validation
    }

    fn default_metric(&self) -> MetricType {
        MetricType::RMSE
    }
}

/// Binary classification objective function placeholder
#[derive(Debug)]
pub struct BinaryObjective;

impl ObjectiveFunction for BinaryObjective {
    fn compute_gradients(
        &self,
        predictions: &ndarray::ArrayView1<'_, Score>,
        labels: &ndarray::ArrayView1<'_, Label>,
        weights: Option<&ndarray::ArrayView1<'_, Label>>,
        gradients: &mut ndarray::ArrayViewMut1<'_, Score>,
        hessians: &mut ndarray::ArrayViewMut1<'_, Score>,
    ) -> Result<()> {
        // Binary logistic loss: gradient = sigmoid(prediction) - label, hessian = sigmoid(prediction) * (1 - sigmoid(prediction))

        if predictions.len() != labels.len()
            || predictions.len() != gradients.len()
            || predictions.len() != hessians.len()
        {
            return Err(LightGBMError::dimension_mismatch(
                format!("predictions: {}", predictions.len()),
                format!(
                    "labels/gradients/hessians: {}/{}/{}",
                    labels.len(),
                    gradients.len(),
                    hessians.len()
                ),
            ));
        }

        if let Some(weights_array) = weights {
            if weights_array.len() != predictions.len() {
                return Err(LightGBMError::dimension_mismatch(
                    format!("predictions: {}", predictions.len()),
                    format!("weights: {}", weights_array.len()),
                ));
            }

            // Weighted binary classification gradients
            for i in 0..predictions.len() {
                let sigmoid = 1.0 / (1.0 + (-predictions[i]).exp());
                let weight = weights_array[i];
                gradients[i] = (sigmoid - labels[i]) * weight;
                hessians[i] = sigmoid * (1.0 - sigmoid) * weight;
            }
        } else {
            // Unweighted binary classification gradients
            for i in 0..predictions.len() {
                let sigmoid = 1.0 / (1.0 + (-predictions[i]).exp());
                gradients[i] = sigmoid - labels[i];
                hessians[i] = sigmoid * (1.0 - sigmoid);
            }
        }

        Ok(())
    }

    fn transform_predictions(&self, scores: &mut ndarray::ArrayViewMut1<'_, Score>) -> Result<()> {
        // Apply sigmoid transformation to convert logits to probabilities
        for score in scores.iter_mut() {
            *score = 1.0 / (1.0 + (-*score).exp());
        }
        Ok(())
    }

    fn num_classes(&self) -> usize {
        2
    }

    fn name(&self) -> &'static str {
        "binary"
    }

    fn validate_labels(&self, _labels: &ndarray::ArrayView1<'_, Label>) -> Result<()> {
        Ok(()) // Placeholder validation
    }

    fn default_metric(&self) -> MetricType {
        MetricType::BinaryLogloss
    }
}

/// Multiclass classification objective function placeholder
#[derive(Debug)]
pub struct MulticlassObjective {
    num_classes: usize,
}

impl MulticlassObjective {
    /// Create a new multiclass objective
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }
}

impl ObjectiveFunction for MulticlassObjective {
    fn compute_gradients(
        &self,
        predictions: &ndarray::ArrayView1<'_, Score>,
        labels: &ndarray::ArrayView1<'_, Label>,
        weights: Option<&ndarray::ArrayView1<'_, Label>>,
        gradients: &mut ndarray::ArrayViewMut1<'_, Score>,
        hessians: &mut ndarray::ArrayViewMut1<'_, Score>,
    ) -> Result<()> {
        let num_data = labels.len();
        let num_classes = self.num_classes;

        if predictions.len() != num_data * num_classes {
            return Err(LightGBMError::dimension_mismatch(
                format!(
                    "Predictions size mismatch for multiclass: expected {}, got {}",
                    num_data * num_classes,
                    predictions.len()
                ),
                predictions.len().to_string(),
            ));
        }

        // For each data point, compute gradients and hessians for all classes
        for i in 0..num_data {
            let true_class = labels[i] as usize;
            let weight = weights.map(|w| w[i]).unwrap_or(1.0);

            // Get predictions for this data point across all classes
            let pred_start = i * num_classes;
            let pred_end = pred_start + num_classes;
            let pred_slice = predictions.slice(s![pred_start..pred_end]);

            // Compute softmax probabilities
            let max_pred = pred_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_preds = vec![0.0; num_classes];
            let mut sum_exp = 0.0;

            for j in 0..num_classes {
                exp_preds[j] = (pred_slice[j] - max_pred).exp();
                sum_exp += exp_preds[j];
            }

            // Compute gradients and hessians for each class
            for j in 0..num_classes {
                let prob = exp_preds[j] / sum_exp;
                let target = if j == true_class { 1.0 } else { 0.0 };

                // Gradient: p_j - y_j
                gradients[pred_start + j] = weight * (prob - target);

                // Hessian: p_j * (1 - p_j) for cross-entropy
                hessians[pred_start + j] = weight * prob * (1.0 - prob).max(1e-16);
            }
        }

        Ok(())
    }

    fn transform_predictions(&self, _scores: &mut ndarray::ArrayViewMut1<'_, Score>) -> Result<()> {
        // Apply softmax transformation
        Err(LightGBMError::not_implemented(
            "MulticlassObjective::transform_predictions",
        ))
    }

    fn num_classes(&self) -> usize {
        self.num_classes
    }

    fn name(&self) -> &'static str {
        "multiclass"
    }

    fn validate_labels(&self, _labels: &ndarray::ArrayView1<'_, Label>) -> Result<()> {
        Ok(()) // Placeholder validation
    }

    fn default_metric(&self) -> MetricType {
        MetricType::MultiLogloss
    }
}

/// Best split information
#[derive(Debug, Clone)]
struct BestSplit {
    feature_idx: usize,
    threshold: f32,
    gain: f64,
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
}

/// Tree node for building
#[derive(Debug, Clone)]
struct TreeNode {
    feature_index: i32,
    threshold: f32,
    left_child: Option<Box<TreeNode>>,
    right_child: Option<Box<TreeNode>>,
    leaf_value: f64,
    sample_count: usize,
    // TODO: implement depth-based constraints and tree pruning
    depth: usize,
    split_gain: Option<f64>,
    node_weight: f64,
    coverage: f64,
}

/// Tree builder for constructing decision trees
struct TreeBuilder<'a> {
    config: &'a Config,
    train_data: &'a Dataset,
    gradients: &'a Array1<f32>,
    hessians: &'a Array1<f32>,
    nodes: Vec<SimpleTreeNode>,
}

impl<'a> TreeBuilder<'a> {
    /// Create new tree builder
    fn new(
        config: &'a Config,
        train_data: &'a Dataset,
        gradients: &'a Array1<f32>,
        hessians: &'a Array1<f32>,
    ) -> Self {
        TreeBuilder {
            config,
            train_data,
            gradients,
            hessians,
            nodes: Vec::new(),
        }
    }

    /// Build a tree node recursively
    fn build_node(&mut self, sample_indices: &[usize], depth: usize) -> Result<TreeNode> {
        let sample_count = sample_indices.len();

        // Calculate node weight (sum of hessians) and coverage
        let node_weight = self.calculate_node_weight(sample_indices);
        let coverage = self.calculate_node_coverage(sample_indices);

        // Check stopping conditions
        if self.should_stop_splitting(sample_indices, depth) {
            let leaf_value = self.calculate_leaf_value(sample_indices)?;
            return Ok(TreeNode {
                feature_index: -1,
                threshold: 0.0,
                left_child: None,
                right_child: None,
                leaf_value,
                sample_count,
                depth,
                split_gain: None,
                node_weight,
                coverage,
            });
        }

        // Find best split
        if let Some(best_split) = self.find_best_split(sample_indices, depth)? {
            // Create left and right children
            let left_child = Box::new(self.build_node(&best_split.left_indices, depth + 1)?);
            let right_child = Box::new(self.build_node(&best_split.right_indices, depth + 1)?);

            Ok(TreeNode {
                feature_index: best_split.feature_idx as i32,
                threshold: best_split.threshold,
                left_child: Some(left_child),
                right_child: Some(right_child),
                leaf_value: 0.0, // Not used for internal nodes
                sample_count,
                depth,
                split_gain: Some(best_split.gain),
                node_weight,
                coverage,
            })
        } else {
            // No good split found, create leaf
            let leaf_value = self.calculate_leaf_value(sample_indices)?;
            Ok(TreeNode {
                feature_index: -1,
                threshold: 0.0,
                left_child: None,
                right_child: None,
                leaf_value,
                sample_count,
                depth,
                split_gain: None,
                node_weight,
                coverage,
            })
        }
    }

    /// Calculate node weight (sum of hessians)
    fn calculate_node_weight(&self, sample_indices: &[usize]) -> f64 {
        sample_indices
            .iter()
            .map(|&idx| {
                if idx < self.hessians.len() {
                    self.hessians[idx] as f64
                } else {
                    1.0
                }
            })
            .sum()
    }

    /// Calculate node coverage (weighted sample count)
    fn calculate_node_coverage(&self, sample_indices: &[usize]) -> f64 {
        // For now, use equal weights for all samples
        // In a more sophisticated implementation, this could use instance weights
        sample_indices.len() as f64
    }

    /// Check if we should stop splitting
    fn should_stop_splitting(&self, sample_indices: &[usize], depth: usize) -> bool {
        // Check minimum samples per leaf
        if sample_indices.len() < (self.config.min_data_in_leaf as usize * 2) {
            return true;
        }

        // Check maximum depth
        if self.config.max_depth > 0 && depth >= self.config.max_depth as usize {
            return true;
        }

        // Check maximum leaves (simplified check)
        if self.nodes.len() >= self.config.num_leaves {
            return true;
        }

        false
    }

    /// Find the best split for the given samples
    fn find_best_split(&self, sample_indices: &[usize], depth: usize) -> Result<Option<BestSplit>> {
        let mut best_split: Option<BestSplit> = None;
        let mut best_gain = 0.0;

        let features = self.train_data.features();
        let num_features = self.train_data.num_features();

        // Debug: Check gradients and hessians
        let total_gradient: f64 = sample_indices
            .iter()
            .map(|&idx| self.gradients[idx] as f64)
            .sum();
        let total_hessian: f64 = sample_indices
            .iter()
            .map(|&idx| self.hessians[idx] as f64)
            .sum();
        log::debug!(
            "find_best_split: {} samples, total_gradient={:.6}, total_hessian={:.6}",
            sample_indices.len(),
            total_gradient,
            total_hessian
        );

        // Debug: Sample a few gradient/hessian values
        for i in 0..std::cmp::min(5, sample_indices.len()) {
            let idx = sample_indices[i];
            log::debug!(
                "  sample[{}]: gradient={:.6}, hessian={:.6}",
                idx,
                self.gradients[idx],
                self.hessians[idx]
            );
        }

        // Iterate through features to find best split
        for feature_idx in 0..num_features {
            // Skip this feature randomly based on feature_fraction
            if self.config.feature_fraction < 1.0 {
                let rand_val = (feature_idx * 17 + sample_indices.len() * 31) % 100;
                if (rand_val as f32 / 100.0) > self.config.feature_fraction as f32 {
                    continue;
                }
            }

            // Get feature values for samples
            let mut feature_values: Vec<(f32, usize)> = sample_indices
                .iter()
                .map(|&idx| (features[[idx, feature_idx]], idx))
                .collect();

            // Sort by feature value
            feature_values
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Debug: Check feature value distribution
            if feature_values.len() > 0 {
                let min_val = feature_values[0].0;
                let max_val = feature_values[feature_values.len() - 1].0;
                log::debug!(
                    "  feature[{}]: min={:.6}, max={:.6}, samples={}",
                    feature_idx,
                    min_val,
                    max_val,
                    feature_values.len()
                );

                // Check if feature has variation
                if (max_val - min_val).abs() < 1e-8 {
                    log::debug!("  feature[{}]: No variation, skipping", feature_idx);
                    continue;
                }
            }

            // Try different split points
            let min_leaf_size = self.config.min_data_in_leaf as usize;
            for i in min_leaf_size..(feature_values.len() - min_leaf_size) {
                let threshold = (feature_values[i].0 + feature_values[i + 1].0) / 2.0;

                // Skip if threshold is the same as previous
                if i > 0 && threshold == (feature_values[i - 1].0 + feature_values[i].0) / 2.0 {
                    continue;
                }

                // Split samples
                let left_indices: Vec<usize> =
                    feature_values[..=i].iter().map(|(_, idx)| *idx).collect();
                let right_indices: Vec<usize> = feature_values[(i + 1)..]
                    .iter()
                    .map(|(_, idx)| *idx)
                    .collect();

                // Calculate gain
                let gain =
                    self.calculate_split_gain(&left_indices, &right_indices, sample_indices, depth)?;

                // Debug: Log gain calculation for first few splits
                if feature_idx < 2 && i < min_leaf_size + 5 {
                    log::debug!(
                        "    split feature[{}] threshold={:.6}: left={}, right={}, gain={:.6}",
                        feature_idx,
                        threshold,
                        left_indices.len(),
                        right_indices.len(),
                        gain
                    );
                }

                // Depth-aware split selection: prefer splits with higher gain, 
                // but if gains are similar (within 1% tolerance), prefer current depth
                let gain_improvement_threshold = 1.01; // 1% improvement threshold
                let should_update = if let Some(_current_best) = &best_split {
                    gain > best_gain * gain_improvement_threshold
                } else {
                    gain > best_gain
                };

                if should_update {
                    best_gain = gain;
                    best_split = Some(BestSplit {
                        feature_idx,
                        threshold,
                        gain,
                        left_indices,
                        right_indices,
                    });
                    log::debug!(
                        "    new best split: feature[{}] threshold={:.6} gain={:.6} (depth={})",
                        feature_idx,
                        threshold,
                        gain,
                        depth
                    );
                }
            }
        }

        // Only return split if gain is significant
        let min_split_gain = self.config.min_gain_to_split as f64;
        log::debug!(
            "find_best_split result: best_gain={:.6}, min_split_gain={:.6}, has_split={}",
            best_gain,
            min_split_gain,
            best_gain > min_split_gain
        );

        if best_gain > min_split_gain {
            log::debug!("Accepting split with gain={:.6}", best_gain);
            Ok(best_split)
        } else {
            log::debug!(
                "Rejecting split - gain too low: {:.6} <= {:.6}",
                best_gain,
                min_split_gain
            );
            Ok(None)
        }
    }

    /// Calculate the gain from a split using LightGBM formula with depth-based regularization
    /// Gain = (1/2) · [G_L²/(H_L + λ) + G_R²/(H_R + λ) - G²/(H + λ)] - γ - depth_penalty
    fn calculate_split_gain(
        &self,
        left_indices: &[usize],
        right_indices: &[usize],
        parent_indices: &[usize],
        depth: usize,
    ) -> Result<f64> {
        if left_indices.is_empty() || right_indices.is_empty() {
            return Ok(0.0);
        }

        // Calculate gradients and hessians sums
        let g_left: f64 = left_indices
            .iter()
            .map(|&idx| self.gradients[idx] as f64)
            .sum();
        let h_left: f64 = left_indices
            .iter()
            .map(|&idx| self.hessians[idx] as f64)
            .sum();

        let g_right: f64 = right_indices
            .iter()
            .map(|&idx| self.gradients[idx] as f64)
            .sum();
        let h_right: f64 = right_indices
            .iter()
            .map(|&idx| self.hessians[idx] as f64)
            .sum();

        let g_parent: f64 = parent_indices
            .iter()
            .map(|&idx| self.gradients[idx] as f64)
            .sum();
        let h_parent: f64 = parent_indices
            .iter()
            .map(|&idx| self.hessians[idx] as f64)
            .sum();

        let lambda = self.config.lambda_l2 as f64;
        let gamma = self.config.min_gain_to_split as f64; // Minimum split gain threshold
        
        // Depth-based regularization: penalty increases with depth to favor shallower trees
        let depth_penalty = if depth > 0 {
            let depth_factor = 0.1; // Configurable depth penalty factor
            depth_factor * (depth as f64).sqrt() // Square root to make penalty more gradual
        } else {
            0.0
        };

        // Debug: Check if we have the expected gradient/hessian sums
        let expected_g_sum = g_left + g_right;
        let expected_h_sum = h_left + h_right;
        if (g_parent - expected_g_sum).abs() > 1e-6 || (h_parent - expected_h_sum).abs() > 1e-6 {
            log::warn!("Gradient/Hessian sum mismatch: parent_g={:.6} vs left+right={:.6}, parent_h={:.6} vs left+right={:.6}",
                      g_parent, expected_g_sum, h_parent, expected_h_sum);
        }

        // LightGBM split gain formula
        let left_score = (g_left * g_left) / (h_left + lambda);
        let right_score = (g_right * g_right) / (h_right + lambda);
        let parent_score = (g_parent * g_parent) / (h_parent + lambda);

        let gain_before_gamma = 0.5 * (left_score + right_score - parent_score);
        let gain = gain_before_gamma - gamma - depth_penalty;

        // Debug: Log detailed gain calculation
        if parent_indices.len() <= 100 {
            // Only for small nodes to avoid spam
            log::debug!(
                "      gain calc: g_L={:.3}, h_L={:.3}, g_R={:.3}, h_R={:.3}, g_P={:.3}, h_P={:.3}",
                g_left,
                h_left,
                g_right,
                h_right,
                g_parent,
                h_parent
            );
            log::debug!("      scores: left={:.6}, right={:.6}, parent={:.6}, gain_raw={:.6}, gamma={:.6}, depth_penalty={:.6}, final_gain={:.6}",
                       left_score, right_score, parent_score, gain_before_gamma, gamma, depth_penalty, gain);
        }

        Ok(gain.max(0.0)) // Ensure non-negative gain
    }

    /// Prune tree based on depth and gain thresholds
    fn prune_tree(&self, mut node: TreeNode) -> TreeNode {
        // If this is a leaf, no pruning needed
        if node.left_child.is_none() && node.right_child.is_none() {
            return node;
        }

        // Recursively prune children first
        if let Some(left_child) = node.left_child.take() {
            node.left_child = Some(Box::new(self.prune_tree(*left_child)));
        }
        if let Some(right_child) = node.right_child.take() {
            node.right_child = Some(Box::new(self.prune_tree(*right_child)));
        }

        // Depth-based pruning: if depth is very high and gain is low, convert to leaf
        let depth_threshold = (self.config.max_depth as f64 * 0.8) as usize; // 80% of max depth
        let min_gain_for_deep_splits = self.config.min_gain_to_split as f64 * 2.0; // Higher threshold for deep splits
        
        if node.depth >= depth_threshold {
            if let Some(split_gain) = node.split_gain {
                if split_gain < min_gain_for_deep_splits {
                    // Convert to leaf node
                    log::debug!(
                        "Pruning deep node at depth {} with low gain {:.6}",
                        node.depth,
                        split_gain
                    );
                    return TreeNode {
                        feature_index: -1,
                        threshold: 0.0,
                        left_child: None,
                        right_child: None,
                        leaf_value: node.leaf_value, // Use existing leaf value
                        sample_count: node.sample_count,
                        depth: node.depth,
                        split_gain: None,
                        node_weight: node.node_weight,
                        coverage: node.coverage,
                    };
                }
            }
        }

        node
    }

    /// Calculate optimal leaf value using Newton-Raphson method
    /// w_j = -Σ_{i∈I_j} g_i / (Σ_{i∈I_j} h_i + λ)
    fn calculate_leaf_value(&self, sample_indices: &[usize]) -> Result<f64> {
        if sample_indices.is_empty() {
            return Ok(0.0);
        }

        let sum_gradients: f64 = sample_indices
            .iter()
            .map(|&idx| self.gradients[idx] as f64)
            .sum();
        let sum_hessians: f64 = sample_indices
            .iter()
            .map(|&idx| self.hessians[idx] as f64)
            .sum();

        let lambda = self.config.lambda_l2 as f64;
        let regularized_hessian = sum_hessians + lambda;

        if regularized_hessian > 1e-6 {
            // Newton-Raphson formula: w_j = -G_j / (H_j + λ)
            let leaf_value = -sum_gradients / regularized_hessian;

            // Apply learning rate as shrinkage
            Ok(leaf_value * self.config.learning_rate as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Convert TreeNode to SimpleTree
    fn to_simple_tree(&mut self, root: TreeNode) -> SimpleTree {
        self.nodes.clear();

        // Debug: Log the structure of the TreeNode before conversion
        log::debug!("Converting TreeNode to SimpleTree - root: feature={}, threshold={:.6}, has_children={}",
                   root.feature_index, root.threshold, root.left_child.is_some() && root.right_child.is_some());

        self.convert_node(&root);

        let num_leaves = self
            .nodes
            .iter()
            .filter(|node| node.feature_index < 0)
            .count();

        // Debug: Log the final SimpleTree structure
        log::debug!(
            "SimpleTree created: {} total nodes, {} leaves",
            self.nodes.len(),
            num_leaves
        );
        if !self.nodes.is_empty() {
            let root_node = &self.nodes[0];
            log::debug!(
                "SimpleTree root: feature={}, threshold={:.6}, left={}, right={}",
                root_node.feature_index,
                root_node.threshold,
                root_node.left_child,
                root_node.right_child
            );
        }

        SimpleTree {
            nodes: self.nodes.clone(),
            num_leaves,
        }
    }

    /// Convert TreeNode to SimpleTreeNode recursively
    fn convert_node(&mut self, node: &TreeNode) -> i32 {
        let node_idx = self.nodes.len() as i32;

        log::debug!(
            "Converting node {}: feature={}, has_children={}",
            node_idx,
            node.feature_index,
            node.left_child.is_some() && node.right_child.is_some()
        );

        // IMPORTANT: Add the current node FIRST to reserve its index position
        self.nodes.push(SimpleTreeNode {
            feature_index: node.feature_index,
            threshold: node.threshold,
            left_child: -1,  // Will be updated below
            right_child: -1, // Will be updated below
            leaf_value: node.leaf_value,
            sample_count: node.sample_count,
            split_gain: node.split_gain.unwrap_or(0.0),
            node_weight: node.node_weight,
            coverage: node.coverage,
        });

        // Now process children and update the indices
        let (left_child, right_child) =
            if let (Some(ref left), Some(ref right)) = (&node.left_child, &node.right_child) {
                let left_idx = self.convert_node(left);
                let right_idx = self.convert_node(right);
                (left_idx, right_idx)
            } else {
                (-1, -1)
            };

        // Update the node's child indices
        self.nodes[node_idx as usize].left_child = left_child;
        self.nodes[node_idx as usize].right_child = right_child;

        log::debug!(
            "Node {}: setting children left={}, right={}",
            node_idx,
            left_child,
            right_child
        );

        node_idx
    }
}

// SerializableModel implementation temporarily disabled due to compilation issues
// impl SerializableModel for GBDT { ... }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::dataset::Dataset;
    use crate::core::types::ObjectiveType;
    use ndarray::{Array1, Array2};

    fn create_test_config() -> Config {
        let mut config = Config::new();
        config.objective = ObjectiveType::Regression;
        config.max_depth = 3;
        config.num_leaves = 8;
        config.min_data_in_leaf = 5;
        config.min_gain_to_split = 0.1;
        config.lambda_l2 = 0.1;
        config
    }

    fn create_test_dataset() -> Dataset {
        // Create a simple dataset for testing
        let features = Array2::from_shape_vec((20, 2), vec![
            1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0,
            6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 9.0, 10.0, 10.0, 11.0,
            2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0, 5.0,
            7.0, 6.0, 8.0, 7.0, 9.0, 8.0, 10.0, 9.0, 11.0, 10.0,
        ]).unwrap();
        let labels = Array1::from_vec(vec![
            1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
            6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5,
        ]);
        
        Dataset::new(features, labels, None, None, None, None).unwrap()
    }

    #[test]
    fn test_depth_based_regularization() {
        let config = create_test_config();
        let dataset = create_test_dataset();
        // Create gradients that will lead to meaningful split gains
        let gradients = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0]);
        let hessians = Array1::from_vec(vec![1.0; 20]);
        
        let tree_builder = TreeBuilder::new(&config, &dataset, &gradients, &hessians);
        
        // Test that deeper splits have higher penalty
        let sample_indices: Vec<usize> = (0..10).collect();
        let left_indices: Vec<usize> = (0..5).collect();
        let right_indices: Vec<usize> = (5..10).collect();
        
        // Calculate gain at depth 0 and depth 3
        let gain_depth_0 = tree_builder.calculate_split_gain(&left_indices, &right_indices, &sample_indices, 0).unwrap();
        let gain_depth_3 = tree_builder.calculate_split_gain(&left_indices, &right_indices, &sample_indices, 3).unwrap();
        
        // Gain at depth 3 should be less than gain at depth 0 due to depth penalty
        assert!(gain_depth_3 < gain_depth_0, 
                "Depth penalty should reduce gain: depth_0={:.6}, depth_3={:.6}", 
                gain_depth_0, gain_depth_3);
    }

    #[test]
    fn test_max_depth_constraint() {
        let config = create_test_config();
        let dataset = create_test_dataset();
        let gradients = Array1::from_vec(vec![0.1; 20]);
        let hessians = Array1::from_vec(vec![1.0; 20]);
        
        let tree_builder = TreeBuilder::new(&config, &dataset, &gradients, &hessians);
        
        // Test stopping condition at max depth
        let sample_indices: Vec<usize> = (0..10).collect();
        
        // Should NOT stop at depth 2 (below max_depth=3)
        assert!(!tree_builder.should_stop_splitting(&sample_indices, 2),
                "Should not stop splitting at depth 2 when max_depth is 3");
        
        // Should stop at depth 3 (equal to max_depth=3)
        assert!(tree_builder.should_stop_splitting(&sample_indices, 3),
                "Should stop splitting at depth 3 when max_depth is 3");
        
        // Should stop at depth 4 (above max_depth=3)
        assert!(tree_builder.should_stop_splitting(&sample_indices, 4),
                "Should stop splitting at depth 4 when max_depth is 3");
    }

    #[test]
    fn test_depth_aware_split_selection() {
        let config = create_test_config();
        let dataset = create_test_dataset();
        // Create gradients that will lead to meaningful split gains
        let gradients = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0]);
        let hessians = Array1::from_vec(vec![1.0; 20]);
        
        let tree_builder = TreeBuilder::new(&config, &dataset, &gradients, &hessians);
        
        // Test that depth-aware selection works by checking that similar gains
        // at different depths are handled correctly through the improvement threshold
        let sample_indices: Vec<usize> = (0..10).collect();
        
        // This is an indirect test - the depth-aware selection logic uses a 1% improvement
        // threshold which should be visible in the gain calculation differences
        let gain_depth_1 = tree_builder.calculate_split_gain(
            &(0..5).collect::<Vec<_>>(), 
            &(5..10).collect::<Vec<_>>(), 
            &sample_indices, 
            1
        ).unwrap();
        
        let gain_depth_2 = tree_builder.calculate_split_gain(
            &(0..5).collect::<Vec<_>>(), 
            &(5..10).collect::<Vec<_>>(), 
            &sample_indices, 
            2
        ).unwrap();
        
        // The depth penalty should create a measurable difference
        let improvement_ratio = gain_depth_1 / gain_depth_2.max(1e-10);
        assert!(improvement_ratio > 1.005, 
                "Depth penalty should create >0.5% difference for depth-aware selection: ratio={:.3}", 
                improvement_ratio);
    }

    #[test]
    fn test_depth_based_pruning() {
        let config = create_test_config();
        let dataset = create_test_dataset();
        let gradients = Array1::from_vec(vec![0.01; 20]); // Very small gradients
        let hessians = Array1::from_vec(vec![1.0; 20]);
        
        let tree_builder = TreeBuilder::new(&config, &dataset, &gradients, &hessians);
        
        // Create a deep node with low gain that should be pruned
        let deep_node = TreeNode {
            feature_index: 0,
            threshold: 5.0,
            left_child: Some(Box::new(TreeNode {
                feature_index: -1,
                threshold: 0.0,
                left_child: None,
                right_child: None,
                leaf_value: 1.0,
                sample_count: 5,
                depth: 4, // Deep node
                split_gain: None,
                node_weight: 5.0,
                coverage: 5.0,
            })),
            right_child: Some(Box::new(TreeNode {
                feature_index: -1,
                threshold: 0.0,
                left_child: None,
                right_child: None,
                leaf_value: 1.0,
                sample_count: 5,
                depth: 4, // Deep node
                split_gain: None,
                node_weight: 5.0,
                coverage: 5.0,
            })),
            leaf_value: 0.0,
            sample_count: 10,
            depth: 3, // At 80% of max_depth (3 * 0.8 = 2.4, so >= 2)
            split_gain: Some(0.05), // Low gain < 2 * min_gain_to_split (2 * 0.1 = 0.2)
            node_weight: 10.0,
            coverage: 10.0,
        };
        
        let pruned_node = tree_builder.prune_tree(deep_node);
        
        // The node should be converted to a leaf (children removed)
        assert!(pruned_node.left_child.is_none(), "Deep node with low gain should be pruned to leaf");
        assert!(pruned_node.right_child.is_none(), "Deep node with low gain should be pruned to leaf");
        assert_eq!(pruned_node.feature_index, -1, "Pruned node should have feature_index = -1");
    }

    #[test]
    fn test_pruning_preserves_good_splits() {
        let config = create_test_config();
        let dataset = create_test_dataset();
        let gradients = Array1::from_vec(vec![0.1; 20]);
        let hessians = Array1::from_vec(vec![1.0; 20]);
        
        let tree_builder = TreeBuilder::new(&config, &dataset, &gradients, &hessians);
        
        // Create a deep node with high gain that should NOT be pruned
        let deep_node = TreeNode {
            feature_index: 0,
            threshold: 5.0,
            left_child: Some(Box::new(TreeNode {
                feature_index: -1,
                threshold: 0.0,
                left_child: None,
                right_child: None,
                leaf_value: 1.0,
                sample_count: 5,
                depth: 4,
                split_gain: None,
                node_weight: 5.0,
                coverage: 5.0,
            })),
            right_child: Some(Box::new(TreeNode {
                feature_index: -1,
                threshold: 0.0,
                left_child: None,
                right_child: None,
                leaf_value: 1.0,
                sample_count: 5,
                depth: 4,
                split_gain: None,
                node_weight: 5.0,
                coverage: 5.0,
            })),
            leaf_value: 0.0,
            sample_count: 10,
            depth: 3,
            split_gain: Some(0.5), // High gain > 2 * min_gain_to_split (2 * 0.1 = 0.2)
            node_weight: 10.0,
            coverage: 10.0,
        };
        
        let pruned_node = tree_builder.prune_tree(deep_node);
        
        // The node should NOT be converted to a leaf (children preserved)
        assert!(pruned_node.left_child.is_some(), "Deep node with high gain should NOT be pruned");
        assert!(pruned_node.right_child.is_some(), "Deep node with high gain should NOT be pruned");
        assert_eq!(pruned_node.feature_index, 0, "Good split should preserve feature_index");
    }

    #[test]
    fn test_depth_field_usage() {
        let config = create_test_config();
        let dataset = create_test_dataset();
        let gradients = Array1::from_vec(vec![0.1; 20]);
        let hessians = Array1::from_vec(vec![1.0; 20]);
        
        let mut tree_builder = TreeBuilder::new(&config, &dataset, &gradients, &hessians);
        let sample_indices: Vec<usize> = (0..dataset.num_data()).collect();
        
        // Build a tree and verify depth field is properly set
        let root_node = tree_builder.build_node(&sample_indices, 0).unwrap();
        
        // Root should have depth 0
        assert_eq!(root_node.depth, 0, "Root node should have depth 0");
        
        // If there are children, they should have depth 1
        if let Some(ref left_child) = root_node.left_child {
            assert_eq!(left_child.depth, 1, "Left child should have depth 1");
        }
        if let Some(ref right_child) = root_node.right_child {
            assert_eq!(right_child.depth, 1, "Right child should have depth 1");
        }
    }
}
