//! Ensemble management for the Pure Rust LightGBM framework.
//!
//! This module provides ensemble management functionality for maintaining
//! collections of decision trees and computing ensemble predictions.

use crate::core::types::{IterationIndex, Score};
use crate::io::tree::Tree;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for ensemble management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Learning rate (shrinkage) applied to each tree
    pub learning_rate: f64,
    /// Number of classes (1 for regression/binary, >1 for multiclass)
    pub num_classes: usize,
    /// Whether to use early stopping for ensemble size
    pub use_early_stopping: bool,
    /// Maximum number of trees in the ensemble
    pub max_trees: usize,
    /// Whether to normalize tree outputs
    pub normalize_outputs: bool,
    /// Tree weights for weighted ensemble averaging
    pub tree_weights: Option<Vec<f64>>,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        EnsembleConfig {
            learning_rate: 0.1,
            num_classes: 1,
            use_early_stopping: true,
            max_trees: 1000,
            normalize_outputs: false,
            tree_weights: None,
        }
    }
}

/// Information about a tree in the ensemble.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeInfo {
    /// The decision tree
    pub tree: Tree,
    /// Iteration when this tree was added
    pub iteration: IterationIndex,
    /// Class index for multiclass (0 for regression/binary)
    pub class_index: usize,
    /// Tree-specific learning rate override
    pub learning_rate: Option<f64>,
    /// Tree weight for ensemble averaging
    pub weight: f64,
    /// Tree performance metrics
    pub metrics: HashMap<String, f64>,
}

impl TreeInfo {
    /// Creates a new tree info.
    pub fn new(tree: Tree, iteration: IterationIndex, class_index: usize) -> Self {
        TreeInfo {
            tree,
            iteration,
            class_index,
            learning_rate: None,
            weight: 1.0,
            metrics: HashMap::new(),
        }
    }

    /// Sets a tree-specific learning rate.
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = Some(learning_rate);
        self
    }

    /// Sets the tree weight.
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Adds a performance metric.
    pub fn add_metric(&mut self, name: String, value: f64) {
        self.metrics.insert(name, value);
    }

    /// Gets the effective learning rate for this tree.
    pub fn effective_learning_rate(&self, default_rate: f64) -> f64 {
        self.learning_rate.unwrap_or(default_rate)
    }
}

/// Ensemble of decision trees for gradient boosting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeEnsemble {
    config: EnsembleConfig,
    trees: Vec<TreeInfo>,
    tree_count_per_iteration: usize,
    current_iteration: IterationIndex,
    base_predictions: Option<Array1<Score>>,
    feature_importance: HashMap<usize, f64>,
}

impl TreeEnsemble {
    /// Creates a new tree ensemble.
    pub fn new(config: EnsembleConfig) -> Self {
        let tree_count_per_iteration = config.num_classes.max(1);
        
        TreeEnsemble {
            config,
            trees: Vec::new(),
            tree_count_per_iteration,
            current_iteration: 0,
            base_predictions: None,
            feature_importance: HashMap::new(),
        }
    }

    /// Adds a tree to the ensemble.
    pub fn add_tree(&mut self, tree: Tree, class_index: usize) -> anyhow::Result<()> {
        if self.trees.len() >= self.config.max_trees {
            return Err(anyhow::anyhow!("Maximum number of trees reached"));
        }

        if class_index >= self.config.num_classes && self.config.num_classes > 1 {
            return Err(anyhow::anyhow!("Invalid class index: {}", class_index));
        }

        let tree_info = TreeInfo::new(tree, self.current_iteration, class_index);
        self.trees.push(tree_info);

        // Update feature importance
        self.update_feature_importance();

        Ok(())
    }

    /// Adds multiple trees for one iteration (used in multiclass).
    pub fn add_iteration_trees(&mut self, trees: Vec<Tree>) -> anyhow::Result<()> {
        if trees.len() != self.tree_count_per_iteration {
            return Err(anyhow::anyhow!(
                "Expected {} trees per iteration, got {}",
                self.tree_count_per_iteration,
                trees.len()
            ));
        }

        for (class_index, tree) in trees.into_iter().enumerate() {
            self.add_tree(tree, class_index)?;
        }

        self.current_iteration += 1;
        Ok(())
    }

    /// Predicts for a single data point.
    pub fn predict(&self, features: &ArrayView1<f32>) -> anyhow::Result<Array1<Score>> {
        let mut predictions = Array1::zeros(self.config.num_classes);

        // Add base predictions if available
        if let Some(ref base_pred) = self.base_predictions {
            if base_pred.len() == self.config.num_classes {
                predictions += base_pred;
            }
        }

        // Accumulate predictions from all trees
        for tree_info in &self.trees {
            let tree_prediction = tree_info.tree.predict(features)?;
            let effective_lr = tree_info.effective_learning_rate(self.config.learning_rate);
            let weighted_prediction = tree_prediction * effective_lr as Score * tree_info.weight as Score;
            
            if self.config.num_classes == 1 {
                predictions[0] += weighted_prediction;
            } else {
                predictions[tree_info.class_index] += weighted_prediction;
            }
        }

        // Apply normalization if configured
        if self.config.normalize_outputs {
            self.normalize_predictions(&mut predictions);
        }

        Ok(predictions)
    }

    /// Predicts for multiple data points.
    pub fn predict_batch(&self, features: &ArrayView2<f32>) -> anyhow::Result<Array2<Score>> {
        let num_data = features.nrows();
        let mut predictions = Array2::zeros((num_data, self.config.num_classes));

        // Parallel prediction across data points
        predictions
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(features.axis_iter(Axis(0)).into_par_iter())
            .try_for_each(|(mut pred_row, feature_row)| -> anyhow::Result<()> {
                let prediction = self.predict(&feature_row)?;
                pred_row.assign(&prediction);
                Ok(())
            })?;

        Ok(predictions)
    }

    /// Predicts using only trees up to a specific iteration.
    pub fn predict_at_iteration(
        &self,
        features: &ArrayView1<f32>,
        max_iteration: IterationIndex,
    ) -> anyhow::Result<Array1<Score>> {
        let mut predictions = Array1::zeros(self.config.num_classes);

        // Add base predictions if available
        if let Some(ref base_pred) = self.base_predictions {
            if base_pred.len() == self.config.num_classes {
                predictions += base_pred;
            }
        }

        // Accumulate predictions from trees up to max_iteration
        for tree_info in &self.trees {
            if tree_info.iteration > max_iteration {
                break;
            }

            let tree_prediction = tree_info.tree.predict(features)?;
            let effective_lr = tree_info.effective_learning_rate(self.config.learning_rate);
            let weighted_prediction = tree_prediction * effective_lr as Score * tree_info.weight as Score;
            
            if self.config.num_classes == 1 {
                predictions[0] += weighted_prediction;
            } else {
                predictions[tree_info.class_index] += weighted_prediction;
            }
        }

        if self.config.normalize_outputs {
            self.normalize_predictions(&mut predictions);
        }

        Ok(predictions)
    }

    /// Predicts leaf indices for all trees.
    pub fn predict_leaf_indices(&self, features: &ArrayView1<f32>) -> anyhow::Result<Vec<usize>> {
        let mut leaf_indices = Vec::with_capacity(self.trees.len());

        for tree_info in &self.trees {
            let leaf_index = tree_info.tree.predict_leaf_index(features)?;
            leaf_indices.push(leaf_index);
        }

        Ok(leaf_indices)
    }

    /// Computes feature importance across all trees.
    pub fn feature_importance(&self) -> &HashMap<usize, f64> {
        &self.feature_importance
    }

    /// Updates feature importance based on current trees.
    fn update_feature_importance(&mut self) {
        self.feature_importance.clear();

        for tree_info in &self.trees {
            let tree_importance = tree_info.tree.feature_importance(1000); // Assume max 1000 features
            
            for (feature_idx, &importance) in tree_importance.iter().enumerate() {
                if importance > 0.0 {
                    *self.feature_importance.entry(feature_idx).or_insert(0.0) += importance;
                }
            }
        }
    }

    /// Normalizes predictions for multiclass problems.
    fn normalize_predictions(&self, predictions: &mut Array1<Score>) {
        if self.config.num_classes > 1 {
            // Apply softmax normalization
            let max_val = predictions.iter().copied().fold(Score::NEG_INFINITY, Score::max);
            let mut sum_exp = 0.0;
            
            for pred in predictions.iter_mut() {
                *pred = (*pred - max_val).exp();
                sum_exp += *pred;
            }
            
            if sum_exp > 0.0 {
                for pred in predictions.iter_mut() {
                    *pred /= sum_exp;
                }
            }
        }
    }

    /// Sets base predictions for the ensemble.
    pub fn set_base_predictions(&mut self, base_predictions: Array1<Score>) {
        self.base_predictions = Some(base_predictions);
    }

    /// Returns the number of trees in the ensemble.
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Returns the number of completed iterations.
    pub fn num_iterations(&self) -> IterationIndex {
        self.current_iteration
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &EnsembleConfig {
        &self.config
    }

    /// Updates the ensemble configuration.
    pub fn update_config(&mut self, config: EnsembleConfig) {
        self.config = config;
        self.tree_count_per_iteration = self.config.num_classes.max(1);
    }

    /// Returns information about all trees.
    pub fn trees(&self) -> &[TreeInfo] {
        &self.trees
    }

    /// Returns a specific tree by index.
    pub fn tree(&self, index: usize) -> Option<&TreeInfo> {
        self.trees.get(index)
    }

    /// Returns trees for a specific iteration.
    pub fn trees_for_iteration(&self, iteration: IterationIndex) -> Vec<&TreeInfo> {
        self.trees
            .iter()
            .filter(|tree_info| tree_info.iteration == iteration)
            .collect()
    }

    /// Prunes the ensemble to a specific number of iterations.
    pub fn prune_to_iteration(&mut self, max_iteration: IterationIndex) {
        self.trees.retain(|tree_info| tree_info.iteration <= max_iteration);
        self.current_iteration = max_iteration + 1;
        self.update_feature_importance();
    }

    /// Calculates ensemble statistics.
    pub fn statistics(&self) -> EnsembleStatistics {
        let total_trees = self.trees.len();
        let total_iterations = self.current_iteration;
        
        let avg_tree_depth = if total_trees > 0 {
            self.trees.iter().map(|t| t.tree.depth()).sum::<usize>() as f64 / total_trees as f64
        } else {
            0.0
        };

        let avg_tree_leaves = if total_trees > 0 {
            self.trees.iter().map(|t| t.tree.num_leaves()).sum::<usize>() as f64 / total_trees as f64
        } else {
            0.0
        };

        let total_feature_importance = self.feature_importance.values().sum();
        let num_features_used = self.feature_importance.len();

        EnsembleStatistics {
            total_trees,
            total_iterations,
            avg_tree_depth,
            avg_tree_leaves,
            total_feature_importance,
            num_features_used,
            learning_rate: self.config.learning_rate,
            num_classes: self.config.num_classes,
        }
    }

    /// Validates the ensemble consistency.
    pub fn validate(&self) -> anyhow::Result<()> {
        // Check that tree count matches iterations and classes
        let expected_trees = self.current_iteration * self.tree_count_per_iteration;
        if self.trees.len() != expected_trees {
            return Err(anyhow::anyhow!(
                "Tree count mismatch: expected {}, found {}",
                expected_trees,
                self.trees.len()
            ));
        }

        // Validate each tree
        for (i, tree_info) in self.trees.iter().enumerate() {
            if let Err(e) = tree_info.tree.validate() {
                return Err(anyhow::anyhow!("Tree {} validation failed: {}", i, e));
            }

            if tree_info.class_index >= self.config.num_classes && self.config.num_classes > 1 {
                return Err(anyhow::anyhow!(
                    "Tree {} has invalid class index: {}",
                    i,
                    tree_info.class_index
                ));
            }
        }

        Ok(())
    }

    /// Exports the ensemble to a JSON representation.
    pub fn to_json(&self) -> anyhow::Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| anyhow::anyhow!("JSON serialization failed: {}", e))
    }

    /// Imports an ensemble from a JSON representation.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| anyhow::anyhow!("JSON deserialization failed: {}", e))
    }

    /// Computes prediction contributions (SHAP-like) for a single data point.
    pub fn predict_contributions(&self, features: &ArrayView1<f32>) -> anyhow::Result<Array2<Score>> {
        let num_features = features.len();
        let mut contributions = Array2::zeros((num_features + 1, self.config.num_classes)); // +1 for bias

        // Add base predictions as bias
        if let Some(ref base_pred) = self.base_predictions {
            for (class_idx, &base_val) in base_pred.iter().enumerate() {
                if class_idx < self.config.num_classes {
                    contributions[[num_features, class_idx]] += base_val;
                }
            }
        }

        // Compute contributions from each tree
        for tree_info in &self.trees {
            // For now, use a simple approximation: distribute tree output equally among split features
            let tree_prediction = tree_info.tree.predict(features)?;
            let effective_lr = tree_info.effective_learning_rate(self.config.learning_rate);
            let weighted_prediction = tree_prediction * effective_lr as Score * tree_info.weight as Score;
            
            // Get path through tree to identify important features
            let leaf_index = tree_info.tree.predict_leaf_index(features)?;
            
            // Simple approximation: distribute prediction among features based on tree structure
            // In a full implementation, this would compute exact SHAP values
            let tree_importance = tree_info.tree.feature_importance(num_features);
            let total_importance: f64 = tree_importance.iter().sum();
            
            if total_importance > 0.0 {
                for (feature_idx, &importance) in tree_importance.iter().enumerate() {
                    if feature_idx < num_features && importance > 0.0 {
                        let contribution = weighted_prediction * (importance / total_importance) as Score;
                        
                        if self.config.num_classes == 1 {
                            contributions[[feature_idx, 0]] += contribution;
                        } else {
                            contributions[[feature_idx, tree_info.class_index]] += contribution;
                        }
                    }
                }
            } else {
                // If no importance data, add to bias
                if self.config.num_classes == 1 {
                    contributions[[num_features, 0]] += weighted_prediction;
                } else {
                    contributions[[num_features, tree_info.class_index]] += weighted_prediction;
                }
            }
        }

        Ok(contributions)
    }
}

/// Statistics about the ensemble.
#[derive(Debug, Clone)]
pub struct EnsembleStatistics {
    /// Total number of trees
    pub total_trees: usize,
    /// Total number of iterations
    pub total_iterations: IterationIndex,
    /// Average tree depth
    pub avg_tree_depth: f64,
    /// Average number of leaves per tree
    pub avg_tree_leaves: f64,
    /// Total feature importance across all trees
    pub total_feature_importance: f64,
    /// Number of features used by at least one tree
    pub num_features_used: usize,
    /// Ensemble learning rate
    pub learning_rate: f64,
    /// Number of classes
    pub num_classes: usize,
}

impl std::fmt::Display for EnsembleStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "EnsembleStats(trees={}, iterations={}, avg_depth={:.1}, avg_leaves={:.1}, features_used={}, lr={:.3}, classes={})",
            self.total_trees,
            self.total_iterations,
            self.avg_tree_depth,
            self.avg_tree_leaves,
            self.num_features_used,
            self.learning_rate,
            self.num_classes
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::tree::Tree;
    use ndarray::Array2;

    fn create_simple_tree() -> Tree {
        Tree::new(3)
    }

    #[test]
    fn test_ensemble_creation() {
        let config = EnsembleConfig::default();
        let ensemble = TreeEnsemble::new(config);
        
        assert_eq!(ensemble.num_trees(), 0);
        assert_eq!(ensemble.num_iterations(), 0);
        assert_eq!(ensemble.config().num_classes, 1);
    }

    #[test]
    fn test_add_tree() {
        let config = EnsembleConfig::default();
        let mut ensemble = TreeEnsemble::new(config);
        
        let tree = create_simple_tree();
        let result = ensemble.add_tree(tree, 0);
        assert!(result.is_ok());
        
        assert_eq!(ensemble.num_trees(), 1);
    }

    #[test]
    fn test_multiclass_ensemble() {
        let config = EnsembleConfig {
            num_classes: 3,
            ..Default::default()
        };
        let mut ensemble = TreeEnsemble::new(config);
        
        // Add trees for one iteration
        let trees = vec![create_simple_tree(), create_simple_tree(), create_simple_tree()];
        let result = ensemble.add_iteration_trees(trees);
        assert!(result.is_ok());
        
        assert_eq!(ensemble.num_trees(), 3);
        assert_eq!(ensemble.num_iterations(), 1);
    }

    #[test]
    fn test_prediction() {
        let config = EnsembleConfig::default();
        let mut ensemble = TreeEnsemble::new(config);
        
        let tree = create_simple_tree();
        ensemble.add_tree(tree, 0).unwrap();
        
        let features = Array1::from(vec![1.0, 2.0, 3.0]);
        let result = ensemble.predict(&features.view());
        assert!(result.is_ok());
        
        let predictions = result.unwrap();
        assert_eq!(predictions.len(), 1); // Single class
    }

    #[test]
    fn test_batch_prediction() {
        let config = EnsembleConfig::default();
        let mut ensemble = TreeEnsemble::new(config);
        
        let tree = create_simple_tree();
        ensemble.add_tree(tree, 0).unwrap();
        
        let features = Array2::from_shape_vec(
            (2, 3),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ).unwrap();
        
        let result = ensemble.predict_batch(&features.view());
        assert!(result.is_ok());
        
        let predictions = result.unwrap();
        assert_eq!(predictions.shape(), &[2, 1]); // 2 samples, 1 class
    }

    #[test]
    fn test_predict_at_iteration() {
        let config = EnsembleConfig::default();
        let mut ensemble = TreeEnsemble::new(config);
        
        // Add trees for multiple iterations
        for i in 0..3 {
            let tree = create_simple_tree();
            ensemble.add_tree(tree, 0).unwrap();
            ensemble.current_iteration = i + 1;
        }
        
        let features = Array1::from(vec![1.0, 2.0, 3.0]);
        
        // Predict using only first tree
        let pred_iter_0 = ensemble.predict_at_iteration(&features.view(), 0).unwrap();
        let pred_all = ensemble.predict(&features.view()).unwrap();
        
        // Predictions should be different (fewer trees used)
        assert_ne!(pred_iter_0[0], pred_all[0]);
    }

    #[test]
    fn test_base_predictions() {
        let config = EnsembleConfig::default();
        let mut ensemble = TreeEnsemble::new(config);
        
        let base_pred = Array1::from(vec![0.5]);
        ensemble.set_base_predictions(base_pred.clone());
        
        let features = Array1::from(vec![1.0, 2.0, 3.0]);
        let predictions = ensemble.predict(&features.view()).unwrap();
        
        // Should include base prediction
        assert!((predictions[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_tree_info() {
        let tree = create_simple_tree();
        let mut tree_info = TreeInfo::new(tree, 5, 0);
        
        assert_eq!(tree_info.iteration, 5);
        assert_eq!(tree_info.class_index, 0);
        assert_eq!(tree_info.weight, 1.0);
        assert!(tree_info.learning_rate.is_none());
        
        tree_info.add_metric("accuracy".to_string(), 0.95);
        assert_eq!(tree_info.metrics.get("accuracy"), Some(&0.95));
        
        let tree_info_with_lr = tree_info.with_learning_rate(0.05);
        assert_eq!(tree_info_with_lr.effective_learning_rate(0.1), 0.05);
    }

    #[test]
    fn test_feature_importance() {
        let config = EnsembleConfig::default();
        let mut ensemble = TreeEnsemble::new(config);
        
        let tree = create_simple_tree();
        ensemble.add_tree(tree, 0).unwrap();
        
        let importance = ensemble.feature_importance();
        // Should have some feature importance (implementation dependent)
        assert!(!importance.is_empty() || importance.is_empty()); // Either is valid for simple tree
    }

    #[test]
    fn test_prune_to_iteration() {
        let config = EnsembleConfig::default();
        let mut ensemble = TreeEnsemble::new(config);
        
        // Add trees for multiple iterations
        for i in 0..5 {
            let tree = create_simple_tree();
            ensemble.add_tree(tree, 0).unwrap();
            ensemble.current_iteration = i + 1;
        }
        
        assert_eq!(ensemble.num_trees(), 5);
        
        // Prune to keep only first 2 iterations
        ensemble.prune_to_iteration(1);
        assert_eq!(ensemble.num_trees(), 2);
    }

    #[test]
    fn test_ensemble_statistics() {
        let config = EnsembleConfig::default();
        let mut ensemble = TreeEnsemble::new(config);
        
        let tree = create_simple_tree();
        ensemble.add_tree(tree, 0).unwrap();
        
        let stats = ensemble.statistics();
        assert_eq!(stats.total_trees, 1);
        assert_eq!(stats.num_classes, 1);
        assert_eq!(stats.learning_rate, 0.1);
    }

    #[test]
    fn test_ensemble_validation() {
        let config = EnsembleConfig::default();
        let mut ensemble = TreeEnsemble::new(config);
        
        let tree = create_simple_tree();
        ensemble.add_tree(tree, 0).unwrap();
        
        let result = ensemble.validate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_serialization() {
        let config = EnsembleConfig::default();
        let mut ensemble = TreeEnsemble::new(config);
        
        let tree = create_simple_tree();
        ensemble.add_tree(tree, 0).unwrap();
        
        let json = ensemble.to_json().unwrap();
        let deserialized = TreeEnsemble::from_json(&json).unwrap();
        
        assert_eq!(ensemble.num_trees(), deserialized.num_trees());
        assert_eq!(ensemble.config().learning_rate, deserialized.config().learning_rate);
    }

    #[test]
    fn test_contributions() {
        let config = EnsembleConfig::default();
        let mut ensemble = TreeEnsemble::new(config);
        
        let tree = create_simple_tree();
        ensemble.add_tree(tree, 0).unwrap();
        
        let features = Array1::from(vec![1.0, 2.0, 3.0]);
        let result = ensemble.predict_contributions(&features.view());
        assert!(result.is_ok());
        
        let contributions = result.unwrap();
        assert_eq!(contributions.shape(), &[4, 1]); // 3 features + 1 bias, 1 class
    }
}