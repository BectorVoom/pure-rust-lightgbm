//! Leaf index prediction module for Pure Rust LightGBM.
//!
//! This module provides functionality to predict which leaf nodes samples
//! end up in, which is useful for advanced analysis and model interpretation.

use crate::core::error::{LightGBMError, Result};

use ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};

/// Configuration for leaf index prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeafIndexConfig {
    /// Number of iterations to use for leaf prediction (None = use all)
    pub num_iterations: Option<usize>,
    /// Whether to return indices for all trees or just the last iteration
    pub all_trees: bool,
    /// Whether to include tree metadata in results
    pub include_metadata: bool,
}

impl LeafIndexConfig {
    /// Create new leaf index configuration
    pub fn new() -> Self {
        Self {
            num_iterations: None,
            all_trees: true,
            include_metadata: false,
        }
    }

    /// Set number of iterations
    pub fn with_iterations(mut self, num_iterations: Option<usize>) -> Self {
        self.num_iterations = num_iterations;
        self
    }

    /// Set whether to include all trees
    pub fn with_all_trees(mut self, all_trees: bool) -> Self {
        self.all_trees = all_trees;
        self
    }

    /// Set whether to include metadata
    pub fn with_metadata(mut self, include_metadata: bool) -> Self {
        self.include_metadata = include_metadata;
        self
    }
}

impl Default for LeafIndexConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Leaf index prediction result
#[derive(Debug, Clone)]
pub struct LeafIndexResult {
    /// Leaf indices for each sample and tree [samples x trees]
    pub leaf_indices: Array2<i32>,
    /// Tree metadata (if requested)
    pub tree_metadata: Option<Vec<TreeMetadata>>,
    /// Number of samples
    pub num_samples: usize,
    /// Number of trees used
    pub num_trees: usize,
}

/// Metadata for a single tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeMetadata {
    /// Tree index in the ensemble
    pub tree_index: usize,
    /// Boosting iteration this tree belongs to
    pub iteration: usize,
    /// Number of leaves in this tree
    pub num_leaves: usize,
    /// Maximum depth of this tree
    pub max_depth: usize,
    /// Tree class (for multiclass classification)
    pub tree_class: Option<usize>,
}

/// Leaf index predictor
#[derive(Debug)]
pub struct LeafIndexPredictor {
    model: Option<crate::boosting::GBDT>,
    config: LeafIndexConfig,
}

impl LeafIndexPredictor {
    /// Create new leaf index predictor
    pub fn new(model: crate::boosting::GBDT, config: LeafIndexConfig) -> Self {
        Self {
            model: Some(model),
            config,
        }
    }

    /// Create predictor without model (for testing)
    pub fn new_without_model(config: LeafIndexConfig) -> Self {
        Self {
            model: None,
            config,
        }
    }

    /// Predict leaf indices for given features
    pub fn predict_leaf_indices(&self, features: &ArrayView2<'_, f32>) -> Result<LeafIndexResult> {
        match &self.model {
            Some(model) => {
                // This would need to be implemented in the GBDT model
                // For now, return a placeholder implementation
                let num_samples = features.nrows();
                let num_trees = self.estimate_num_trees(model)?;

                // Placeholder: return dummy leaf indices
                let leaf_indices = Array2::zeros((num_samples, num_trees));

                let tree_metadata = if self.config.include_metadata {
                    Some(self.generate_tree_metadata(model, num_trees)?)
                } else {
                    None
                };

                Ok(LeafIndexResult {
                    leaf_indices,
                    tree_metadata,
                    num_samples,
                    num_trees,
                })
            }
            None => Err(LightGBMError::prediction(
                "No model available for leaf index prediction",
            )),
        }
    }

    /// Predict leaf indices for a single sample
    pub fn predict_leaf_indices_single(&self, _features: &[f32]) -> Result<Array1<i32>> {
        match &self.model {
            Some(model) => {
                let num_trees = self.estimate_num_trees(model)?;

                // TODO: Implement actual leaf index prediction by traversing trees with features
                // This should use the features parameter to navigate through each tree
                // Placeholder: return dummy leaf indices
                let leaf_indices = Array1::zeros(num_trees);

                Ok(leaf_indices)
            }
            None => Err(LightGBMError::prediction(
                "No model available for leaf index prediction",
            )),
        }
    }

    /// Get leaf value for a specific tree and leaf index
    pub fn get_leaf_value(&self, _tree_index: usize, _leaf_index: i32) -> Result<f64> {
        match &self.model {
            Some(_model) => {
                // TODO: Implement actual leaf value retrieval from model trees
                // This should use tree_index and leaf_index to get the actual leaf value
                // This would need to be implemented by accessing the tree structure
                // For now, return a placeholder
                Ok(0.0)
            }
            None => Err(LightGBMError::prediction(
                "No model available for leaf value lookup",
            )),
        }
    }

    /// Analyze leaf distribution statistics
    pub fn analyze_leaf_distribution(
        &self,
        leaf_result: &LeafIndexResult,
    ) -> LeafDistributionStats {
        let mut leaf_counts: std::collections::HashMap<i32, usize> =
            std::collections::HashMap::new();

        for &leaf_idx in leaf_result.leaf_indices.iter() {
            *leaf_counts.entry(leaf_idx).or_insert(0) += 1;
        }

        let total_predictions = leaf_result.leaf_indices.len();
        let unique_leaves = leaf_counts.len();
        let most_common_leaf = leaf_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&leaf, &count)| (leaf, count))
            .unwrap_or((0, 0));

        let leaf_entropy = self.calculate_leaf_entropy(&leaf_counts, total_predictions);

        LeafDistributionStats {
            total_predictions,
            unique_leaves,
            most_common_leaf,
            leaf_entropy,
            leaf_counts,
        }
    }

    /// Calculate entropy of leaf distribution
    fn calculate_leaf_entropy(
        &self,
        leaf_counts: &std::collections::HashMap<i32, usize>,
        total: usize,
    ) -> f64 {
        if total == 0 {
            return 0.0;
        }

        leaf_counts
            .values()
            .map(|&count| {
                let prob = count as f64 / total as f64;
                if prob > 0.0 {
                    -prob * prob.log2()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Estimate number of trees in the model
    fn estimate_num_trees(&self, _model: &crate::boosting::GBDT) -> Result<usize> {
        // TODO: Implement actual tree count estimation from model
        // This would need access to the model's internal structure
        // For now, use a placeholder calculation
        let num_iterations = self.config.num_iterations.unwrap_or(100); // Default
        let num_tree_per_iteration = 1; // Would need to get from model config
        Ok(num_iterations * num_tree_per_iteration)
    }

    /// Generate tree metadata
    fn generate_tree_metadata(
        &self,
        _model: &crate::boosting::GBDT,
        num_trees: usize,
    ) -> Result<Vec<TreeMetadata>> {
        // Placeholder implementation
        let mut metadata = Vec::new();

        for tree_idx in 0..num_trees {
            metadata.push(TreeMetadata {
                tree_index: tree_idx,
                iteration: tree_idx, // Simplified
                num_leaves: 31,      // Default
                max_depth: 6,        // Default
                tree_class: None,
            });
        }

        Ok(metadata)
    }
}

/// Statistics about leaf distribution
#[derive(Debug, Clone)]
pub struct LeafDistributionStats {
    /// Total number of predictions
    pub total_predictions: usize,
    /// Number of unique leaves used
    pub unique_leaves: usize,
    /// Most commonly used leaf (leaf_index, count)
    pub most_common_leaf: (i32, usize),
    /// Entropy of leaf distribution
    pub leaf_entropy: f64,
    /// Count of samples in each leaf
    pub leaf_counts: std::collections::HashMap<i32, usize>,
}

impl LeafDistributionStats {
    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "Leaf Distribution Statistics:\n\
             - Total predictions: {}\n\
             - Unique leaves used: {}\n\
             - Most common leaf: {} (used {} times)\n\
             - Leaf entropy: {:.4}\n\
             - Average samples per leaf: {:.2}",
            self.total_predictions,
            self.unique_leaves,
            self.most_common_leaf.0,
            self.most_common_leaf.1,
            self.leaf_entropy,
            self.total_predictions as f64 / self.unique_leaves as f64
        )
    }
}
