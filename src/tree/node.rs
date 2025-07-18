//! Tree node implementation for the Pure Rust LightGBM framework.
//!
//! This module provides the fundamental tree node structure that can represent
//! both internal nodes (with splits) and leaf nodes (with prediction values).

use crate::core::types::{BinIndex, DataSize, FeatureIndex, NodeIndex, Score};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Tree node representation supporting both internal and leaf nodes.
///
/// Internal nodes contain split information (feature index, threshold) and
/// child node references. Leaf nodes contain prediction values and statistics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TreeNode {
    /// Left child node index (for internal nodes only)
    left_child: Option<NodeIndex>,
    /// Right child node index (for internal nodes only)
    right_child: Option<NodeIndex>,
    /// Parent node index (None for root node)
    parent: Option<NodeIndex>,
    /// Split feature index (for internal nodes only)
    split_feature: Option<FeatureIndex>,
    /// Split threshold value (for internal nodes only)
    split_threshold: Option<f64>,
    /// Bin threshold for the split (for internal nodes only)
    split_bin: Option<BinIndex>,
    /// Prediction value (for leaf nodes only)
    leaf_output: Option<Score>,
    /// Sum of gradients in this node
    sum_gradients: f64,
    /// Sum of hessians in this node
    sum_hessians: f64,
    /// Number of data points in this node
    data_count: DataSize,
    /// Split gain (improvement in loss function)
    split_gain: f64,
    /// Node depth in the tree
    depth: usize,
    /// Whether this node is a leaf
    is_leaf: bool,
    /// Default direction for missing values (true = left, false = right)
    default_left: bool,
}

impl TreeNode {
    /// Creates a new leaf node with the given statistics.
    pub fn new_leaf(
        sum_gradients: f64,
        sum_hessians: f64,
        data_count: DataSize,
        depth: usize,
        parent: Option<NodeIndex>,
    ) -> Self {
        TreeNode {
            left_child: None,
            right_child: None,
            parent,
            split_feature: None,
            split_threshold: None,
            split_bin: None,
            leaf_output: None,
            sum_gradients,
            sum_hessians,
            data_count,
            split_gain: 0.0,
            depth,
            is_leaf: true,
            default_left: false,
        }
    }

    /// Creates a new internal node with split information.
    pub fn new_internal(
        left_child: NodeIndex,
        right_child: NodeIndex,
        parent: Option<NodeIndex>,
        split_feature: FeatureIndex,
        split_threshold: f64,
        split_bin: BinIndex,
        split_gain: f64,
        sum_gradients: f64,
        sum_hessians: f64,
        data_count: DataSize,
        depth: usize,
        default_left: bool,
    ) -> Self {
        TreeNode {
            left_child: Some(left_child),
            right_child: Some(right_child),
            parent,
            split_feature: Some(split_feature),
            split_threshold: Some(split_threshold),
            split_bin: Some(split_bin),
            leaf_output: None,
            sum_gradients,
            sum_hessians,
            data_count,
            split_gain,
            depth,
            is_leaf: false,
            default_left,
        }
    }

    /// Returns true if this node is a leaf node.
    pub fn is_leaf(&self) -> bool {
        self.is_leaf
    }

    /// Returns the left child node index (for internal nodes).
    pub fn left_child(&self) -> Option<NodeIndex> {
        self.left_child
    }

    /// Returns the right child node index (for internal nodes).
    pub fn right_child(&self) -> Option<NodeIndex> {
        self.right_child
    }

    /// Returns the parent node index.
    pub fn parent(&self) -> Option<NodeIndex> {
        self.parent
    }

    /// Returns the split feature index (for internal nodes).
    pub fn split_feature(&self) -> Option<FeatureIndex> {
        self.split_feature
    }

    /// Returns the split threshold value (for internal nodes).
    pub fn split_threshold(&self) -> Option<f64> {
        self.split_threshold
    }

    /// Returns the split bin index (for internal nodes).
    pub fn split_bin(&self) -> Option<BinIndex> {
        self.split_bin
    }

    /// Returns the leaf output value (for leaf nodes).
    pub fn leaf_output(&self) -> Option<Score> {
        self.leaf_output
    }

    /// Sets the leaf output value.
    pub fn set_leaf_output(&mut self, output: Score) {
        self.leaf_output = Some(output);
    }

    /// Returns the sum of gradients in this node.
    pub fn sum_gradients(&self) -> f64 {
        self.sum_gradients
    }

    /// Returns the sum of hessians in this node.
    pub fn sum_hessians(&self) -> f64 {
        self.sum_hessians
    }

    /// Returns the number of data points in this node.
    pub fn data_count(&self) -> DataSize {
        self.data_count
    }

    /// Returns the split gain value.
    pub fn split_gain(&self) -> f64 {
        self.split_gain
    }

    /// Returns the node depth in the tree.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Returns the default direction for missing values.
    pub fn default_left(&self) -> bool {
        self.default_left
    }

    /// Converts this node from leaf to internal node with the given split.
    pub fn set_split(
        &mut self,
        left_child: NodeIndex,
        right_child: NodeIndex,
        split_feature: FeatureIndex,
        split_threshold: f64,
        split_bin: BinIndex,
        split_gain: f64,
        default_left: bool,
    ) {
        self.left_child = Some(left_child);
        self.right_child = Some(right_child);
        self.split_feature = Some(split_feature);
        self.split_threshold = Some(split_threshold);
        self.split_bin = Some(split_bin);
        self.split_gain = split_gain;
        self.default_left = default_left;
        self.is_leaf = false;
        self.leaf_output = None; // Clear leaf output
    }

    /// Updates the node statistics.
    pub fn update_statistics(
        &mut self,
        sum_gradients: f64,
        sum_hessians: f64,
        data_count: DataSize,
    ) {
        self.sum_gradients = sum_gradients;
        self.sum_hessians = sum_hessians;
        self.data_count = data_count;
    }

    /// Calculates the optimal leaf output using the given regularization parameters.
    pub fn calculate_leaf_output(&self, lambda_l1: f64, lambda_l2: f64) -> Score {
        if self.sum_hessians <= 0.0 {
            return 0.0;
        }

        let numerator = if lambda_l1 > 0.0 {
            if self.sum_gradients > lambda_l1 {
                self.sum_gradients - lambda_l1
            } else if self.sum_gradients < -lambda_l1 {
                self.sum_gradients + lambda_l1
            } else {
                0.0
            }
        } else {
            self.sum_gradients
        };

        let denominator = self.sum_hessians + lambda_l2;
        (-numerator / denominator) as Score
    }

    /// Calculates the split gain for this node's split.
    pub fn calculate_split_gain(
        &self,
        left_sum_gradients: f64,
        left_sum_hessians: f64,
        right_sum_gradients: f64,
        right_sum_hessians: f64,
        lambda_l1: f64,
        lambda_l2: f64,
    ) -> f64 {
        let parent_gain =
            self.calculate_gain(self.sum_gradients, self.sum_hessians, lambda_l1, lambda_l2);
        let left_gain =
            self.calculate_gain(left_sum_gradients, left_sum_hessians, lambda_l1, lambda_l2);
        let right_gain = self.calculate_gain(
            right_sum_gradients,
            right_sum_hessians,
            lambda_l1,
            lambda_l2,
        );

        left_gain + right_gain - parent_gain
    }

    /// Calculates the gain for given gradient and hessian sums.
    fn calculate_gain(
        &self,
        sum_gradients: f64,
        sum_hessians: f64,
        lambda_l1: f64,
        lambda_l2: f64,
    ) -> f64 {
        if sum_hessians <= 0.0 {
            return 0.0;
        }

        let numerator = if lambda_l1 > 0.0 {
            if sum_gradients > lambda_l1 {
                (sum_gradients - lambda_l1).powi(2)
            } else if sum_gradients < -lambda_l1 {
                (sum_gradients + lambda_l1).powi(2)
            } else {
                0.0
            }
        } else {
            sum_gradients.powi(2)
        };

        numerator / (2.0 * (sum_hessians + lambda_l2))
    }

    /// Returns true if this node can be split further.
    pub fn can_split(&self, min_data_in_leaf: DataSize, min_sum_hessian_in_leaf: f64) -> bool {
        self.is_leaf
            && self.data_count >= min_data_in_leaf * 2
            && self.sum_hessians >= min_sum_hessian_in_leaf * 2.0
    }
}

impl fmt::Display for TreeNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_leaf {
            write!(
                f,
                "Leaf(output={:.4}, data_count={}, sum_gradients={:.4}, sum_hessians={:.4})",
                self.leaf_output.unwrap_or(0.0),
                self.data_count,
                self.sum_gradients,
                self.sum_hessians
            )
        } else {
            write!(
                f,
                "Internal(feature={}, threshold={:.4}, gain={:.4}, data_count={})",
                self.split_feature.unwrap_or(0),
                self.split_threshold.unwrap_or(0.0),
                self.split_gain,
                self.data_count
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_leaf_node() {
        let node = TreeNode::new_leaf(10.0, 5.0, 100, 2, Some(0));

        assert!(node.is_leaf());
        assert_eq!(node.sum_gradients(), 10.0);
        assert_eq!(node.sum_hessians(), 5.0);
        assert_eq!(node.data_count(), 100);
        assert_eq!(node.depth(), 2);
        assert_eq!(node.parent(), Some(0));
        assert!(node.left_child().is_none());
        assert!(node.right_child().is_none());
    }

    #[test]
    fn test_new_internal_node() {
        let node = TreeNode::new_internal(1, 2, Some(0), 5, 2.5, 10, 1.5, 20.0, 10.0, 200, 1, true);

        assert!(!node.is_leaf());
        assert_eq!(node.left_child(), Some(1));
        assert_eq!(node.right_child(), Some(2));
        assert_eq!(node.split_feature(), Some(5));
        assert_eq!(node.split_threshold(), Some(2.5));
        assert_eq!(node.split_bin(), Some(10));
        assert_eq!(node.split_gain(), 1.5);
        assert!(node.default_left());
    }

    #[test]
    fn test_calculate_leaf_output() {
        let node = TreeNode::new_leaf(-10.0, 5.0, 100, 0, None);
        let output = node.calculate_leaf_output(0.0, 0.1);
        assert!((output - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_leaf_output_with_l1() {
        let node = TreeNode::new_leaf(-10.0, 5.0, 100, 0, None);
        let output = node.calculate_leaf_output(1.0, 0.1);
        assert!((output - 1.764706).abs() < 1e-6);
    }

    #[test]
    fn test_set_split() {
        let mut node = TreeNode::new_leaf(10.0, 5.0, 100, 1, Some(0));
        assert!(node.is_leaf());

        node.set_split(1, 2, 3, 2.5, 10, 1.2, false);

        assert!(!node.is_leaf());
        assert_eq!(node.left_child(), Some(1));
        assert_eq!(node.right_child(), Some(2));
        assert_eq!(node.split_feature(), Some(3));
        assert_eq!(node.split_threshold(), Some(2.5));
        assert!(node.leaf_output().is_none());
    }

    #[test]
    fn test_can_split() {
        let node = TreeNode::new_leaf(10.0, 5.0, 100, 1, None);
        assert!(node.can_split(20, 1.0));
        assert!(!node.can_split(60, 1.0));
        assert!(!node.can_split(20, 3.0));
    }

    #[test]
    fn test_calculate_split_gain() {
        let node = TreeNode::new_leaf(20.0, 10.0, 100, 1, None);
        let gain = node.calculate_split_gain(15.0, 6.0, 5.0, 4.0, 0.0, 0.1);
        assert!(gain > 0.0);
    }
}
