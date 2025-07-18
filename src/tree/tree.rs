//! Decision tree implementation for the Pure Rust LightGBM framework.
//!
//! This module provides the core decision tree structure with efficient
//! prediction, serialization, and tree manipulation capabilities.

use crate::core::types::{BinIndex, DataSize, FeatureIndex, NodeIndex, Score};
use crate::tree::node::TreeNode;
use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;

/// Decision tree structure representing a single tree in the ensemble.
///
/// The tree stores nodes in a contiguous vector for cache-efficient access
/// and provides methods for prediction, tree construction, and analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tree {
    /// Vector of tree nodes (index 0 is always the root)
    nodes: Vec<TreeNode>,
    /// Maximum number of leaves allowed in the tree
    max_leaves: usize,
    /// Current number of leaf nodes
    num_leaves: usize,
    /// Tree shrinkage factor (learning rate)
    shrinkage: f64,
    /// Maximum tree depth
    max_depth: usize,
}

impl Tree {
    /// Creates a new tree with a single root node.
    pub fn new(max_leaves: usize) -> Self {
        let root = TreeNode::new_leaf(0.0, 0.0, 0, 0, None);
        
        Tree {
            nodes: vec![root],
            max_leaves,
            num_leaves: 1,
            shrinkage: 1.0,
            max_depth: 0,
        }
    }

    /// Creates a new tree with specified capacity and shrinkage.
    pub fn with_capacity(max_leaves: usize, shrinkage: f64) -> Self {
        let mut tree = Self::new(max_leaves);
        tree.shrinkage = shrinkage;
        tree
    }

    /// Returns the number of nodes in the tree.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of leaf nodes in the tree.
    pub fn num_leaves(&self) -> usize {
        self.num_leaves
    }

    /// Returns the maximum number of leaves allowed.
    pub fn max_leaves(&self) -> usize {
        self.max_leaves
    }

    /// Returns the tree depth (maximum depth of any node).
    pub fn depth(&self) -> usize {
        self.max_depth
    }

    /// Returns the shrinkage factor.
    pub fn shrinkage(&self) -> f64 {
        self.shrinkage
    }

    /// Returns a reference to the node at the given index.
    pub fn node(&self, index: NodeIndex) -> Option<&TreeNode> {
        self.nodes.get(index)
    }

    /// Returns a mutable reference to the node at the given index.
    pub fn node_mut(&mut self, index: NodeIndex) -> Option<&mut TreeNode> {
        self.nodes.get_mut(index)
    }

    /// Returns the root node of the tree.
    pub fn root(&self) -> &TreeNode {
        &self.nodes[0]
    }

    /// Predicts the output for a single data point.
    pub fn predict(&self, features: &ArrayView1<f32>) -> anyhow::Result<Score> {
        let mut node_index = 0;

        loop {
            let node = &self.nodes[node_index];

            if node.is_leaf() {
                return Ok(node.leaf_output().unwrap_or(0.0) * self.shrinkage as Score);
            }

            let feature_idx = node.split_feature().unwrap();
            let threshold = node.split_threshold().unwrap();
            let feature_value = features[feature_idx] as f64;

            // Handle missing values
            let go_left = if feature_value.is_nan() {
                node.default_left()
            } else {
                feature_value <= threshold
            };

            node_index = if go_left {
                node.left_child().unwrap()
            } else {
                node.right_child().unwrap()
            };
        }
    }

    /// Predicts the output for multiple data points.
    pub fn predict_batch(&self, features: &ndarray::Array2<f32>) -> anyhow::Result<Array1<Score>> {
        let num_data = features.nrows();
        let mut predictions = Array1::zeros(num_data);

        for (i, row) in features.axis_iter(ndarray::Axis(0)).enumerate() {
            predictions[i] = self.predict(&row)?;
        }

        Ok(predictions)
    }

    /// Predicts the leaf index for a single data point.
    pub fn predict_leaf_index(&self, features: &ArrayView1<f32>) -> anyhow::Result<NodeIndex> {
        let mut node_index = 0;

        loop {
            let node = &self.nodes[node_index];

            if node.is_leaf() {
                return Ok(node_index);
            }

            let feature_idx = node.split_feature().unwrap();
            let threshold = node.split_threshold().unwrap();
            let feature_value = features[feature_idx] as f64;

            let go_left = if feature_value.is_nan() {
                node.default_left()
            } else {
                feature_value <= threshold
            };

            node_index = if go_left {
                node.left_child().unwrap()
            } else {
                node.right_child().unwrap()
            };
        }
    }

    /// Sets the leaf output for the node at the given index.
    pub fn set_leaf_output(&mut self, node_index: NodeIndex, output: Score) -> anyhow::Result<()> {
        if node_index >= self.nodes.len() {
            return Err(anyhow::anyhow!("Node index {} out of bounds", node_index));
        }

        if let Some(node) = self.nodes.get_mut(node_index) {
            if !node.is_leaf() {
                return Err(anyhow::anyhow!("Cannot set output for non-leaf node"));
            }
            node.set_leaf_output(output);
        }

        Ok(())
    }

    /// Splits a leaf node into an internal node with two children.
    pub fn split_node(
        &mut self,
        node_index: NodeIndex,
        split_feature: FeatureIndex,
        split_threshold: f64,
        split_bin: BinIndex,
        split_gain: f64,
        left_sum_gradients: f64,
        left_sum_hessians: f64,
        left_data_count: DataSize,
        right_sum_gradients: f64,
        right_sum_hessians: f64,
        right_data_count: DataSize,
        default_left: bool,
    ) -> anyhow::Result<(NodeIndex, NodeIndex)> {
        if node_index >= self.nodes.len() {
            return Err(anyhow::anyhow!("Node index {} out of bounds", node_index));
        }

        if !self.nodes[node_index].is_leaf() {
            return Err(anyhow::anyhow!("Cannot split non-leaf node"));
        }

        if self.num_leaves >= self.max_leaves {
            return Err(anyhow::anyhow!("Maximum number of leaves reached"));
        }

        let parent_depth = self.nodes[node_index].depth();
        let child_depth = parent_depth + 1;
        self.max_depth = self.max_depth.max(child_depth);

        // Create left and right child nodes
        let left_child_index = self.nodes.len();
        let right_child_index = self.nodes.len() + 1;

        let left_child = TreeNode::new_leaf(
            left_sum_gradients,
            left_sum_hessians,
            left_data_count,
            child_depth,
            Some(node_index),
        );

        let right_child = TreeNode::new_leaf(
            right_sum_gradients,
            right_sum_hessians,
            right_data_count,
            child_depth,
            Some(node_index),
        );

        self.nodes.push(left_child);
        self.nodes.push(right_child);

        // Convert the parent node to an internal node
        self.nodes[node_index].set_split(
            left_child_index,
            right_child_index,
            split_feature,
            split_threshold,
            split_bin,
            split_gain,
            default_left,
        );

        // Update leaf count (added 2 leaves, removed 1)
        self.num_leaves += 1;

        Ok((left_child_index, right_child_index))
    }

    /// Returns all leaf node indices.
    pub fn leaf_indices(&self) -> Vec<NodeIndex> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, node)| if node.is_leaf() { Some(i) } else { None })
            .collect()
    }

    /// Returns the best leaf to split based on split gain.
    pub fn best_leaf_to_split(&self) -> Option<NodeIndex> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.is_leaf())
            .max_by(|(_, a), (_, b)| a.split_gain().partial_cmp(&b.split_gain()).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, _)| index)
    }

    /// Calculates feature importance based on split gains.
    pub fn feature_importance(&self, num_features: usize) -> Array1<f64> {
        let mut importance = Array1::zeros(num_features);

        for node in &self.nodes {
            if !node.is_leaf() {
                if let Some(feature_idx) = node.split_feature() {
                    if feature_idx < num_features {
                        importance[feature_idx] += node.split_gain();
                    }
                }
            }
        }

        importance
    }

    /// Returns a textual representation of the tree structure.
    pub fn to_string_representation(&self) -> String {
        if self.nodes.is_empty() {
            return "Empty tree".to_string();
        }

        let mut result = String::new();
        self.tree_to_string_recursive(0, "", true, &mut result);
        result
    }

    fn tree_to_string_recursive(
        &self,
        node_index: NodeIndex,
        prefix: &str,
        is_last: bool,
        result: &mut String,
    ) {
        if node_index >= self.nodes.len() {
            return;
        }

        let node = &self.nodes[node_index];
        let current_prefix = if is_last { "└── " } else { "├── " };
        result.push_str(&format!("{}{}{}\n", prefix, current_prefix, node));

        if !node.is_leaf() {
            let new_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });

            if let Some(left_child) = node.left_child() {
                self.tree_to_string_recursive(left_child, &new_prefix, false, result);
            }

            if let Some(right_child) = node.right_child() {
                self.tree_to_string_recursive(right_child, &new_prefix, true, result);
            }
        }
    }

    /// Validates the tree structure consistency.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.nodes.is_empty() {
            return Err(anyhow::anyhow!("Tree has no nodes"));
        }

        // Check root node
        if self.nodes[0].parent().is_some() {
            return Err(anyhow::anyhow!("Root node should not have a parent"));
        }

        let mut leaf_count = 0;
        for (i, node) in self.nodes.iter().enumerate() {
            if node.is_leaf() {
                leaf_count += 1;
                if node.left_child().is_some() || node.right_child().is_some() {
                    return Err(anyhow::anyhow!("Leaf node {} has children", i));
                }
            } else {
                let left_child = node.left_child();
                let right_child = node.right_child();

                if left_child.is_none() || right_child.is_none() {
                    return Err(anyhow::anyhow!("Internal node {} missing children", i));
                }

                let left_idx = left_child.unwrap();
                let right_idx = right_child.unwrap();

                if left_idx >= self.nodes.len() || right_idx >= self.nodes.len() {
                    return Err(anyhow::anyhow!("Node {} has invalid child indices", i));
                }

                // Check that children point back to parent
                if self.nodes[left_idx].parent() != Some(i) {
                    return Err(anyhow::anyhow!("Left child {} parent mismatch", left_idx));
                }

                if self.nodes[right_idx].parent() != Some(i) {
                    return Err(anyhow::anyhow!("Right child {} parent mismatch", right_idx));
                }
            }
        }

        if leaf_count != self.num_leaves {
            return Err(anyhow::anyhow!(
                "Leaf count mismatch: expected {}, found {}",
                self.num_leaves,
                leaf_count
            ));
        }

        Ok(())
    }

    /// Converts the tree to a JSON representation.
    pub fn to_json(&self) -> anyhow::Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| anyhow::anyhow!("JSON serialization failed: {}", e))
    }

    /// Creates a tree from a JSON representation.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        serde_json::from_str(json).map_err(|e| anyhow::anyhow!("JSON deserialization failed: {}", e))
    }
}

impl fmt::Display for Tree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tree(nodes={}, leaves={}, depth={}, shrinkage={})",
            self.num_nodes(),
            self.num_leaves(),
            self.depth(),
            self.shrinkage()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_new_tree() {
        let tree = Tree::new(31);
        assert_eq!(tree.num_nodes(), 1);
        assert_eq!(tree.num_leaves(), 1);
        assert_eq!(tree.max_leaves(), 31);
        assert_eq!(tree.depth(), 0);
        assert!(tree.root().is_leaf());
    }

    #[test]
    fn test_split_node() {
        let mut tree = Tree::new(31);
        
        // Set up root node
        tree.node_mut(0).unwrap().update_statistics(20.0, 10.0, 100);
        
        let result = tree.split_node(
            0, 5, 2.5, 10, 1.5, 
            15.0, 6.0, 60, 
            5.0, 4.0, 40, 
            true
        );
        
        assert!(result.is_ok());
        let (left_idx, right_idx) = result.unwrap();
        
        assert_eq!(tree.num_nodes(), 3);
        assert_eq!(tree.num_leaves(), 2);
        assert_eq!(left_idx, 1);
        assert_eq!(right_idx, 2);
        assert!(!tree.root().is_leaf());
        assert_eq!(tree.root().split_feature(), Some(5));
    }

    #[test]
    fn test_predict_simple() {
        let mut tree = Tree::new(31);
        
        // Create a simple tree: if feature[0] <= 2.5 then 1.0 else -1.0
        tree.node_mut(0).unwrap().update_statistics(0.0, 10.0, 100);
        let _ = tree.split_node(
            0, 0, 2.5, 5, 1.0,
            10.0, 5.0, 60,
            -10.0, 5.0, 40,
            true
        );
        
        tree.set_leaf_output(1, 1.0).unwrap();
        tree.set_leaf_output(2, -1.0).unwrap();
        
        let features1 = Array1::from(vec![2.0, 0.0, 0.0]);
        let features2 = Array1::from(vec![3.0, 0.0, 0.0]);
        
        let pred1 = tree.predict(&features1.view()).unwrap();
        let pred2 = tree.predict(&features2.view()).unwrap();
        
        assert_eq!(pred1, 1.0);
        assert_eq!(pred2, -1.0);
    }

    #[test]
    fn test_predict_leaf_index() {
        let mut tree = Tree::new(31);
        
        tree.node_mut(0).unwrap().update_statistics(0.0, 10.0, 100);
        let _ = tree.split_node(
            0, 0, 2.5, 5, 1.0,
            10.0, 5.0, 60,
            -10.0, 5.0, 40,
            true
        );
        
        let features1 = Array1::from(vec![2.0]);
        let features2 = Array1::from(vec![3.0]);
        
        let leaf1 = tree.predict_leaf_index(&features1.view()).unwrap();
        let leaf2 = tree.predict_leaf_index(&features2.view()).unwrap();
        
        assert_eq!(leaf1, 1);
        assert_eq!(leaf2, 2);
    }

    #[test]
    fn test_feature_importance() {
        let mut tree = Tree::new(31);
        
        tree.node_mut(0).unwrap().update_statistics(0.0, 10.0, 100);
        let _ = tree.split_node(
            0, 2, 2.5, 5, 1.5,
            10.0, 5.0, 60,
            -10.0, 5.0, 40,
            true
        );
        
        let importance = tree.feature_importance(5);
        assert_eq!(importance[2], 1.5);
        assert_eq!(importance[0], 0.0);
        assert_eq!(importance[1], 0.0);
    }

    #[test]
    fn test_tree_validation() {
        let tree = Tree::new(31);
        assert!(tree.validate().is_ok());
        
        let mut tree = Tree::new(31);
        tree.node_mut(0).unwrap().update_statistics(0.0, 10.0, 100);
        let _ = tree.split_node(
            0, 0, 2.5, 5, 1.0,
            10.0, 5.0, 60,
            -10.0, 5.0, 40,
            true
        );
        
        assert!(tree.validate().is_ok());
    }

    #[test]
    fn test_serialization() {
        let tree = Tree::new(31);
        let json = tree.to_json().unwrap();
        let deserialized = Tree::from_json(&json).unwrap();
        
        assert_eq!(tree.num_nodes(), deserialized.num_nodes());
        assert_eq!(tree.num_leaves(), deserialized.num_leaves());
        assert_eq!(tree.max_leaves(), deserialized.max_leaves());
    }
}