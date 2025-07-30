//! Tree data structure and operations for LightGBM.
//!
//! This module provides the Tree implementation that is semantically equivalent
//! to the C++ LightGBM Tree class. It handles tree construction, prediction,
//! serialization, and SHAP value computation.

use crate::core::types::{DataSize, MissingType};
use crate::dataset::Dataset;
use anyhow::{bail, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Decision type bit masks for tree node decisions
const CATEGORICAL_MASK: u8 = 1;
const DEFAULT_LEFT_MASK: u8 = 2;

/// Path element for SHAP value computation
#[derive(Debug, Clone, Copy)]
pub struct PathElement {
    /// Feature index for this path element
    pub feature_index: i32,
    /// Zero fraction for this path element
    pub zero_fraction: f64,
    /// One fraction for this path element
    pub one_fraction: f64,
    /// Path weight for this element
    pub pweight: f64,
}

/// Tree structure for gradient boosting decision trees.
///
/// This struct represents a single decision tree in the ensemble and is
/// semantically equivalent to the C++ Tree class in LightGBM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tree {
    /// Maximum number of leaves allowed in this tree
    max_leaves: i32,
    /// Current number of leaves in the tree
    num_leaves: i32,
    /// Whether to track branch features
    track_branch_features: bool,
    /// Whether this is a linear tree
    is_linear: bool,
    /// Shrinkage factor applied to leaf values
    shrinkage: f32,
    /// Number of categorical features
    num_cat: i32,
    /// Maximum depth of the tree (-1 if not computed)
    max_depth: i32,

    // Tree structure arrays
    /// Left child indices for internal nodes
    left_child: Vec<i32>,
    /// Right child indices for internal nodes
    right_child: Vec<i32>,
    /// Feature indices used for splits (inner representation)
    split_feature_inner: Vec<i32>,
    /// Feature indices used for splits (original representation)
    split_feature: Vec<i32>,
    /// Threshold values in bin representation
    threshold_in_bin: Vec<u32>,
    /// Threshold values in double representation
    threshold: Vec<f64>,
    /// Decision type information for each split
    decision_type: Vec<i8>,
    /// Split gain values
    split_gain: Vec<f32>,

    // Leaf information
    /// Parent node index for each leaf
    leaf_parent: Vec<i32>,
    /// Prediction values for each leaf
    leaf_value: Vec<f64>,
    /// Weight values for each leaf
    leaf_weight: Vec<f64>,
    /// Sample count for each leaf
    leaf_count: Vec<i32>,
    /// Depth of each leaf in the tree
    leaf_depth: Vec<i32>,

    // Internal node information
    /// Prediction values for internal nodes
    internal_value: Vec<f64>,
    /// Weight values for internal nodes
    internal_weight: Vec<f64>,
    /// Sample count for internal nodes
    internal_count: Vec<i32>,

    // Categorical feature support
    /// Boundaries for categorical thresholds
    cat_boundaries: Vec<i32>,
    /// Boundaries for categorical thresholds (inner representation)
    cat_boundaries_inner: Vec<i32>,
    /// Categorical threshold values
    cat_threshold: Vec<u32>,
    /// Categorical threshold values (inner representation)
    cat_threshold_inner: Vec<u32>,

    // Linear tree support
    /// Constant values for linear leaves
    leaf_const: Vec<f64>,
    /// Coefficients for linear leaves
    leaf_coeff: Vec<Vec<f64>>,
    /// Feature indices for linear leaves
    leaf_features: Vec<Vec<i32>>,
    /// Feature indices for linear leaves (inner representation)
    leaf_features_inner: Vec<Vec<i32>>,

    // Optional branch features tracking
    /// Branch features for each leaf (if tracked)
    branch_features: Option<Vec<Vec<i32>>>,
}

impl Tree {
    /// Create a new tree with the specified parameters.
    ///
    /// # Arguments
    /// * `max_leaves` - Maximum number of leaves allowed
    /// * `track_branch_features` - Whether to track branch features
    /// * `is_linear` - Whether this is a linear tree
    ///
    /// # Returns
    /// A new Tree instance
    pub fn new(max_leaves: i32, track_branch_features: bool, is_linear: bool) -> Self {
        let mut tree = Tree {
            max_leaves,
            num_leaves: 1,
            track_branch_features,
            is_linear,
            shrinkage: 1.0,
            num_cat: 0,
            max_depth: -1,

            // Initialize vectors with appropriate sizes
            left_child: vec![0; (max_leaves - 1) as usize],
            right_child: vec![0; (max_leaves - 1) as usize],
            split_feature_inner: vec![0; (max_leaves - 1) as usize],
            split_feature: vec![0; (max_leaves - 1) as usize],
            threshold_in_bin: vec![0; (max_leaves - 1) as usize],
            threshold: vec![0.0; (max_leaves - 1) as usize],
            decision_type: vec![0; (max_leaves - 1) as usize],
            split_gain: vec![0.0; (max_leaves - 1) as usize],

            leaf_parent: vec![0; max_leaves as usize],
            leaf_value: vec![0.0; max_leaves as usize],
            leaf_weight: vec![0.0; max_leaves as usize],
            leaf_count: vec![0; max_leaves as usize],
            leaf_depth: vec![0; max_leaves as usize],

            internal_value: vec![0.0; (max_leaves - 1) as usize],
            internal_weight: vec![0.0; (max_leaves - 1) as usize],
            internal_count: vec![0; (max_leaves - 1) as usize],

            cat_boundaries: vec![0],
            cat_boundaries_inner: vec![0],
            cat_threshold: Vec::new(),
            cat_threshold_inner: Vec::new(),

            leaf_const: Vec::new(),
            leaf_coeff: Vec::new(),
            leaf_features: Vec::new(),
            leaf_features_inner: Vec::new(),

            branch_features: None,
        };

        // Initialize branch features if tracking is enabled
        if track_branch_features {
            tree.branch_features = Some(vec![Vec::new(); max_leaves as usize]);
        }

        // Initialize root node
        tree.leaf_depth[0] = 0;
        tree.leaf_value[0] = 0.0;
        tree.leaf_weight[0] = 0.0;
        tree.leaf_parent[0] = -1;

        // Initialize linear tree components if needed
        if is_linear {
            tree.leaf_coeff = vec![Vec::new(); max_leaves as usize];
            tree.leaf_const = vec![0.0; max_leaves as usize];
            tree.leaf_features = vec![Vec::new(); max_leaves as usize];
            tree.leaf_features_inner = vec![Vec::new(); max_leaves as usize];
        }

        tree
    }

    /// Create a tree from string representation (simplified version).
    /// This is a basic implementation - the full string parsing would require
    /// more complex parsing logic similar to the C++ version.
    pub fn from_string(tree_str: &str) -> Result<Self> {
        // For now, just create a default tree
        // A full implementation would parse the string format used by LightGBM
        let mut tree = Self::new(31, false, false);

        // Parse basic parameters from string
        for line in tree_str.lines() {
            if let Some(eq_pos) = line.find('=') {
                let key = line[..eq_pos].trim();
                let value = line[eq_pos + 1..].trim();

                match key {
                    "num_leaves" => {
                        if let Ok(num) = value.parse::<i32>() {
                            tree.num_leaves = num;
                        }
                    }
                    "num_cat" => {
                        if let Ok(num) = value.parse::<i32>() {
                            tree.num_cat = num;
                        }
                    }
                    "shrinkage" => {
                        if let Ok(shrink) = value.parse::<f32>() {
                            tree.shrinkage = shrink;
                        }
                    }
                    _ => {} // Skip other fields for now
                }
            }
        }

        Ok(tree)
    }

    /// Split a leaf node for numerical features.
    ///
    /// # Arguments
    /// * `leaf` - Index of the leaf to split
    /// * `feature` - Feature index for the split
    /// * `real_feature` - Real feature index
    /// * `threshold_bin` - Threshold in bin representation
    /// * `threshold_double` - Threshold in double representation
    /// * `left_value` - Prediction value for left child
    /// * `right_value` - Prediction value for right child
    /// * `left_cnt` - Sample count for left child
    /// * `right_cnt` - Sample count for right child
    /// * `left_weight` - Weight for left child
    /// * `right_weight` - Weight for right child
    /// * `gain` - Split gain
    /// * `missing_type` - How to handle missing values
    /// * `default_left` - Whether missing values go left by default
    ///
    /// # Returns
    /// Index of the new internal node
    pub fn split(
        &mut self,
        leaf: i32,
        feature: i32,
        real_feature: i32,
        threshold_bin: u32,
        threshold_double: f64,
        left_value: f64,
        right_value: f64,
        left_cnt: i32,
        right_cnt: i32,
        left_weight: f64,
        right_weight: f64,
        gain: f32,
        missing_type: MissingType,
        default_left: bool,
    ) -> i32 {
        // Call the general split method first
        self.split_general(
            leaf,
            feature,
            real_feature,
            left_value,
            right_value,
            left_cnt,
            right_cnt,
            left_weight,
            right_weight,
            gain,
        );

        let new_node_idx = (self.num_leaves - 1) as usize;

        // Set decision type for numerical split
        self.decision_type[new_node_idx] = 0;
        self.set_decision_type(new_node_idx, false, CATEGORICAL_MASK);
        self.set_decision_type(new_node_idx, default_left, DEFAULT_LEFT_MASK);
        self.set_missing_type(new_node_idx, missing_type as i8);

        // Set threshold values
        self.threshold_in_bin[new_node_idx] = threshold_bin;
        self.threshold[new_node_idx] = threshold_double;

        self.num_leaves += 1;
        self.num_leaves - 1
    }

    /// Split a leaf node for categorical features.
    ///
    /// # Arguments
    /// * `leaf` - Index of the leaf to split
    /// * `feature` - Feature index for the split
    /// * `real_feature` - Real feature index
    /// * `threshold_bin` - Threshold bins for categorical values
    /// * `threshold` - Threshold values for categorical values
    /// * `left_value` - Prediction value for left child
    /// * `right_value` - Prediction value for right child
    /// * `left_cnt` - Sample count for left child
    /// * `right_cnt` - Sample count for right child
    /// * `left_weight` - Weight for left child
    /// * `right_weight` - Weight for right child
    /// * `gain` - Split gain
    /// * `missing_type` - How to handle missing values
    ///
    /// # Returns
    /// Index of the new internal node
    pub fn split_categorical(
        &mut self,
        leaf: i32,
        feature: i32,
        real_feature: i32,
        threshold_bin: &[u32],
        threshold: &[u32],
        left_value: f64,
        right_value: f64,
        left_cnt: DataSize,
        right_cnt: DataSize,
        left_weight: f64,
        right_weight: f64,
        gain: f32,
        missing_type: MissingType,
    ) -> i32 {
        // Call the general split method first
        self.split_general(
            leaf,
            feature,
            real_feature,
            left_value,
            right_value,
            left_cnt,
            right_cnt,
            left_weight,
            right_weight,
            gain,
        );

        let new_node_idx = (self.num_leaves - 1) as usize;

        // Set decision type for categorical split
        self.decision_type[new_node_idx] = 0;
        self.set_decision_type(new_node_idx, true, CATEGORICAL_MASK);
        self.set_missing_type(new_node_idx, missing_type as i8);

        // Set threshold values for categorical split
        self.threshold_in_bin[new_node_idx] = self.num_cat as u32;
        self.threshold[new_node_idx] = self.num_cat as f64;

        self.num_cat += 1;

        // Update categorical boundaries and thresholds
        self.cat_boundaries
            .push(*self.cat_boundaries.last().unwrap() + threshold.len() as i32);
        for &thresh in threshold {
            self.cat_threshold.push(thresh);
        }

        self.cat_boundaries_inner
            .push(*self.cat_boundaries_inner.last().unwrap() + threshold_bin.len() as i32);
        for &thresh_bin in threshold_bin {
            self.cat_threshold_inner.push(thresh_bin);
        }

        self.num_leaves += 1;
        self.num_leaves - 1
    }

    /// General split method called by both numerical and categorical splits.
    fn split_general(
        &mut self,
        leaf: i32,
        feature: i32,
        real_feature: i32,
        left_value: f64,
        right_value: f64,
        left_cnt: DataSize,
        right_cnt: DataSize,
        left_weight: f64,
        right_weight: f64,
        gain: f32,
    ) {
        let leaf_idx = leaf as usize;
        let new_node_idx = (self.num_leaves - 1) as usize;
        let left_idx = self.num_leaves as usize;
        let right_idx = (self.num_leaves + 1) as usize;

        // Set split information for the new internal node
        self.split_feature_inner[new_node_idx] = feature;
        self.split_feature[new_node_idx] = real_feature;
        self.split_gain[new_node_idx] = gain;

        // Set child indices
        self.left_child[new_node_idx] = self.num_leaves;
        self.right_child[new_node_idx] = self.num_leaves + 1;

        // Set leaf values
        self.leaf_value[left_idx] = left_value;
        self.leaf_value[right_idx] = right_value;

        // Set leaf weights
        self.leaf_weight[left_idx] = left_weight;
        self.leaf_weight[right_idx] = right_weight;

        // Set leaf counts
        self.leaf_count[left_idx] = left_cnt;
        self.leaf_count[right_idx] = right_cnt;

        // Set leaf parents
        self.leaf_parent[left_idx] = new_node_idx as i32;
        self.leaf_parent[right_idx] = new_node_idx as i32;

        // Set leaf depths
        self.leaf_depth[left_idx] = self.leaf_depth[leaf_idx] + 1;
        self.leaf_depth[right_idx] = self.leaf_depth[leaf_idx] + 1;

        // Set internal node information
        self.internal_value[new_node_idx] = self.leaf_value[leaf_idx];
        self.internal_weight[new_node_idx] = self.leaf_weight[leaf_idx];
        self.internal_count[new_node_idx] = self.leaf_count[leaf_idx];

        // Update branch features if tracking is enabled
        if let Some(ref mut branch_features) = self.branch_features {
            // Copy parent's branch features to children
            branch_features[left_idx] = branch_features[leaf_idx].clone();
            branch_features[right_idx] = branch_features[leaf_idx].clone();

            // Add current feature to both children
            branch_features[left_idx].push(real_feature);
            branch_features[right_idx].push(real_feature);
        }
    }

    /// Get the upper bound value among all leaf values.
    pub fn get_upper_bound_value(&self) -> f64 {
        let mut upper_bound = self.leaf_value[0];
        for i in 1..self.num_leaves as usize {
            if self.leaf_value[i] > upper_bound {
                upper_bound = self.leaf_value[i];
            }
        }
        upper_bound
    }

    /// Get the lower bound value among all leaf values.
    pub fn get_lower_bound_value(&self) -> f64 {
        let mut lower_bound = self.leaf_value[0];
        for i in 1..self.num_leaves as usize {
            if self.leaf_value[i] < lower_bound {
                lower_bound = self.leaf_value[i];
            }
        }
        lower_bound
    }

    /// Convert tree to string representation (LightGBM format).
    pub fn to_string(&self) -> String {
        let mut result = String::new();

        result.push_str(&format!("num_leaves={}\n", self.num_leaves));
        result.push_str(&format!("num_cat={}\n", self.num_cat));
        result.push_str(&format!(
            "split_feature={}\n",
            array_to_string(&self.split_feature[0..(self.num_leaves - 1) as usize])
        ));
        result.push_str(&format!(
            "split_gain={}\n",
            array_to_string(&self.split_gain[0..(self.num_leaves - 1) as usize])
        ));
        result.push_str(&format!(
            "threshold={}\n",
            array_to_string_precise(&self.threshold[0..(self.num_leaves - 1) as usize])
        ));
        result.push_str(&format!(
            "decision_type={}\n",
            array_to_string(&self.decision_type[0..(self.num_leaves - 1) as usize])
        ));
        result.push_str(&format!(
            "left_child={}\n",
            array_to_string(&self.left_child[0..(self.num_leaves - 1) as usize])
        ));
        result.push_str(&format!(
            "right_child={}\n",
            array_to_string(&self.right_child[0..(self.num_leaves - 1) as usize])
        ));
        result.push_str(&format!(
            "leaf_value={}\n",
            array_to_string_precise(&self.leaf_value[0..self.num_leaves as usize])
        ));
        result.push_str(&format!(
            "leaf_weight={}\n",
            array_to_string_precise(&self.leaf_weight[0..self.num_leaves as usize])
        ));
        result.push_str(&format!(
            "leaf_count={}\n",
            array_to_string(&self.leaf_count[0..self.num_leaves as usize])
        ));
        result.push_str(&format!(
            "internal_value={}\n",
            array_to_string(&self.internal_value[0..(self.num_leaves - 1) as usize])
        ));
        result.push_str(&format!(
            "internal_weight={}\n",
            array_to_string(&self.internal_weight[0..(self.num_leaves - 1) as usize])
        ));
        result.push_str(&format!(
            "internal_count={}\n",
            array_to_string(&self.internal_count[0..(self.num_leaves - 1) as usize])
        ));

        if self.num_cat > 0 {
            result.push_str(&format!(
                "cat_boundaries={}\n",
                array_to_string(&self.cat_boundaries)
            ));
            result.push_str(&format!(
                "cat_threshold={}\n",
                array_to_string(&self.cat_threshold)
            ));
        }

        result.push_str(&format!(
            "is_linear={}\n",
            if self.is_linear { 1 } else { 0 }
        ));

        if self.is_linear {
            result.push_str(&format!(
                "leaf_const={}\n",
                array_to_string_precise(&self.leaf_const)
            ));
            let num_feat: Vec<i32> = (0..self.num_leaves)
                .map(|i| self.leaf_coeff[i as usize].len() as i32)
                .collect();
            result.push_str(&format!("num_features={}\n", array_to_string(&num_feat)));

            result.push_str("leaf_features=");
            for i in 0..self.num_leaves as usize {
                if !self.leaf_features[i].is_empty() {
                    result.push_str(&array_to_string(&self.leaf_features[i]));
                    result.push(' ');
                }
                result.push(' ');
            }
            result.push('\n');

            result.push_str("leaf_coeff=");
            for i in 0..self.num_leaves as usize {
                if !self.leaf_coeff[i].is_empty() {
                    result.push_str(&array_to_string_precise(&self.leaf_coeff[i]));
                    result.push(' ');
                }
                result.push(' ');
            }
            result.push('\n');
        }

        result.push_str(&format!("shrinkage={}\n", self.shrinkage));
        result.push('\n');

        result
    }

    /// Helper method to set decision type bits.
    fn set_decision_type(&mut self, node_idx: usize, value: bool, mask: u8) {
        if value {
            self.decision_type[node_idx] |= mask as i8;
        } else {
            self.decision_type[node_idx] &= !(mask as i8);
        }
    }

    /// Helper method to set missing type in decision type.
    fn set_missing_type(&mut self, node_idx: usize, missing_type: i8) {
        // Clear the missing type bits (bits 2-3) and set new value
        self.decision_type[node_idx] &= !0x0C; // Clear bits 2-3
        self.decision_type[node_idx] |= (missing_type & 0x03) << 2; // Set bits 2-3
    }

    /// Helper method to get decision type bit.
    fn get_decision_type(&self, decision_type: i8, mask: u8) -> bool {
        (decision_type & mask as i8) != 0
    }

    /// Helper method to get missing type from decision type.
    fn get_missing_type(&self, decision_type: i8) -> u8 {
        ((decision_type >> 2) & 0x03) as u8
    }

    /// Get the number of leaves in the tree.
    pub fn num_leaves(&self) -> i32 {
        self.num_leaves
    }

    /// Get leaf output value (for linear trees, this may include linear model computation).
    pub fn leaf_output(&self, leaf_idx: usize) -> f64 {
        if self.is_linear && !self.leaf_coeff[leaf_idx].is_empty() {
            // For linear trees, this would need feature values to compute the full output
            // For now, return the constant term
            self.leaf_const[leaf_idx]
        } else {
            self.leaf_value[leaf_idx]
        }
    }

    /// Get data count for a node (leaf or internal).
    pub fn data_count(&self, node_idx: i32) -> f64 {
        if node_idx >= 0 {
            // Internal node
            self.internal_count[node_idx as usize] as f64
        } else {
            // Leaf node (negative index)
            self.leaf_count[(!node_idx) as usize] as f64
        }
    }

    /// Make a decision at an internal node.
    pub fn decision(&self, feature_value: f64, node_idx: i32) -> i32 {
        if self.get_decision_type(self.decision_type[node_idx as usize], CATEGORICAL_MASK) {
            self.categorical_decision(feature_value, node_idx)
        } else {
            self.numerical_decision(feature_value, node_idx)
        }
    }

    /// Make a numerical decision at an internal node.
    fn numerical_decision(&self, feature_value: f64, node_idx: i32) -> i32 {
        let node_idx = node_idx as usize;
        let missing_type = self.get_missing_type(self.decision_type[node_idx]);
        let default_left = self.get_decision_type(self.decision_type[node_idx], DEFAULT_LEFT_MASK);

        let mut fval = feature_value;

        // Handle missing values
        if missing_type != MissingType::NaN as u8 && fval.is_nan() {
            fval = 0.0;
        }

        let go_left = if missing_type == MissingType::Zero as u8 {
            if Self::is_zero_internal(fval) {
                default_left
            } else {
                !default_left
            }
        } else if missing_type == MissingType::NaN as u8 {
            if fval.is_nan() {
                default_left
            } else {
                fval <= self.threshold[node_idx]
            }
        } else {
            fval <= self.threshold[node_idx]
        };

        if go_left {
            self.left_child[node_idx]
        } else {
            self.right_child[node_idx]
        }
    }

    /// Make a categorical decision at an internal node.
    fn categorical_decision(&self, feature_value: f64, node_idx: i32) -> i32 {
        let node_idx = node_idx as usize;
        let cat_idx = self.threshold[node_idx] as usize;

        let int_fval = if feature_value.is_nan() {
            -1
        } else {
            feature_value as i32
        };

        let go_left = if int_fval >= 0
            && int_fval < (32 * (self.cat_boundaries[cat_idx + 1] - self.cat_boundaries[cat_idx]))
            && self.find_in_bitset(
                &self.cat_threshold[self.cat_boundaries[cat_idx] as usize..],
                (self.cat_boundaries[cat_idx + 1] - self.cat_boundaries[cat_idx]) as usize,
                int_fval,
            ) {
            true
        } else {
            false
        };

        if go_left {
            self.left_child[node_idx]
        } else {
            self.right_child[node_idx]
        }
    }

    /// Helper function to check if a value is effectively zero.
    fn is_zero_internal(value: f64) -> bool {
        value.abs() < f64::EPSILON
    }

    /// Helper function to find a value in a bitset representation.
    fn find_in_bitset(&self, bitset: &[u32], size: usize, val: i32) -> bool {
        if val < 0 || val >= (size * 32) as i32 {
            return false;
        }
        let byte_idx = (val / 32) as usize;
        let bit_idx = val % 32;
        if byte_idx < bitset.len() {
            (bitset[byte_idx] >> bit_idx) & 1 != 0
        } else {
            false
        }
    }

    /// Add prediction contributions to score array for all data points.
    /// This is equivalent to the C++ AddPredictionToScore method.
    pub fn add_prediction_to_score(&self, dataset: &Dataset, scores: &mut [f64]) -> Result<()> {
        let _num_data = scores.len();

        // Handle single leaf case
        if !self.is_linear && self.num_leaves <= 1 {
            if self.leaf_value[0] != 0.0 {
                scores.par_iter_mut().for_each(|score| {
                    *score += self.leaf_value[0];
                });
            }
            return Ok(());
        }

        // Get default and max bins for each internal node
        let mut default_bins = vec![0u32; (self.num_leaves - 1) as usize];
        let mut max_bins = vec![0u32; (self.num_leaves - 1) as usize];

        for i in 0..(self.num_leaves - 1) {
            let fidx = self.split_feature_inner[i as usize];
            if let Some(bin_mapper) = dataset.bin_mapper(fidx as usize) {
                default_bins[i as usize] = bin_mapper.default_bin;
                max_bins[i as usize] = (bin_mapper.num_bins - 1) as u32;
            } else {
                bail!("No bin mapper found for feature index {}", fidx);
            }
        }

        if self.is_linear {
            self.add_prediction_to_score_linear(dataset, scores, &default_bins, &max_bins)
        } else {
            self.add_prediction_to_score_standard(dataset, scores, &default_bins, &max_bins)
        }
    }

    /// Add prediction contributions to score array for selected data indices.
    pub fn add_prediction_to_score_with_indices(
        &self,
        dataset: &Dataset,
        used_data_indices: &[DataSize],
        scores: &mut [f64],
    ) -> Result<()> {
        // Handle single leaf case
        if !self.is_linear && self.num_leaves <= 1 {
            if self.leaf_value[0] != 0.0 {
                for &idx in used_data_indices {
                    scores[idx as usize] += self.leaf_value[0];
                }
            }
            return Ok(());
        }

        // Get default and max bins for each internal node
        let mut default_bins = vec![0u32; (self.num_leaves - 1) as usize];
        let mut max_bins = vec![0u32; (self.num_leaves - 1) as usize];

        for i in 0..(self.num_leaves - 1) {
            let fidx = self.split_feature_inner[i as usize];
            if let Some(bin_mapper) = dataset.bin_mapper(fidx as usize) {
                default_bins[i as usize] = bin_mapper.default_bin;
                max_bins[i as usize] = (bin_mapper.num_bins - 1) as u32;
            } else {
                bail!("No bin mapper found for feature index {}", fidx);
            }
        }

        if self.is_linear {
            self.add_prediction_to_score_linear_with_indices(
                dataset,
                used_data_indices,
                scores,
                &default_bins,
                &max_bins,
            )
        } else {
            self.add_prediction_to_score_standard_with_indices(
                dataset,
                used_data_indices,
                scores,
                &default_bins,
                &max_bins,
            )
        }
    }

    /// Standard prediction for non-linear trees.
    fn add_prediction_to_score_standard(
        &self,
        dataset: &Dataset,
        scores: &mut [f64],
        default_bins: &[u32],
        max_bins: &[u32],
    ) -> Result<()> {
        let features = dataset.features();

        scores
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(data_idx, score)| {
                let mut node = 0i32;

                // Traverse tree until we reach a leaf
                while node >= 0 {
                    let node_idx = node as usize;
                    let feature_idx = self.split_feature_inner[node_idx];
                    let feature_value = features[[data_idx, feature_idx as usize]] as f64;

                    node = self.decision_inner(
                        feature_value,
                        node,
                        default_bins[node_idx],
                        max_bins[node_idx],
                    );
                }

                // Add leaf value to score
                *score += self.leaf_value[(!node) as usize];

                Ok::<(), anyhow::Error>(())
            })?;

        Ok(())
    }

    /// Standard prediction for non-linear trees with specific indices.
    fn add_prediction_to_score_standard_with_indices(
        &self,
        dataset: &Dataset,
        used_data_indices: &[DataSize],
        scores: &mut [f64],
        default_bins: &[u32],
        max_bins: &[u32],
    ) -> Result<()> {
        let features = dataset.features();

        for &data_idx in used_data_indices {
            let mut node = 0i32;

            // Traverse tree until we reach a leaf
            while node >= 0 {
                let node_idx = node as usize;
                let feature_idx = self.split_feature_inner[node_idx];
                let feature_value = features[[data_idx as usize, feature_idx as usize]] as f64;

                node = self.decision_inner(
                    feature_value,
                    node,
                    default_bins[node_idx],
                    max_bins[node_idx],
                );
            }

            // Add leaf value to score
            scores[data_idx as usize] += self.leaf_value[(!node) as usize];
        }

        Ok(())
    }

    /// Linear prediction for linear trees.
    fn add_prediction_to_score_linear(
        &self,
        dataset: &Dataset,
        scores: &mut [f64],
        default_bins: &[u32],
        max_bins: &[u32],
    ) -> Result<()> {
        let features = dataset.features();

        // Prepare feature pointers for linear leaves
        let mut feat_ptr = vec![Vec::new(); self.num_leaves as usize];
        for leaf_num in 0..self.num_leaves as usize {
            for &feat in &self.leaf_features_inner[leaf_num] {
                feat_ptr[leaf_num].push(feat);
            }
        }

        scores
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(data_idx, score)| {
                let mut node = 0i32;

                if self.num_leaves > 1 {
                    // Traverse tree until we reach a leaf
                    while node >= 0 {
                        let node_idx = node as usize;
                        let feature_idx = self.split_feature_inner[node_idx];
                        let feature_value = features[[data_idx, feature_idx as usize]] as f64;

                        node = self.decision_inner(
                            feature_value,
                            node,
                            default_bins[node_idx],
                            max_bins[node_idx],
                        );
                    }
                    node = !node;
                }

                let leaf_idx = node as usize;
                let mut add_score = self.leaf_const[leaf_idx];
                let mut nan_found = false;

                // Compute linear model contribution
                for (j, &feat_idx) in feat_ptr[leaf_idx].iter().enumerate() {
                    let feat_val = features[[data_idx, feat_idx as usize]] as f64;
                    if feat_val.is_nan() {
                        nan_found = true;
                        break;
                    }
                    add_score += self.leaf_coeff[leaf_idx][j] * feat_val;
                }

                if nan_found {
                    *score += self.leaf_value[leaf_idx];
                } else {
                    *score += add_score;
                }

                Ok::<(), anyhow::Error>(())
            })?;

        Ok(())
    }

    /// Linear prediction for linear trees with specific indices.
    fn add_prediction_to_score_linear_with_indices(
        &self,
        dataset: &Dataset,
        used_data_indices: &[DataSize],
        scores: &mut [f64],
        default_bins: &[u32],
        max_bins: &[u32],
    ) -> Result<()> {
        let features = dataset.features();

        // Prepare feature pointers for linear leaves
        let mut feat_ptr = vec![Vec::new(); self.num_leaves as usize];
        for leaf_num in 0..self.num_leaves as usize {
            for &feat in &self.leaf_features_inner[leaf_num] {
                feat_ptr[leaf_num].push(feat);
            }
        }

        for &data_idx in used_data_indices {
            let mut node = 0i32;

            if self.num_leaves > 1 {
                // Traverse tree until we reach a leaf
                while node >= 0 {
                    let node_idx = node as usize;
                    let feature_idx = self.split_feature_inner[node_idx];
                    let feature_value = features[[data_idx as usize, feature_idx as usize]] as f64;

                    node = self.decision_inner(
                        feature_value,
                        node,
                        default_bins[node_idx],
                        max_bins[node_idx],
                    );
                }
                node = !node;
            }

            let leaf_idx = node as usize;
            let mut add_score = self.leaf_const[leaf_idx];
            let mut nan_found = false;

            // Compute linear model contribution
            for (j, &feat_idx) in feat_ptr[leaf_idx].iter().enumerate() {
                let feat_val = features[[data_idx as usize, feat_idx as usize]] as f64;
                if feat_val.is_nan() {
                    nan_found = true;
                    break;
                }
                add_score += self.leaf_coeff[leaf_idx][j] * feat_val;
            }

            if nan_found {
                scores[data_idx as usize] += self.leaf_value[leaf_idx];
            } else {
                scores[data_idx as usize] += add_score;
            }
        }

        Ok(())
    }

    /// Internal decision function that considers bin bounds.
    fn decision_inner(
        &self,
        feature_value: f64,
        node_idx: i32,
        default_bin: u32,
        max_bin: u32,
    ) -> i32 {
        if self.get_decision_type(self.decision_type[node_idx as usize], CATEGORICAL_MASK) {
            self.categorical_decision_inner(feature_value, node_idx, default_bin, max_bin)
        } else {
            self.numerical_decision_inner(feature_value, node_idx, default_bin, max_bin)
        }
    }

    /// Numerical decision with bin considerations.
    fn numerical_decision_inner(
        &self,
        feature_value: f64,
        node_idx: i32,
        _default_bin: u32,
        _max_bin: u32,
    ) -> i32 {
        // For numerical decisions, we can use the standard decision logic
        self.numerical_decision(feature_value, node_idx)
    }

    /// Categorical decision with bin considerations.
    fn categorical_decision_inner(
        &self,
        feature_value: f64,
        node_idx: i32,
        _default_bin: u32,
        _max_bin: u32,
    ) -> i32 {
        // For categorical decisions, we can use the standard decision logic
        self.categorical_decision(feature_value, node_idx)
    }

    /// Convert tree to JSON representation.
    pub fn to_json(&self) -> String {
        let mut result = String::new();

        result.push_str(&format!("\"num_leaves\":{},\n", self.num_leaves));
        result.push_str(&format!("\"num_cat\":{},\n", self.num_cat));
        result.push_str(&format!("\"shrinkage\":{},\n", self.shrinkage));

        if self.num_leaves == 1 {
            result.push_str("\"tree_structure\":{");
            result.push_str(&format!("\"leaf_value\":{}, \n", self.leaf_value[0]));
            if self.is_linear {
                result.push_str(&format!("\"leaf_count\":{}, \n", self.leaf_count[0]));
                result.push_str(&self.linear_model_to_json(0));
            } else {
                result.push_str(&format!("\"leaf_count\":{}", self.leaf_count[0]));
            }
            result.push_str("}\n");
        } else {
            result.push_str(&format!("\"tree_structure\":{}\n", self.node_to_json(0)));
        }

        result
    }

    /// Convert a linear model to JSON representation.
    fn linear_model_to_json(&self, index: usize) -> String {
        let mut result = String::new();

        result.push_str(&format!("\"leaf_const\":{},\n", self.leaf_const[index]));
        let num_features = self.leaf_features[index].len();

        if num_features > 0 {
            result.push_str("\"leaf_features\":[");
            for i in 0..num_features {
                if i > 0 {
                    result.push_str(", ");
                }
                result.push_str(&self.leaf_features[index][i].to_string());
            }
            result.push_str("], \n");

            result.push_str("\"leaf_coeff\":[");
            for i in 0..num_features {
                if i > 0 {
                    result.push_str(", ");
                }
                result.push_str(&format!("{:.17}", self.leaf_coeff[index][i]));
            }
            result.push_str("]\n");
        } else {
            result.push_str("\"leaf_features\":[],\n");
            result.push_str("\"leaf_coeff\":[]\n");
        }

        result
    }

    /// Convert a tree node to JSON representation recursively.
    fn node_to_json(&self, index: i32) -> String {
        let mut result = String::new();

        if index >= 0 {
            // Internal node
            let idx = index as usize;
            result.push_str("{\n");
            result.push_str(&format!("\"split_index\":{},\n", index));
            result.push_str(&format!("\"split_feature\":{},\n", self.split_feature[idx]));
            result.push_str(&format!(
                "\"split_gain\":{},\n",
                self.avoid_inf_f32(self.split_gain[idx])
            ));

            if self.get_decision_type(self.decision_type[idx], CATEGORICAL_MASK) {
                let cat_idx = self.threshold[idx] as usize;
                let mut cats = Vec::new();

                for i in self.cat_boundaries[cat_idx]..self.cat_boundaries[cat_idx + 1] {
                    for j in 0..32 {
                        let cat = (i - self.cat_boundaries[cat_idx]) * 32 + j;
                        if self.find_in_bitset(
                            &self.cat_threshold[self.cat_boundaries[cat_idx] as usize..],
                            (self.cat_boundaries[cat_idx + 1] - self.cat_boundaries[cat_idx])
                                as usize,
                            cat as i32,
                        ) {
                            cats.push(cat.to_string());
                        }
                    }
                }

                result.push_str(&format!("\"threshold\":\"{}\",\n", cats.join("||")));
                result.push_str("\"decision_type\":\"==\",\n");
            } else {
                result.push_str(&format!(
                    "\"threshold\":{},\n",
                    self.avoid_inf_f64(self.threshold[idx])
                ));
                result.push_str("\"decision_type\":\"<=\",\n");
            }

            if self.get_decision_type(self.decision_type[idx], DEFAULT_LEFT_MASK) {
                result.push_str("\"default_left\":true,\n");
            } else {
                result.push_str("\"default_left\":false,\n");
            }

            let missing_type = self.get_missing_type(self.decision_type[idx]);
            match missing_type {
                0 => result.push_str("\"missing_type\":\"None\",\n"), // MissingType::None
                1 => result.push_str("\"missing_type\":\"Zero\",\n"), // MissingType::Zero
                2 => result.push_str("\"missing_type\":\"NaN\",\n"),  // MissingType::NaN
                _ => result.push_str("\"missing_type\":\"NaN\",\n"),
            }

            result.push_str(&format!(
                "\"internal_value\":{},\n",
                self.internal_value[idx]
            ));
            result.push_str(&format!(
                "\"internal_weight\":{},\n",
                self.internal_weight[idx]
            ));
            result.push_str(&format!(
                "\"internal_count\":{},\n",
                self.internal_count[idx]
            ));
            result.push_str(&format!(
                "\"left_child\":{},\n",
                self.node_to_json(self.left_child[idx])
            ));
            result.push_str(&format!(
                "\"right_child\":{}\n",
                self.node_to_json(self.right_child[idx])
            ));
            result.push('}');
        } else {
            // Leaf node
            let leaf_idx = (!index) as usize;
            result.push_str("{\n");
            result.push_str(&format!("\"leaf_index\":{},\n", leaf_idx));
            result.push_str(&format!("\"leaf_value\":{},\n", self.leaf_value[leaf_idx]));
            result.push_str(&format!(
                "\"leaf_weight\":{},\n",
                self.leaf_weight[leaf_idx]
            ));

            if self.is_linear {
                result.push_str(&format!("\"leaf_count\":{},\n", self.leaf_count[leaf_idx]));
                result.push_str(&self.linear_model_to_json(leaf_idx));
            } else {
                result.push_str(&format!("\"leaf_count\":{}\n", self.leaf_count[leaf_idx]));
            }
            result.push('}');
        }

        result
    }

    /// Avoid infinite values in f32.
    fn avoid_inf_f32(&self, value: f32) -> f32 {
        if value.is_infinite() {
            if value > 0.0 {
                1e30_f32
            } else {
                -1e30_f32
            }
        } else {
            value
        }
    }

    /// Avoid infinite values in f64.
    fn avoid_inf_f64(&self, value: f64) -> f64 {
        if value.is_infinite() {
            if value > 0.0 {
                1e30_f64
            } else {
                -1e30_f64
            }
        } else {
            value
        }
    }

    /// Get expected value of the tree.
    pub fn expected_value(&self) -> f64 {
        if self.num_leaves == 1 {
            return self.leaf_output(0);
        }

        let total_count = self.internal_count[0] as f64;
        let mut exp_value = 0.0;

        for i in 0..self.num_leaves as usize {
            exp_value += (self.leaf_count[i] as f64 / total_count) * self.leaf_output(i);
        }

        exp_value
    }

    /// Recompute maximum depth of the tree.
    pub fn recompute_max_depth(&mut self) {
        if self.num_leaves == 1 {
            self.max_depth = 0;
        } else {
            if self.leaf_depth.is_empty() {
                self.recompute_leaf_depths(0, 0);
            }
            self.max_depth = *self.leaf_depth[0..self.num_leaves as usize]
                .iter()
                .max()
                .unwrap_or(&0);
        }
    }

    /// Recompute leaf depths recursively.
    fn recompute_leaf_depths(&mut self, node_idx: i32, depth: i32) {
        if node_idx < 0 {
            // Leaf node
            let leaf_idx = (!node_idx) as usize;
            if leaf_idx < self.leaf_depth.len() {
                self.leaf_depth[leaf_idx] = depth;
            }
        } else {
            // Internal node
            let idx = node_idx as usize;
            if idx < self.left_child.len() && idx < self.right_child.len() {
                self.recompute_leaf_depths(self.left_child[idx], depth + 1);
                self.recompute_leaf_depths(self.right_child[idx], depth + 1);
            }
        }
    }

    /// Compute SHAP values for the given feature values.
    pub fn tree_shap(
        &self,
        feature_values: &[f64],
        phi: &mut [f64],
        node: i32,
        unique_depth: usize,
        parent_unique_path: &mut [PathElement],
        parent_zero_fraction: f64,
        parent_one_fraction: f64,
        parent_feature_index: i32,
    ) {
        // Extend the unique path
        if unique_depth > 0 {
            for i in 0..unique_depth {
                parent_unique_path[unique_depth + i] = parent_unique_path[i];
            }
        }
        let unique_path = &mut parent_unique_path[unique_depth..];

        Self::extend_path(
            unique_path,
            unique_depth,
            parent_zero_fraction,
            parent_one_fraction,
            parent_feature_index,
        );

        if node < 0 {
            // Leaf node
            let leaf_idx = (!node) as usize;
            for i in 1..=unique_depth {
                let w = Self::unwound_path_sum(unique_path, unique_depth, i);
                let el = &unique_path[i];
                phi[el.feature_index as usize] +=
                    w * (el.one_fraction - el.zero_fraction) * self.leaf_value[leaf_idx];
            }
        } else {
            // Internal node
            let node_idx = node as usize;
            let hot_index =
                self.decision(feature_values[self.split_feature[node_idx] as usize], node);
            let cold_index = if hot_index == self.left_child[node_idx] {
                self.right_child[node_idx]
            } else {
                self.left_child[node_idx]
            };

            let w = self.data_count(node);
            let hot_zero_fraction = self.data_count(hot_index) / w;
            let cold_zero_fraction = self.data_count(cold_index) / w;
            let mut incoming_zero_fraction = 1.0;
            let mut incoming_one_fraction = 1.0;

            // Check if we have already split on this feature
            let mut path_index = 0;
            for i in 0..=unique_depth {
                if unique_path[i].feature_index == self.split_feature[node_idx] {
                    path_index = i;
                    break;
                }
            }

            if path_index <= unique_depth {
                incoming_zero_fraction = unique_path[path_index].zero_fraction;
                incoming_one_fraction = unique_path[path_index].one_fraction;
                Self::unwind_path(unique_path, unique_depth, path_index);
                let new_unique_depth = unique_depth - 1;

                self.tree_shap(
                    feature_values,
                    phi,
                    hot_index,
                    new_unique_depth + 1,
                    parent_unique_path,
                    hot_zero_fraction * incoming_zero_fraction,
                    incoming_one_fraction,
                    self.split_feature[node_idx],
                );

                self.tree_shap(
                    feature_values,
                    phi,
                    cold_index,
                    new_unique_depth + 1,
                    parent_unique_path,
                    cold_zero_fraction * incoming_zero_fraction,
                    0.0,
                    self.split_feature[node_idx],
                );
            } else {
                self.tree_shap(
                    feature_values,
                    phi,
                    hot_index,
                    unique_depth + 1,
                    parent_unique_path,
                    hot_zero_fraction * incoming_zero_fraction,
                    incoming_one_fraction,
                    self.split_feature[node_idx],
                );

                self.tree_shap(
                    feature_values,
                    phi,
                    cold_index,
                    unique_depth + 1,
                    parent_unique_path,
                    cold_zero_fraction * incoming_zero_fraction,
                    0.0,
                    self.split_feature[node_idx],
                );
            }
        }
    }

    /// Compute SHAP values using a sparse feature map.
    pub fn tree_shap_by_map(
        &self,
        feature_values: &HashMap<i32, f64>,
        phi: &mut HashMap<i32, f64>,
        node: i32,
        unique_depth: usize,
        parent_unique_path: &mut [PathElement],
        parent_zero_fraction: f64,
        parent_one_fraction: f64,
        parent_feature_index: i32,
    ) {
        // Extend the unique path
        if unique_depth > 0 {
            for i in 0..unique_depth {
                parent_unique_path[unique_depth + i] = parent_unique_path[i];
            }
        }
        let unique_path = &mut parent_unique_path[unique_depth..];

        Self::extend_path(
            unique_path,
            unique_depth,
            parent_zero_fraction,
            parent_one_fraction,
            parent_feature_index,
        );

        if node < 0 {
            // Leaf node
            let leaf_idx = (!node) as usize;
            for i in 1..=unique_depth {
                let w = Self::unwound_path_sum(unique_path, unique_depth, i);
                let el = &unique_path[i];
                *phi.entry(el.feature_index).or_insert(0.0) +=
                    w * (el.one_fraction - el.zero_fraction) * self.leaf_value[leaf_idx];
            }
        } else {
            // Internal node
            let node_idx = node as usize;
            let feature_value = feature_values
                .get(&self.split_feature[node_idx])
                .copied()
                .unwrap_or(0.0);
            let hot_index = self.decision(feature_value, node);
            let cold_index = if hot_index == self.left_child[node_idx] {
                self.right_child[node_idx]
            } else {
                self.left_child[node_idx]
            };

            let w = self.data_count(node);
            let hot_zero_fraction = self.data_count(hot_index) / w;
            let cold_zero_fraction = self.data_count(cold_index) / w;
            let mut incoming_zero_fraction = 1.0;
            let mut incoming_one_fraction = 1.0;

            // Check if we have already split on this feature
            let mut path_index = 0;
            for i in 0..=unique_depth {
                if unique_path[i].feature_index == self.split_feature[node_idx] {
                    path_index = i;
                    break;
                }
            }

            if path_index <= unique_depth {
                incoming_zero_fraction = unique_path[path_index].zero_fraction;
                incoming_one_fraction = unique_path[path_index].one_fraction;
                Self::unwind_path(unique_path, unique_depth, path_index);
                let new_unique_depth = unique_depth - 1;

                self.tree_shap_by_map(
                    feature_values,
                    phi,
                    hot_index,
                    new_unique_depth + 1,
                    parent_unique_path,
                    hot_zero_fraction * incoming_zero_fraction,
                    incoming_one_fraction,
                    self.split_feature[node_idx],
                );

                self.tree_shap_by_map(
                    feature_values,
                    phi,
                    cold_index,
                    new_unique_depth + 1,
                    parent_unique_path,
                    cold_zero_fraction * incoming_zero_fraction,
                    0.0,
                    self.split_feature[node_idx],
                );
            } else {
                self.tree_shap_by_map(
                    feature_values,
                    phi,
                    hot_index,
                    unique_depth + 1,
                    parent_unique_path,
                    hot_zero_fraction * incoming_zero_fraction,
                    incoming_one_fraction,
                    self.split_feature[node_idx],
                );

                self.tree_shap_by_map(
                    feature_values,
                    phi,
                    cold_index,
                    unique_depth + 1,
                    parent_unique_path,
                    cold_zero_fraction * incoming_zero_fraction,
                    0.0,
                    self.split_feature[node_idx],
                );
            }
        }
    }

    /// Extend path for SHAP computation.
    fn extend_path(
        unique_path: &mut [PathElement],
        unique_depth: usize,
        zero_fraction: f64,
        one_fraction: f64,
        feature_index: i32,
    ) {
        unique_path[unique_depth].feature_index = feature_index;
        unique_path[unique_depth].zero_fraction = zero_fraction;
        unique_path[unique_depth].one_fraction = one_fraction;
        unique_path[unique_depth].pweight = if unique_depth == 0 { 1.0 } else { 0.0 };

        for i in (0..unique_depth).rev() {
            unique_path[i + 1].pweight +=
                one_fraction * unique_path[i].pweight * (i + 1) as f64 / (unique_depth + 1) as f64;
            unique_path[i].pweight =
                zero_fraction * unique_path[i].pweight * (unique_depth - i) as f64
                    / (unique_depth + 1) as f64;
        }
    }

    /// Unwind path for SHAP computation.
    fn unwind_path(unique_path: &mut [PathElement], unique_depth: usize, path_index: usize) {
        let one_fraction = unique_path[path_index].one_fraction;
        let zero_fraction = unique_path[path_index].zero_fraction;
        let mut next_one_portion = unique_path[unique_depth].pweight;

        for i in (0..unique_depth).rev() {
            if one_fraction != 0.0 {
                let tmp = unique_path[i].pweight;
                unique_path[i].pweight =
                    next_one_portion * (unique_depth + 1) as f64 / ((i + 1) as f64 * one_fraction);
                next_one_portion = tmp
                    - unique_path[i].pweight * zero_fraction * (unique_depth - i) as f64
                        / (unique_depth + 1) as f64;
            } else {
                unique_path[i].pweight = (unique_path[i].pweight * (unique_depth + 1) as f64)
                    / (zero_fraction * (unique_depth - i) as f64);
            }
        }

        for i in path_index..unique_depth {
            unique_path[i].feature_index = unique_path[i + 1].feature_index;
            unique_path[i].zero_fraction = unique_path[i + 1].zero_fraction;
            unique_path[i].one_fraction = unique_path[i + 1].one_fraction;
        }
    }

    /// Compute unwound path sum for SHAP computation.
    fn unwound_path_sum(
        unique_path: &[PathElement],
        unique_depth: usize,
        path_index: usize,
    ) -> f64 {
        let one_fraction = unique_path[path_index].one_fraction;
        let zero_fraction = unique_path[path_index].zero_fraction;
        let mut next_one_portion = unique_path[unique_depth].pweight;
        let mut total = 0.0;

        for i in (0..unique_depth).rev() {
            if one_fraction != 0.0 {
                let tmp =
                    next_one_portion * (unique_depth + 1) as f64 / ((i + 1) as f64 * one_fraction);
                total += tmp;
                next_one_portion = unique_path[i].pweight
                    - tmp * zero_fraction * ((unique_depth - i) as f64 / (unique_depth + 1) as f64);
            } else {
                total += (unique_path[i].pweight / zero_fraction)
                    / ((unique_depth - i) as f64 / (unique_depth + 1) as f64);
            }
        }

        total
    }

    /// Convert tree to if-else code representation.
    pub fn to_if_else(&self, index: i32, predict_leaf_index: bool) -> String {
        let mut result = String::new();

        result.push_str(&format!("double PredictTree{}", index));
        if predict_leaf_index {
            result.push_str("Leaf");
        }
        result.push_str("(const double* arr) { ");

        if self.num_leaves <= 1 {
            result.push_str(&format!("return {};", self.leaf_value[0]));
        } else {
            result.push_str("const std::vector<uint32_t> cat_threshold = {");
            for (i, &thresh) in self.cat_threshold.iter().enumerate() {
                if i != 0 {
                    result.push(',');
                }
                result.push_str(&thresh.to_string());
            }
            result.push_str("};");

            // Use this for the missing value conversion
            result.push_str("double fval = 0.0f; ");
            if self.num_cat > 0 {
                result.push_str("int int_fval = 0; ");
            }
            result.push_str(&self.node_to_if_else(0, predict_leaf_index));
        }
        result.push_str(" }\n");

        // Predict func by Map to ifelse
        result.push_str(&format!("double PredictTree{}", index));
        if predict_leaf_index {
            result.push_str("LeafByMap");
        } else {
            result.push_str("ByMap");
        }
        result.push_str("(const std::unordered_map<int, double>& arr) { ");

        if self.num_leaves <= 1 {
            result.push_str(&format!("return {};", self.leaf_value[0]));
        } else {
            result.push_str("const std::vector<uint32_t> cat_threshold = {");
            for (i, &thresh) in self.cat_threshold.iter().enumerate() {
                if i != 0 {
                    result.push(',');
                }
                result.push_str(&thresh.to_string());
            }
            result.push_str("};");

            // Use this for the missing value conversion
            result.push_str("double fval = 0.0f; ");
            if self.num_cat > 0 {
                result.push_str("int int_fval = 0; ");
            }
            result.push_str(&self.node_to_if_else_by_map(0, predict_leaf_index));
        }
        result.push_str(" }\n");

        result
    }

    /// Convert node to if-else representation.
    fn node_to_if_else(&self, index: i32, predict_leaf_index: bool) -> String {
        let mut result = String::new();

        if index >= 0 {
            // Non-leaf
            let idx = index as usize;
            result.push_str(&format!("fval = arr[{}];", self.split_feature[idx]));

            if !self.get_decision_type(self.decision_type[idx], CATEGORICAL_MASK) {
                result.push_str(&self.numerical_decision_if_else(index));
            } else {
                result.push_str(&self.categorical_decision_if_else(index));
            }

            // Left subtree
            result.push_str(&self.node_to_if_else(self.left_child[idx], predict_leaf_index));
            result.push_str(" } else { ");
            // Right subtree
            result.push_str(&self.node_to_if_else(self.right_child[idx], predict_leaf_index));
            result.push_str(" }");
        } else {
            // Leaf
            result.push_str("return ");
            if predict_leaf_index {
                result.push_str(&(!index).to_string());
            } else {
                result.push_str(&self.leaf_value[(!index) as usize].to_string());
            }
            result.push(';');
        }

        result
    }

    /// Convert node to if-else representation using map.
    fn node_to_if_else_by_map(&self, index: i32, predict_leaf_index: bool) -> String {
        let mut result = String::new();

        if index >= 0 {
            // Non-leaf
            let idx = index as usize;
            result.push_str(&format!(
                "fval = arr.count({}) > 0 ? arr.at({}) : 0.0f;",
                self.split_feature[idx], self.split_feature[idx]
            ));

            if !self.get_decision_type(self.decision_type[idx], CATEGORICAL_MASK) {
                result.push_str(&self.numerical_decision_if_else(index));
            } else {
                result.push_str(&self.categorical_decision_if_else(index));
            }

            // Left subtree
            result.push_str(&self.node_to_if_else_by_map(self.left_child[idx], predict_leaf_index));
            result.push_str(" } else { ");
            // Right subtree
            result
                .push_str(&self.node_to_if_else_by_map(self.right_child[idx], predict_leaf_index));
            result.push_str(" }");
        } else {
            // Leaf
            result.push_str("return ");
            if predict_leaf_index {
                result.push_str(&(!index).to_string());
            } else {
                result.push_str(&self.leaf_value[(!index) as usize].to_string());
            }
            result.push(';');
        }

        result
    }

    /// Generate numerical decision if-else code.
    fn numerical_decision_if_else(&self, node: i32) -> String {
        let node_idx = node as usize;
        let mut result = String::new();

        let missing_type = self.get_missing_type(self.decision_type[node_idx]);
        let default_left = self.get_decision_type(self.decision_type[node_idx], DEFAULT_LEFT_MASK);

        if missing_type != MissingType::NaN as u8 {
            result.push_str("if (std::isnan(fval)) fval = 0.0;");
        }

        if missing_type == MissingType::Zero as u8 {
            if default_left {
                result.push_str("if (Tree::IsZero(fval)) {");
            } else {
                result.push_str("if (!Tree::IsZero(fval)) {");
            }
        } else if missing_type == MissingType::NaN as u8 {
            if default_left {
                result.push_str("if (std::isnan(fval)) {");
            } else {
                result.push_str("if (!std::isnan(fval)) {");
            }
        } else {
            result.push_str(&format!("if (fval <= {}) {{", self.threshold[node_idx]));
        }

        result
    }

    /// Generate categorical decision if-else code.
    fn categorical_decision_if_else(&self, node: i32) -> String {
        let node_idx = node as usize;
        let mut result = String::new();

        let cat_idx = self.threshold[node_idx] as usize;
        result.push_str(
            "if (std::isnan(fval)) { int_fval = -1; } else { int_fval = static_cast<int>(fval); }",
        );
        result.push_str(&format!("if (int_fval >= 0 && int_fval < 32 * ({}) && (((cat_threshold[{} + int_fval / 32] >> (int_fval & 31)) & 1))) {{",
            self.cat_boundaries[cat_idx + 1] - self.cat_boundaries[cat_idx],
            self.cat_boundaries[cat_idx]));

        result
    }

    /// Get features used in the branch leading to the given leaf.
    pub fn branch_features(&self, leaf: i32) -> Vec<i32> {
        if let Some(ref branch_features) = self.branch_features {
            if leaf >= 0 && (leaf as usize) < branch_features.len() {
                branch_features[leaf as usize].clone()
            } else {
                Vec::new()
            }
        } else {
            // If branch features tracking is not enabled, return empty vector
            Vec::new()
        }
    }

    /// Get the depth of a specific leaf.
    ///
    /// # Arguments
    /// * `leaf_idx` - Index of the leaf
    ///
    /// # Returns
    /// The depth of the leaf
    pub fn leaf_depth(&self, leaf_idx: i32) -> i32 {
        if leaf_idx >= 0 && (leaf_idx as usize) < self.leaf_depth.len() {
            self.leaf_depth[leaf_idx as usize]
        } else {
            0
        }
    }
}

/// Helper function to convert array to string representation.
fn array_to_string<T: fmt::Display>(arr: &[T]) -> String {
    arr.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Helper function to convert array to string with high precision.
fn array_to_string_precise<T: fmt::Display>(arr: &[T]) -> String {
    arr.iter()
        .map(|x| format!("{:.17}", x))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Check if a floating point value is effectively zero.
pub fn is_zero(value: f64) -> bool {
    value.abs() < f64::EPSILON
}

impl Default for Tree {
    fn default() -> Self {
        Self::new(31, false, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_creation() {
        let tree = Tree::new(31, false, false);
        assert_eq!(tree.num_leaves(), 1);
        assert_eq!(tree.max_leaves, 31);
        assert!(!tree.is_linear);
        assert!(!tree.track_branch_features);
    }

    #[test]
    fn test_tree_with_branch_features() {
        let tree = Tree::new(15, true, false);
        assert!(tree.branch_features.is_some());
        assert_eq!(tree.branch_features.as_ref().unwrap().len(), 15);
    }

    #[test]
    fn test_linear_tree() {
        let tree = Tree::new(10, false, true);
        assert!(tree.is_linear);
        assert_eq!(tree.leaf_coeff.len(), 10);
        assert_eq!(tree.leaf_const.len(), 10);
    }

    #[test]
    fn test_bound_values() {
        let mut tree = Tree::new(5, false, false);
        tree.leaf_value[0] = 1.0;
        tree.leaf_value[1] = 3.0;
        tree.leaf_value[2] = -1.0;
        tree.num_leaves = 3;

        assert_eq!(tree.get_upper_bound_value(), 3.0);
        assert_eq!(tree.get_lower_bound_value(), -1.0);
    }

    #[test]
    fn test_tree_string_conversion() {
        let tree = Tree::new(3, false, false);
        let tree_str = tree.to_string();
        assert!(tree_str.contains("num_leaves=1"));
        assert!(tree_str.contains("num_cat=0"));
        assert!(tree_str.contains("is_linear=0"));
    }
}
