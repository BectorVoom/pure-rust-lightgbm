#!/usr/bin/env rust-script

//! Test script to validate the leaf index implementation

use lightgbm_rust::boosting::{SimpleTree, SimpleTreeNode};
use lightgbm_rust::prediction::leaf_index::{LeafIndexPredictor, LeafIndexConfig};
use lightgbm_rust::boosting::GBDT;
use lightgbm_rust::config::Config;
use lightgbm_rust::dataset::Dataset;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing leaf index implementation...");

    // Test SimpleTree::predict_leaf_index method
    test_simple_tree_leaf_index()?;
    
    println!("All tests passed!");
    Ok(())
}

fn test_simple_tree_leaf_index() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing SimpleTree::predict_leaf_index...");
    
    // Create a simple tree with 3 nodes: root (split), left leaf, right leaf
    let root_node = SimpleTreeNode {
        feature_index: 0,      // Split on feature 0
        threshold: 5.0,        // Threshold of 5.0
        left_child: 1,         // Left child at index 1
        right_child: 2,        // Right child at index 2
        leaf_value: 0.0,       // Not used for internal nodes
        sample_count: 100,
        split_gain: 1.0,
        node_weight: 100.0,
        coverage: 100.0,
    };
    
    let left_leaf = SimpleTreeNode {
        feature_index: -1,     // -1 indicates leaf node
        threshold: 0.0,
        left_child: -1,
        right_child: -1,
        leaf_value: 1.0,       // Leaf value for left side
        sample_count: 60,
        split_gain: 0.0,
        node_weight: 60.0,
        coverage: 60.0,
    };
    
    let right_leaf = SimpleTreeNode {
        feature_index: -1,     // -1 indicates leaf node
        threshold: 0.0,
        left_child: -1,
        right_child: -1,
        leaf_value: -1.0,      // Leaf value for right side
        sample_count: 40,
        split_gain: 0.0,
        node_weight: 40.0,
        coverage: 40.0,
    };
    
    let tree = SimpleTree {
        nodes: vec![root_node, left_leaf, right_leaf],
        num_leaves: 2,
    };
    
    // Test features that should go to left leaf (feature[0] <= 5.0)
    let features_left = vec![3.0, 10.0, 20.0];  // feature[0] = 3.0 <= 5.0
    let leaf_index_left = tree.predict_leaf_index(&features_left);
    println!("Features {:?} -> Leaf index: {}", features_left, leaf_index_left);
    assert_eq!(leaf_index_left, 1, "Should go to left leaf (index 1)");
    
    // Test features that should go to right leaf (feature[0] > 5.0)
    let features_right = vec![7.0, 15.0, 25.0];  // feature[0] = 7.0 > 5.0
    let leaf_index_right = tree.predict_leaf_index(&features_right);
    println!("Features {:?} -> Leaf index: {}", features_right, leaf_index_right);
    assert_eq!(leaf_index_right, 2, "Should go to right leaf (index 2)");
    
    // Test boundary case (feature[0] = 5.0 should go left)
    let features_boundary = vec![5.0, 0.0, 0.0];  // feature[0] = 5.0 == 5.0
    let leaf_index_boundary = tree.predict_leaf_index(&features_boundary);
    println!("Features {:?} -> Leaf index: {}", features_boundary, leaf_index_boundary);
    assert_eq!(leaf_index_boundary, 1, "Boundary case should go to left leaf (index 1)");
    
    println!("SimpleTree::predict_leaf_index tests passed!");
    Ok(())
}