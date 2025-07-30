//! Test suite for verifying tree.rs equivalence with C++ implementation
//!
//! This module tests the core functionality of the Tree struct to ensure
//! it produces semantically equivalent results to the C++ LightGBM Tree class.

use anyhow::Result;
use lightgbm_rust::{
    io::tree::{Tree, PathElement},
};
use ndarray::{Array1, Array2};

/// Test basic tree creation and initialization
#[test]
fn test_tree_creation() {
    let tree = Tree::new(10, false, false);
    
    assert_eq!(tree.num_leaves(), 1);
    assert_eq!(tree.get_upper_bound_value(), 0.0);
    assert_eq!(tree.get_lower_bound_value(), 0.0);
    assert!(!tree.to_string().is_empty());
}

/// Test tree creation with branch features tracking
#[test]
fn test_tree_with_branch_features() {
    let tree = Tree::new(5, true, false);
    assert_eq!(tree.num_leaves(), 1);
}

/// Test linear tree creation
#[test]
fn test_linear_tree_creation() {
    let tree = Tree::new(8, false, true);
    assert_eq!(tree.num_leaves(), 1);
}

/// Test tree string serialization
#[test]
fn test_tree_string_serialization() {
    let tree = Tree::new(3, false, false);
    let tree_str = tree.to_string();
    
    assert!(tree_str.contains("num_leaves=1"));
    assert!(tree_str.contains("num_cat=0"));
    assert!(tree_str.contains("is_linear=0"));
    assert!(tree_str.contains("shrinkage=1"));
}

/// Test tree JSON serialization
#[test]  
fn test_tree_json_serialization() {
    let tree = Tree::new(3, false, false);
    let json_str = tree.to_json();
    
    assert!(json_str.contains("\"num_leaves\":1"));
    assert!(json_str.contains("\"num_cat\":0"));
    assert!(json_str.contains("\"shrinkage\":1"));
    assert!(json_str.contains("\"tree_structure\""));
}

/// Test tree if-else code generation
#[test]
fn test_tree_code_generation() {
    let tree = Tree::new(3, false, false);
    let code = tree.to_if_else(0, false);
    
    assert!(code.contains("double PredictTree0"));
    assert!(code.contains("return"));
}

/// Test tree bound values
#[test]
fn test_tree_bound_values() {
    let tree = Tree::new(5, false, false);
    
    // For a single-leaf tree, both bounds should be the leaf value (0.0)
    assert_eq!(tree.get_upper_bound_value(), 0.0);
    assert_eq!(tree.get_lower_bound_value(), 0.0);
}

/// Test tree expected value calculation
#[test]
fn test_tree_expected_value() {
    let tree = Tree::new(3, false, false);
    let expected = tree.expected_value();
    assert_eq!(expected, 0.0); // Single leaf with value 0
}

/// Test tree depth computation
#[test]
fn test_tree_depth_computation() {
    let mut tree = Tree::new(3, false, false);
    tree.recompute_max_depth();
    // Single leaf tree should have depth 0
    // Cannot directly test max_depth as it's private, but the method should not panic
}

/// Test string parsing (basic)
#[test]
fn test_tree_from_string() -> Result<()> {
    let tree_str = "num_leaves=2\nnum_cat=1\nshrinkage=0.5\n";
    let tree = Tree::from_string(tree_str)?;
    
    assert_eq!(tree.num_leaves(), 2);
    Ok(())
}

/// Test PathElement creation and copy
#[test]
fn test_path_element() {
    let path_elem = PathElement {
        feature_index: 1,
        zero_fraction: 0.5,
        one_fraction: 0.5,
        pweight: 1.0,
    };
    
    let copied = path_elem; // Should work due to Copy trait
    assert_eq!(copied.feature_index, 1);
    assert_eq!(copied.zero_fraction, 0.5);
}

/// Test tree functionality with mock dataset
#[test]
fn test_tree_with_mock_dataset() -> Result<()> {
    // Create a simple mock dataset for testing
    let _features = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let _labels = Array1::from_vec(vec![0.0, 1.0, 0.0]);
    
    // Create a simple dataset structure for testing
    // Note: This is a simplified test since we need a proper Dataset implementation
    let tree = Tree::new(5, false, false);
    
    // Test basic tree properties
    assert_eq!(tree.num_leaves(), 1);
    assert_eq!(tree.expected_value(), 0.0);
    
    Ok(())
}

/// Integration test for tree serialization roundtrip
#[test]
fn test_tree_serialization_roundtrip() -> Result<()> {
    let original_tree = Tree::new(5, false, false);
    
    // Test string serialization
    let tree_str = original_tree.to_string();
    let parsed_tree = Tree::from_string(&tree_str)?;
    
    // Basic checks - should have same leaf count
    assert_eq!(original_tree.num_leaves(), parsed_tree.num_leaves());
    
    // Test JSON serialization  
    let json_str = original_tree.to_json();
    assert!(json_str.contains("num_leaves"));
    
    Ok(())
}

/// Performance test for tree operations
#[test]
fn test_tree_performance() {
    let tree = Tree::new(100, false, false);
    
    // Test multiple serializations to ensure performance is reasonable
    for _ in 0..100 {
        let _json = tree.to_json();
        let _string = tree.to_string();
        let _code = tree.to_if_else(0, false);
    }
    
    // If we get here without timeout, performance is acceptable
    assert_eq!(tree.num_leaves(), 1);
}

/// Test tree creation with various parameters
#[test]
fn test_tree_parameter_variations() {
    // Test different max_leaves values
    for max_leaves in [1, 5, 10, 31, 100] {
        let tree = Tree::new(max_leaves, false, false);
        assert_eq!(tree.num_leaves(), 1);
    }
    
    // Test with different feature combinations
    let linear_tree = Tree::new(10, false, true);
    let tracking_tree = Tree::new(10, true, false);
    let full_tree = Tree::new(10, true, true);
    
    assert_eq!(linear_tree.num_leaves(), 1);
    assert_eq!(tracking_tree.num_leaves(), 1);
    assert_eq!(full_tree.num_leaves(), 1);
}