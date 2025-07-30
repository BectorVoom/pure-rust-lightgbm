//! Semantic equivalence tests for Rust SparseBin implementation
//!
//! This test suite verifies that the Rust SparseBin implementation produces
//! identical results to the expected C++ behavior.

use lightgbm_rust::core::types::MissingType;
use lightgbm_rust::io::sparse_bin::{SparseBin, SparseBinU16, SparseBinU8};

#[test]
fn test_sparse_bin_basic_functionality() {
    // Test basic sparse bin creation and operations
    let mut bin: SparseBinU8 = SparseBin::new(100);

    // Test initial state
    assert_eq!(bin.num_data(), 100);

    // Test pushing values
    bin.push(0, 10, 3);
    bin.push(0, 25, 7);
    bin.push(0, 50, 2);

    // Finish loading to process the sparse data
    bin.finish_load();

    // Should have 3 non-zero values
    assert_eq!(bin.num_vals(), 3);

    // Test that deltas and values are properly stored
    assert_eq!(bin.vals().len(), 3);
    assert_eq!(bin.deltas().len(), 4); // 3 values + terminator

    // Values should be stored in order
    assert_eq!(bin.vals()[0], 3);
    assert_eq!(bin.vals()[1], 7);
    assert_eq!(bin.vals()[2], 2);
}

#[test]
fn test_sparse_bin_iterator() {
    let mut bin: SparseBinU8 = SparseBin::new(100);

    // Push some test data
    bin.push(0, 5, 1);
    bin.push(0, 15, 2);
    bin.push(0, 25, 3);
    bin.finish_load();

    // Test iterator functionality
    let mut iterator = bin.get_iterator(1, 10, 0);

    // Test getting values at specific indices
    assert_eq!(iterator.inner_raw_get(5), 1);
    assert_eq!(iterator.inner_raw_get(15), 2);
    assert_eq!(iterator.inner_raw_get(25), 3);
    assert_eq!(iterator.inner_raw_get(10), 0); // Should return 0 for empty position
}

#[test]
fn test_histogram_construction() {
    let mut bin: SparseBinU8 = SparseBin::new(100);

    // Create sparse data: indices [10, 20, 30] with values [1, 2, 1]
    bin.push(0, 10, 1);
    bin.push(0, 20, 2);
    bin.push(0, 30, 1);
    bin.finish_load();

    // Test data indices and gradient values
    let data_indices = vec![10, 20, 30];
    let gradients = vec![0.5, 1.5, 0.8];
    let hessians = vec![0.2, 0.3, 0.1];

    // Create histogram output buffer (2 values per bin: gradient, hessian)
    let mut histogram = vec![0.0; 6]; // 3 bins * 2 values each

    bin.construct_histogram(&data_indices, 0, 3, &gradients, &hessians, &mut histogram);

    // Verify histogram construction
    // Bin 1: gradients[0] + gradients[2] = 0.5 + 0.8 = 1.3
    // Bin 1: hessians[0] + hessians[2] = 0.2 + 0.1 = 0.3
    // Bin 2: gradients[1] = 1.5, hessians[1] = 0.3

    assert!((histogram[2] - 1.3).abs() < 1e-6); // Bin 1 gradient
    assert!((histogram[3] - 0.3).abs() < 1e-6); // Bin 1 hessian
    assert!((histogram[4] - 1.5).abs() < 1e-6); // Bin 2 gradient
    assert!((histogram[5] - 0.3).abs() < 1e-6); // Bin 2 hessian
}

#[test]
fn test_split_functionality() {
    let mut bin: SparseBinU8 = SparseBin::new(100);

    // Create test data with different bin values
    bin.push(0, 10, 1); // Bin 1 - should go left (<=2)
    bin.push(0, 20, 3); // Bin 3 - should go right (>2)
    bin.push(0, 30, 2); // Bin 2 - should go left (<=2)
    bin.push(0, 40, 4); // Bin 4 - should go right (>2)
    bin.finish_load();

    let data_indices = vec![10, 20, 30, 40];
    let mut lte_indices = Vec::new();
    let mut gt_indices = Vec::new();

    // Split with threshold 2 (bins <= 2 go left, bins > 2 go right)
    let lte_count = bin.split(
        1,
        5,
        0,
        0,
        MissingType::None,
        false,
        2,
        &data_indices,
        &mut lte_indices,
        &mut gt_indices,
    );

    // Should have 2 indices going left and 2 going right
    assert_eq!(lte_count, 2);
    assert_eq!(lte_indices.len(), 2);
    assert_eq!(gt_indices.len(), 2);

    // Check that indices are correctly assigned
    assert!(lte_indices.contains(&10)); // Bin 1
    assert!(lte_indices.contains(&30)); // Bin 2
    assert!(gt_indices.contains(&20)); // Bin 3
    assert!(gt_indices.contains(&40)); // Bin 4
}

#[test]
fn test_categorical_split() {
    let mut bin: SparseBinU8 = SparseBin::new(100);

    // Create categorical data
    bin.push(0, 10, 1); // Category 1
    bin.push(0, 20, 3); // Category 3
    bin.push(0, 30, 2); // Category 2
    bin.push(0, 40, 4); // Category 4
    bin.finish_load();

    let data_indices = vec![10, 20, 30, 40];
    let mut lte_indices = Vec::new();
    let mut gt_indices = Vec::new();

    // Create threshold bitset where categories 1 and 3 go left
    // Bitset: bit 1 and bit 3 are set
    let threshold_bitset = vec![0b00000000_00000000_00000000_00001010u32]; // bits 1 and 3 set

    let lte_count = bin.split_categorical(
        1,
        5,
        0,
        &threshold_bitset,
        &data_indices,
        &mut lte_indices,
        &mut gt_indices,
    );

    // Should have 2 indices going left (categories 1 and 3)
    assert_eq!(lte_count, 2);
    assert_eq!(lte_indices.len(), 2);
    assert_eq!(gt_indices.len(), 2);

    // Check correct assignment
    assert!(lte_indices.contains(&10)); // Category 1
    assert!(lte_indices.contains(&20)); // Category 3
    assert!(gt_indices.contains(&30)); // Category 2
    assert!(gt_indices.contains(&40)); // Category 4
}

#[test]
fn test_fast_index_functionality() {
    let mut bin: SparseBinU8 = SparseBin::new(1000);

    // Create sparse data across a wide range
    for i in (0..1000).step_by(50) {
        bin.push(0, i, ((i / 50) % 5 + 1) as u32);
    }
    bin.finish_load();

    // Test that fast index allows efficient lookup
    let mut i_delta = 0;
    let mut cur_pos = 0;

    // Initialize at position 500
    bin.init_index(500, &mut i_delta, &mut cur_pos);

    // Should be able to find nearby positions efficiently
    assert!(i_delta >= 0);
    assert!(cur_pos <= 500);
}

#[test]
fn test_different_value_types() {
    // Test with u16 values
    let mut bin16: SparseBinU16 = SparseBin::new(100);

    bin16.push(0, 10, 300); // Value > 255, needs u16
    bin16.push(0, 20, 500);
    bin16.finish_load();

    assert_eq!(bin16.vals()[0], 300);
    assert_eq!(bin16.vals()[1], 500);
}

#[test]
fn test_missing_value_handling() {
    let mut bin: SparseBinU8 = SparseBin::new(100);

    // Create data with explicit missing value handling
    bin.push(0, 10, 1);
    bin.push(0, 30, 2);
    // Index 20 is "missing" (no value pushed)
    bin.finish_load();

    let data_indices = vec![10, 20, 30];
    let mut lte_indices = Vec::new();
    let mut gt_indices = Vec::new();

    // Test split with missing value handling (default_left = true)
    let lte_count = bin.split(
        1,
        3,
        0,
        0,
        MissingType::Zero,
        true, // default_left
        1,
        &data_indices,
        &mut lte_indices,
        &mut gt_indices,
    );

    // Missing value (index 20) should go left due to default_left=true
    assert!(lte_indices.contains(&20));
}

#[cfg(test)]
mod bench_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_performance_large_sparse_data() {
        let mut bin: SparseBinU16 = SparseBin::new(100000);

        // Create sparse data (only 1% density)
        for i in (0..100000).step_by(100) {
            bin.push(0, i, (i / 100 % 1000) as u32);
        }

        let start = Instant::now();
        bin.finish_load();
        let load_time = start.elapsed();

        println!(
            "Load time for 1000 sparse entries in 100K dataset: {:?}",
            load_time
        );

        // Test iterator performance
        let mut iterator = bin.get_iterator(1, 1000, 0);
        let start = Instant::now();

        for i in (0..100000).step_by(1000) {
            iterator.inner_raw_get(i);
        }

        let lookup_time = start.elapsed();
        println!("Lookup time for 100 queries: {:?}", lookup_time);

        // Performance should be reasonable
        assert!(load_time.as_millis() < 100); // Should load in < 100ms
        assert!(lookup_time.as_micros() < 1000); // Should lookup in < 1ms
    }
}
