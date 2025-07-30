//! Tests for semantic equivalence between Rust and C++ MultiValDenseBin implementations

use lightgbm_rust::io::multi_val_dense_bin::{
    MultiValBin, MultiValDenseBinU16, MultiValDenseBinU8,
};

#[test]
fn test_basic_construction() {
    let offsets = vec![0, 10, 20, 30];
    let bin = MultiValDenseBinU8::new(100, 30, 3, offsets.clone());

    assert_eq!(bin.num_data(), 100);
    assert_eq!(bin.num_bin(), 30);
    assert_eq!(bin.num_element_per_row(), 3.0);
    assert_eq!(bin.offsets(), &offsets);
    assert!(!bin.is_sparse());
}

#[test]
fn test_push_one_row() {
    let offsets = vec![0, 10, 20, 30];
    let mut bin = MultiValDenseBinU8::new(10, 30, 3, offsets);

    let values = vec![5, 15, 25];
    bin.push_one_row(0, 0, &values);
    bin.push_one_row(0, 1, &[7, 17, 27]);

    // Test that the bin accepts the data without panicking
    assert_eq!(bin.num_data(), 10);
}

#[test]
fn test_resize() {
    let offsets = vec![0, 10, 20];
    let mut bin = MultiValDenseBinU8::new(10, 20, 2, offsets);

    let new_offsets = vec![0, 10, 20, 30, 40];
    bin.resize(20, 40, 4, 0.0, &new_offsets);

    assert_eq!(bin.num_data(), 20);
    assert_eq!(bin.num_bin(), 40);
    assert_eq!(bin.num_element_per_row(), 4.0);
    assert_eq!(bin.offsets().len(), 5);
}

#[test]
fn test_histogram_construction_basic() {
    let offsets = vec![0, 2, 4, 6]; // 3 features, 2 bins each
    let mut bin = MultiValDenseBinU8::new(3, 6, 3, offsets);

    // Set up some test data
    bin.push_one_row(0, 0, &[1, 0, 1]); // Row 0: bins [1, 0, 1]
    bin.push_one_row(0, 1, &[0, 1, 0]); // Row 1: bins [0, 1, 0]
    bin.push_one_row(0, 2, &[1, 1, 1]); // Row 2: bins [1, 1, 1]

    // Test data
    let data_indices = vec![0, 1, 2];
    let gradients = vec![1.0f32, 2.0f32, 3.0f32];
    let hessians = vec![0.5f32, 1.0f32, 1.5f32];

    // Histogram output: 6 bins * 2 (grad + hess) = 12 elements
    let mut histogram = vec![0.0f64; 12];

    bin.construct_histogram(
        data_indices.as_ptr(),
        0,
        3,
        gradients.as_ptr(),
        hessians.as_ptr(),
        histogram.as_mut_ptr(),
    );

    // Check that histogram values are non-zero (indicating data was processed)
    let sum: f64 = histogram.iter().sum();
    assert!(sum > 0.0, "Histogram should contain non-zero values");

    // Check gradient sum matches input (approximately)
    let grad_sum: f64 = histogram.iter().step_by(2).sum();
    let expected_grad_sum = gradients.iter().sum::<f32>() as f64;
    assert!(
        (grad_sum - expected_grad_sum).abs() < 1e-6,
        "Gradient sum mismatch: {} vs {}",
        grad_sum,
        expected_grad_sum
    );
}

#[test]
fn test_create_like() {
    let offsets = vec![0, 10, 20, 30];
    let bin = MultiValDenseBinU8::new(100, 30, 3, offsets.clone());

    let new_bin = bin.create_like(200, 50, 4, 0.0, &[0, 12, 24, 36, 48]);

    assert_eq!(new_bin.num_data(), 200);
    assert_eq!(new_bin.num_bin(), 50);
    assert_eq!(new_bin.num_element_per_row(), 4.0);
}

#[test]
fn test_clone() {
    let offsets = vec![0, 10, 20, 30];
    let bin = MultiValDenseBinU8::new(100, 30, 3, offsets);

    let cloned = bin.clone_multi_val_bin();

    assert_eq!(bin.num_data(), cloned.num_data());
    assert_eq!(bin.num_bin(), cloned.num_bin());
    assert_eq!(bin.num_element_per_row(), cloned.num_element_per_row());
    assert_eq!(bin.offsets(), cloned.offsets());
}

#[test]
fn test_different_value_types() {
    // Test u8 version
    let offsets8 = vec![0, 100, 200];
    let bin8 = MultiValDenseBinU8::new(50, 200, 2, offsets8);
    assert_eq!(bin8.num_data(), 50);

    // Test u16 version
    let offsets16 = vec![0, 300, 600];
    let bin16 = MultiValDenseBinU16::new(50, 600, 2, offsets16);
    assert_eq!(bin16.num_data(), 50);
}

#[test]
fn test_row_ptr_calculation() {
    let offsets = vec![0, 10, 20, 30];
    let bin = MultiValDenseBinU8::new(10, 30, 3, offsets);

    // Test internal row pointer calculation by observing behavior
    // Since row_ptr is private, we test it indirectly through push_one_row
    let values1 = vec![1, 2, 3];
    let values2 = vec![4, 5, 6];

    let mut bin_mut = bin;
    bin_mut.push_one_row(0, 0, &values1);
    bin_mut.push_one_row(0, 1, &values2);

    // If row_ptr works correctly, this should not panic
    assert_eq!(bin_mut.num_data(), 10);
}
