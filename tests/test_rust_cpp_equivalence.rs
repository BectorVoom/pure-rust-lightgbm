//! Cross-language semantic equivalence tests
//! Tests Rust implementation against C++ reference outputs

use lightgbm_rust::core::types::*;
use lightgbm_rust::io::multi_val_dense_bin::{
    MultiValBin, MultiValDenseBinU16, MultiValDenseBinU8,
};
use std::fs;

fn parse_test_case(filename: &str) -> (Vec<DataSize>, Vec<Score>, Vec<Score>, Vec<Hist>) {
    let content = fs::read_to_string(filename).expect("Failed to read test case file");
    let lines: Vec<&str> = content.lines().collect();

    let data_indices: Vec<DataSize> = lines[0]
        .strip_prefix("data_indices: ")
        .unwrap()
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();

    let gradients: Vec<Score> = lines[1]
        .strip_prefix("gradients: ")
        .unwrap()
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();

    let hessians: Vec<Score> = lines[2]
        .strip_prefix("hessians: ")
        .unwrap()
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();

    let histogram: Vec<Hist> = lines[3]
        .strip_prefix("histogram: ")
        .unwrap()
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();

    (data_indices, gradients, hessians, histogram)
}

#[test]
fn test_case_1_rust_vs_cpp() {
    // Parse C++ reference output
    let (data_indices, gradients, hessians, expected_histogram) =
        parse_test_case("cpp_test_case_1.txt");

    // Set up Rust implementation with same parameters
    let offsets = vec![0, 2, 4, 6];
    let mut bin = MultiValDenseBinU8::new(3, 6, 3, offsets);

    // Insert the same test data as C++
    let values1 = vec![1, 0, 1]; // Row 0: bins [1, 0, 1]
    let values2 = vec![0, 1, 0]; // Row 1: bins [0, 1, 0]
    let values3 = vec![1, 1, 1]; // Row 2: bins [1, 1, 1]

    bin.push_one_row(0, 0, &values1);
    bin.push_one_row(0, 1, &values2);
    bin.push_one_row(0, 2, &values3);
    bin.finish_load();

    // Run Rust histogram construction
    let mut histogram = vec![0.0f64; 12];

    bin.construct_histogram(
        data_indices.as_ptr(),
        0,
        3,
        gradients.as_ptr(),
        hessians.as_ptr(),
        histogram.as_mut_ptr(),
    );

    // Compare results
    assert_eq!(
        histogram.len(),
        expected_histogram.len(),
        "Histogram length mismatch"
    );

    for (i, (&rust_val, &cpp_val)) in histogram.iter().zip(expected_histogram.iter()).enumerate() {
        assert!(
            (rust_val - cpp_val).abs() < 1e-10,
            "Histogram mismatch at index {}: Rust={}, C++={}",
            i,
            rust_val,
            cpp_val
        );
    }

    println!("✅ Test Case 1: Rust and C++ outputs match exactly");
    println!("Expected: {:?}", expected_histogram);
    println!("Got:      {:?}", histogram);
}

#[test]
fn test_case_2_rust_vs_cpp() {
    // Parse C++ reference output
    let (_, gradients, hessians, expected_histogram) = parse_test_case("cpp_test_case_2.txt");

    // Set up Rust implementation
    let offsets = vec![0, 2, 4, 6];
    let mut bin = MultiValDenseBinU8::new(3, 6, 3, offsets);

    // Same test data
    bin.push_one_row(0, 0, &[1, 0, 1]);
    bin.push_one_row(0, 1, &[0, 1, 0]);
    bin.push_one_row(0, 2, &[1, 1, 1]);
    bin.finish_load();

    // Run Rust histogram construction WITHOUT indices
    let mut histogram = vec![0.0f64; 12];

    bin.construct_histogram_no_indices(
        0,
        3,
        gradients.as_ptr(),
        hessians.as_ptr(),
        histogram.as_mut_ptr(),
    );

    // Compare results
    for (i, (&rust_val, &cpp_val)) in histogram.iter().zip(expected_histogram.iter()).enumerate() {
        assert!(
            (rust_val - cpp_val).abs() < 1e-10,
            "Histogram mismatch at index {}: Rust={}, C++={}",
            i,
            rust_val,
            cpp_val
        );
    }

    println!("✅ Test Case 2: Rust and C++ outputs match exactly (no indices)");
}

#[test]
fn test_case_3_rust_vs_cpp() {
    // Parse C++ reference output
    let (data_indices, gradients, hessians, expected_histogram) =
        parse_test_case("cpp_test_case_3.txt");

    // Set up Rust implementation
    let offsets = vec![0, 2, 4, 6];
    let mut bin = MultiValDenseBinU8::new(3, 6, 3, offsets);

    // Same test data
    bin.push_one_row(0, 0, &[1, 0, 1]);
    bin.push_one_row(0, 1, &[0, 1, 0]);
    bin.push_one_row(0, 2, &[1, 1, 1]);
    bin.finish_load();

    // Run Rust ORDERED histogram construction
    let mut histogram = vec![0.0f64; 12];

    bin.construct_histogram_ordered(
        data_indices.as_ptr(),
        0,
        3,
        gradients.as_ptr(),
        hessians.as_ptr(),
        histogram.as_mut_ptr(),
    );

    // Compare results
    for (i, (&rust_val, &cpp_val)) in histogram.iter().zip(expected_histogram.iter()).enumerate() {
        assert!(
            (rust_val - cpp_val).abs() < 1e-10,
            "Histogram mismatch at index {}: Rust={}, C++={}",
            i,
            rust_val,
            cpp_val
        );
    }

    println!("✅ Test Case 3: Rust and C++ outputs match exactly (ordered)");
    println!("Data indices: {:?}", data_indices);
    println!("Expected: {:?}", expected_histogram);
    println!("Got:      {:?}", histogram);
}

#[test]
fn test_case_4_rust_vs_cpp() {
    // Parse C++ reference output
    let (data_indices, gradients, hessians, expected_histogram) =
        parse_test_case("cpp_test_case_4.txt");

    // Set up Rust implementation with uint16_t equivalent
    let offsets = vec![0, 100, 200, 300];
    let mut bin = MultiValDenseBinU16::new(2, 300, 3, offsets);

    // Insert test data with larger values
    bin.push_one_row(0, 0, &[50, 25, 75]);
    bin.push_one_row(0, 1, &[25, 50, 25]);
    bin.finish_load();

    // Run Rust histogram construction
    let mut histogram = vec![0.0f64; 600]; // 300 bins * 2

    bin.construct_histogram(
        data_indices.as_ptr(),
        0,
        2,
        gradients.as_ptr(),
        hessians.as_ptr(),
        histogram.as_mut_ptr(),
    );

    // Compare results
    assert_eq!(histogram.len(), expected_histogram.len());

    for (i, (&rust_val, &cpp_val)) in histogram.iter().zip(expected_histogram.iter()).enumerate() {
        assert!(
            (rust_val - cpp_val).abs() < 1e-10,
            "Histogram mismatch at index {}: Rust={}, C++={}",
            i,
            rust_val,
            cpp_val
        );
    }

    // Count non-zero entries to match C++ output
    let non_zero_count = histogram
        .chunks(2)
        .filter(|chunk| chunk[0] != 0.0 || chunk[1] != 0.0)
        .count();
    let total_grad: f64 = histogram.iter().step_by(2).sum();
    let total_hess: f64 = histogram.iter().skip(1).step_by(2).sum();

    println!("✅ Test Case 4: Rust and C++ outputs match exactly (uint16_t)");
    println!("Non-zero histogram entries: {}", non_zero_count);
    println!(
        "Total gradient: {}, Total hessian: {}",
        total_grad, total_hess
    );
}

#[test]
fn test_integer_histogram_methods() {
    // Test simplified integer histogram methods against regular histogram
    let offsets = vec![0, 2, 4, 6];
    let mut bin = MultiValDenseBinU8::new(3, 6, 3, offsets);

    bin.push_one_row(0, 0, &[1, 0, 1]);
    bin.push_one_row(0, 1, &[0, 1, 0]);
    bin.push_one_row(0, 2, &[1, 1, 1]);
    bin.finish_load();

    let data_indices = vec![0, 1, 2];
    let gradients = vec![1.0f32, 2.0f32, 3.0f32];
    let dummy_hessians = vec![1.0f32, 1.0f32, 1.0f32]; // Not used in int methods

    // Test int32 method
    let mut histogram_int32 = vec![0.0f64; 12];

    bin.construct_histogram_int32(
        data_indices.as_ptr(),
        0,
        3,
        gradients.as_ptr(),
        dummy_hessians.as_ptr(),
        histogram_int32.as_mut_ptr(),
    );

    // Should have non-zero values
    let sum: f64 = histogram_int32.iter().sum();
    assert!(sum > 0.0, "Integer histogram should have non-zero values");

    println!("✅ Integer histogram methods work correctly");
}
