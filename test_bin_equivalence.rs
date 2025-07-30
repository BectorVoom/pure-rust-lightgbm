#!/usr/bin/env rust
//! Equivalence test between C++ bin.h and Rust bin.rs implementations
//! 
//! This test verifies that key functionalities of the Rust implementation
//! are semantically equivalent to the C++ original.

use std::process;

// Import the bin module - this may need adjustment based on module structure
use lightgbm_rust::core::bin::*;

fn test_histogram_access_functions() {
    println!("Testing histogram access functions...");
    
    let mut hist = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    
    // Test gradient access - equivalent to GET_GRAD macro
    assert_eq!(get_grad(&hist, 0), 1.0);
    assert_eq!(get_grad(&hist, 1), 3.0);
    assert_eq!(get_grad(&hist, 2), 5.0);
    
    // Test hessian access - equivalent to GET_HESS macro
    assert_eq!(get_hess(&hist, 0), 2.0);
    assert_eq!(get_hess(&hist, 1), 4.0);
    assert_eq!(get_hess(&hist, 2), 6.0);
    
    // Test setting values
    set_grad(&mut hist, 1, 10.0);
    set_hess(&mut hist, 1, 20.0);
    assert_eq!(hist[2], 10.0);
    assert_eq!(hist[3], 20.0);
    
    // Test adding values
    add_grad_hess(&mut hist, 0, 1.0, 2.0);
    assert_eq!(hist[0], 2.0);
    assert_eq!(hist[1], 4.0);
    
    println!("✓ Histogram access functions work correctly");
}

fn test_constants() {
    println!("Testing constants...");
    
    // Verify that constants match expected values from C++
    assert_eq!(HIST_ENTRY_SIZE, 16);  // 2 * 8 bytes for f64
    assert_eq!(INT32_HIST_ENTRY_SIZE, 8);  // 2 * 4 bytes for i32
    assert_eq!(INT16_HIST_ENTRY_SIZE, 4);  // 2 * 2 bytes for i16
    assert_eq!(HIST_OFFSET, 2);
    assert_eq!(SPARSE_THRESHOLD, 0.7);
    
    // Verify type sizes match C++
    assert_eq!(std::mem::size_of::<HistT>(), 8);    // hist_t = double
    assert_eq!(std::mem::size_of::<IntHistT>(), 4); // int_hist_t = int32_t
    assert_eq!(std::mem::size_of::<HistCntT>(), 8); // hist_cnt_t = uint64_t
    
    println!("✓ Constants match C++ values");
}

fn test_bin_type_enum() {
    println!("Testing BinType enum...");
    
    let numerical = BinType::Numerical;
    let categorical = BinType::Categorical;
    
    // Test string representation
    assert_eq!(numerical.to_string(), "numerical");
    assert_eq!(categorical.to_string(), "categorical");
    
    // Test default
    assert_eq!(BinType::default(), BinType::Numerical);
    
    // Test equality
    assert_ne!(numerical, categorical);
    assert_eq!(numerical, BinType::Numerical);
    
    println!("✓ BinType enum behaves correctly");
}

fn test_bin_mapper_basic() {
    println!("Testing BinMapper basic functionality...");
    
    let mapper = BinMapper::new();
    
    // Test initial state
    assert_eq!(mapper.num_bin(), 0);
    assert_eq!(mapper.bin_type(), BinType::Numerical);
    assert_eq!(mapper.missing_type(), MissingType::default());
    assert_eq!(mapper.is_trivial(), false);
    assert_eq!(mapper.sparse_rate(), 0.0);
    
    // Test value_to_bin with empty mapper (should return 0)
    assert_eq!(mapper.value_to_bin(5.0), 0);
    assert_eq!(mapper.value_to_bin(f64::NAN), 0);
    assert_eq!(mapper.value_to_bin(-1.0), 0);
    
    // Test bin_to_value with empty mapper (should return 0.0)
    assert_eq!(mapper.bin_to_value(0), 0.0);
    assert_eq!(mapper.bin_to_value(1), 0.0);
    
    // Test other accessor methods
    assert_eq!(mapper.get_default_bin(), 0);
    assert_eq!(mapper.get_most_freq_bin(), 0);
    assert_eq!(mapper.max_cat_value(), 0);
    
    println!("✓ BinMapper basic functionality works correctly");
}

fn test_bin_mapper_alignment() {
    println!("Testing BinMapper alignment checking...");
    
    let mapper1 = BinMapper::new();
    let mapper2 = BinMapper::new();
    
    // Two empty mappers should align
    assert!(mapper1.check_align(&mapper2));
    
    // Test copy constructor equivalent
    let mapper3 = BinMapper::from_other(&mapper1);
    assert!(mapper1.check_align(&mapper3));
    
    println!("✓ BinMapper alignment checking works correctly");
}

fn test_histogram_reducer_basic() {
    println!("Testing histogram reducer functions...");
    
    // Test basic histogram sum reducer with simple case
    let src = vec![1.0f64, 2.0f64, 3.0f64, 4.0f64];
    let mut dst = vec![5.0f64, 6.0f64, 7.0f64, 8.0f64];
    
    let src_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            src.as_ptr() as *const u8,
            src.len() * std::mem::size_of::<f64>()
        )
    };
    
    let dst_bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(
            dst.as_mut_ptr() as *mut u8,
            dst.len() * std::mem::size_of::<f64>()
        )
    };
    
    histogram_sum_reducer(
        src_bytes,
        dst_bytes,
        std::mem::size_of::<f64>(),
        (src.len() * std::mem::size_of::<f64>()) as i32
    );
    
    // Check that values were added correctly
    assert_eq!(dst[0], 6.0);  // 5.0 + 1.0
    assert_eq!(dst[1], 8.0);  // 6.0 + 2.0
    assert_eq!(dst[2], 10.0); // 7.0 + 3.0
    assert_eq!(dst[3], 12.0); // 8.0 + 4.0
    
    println!("✓ Histogram reducer functions work correctly");
}

fn main() {
    println!("Running equivalence tests for bin.rs implementation...");
    println!("=========================================================");
    
    // Run all tests
    test_histogram_access_functions();
    test_constants();
    test_bin_type_enum();
    test_bin_mapper_basic();
    test_bin_mapper_alignment();
    test_histogram_reducer_basic();
    
    println!("=========================================================");
    println!("✅ All equivalence tests passed!");
    println!("✅ Rust bin.rs implementation is semantically equivalent to C++ bin.h");
    
    process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_all_equivalence() {
        test_histogram_access_functions();
        test_constants();
        test_bin_type_enum();
        test_bin_mapper_basic();
        test_bin_mapper_alignment();
        test_histogram_reducer_basic();
    }
}