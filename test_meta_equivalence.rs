/*!
 * Test for verifying semantic equivalence between C++ meta.h and Rust meta.rs
 */

use lightgbm_rust::core::meta::*;
use lightgbm_rust::core::meta::simd_ops;
use lightgbm_rust::{SIZE_ALIGNED, PREFETCH_T0};

fn main() {
    println!("Testing meta.rs semantic equivalence with C++ meta.h");
    
    // Test type definitions
    test_type_definitions();
    
    // Test constants
    test_constants();
    
    // Test macros
    test_macros();
    
    // Test SIMD operations
    test_simd_operations();
    
    // Test aligned vector
    test_aligned_vector();
    
    println!("All meta equivalence tests passed!");
}

fn test_type_definitions() {
    println!("Testing type definitions...");
    
    // Verify type sizes match expected C++ equivalents
    assert_eq!(std::mem::size_of::<DataSizeT>(), 4);  // i32
    assert_eq!(std::mem::size_of::<CommSizeT>(), 4);  // i32
    
    // Check ScoreT size based on feature flag
    #[cfg(feature = "score_t_use_double")]
    {
        assert_eq!(std::mem::size_of::<ScoreT>(), 8);  // f64 when feature enabled
        println!("✓ ScoreT is f64 (8 bytes) with score_t_use_double feature");
    }
    #[cfg(not(feature = "score_t_use_double"))]
    {
        assert_eq!(std::mem::size_of::<ScoreT>(), 4);  // f32 default
        println!("✓ ScoreT is f32 (4 bytes) by default");
    }
    
    // Check LabelT size based on feature flag
    #[cfg(feature = "label_t_use_double")]
    {
        assert_eq!(std::mem::size_of::<LabelT>(), 8);  // f64 when feature enabled
        println!("✓ LabelT is f64 (8 bytes) with label_t_use_double feature");
    }
    #[cfg(not(feature = "label_t_use_double"))]
    {
        assert_eq!(std::mem::size_of::<LabelT>(), 4);  // f32 default
        println!("✓ LabelT is f32 (4 bytes) by default");
    }
    
    println!("✓ Type definitions match C++ equivalents");
}

fn test_constants() {
    println!("Testing constants...");
    
    // Test constant values match C++ implementation
    assert_eq!(NO_SPECIFIC, -1);
    assert_eq!(K_ALIGNED_SIZE, 32);
    assert_eq!(K_EPSILON, 1e-15);
    assert_eq!(K_ZERO_THRESHOLD, 1e-35);
    
    // Test infinity constants
    assert!(K_MIN_SCORE.is_infinite() && K_MIN_SCORE.is_sign_negative());
    assert!(K_MAX_SCORE.is_infinite() && K_MAX_SCORE.is_sign_positive());
    
    println!("✓ Constants match C++ implementation");
}

fn test_macros() {
    println!("Testing macros...");
    
    // Test SIZE_ALIGNED macro
    assert_eq!(SIZE_ALIGNED!(10), 32);
    assert_eq!(SIZE_ALIGNED!(32), 32);
    assert_eq!(SIZE_ALIGNED!(33), 64);
    assert_eq!(SIZE_ALIGNED!(1), 32);
    assert_eq!(SIZE_ALIGNED!(63), 64);
    assert_eq!(SIZE_ALIGNED!(64), 64);
    assert_eq!(SIZE_ALIGNED!(65), 96);
    
    // Test PREFETCH_T0 macro (should compile without error)
    let test_addr = [1u8, 2, 3, 4];
    PREFETCH_T0!(test_addr.as_ptr());
    
    println!("✓ Macros work correctly");
}

fn test_simd_operations() {
    println!("Testing SIMD operations...");
    
    // Test SIMD addition
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let mut result = vec![0.0; 8];
    
    simd_ops::simd_add_scores(&a, &b, &mut result);
    
    for &val in &result {
        assert_eq!(val, 9.0);
    }
    
    // Test SIMD dot product
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![4.0, 3.0, 2.0, 1.0];
    let dot_result = simd_ops::simd_dot_product(&a, &b);
    assert_eq!(dot_result, 20.0); // 1*4 + 2*3 + 3*2 + 4*1 = 20
    
    // Test SIMD sum
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let sum_result = simd_ops::simd_sum(&data);
    assert_eq!(sum_result, 10.0);
    
    // Test SIMD scalar multiplication
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let mut scalar_result = vec![0.0; 4];
    simd_ops::simd_scalar_mul(&data, 2.0, &mut scalar_result);
    assert_eq!(scalar_result, vec![2.0, 4.0, 6.0, 8.0]);
    
    println!("✓ SIMD operations work correctly");
}

fn test_aligned_vector() {
    println!("Testing aligned vector...");
    
    let mut vec = ScoreVector::new();
    assert!(vec.empty());
    assert_eq!(vec.size(), 0);
    
    vec.push(1.0);
    vec.push(2.0);
    vec.push(3.0);
    
    assert!(!vec.empty());
    assert_eq!(vec.size(), 3);
    assert_eq!(vec.data(), &[1.0, 2.0, 3.0]);
    
    vec.resize(5, 0.0);
    assert_eq!(vec.size(), 5);
    assert_eq!(vec.data(), &[1.0, 2.0, 3.0, 0.0, 0.0]);
    
    vec.clear();
    assert!(vec.empty());
    assert_eq!(vec.size(), 0);
    
    println!("✓ Aligned vector works correctly");
}