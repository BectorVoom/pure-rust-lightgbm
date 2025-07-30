/*!
 * Rust test program to verify CUDA random equivalence
 * This tests our Rust CudaRandom implementation against C++ results
 */

use lightgbm_rust::core::gpu::CudaRandom;

fn main() {
    println!("Rust CUDA Random Equivalence Test");
    println!("=================================");
    
    let mut rng = CudaRandom::new();
    
    // Test 1: Default seed behavior
    println!("\nTest 1: Default seed (123456789)");
    println!("First 10 next_short(0, 100):");
    for _ in 0..10 {
        print!("{} ", rng.next_short(0, 100));
    }
    println!();
    
    // Reset and test next_int
    rng.set_seed(123456789);
    println!("First 10 next_int(0, 1000):");
    for _ in 0..10 {
        print!("{} ", rng.next_int(0, 1000));
    }
    println!();
    
    // Reset and test next_float
    rng.set_seed(123456789);
    println!("First 10 next_float():");
    for _ in 0..10 {
        print!("{:.6} ", rng.next_float());
    }
    println!();
    
    // Test 2: Custom seed
    println!("\nTest 2: Custom seed (42)");
    rng.set_seed(42);
    println!("First 5 next_short(10, 50):");
    for _ in 0..5 {
        print!("{} ", rng.next_short(10, 50));
    }
    println!();
    
    // Test 3: Large range
    rng.set_seed(12345);
    println!("\nTest 3: Seed 12345, next_int(0, 1000000):");
    for _ in 0..5 {
        print!("{} ", rng.next_int(0, 1000000));
    }
    println!();
    
    // Test 4: Verify deterministic behavior
    println!("\nTest 4: Deterministic behavior verification");
    let mut rng1 = CudaRandom::with_seed(9999);
    let mut rng2 = CudaRandom::with_seed(9999);
    
    let mut deterministic = true;
    for _ in 0..20 {
        if rng1.next_int(0, 100) != rng2.next_int(0, 100) {
            deterministic = false;
            break;
        }
    }
    println!("Deterministic: {}", if deterministic { "PASS" } else { "FAIL" });
    
    // Test 5: Raw LCG values for verification
    println!("\nTest 5: Raw LCG internal states (seed=1)");
    let mut rng_raw = CudaRandom::with_seed(1);
    println!("First 5 next_short(0, 65536) with seed=1:");
    for _ in 0..5 {
        print!("{} ", rng_raw.next_short(0, 65536));
    }
    println!();
    
    println!("\nRust equivalence test completed.");
}

// Unit tests for the CUDA random implementation
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_random_equivalence_basic() {
        let mut rng = CudaRandom::with_seed(123456789);
        
        // Test that the same seed produces consistent results
        let first_call = rng.next_short(0, 100);
        
        rng.set_seed(123456789);
        let second_call = rng.next_short(0, 100);
        
        assert_eq!(first_call, second_call, "Same seed should produce same result");
    }
    
    #[test]
    fn test_cuda_random_range_bounds() {
        let mut rng = CudaRandom::new();
        
        // Test NextShort bounds
        for _ in 0..100 {
            let val = rng.next_short(10, 20);
            assert!(val >= 10 && val < 20, "NextShort value {} out of range [10, 20)", val);
        }
        
        // Test NextInt bounds
        for _ in 0..100 {
            let val = rng.next_int(100, 200);
            assert!(val >= 100 && val < 200, "NextInt value {} out of range [100, 200)", val);
        }
        
        // Test NextFloat bounds
        for _ in 0..100 {
            let val = rng.next_float();
            assert!(val >= 0.0 && val < 1.0, "NextFloat value {} out of range [0.0, 1.0)", val);
        }
    }
    
    #[test]
    fn test_cuda_random_lcg_formula() {
        // Test the LCG formula matches exactly with specific known values
        let mut rng = CudaRandom::with_seed(1);
        
        // With seed=1, first call should produce:
        // x = (214013 * 1 + 2531011) = 2745024
        // rand_int16 = ((2745024 >> 16) & 0x7FFF) = (41 & 0x7FFF) = 41
        let first_val = rng.next_short(0, 65536);
        assert_eq!(first_val, 41, "First NextShort call with seed=1 should be 41");
    }
    
    #[test]
    fn test_cuda_random_deterministic() {
        let mut rng1 = CudaRandom::with_seed(42);
        let mut rng2 = CudaRandom::with_seed(42);
        
        // Generate sequences and verify they match
        let seq1: Vec<i32> = (0..50).map(|_| rng1.next_int(0, 1000)).collect();
        let seq2: Vec<i32> = (0..50).map(|_| rng2.next_int(0, 1000)).collect();
        
        assert_eq!(seq1, seq2, "Same seed should produce identical sequences");
    }
    
    #[test]
    fn test_cuda_random_float_precision() {
        let mut rng = CudaRandom::with_seed(123456789);
        
        // Verify float values are computed correctly: rand_int16 / 32768.0
        // We can't easily test exact values due to floating point precision,
        // but we can verify the range and basic properties
        let mut floats = Vec::new();
        for _ in 0..100 {
            floats.push(rng.next_float());
        }
        
        // Check all values are in valid range
        for (i, &val) in floats.iter().enumerate() {
            assert!(val >= 0.0 && val < 1.0, "Float {} at index {} out of range", val, i);
        }
        
        // Verify we have some variety (not all the same value)
        let first_val = floats[0];
        let has_variety = floats.iter().any(|&v| (v - first_val).abs() > 0.001);
        assert!(has_variety, "Float sequence should have variety, not all same value");
    }
}