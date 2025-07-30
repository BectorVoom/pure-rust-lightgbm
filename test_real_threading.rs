#!/usr/bin/env cargo

// Test the actual threading implementation from the library
use lightgbm_rust::core::utils::threading::Threading;

fn main() {
    println!("=== Testing Real Library Threading Implementation ===\n");

    // Test SIZE_ALIGNED with actual library code
    let mut nblock = 0i32;
    let mut block_size = 0usize;
    
    // Test case: cnt=1000, threads=8, min_cnt=10
    Threading::block_info(8, 1000usize, 10usize, &mut nblock, &mut block_size);
    
    println!("BlockInfo(threads=8, cnt=1000, min_cnt=10):");
    println!("  -> nblock={}, block_size={}", nblock, block_size);
    
    // Manual calculation verification
    let cnt = 1000usize;
    let threads = 8i32;
    let min_cnt = 10usize;
    
    let expected_nblock = std::cmp::min(threads, ((cnt + min_cnt - 1) / min_cnt) as i32);
    let raw_size = (cnt + expected_nblock as usize - 1) / expected_nblock as usize;
    let expected_aligned_size = ((raw_size + 32 - 1) / 32) * 32;
    
    println!("Manual calculation:");
    println!("  nblock = min({}, {}) = {}", threads, ((cnt + min_cnt - 1) / min_cnt), expected_nblock);
    println!("  raw_size = ({} + {} - 1) / {} = {}", cnt, expected_nblock, expected_nblock, raw_size);
    println!("  aligned_size = (({} + 31) / 32) * 32 = {}", raw_size, expected_aligned_size);
    
    if nblock == expected_nblock {
        println!("  ✅ nblock matches expected value");
    } else {
        println!("  ❌ nblock mismatch: got {} expected {}", nblock, expected_nblock);
    }
    
    if block_size == expected_aligned_size {
        println!("  ✅ block_size is correctly 32-byte aligned");
    } else {
        println!("  ❌ block_size mismatch: got {} expected {} (aligned)", block_size, expected_aligned_size);
        println!("    -> block_size % 32 = {}", block_size % 32);
    }
    
    // Test BlockInfoForceSize for comparison
    Threading::block_info_force_size(8, 1000usize, 10usize, &mut nblock, &mut block_size);
    let expected_force_size = ((raw_size + min_cnt - 1) / min_cnt) * min_cnt;
    
    println!("\nBlockInfoForceSize(threads=8, cnt=1000, min_cnt=10):");
    println!("  -> nblock={}, block_size={}", nblock, block_size);
    println!("  expected_force_size = (({} + {} - 1) / {}) * {} = {}", 
        raw_size, min_cnt, min_cnt, min_cnt, expected_force_size);
    
    if block_size == expected_force_size {
        println!("  ✅ block_size is correctly min_cnt aligned");
    } else {
        println!("  ❌ block_size mismatch: got {} expected {}", block_size, expected_force_size);
    }
}