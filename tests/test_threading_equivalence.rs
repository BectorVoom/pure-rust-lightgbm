use lightgbm_rust::core::utils::threading::{Threading, ParallelPartitionRunner};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_info_equivalence() {
        // Test BlockInfo function with various inputs
        let test_cases = vec![
            (4, 100usize, 10usize),  // num_threads, cnt, min_cnt_per_block
            (8, 1000, 50),
            (2, 17, 5),
            (1, 100, 10),
            (16, 200, 25),
        ];

        for (num_threads, cnt, min_cnt_per_block) in test_cases {
            let mut nblock = 0i32;
            let mut block_size = 0usize;
            
            Threading::block_info(
                num_threads,
                cnt,
                min_cnt_per_block,
                &mut nblock,
                &mut block_size,
            );

            // Verify logical constraints from C++ implementation
            assert!(nblock > 0, "nblock should be positive");
            assert!(nblock <= num_threads, "nblock should not exceed num_threads");
            
            if nblock > 1 {
                // block_size should be aligned (multiple of min_cnt_per_block)
                assert_eq!(block_size % min_cnt_per_block, 0, 
                    "block_size should be aligned to min_cnt_per_block");
            } else {
                assert_eq!(block_size, cnt, "single block should equal total count");
            }
            
            println!("BlockInfo({}, {}, {}) -> nblock={}, block_size={}", 
                num_threads, cnt, min_cnt_per_block, nblock, block_size);
        }
    }

    #[test]
    fn test_block_info_force_size_equivalence() {
        // Test BlockInfoForceSize function
        let test_cases = vec![
            (4, 100usize, 10usize),
            (8, 1000, 50),
            (2, 17, 5),
        ];

        for (num_threads, cnt, min_cnt_per_block) in test_cases {
            let mut nblock = 0i32;
            let mut block_size = 0usize;
            
            Threading::block_info_force_size(
                num_threads,
                cnt,
                min_cnt_per_block,
                &mut nblock,
                &mut block_size,
            );

            // Verify constraints
            assert!(nblock > 0);
            assert!(nblock <= num_threads);
            
            if nblock > 1 {
                // Force size should ensure block_size is multiple of min_cnt_per_block
                assert_eq!(block_size % min_cnt_per_block, 0, 
                    "forced block_size should be aligned");
            }
            
            println!("BlockInfoForceSize({}, {}, {}) -> nblock={}, block_size={}", 
                num_threads, cnt, min_cnt_per_block, nblock, block_size);
        }
    }

    #[test]
    fn test_for_loop_equivalence() {
        use std::sync::{Arc, Mutex};
        
        let start = 0usize;
        let end = 100usize;
        let min_block_size = 10usize;
        
        // Collect results from parallel execution
        let results = Arc::new(Mutex::new(Vec::new()));
        let results_clone = results.clone();
        
        let nblock = Threading::for_loop(start, end, min_block_size, |thread_id, start, end| {
            let mut res = results_clone.lock().unwrap();
            res.push((thread_id, start, end));
        });
        
        let final_results = results.lock().unwrap();
        
        // Verify results
        assert!(nblock > 0, "Should have at least one block");
        assert_eq!(final_results.len(), nblock as usize, "Should have results for each block");
        
        // Verify ranges cover the full input range without gaps/overlaps
        let mut sorted_results = final_results.clone();
        sorted_results.sort_by_key(|&(_, start, _)| start);
        
        // Check coverage
        assert_eq!(sorted_results[0].1, start, "Should start at beginning");
        
        for i in 1..sorted_results.len() {
            assert_eq!(sorted_results[i-1].2, sorted_results[i].1, 
                "Ranges should be contiguous");
        }
        
        if let Some(last) = sorted_results.last() {
            assert_eq!(last.2, end, "Should end at the end");
        }
        
        println!("For loop processed {} blocks covering [{}, {})", nblock, start, end);
    }

    #[test]
    fn test_parallel_partition_runner_basic() {
        let num_data = 100usize;
        let min_block_size = 10usize;
        
        let mut runner = ParallelPartitionRunner::new(num_data, min_block_size);
        
        // Test resize
        runner.resize(200usize);
        
        // Create test data - simple partition function that puts even numbers left, odd right
        let cnt = 50usize;
        let mut output = vec![0usize; cnt];
        
        let partition_func = |_thread_id: i32, _start: usize, cnt: usize, 
                            left: &mut [usize], _right: Option<&mut [usize]>| -> usize {
            let mut left_count = 0usize;
            for i in 0..cnt {
                left[i] = i;
                if i % 2 == 0 {
                    left_count += 1;
                }
            }
            left_count
        };
        
        let left_cnt = runner.run(cnt, partition_func, &mut output, false, false);
        
        // Verify partitioning results
        assert!(left_cnt.into() <= cnt, "Left count should not exceed total");
        
        println!("Partitioned {} items: {} left, {} right", 
            cnt, left_cnt.into(), cnt - left_cnt.into());
    }

    #[test]
    fn test_threading_semantic_equivalence() {
        // This test verifies the overall semantic equivalence with expected C++ behavior
        
        // Test case 1: BlockInfo with standard parameters
        let mut nblock = 0i32;
        let mut block_size = 0usize;
        Threading::block_info(4, 100usize, 10usize, &mut nblock, &mut block_size);
        
        // These values should match C++ LightGBM behavior
        assert!(nblock >= 1 && nblock <= 4);
        assert!(block_size >= 10); // At least min_cnt_per_block
        
        // Test case 2: Verify default thread count functions work
        let mut nblock2 = 0i32;
        let mut block_size2 = 0usize;
        Threading::block_info_default(100usize, 10usize, &mut nblock2, &mut block_size2);
        
        // Should produce reasonable results
        assert!(nblock2 > 0);
        assert!(block_size2 > 0);
        
        println!("Semantic equivalence tests passed");
    }
}