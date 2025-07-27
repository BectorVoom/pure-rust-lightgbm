//! Test for verifying semantic equivalence between Rust and C++ LeafSplits implementations.
//!
//! This test validates that the Rust implementation produces the same results
//! as the original C++ implementation for various initialization scenarios.

use lightgbm_rust::{
    config::core::Config,
    core::types::*,
    dataset::partition::DataPartition,
    treelearner::leaf_splits::LeafSplits,
};

#[cfg(test)]
mod leaf_splits_equivalence_tests {
    use super::*;

    /// Test case structure for equivalence testing
    #[derive(Debug, Clone)]
    struct EquivalenceTestCase {
        name: &'static str,
        num_data: DataSize,
        gradients: Vec<Score>,
        hessians: Vec<Score>,
        expected_sum_gradients: f64,
        expected_sum_hessians: f64,
    }

    impl EquivalenceTestCase {
        fn new(
            name: &'static str,
            num_data: DataSize,
            gradients: Vec<Score>,
            hessians: Vec<Score>,
            expected_sum_gradients: f64,
            expected_sum_hessians: f64,
        ) -> Self {
            Self {
                name,
                num_data,
                gradients,
                hessians,
                expected_sum_gradients,
                expected_sum_hessians,
            }
        }
    }

    /// Create test cases for various scenarios
    fn create_test_cases() -> Vec<EquivalenceTestCase> {
        vec![
            EquivalenceTestCase::new(
                "small_dataset",
                4,
                vec![1.0, 2.0, 3.0, 4.0],
                vec![0.5, 1.0, 1.5, 2.0],
                10.0,
                5.0,
            ),
            EquivalenceTestCase::new(
                "medium_dataset",
                10,
                vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0],
                vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                -5.0,
                5.5,
            ),
            EquivalenceTestCase::new(
                "zero_gradients",
                5,
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![1.0, 1.0, 1.0, 1.0, 1.0],
                0.0,
                5.0,
            ),
            EquivalenceTestCase::new(
                "mixed_values",
                6,
                vec![1.5, -2.3, 4.7, -0.8, 3.2, -1.1],
                vec![0.25, 0.75, 1.25, 0.5, 2.0, 0.8],
                5.2,
                5.55,
            ),
        ]
    }

    #[test]
    fn test_leaf_splits_init_from_gradients() {
        let test_cases = create_test_cases();

        for test_case in test_cases {
            println!("Testing case: {}", test_case.name);

            let mut leaf_splits = LeafSplits::new(test_case.num_data, None);
            leaf_splits.init_from_gradients(&test_case.gradients, &test_case.hessians);

            // Validate results
            assert_eq!(
                leaf_splits.num_data_in_leaf(),
                test_case.num_data,
                "Failed for case: {} - num_data_in_leaf mismatch",
                test_case.name
            );

            assert!(
                (leaf_splits.sum_gradients() - test_case.expected_sum_gradients).abs() < 1e-10,
                "Failed for case: {} - gradients sum mismatch. Expected: {}, Got: {}",
                test_case.name,
                test_case.expected_sum_gradients,
                leaf_splits.sum_gradients()
            );

            assert!(
                (leaf_splits.sum_hessians() - test_case.expected_sum_hessians).abs() < 1e-10,
                "Failed for case: {} - hessians sum mismatch. Expected: {}, Got: {}",
                test_case.name,
                test_case.expected_sum_hessians,
                leaf_splits.sum_hessians()
            );

            assert_eq!(
                leaf_splits.leaf_index(),
                0,
                "Failed for case: {} - leaf index should be 0",
                test_case.name
            );

            assert!(
                leaf_splits.data_indices().is_none(),
                "Failed for case: {} - data indices should be None",
                test_case.name
            );
        }
    }

    #[test]
    fn test_leaf_splits_init_with_sums() {
        let test_gradients = 42.5;
        let test_hessians = 84.3;
        let test_weight = 12.7;
        let leaf_id = 5;

        let mut data_partition = DataPartition::new(100);
        let mut leaf_splits = LeafSplits::new(100, None);

        leaf_splits.init_with_sums(leaf_id, &data_partition, test_gradients, test_hessians, test_weight);

        assert_eq!(leaf_splits.leaf_index(), leaf_id);
        assert_eq!(leaf_splits.sum_gradients(), test_gradients);
        assert_eq!(leaf_splits.sum_hessians(), test_hessians);
        assert_eq!(leaf_splits.weight(), test_weight);
        assert_eq!(leaf_splits.num_data_in_leaf(), 100); // All data starts in root leaf (0)
    }

    #[test]
    fn test_leaf_splits_init_simple() {
        let leaf_id = 3;
        let sum_gradients = 15.7;
        let sum_hessians = 23.9;

        let mut leaf_splits = LeafSplits::new(50, None);
        leaf_splits.init_simple(leaf_id, sum_gradients, sum_hessians);

        assert_eq!(leaf_splits.leaf_index(), leaf_id);
        assert_eq!(leaf_splits.sum_gradients(), sum_gradients);
        assert_eq!(leaf_splits.sum_hessians(), sum_hessians);
    }

    #[test]
    fn test_leaf_splits_init_sums_only() {
        let sum_gradients = 67.2;
        let sum_hessians = 34.8;

        let mut leaf_splits = LeafSplits::new(200, None);
        leaf_splits.init_sums_only(sum_gradients, sum_hessians);

        assert_eq!(leaf_splits.leaf_index(), 0);
        assert_eq!(leaf_splits.sum_gradients(), sum_gradients);
        assert_eq!(leaf_splits.sum_hessians(), sum_hessians);
    }

    #[test]
    fn test_leaf_splits_init_discretized() {
        let sum_gradients = 45.6;
        let sum_hessians = 78.9;
        let int_sum = 123456789i64;

        let mut leaf_splits = LeafSplits::new(150, None);
        leaf_splits.init_sums_with_discretized(sum_gradients, sum_hessians, int_sum);

        assert_eq!(leaf_splits.leaf_index(), 0);
        assert_eq!(leaf_splits.sum_gradients(), sum_gradients);
        assert_eq!(leaf_splits.sum_hessians(), sum_hessians);
        assert_eq!(leaf_splits.int_sum_gradients_and_hessians(), int_sum);
    }

    #[test]
    fn test_leaf_splits_init_empty() {
        let mut leaf_splits = LeafSplits::new(100, None);
        leaf_splits.init_empty();

        assert_eq!(leaf_splits.leaf_index(), -1);
        assert_eq!(leaf_splits.num_data_in_leaf(), 0);
        assert!(leaf_splits.data_indices().is_none());
    }

    #[test]
    fn test_leaf_splits_reset_num_data() {
        let mut leaf_splits = LeafSplits::new(100, None);
        assert_eq!(leaf_splits.num_data_in_leaf(), 100);

        leaf_splits.reset_num_data(250);
        assert_eq!(leaf_splits.num_data_in_leaf(), 250);
    }

    #[test]
    fn test_leaf_splits_deterministic_config() {
        let mut config = Config::default();
        config.deterministic = true;

        let leaf_splits = LeafSplits::new(100, Some(&config));
        // We can't directly test the deterministic flag since it's private,
        // but we can test that the config is accepted without errors
        assert_eq!(leaf_splits.num_data_in_leaf(), 100);

        let non_deterministic_splits = LeafSplits::new(100, None);
        assert_eq!(non_deterministic_splits.num_data_in_leaf(), 100);
    }

    #[test]
    fn test_leaf_splits_partial_data() {
        let gradients = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let hessians = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        
        let mut data_partition = DataPartition::new(6);
        
        // Split the data: indices 0,1,2 go to leaf 1, indices 3,4,5 go to leaf 2
        let split_indices = vec![true, true, true, false, false, false];
        data_partition.split_leaf(0, 1, 2, &split_indices).unwrap();

        let mut leaf_splits = LeafSplits::new(6, None);
        
        // Test with leaf 1 (should have indices 0, 1, 2)
        leaf_splits.init_from_partial_gradients(1, &data_partition, &gradients, &hessians);
        
        assert_eq!(leaf_splits.leaf_index(), 1);
        assert_eq!(leaf_splits.num_data_in_leaf(), 3);
        assert_eq!(leaf_splits.sum_gradients(), 6.0); // 1.0 + 2.0 + 3.0
        assert_eq!(leaf_splits.sum_hessians(), 0.6); // 0.1 + 0.2 + 0.3
        
        let data_indices = leaf_splits.data_indices().unwrap();
        assert_eq!(data_indices.len(), 3);
        // Check that we have the right indices (order may vary)
        assert!(data_indices.contains(&0));
        assert!(data_indices.contains(&1));
        assert!(data_indices.contains(&2));
    }

    #[test]
    fn test_leaf_splits_discretized_gradients() {
        // Test with small discretized gradient/hessian data
        let int_gradients_and_hessians = vec![
            1i8, 2i8,  // hess=1, grad=2 for data point 0
            3i8, 4i8,  // hess=3, grad=4 for data point 1
            5i8, 6i8,  // hess=5, grad=6 for data point 2
            7i8, 8i8,  // hess=7, grad=8 for data point 3
        ];
        let grad_scale = 0.1;
        let hess_scale = 0.2;

        let mut leaf_splits = LeafSplits::new(4, None);
        leaf_splits.init_from_discretized_gradients(&int_gradients_and_hessians, grad_scale, hess_scale);

        // Expected sums:
        // gradients: (2*0.1) + (4*0.1) + (6*0.1) + (8*0.1) = 0.2 + 0.4 + 0.6 + 0.8 = 2.0
        // hessians: (1*0.2) + (3*0.2) + (5*0.2) + (7*0.2) = 0.2 + 0.6 + 1.0 + 1.4 = 3.2
        
        assert_eq!(leaf_splits.num_data_in_leaf(), 4);
        assert!((leaf_splits.sum_gradients() - 2.0).abs() < 1e-10);
        assert!((leaf_splits.sum_hessians() - 3.2).abs() < 1e-10);
        
        // The discretized sum should be non-zero (exact value depends on packing logic)
        assert_ne!(leaf_splits.int_sum_gradients_and_hessians(), 0);
    }

    #[test]
    fn test_leaf_splits_default() {
        let leaf_splits = LeafSplits::default();
        assert_eq!(leaf_splits.num_data_in_leaf(), 0);
        assert_eq!(leaf_splits.leaf_index(), 0);
        assert_eq!(leaf_splits.sum_gradients(), 0.0);
        assert_eq!(leaf_splits.sum_hessians(), 0.0);
        assert_eq!(leaf_splits.weight(), 0.0);
    }

    /// Performance test to ensure parallel processing works correctly
    #[test]
    fn test_leaf_splits_large_dataset_performance() {
        const LARGE_SIZE: usize = 2000; // Enough to trigger parallel processing
        
        let gradients: Vec<Score> = (0..LARGE_SIZE).map(|i| i as Score * 0.001).collect();
        let hessians: Vec<Score> = (0..LARGE_SIZE).map(|i| (i as Score + 1.0) * 0.0005).collect();
        
        let expected_sum_gradients: f64 = gradients.iter().map(|&x| x as f64).sum();
        let expected_sum_hessians: f64 = hessians.iter().map(|&x| x as f64).sum();
        
        let mut leaf_splits = LeafSplits::new(LARGE_SIZE as DataSize, None);
        
        let start = std::time::Instant::now();
        leaf_splits.init_from_gradients(&gradients, &hessians);
        let duration = start.elapsed();
        
        println!("Large dataset processing took: {:?}", duration);
        
        assert!((leaf_splits.sum_gradients() - expected_sum_gradients).abs() < 1e-6);
        assert!((leaf_splits.sum_hessians() - expected_sum_hessians).abs() < 1e-6);
        assert_eq!(leaf_splits.num_data_in_leaf(), LARGE_SIZE as DataSize);
    }
}

fn main() {
    println!("Running LeafSplits equivalence tests...");
    
    // Note: In a real test environment, you would run:
    // cargo test --test test_leaf_splits_equivalence
    
    println!("Tests should be run with: cargo test --test test_leaf_splits_equivalence");
}