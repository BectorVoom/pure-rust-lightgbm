//! Split finding and evaluation module for the Pure Rust LightGBM framework.
//!
//! This module provides comprehensive split finding algorithms, evaluation
//! mechanisms, and constraint enforcement for decision tree construction.

pub mod constraints;
pub mod evaluator;
pub mod finder;

// Re-export key types and traits
pub use constraints::{
    ConstraintManager, ConstraintType, ConstraintValidationResult, ConstraintValidator,
    InteractionConstraint, MaxFeatureUsageConstraint, MonotonicConstraint,
};
pub use evaluator::{
    CategoricalSplitHandler, MonotonicConstraint as EvaluatorMonotonicConstraint,
    SplitEvaluationResult, SplitEvaluator, SplitEvaluatorConfig,
};
pub use finder::{SplitFinder, SplitFinderConfig, SplitInfo};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::FeatureIndex;
    use ndarray::Array1;

    #[test]
    fn test_module_integration() {
        // Test that all modules work together
        let config = SplitFinderConfig::default();
        let finder = SplitFinder::new(config);

        let evaluator_config = SplitEvaluatorConfig::default();
        let evaluator = SplitEvaluator::new(evaluator_config);

        let mut constraint_manager = ConstraintManager::new();
        constraint_manager.set_monotonic_constraint(0, MonotonicConstraint::Increasing);

        // Verify all components are accessible
        assert_eq!(finder.config().min_data_in_leaf, 20);
        assert_eq!(evaluator.config().max_cat_to_onehot, 4);
        assert!(constraint_manager.monotonic_constraints().contains_key(&0));
    }

    #[test]
    fn test_split_workflow() {
        // Test a complete split finding and evaluation workflow
        let finder_config = SplitFinderConfig {
            min_data_in_leaf: 5,
            min_sum_hessian_in_leaf: 1.0,
            lambda_l1: 0.0,
            lambda_l2: 0.1,
            min_split_gain: 0.1,
            max_bin: 10,
        };

        let finder = SplitFinder::new(finder_config);

        // Create sample histogram data
        let histogram = Array1::from(vec![
            -10.0, 5.0,  // bin 0
            -5.0, 3.0,   // bin 1
            5.0, 2.0,    // bin 2
            10.0, 4.0,   // bin 3
        ]);

        let bin_boundaries = vec![1.0, 2.0, 3.0, 4.0];

        // Find best split
        let split = finder.find_best_split_for_feature(
            0,
            &histogram.view(),
            0.0,  // total_sum_gradient
            14.0, // total_sum_hessian
            100,  // total_count
            &bin_boundaries,
        );

        assert!(split.is_some());
        let split = split.unwrap();
        assert!(split.is_valid());

        // Evaluate the split
        let evaluator_config = SplitEvaluatorConfig::default();
        let evaluator = SplitEvaluator::new(evaluator_config);
        let evaluation = evaluator.evaluate_split(&split, 0.0);
        assert!(evaluation.is_valid);

        // Check constraints
        let constraint_manager = ConstraintManager::new();
        let validation = constraint_manager.validate_split(&split, 1, 0.0, &[]);
        assert!(validation.is_valid());
    }
}