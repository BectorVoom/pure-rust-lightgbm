//! Tree learning module for the Pure Rust LightGBM framework.
//!
//! This module provides a complete tree learning subsystem including decision tree
//! structures, various learning algorithms, histogram construction, split finding,
//! and feature sampling utilities.

pub mod histogram;
pub mod learner;
pub mod node;
pub mod sampling;
pub mod split;
pub mod tree;

// Re-export key types and traits for easy access
pub use histogram::{
    BinMapper, FeatureType, HistogramBuilder, HistogramBuilderConfig, HistogramPool,
    HistogramPoolConfig, SimdConfig, SimdHistogramAccumulator,
};
pub use learner::{
    Dataset, FeatureGroupingStrategy, FeatureParallelTreeLearner, FeatureParallelTreeLearnerConfig,
    LoadBalancingStrategy, ParallelTreeLearner, ParallelTreeLearnerConfig, SerialTreeLearner,
    SerialTreeLearnerConfig, TreeLearnerAutoConfig, TreeLearnerFactory, TreeLearnerTrait,
    TreeLearnerType, TreeLearnerUtils,
};
pub use node::TreeNode;
pub use sampling::{
    FeatureSampler, FeatureSamplingConfig, SamplingStrategy,
};
pub use split::{
    ConstraintManager, ConstraintType, ConstraintValidator, MonotonicConstraint, SplitEvaluator,
    SplitEvaluatorConfig, SplitFinder, SplitFinderConfig, SplitInfo,
};
pub use tree::Tree;

/// Tree learning subsystem providing high-level APIs for decision tree construction.
pub struct TreeLearningSubsystem {
    learner: Box<dyn TreeLearnerTrait>,
    config: TreeLearnerAutoConfig,
}

impl TreeLearningSubsystem {
    /// Creates a new tree learning subsystem with automatic configuration.
    pub fn new(
        num_data: usize,
        num_features: usize,
        available_memory_gb: f64,
    ) -> anyhow::Result<Self> {
        let config = TreeLearnerUtils::create_optimized_config(
            num_data,
            num_features,
            available_memory_gb,
        );

        let learner = TreeLearnerFactory::create_auto(&config)?;

        Ok(TreeLearningSubsystem { learner, config })
    }

    /// Creates a tree learning subsystem with explicit configuration.
    pub fn with_config(config: TreeLearnerAutoConfig) -> anyhow::Result<Self> {
        let learner = TreeLearnerFactory::create_auto(&config)?;
        Ok(TreeLearningSubsystem { learner, config })
    }

    /// Trains a decision tree using the configured learner.
    pub fn train_tree(
        &mut self,
        dataset: &Dataset,
        gradients: &ndarray::ArrayView1<crate::core::types::Score>,
        hessians: &ndarray::ArrayView1<crate::core::types::Score>,
        iteration: usize,
    ) -> anyhow::Result<Tree> {
        self.learner.train(dataset, gradients, hessians, iteration)
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &TreeLearnerAutoConfig {
        &self.config
    }

    /// Returns the learner type being used.
    pub fn learner_type(&self) -> TreeLearnerType {
        self.config.learner_type
    }

    /// Returns the learner name.
    pub fn learner_name(&self) -> &'static str {
        self.learner.name()
    }

    /// Resets the learner state.
    pub fn reset(&mut self) {
        self.learner.reset();
    }

    /// Updates the configuration and recreates the learner if necessary.
    pub fn update_config(&mut self, config: TreeLearnerAutoConfig) -> anyhow::Result<()> {
        if config.learner_type != self.config.learner_type {
            // Need to recreate the learner
            self.learner = TreeLearnerFactory::create_auto(&config)?;
        }
        self.config = config;
        Ok(())
    }

    /// Estimates the memory usage of the current configuration.
    pub fn estimate_memory_usage(&self, num_data: usize, num_features: usize) -> f64 {
        let base_memory = (num_data * num_features * 4) as f64; // 4 bytes per f32
        
        match self.config.learner_type {
            TreeLearnerType::Serial => {
                // Histogram memory + tree memory
                let histogram_memory = num_features * self.config.serial_config.max_bin * 16; // 8 bytes per f64 * 2
                base_memory + histogram_memory as f64
            }
            TreeLearnerType::Parallel => {
                // Additional memory for parallel processing
                let thread_overhead = self.config.parallel_config.num_threads as f64 * 1024.0 * 1024.0; // 1MB per thread
                let histogram_memory = num_features * self.config.parallel_config.base_config.max_bin * 16;
                base_memory + histogram_memory as f64 + thread_overhead
            }
            TreeLearnerType::FeatureParallel => {
                // Feature-parallel specific memory
                let thread_overhead = self.config.feature_parallel_config.num_feature_threads as f64 * 512.0 * 1024.0; // 512KB per thread
                let histogram_memory = num_features * self.config.feature_parallel_config.base_config.max_bin * 16;
                base_memory + histogram_memory as f64 + thread_overhead
            }
        }
    }
}

/// Convenience functions for common tree learning operations.
pub mod convenience {
    use super::*;
    use crate::core::types::Score;
    use ndarray::{Array1, Array2, ArrayView1};

    /// Trains a single decision tree with default configuration.
    pub fn train_simple_tree(
        features: Array2<f32>,
        gradients: Array1<Score>,
        hessians: Array1<Score>,
    ) -> anyhow::Result<Tree> {
        let dataset = Dataset::new(features, 255)?;
        let config = SerialTreeLearnerConfig::default();
        let mut learner = SerialTreeLearner::new(config)?;
        
        learner.train(&dataset, &gradients.view(), &hessians.view(), 0)
    }

    /// Trains a decision tree optimized for the given dataset size.
    pub fn train_optimized_tree(
        features: Array2<f32>,
        gradients: Array1<Score>,
        hessians: Array1<Score>,
        available_memory_gb: f64,
    ) -> anyhow::Result<Tree> {
        let (num_data, num_features) = features.dim();
        let dataset = Dataset::new(features, 255)?;
        
        let mut subsystem = TreeLearningSubsystem::new(
            num_data,
            num_features,
            available_memory_gb,
        )?;
        
        subsystem.train_tree(&dataset, &gradients.view(), &hessians.view(), 0)
    }

    /// Creates a dataset from feature matrix with automatic bin mapping.
    pub fn create_dataset(features: Array2<f32>, max_bin: Option<usize>) -> anyhow::Result<Dataset> {
        let max_bin = max_bin.unwrap_or(255);
        Dataset::new(features, max_bin)
    }

    /// Validates that a dataset is compatible with given gradients and hessians.
    pub fn validate_training_data(
        dataset: &Dataset,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
    ) -> anyhow::Result<()> {
        if dataset.num_data as usize != gradients.len() {
            return Err(anyhow::anyhow!(
                "Dataset has {} data points but gradients array has {} elements",
                dataset.num_data,
                gradients.len()
            ));
        }

        if dataset.num_data as usize != hessians.len() {
            return Err(anyhow::anyhow!(
                "Dataset has {} data points but hessians array has {} elements",
                dataset.num_data,
                hessians.len()
            ));
        }

        if gradients.iter().any(|&g| !g.is_finite()) {
            return Err(anyhow::anyhow!("Gradients contain non-finite values"));
        }

        if hessians.iter().any(|&h| !h.is_finite() || h <= 0.0) {
            return Err(anyhow::anyhow!("Hessians contain non-finite or non-positive values"));
        }

        Ok(())
    }

    /// Estimates the optimal number of leaves for a given dataset.
    pub fn estimate_optimal_leaves(num_data: usize, num_features: usize) -> usize {
        // Heuristic: roughly log2(num_data) leaves, capped by features and practical limits
        let log_based = (num_data as f64).log2().ceil() as usize;
        let feature_based = num_features.min(31);
        let practical_max = 255;
        
        log_based.min(feature_based).min(practical_max).max(3)
    }

    /// Estimates the optimal maximum depth for a given dataset.
    pub fn estimate_optimal_depth(num_data: usize, num_features: usize) -> usize {
        // Heuristic: depth should allow for reasonable tree size without overfitting
        let log_based = (num_data as f64).log2().ceil() as usize / 2;
        let feature_based = (num_features as f64).log2().ceil() as usize + 2;
        
        log_based.min(feature_based).min(15).max(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Score;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_tree_learning_subsystem_creation() {
        let subsystem = TreeLearningSubsystem::new(1000, 50, 4.0);
        assert!(subsystem.is_ok());
        
        let subsystem = subsystem.unwrap();
        assert!(matches!(
            subsystem.learner_type(),
            TreeLearnerType::Serial | TreeLearnerType::Parallel | TreeLearnerType::FeatureParallel
        ));
    }

    #[test]
    fn test_subsystem_with_config() {
        let config = TreeLearnerAutoConfig {
            learner_type: TreeLearnerType::Serial,
            ..Default::default()
        };

        let subsystem = TreeLearningSubsystem::with_config(config);
        assert!(subsystem.is_ok());
        
        let subsystem = subsystem.unwrap();
        assert_eq!(subsystem.learner_type(), TreeLearnerType::Serial);
        assert_eq!(subsystem.learner_name(), "serial");
    }

    #[test]
    fn test_convenience_simple_tree() {
        let features = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ).unwrap();
        let gradients = Array1::from(vec![-1.0, -0.5, 0.5, 1.0]);
        let hessians = Array1::from(vec![1.0; 4]);

        let result = convenience::train_simple_tree(features, gradients, hessians);
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert!(tree.num_nodes() >= 1);
        assert!(tree.num_leaves() >= 1);
    }

    #[test]
    fn test_convenience_optimized_tree() {
        let features = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 2.0, 3.0,
                2.0, 3.0, 1.0,
                3.0, 1.0, 2.0,
                1.0, 3.0, 2.0,
                2.0, 1.0, 3.0,
                3.0, 2.0, 1.0,
            ],
        ).unwrap();
        let gradients = Array1::from(vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]);
        let hessians = Array1::from(vec![1.0; 6]);

        let result = convenience::train_optimized_tree(features, gradients, hessians, 4.0);
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert!(tree.num_nodes() >= 1);
    }

    #[test]
    fn test_convenience_dataset_creation() {
        let features = Array2::from_shape_vec(
            (3, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ).unwrap();

        let dataset = convenience::create_dataset(features, Some(10));
        assert!(dataset.is_ok());

        let dataset = dataset.unwrap();
        assert_eq!(dataset.num_data, 3);
        assert_eq!(dataset.num_features, 2);
    }

    #[test]
    fn test_convenience_data_validation() {
        let features = Array2::from_shape_vec(
            (3, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ).unwrap();
        let dataset = Dataset::new(features, 10).unwrap();

        let gradients = Array1::from(vec![0.1, 0.2, 0.3]);
        let hessians = Array1::from(vec![1.0, 1.0, 1.0]);

        let result = convenience::validate_training_data(
            &dataset,
            &gradients.view(),
            &hessians.view(),
        );
        assert!(result.is_ok());

        // Test mismatched sizes
        let wrong_gradients = Array1::from(vec![0.1, 0.2]);
        let result = convenience::validate_training_data(
            &dataset,
            &wrong_gradients.view(),
            &hessians.view(),
        );
        assert!(result.is_err());

        // Test invalid values
        let invalid_hessians = Array1::from(vec![1.0, 0.0, -1.0]); // Contains zero and negative
        let result = convenience::validate_training_data(
            &dataset,
            &gradients.view(),
            &invalid_hessians.view(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_optimal_parameter_estimation() {
        // Test leaf estimation
        let leaves = convenience::estimate_optimal_leaves(1000, 20);
        assert!(leaves >= 3 && leaves <= 31);

        let leaves_large = convenience::estimate_optimal_leaves(100000, 100);
        assert!(leaves_large >= leaves);

        // Test depth estimation
        let depth = convenience::estimate_optimal_depth(1000, 20);
        assert!(depth >= 3 && depth <= 15);

        let depth_large = convenience::estimate_optimal_depth(100000, 100);
        assert!(depth_large >= depth || depth_large == 15); // May be capped at 15
    }

    #[test]
    fn test_memory_estimation() {
        let config = TreeLearnerAutoConfig {
            learner_type: TreeLearnerType::Serial,
            ..Default::default()
        };

        let subsystem = TreeLearningSubsystem::with_config(config).unwrap();
        let memory_usage = subsystem.estimate_memory_usage(1000, 50);
        
        assert!(memory_usage > 0.0);
        
        // Parallel should use more memory
        let config_parallel = TreeLearnerAutoConfig {
            learner_type: TreeLearnerType::Parallel,
            ..Default::default()
        };
        
        let subsystem_parallel = TreeLearningSubsystem::with_config(config_parallel).unwrap();
        let memory_usage_parallel = subsystem_parallel.estimate_memory_usage(1000, 50);
        
        assert!(memory_usage_parallel > memory_usage);
    }

    #[test]
    fn test_config_update() {
        let config = TreeLearnerAutoConfig {
            learner_type: TreeLearnerType::Serial,
            ..Default::default()
        };

        let mut subsystem = TreeLearningSubsystem::with_config(config).unwrap();
        assert_eq!(subsystem.learner_type(), TreeLearnerType::Serial);

        // Update to different learner type
        let new_config = TreeLearnerAutoConfig {
            learner_type: TreeLearnerType::Parallel,
            ..Default::default()
        };

        let result = subsystem.update_config(new_config);
        assert!(result.is_ok());
        assert_eq!(subsystem.learner_type(), TreeLearnerType::Parallel);
    }
}