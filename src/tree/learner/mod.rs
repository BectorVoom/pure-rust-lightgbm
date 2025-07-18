//! Tree learning algorithms for the Pure Rust LightGBM framework.
//!
//! This module provides various tree learning strategies including serial,
//! parallel, and feature-parallel algorithms for decision tree construction.

pub mod feature_parallel;
pub mod parallel;
pub mod serial;

// Re-export key types and traits
pub use feature_parallel::{
    FeatureGroupingStrategy, FeatureParallelTreeLearner, FeatureParallelTreeLearnerConfig,
};
pub use parallel::{
    LoadBalancingStrategy, ParallelTreeLearner, ParallelTreeLearnerConfig,
};
pub use serial::{
    Dataset, SerialTreeLearner, SerialTreeLearnerConfig,
};

/// Tree learner factory for creating different types of learners.
pub struct TreeLearnerFactory;

impl TreeLearnerFactory {
    /// Creates a serial tree learner.
    pub fn create_serial(config: SerialTreeLearnerConfig) -> anyhow::Result<SerialTreeLearner> {
        SerialTreeLearner::new(config)
    }

    /// Creates a parallel tree learner.
    pub fn create_parallel(config: ParallelTreeLearnerConfig) -> anyhow::Result<ParallelTreeLearner> {
        ParallelTreeLearner::new(config)
    }

    /// Creates a feature-parallel tree learner.
    pub fn create_feature_parallel(
        config: FeatureParallelTreeLearnerConfig,
    ) -> anyhow::Result<FeatureParallelTreeLearner> {
        FeatureParallelTreeLearner::new(config)
    }

    /// Creates the appropriate tree learner based on configuration.
    pub fn create_auto(
        config: &TreeLearnerAutoConfig,
    ) -> anyhow::Result<Box<dyn TreeLearnerTrait>> {
        match config.learner_type {
            TreeLearnerType::Serial => {
                let learner = Self::create_serial(config.serial_config.clone())?;
                Ok(Box::new(learner))
            }
            TreeLearnerType::Parallel => {
                let learner = Self::create_parallel(config.parallel_config.clone())?;
                Ok(Box::new(learner))
            }
            TreeLearnerType::FeatureParallel => {
                let learner = Self::create_feature_parallel(config.feature_parallel_config.clone())?;
                Ok(Box::new(learner))
            }
        }
    }
}

/// Configuration for automatic tree learner selection.
#[derive(Debug, Clone)]
pub struct TreeLearnerAutoConfig {
    pub learner_type: TreeLearnerType,
    pub serial_config: SerialTreeLearnerConfig,
    pub parallel_config: ParallelTreeLearnerConfig,
    pub feature_parallel_config: FeatureParallelTreeLearnerConfig,
}

impl Default for TreeLearnerAutoConfig {
    fn default() -> Self {
        TreeLearnerAutoConfig {
            learner_type: TreeLearnerType::Serial,
            serial_config: SerialTreeLearnerConfig::default(),
            parallel_config: ParallelTreeLearnerConfig::default(),
            feature_parallel_config: FeatureParallelTreeLearnerConfig::default(),
        }
    }
}

/// Tree learner type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeLearnerType {
    /// Serial tree learning
    Serial,
    /// Data-parallel tree learning
    Parallel,
    /// Feature-parallel tree learning
    FeatureParallel,
}

impl Default for TreeLearnerType {
    fn default() -> Self {
        TreeLearnerType::Serial
    }
}

impl std::fmt::Display for TreeLearnerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TreeLearnerType::Serial => write!(f, "serial"),
            TreeLearnerType::Parallel => write!(f, "parallel"),
            TreeLearnerType::FeatureParallel => write!(f, "feature_parallel"),
        }
    }
}

/// Common trait for all tree learners to enable polymorphism.
pub trait TreeLearnerTrait: Send + Sync {
    /// Trains a decision tree using the given dataset and gradients.
    fn train(
        &mut self,
        dataset: &Dataset,
        gradients: &ndarray::ArrayView1<crate::core::types::Score>,
        hessians: &ndarray::ArrayView1<crate::core::types::Score>,
        iteration: usize,
    ) -> anyhow::Result<crate::tree::tree::Tree>;

    /// Resets the learner state.
    fn reset(&mut self);

    /// Returns the learner type name.
    fn name(&self) -> &'static str;
}

// Implement the trait for all learner types
impl TreeLearnerTrait for SerialTreeLearner {
    fn train(
        &mut self,
        dataset: &Dataset,
        gradients: &ndarray::ArrayView1<crate::core::types::Score>,
        hessians: &ndarray::ArrayView1<crate::core::types::Score>,
        iteration: usize,
    ) -> anyhow::Result<crate::tree::tree::Tree> {
        self.train(dataset, gradients, hessians, iteration)
    }

    fn reset(&mut self) {
        self.reset()
    }

    fn name(&self) -> &'static str {
        "serial"
    }
}

impl TreeLearnerTrait for ParallelTreeLearner {
    fn train(
        &mut self,
        dataset: &Dataset,
        gradients: &ndarray::ArrayView1<crate::core::types::Score>,
        hessians: &ndarray::ArrayView1<crate::core::types::Score>,
        iteration: usize,
    ) -> anyhow::Result<crate::tree::tree::Tree> {
        self.train(dataset, gradients, hessians, iteration)
    }

    fn reset(&mut self) {
        // Note: Parallel learner doesn't have a reset method, so we skip it
    }

    fn name(&self) -> &'static str {
        "parallel"
    }
}

impl TreeLearnerTrait for FeatureParallelTreeLearner {
    fn train(
        &mut self,
        dataset: &Dataset,
        gradients: &ndarray::ArrayView1<crate::core::types::Score>,
        hessians: &ndarray::ArrayView1<crate::core::types::Score>,
        iteration: usize,
    ) -> anyhow::Result<crate::tree::tree::Tree> {
        self.train(dataset, gradients, hessians, iteration)
    }

    fn reset(&mut self) {
        self.reset()
    }

    fn name(&self) -> &'static str {
        "feature_parallel"
    }
}

/// Utility functions for tree learner selection and optimization.
pub struct TreeLearnerUtils;

impl TreeLearnerUtils {
    /// Selects the optimal tree learner type based on dataset characteristics.
    pub fn select_optimal_learner_type(
        num_data: usize,
        num_features: usize,
        num_threads: usize,
    ) -> TreeLearnerType {
        // Heuristic for learner selection
        if num_threads <= 1 {
            TreeLearnerType::Serial
        } else if num_features > num_threads * 50 {
            // Many features: use feature parallelism
            TreeLearnerType::FeatureParallel
        } else if num_data > 10000 {
            // Large dataset: use data parallelism
            TreeLearnerType::Parallel
        } else {
            // Default to serial for small datasets
            TreeLearnerType::Serial
        }
    }

    /// Estimates the computational cost for different learner types.
    pub fn estimate_training_cost(
        learner_type: TreeLearnerType,
        num_data: usize,
        num_features: usize,
        num_threads: usize,
    ) -> f64 {
        let base_cost = (num_data as f64) * (num_features as f64).ln();
        
        match learner_type {
            TreeLearnerType::Serial => base_cost,
            TreeLearnerType::Parallel => {
                // Parallel efficiency typically 70-90% due to overhead
                base_cost / (num_threads as f64 * 0.8)
            }
            TreeLearnerType::FeatureParallel => {
                // Feature parallel efficiency depends on feature distribution
                let feature_parallel_efficiency = if num_features >= num_threads * 10 {
                    0.85
                } else {
                    0.6
                };
                base_cost / (num_threads as f64 * feature_parallel_efficiency)
            }
        }
    }

    /// Creates an optimized configuration based on dataset characteristics.
    pub fn create_optimized_config(
        num_data: usize,
        num_features: usize,
        available_memory_gb: f64,
    ) -> TreeLearnerAutoConfig {
        let num_threads = num_cpus::get();
        let learner_type = Self::select_optimal_learner_type(num_data, num_features, num_threads);

        // Adjust parameters based on memory constraints
        let max_bin = if available_memory_gb > 8.0 { 255 } else { 127 };
        let max_leaves = if available_memory_gb > 4.0 { 31 } else { 15 };

        let base_serial_config = SerialTreeLearnerConfig {
            max_leaves,
            max_bin,
            ..Default::default()
        };

        let parallel_config = ParallelTreeLearnerConfig {
            base_config: base_serial_config.clone(),
            num_threads,
            ..Default::default()
        };

        let feature_parallel_config = FeatureParallelTreeLearnerConfig {
            base_config: base_serial_config.clone(),
            num_feature_threads: num_threads,
            features_per_thread: (num_features + num_threads - 1) / num_threads,
            ..Default::default()
        };

        TreeLearnerAutoConfig {
            learner_type,
            serial_config: base_serial_config,
            parallel_config,
            feature_parallel_config,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_learner_factory() {
        // Test serial learner creation
        let serial_config = SerialTreeLearnerConfig::default();
        let serial_learner = TreeLearnerFactory::create_serial(serial_config);
        assert!(serial_learner.is_ok());

        // Test parallel learner creation
        let parallel_config = ParallelTreeLearnerConfig::default();
        let parallel_learner = TreeLearnerFactory::create_parallel(parallel_config);
        assert!(parallel_learner.is_ok());

        // Test feature-parallel learner creation
        let feature_parallel_config = FeatureParallelTreeLearnerConfig::default();
        let feature_parallel_learner = TreeLearnerFactory::create_feature_parallel(feature_parallel_config);
        assert!(feature_parallel_learner.is_ok());
    }

    #[test]
    fn test_learner_type_selection() {
        // Small dataset should use serial
        let learner_type = TreeLearnerUtils::select_optimal_learner_type(1000, 10, 4);
        assert_eq!(learner_type, TreeLearnerType::Serial);

        // Large dataset should use parallel
        let learner_type = TreeLearnerUtils::select_optimal_learner_type(50000, 20, 4);
        assert_eq!(learner_type, TreeLearnerType::Parallel);

        // Many features should use feature parallel
        let learner_type = TreeLearnerUtils::select_optimal_learner_type(10000, 500, 4);
        assert_eq!(learner_type, TreeLearnerType::FeatureParallel);

        // Single thread should use serial
        let learner_type = TreeLearnerUtils::select_optimal_learner_type(10000, 100, 1);
        assert_eq!(learner_type, TreeLearnerType::Serial);
    }

    #[test]
    fn test_training_cost_estimation() {
        let num_data = 10000;
        let num_features = 100;
        let num_threads = 4;

        let serial_cost = TreeLearnerUtils::estimate_training_cost(
            TreeLearnerType::Serial, num_data, num_features, num_threads
        );

        let parallel_cost = TreeLearnerUtils::estimate_training_cost(
            TreeLearnerType::Parallel, num_data, num_features, num_threads
        );

        let feature_parallel_cost = TreeLearnerUtils::estimate_training_cost(
            TreeLearnerType::FeatureParallel, num_data, num_features, num_threads
        );

        // Parallel should be faster than serial
        assert!(parallel_cost < serial_cost);
        assert!(feature_parallel_cost < serial_cost);
    }

    #[test]
    fn test_optimized_config_creation() {
        let config = TreeLearnerUtils::create_optimized_config(10000, 200, 8.0);
        
        // Should select feature parallel for many features
        assert_eq!(config.learner_type, TreeLearnerType::FeatureParallel);
        
        // Should use higher limits with more memory
        assert_eq!(config.serial_config.max_bin, 255);
        assert_eq!(config.serial_config.max_leaves, 31);

        // Test with limited memory
        let config_limited = TreeLearnerUtils::create_optimized_config(10000, 200, 2.0);
        assert_eq!(config_limited.serial_config.max_bin, 127);
        assert_eq!(config_limited.serial_config.max_leaves, 15);
    }

    #[test]
    fn test_auto_learner_creation() {
        let config = TreeLearnerAutoConfig {
            learner_type: TreeLearnerType::Serial,
            ..Default::default()
        };

        let learner = TreeLearnerFactory::create_auto(&config);
        assert!(learner.is_ok());
        
        let learner = learner.unwrap();
        assert_eq!(learner.name(), "serial");
    }

    #[test]
    fn test_learner_type_display() {
        assert_eq!(TreeLearnerType::Serial.to_string(), "serial");
        assert_eq!(TreeLearnerType::Parallel.to_string(), "parallel");
        assert_eq!(TreeLearnerType::FeatureParallel.to_string(), "feature_parallel");
    }
}