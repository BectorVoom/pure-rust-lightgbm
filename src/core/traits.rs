//! Core trait definitions for Pure Rust LightGBM.
//!
//! This module defines the fundamental trait abstractions that provide
//! consistent interfaces across the entire LightGBM implementation,
//! enabling polymorphism and extensibility.

use crate::core::error::Result;
use crate::core::types::*;
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};
use serde::{Deserialize, Serialize};

use std::fmt::Debug;

/// Core trait for all LightGBM components that can be configured.
pub trait Configurable {
    /// Configuration type associated with this component.
    type Config;

    /// Apply configuration to this component.
    fn configure(&mut self, config: &Self::Config) -> Result<()>;

    /// Get current configuration.
    fn config(&self) -> &Self::Config;

    /// Validate configuration parameters.
    fn validate_config(config: &Self::Config) -> Result<()>;
}

/// Trait for components that can be reset to initial state.
pub trait Resettable {
    /// Reset the component to its initial state.
    fn reset(&mut self);

    /// Check if the component is in its initial state.
    fn is_reset(&self) -> bool;
}

/// Trait for components that can provide debugging information.
pub trait Debuggable {
    /// Get debug information as a string.
    fn debug_info(&self) -> String;

    /// Get detailed state information for debugging.
    fn detailed_state(&self) -> std::collections::HashMap<String, String>;
}

/// Trait for serializable components.
pub trait Persistable: Serialize + for<'de> Deserialize<'de> {
    /// Save component to a file.
    fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let file = std::fs::File::create(path)?;
        bincode::serialize_into(file, self).map_err(|e| {
            crate::core::error::LightGBMError::Serialization {
                message: format!("Failed to serialize: {}", e),
            }
        })
    }

    /// Load component from a file.
    fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self>
    where
        Self: Sized,
    {
        let file = std::fs::File::open(path)?;
        bincode::deserialize_from(file).map_err(|e| {
            crate::core::error::LightGBMError::Serialization {
                message: format!("Failed to deserialize: {}", e),
            }
        })
    }
}

/// Trait for objective functions that compute gradients and hessians.
pub trait ObjectiveFunction: Send + Sync + Debug {
    /// Compute gradients and hessians for the given predictions and labels.
    fn compute_gradients(
        &self,
        predictions: &ArrayView1<'_, Score>,
        labels: &ArrayView1<'_, Label>,
        weights: Option<&ArrayView1<'_, Label>>,
        gradients: &mut ArrayViewMut1<'_, Score>,
        hessians: &mut ArrayViewMut1<'_, Score>,
    ) -> Result<()>;

    /// Transform raw prediction scores to final output format.
    fn transform_predictions(&self, scores: &mut ArrayViewMut1<'_, Score>) -> Result<()>;

    /// Get the number of classes (1 for regression/binary, >1 for multiclass).
    fn num_classes(&self) -> usize;

    /// Get the objective function name.
    fn name(&self) -> &'static str;

    /// Check if this objective requires class information.
    fn requires_class_info(&self) -> bool {
        self.num_classes() > 2
    }

    /// Validate labels for this objective function.
    fn validate_labels(&self, labels: &ArrayView1<'_, Label>) -> Result<()>;

    /// Alias for compute_gradients for backward compatibility
    fn get_gradients(
        &self,
        predictions: &ArrayView1<'_, Score>,
        labels: &ArrayView1<'_, Label>,
        weights: Option<&ArrayView1<'_, Label>>,
        gradients: &mut ArrayViewMut1<'_, Score>,
        hessians: &mut ArrayViewMut1<'_, Score>,
    ) -> Result<()> {
        self.compute_gradients(predictions, labels, weights, gradients, hessians)
    }

    /// Get default evaluation metric for this objective.
    fn default_metric(&self) -> MetricType;
}

/// Trait for tree learning algorithms.
pub trait TreeLearner: Send + Sync {
    /// Train a single decision tree.
    fn train_tree(
        &mut self,
        features: &Array2<f32>,
        gradients: &ArrayView1<'_, Score>,
        hessians: &ArrayView1<'_, Score>,
        data_indices: Option<&[DataSize]>,
    ) -> Result<Box<dyn DecisionTree>>;

    /// Get the tree learner type.
    fn learner_type(&self) -> TreeLearnerType;

    /// Get the tree learner name.
    fn name(&self) -> &'static str;

    /// Check if this learner supports parallel training.
    fn supports_parallel(&self) -> bool;

    /// Set the number of threads for parallel processing.
    fn set_num_threads(&mut self, num_threads: usize);

    /// Set bagging data for the tree learner.
    /// 
    /// # Arguments
    /// * `subset` - Optional subset dataset to use for training
    /// * `used_indices` - Optional array of data indices to use for training
    /// * `num_data` - Number of data points to use
    fn set_bagging_data(
        &mut self,
        subset: Option<&crate::dataset::Dataset>,
        used_indices: Option<&[DataSize]>,
        num_data: DataSize,
    ) -> Result<()>;
}

/// Trait for decision tree implementations.
pub trait DecisionTree: Send + Sync + Debug {
    /// Predict a single sample.
    fn predict(&self, features: &ArrayView1<'_, f32>) -> Result<Score>;

    /// Predict multiple samples.
    fn predict_batch(&self, features: &Array2<f32>) -> Result<Array1<Score>> {
        let mut predictions = Array1::zeros(features.nrows());
        for (i, row) in features.axis_iter(ndarray::Axis(0)).enumerate() {
            predictions[i] = self.predict(&row)?;
        }
        Ok(predictions)
    }

    /// Get leaf index for a single sample.
    fn predict_leaf_index(&self, features: &ArrayView1<'_, f32>) -> Result<NodeIndex>;

    /// Get the number of leaves in the tree.
    fn num_leaves(&self) -> usize;

    /// Get the tree depth.
    fn depth(&self) -> usize;

    /// Get feature importance scores.
    fn feature_importance(&self, importance_type: ImportanceType) -> Array1<f64>;

    /// Check if the tree is a leaf (single node).
    fn is_leaf(&self) -> bool {
        self.num_leaves() == 1
    }

    /// Serialize tree to JSON format.
    fn to_json(&self) -> Result<String>;

    /// Get tree structure information.
    fn tree_info(&self) -> TreeInfo;
}

/// Information about a decision tree's structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeInfo {
    /// Number of leaf nodes in the tree
    pub num_leaves: usize,
    /// Maximum depth of the tree
    pub depth: usize,
    /// Total number of nodes (internal + leaf)
    pub num_nodes: usize,
    /// Number of internal (non-leaf) nodes
    pub num_internal_nodes: usize,
    /// Map of feature indices to their usage count in the tree
    pub feature_usage: std::collections::HashMap<FeatureIndex, usize>,
}

/// Trait for prediction engines.
pub trait Predictor: Send + Sync + Debug {
    /// Predict on a batch of samples.
    fn predict(&self, features: &Array2<f32>) -> Result<Array1<Score>>;

    /// Predict with early stopping.
    fn predict_with_early_stopping(
        &self,
        features: &Array2<f32>,
        num_iterations: Option<usize>,
    ) -> Result<Array1<Score>>;

    /// Predict leaf indices.
    fn predict_leaf_indices(&self, features: &Array2<f32>) -> Result<Array2<NodeIndex>>;

    /// Predict feature contributions (SHAP values).
    fn predict_contributions(&self, features: &Array2<f32>) -> Result<Array2<Score>>;

    /// Get the number of trees in the ensemble.
    fn num_trees(&self) -> usize;

    /// Get the number of features.
    fn num_features(&self) -> usize;

    /// Get the objective type.
    fn objective_type(&self) -> ObjectiveType;
}

/// Trait for evaluation metrics.
pub trait Metric: Send + Sync + Debug {
    /// Compute the metric value.
    fn evaluate(
        &self,
        predictions: &ArrayView1<'_, Score>,
        labels: &ArrayView1<'_, Label>,
        weights: Option<&ArrayView1<'_, Label>>,
    ) -> Result<f64>;

    /// Get the metric name.
    fn name(&self) -> &'static str;

    /// Check if higher values are better for this metric.
    fn higher_is_better(&self) -> bool;

    /// Get the metric type.
    fn metric_type(&self) -> MetricType;

    /// Validate that predictions and labels are compatible.
    fn validate_inputs(
        &self,
        predictions: &ArrayView1<'_, Score>,
        labels: &ArrayView1<'_, Label>,
    ) -> Result<()> {
        if predictions.len() != labels.len() {
            return Err(crate::core::error::LightGBMError::dimension_mismatch(
                format!("predictions: {}", predictions.len()),
                format!("labels: {}", labels.len()),
            ));
        }
        Ok(())
    }
}

/// Trait for feature importance calculation methods.
pub trait FeatureImportanceCalculator: Send + Sync + Debug {
    /// Calculate feature importance from a trained model.
    fn calculate_importance(
        &self,
        trees: &[Box<dyn DecisionTree>],
        importance_type: ImportanceType,
    ) -> Result<Array1<f64>>;

    /// Get the calculator name.
    fn name(&self) -> &'static str;
}

/// Trait for data preprocessing components.
pub trait Preprocessor: Send + Sync + Debug {
    /// Preprocess features.
    fn preprocess(&mut self, features: &mut Array2<f32>) -> Result<()>;

    /// Fit the preprocessor to training data.
    fn fit(&mut self, features: &Array2<f32>) -> Result<()>;

    /// Transform features using fitted parameters.
    fn transform(&self, features: &Array2<f32>) -> Result<Array2<f32>>;

    /// Fit and transform in one step.
    fn fit_transform(&mut self, features: &Array2<f32>) -> Result<Array2<f32>> {
        self.fit(features)?;
        self.transform(features)
    }

    /// Check if the preprocessor has been fitted.
    fn is_fitted(&self) -> bool;

    /// Get preprocessor name.
    fn name(&self) -> &'static str;
}

/// Trait for feature binning strategies.
pub trait FeatureBinner: Send + Sync + Debug {
    /// Create bins for a feature.
    fn create_bins(&mut self, values: &[f32], max_bins: usize) -> Result<Vec<f32>>;

    /// Map a value to its bin index.
    fn value_to_bin(&self, value: f32, feature_index: FeatureIndex) -> Result<BinIndex>;

    /// Get bin boundaries for a feature.
    fn get_bins(&self, feature_index: FeatureIndex) -> Option<&[f32]>;

    /// Get the number of bins for a feature.
    fn num_bins(&self, feature_index: FeatureIndex) -> usize;

    /// Get binner name.
    fn name(&self) -> &'static str;
}

/// Trait for early stopping strategies.
pub trait EarlyStoppingStrategy: Send + Sync + Debug {
    /// Check if training should stop early.
    fn should_stop(&mut self, current_metric: f64, iteration: usize) -> bool;

    /// Reset the early stopping state.
    fn reset(&mut self);

    /// Get the best metric value seen so far.
    fn best_metric(&self) -> Option<f64>;

    /// Get the iteration with the best metric.
    fn best_iteration(&self) -> Option<usize>;

    /// Get strategy name.
    fn name(&self) -> &'static str;
}

/// Trait for dataset loading strategies.
pub trait DatasetLoader: Send + Sync + Debug {
    /// Load dataset from a source.
    fn load(&self, source: &str) -> Result<(Array2<f32>, Array1<Label>)>;

    /// Check if the loader can handle the given source.
    fn can_load(&self, source: &str) -> bool;

    /// Get supported file extensions.
    fn supported_extensions(&self) -> &[&'static str];

    /// Get loader name.
    fn name(&self) -> &'static str;
}

/// Trait for feature selection methods.
pub trait FeatureSelector: Send + Sync + Debug {
    /// Select features based on importance scores.
    fn select_features(
        &self,
        importance_scores: &ArrayView1<'_, f64>,
        k: usize,
    ) -> Result<Vec<FeatureIndex>>;

    /// Get selector name.
    fn name(&self) -> &'static str;
}

/// Trait for hyperparameter optimization.
pub trait HyperparameterOptimizer: Send + Sync + Debug {
    /// Configuration type for the optimizer.
    type Config;

    /// Optimize hyperparameters.
    fn optimize(
        &mut self,
        objective: &dyn ObjectiveFunction,
        train_data: &Array2<f32>,
        train_labels: &Array1<Label>,
        valid_data: &Array2<f32>,
        valid_labels: &Array1<Label>,
    ) -> Result<Self::Config>;

    /// Get optimizer name.
    fn name(&self) -> &'static str;
}

/// Trait for components that can be parallelized.
pub trait Parallelizable {
    /// Set the number of threads.
    fn set_num_threads(&mut self, num_threads: usize);

    /// Get the current number of threads.
    fn num_threads(&self) -> usize;

    /// Check if parallel processing is enabled.
    fn is_parallel(&self) -> bool {
        self.num_threads() > 1
    }
}

/// Trait for components that support GPU acceleration.
pub trait GPUAccelerated {
    /// Check if GPU acceleration is available.
    fn gpu_available(&self) -> bool;

    /// Enable GPU acceleration.
    fn enable_gpu(&mut self) -> Result<()>;

    /// Disable GPU acceleration.
    fn disable_gpu(&mut self);

    /// Check if GPU is currently enabled.
    fn is_gpu_enabled(&self) -> bool;

    /// Get GPU device information.
    fn gpu_device_info(&self) -> Option<String>;
}

/// Marker trait for components that are thread-safe.
pub trait ThreadSafe: Send + Sync {}

/// Auto-implement ThreadSafe for types that are Send + Sync.
impl<T: Send + Sync> ThreadSafe for T {}

/// Trait for components that provide progress reporting.
pub trait ProgressReporter {
    /// Report progress with a percentage (0.0 to 1.0).
    fn report_progress(&self, progress: f64, message: Option<&str>);

    /// Report completion.
    fn report_completion(&self);

    /// Check if progress reporting is enabled.
    fn is_reporting_enabled(&self) -> bool;
}

/// Trait for components that can be validated.
pub trait Validatable {
    /// Validate the component's current state.
    fn validate(&self) -> Result<()>;

    /// Get validation warnings (non-fatal issues).
    fn validation_warnings(&self) -> Vec<String>;
}

/// Blanket implementations for common trait combinations
impl<T> Debuggable for T
where
    T: Debug,
{
    fn debug_info(&self) -> String {
        format!("{:?}", self)
    }

    fn detailed_state(&self) -> std::collections::HashMap<String, String> {
        let mut state = std::collections::HashMap::new();
        state.insert("debug".to_string(), self.debug_info());
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock implementations for testing
    #[derive(Debug)]
    struct MockObjective;

    impl ObjectiveFunction for MockObjective {
        fn compute_gradients(
            &self,
            _predictions: &ArrayView1<'_, Score>,
            _labels: &ArrayView1<'_, Label>,
            _weights: Option<&ArrayView1<'_, Label>>,
            _gradients: &mut ArrayViewMut1<'_, Score>,
            _hessians: &mut ArrayViewMut1<'_, Score>,
        ) -> Result<()> {
            Ok(())
        }

        fn transform_predictions(&self, _scores: &mut ArrayViewMut1<'_, Score>) -> Result<()> {
            Ok(())
        }

        fn num_classes(&self) -> usize {
            1
        }

        fn name(&self) -> &'static str {
            "mock"
        }

        fn validate_labels(&self, _labels: &ArrayView1<'_, Label>) -> Result<()> {
            Ok(())
        }

        fn default_metric(&self) -> MetricType {
            MetricType::MSE
        }
    }

    #[test]
    fn test_objective_function_trait() {
        let objective = MockObjective;
        assert_eq!(objective.name(), "mock");
        assert_eq!(objective.num_classes(), 1);
        assert!(!objective.requires_class_info());
        assert_eq!(objective.default_metric(), MetricType::MSE);
    }

    #[test]
    fn test_thread_safe_trait() {
        fn assert_thread_safe<T: ThreadSafe>(_: T) {}
        assert_thread_safe(MockObjective);
    }

    #[test]
    fn test_debuggable_blanket_impl() {
        let objective = MockObjective;
        let debug_info = objective.debug_info();
        assert!(debug_info.contains("MockObjective"));

        let state = objective.detailed_state();
        assert!(state.contains_key("debug"));
    }
}
