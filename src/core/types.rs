//! Core data types for the Pure Rust LightGBM implementation.
//!
//! This module defines fundamental data types that maintain compatibility
//! with the original LightGBM C++ implementation while leveraging Rust's
//! type system for enhanced safety and performance.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Data indexing type, equivalent to `data_size_t` in LightGBM C++.
/// 32-bit integer supporting up to 2 billion data points.
pub type DataSize = i32;

/// Prediction and gradient value type, equivalent to `score_t` in LightGBM C++.
/// 32-bit float optimized for SIMD operations.
pub type Score = f32;

/// Target value and sample weight type, equivalent to `label_t` in LightGBM C++.
/// 32-bit float for target values and sample weights.
pub type Label = f32;

/// Histogram accumulation type, equivalent to `hist_t` in LightGBM C++.
/// 64-bit float providing numerical stability for histogram operations.
pub type Hist = f64;

/// Feature index type for identifying features in the dataset.
pub type FeatureIndex = usize;

/// Bin index type for discretized feature values.
pub type BinIndex = u32;

/// Tree node identifier type.
pub type NodeIndex = usize;

/// Iteration number type for boosting iterations.
pub type IterationIndex = usize;

/// Configuration enumeration for device selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    /// CPU-based computation
    CPU,
    /// GPU-based computation (requires CUDA or OpenCL)
    GPU,
}

impl Default for DeviceType {
    fn default() -> Self {
        DeviceType::CPU
    }
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::CPU => write!(f, "cpu"),
            DeviceType::GPU => write!(f, "gpu"),
        }
    }
}

/// Objective function types supported by the LightGBM implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObjectiveType {
    /// Regression task (continuous target)
    Regression,
    /// Binary classification task
    Binary,
    /// Multiclass classification task
    Multiclass,
    /// Learning to rank task
    Ranking,
    /// Poisson regression
    Poisson,
    /// Gamma regression
    Gamma,
    /// Tweedie regression
    Tweedie,
}

impl Default for ObjectiveType {
    fn default() -> Self {
        ObjectiveType::Regression
    }
}

impl fmt::Display for ObjectiveType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObjectiveType::Regression => write!(f, "regression"),
            ObjectiveType::Binary => write!(f, "binary"),
            ObjectiveType::Multiclass => write!(f, "multiclass"),
            ObjectiveType::Ranking => write!(f, "ranking"),
            ObjectiveType::Poisson => write!(f, "poisson"),
            ObjectiveType::Gamma => write!(f, "gamma"),
            ObjectiveType::Tweedie => write!(f, "tweedie"),
        }
    }
}

/// Boosting strategy types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoostingType {
    /// Gradient Boosting Decision Tree
    GBDT,
    /// Random Forest
    RandomForest,
    /// Dropouts meet Multiple Additive Regression Trees
    DART,
    /// Gradient One-Side Sampling
    GOSS,
}

impl Default for BoostingType {
    fn default() -> Self {
        BoostingType::GBDT
    }
}

impl fmt::Display for BoostingType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoostingType::GBDT => write!(f, "gbdt"),
            BoostingType::RandomForest => write!(f, "rf"),
            BoostingType::DART => write!(f, "dart"),
            BoostingType::GOSS => write!(f, "goss"),
        }
    }
}

/// Tree learning algorithm types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TreeLearnerType {
    /// Serial tree learning
    Serial,
    /// Feature PARALLEL tree learning
    Feature,
    /// Data PARALLEL tree learning
    Data,
    /// Voting PARALLEL tree learning
    Voting,
}

impl Default for TreeLearnerType {
    fn default() -> Self {
        TreeLearnerType::Serial
    }
}

impl fmt::Display for TreeLearnerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TreeLearnerType::Serial => write!(f, "serial"),
            TreeLearnerType::Feature => write!(f, "feature"),
            TreeLearnerType::Data => write!(f, "data"),
            TreeLearnerType::Voting => write!(f, "voting"),
        }
    }
}

/// Feature importance calculation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImportanceType {
    /// Split-based importance (frequency of splits)
    Split,
    /// Gain-based importance (sum of gains from splits)
    Gain,
    /// Coverage-based importance (sum of sample coverage)
    Coverage,
    /// Total gain importance (normalized by total feature contribution)
    TotalGain,
    /// Permutation importance (decrease in accuracy when feature is permuted)
    Permutation,
}

impl Default for ImportanceType {
    fn default() -> Self {
        ImportanceType::Split
    }
}

/// Metric types for model evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricType {
    /// No metric
    None,
    /// Mean Absolute Error
    MAE,
    /// Mean Squared Error
    MSE,
    /// Root Mean Squared Error
    RMSE,
    /// Area Under Curve
    AUC,
    /// Binary log loss
    BinaryLogloss,
    /// Multi-class log loss
    MultiLogloss,
    /// Multi-class error rate
    MultiError,
    /// Cross entropy
    CrossEntropy,
    /// Custom metric with name
    Custom(String),
}

impl Default for MetricType {
    fn default() -> Self {
        MetricType::None
    }
}

impl fmt::Display for MetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetricType::None => write!(f, "none"),
            MetricType::MAE => write!(f, "mae"),
            MetricType::MSE => write!(f, "mse"),
            MetricType::RMSE => write!(f, "rmse"),
            MetricType::AUC => write!(f, "auc"),
            MetricType::BinaryLogloss => write!(f, "binary_logloss"),
            MetricType::MultiLogloss => write!(f, "multi_logloss"),
            MetricType::MultiError => write!(f, "multi_error"),
            MetricType::CrossEntropy => write!(f, "cross_entropy"),
            MetricType::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Verbosity levels for logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum VerbosityLevel {
    /// Fatal errors only
    Fatal = -1,
    /// Warnings and errors
    Warning = 0,
    /// Information, warnings, and errors
    Info = 1,
    /// Debug information
    Debug = 2,
}

impl Default for VerbosityLevel {
    fn default() -> Self {
        VerbosityLevel::Info
    }
}

/// Feature type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureType {
    /// Numerical feature
    Numerical,
    /// Categorical feature
    Categorical,
}

/// Missing value handling strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissingType {
    /// No missing values
    None,
    /// Missing values represented as zero
    Zero,
    /// Missing values represented as NaN
    NaN,
}

impl Default for MissingType {
    fn default() -> Self {
        MissingType::NaN
    }
}

/// Prediction type for model inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionType {
    /// Raw prediction scores
    RawScore,
    /// Probability predictions (for classification)
    Probability,
    /// Leaf index predictions
    LeafIndex,
    /// Feature contributions (SHAP values)
    Contrib,
}

impl Default for PredictionType {
    fn default() -> Self {
        PredictionType::RawScore
    }
}

/// Training history for tracking metrics over iterations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHistory {
    /// Training metrics by iteration
    pub train_metrics: Vec<std::collections::HashMap<String, f64>>,
    /// Validation metrics by iteration
    pub valid_metrics: Vec<std::collections::HashMap<String, f64>>,
}

impl TrainingHistory {
    /// Create a new empty training history
    pub fn new() -> Self {
        Self {
            train_metrics: Vec::new(),
            valid_metrics: Vec::new(),
        }
    }
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Verbosity levels with SILENT option
impl VerbosityLevel {
    /// SILENT mode (no  output)
    pub const SILENT: Self = VerbosityLevel::Fatal;
}

/// PARALLEL tree learner types (aliases for existing types)
impl TreeLearnerType {
    /// PARALLEL tree learning (alias for Feature)
    pub const PARALLEL: Self = TreeLearnerType::Feature;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_sizes() {
        // Verify type sizes match expectations
        assert_eq!(std::mem::size_of::<DataSize>(), 4);
        assert_eq!(std::mem::size_of::<Score>(), 4);
        assert_eq!(std::mem::size_of::<Label>(), 4);
        assert_eq!(std::mem::size_of::<Hist>(), 8);
        assert_eq!(std::mem::size_of::<BinIndex>(), 4);
    }

    #[test]
    fn test_device_type_display() {
        assert_eq!(DeviceType::CPU.to_string(), "cpu");
        assert_eq!(DeviceType::GPU.to_string(), "gpu");
    }

    #[test]
    fn test_objective_type_display() {
        assert_eq!(ObjectiveType::Regression.to_string(), "regression");
        assert_eq!(ObjectiveType::Binary.to_string(), "binary");
        assert_eq!(ObjectiveType::Multiclass.to_string(), "multiclass");
    }

    #[test]
    fn test_boosting_type_display() {
        assert_eq!(BoostingType::GBDT.to_string(), "gbdt");
        assert_eq!(BoostingType::RandomForest.to_string(), "rf");
        assert_eq!(BoostingType::DART.to_string(), "dart");
        assert_eq!(BoostingType::GOSS.to_string(), "goss");
    }

    #[test]
    fn test_metric_type_display() {
        assert_eq!(MetricType::MAE.to_string(), "mae");
        assert_eq!(MetricType::MSE.to_string(), "mse");
        assert_eq!(
            MetricType::Custom("my_metric".to_string()).to_string(),
            "my_metric"
        );
    }

    #[test]
    fn test_defaults() {
        assert_eq!(DeviceType::default(), DeviceType::CPU);
        assert_eq!(ObjectiveType::default(), ObjectiveType::Regression);
        assert_eq!(BoostingType::default(), BoostingType::GBDT);
        assert_eq!(TreeLearnerType::default(), TreeLearnerType::Serial);
        assert_eq!(ImportanceType::default(), ImportanceType::Split);
        assert_eq!(MetricType::default(), MetricType::None);
        assert_eq!(VerbosityLevel::default(), VerbosityLevel::Info);
        assert_eq!(MissingType::default(), MissingType::NaN);
        assert_eq!(PredictionType::default(), PredictionType::RawScore);
    }

    #[test]
    fn test_serialization() {
        let device = DeviceType::GPU;
        let serialized = serde_json::to_string(&device).unwrap();
        let deserialized: DeviceType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(device, deserialized);

        let objective = ObjectiveType::Multiclass;
        let serialized = serde_json::to_string(&objective).unwrap();
        let deserialized: ObjectiveType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(objective, deserialized);
    }
}
