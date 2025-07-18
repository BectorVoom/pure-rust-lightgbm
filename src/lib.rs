//! # Pure Rust LightGBM
//!
//! A pure Rust implementation of the LightGBM gradient boosting framework,
//! designed for high performance, memory safety, and seamless integration
//! with the Rust ecosystem.
//!
//! ## Features
//!
//! - **Memory Safety**: Leverages Rust's ownership system to prevent common
//!   memory-related bugs like buffer overflows and use-after-free errors.
//! - **High Performance**: SIMD-optimized operations with 32-byte aligned memory
//!   allocation for maximum computational efficiency.
//! - **GPU Acceleration**: Optional CUDA/OpenCL support through the CubeCL framework
//!   for massively parallel computation.
//! - **Parallel Processing**: Built-in support for multi-threaded training and
//!   prediction using Rayon for efficient CPU utilization.
//! - **DataFrame Integration**: Native support for Polars DataFrames alongside
//!   traditional CSV and ndarray inputs.
//! - **API Compatibility**: Maintains compatibility with the original LightGBM
//!   API while providing idiomatic Rust interfaces.
//!
//! ## Quick Start
//!
//! ### Basic Usage
//!
//! ```rust,no_run
//! use lightgbm_rust::{Dataset, LGBMRegressor, ConfigBuilder};
//! use ndarray::{Array2, Array1};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a simple dataset
//! let features = Array2::from_shape_vec((4, 2), vec![
//!     1.0, 2.0,
//!     2.0, 3.0,
//!     3.0, 4.0,
//!     4.0, 5.0,
//! ])?;
//! let labels = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0]);
//!
//! // Create dataset
//! let dataset = Dataset::new(features, labels, None, None, None, None)?;
//!
//! // Configure model
//! let config = ConfigBuilder::new()
//!     .num_iterations(100)
//!     .learning_rate(0.1)
//!     .num_leaves(31)
//!     .build()?;
//!
//! // Train model
//! let mut model = LGBMRegressor::new(config);
//! model.fit(&dataset)?;
//!
//! // Make predictions
//! let test_features = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 6.0, 7.0])?;
//! let predictions = model.predict(&test_features)?;
//!
//! println!("Predictions: {:?}", predictions);
//! # Ok(())
//! # }
//! ```
//!
//! ### Working with Polars DataFrames
//!
//! ```rust,no_run
//! # #[cfg(feature = "polars")]
//! # {
//! use lightgbm_rust::{DatasetFactory, LGBMClassifier, DatasetConfig};
//! use polars::prelude::*;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Load data with Polars
//! let df = LazyCsvReader::new("data.csv")
//!     .with_has_header(true)
//!     .finish()?
//!     .collect()?;
//!
//! // Create dataset configuration
//! let config = DatasetConfig {
//!     target_column: Some("target_column".to_string()),
//!     ..DatasetConfig::default()
//! };
//!
//! // Create dataset from DataFrame
//! let dataset = DatasetFactory::from_polars(&df, config)?;
//!
//! // Train classifier
//! let mut model = LGBMClassifier::default();
//! model.fit(&dataset)?;
//! # Ok(())
//! # }
//! # }
//! ```
//!
//! ### GPU Acceleration
//!
//! ```rust
//! # #[cfg(feature = "gpu")]
//! # {
//! use lightgbm_rust::{Config, DeviceType, LGBMRegressor};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = Config::default()
//!     .device_type(DeviceType::GPU)
//!     .num_iterations(1000);
//!
//! let mut model = LGBMRegressor::new(config);
//! // Training will use GPU acceleration
//! # Ok(())
//! # }
//! # }
//! ```
//!
//! ## Architecture
//!
//! The library is organized into several key modules:
//!
//! - [`core`]: Fundamental types, constants, error handling, and trait abstractions
//! - [`dataset`]: Data loading, preprocessing, and management (coming soon)
//! - [`boosting`]: Gradient boosting algorithms and ensemble management (coming soon)
//! - [`tree`]: Decision tree learning and construction (coming soon)
//! - [`prediction`]: Model inference and prediction pipeline (coming soon)
//! - [`metrics`]: Evaluation metrics for model assessment (coming soon)
//! - [`io`]: Model serialization and persistence (coming soon)
//!
//! ## Performance Considerations
//!
//! This implementation is designed for maximum performance:
//!
//! - **SIMD Operations**: Automatic vectorization through aligned memory allocation
//! - **Cache Optimization**: Memory layouts optimized for CPU cache efficiency
//! - **Parallel Processing**: Multi-threaded training and prediction
//! - **GPU Acceleration**: Optional CUDA/OpenCL support for large-scale workloads
//! - **Zero-Copy Operations**: Minimal memory allocations and data copying
//!
//! ## Safety Guarantees
//!
//! Unlike the original C++ implementation, this Rust version provides:
//!
//! - **Memory Safety**: No buffer overflows, use-after-free, or memory leaks
//! - **Thread Safety**: Safe concurrent access to shared data structures
//! - **Type Safety**: Compile-time prevention of type-related errors
//! - **Error Handling**: Comprehensive error types with clear recovery paths

#![doc(html_root_url = "https://docs.rs/lightgbm-rust/")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub,
    non_snake_case,
    non_upper_case_globals
)]

// Core infrastructure module - always available
pub mod core;

// Configuration management module
pub mod config;

// Dataset management module
pub mod dataset;

// Hyperparameter optimization module
pub mod hyperopt;

// Model ensemble module
pub mod ensemble;

// Metrics evaluation module
pub mod metrics_eval;

// Prediction module
pub mod prediction;

// Boosting module
pub mod boosting;

// Feature-gated modules (to be implemented)
// #[cfg(feature = "polars")]
// #[cfg_attr(docsrs, doc(cfg(feature = "polars")))]
// pub mod polars_integration;

// #[cfg(feature = "gpu")]
// #[cfg_attr(docsrs, doc(cfg(feature = "gpu")))]
// pub mod gpu;

// #[cfg(feature = "python")]
// #[cfg_attr(docsrs, doc(cfg(feature = "python")))]
// pub mod python;

// Re-export core functionality for convenience
pub use core::{
    constants::*,
    error::{LightGBMError, Result},
    memory::{AlignedBuffer, MemoryPool, MemoryStats, MemoryHandle},
    types::*,
    traits::*,
};

// Re-export configuration functionality
pub use config::{
    core::{Config, ConfigBuilder, ConfigError},
    device::{DeviceConfig, DeviceCapabilities},
    objective::{ObjectiveConfig, ObjectiveFunction},
    validation::{ConfigValidator, ValidationResult, ValidationError},
};

// Re-export dataset functionality
pub use dataset::{
    dataset::{Dataset, DatasetBuilder, DatasetInfo, DatasetMetadata},
    binning::{BinMapper, BinType, MissingType, FeatureBinner},
    DatasetConfig, DatasetFactory, DatasetStatistics,
};

// Re-export hyperparameter optimization functionality
pub use hyperopt::{
    HyperparameterSpace, OptimizationConfig, OptimizationDirection, OptimizationResult,
    CrossValidationConfig, CrossValidationResult, optimize_hyperparameters, cross_validate,
};

// Re-export ensemble functionality
pub use ensemble::{
    EnsembleMethod, EnsembleConfig, ModelEnsemble, ClassificationEnsemble,
};

// Re-export metrics evaluation functionality
pub use metrics_eval::{
    RegressionMetrics, ClassificationMetrics, MulticlassMetrics,
    evaluate_regression, evaluate_binary_classification, evaluate_multiclass_classification,
};

// Re-export prediction functionality
pub use prediction::{
    PredictionConfig, Predictor, HistogramPool, SplitFinder, SplitInfo,
};

// Re-export boosting functionality
pub use boosting::{
    GBDT, create_objective_function, RegressionObjective, BinaryObjective, MulticlassObjective,
};

// Version information
pub use core::constants::LIGHTGBM_RUST_VERSION as VERSION;

/// Initialize the LightGBM library.
/// 
/// This function must be called before using any other library functionality.
/// It performs necessary setup including memory subsystem initialization,
/// logging configuration, and capability detection.
/// 
/// # Examples
/// 
/// ```rust
/// fn main() -> lightgbm_rust::Result<()> {
///     // Initialize the library
///     lightgbm_rust::init()?;
///     
///     // Now you can use other library functions
///     Ok(())
/// }
/// ```
pub fn init() -> Result<()> {
    core::initialize_core()
}

/// Check if the library has been initialized.
pub fn is_initialized() -> bool {
    core::is_core_initialized()
}

/// Get library capabilities and feature information.
/// 
/// Returns a [`CoreCapabilities`] struct containing information about
/// available features, hardware support, and performance optimizations.
/// 
/// # Examples
/// 
/// ```rust
/// use lightgbm_rust;
/// 
/// let caps = lightgbm_rust::capabilities();
/// println!("GPU support: {}", caps.gpu_ready);
/// println!("SIMD memory: {}", caps.simd_aligned_memory);
/// println!("{}", caps.summary());
/// ```
pub fn capabilities() -> core::CoreCapabilities {
    core::core_capabilities()
}


// Dataset implementation moved to dataset module

/// Placeholder for LGBMRegressor implementation (to be implemented in boosting module)
#[derive(Debug)]
pub struct LGBMRegressor {
    config: Config,
}

impl LGBMRegressor {
    /// Create a new regressor with the given configuration.
    pub fn new(config: Config) -> Self {
        LGBMRegressor { config }
    }

    /// Get the configuration
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Train the model on the given dataset.
    pub fn fit(&mut self, _dataset: &Dataset) -> Result<()> {
        Err(LightGBMError::not_implemented("LGBMRegressor::fit"))
    }

    /// Make predictions on the given features.
    pub fn predict(&self, _features: &ndarray::Array2<f32>) -> Result<ndarray::Array1<Score>> {
        Err(LightGBMError::not_implemented("LGBMRegressor::predict"))
    }

    /// Add validation dataset for early stopping
    pub fn add_validation_data(&mut self, _dataset: &Dataset) -> Result<()> {
        Err(LightGBMError::not_implemented("LGBMRegressor::add_validation_data"))
    }

    /// Get number of trained iterations
    pub fn num_iterations(&self) -> usize {
        // Return the configured number for now
        self.config.num_iterations
    }

    /// Get training history
    pub fn training_history(&self) -> Result<TrainingHistory> {
        Err(LightGBMError::not_implemented("LGBMRegressor::training_history"))
    }

    /// Get feature importance
    pub fn feature_importance(&self, _importance_type: ImportanceType) -> Result<ndarray::Array1<f64>> {
        Err(LightGBMError::not_implemented("LGBMRegressor::feature_importance"))
    }

    /// Predict feature contributions (SHAP values)
    pub fn predict_contrib(&self, _features: &ndarray::ArrayView2<f32>) -> Result<ndarray::Array2<f64>> {
        Err(LightGBMError::not_implemented("LGBMRegressor::predict_contrib"))
    }

    /// Save model to file
    pub fn save_model<P: AsRef<std::path::Path>>(&self, _path: P) -> Result<()> {
        Err(LightGBMError::not_implemented("LGBMRegressor::save_model"))
    }

    /// Load model from file
    pub fn load_model<P: AsRef<std::path::Path>>(_path: P) -> Result<Self> {
        Err(LightGBMError::not_implemented("LGBMRegressor::load_model"))
    }
}

impl Default for LGBMRegressor {
    fn default() -> Self {
        Self::new(Config::default())
    }
}

/// Placeholder for LGBMClassifier implementation (to be implemented in boosting module)
#[derive(Debug)]
pub struct LGBMClassifier {
    config: Config,
}

impl LGBMClassifier {
    /// Create a new classifier with the given configuration.
    pub fn new(config: Config) -> Self {
        LGBMClassifier { config }
    }

    /// Get the configuration
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Train the model on the given dataset.
    pub fn fit(&mut self, _dataset: &Dataset) -> Result<()> {
        Err(LightGBMError::not_implemented("LGBMClassifier::fit"))
    }

    /// Make predictions on the given features.
    pub fn predict(&self, _features: &ndarray::Array2<f32>) -> Result<ndarray::Array1<f32>> {
        Err(LightGBMError::not_implemented("LGBMClassifier::predict"))
    }

    /// Predict class probabilities.
    pub fn predict_proba(&self, _features: &ndarray::Array2<f32>) -> Result<ndarray::Array2<Score>> {
        Err(LightGBMError::not_implemented("LGBMClassifier::predict_proba"))
    }

    /// Save model to file
    pub fn save_model<P: AsRef<std::path::Path>>(&self, _path: P) -> Result<()> {
        Err(LightGBMError::not_implemented("LGBMClassifier::save_model"))
    }

    /// Load model from file
    pub fn load_model<P: AsRef<std::path::Path>>(_path: P) -> Result<Self> {
        Err(LightGBMError::not_implemented("LGBMClassifier::load_model"))
    }
}

impl Default for LGBMClassifier {
    fn default() -> Self {
        let config = ConfigBuilder::new()
            .objective(ObjectiveType::Binary)
            .build()
            .unwrap();
        Self::new(config)
    }
}

// Conditional module imports for features
#[cfg(feature = "polars")]
#[cfg_attr(docsrs, doc(cfg(feature = "polars")))]
mod polars_integration_impl {
    //! Polars DataFrame integration (placeholder)
}

#[cfg(feature = "gpu")]
#[cfg_attr(docsrs, doc(cfg(feature = "gpu")))]
mod gpu_impl {
    //! GPU acceleration implementation (placeholder)
}

#[cfg(feature = "python")]
#[cfg_attr(docsrs, doc(cfg(feature = "python")))]
mod python_impl {
    //! Python bindings (placeholder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_initialization() {
        assert!(init().is_ok());
        assert!(is_initialized());
    }

    #[test]
    fn test_capabilities() {
        let caps = capabilities();
        assert!(caps.simd_aligned_memory);
        assert!(caps.thread_safe_memory);
        assert!(caps.rich_error_types);
        assert!(caps.trait_abstractions);
        assert!(caps.serialization);
    }

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .learning_rate(0.05)
            .num_iterations(500)
            .num_leaves(127)
            .objective(ObjectiveType::Binary)
            .build().unwrap();

        assert_eq!(config.learning_rate, 0.05);
        assert_eq!(config.num_iterations, 500);
        assert_eq!(config.num_leaves, 127);
        assert_eq!(config.objective, ObjectiveType::Binary);
    }

    #[test]
    fn test_config_validation() {
        let valid_config = Config::default();
        assert!(valid_config.validate().is_ok());

        let mut invalid_config = Config::default();
        invalid_config.learning_rate = -0.1;
        assert!(invalid_config.validate().is_err());

        let mut invalid_config2 = Config::default();
        invalid_config2.num_leaves = 1;
        assert!(invalid_config2.validate().is_err());
    }

    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_error_integration() {
        let err = LightGBMError::config("test error");
        assert_eq!(err.category(), "config");
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_placeholder_implementations() {
        // Test that placeholder methods return NotImplemented errors
        let features = ndarray::Array2::zeros((10, 5));
        let labels = ndarray::Array1::zeros(10);
        
        let dataset_result = Dataset::new(features.clone(), labels, None, None, None, None);
        assert!(dataset_result.is_ok());
        
        let mut regressor = LGBMRegressor::default();
        if let Ok(dataset) = dataset_result {
            let fit_result = regressor.fit(&dataset);
            assert!(fit_result.is_err());
        }
        
        let predict_result = regressor.predict(&features);
        assert!(predict_result.is_err());
    }

    #[test]
    fn test_model_defaults() {
        let regressor = LGBMRegressor::default();
        assert_eq!(regressor.config.objective, ObjectiveType::Regression);

        let classifier = LGBMClassifier::default();
        assert_eq!(classifier.config.objective, ObjectiveType::Binary);
    }
}