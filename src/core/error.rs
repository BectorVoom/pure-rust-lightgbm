//! Error handling and error types for Pure Rust LightGBM.
//!
//! This module provides comprehensive error handling using Rust's Result type
//! system, ensuring clear error propagation and recovery mechanisms throughout
//! the entire LightGBM implementation.

use std::io;
use thiserror::Error;

/// Main error type for the LightGBM library.
///
/// This enum covers all possible error conditions that can occur during
/// dataset loading, model training, prediction, and other operations.
#[derive(Error, Debug)]
pub enum LightGBMError {
    /// Configuration and validation errors
    #[error("Configuration error: {message}")]
    Config { message: String },

    /// Dataset-related errors
    #[error("Dataset error: {message}")]
    Dataset { message: String },

    /// Data dimension mismatch errors
    #[error("Data dimension mismatch: {message}")]
    DataDimensionMismatch { message: String },

    /// Data loading and parsing errors
    #[error("Data loading error: {message}")]
    DataLoading { message: String },

    /// Feature processing errors
    #[error("Feature processing error: {message}")]
    FeatureProcessing { message: String },

    /// Training-related errors
    #[error("Training error: {message}")]
    Training { message: String },

    /// Tree construction errors
    #[error("Tree construction error: {message}")]
    TreeConstruction { message: String },

    /// Prediction errors
    #[error("Prediction error: {message}")]
    Prediction { message: String },

    /// GPU/CUDA computation errors
    #[error("GPU computation error: {message}")]
    GPU { message: String },

    /// Memory allocation and management errors
    #[error("Memory error: {message}")]
    Memory { message: String },

    /// Numerical computation errors (overflow, underflow, NaN)
    #[error("Numerical error: {message}")]
    Numerical { message: String },

    /// Model serialization/deserialization errors
    #[error("Serialization error: {message}")]
    Serialization { message: String },

    /// File I/O errors
    #[error("I/O error: {source}")]
    IO {
        #[from]
        source: io::Error,
    },

    /// CSV parsing errors
    #[error("CSV parsing error: {source}")]
    Csv {
        #[from]
        source: csv::Error,
    },

    /// JSON serialization errors
    #[error("JSON error: {source}")]
    Json {
        #[from]
        source: serde_json::Error,
    },

    /// Bincode serialization errors
    #[error("Bincode error: {source}")]
    Bincode {
        #[from]
        source: bincode::Error,
    },

    /// Thread synchronization errors
    #[error("Threading error: {message}")]
    Threading { message: String },

    /// Invalid input parameters
    #[error("Invalid parameter: {parameter} = {value}, {reason}")]
    InvalidParameter {
        parameter: String,
        value: String,
        reason: String,
    },

    /// Dimension mismatch errors
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },

    /// Out of bounds access
    #[error("Index out of bounds: index {index}, length {length}")]
    IndexOutOfBounds { index: usize, length: usize },

    /// Early stopping conditions
    #[error("Early stopping triggered: {reason}")]
    EarlyStopping { reason: String },

    /// Not implemented functionality
    #[error("Not implemented: {feature}")]
    NotImplemented { feature: String },

    /// Internal library errors (should not occur in normal usage)
    #[error("Internal error: {message}")]
    Internal { message: String },
}

/// Specialized error types for specific components
#[derive(Error, Debug)]
pub enum DatasetError {
    #[error("Empty dataset provided")]
    Empty,

    #[error("Feature count mismatch: expected {expected}, got {actual}")]
    FeatureMismatch { expected: usize, actual: usize },

    #[error("Invalid feature type for feature {index}: expected {expected:?}, got {actual:?}")]
    InvalidFeatureType {
        index: usize,
        expected: crate::core::types::FeatureType,
        actual: crate::core::types::FeatureType,
    },

    #[error("Missing values not supported for feature {index}")]
    MissingValuesNotSupported { index: usize },

    #[error("Categorical feature {index} has too many categories: {count} > {max}")]
    TooManyCategories {
        index: usize,
        count: usize,
        max: usize,
    },

    #[error("Feature {index} has invalid value: {value}")]
    InvalidFeatureValue { index: usize, value: f64 },
}

/// Training-specific errors
#[derive(Error, Debug)]
pub enum TrainingError {
    #[error("Convergence failed after {iterations} iterations")]
    ConvergenceFailed { iterations: usize },

    #[error("Insufficient data: need at least {required} samples, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    #[error("All features have zero importance")]
    ZeroImportance,

    #[error("Gradient computation failed: {reason}")]
    GradientComputation { reason: String },

    #[error("Tree learning failed at iteration {iteration}: {reason}")]
    TreeLearning { iteration: usize, reason: String },

    #[error("Objective function error: {reason}")]
    ObjectiveFunction { reason: String },
}

/// GPU computation errors
#[derive(Error, Debug)]
pub enum GPUError {
    #[error("GPU device not available")]
    DeviceNotAvailable,

    #[error("CUDA error: {code}")]
    CUDA { code: i32 },

    #[error("OpenCL error: {message}")]
    OpenCL { message: String },

    #[error("GPU memory allocation failed: requested {size} bytes")]
    MemoryAllocation { size: usize },

    #[error("GPU kernel execution failed: {kernel}")]
    KernelExecution { kernel: String },

    #[error("GPU memory transfer failed: {direction}")]
    MemoryTransfer { direction: String },
}

/// Memory-related errors
#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Allocation failed: requested {size} bytes")]
    AllocationFailed { size: usize },

    #[error("Alignment constraint violated: address {address:#x}, required alignment {alignment}")]
    AlignmentViolation { address: usize, alignment: usize },

    #[error("Buffer overflow: capacity {capacity}, attempted write at {offset}")]
    BufferOverflow { capacity: usize, offset: usize },

    #[error("Out of memory: available {available} bytes, requested {requested} bytes")]
    OutOfMemory { available: usize, requested: usize },
}

/// Type alias for Results using LightGBMError
pub type Result<T> = std::result::Result<T, LightGBMError>;

/// Utility functions for error handling
impl LightGBMError {
    /// Create a configuration error
    pub fn config<S: Into<String>>(message: S) -> Self {
        LightGBMError::Config {
            message: message.into(),
        }
    }

    /// Create a dataset error
    pub fn dataset<S: Into<String>>(message: S) -> Self {
        LightGBMError::Dataset {
            message: message.into(),
        }
    }

    /// Create a data validation error (alias for dataset error)
    pub fn data_validation<S: Into<String>>(message: S) -> Self {
        LightGBMError::Dataset {
            message: message.into(),
        }
    }

    /// Create a data dimension mismatch error
    pub fn data_dimension_mismatch<S: Into<String>>(message: S) -> Self {
        LightGBMError::DataDimensionMismatch {
            message: message.into(),
        }
    }

    /// Create a data loading error
    pub fn data_loading<S: Into<String>>(message: S) -> Self {
        LightGBMError::DataLoading {
            message: message.into(),
        }
    }

    /// Create a training error
    pub fn training<S: Into<String>>(message: S) -> Self {
        LightGBMError::Training {
            message: message.into(),
        }
    }

    /// Create a prediction error
    pub fn prediction<S: Into<String>>(message: S) -> Self {
        LightGBMError::Prediction {
            message: message.into(),
        }
    }

    /// Create a GPU error
    pub fn gpu<S: Into<String>>(message: S) -> Self {
        LightGBMError::GPU {
            message: message.into(),
        }
    }

    /// Create a memory error
    pub fn memory<S: Into<String>>(message: S) -> Self {
        LightGBMError::Memory {
            message: message.into(),
        }
    }

    /// Create a numerical error
    pub fn numerical<S: Into<String>>(message: S) -> Self {
        LightGBMError::Numerical {
            message: message.into(),
        }
    }

    /// Create an invalid parameter error
    pub fn invalid_parameter<P, V, R>(parameter: P, value: V, reason: R) -> Self
    where
        P: Into<String>,
        V: Into<String>,
        R: Into<String>,
    {
        LightGBMError::InvalidParameter {
            parameter: parameter.into(),
            value: value.into(),
            reason: reason.into(),
        }
    }

    /// Create a dimension mismatch error
    pub fn dimension_mismatch<E, A>(expected: E, actual: A) -> Self
    where
        E: Into<String>,
        A: Into<String>,
    {
        LightGBMError::DimensionMismatch {
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Create an index out of bounds error
    pub fn index_out_of_bounds(index: usize, length: usize) -> Self {
        LightGBMError::IndexOutOfBounds { index, length }
    }

    /// Create an internal error (should be used sparingly)
    pub fn internal<S: Into<String>>(message: S) -> Self {
        LightGBMError::Internal {
            message: message.into(),
        }
    }

    /// Create a not implemented error
    pub fn not_implemented<S: Into<String>>(feature: S) -> Self {
        LightGBMError::NotImplemented {
            feature: feature.into(),
        }
    }

    /// Create a serialization error
    pub fn serialization<S: Into<String>>(message: S) -> Self {
        LightGBMError::Serialization {
            message: message.into(),
        }
    }

    /// Create an IO error with custom message
    pub fn io_error<S: Into<String>>(message: S) -> Self {
        LightGBMError::IO {
            source: std::io::Error::new(std::io::ErrorKind::Other, message.into()),
        }
    }

    /// Create a model error
    pub fn model<S: Into<String>>(message: S) -> Self {
        LightGBMError::Training {
            message: message.into(),
        }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            LightGBMError::Config { .. } => false,
            LightGBMError::Dataset { .. } => false,
            LightGBMError::DataDimensionMismatch { .. } => false,
            LightGBMError::DataLoading { .. } => false,
            LightGBMError::FeatureProcessing { .. } => true,
            LightGBMError::Training { .. } => true,
            LightGBMError::TreeConstruction { .. } => true,
            LightGBMError::Prediction { .. } => true,
            LightGBMError::GPU { .. } => true,
            LightGBMError::Memory { .. } => false,
            LightGBMError::Numerical { .. } => true,
            LightGBMError::Serialization { .. } => false,
            LightGBMError::IO { .. } => false,
            LightGBMError::Csv { .. } => false,
            LightGBMError::Json { .. } => false,
            LightGBMError::Bincode { .. } => false,
            LightGBMError::Threading { .. } => true,
            LightGBMError::InvalidParameter { .. } => false,
            LightGBMError::DimensionMismatch { .. } => false,
            LightGBMError::IndexOutOfBounds { .. } => false,
            LightGBMError::EarlyStopping { .. } => true,
            LightGBMError::NotImplemented { .. } => false,
            LightGBMError::Internal { .. } => false,
        }
    }

    /// Get error category for logging and metrics
    pub fn category(&self) -> &'static str {
        match self {
            LightGBMError::Config { .. } => "config",
            LightGBMError::Dataset { .. } => "dataset",
            LightGBMError::DataDimensionMismatch { .. } => "data_dimension_mismatch",
            LightGBMError::DataLoading { .. } => "data_loading",
            LightGBMError::FeatureProcessing { .. } => "feature_processing",
            LightGBMError::Training { .. } => "training",
            LightGBMError::TreeConstruction { .. } => "tree_construction",
            LightGBMError::Prediction { .. } => "prediction",
            LightGBMError::GPU { .. } => "gpu",
            LightGBMError::Memory { .. } => "memory",
            LightGBMError::Numerical { .. } => "numerical",
            LightGBMError::Serialization { .. } => "serialization",
            LightGBMError::IO { .. } => "io",
            LightGBMError::Csv { .. } => "csv",
            LightGBMError::Json { .. } => "json",
            LightGBMError::Bincode { .. } => "bincode",
            LightGBMError::Threading { .. } => "threading",
            LightGBMError::InvalidParameter { .. } => "invalid_parameter",
            LightGBMError::DimensionMismatch { .. } => "dimension_mismatch",
            LightGBMError::IndexOutOfBounds { .. } => "index_out_of_bounds",
            LightGBMError::EarlyStopping { .. } => "early_stopping",
            LightGBMError::NotImplemented { .. } => "not_implemented",
            LightGBMError::Internal { .. } => "internal",
        }
    }
}

/// Convert specialized error types to LightGBMError
impl From<DatasetError> for LightGBMError {
    fn from(err: DatasetError) -> Self {
        LightGBMError::Dataset {
            message: err.to_string(),
        }
    }
}

impl From<TrainingError> for LightGBMError {
    fn from(err: TrainingError) -> Self {
        LightGBMError::Training {
            message: err.to_string(),
        }
    }
}

impl From<GPUError> for LightGBMError {
    fn from(err: GPUError) -> Self {
        LightGBMError::GPU {
            message: err.to_string(),
        }
    }
}

impl From<MemoryError> for LightGBMError {
    fn from(err: MemoryError) -> Self {
        LightGBMError::Memory {
            message: err.to_string(),
        }
    }
}

/// Convenience macros for error creation
#[macro_export]
macro_rules! config_error {
    ($msg:expr) => {
        $crate::core::error::LightGBMError::config($msg)
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::core::error::LightGBMError::config(format!($fmt, $($arg)*))
    };
}

#[macro_export]
macro_rules! dataset_error {
    ($msg:expr) => {
        $crate::core::error::LightGBMError::dataset($msg)
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::core::error::LightGBMError::dataset(format!($fmt, $($arg)*))
    };
}

#[macro_export]
macro_rules! training_error {
    ($msg:expr) => {
        $crate::core::error::LightGBMError::training($msg)
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::core::error::LightGBMError::training(format!($fmt, $($arg)*))
    };
}

#[macro_export]
macro_rules! ensure {
    ($cond:expr, $err:expr) => {
        if !($cond) {
            return Err($err.into());
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = LightGBMError::config("test configuration error");
        assert_eq!(err.category(), "config");
        assert!(!err.is_recoverable());

        let err = LightGBMError::training("test training error");
        assert_eq!(err.category(), "training");
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_error_macros() {
        let err = config_error!("test error");
        assert!(matches!(err, LightGBMError::Config { .. }));

        let err = dataset_error!("test error with param: {}", 42);
        assert!(matches!(err, LightGBMError::Dataset { .. }));
    }

    #[test]
    fn test_specialized_errors() {
        let dataset_err = DatasetError::Empty;
        let lightgbm_err: LightGBMError = dataset_err.into();
        assert!(matches!(lightgbm_err, LightGBMError::Dataset { .. }));

        let training_err = TrainingError::ConvergenceFailed { iterations: 100 };
        let lightgbm_err: LightGBMError = training_err.into();
        assert!(matches!(lightgbm_err, LightGBMError::Training { .. }));
    }

    #[test]
    fn test_parameter_errors() {
        let err = LightGBMError::invalid_parameter("learning_rate", "-0.5", "must be positive");
        assert_eq!(err.category(), "invalid_parameter");
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_dimension_mismatch() {
        let err = LightGBMError::dimension_mismatch("(100, 10)", "(100, 5)");
        assert_eq!(err.category(), "dimension_mismatch");
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_error_display() {
        let err = LightGBMError::config("test message");
        let error_string = format!("{}", err);
        assert!(error_string.contains("Configuration error"));
        assert!(error_string.contains("test message"));
    }

    #[test]
    fn test_error_debug() {
        let err = LightGBMError::internal("debug test");
        let debug_string = format!("{:?}", err);
        assert!(debug_string.contains("Internal"));
        assert!(debug_string.contains("debug test"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let lightgbm_err: LightGBMError = io_err.into();
        assert!(matches!(lightgbm_err, LightGBMError::IO { .. }));
        assert_eq!(lightgbm_err.category(), "io");
    }
}
