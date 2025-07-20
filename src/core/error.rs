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
    Config {
        /// Error message describing the configuration issue
        message: String
    },

    /// Dataset-related errors
    #[error("Dataset error: {message}")]
    Dataset {
        /// Error message describing the dataset issue
        message: String
    },

    /// Data dimension mismatch errors
    #[error("Data dimension mismatch: {message}")]
    DataDimensionMismatch {
        /// Error message describing the dimension mismatch
        message: String
    },

    /// Data loading and parsing errors
    #[error("Data loading error: {message}")]
    DataLoading {
        /// Error message describing the data loading issue
        message: String
    },

    /// Feature processing errors
    #[error("Feature processing error: {message}")]
    FeatureProcessing {
        /// Error message describing the feature processing issue
        message: String
    },

    /// Training-related errors
    #[error("Training error: {message}")]
    Training {
        /// Error message describing the training issue
        message: String
    },

    /// Tree construction errors
    #[error("Tree construction error: {message}")]
    TreeConstruction {
        /// Error message describing the tree construction issue
        message: String
    },

    /// Prediction errors
    #[error("Prediction error: {message}")]
    Prediction {
        /// Error message describing the prediction issue
        message: String
    },

    /// GPU/CUDA computation errors
    #[error("GPU computation error: {message}")]
    GPU {
        /// Error message describing the GPU computation issue
        message: String
    },

    /// Memory allocation and management errors
    #[error("Memory error: {message}")]
    Memory {
        /// Error message describing the memory issue
        message: String
    },

    /// Numerical computation errors (overflow, underflow, NaN)
    #[error("Numerical error: {message}")]
    Numerical {
        /// Error message describing the numerical computation issue
        message: String
    },

    /// Model serialization/deserialization errors
    #[error("Serialization error: {message}")]
    Serialization {
        /// Error message describing the serialization issue
        message: String
    },

    /// File I/O errors
    #[error("I/O error: {source}")]
    IO {
        /// The underlying I/O error
        #[from]
        source: io::Error,
    },

    /// CSV parsing errors
    #[error("CSV parsing error: {source}")]
    Csv {
        /// The underlying CSV parsing error
        #[from]
        source: csv::Error,
    },

    /// JSON serialization errors
    #[error("JSON error: {source}")]
    Json {
        /// The underlying JSON serialization error
        #[from]
        source: serde_json::Error,
    },

    /// Bincode serialization errors
    #[error("Bincode error: {source}")]
    Bincode {
        /// The underlying bincode serialization error
        #[from]
        source: bincode::Error,
    },

    /// Thread synchronization errors
    #[error("Threading error: {message}")]
    Threading { 
        /// Error message describing the threading issue
        message: String 
    },

    /// Invalid input parameters
    #[error("Invalid parameter: {parameter} = {value}, {reason}")]
    InvalidParameter {
        /// Name of the invalid parameter
        parameter: String,
        /// Value that was provided
        value: String,
        /// Reason why the parameter is invalid
        reason: String,
    },

    /// Dimension mismatch errors
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { 
        /// Expected dimension format
        expected: String, 
        /// Actual dimension format received
        actual: String 
    },

    /// Out of bounds access
    #[error("Index out of bounds: index {index}, length {length}")]
    IndexOutOfBounds { 
        /// The index that was accessed
        index: usize, 
        /// The maximum valid length
        length: usize 
    },

    /// Early stopping conditions
    #[error("Early stopping triggered: {reason}")]
    EarlyStopping { 
        /// Reason for early stopping
        reason: String 
    },

    /// Not implemented functionality
    #[error("Not implemented: {feature}")]
    NotImplemented { 
        /// Name of the feature that is not implemented
        feature: String 
    },

    /// Internal library errors (should not occur in normal usage)
    #[error("Internal error: {message}")]
    Internal { 
        /// Internal error message
        message: String 
    },
}

/// Specialized error types for specific components
#[derive(Error, Debug)]
pub enum DatasetError {
    /// Empty dataset was provided
    #[error("Empty dataset provided")]
    Empty,

    /// Feature count mismatch between expected and actual
    #[error("Feature count mismatch: expected {expected}, got {actual}")]
    FeatureMismatch { 
        /// Expected number of features
        expected: usize, 
        /// Actual number of features
        actual: usize 
    },

    /// Invalid feature type detected
    #[error("Invalid feature type for feature {index}: expected {expected:?}, got {actual:?}")]
    InvalidFeatureType {
        /// Index of the feature with invalid type
        index: usize,
        /// Expected feature type
        expected: crate::core::types::FeatureType,
        /// Actual feature type found
        actual: crate::core::types::FeatureType,
    },

    /// Missing values are not supported for this feature
    #[error("Missing values not supported for feature {index}")]
    MissingValuesNotSupported { 
        /// Index of the feature that has unsupported missing values
        index: usize 
    },

    /// Categorical feature has too many categories
    #[error("Categorical feature {index} has too many categories: {count} > {max}")]
    TooManyCategories {
        /// Index of the categorical feature
        index: usize,
        /// Number of categories found
        count: usize,
        /// Maximum allowed categories
        max: usize,
    },

    /// Feature contains an invalid value
    #[error("Feature {index} has invalid value: {value}")]
    InvalidFeatureValue { 
        /// Index of the feature with invalid value
        index: usize, 
        /// The invalid value
        value: f64 
    },
}

/// Training-specific errors
#[derive(Error, Debug)]
pub enum TrainingError {
    /// Training failed to converge within the maximum iterations
    #[error("Convergence failed after {iterations} iterations")]
    ConvergenceFailed { 
        /// Number of iterations attempted before failure
        iterations: usize 
    },

    /// Insufficient data provided for training
    #[error("Insufficient data: need at least {required} samples, got {actual}")]
    InsufficientData { 
        /// Minimum required number of samples
        required: usize, 
        /// Actual number of samples provided
        actual: usize 
    },

    /// All features have zero importance, cannot continue training
    #[error("All features have zero importance")]
    ZeroImportance,

    /// Error in gradient computation
    #[error("Gradient computation failed: {reason}")]
    GradientComputation { 
        /// Reason for gradient computation failure
        reason: String 
    },

    /// Error during tree learning process
    #[error("Tree learning failed at iteration {iteration}: {reason}")]
    TreeLearning { 
        /// Iteration at which tree learning failed
        iteration: usize, 
        /// Reason for tree learning failure
        reason: String 
    },

    /// Error in objective function computation
    #[error("Objective function error: {reason}")]
    ObjectiveFunction { 
        /// Reason for objective function error
        reason: String 
    },
}

/// GPU computation errors
#[derive(Error, Debug)]
pub enum GPUError {
    /// No GPU device is available for computation
    #[error("GPU device not available")]
    DeviceNotAvailable,

    /// CUDA runtime error
    #[error("CUDA error: {code}")]
    CUDA { 
        /// CUDA error code
        code: i32 
    },

    /// OpenCL computation error
    #[error("OpenCL error: {message}")]
    OpenCL { 
        /// OpenCL error message
        message: String 
    },

    /// GPU memory allocation failed
    #[error("GPU memory allocation failed: requested {size} bytes")]
    MemoryAllocation { 
        /// Size of memory allocation that failed
        size: usize 
    },

    /// GPU kernel execution failed
    #[error("GPU kernel execution failed: {kernel}")]
    KernelExecution { 
        /// Name of the kernel that failed
        kernel: String 
    },

    /// GPU memory transfer operation failed
    #[error("GPU memory transfer failed: {direction}")]
    MemoryTransfer { 
        /// Direction of memory transfer (host-to-device, device-to-host, etc.)
        direction: String 
    },
}

/// Memory-related errors
#[derive(Error, Debug)]
pub enum MemoryError {
    /// Memory allocation failed
    #[error("Allocation failed: requested {size} bytes")]
    AllocationFailed { 
        /// Size of the failed allocation in bytes
        size: usize 
    },

    /// Memory alignment constraint violation
    #[error("Alignment constraint violated: address {address:#x}, required alignment {alignment}")]
    AlignmentViolation { 
        /// Memory address that violates alignment
        address: usize, 
        /// Required memory alignment
        alignment: usize 
    },

    /// Buffer overflow detected
    #[error("Buffer overflow: capacity {capacity}, attempted write at {offset}")]
    BufferOverflow { 
        /// Buffer capacity
        capacity: usize, 
        /// Offset where overflow was attempted
        offset: usize 
    },

    /// System is out of memory
    #[error("Out of memory: available {available} bytes, requested {requested} bytes")]
    OutOfMemory { 
        /// Available memory in bytes
        available: usize, 
        /// Requested memory in bytes
        requested: usize 
    },
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

/// Macro for creating dataset errors with formatted messages
#[macro_export]
macro_rules! dataset_error {
    ($msg:expr) => {
        $crate::core::error::LightGBMError::dataset($msg)
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::core::error::LightGBMError::dataset(format!($fmt, $($arg)*))
    };
}

/// Macro for creating training errors with formatted messages
#[macro_export]
macro_rules! training_error {
    ($msg:expr) => {
        $crate::core::error::LightGBMError::training($msg)
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::core::error::LightGBMError::training(format!($fmt, $($arg)*))
    };
}

/// Macro for conditional error handling - returns an error if condition is false
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
