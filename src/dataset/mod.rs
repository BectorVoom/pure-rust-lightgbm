//! Dataset management module for Pure Rust LightGBM.
//!
//! This module provides comprehensive data loading, preprocessing, and management
//! capabilities for LightGBM training and prediction. It supports multiple data
//! formats and provides efficient data structures optimized for gradient boosting.

pub mod binning;
pub mod dataset;
pub mod loader;
pub mod partition;
pub mod preprocessing;

// Re-export commonly used types
pub use binning::{BinMapper, BinType, FeatureBinner, MissingType};
pub use dataset::{Dataset, DatasetBuilder, DatasetInfo, DatasetMetadata};
pub use loader::{DataLoader, LoaderConfig, LoaderError};
pub use partition::{DataPartition, MemoryOptimization, PartitionConfig};
pub use preprocessing::{Preprocessor, PreprocessorConfig, PreprocessorError};

use crate::core::constants::*;
use crate::core::error::{LightGBMError, Result};
use crate::core::types::*;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use std::path::Path;

/// Dataset configuration structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Target column name (for supervised learning)
    pub target_column: Option<String>,
    /// Feature columns to use (None means all except target)
    pub feature_columns: Option<Vec<String>>,
    /// Weight column name (for weighted learning)
    pub weight_column: Option<String>,
    /// Group column name (for ranking)
    pub group_column: Option<String>,
    /// Maximum number of bins for feature discretization
    pub max_bin: usize,
    /// Minimum data points per bin
    pub min_data_per_bin: usize,
    /// Use missing value as zero
    pub use_missing_as_zero: bool,
    /// Force column-wise binning
    pub force_col_wise: bool,
    /// Force row-wise binning
    pub force_row_wise: bool,
    /// Pre-filter features with low variance
    pub pre_filter: bool,
    /// Categorical feature indices
    pub categorical_features: Option<Vec<usize>>,
    /// Feature names
    pub feature_names: Option<Vec<String>>,
    /// Enable two-round loading for large datasets
    pub two_round: bool,
    /// Save binary dataset file
    pub save_binary: bool,
    /// Binary dataset file path
    pub binary_file: Option<String>,
    /// Enable histogram construction caching
    pub enable_histogram_cache: bool,
    /// Memory limit for dataset (MB)
    pub memory_limit_mb: Option<usize>,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        DatasetConfig {
            target_column: None,
            feature_columns: None,
            weight_column: None,
            group_column: None,
            max_bin: DEFAULT_MAX_BIN,
            min_data_per_bin: 3,
            use_missing_as_zero: false,
            force_col_wise: false,
            force_row_wise: false,
            pre_filter: false,
            categorical_features: None,
            feature_names: None,
            two_round: false,
            save_binary: false,
            binary_file: None,
            enable_histogram_cache: true,
            memory_limit_mb: None,
        }
    }
}

impl DatasetConfig {
    /// Create a new dataset configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set target column name
    pub fn with_target_column<S: Into<String>>(mut self, target: S) -> Self {
        self.target_column = Some(target.into());
        self
    }

    /// Set feature columns
    pub fn with_feature_columns(mut self, features: Vec<String>) -> Self {
        self.feature_columns = Some(features);
        self
    }

    /// Set weight column name
    pub fn with_weight_column<S: Into<String>>(mut self, weight: S) -> Self {
        self.weight_column = Some(weight.into());
        self
    }

    /// Set maximum number of bins
    pub fn with_max_bin(mut self, max_bin: usize) -> Self {
        self.max_bin = max_bin;
        self
    }

    /// Set minimum data points per bin
    pub fn with_min_data_in_bin(mut self, min_data: usize) -> Self {
        self.min_data_per_bin = min_data;
        self
    }

    /// Set categorical feature indices
    pub fn with_categorical_features(mut self, categorical: Vec<usize>) -> Self {
        self.categorical_features = Some(categorical);
        self
    }

    /// Enable two-round loading
    pub fn with_two_round(mut self, two_round: bool) -> Self {
        self.two_round = two_round;
        self
    }

    /// Set memory limit
    pub fn with_memory_limit(mut self, limit_mb: usize) -> Self {
        self.memory_limit_mb = Some(limit_mb);
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.max_bin < 2 {
            return Err(LightGBMError::invalid_parameter(
                "max_bin",
                self.max_bin.to_string(),
                "must be at least 2",
            ));
        }

        if self.max_bin > 65535 {
            return Err(LightGBMError::invalid_parameter(
                "max_bin",
                self.max_bin.to_string(),
                "cannot exceed 65535",
            ));
        }

        if self.min_data_per_bin < 1 {
            return Err(LightGBMError::invalid_parameter(
                "min_data_per_bin",
                self.min_data_per_bin.to_string(),
                "must be at least 1",
            ));
        }

        if self.force_col_wise && self.force_row_wise {
            return Err(LightGBMError::config(
                "Cannot force both column-wise and row-wise binning",
            ));
        }

        if let Some(memory_limit) = self.memory_limit_mb {
            if memory_limit == 0 {
                return Err(LightGBMError::invalid_parameter(
                    "memory_limit_mb",
                    "0".to_string(),
                    "must be positive",
                ));
            }
        }

        Ok(())
    }
}

/// Dataset factory for creating datasets from various sources
pub struct DatasetFactory;

impl DatasetFactory {
    /// Create dataset from CSV file
    pub fn from_csv<P: AsRef<Path>>(path: P, config: DatasetConfig) -> Result<Dataset> {
        println!("DEBUG: DatasetFactory::from_csv called");
        config.validate()?;
        let loader = loader::CsvLoader::new(config.clone())?;
        loader.load(path)
    }

    /// Create dataset from Polars DataFrame
    #[cfg(feature = "polars")]
    pub fn from_polars(df: &polars::prelude::DataFrame, config: DatasetConfig) -> Result<Dataset> {
        config.validate()?;
        let loader = loader::PolarsLoader::new(config.clone())?;
        loader.load_dataframe(df)
    }

    /// Create dataset from Arrow Table
    #[cfg(feature = "arrow")]
    pub fn from_arrow(table: &arrow::array::RecordBatch, config: DatasetConfig) -> Result<Dataset> {
        config.validate()?;
        let loader = loader::ArrowLoader::new(config.clone())?;
        loader.load_table(table)
    }

    /// Create dataset from Parquet file
    #[cfg(feature = "parquet")]
    pub fn from_parquet<P: AsRef<Path>>(path: P, config: DatasetConfig) -> Result<Dataset> {
        config.validate()?;
        let loader = loader::ParquetLoader::new(config.clone())?;
        loader.load(path)
    }

    /// Create dataset from numpy-style arrays
    pub fn from_arrays(
        features: Array2<f32>,
        labels: Array1<f32>,
        weights: Option<Array1<f32>>,
        config: DatasetConfig,
    ) -> Result<Dataset> {
        config.validate()?;
        let loader = loader::ArrayLoader::new(config.clone())?;
        loader.load_arrays(features, labels, weights)
    }

    /// Create dataset from binary file
    pub fn from_binary<P: AsRef<Path>>(path: P, config: DatasetConfig) -> Result<Dataset> {
        config.validate()?;
        let loader = loader::BinaryLoader::new(config.clone())?;
        loader.load(path)
    }

    /// Create dataset from memory-mapped file
    pub fn from_memory_mapped<P: AsRef<Path>>(path: P, config: DatasetConfig) -> Result<Dataset> {
        config.validate()?;
        let loader = loader::MemoryMappedLoader::new(config.clone())?;
        loader.load(path)
    }
}

/// Dataset statistics for analysis and validation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetStatistics {
    /// Number of samples
    pub num_samples: usize,
    /// Number of features
    pub num_features: usize,
    /// Number of classes (for classification)
    pub num_classes: Option<usize>,
    /// Feature statistics
    pub feature_stats: Vec<FeatureStatistics>,
    /// Missing value counts
    pub missing_counts: Vec<usize>,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Sparsity ratio
    pub sparsity: f64,
}

impl Default for DatasetStatistics {
    fn default() -> Self {
        DatasetStatistics {
            num_samples: 0,
            num_features: 0,
            num_classes: None,
            feature_stats: Vec::new(),
            missing_counts: Vec::new(),
            memory_usage: 0,
            sparsity: 0.0,
        }
    }
}

/// Statistics for individual features
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureStatistics {
    /// Feature name
    pub name: String,
    /// Feature type
    pub feature_type: FeatureType,
    /// Minimum value
    pub min_value: f64,
    /// Maximum value
    pub max_value: f64,
    /// Mean value
    pub mean_value: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Number of unique values
    pub num_unique: usize,
    /// Missing value count
    pub missing_count: usize,
    /// Histogram of values
    pub histogram: Vec<usize>,
}

/// Dataset validation result
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetValidationResult {
    /// Validation passed
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Suggested fixes
    pub suggestions: Vec<String>,
}

/// Dataset utilities
pub mod utils {
    use super::*;

    /// Detect feature types automatically
    pub fn detect_feature_types(features: &Array2<f32>) -> Vec<FeatureType> {
        let mut types = Vec::new();

        for col in 0..features.ncols() {
            let column = features.column(col);
            let unique_values: std::collections::HashSet<_> = column
                .iter()
                .filter(|&&x| !x.is_nan())
                .map(|&x| x as i32)
                .collect();

            // If all values are integers and there are few unique values, it's likely categorical
            if unique_values.len() <= 20 && column.iter().all(|&x| x.is_nan() || x.fract() == 0.0) {
                types.push(FeatureType::Categorical);
            } else {
                types.push(FeatureType::Numerical);
            }
        }

        types
    }

    /// Calculate dataset statistics
    pub fn calculate_statistics(dataset: &Dataset) -> DatasetStatistics {
        let num_samples = dataset.num_data();
        let num_features = dataset.num_features();
        let memory_usage = dataset.memory_usage();

        let mut feature_stats = Vec::new();
        let mut missing_counts = Vec::new();

        for feature_idx in 0..num_features {
            let feature_data = dataset.feature_data(feature_idx);
            let feature_name = dataset
                .feature_name(feature_idx)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("feature_{}", feature_idx));

            let (min_val, max_val, mean_val, std_dev, missing_count) =
                calculate_feature_statistics(&feature_data);

            let unique_values: std::collections::HashSet<_> = feature_data
                .iter()
                .filter(|&&x| !x.is_nan())
                .map(|&x| x as i32)
                .collect();

            feature_stats.push(FeatureStatistics {
                name: feature_name,
                feature_type: if unique_values.len() <= 20 {
                    FeatureType::Categorical
                } else {
                    FeatureType::Numerical
                },
                min_value: min_val,
                max_value: max_val,
                mean_value: mean_val,
                std_dev,
                num_unique: unique_values.len(),
                missing_count,
                histogram: vec![], // TODO: Implement histogram calculation
            });

            missing_counts.push(missing_count);
        }

        let total_values = num_samples * num_features;
        let total_missing: usize = missing_counts.iter().sum();
        let sparsity = total_missing as f64 / total_values as f64;

        DatasetStatistics {
            num_samples,
            num_features,
            num_classes: None, // TODO: Detect from labels
            feature_stats,
            missing_counts,
            memory_usage,
            sparsity,
        }
    }

    /// Calculate statistics for a single feature
    pub fn calculate_feature_statistics(data: &[f32]) -> (f64, f64, f64, f64, usize) {
        let valid_values: Vec<f32> = data.iter().copied().filter(|x| !x.is_nan()).collect();

        if valid_values.is_empty() {
            return (0.0, 0.0, 0.0, 0.0, data.len());
        }

        let min_val = valid_values.iter().fold(f32::INFINITY, |a, &b| a.min(b)) as f64;
        let max_val = valid_values
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
        let mean_val = valid_values.iter().sum::<f32>() as f64 / valid_values.len() as f64;

        let variance = valid_values
            .iter()
            .map(|&x| (x as f64 - mean_val).powi(2))
            .sum::<f64>()
            / valid_values.len() as f64;
        let std_dev = variance.sqrt();

        let missing_count = data.len() - valid_values.len();

        (min_val, max_val, mean_val, std_dev, missing_count)
    }

    /// Validate dataset for training
    pub fn validate_dataset(dataset: &Dataset) -> DatasetValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();

        // Check minimum data requirements
        if dataset.num_data() < 10 {
            errors.push("Dataset has fewer than 10 samples".to_string());
            suggestions.push("Collect more training data".to_string());
        }

        // Check feature requirements
        if dataset.num_features() == 0 {
            errors.push("Dataset has no features".to_string());
        }

        // Check for too many missing values
        let stats = calculate_statistics(dataset);
        if stats.sparsity > 0.8 {
            warnings.push(format!("Dataset has high sparsity ({:.2})", stats.sparsity));
            suggestions
                .push("Consider imputing missing values or removing sparse features".to_string());
        }

        // Check for constant features
        for (i, feature_stat) in stats.feature_stats.iter().enumerate() {
            if feature_stat.std_dev < 1e-10 {
                warnings.push(format!("Feature {} has constant values", i));
                suggestions.push(format!("Consider removing feature {}", i));
            }
        }

        DatasetValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            suggestions,
        }
    }

    /// Convert dataset to different format
    pub fn convert_dataset(dataset: &Dataset, format: &str) -> Result<Vec<u8>> {
        match format {
            "csv" => {
                // TODO: Implement CSV conversion
                Err(LightGBMError::not_implemented("CSV conversion"))
            }
            "json" => {
                // TODO: Implement JSON conversion
                Err(LightGBMError::not_implemented("JSON conversion"))
            }
            "parquet" => {
                // TODO: Implement Parquet conversion
                Err(LightGBMError::not_implemented("Parquet conversion"))
            }
            _ => Err(LightGBMError::invalid_parameter(
                "format",
                format.to_string(),
                "supported formats: csv, json, parquet",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_config_default() {
        let config = DatasetConfig::default();
        assert_eq!(config.max_bin, DEFAULT_MAX_BIN);
        assert_eq!(config.min_data_per_bin, 3);
        assert!(!config.use_missing_as_zero);
        assert!(!config.force_col_wise);
        assert!(!config.force_row_wise);
    }

    #[test]
    fn test_dataset_config_builder() {
        let config = DatasetConfig::new()
            .with_target_column("target")
            .with_max_bin(128)
            .with_categorical_features(vec![0, 1, 2])
            .with_two_round(true);

        assert_eq!(config.target_column, Some("target".to_string()));
        assert_eq!(config.max_bin, 128);
        assert_eq!(config.categorical_features, Some(vec![0, 1, 2]));
        assert!(config.two_round);
    }

    #[test]
    fn test_dataset_config_validation() {
        let mut config = DatasetConfig::default();
        assert!(config.validate().is_ok());

        config.max_bin = 1;
        assert!(config.validate().is_err());

        config.max_bin = 128;
        config.force_col_wise = true;
        config.force_row_wise = true;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_feature_type_detection() {
        // Create features where column 0 is clearly numerical (many unique values or non-integers)
        // and column 1 is clearly categorical (few unique integer values)
        let features = Array2::from_shape_vec(
            (6, 2), 
            vec![
                1.5, 1.0,   // Row 0: [1.5, 1.0]
                2.7, 1.0,   // Row 1: [2.7, 1.0] 
                3.2, 2.0,   // Row 2: [3.2, 2.0]
                4.8, 2.0,   // Row 3: [4.8, 2.0]
                5.1, 1.0,   // Row 4: [5.1, 1.0]
                6.9, 2.0,   // Row 5: [6.9, 2.0]
            ]
        ).unwrap();

        let types = utils::detect_feature_types(&features);
        assert_eq!(types.len(), 2);
        // Column 0: [1.5, 2.7, 3.2, 4.8, 5.1, 6.9] - has non-integer values, should be Numerical
        assert_eq!(types[0], FeatureType::Numerical);
        // Column 1: [1.0, 1.0, 2.0, 2.0, 1.0, 2.0] - only 2 unique integer values, should be Categorical
        assert_eq!(types[1], FeatureType::Categorical);
    }

    #[test]
    fn test_feature_statistics() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, f32::NAN];
        let (min_val, max_val, mean_val, std_dev, missing_count) =
            utils::calculate_feature_statistics(&data);

        assert_eq!(min_val, 1.0);
        assert_eq!(max_val, 5.0);
        assert_eq!(mean_val, 3.0);
        assert!(std_dev > 0.0);
        assert_eq!(missing_count, 1);
    }
}
