//! Feature binning system for Pure Rust LightGBM.
//!
//! This module provides feature discretization (binning) capabilities that are
//! essential for efficient histogram construction in gradient boosting. It supports
//! both numerical and categorical features with various binning strategies.

pub mod categorical;
pub mod mapper;
pub mod numerical;

// Re-export commonly used types
pub use categorical::CategoricalBinner;
pub use mapper::{BinConfig, BinMapper, BinType, MissingType};
pub use numerical::NumericalBinner;

use crate::core::constants::*;
use crate::core::error::{LightGBMError, Result};
use crate::core::types::*;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Feature binner that handles both numerical and categorical features
#[derive(Debug)]
pub struct FeatureBinner {
    /// Binning configuration
    config: BinningConfig,
    /// Maximum number of bins per feature
    // TODO: implement max_bins usage in binning algorithm
    max_bins: usize,
    /// Minimum data points per bin
    // TODO: implement min_data_per_bin enforcement in binning
    min_data_per_bin: usize,
    /// Bin mappers for each feature
    bin_mappers: Vec<BinMapper>,
    /// Feature types
    feature_types: Vec<FeatureType>,
    /// Missing value handling
    missing_types: Vec<MissingType>,
}

/// Configuration for the binning process
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinningConfig {
    /// Maximum number of bins per feature
    pub max_bins: usize,
    /// Minimum data points per bin
    pub min_data_per_bin: usize,
    /// Binning strategy
    pub strategy: BinningStrategy,
    /// Missing value handling
    pub missing_handling: MissingHandling,
    /// Enable sparse optimization
    pub sparse_optimization: bool,
    /// Categorical feature indices
    pub categorical_features: Option<Vec<usize>>,
    /// Force column-wise binning
    pub force_col_wise: bool,
    /// Force row-wise binning
    pub force_row_wise: bool,
    /// Enable pre-filtering
    pub pre_filter: bool,
    /// Memory limit for binning (MB)
    pub memory_limit_mb: Option<usize>,
}

impl Default for BinningConfig {
    fn default() -> Self {
        BinningConfig {
            max_bins: DEFAULT_MAX_BIN,
            min_data_per_bin: 3,
            strategy: BinningStrategy::Quantile,
            missing_handling: MissingHandling::Zero,
            sparse_optimization: false,
            categorical_features: None,
            force_col_wise: false,
            force_row_wise: false,
            pre_filter: false,
            memory_limit_mb: None,
        }
    }
}

/// Binning strategy enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinningStrategy {
    /// Uniform binning (equal-width bins)
    Uniform,
    /// Quantile binning (equal-frequency bins)
    Quantile,
    /// Adaptive binning (optimizes for split quality)
    Adaptive,
    /// Custom binning (user-defined bin boundaries)
    Custom,
}

/// Missing value handling strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissingHandling {
    /// Treat missing values as zero
    Zero,
    /// Treat missing values as NaN
    NaN,
    /// No missing values expected
    None,
    /// Missing values get a separate bin
    Separate,
}

impl From<MissingHandling> for MissingType {
    fn from(handling: MissingHandling) -> Self {
        match handling {
            MissingHandling::Zero => MissingType::Zero,
            MissingHandling::NaN => MissingType::NaN,
            MissingHandling::None => MissingType::None,
            MissingHandling::Separate => MissingType::Separate,
        }
    }
}

/// Binning statistics for analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinningStatistics {
    /// Number of features processed
    pub num_features: usize,
    /// Number of numerical features
    pub num_numerical: usize,
    /// Number of categorical features
    pub num_categorical: usize,
    /// Average number of bins per feature
    pub avg_bins_per_feature: f64,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Feature statistics
    pub feature_stats: Vec<FeatureBinningStats>,
}

/// Binning statistics for individual features
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureBinningStats {
    /// Feature index
    pub feature_index: usize,
    /// Feature type
    pub feature_type: FeatureType,
    /// Number of bins
    pub num_bins: usize,
    /// Number of unique values
    pub num_unique_values: usize,
    /// Missing value count
    pub missing_count: usize,
    /// Bin distribution
    pub bin_distribution: Vec<usize>,
}

impl FeatureBinner {
    /// Create a new feature binner
    pub fn new(config: BinningConfig) -> Result<Self> {
        config.validate()?;

        let max_bins = config.max_bins;
        let min_data_per_bin = config.min_data_per_bin;

        Ok(FeatureBinner {
            config,
            max_bins,
            min_data_per_bin,
            bin_mappers: Vec::new(),
            feature_types: Vec::new(),
            missing_types: Vec::new(),
        })
    }

    /// Fit the binner on feature data
    pub fn fit(
        &mut self,
        features: &Array2<f32>,
        feature_types: Option<&[FeatureType]>,
    ) -> Result<()> {
        let num_features = features.ncols();
        let _num_samples = features.nrows();

        // Determine feature types
        self.feature_types = match feature_types {
            Some(types) => {
                if types.len() != num_features {
                    return Err(LightGBMError::dimension_mismatch(
                        format!("features: {}", num_features),
                        format!("feature_types: {}", types.len()),
                    ));
                }
                types.to_vec()
            }
            None => self.auto_detect_feature_types(features),
        };

        // Initialize bin mappers
        self.bin_mappers = Vec::with_capacity(num_features);
        self.missing_types = Vec::with_capacity(num_features);

        // Process each feature
        for feature_idx in 0..num_features {
            let feature_data = features.column(feature_idx);
            let feature_type = self.feature_types[feature_idx];

            let (bin_mapper, missing_type) = match feature_type {
                FeatureType::Numerical => {
                    let binner = NumericalBinner::new(self.config.clone())?;
                    binner.fit(&feature_data)?
                }
                FeatureType::Categorical => {
                    let binner = CategoricalBinner::new(self.config.clone())?;
                    binner.fit(&feature_data)?
                }
            };

            self.bin_mappers.push(bin_mapper);
            self.missing_types.push(missing_type);
        }

        Ok(())
    }

    /// Transform features to bins
    pub fn transform(&self, features: &Array2<f32>) -> Result<Array2<BinIndex>> {
        if features.ncols() != self.bin_mappers.len() {
            return Err(LightGBMError::dimension_mismatch(
                format!("fitted features: {}", self.bin_mappers.len()),
                format!("input features: {}", features.ncols()),
            ));
        }

        let num_samples = features.nrows();
        let num_features = features.ncols();
        let mut binned_features = Array2::zeros((num_samples, num_features));

        for feature_idx in 0..num_features {
            let feature_data = features.column(feature_idx);
            let bin_mapper = &self.bin_mappers[feature_idx];

            for (sample_idx, &value) in feature_data.iter().enumerate() {
                let bin = bin_mapper.value_to_bin(value);
                binned_features[[sample_idx, feature_idx]] = bin;
            }
        }

        Ok(binned_features)
    }

    /// Fit and transform in one step
    pub fn fit_transform(
        &mut self,
        features: &Array2<f32>,
        feature_types: Option<&[FeatureType]>,
    ) -> Result<Array2<BinIndex>> {
        self.fit(features, feature_types)?;
        self.transform(features)
    }

    /// Get bin mappers
    pub fn bin_mappers(&self) -> &[BinMapper] {
        &self.bin_mappers
    }

    /// Get bin mapper for specific feature
    pub fn bin_mapper(&self, feature_idx: usize) -> Option<&BinMapper> {
        self.bin_mappers.get(feature_idx)
    }

    /// Get feature types
    pub fn feature_types(&self) -> &[FeatureType] {
        &self.feature_types
    }

    /// Get missing types
    pub fn missing_types(&self) -> &[MissingType] {
        &self.missing_types
    }

    /// Get binning statistics
    pub fn statistics(&self) -> BinningStatistics {
        let num_features = self.bin_mappers.len();
        let num_numerical = self
            .feature_types
            .iter()
            .filter(|&&ft| ft == FeatureType::Numerical)
            .count();
        let num_categorical = num_features - num_numerical;

        let avg_bins_per_feature = if num_features > 0 {
            self.bin_mappers
                .iter()
                .map(|bm| bm.num_bins())
                .sum::<usize>() as f64
                / num_features as f64
        } else {
            0.0
        };

        let memory_usage = self.calculate_memory_usage();

        let feature_stats = self
            .bin_mappers
            .iter()
            .enumerate()
            .map(|(idx, bm)| FeatureBinningStats {
                feature_index: idx,
                feature_type: self.feature_types[idx],
                num_bins: bm.num_bins(),
                num_unique_values: bm.num_unique_values(),
                missing_count: 0, // TODO: Track missing counts during fitting
                bin_distribution: vec![], // TODO: Track bin distributions
            })
            .collect();

        BinningStatistics {
            num_features,
            num_numerical,
            num_categorical,
            avg_bins_per_feature,
            memory_usage,
            processing_time_ms: 0, // TODO: Track processing time
            feature_stats,
        }
    }

    /// Validate binning configuration
    // TODO: implement comprehensive configuration validation logic
    fn validate_config(&self) -> Result<()> {
        self.config.validate()
    }

    /// Auto-detect feature types
    fn auto_detect_feature_types(&self, features: &Array2<f32>) -> Vec<FeatureType> {
        let mut types = Vec::new();

        for col_idx in 0..features.ncols() {
            let column = features.column(col_idx);
            let unique_values: std::collections::HashSet<_> = column
                .iter()
                .filter(|&&x| !x.is_nan())
                .map(|&x| x as i32)
                .collect();

            // Heuristic: if all values are integers and there are very few unique values,
            // treat as categorical
            if unique_values.len() <= 3 && column.iter().all(|&x| x.is_nan() || x.fract() == 0.0) {
                types.push(FeatureType::Categorical);
            } else {
                types.push(FeatureType::Numerical);
            }
        }

        types
    }

    /// Calculate memory usage
    fn calculate_memory_usage(&self) -> usize {
        let mut usage = 0;

        // Bin mappers
        for bin_mapper in &self.bin_mappers {
            usage += bin_mapper.memory_usage();
        }

        // Feature types
        usage += self.feature_types.len() * std::mem::size_of::<FeatureType>();

        // Missing types
        usage += self.missing_types.len() * std::mem::size_of::<MissingType>();

        usage
    }

    /// Save binning configuration and mappers
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let data = BinningData {
            config: self.config.clone(),
            bin_mappers: self.bin_mappers.clone(),
            feature_types: self.feature_types.clone(),
            missing_types: self.missing_types.clone(),
        };

        let serialized = bincode::serialize(&data).map_err(|e| {
            LightGBMError::config(format!("Failed to serialize binning data: {}", e))
        })?;

        std::fs::write(path, serialized)
            .map_err(|e| LightGBMError::config(format!("Failed to write binning file: {}", e)))?;

        Ok(())
    }

    /// Load binning configuration and mappers
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let data = std::fs::read(path)
            .map_err(|e| LightGBMError::config(format!("Failed to read binning file: {}", e)))?;

        let binning_data: BinningData = bincode::deserialize(&data).map_err(|e| {
            LightGBMError::config(format!("Failed to deserialize binning data: {}", e))
        })?;

        Ok(FeatureBinner {
            config: binning_data.config.clone(),
            max_bins: binning_data.config.max_bins,
            min_data_per_bin: binning_data.config.min_data_per_bin,
            bin_mappers: binning_data.bin_mappers,
            feature_types: binning_data.feature_types,
            missing_types: binning_data.missing_types,
        })
    }
}

/// Serializable binning data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BinningData {
    config: BinningConfig,
    bin_mappers: Vec<BinMapper>,
    feature_types: Vec<FeatureType>,
    missing_types: Vec<MissingType>,
}

impl BinningConfig {
    /// Validate the binning configuration
    pub fn validate(&self) -> Result<()> {
        if self.max_bins < 2 {
            return Err(LightGBMError::invalid_parameter(
                "max_bins",
                self.max_bins.to_string(),
                "must be at least 2",
            ));
        }

        if self.max_bins > 65535 {
            return Err(LightGBMError::invalid_parameter(
                "max_bins",
                self.max_bins.to_string(),
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

/// Utility functions for binning
pub mod utils {
    use super::*;

    /// Create optimal binning configuration for dataset
    pub fn create_optimal_config(
        num_samples: usize,
        num_features: usize,
        memory_limit_mb: Option<usize>,
    ) -> BinningConfig {
        let mut config = BinningConfig::default();

        // Adjust max_bins based on data size
        if num_samples <= 1000 {
            config.max_bins = 32;
        } else if num_samples <= 10000 {
            config.max_bins = 64;
        } else if num_samples < 100000 {
            config.max_bins = 128;
        } else {
            config.max_bins = 255;
        }

        // Adjust min_data_per_bin based on data size
        config.min_data_per_bin = std::cmp::max(3, num_samples / (config.max_bins * 100));

        // Set memory limit
        config.memory_limit_mb = memory_limit_mb;

        // Choose strategy based on data characteristics
        if num_features > 1000 {
            config.strategy = BinningStrategy::Uniform; // Faster for high-dimensional data
        } else {
            config.strategy = BinningStrategy::Quantile; // Better quality for low-dimensional data
        }

        config
    }

    /// Estimate memory usage for binning
    pub fn estimate_memory_usage(
        num_samples: usize,
        num_features: usize,
        max_bins: usize,
    ) -> usize {
        // Binned features array
        let binned_features_size = num_samples * num_features * std::mem::size_of::<BinIndex>();

        // Bin mappers (approximate)
        let bin_mappers_size = num_features * max_bins * std::mem::size_of::<f64>();

        // Additional overhead
        let overhead = (binned_features_size + bin_mappers_size) / 10;

        binned_features_size + bin_mappers_size + overhead
    }

    /// Choose optimal binning strategy
    pub fn choose_strategy(
        num_samples: usize,
        num_unique_values: usize,
        feature_type: FeatureType,
    ) -> BinningStrategy {
        match feature_type {
            FeatureType::Categorical => BinningStrategy::Custom,
            FeatureType::Numerical => {
                if num_unique_values <= 50 {
                    BinningStrategy::Custom
                } else if num_samples < 10000 {
                    BinningStrategy::Uniform
                } else {
                    BinningStrategy::Quantile
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_binning_config_default() {
        let config = BinningConfig::default();
        assert_eq!(config.max_bins, DEFAULT_MAX_BIN);
        assert_eq!(config.min_data_per_bin, 3);
        assert_eq!(config.strategy, BinningStrategy::Quantile);
        assert_eq!(config.missing_handling, MissingHandling::Zero);
    }

    #[test]
    fn test_binning_config_validation() {
        let mut config = BinningConfig::default();
        assert!(config.validate().is_ok());

        config.max_bins = 1;
        assert!(config.validate().is_err());

        config.max_bins = 128;
        config.min_data_per_bin = 0;
        assert!(config.validate().is_err());

        config.min_data_per_bin = 3;
        config.force_col_wise = true;
        config.force_row_wise = true;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_feature_binner_creation() {
        let config = BinningConfig::default();
        let binner = FeatureBinner::new(config).unwrap();

        assert_eq!(binner.max_bins, DEFAULT_MAX_BIN);
        assert_eq!(binner.min_data_per_bin, 3);
        assert_eq!(binner.bin_mappers.len(), 0);
    }

    #[test]
    fn test_feature_type_detection() {
        let features =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 2.0]).unwrap();

        let config = BinningConfig::default();
        let binner = FeatureBinner::new(config).unwrap();
        let types = binner.auto_detect_feature_types(&features);

        assert_eq!(types.len(), 2);
        assert_eq!(types[0], FeatureType::Numerical);
        assert_eq!(types[1], FeatureType::Categorical);
    }

    #[test]
    fn test_memory_estimation() {
        let memory = utils::estimate_memory_usage(1000, 10, 128);
        assert!(memory > 0);

        let memory_large = utils::estimate_memory_usage(10000, 100, 255);
        assert!(memory_large > memory);
    }

    #[test]
    fn test_optimal_config_creation() {
        let config = utils::create_optimal_config(1000, 10, Some(1024));
        assert_eq!(config.max_bins, 32);
        assert_eq!(config.memory_limit_mb, Some(1024));

        let config_large = utils::create_optimal_config(100000, 100, None);
        assert_eq!(config_large.max_bins, 255);
        assert!(config_large.memory_limit_mb.is_none());
    }

    #[test]
    fn test_strategy_choice() {
        let strategy = utils::choose_strategy(1000, 20, FeatureType::Numerical);
        assert_eq!(strategy, BinningStrategy::Custom);

        let strategy = utils::choose_strategy(1000, 200, FeatureType::Numerical);
        assert_eq!(strategy, BinningStrategy::Uniform);

        let strategy = utils::choose_strategy(100000, 1000, FeatureType::Numerical);
        assert_eq!(strategy, BinningStrategy::Quantile);

        let strategy = utils::choose_strategy(1000, 10, FeatureType::Categorical);
        assert_eq!(strategy, BinningStrategy::Custom);
    }
}
