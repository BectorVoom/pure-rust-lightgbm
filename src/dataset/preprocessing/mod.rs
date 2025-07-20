//! Data preprocessing utilities for Pure Rust LightGBM.
//!
//! This module provides preprocessing capabilities including missing value
//! imputation, feature encoding, and data validation.

use crate::core::error::{LightGBMError, Result};
use crate::core::types::*;
use crate::dataset::Dataset;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod missing;

/// Preprocessor configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreprocessorConfig {
    /// Missing value imputation strategy
    pub imputation_strategy: ImputationStrategy,
    /// Encoding strategy for categorical features
    pub encoding_strategy: EncodingStrategy,
    /// Feature scaling strategy
    pub scaling_strategy: ScalingStrategy,
    /// Outlier detection and handling
    pub outlier_handling: OutlierHandling,
    /// Feature selection parameters
    pub feature_selection: FeatureSelectionConfig,
    /// Validation parameters
    pub validation: ValidationConfig,
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        PreprocessorConfig {
            imputation_strategy: ImputationStrategy::Mean,
            encoding_strategy: EncodingStrategy::OneHot,
            scaling_strategy: ScalingStrategy::None,
            outlier_handling: OutlierHandling::None,
            feature_selection: FeatureSelectionConfig::default(),
            validation: ValidationConfig::default(),
        }
    }
}

/// Missing value imputation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImputationStrategy {
    /// Remove samples with missing values
    Remove,
    /// Fill with mean value
    Mean,
    /// Fill with median value
    Median,
    /// Fill with mode (most frequent value)
    Mode,
    /// Fill with constant value
    Constant(i32),
    /// Forward fill
    Forward,
    /// Backward fill
    Backward,
    /// Interpolation
    Interpolate,
}

/// Categorical encoding strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncodingStrategy {
    /// One-hot encoding
    OneHot,
    /// Label encoding
    Label,
    /// Target encoding
    Target,
    /// Binary encoding
    Binary,
    /// Frequency encoding
    Frequency,
}

/// Feature scaling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalingStrategy {
    /// No scaling
    None,
    /// Min-max scaling
    MinMax,
    /// Standard scaling (z-score)
    Standard,
    /// Robust scaling
    Robust,
    /// Quantile uniform scaling
    QuantileUniform,
}

/// Outlier handling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutlierHandling {
    /// No outlier handling
    None,
    /// Remove outliers
    Remove,
    /// Cap outliers
    Cap,
    /// Transform outliers
    Transform,
}

/// Feature selection configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureSelectionConfig {
    /// Enable feature selection
    pub enabled: bool,
    /// Selection method
    pub method: FeatureSelectionMethod,
    /// Number of features to select
    pub num_features: Option<usize>,
    /// Feature importance threshold
    pub importance_threshold: Option<f64>,
    /// Correlation threshold for removing correlated features
    pub correlation_threshold: Option<f64>,
}

impl Default for FeatureSelectionConfig {
    fn default() -> Self {
        FeatureSelectionConfig {
            enabled: false,
            method: FeatureSelectionMethod::VarianceThreshold,
            num_features: None,
            importance_threshold: None,
            correlation_threshold: Some(0.95),
        }
    }
}

/// Feature selection methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureSelectionMethod {
    /// Variance threshold
    VarianceThreshold,
    /// Univariate selection
    Univariate,
    /// Recursive feature elimination
    RFE,
    /// L1-based selection
    L1Based,
    /// Tree-based selection
    TreeBased,
}

/// Validation configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable data validation
    pub enabled: bool,
    /// Check for missing values
    pub check_missing: bool,
    /// Check for infinite values
    pub check_infinite: bool,
    /// Check for duplicates
    pub check_duplicates: bool,
    /// Check data types
    pub check_types: bool,
    /// Check value ranges
    pub check_ranges: bool,
    /// Custom validation rules
    pub custom_rules: Vec<String>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        ValidationConfig {
            enabled: true,
            check_missing: true,
            check_infinite: true,
            check_duplicates: false,
            check_types: true,
            check_ranges: true,
            custom_rules: Vec::new(),
        }
    }
}

/// Data preprocessor
#[derive(Debug)]
pub struct Preprocessor {
    /// Configuration
    config: PreprocessorConfig,
    /// Fitted parameters
    fitted_params: Option<FittedParams>,
}

/// Fitted preprocessing parameters
#[derive(Debug, Clone)]
pub struct FittedParams {
    /// Feature means for imputation
    pub feature_means: HashMap<usize, f64>,
    /// Feature medians for imputation
    pub feature_medians: HashMap<usize, f64>,
    /// Feature modes for imputation
    pub feature_modes: HashMap<usize, f32>,
    /// Encoding mappings
    pub encoding_mappings: HashMap<usize, HashMap<i32, usize>>,
    /// Scaling parameters
    pub scaling_params: HashMap<usize, ScalingParams>,
    /// Selected features
    pub selected_features: Option<Vec<usize>>,
}

/// Scaling parameters for a feature
#[derive(Debug, Clone)]
pub struct ScalingParams {
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
    /// Quantiles for robust scaling
    pub quantiles: Option<(f64, f64)>,
}

/// Preprocessor error type
#[derive(Debug, thiserror::Error)]
pub enum PreprocessorError {
    /// Preprocessor has not been fitted on training data
    #[error("Preprocessor not fitted")]
    NotFitted,
    /// Invalid feature index provided
    #[error("Invalid feature index: {0}")]
    InvalidFeatureIndex(usize),
    /// Data dimension mismatch between expected and actual
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected number of dimensions
        expected: usize,
        /// Actual number of dimensions
        actual: usize,
    },
    /// General preprocessing error
    #[error("Preprocessing error: {0}")]
    Processing(String),
}

impl Preprocessor {
    /// Create a new preprocessor
    pub fn new(config: PreprocessorConfig) -> Self {
        Preprocessor {
            config,
            fitted_params: None,
        }
    }

    /// Fit the preprocessor on data
    pub fn fit(&mut self, dataset: &Dataset) -> Result<()> {
        log::info!(
            "Fitting preprocessor on dataset with {} features",
            dataset.num_features()
        );

        let features = dataset.features();
        let num_features = dataset.num_features();
        let _num_samples = dataset.num_data();

        let mut fitted_params = FittedParams {
            feature_means: HashMap::new(),
            feature_medians: HashMap::new(),
            feature_modes: HashMap::new(),
            encoding_mappings: HashMap::new(),
            scaling_params: HashMap::new(),
            selected_features: None,
        };

        // Fit imputation parameters
        for feat_idx in 0..num_features {
            let feature_data: Vec<f32> = features.column(feat_idx).to_vec();
            let valid_values: Vec<f32> = feature_data
                .iter()
                .filter(|&&x| !x.is_nan() && x.is_finite())
                .copied()
                .collect();

            if !valid_values.is_empty() {
                // Calculate mean
                let mean = valid_values.iter().sum::<f32>() as f64 / valid_values.len() as f64;
                fitted_params.feature_means.insert(feat_idx, mean);

                // Calculate median
                let mut sorted_values = valid_values.clone();
                sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = if sorted_values.len() % 2 == 0 {
                    let mid1 = sorted_values[sorted_values.len() / 2 - 1];
                    let mid2 = sorted_values[sorted_values.len() / 2];
                    (mid1 + mid2) / 2.0
                } else {
                    sorted_values[sorted_values.len() / 2]
                };
                fitted_params
                    .feature_medians
                    .insert(feat_idx, median as f64);

                // Calculate mode (most frequent value for categorical features)
                let mut value_counts = HashMap::new();
                for &val in &valid_values {
                    *value_counts.entry(val as i32).or_insert(0) += 1;
                }
                if let Some((&mode_val, _)) = value_counts.iter().max_by_key(|(_, &count)| count) {
                    fitted_params
                        .feature_modes
                        .insert(feat_idx, mode_val as f32);
                }

                // Calculate scaling parameters
                let var = valid_values
                    .iter()
                    .map(|&x| (x as f64 - mean).powi(2))
                    .sum::<f64>()
                    / valid_values.len() as f64;
                let std = var.sqrt();

                let min_val = valid_values.iter().fold(f32::INFINITY, |a, &b| a.min(b)) as f64;
                let max_val = valid_values
                    .iter()
                    .fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;

                // Calculate quantiles for robust scaling
                let q25_idx = (sorted_values.len() as f64 * 0.25) as usize;
                let q75_idx = (sorted_values.len() as f64 * 0.75) as usize;
                let quantiles = if q25_idx < sorted_values.len() && q75_idx < sorted_values.len() {
                    Some((sorted_values[q25_idx] as f64, sorted_values[q75_idx] as f64))
                } else {
                    None
                };

                fitted_params.scaling_params.insert(
                    feat_idx,
                    ScalingParams {
                        min: min_val,
                        max: max_val,
                        mean,
                        std,
                        quantiles,
                    },
                );
            }
        }

        // Fit feature selection if enabled
        if self.config.feature_selection.enabled {
            fitted_params.selected_features = Some(self.select_features(dataset, &fitted_params)?);
        }

        self.fitted_params = Some(fitted_params);
        log::info!("Preprocessor fitting completed");
        Ok(())
    }

    /// Transform data using fitted parameters
    pub fn transform(&self, dataset: &Dataset) -> Result<Dataset> {
        let fitted_params = self
            .fitted_params
            .as_ref()
            .ok_or_else(|| LightGBMError::internal("Preprocessor not fitted. Call fit() first."))?;

        log::info!(
            "Transforming dataset with {} features",
            dataset.num_features()
        );

        let features = dataset.features();
        let mut transformed_features = features.to_owned();

        // Apply missing value imputation
        self.apply_imputation(&mut transformed_features, fitted_params)?;

        // Apply scaling
        self.apply_scaling(&mut transformed_features, fitted_params)?;

        // Apply feature selection
        let final_features = if let Some(ref selected_indices) = fitted_params.selected_features {
            self.apply_feature_selection(&transformed_features, selected_indices)?
        } else {
            transformed_features
        };

        // Create transformed dataset
        let transformed_feature_names =
            if let Some(ref selected_indices) = fitted_params.selected_features {
                dataset.feature_names().map(|names| {
                    selected_indices
                        .iter()
                        .map(|&idx| names[idx].clone())
                        .collect()
                })
            } else {
                dataset.feature_names().map(|names| names.to_vec())
            };

        Dataset::new(
            final_features,
            dataset.labels().to_owned(),
            dataset.weights().map(|w| w.to_owned()),
            dataset.groups().map(|g| g.to_owned()),
            transformed_feature_names,
            None, // feature_types will be inferred
        )
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, dataset: &Dataset) -> Result<Dataset> {
        self.fit(dataset)?;
        self.transform(dataset)
    }

    /// Check if preprocessor is fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted_params.is_some()
    }

    /// Get fitted parameters
    pub fn fitted_params(&self) -> Option<&FittedParams> {
        self.fitted_params.as_ref()
    }

    /// Get configuration
    pub fn config(&self) -> &PreprocessorConfig {
        &self.config
    }

    /// Apply missing value imputation
    fn apply_imputation(
        &self,
        features: &mut Array2<f32>,
        fitted_params: &FittedParams,
    ) -> Result<()> {
        let (num_rows, num_cols) = features.dim();

        for feat_idx in 0..num_cols {
            let impute_value = match self.config.imputation_strategy {
                ImputationStrategy::Remove => continue, // Skip, removal should be done at dataset level
                ImputationStrategy::Mean => fitted_params
                    .feature_means
                    .get(&feat_idx)
                    .copied()
                    .unwrap_or(0.0) as f32,
                ImputationStrategy::Median => fitted_params
                    .feature_medians
                    .get(&feat_idx)
                    .copied()
                    .unwrap_or(0.0) as f32,
                ImputationStrategy::Mode => fitted_params
                    .feature_modes
                    .get(&feat_idx)
                    .copied()
                    .unwrap_or(0.0),
                ImputationStrategy::Constant(value) => value as f32,
                ImputationStrategy::Forward => {
                    // Forward fill implementation
                    let mut last_valid = 0.0f32;
                    for row_idx in 0..num_rows {
                        let current_val = features[[row_idx, feat_idx]];
                        if current_val.is_nan() {
                            features[[row_idx, feat_idx]] = last_valid;
                        } else {
                            last_valid = current_val;
                        }
                    }
                    continue;
                }
                ImputationStrategy::Backward => {
                    // Backward fill implementation
                    let mut next_valid = 0.0f32;
                    for row_idx in (0..num_rows).rev() {
                        let current_val = features[[row_idx, feat_idx]];
                        if current_val.is_nan() {
                            features[[row_idx, feat_idx]] = next_valid;
                        } else {
                            next_valid = current_val;
                        }
                    }
                    continue;
                }
                ImputationStrategy::Interpolate => {
                    // Linear interpolation
                    self.apply_linear_interpolation(features, feat_idx)?;
                    continue;
                }
            };

            // Apply simple imputation
            for row_idx in 0..num_rows {
                if features[[row_idx, feat_idx]].is_nan() {
                    features[[row_idx, feat_idx]] = impute_value;
                }
            }
        }

        Ok(())
    }

    /// Apply linear interpolation for missing values
    fn apply_linear_interpolation(
        &self,
        features: &mut Array2<f32>,
        feat_idx: usize,
    ) -> Result<()> {
        let num_rows = features.nrows();
        let mut missing_indices = Vec::new();

        // Find missing value indices
        for row_idx in 0..num_rows {
            if features[[row_idx, feat_idx]].is_nan() {
                missing_indices.push(row_idx);
            }
        }

        // Interpolate each missing value
        for &missing_idx in &missing_indices {
            // Find previous valid value
            let mut prev_idx = None;
            let mut prev_val = 0.0f32;
            for i in (0..missing_idx).rev() {
                if !features[[i, feat_idx]].is_nan() {
                    prev_idx = Some(i);
                    prev_val = features[[i, feat_idx]];
                    break;
                }
            }

            // Find next valid value
            let mut next_idx = None;
            let mut next_val = 0.0f32;
            for i in (missing_idx + 1)..num_rows {
                if !features[[i, feat_idx]].is_nan() {
                    next_idx = Some(i);
                    next_val = features[[i, feat_idx]];
                    break;
                }
            }

            // Interpolate
            let interpolated_val = match (prev_idx, next_idx) {
                (Some(p_idx), Some(n_idx)) => {
                    // Linear interpolation
                    let ratio = (missing_idx - p_idx) as f32 / (n_idx - p_idx) as f32;
                    prev_val + ratio * (next_val - prev_val)
                }
                (Some(_), None) => prev_val, // Use previous value
                (None, Some(_)) => next_val, // Use next value
                (None, None) => 0.0,         // No valid values found
            };

            features[[missing_idx, feat_idx]] = interpolated_val;
        }

        Ok(())
    }

    /// Apply feature scaling
    fn apply_scaling(
        &self,
        features: &mut Array2<f32>,
        fitted_params: &FittedParams,
    ) -> Result<()> {
        if self.config.scaling_strategy == ScalingStrategy::None {
            return Ok(());
        }

        let (num_rows, num_cols) = features.dim();

        for feat_idx in 0..num_cols {
            if let Some(scaling_params) = fitted_params.scaling_params.get(&feat_idx) {
                for row_idx in 0..num_rows {
                    let original_val = features[[row_idx, feat_idx]] as f64;

                    let scaled_val = match self.config.scaling_strategy {
                        ScalingStrategy::None => original_val,
                        ScalingStrategy::MinMax => {
                            if scaling_params.max != scaling_params.min {
                                (original_val - scaling_params.min)
                                    / (scaling_params.max - scaling_params.min)
                            } else {
                                0.0
                            }
                        }
                        ScalingStrategy::Standard => {
                            if scaling_params.std != 0.0 {
                                (original_val - scaling_params.mean) / scaling_params.std
                            } else {
                                0.0
                            }
                        }
                        ScalingStrategy::Robust => {
                            if let Some((q25, q75)) = scaling_params.quantiles {
                                if q75 != q25 {
                                    (original_val - scaling_params.mean) / (q75 - q25)
                                } else {
                                    0.0
                                }
                            } else {
                                original_val
                            }
                        }
                        ScalingStrategy::QuantileUniform => {
                            // Simple quantile uniform scaling (could be enhanced)
                            if scaling_params.max != scaling_params.min {
                                (original_val - scaling_params.min)
                                    / (scaling_params.max - scaling_params.min)
                            } else {
                                0.0
                            }
                        }
                    };

                    features[[row_idx, feat_idx]] = scaled_val as f32;
                }
            }
        }

        Ok(())
    }

    /// Apply feature selection
    fn apply_feature_selection(
        &self,
        features: &Array2<f32>,
        selected_indices: &[usize],
    ) -> Result<Array2<f32>> {
        let num_rows = features.nrows();
        let num_selected = selected_indices.len();
        let mut selected_features = Array2::<f32>::zeros((num_rows, num_selected));

        for (new_idx, &orig_idx) in selected_indices.iter().enumerate() {
            for row_idx in 0..num_rows {
                selected_features[[row_idx, new_idx]] = features[[row_idx, orig_idx]];
            }
        }

        Ok(selected_features)
    }

    /// Select features based on configuration
    fn select_features(
        &self,
        dataset: &Dataset,
        fitted_params: &FittedParams,
    ) -> Result<Vec<usize>> {
        let num_features = dataset.num_features();
        let mut selected_indices: Vec<usize> = (0..num_features).collect();

        match self.config.feature_selection.method {
            FeatureSelectionMethod::VarianceThreshold => {
                // Remove low-variance features
                let threshold = 0.01; // Default threshold
                selected_indices.retain(|&idx| {
                    if let Some(scaling_params) = fitted_params.scaling_params.get(&idx) {
                        scaling_params.std.powi(2) > threshold
                    } else {
                        false
                    }
                });
            }
            FeatureSelectionMethod::Univariate => {
                // For now, just return all features (could implement statistical tests)
                log::warn!(
                    "Univariate feature selection not fully implemented, using all features"
                );
            }
            FeatureSelectionMethod::RFE => {
                // Recursive feature elimination (placeholder)
                log::warn!("RFE feature selection not implemented, using all features");
            }
            FeatureSelectionMethod::L1Based => {
                // L1-based feature selection (placeholder)
                log::warn!("L1-based feature selection not implemented, using all features");
            }
            FeatureSelectionMethod::TreeBased => {
                // Tree-based feature selection (placeholder)
                log::warn!("Tree-based feature selection not implemented, using all features");
            }
        }

        // Apply number limit if specified
        if let Some(max_features) = self.config.feature_selection.num_features {
            if selected_indices.len() > max_features {
                selected_indices.truncate(max_features);
            }
        }

        // Remove highly correlated features if threshold is set
        if let Some(corr_threshold) = self.config.feature_selection.correlation_threshold {
            selected_indices =
                self.remove_correlated_features(dataset, selected_indices, corr_threshold)?;
        }

        Ok(selected_indices)
    }

    /// Remove highly correlated features
    fn remove_correlated_features(
        &self,
        dataset: &Dataset,
        mut indices: Vec<usize>,
        threshold: f64,
    ) -> Result<Vec<usize>> {
        let features = dataset.features();
        let mut to_remove = std::collections::HashSet::new();

        // Simple correlation-based removal
        for i in 0..indices.len() {
            if to_remove.contains(&i) {
                continue;
            }

            for j in (i + 1)..indices.len() {
                if to_remove.contains(&j) {
                    continue;
                }

                let idx_i = indices[i];
                let idx_j = indices[j];

                // Calculate correlation between features i and j
                let correlation =
                    self.calculate_correlation(&features.column(idx_i), &features.column(idx_j))?;

                if correlation.abs() > threshold {
                    // Remove the feature with higher index (arbitrary choice)
                    to_remove.insert(j);
                }
            }
        }

        // Remove marked indices
        let mut result = Vec::new();
        for (i, &idx) in indices.iter().enumerate() {
            if !to_remove.contains(&i) {
                result.push(idx);
            }
        }
        indices = result;

        Ok(indices)
    }

    /// Calculate Pearson correlation between two feature vectors
    fn calculate_correlation(
        &self,
        x: &ndarray::ArrayView1<'_, f32>,
        y: &ndarray::ArrayView1<'_, f32>,
    ) -> Result<f64> {
        let n = x.len() as f64;
        if n == 0.0 {
            return Ok(0.0);
        }

        // Filter out NaN and infinite values
        let mut valid_pairs = Vec::new();
        for i in 0..x.len() {
            let xi = x[i] as f64;
            let yi = y[i] as f64;
            if xi.is_finite() && yi.is_finite() {
                valid_pairs.push((xi, yi));
            }
        }

        if valid_pairs.len() < 2 {
            return Ok(0.0);
        }

        let n_valid = valid_pairs.len() as f64;
        let mean_x = valid_pairs.iter().map(|(x, _)| x).sum::<f64>() / n_valid;
        let mean_y = valid_pairs.iter().map(|(_, y)| y).sum::<f64>() / n_valid;

        let mut num = 0.0;
        let mut den_x = 0.0;
        let mut den_y = 0.0;

        for (xi, yi) in valid_pairs {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            num += dx * dy;
            den_x += dx * dx;
            den_y += dy * dy;
        }

        if den_x == 0.0 || den_y == 0.0 {
            Ok(0.0)
        } else {
            Ok(num / (den_x * den_y).sqrt())
        }
    }
}

/// Utility functions for preprocessing
pub mod utils {
    use super::*;

    /// Detect missing values in dataset
    pub fn detect_missing_values(_dataset: &Dataset) -> Result<Array2<bool>> {
        // TODO: Implement missing value detection
        Err(LightGBMError::not_implemented("Missing value detection"))
    }

    /// Calculate feature statistics
    pub fn calculate_feature_statistics(_dataset: &Dataset) -> Result<Vec<FeatureStats>> {
        // TODO: Implement feature statistics calculation
        Err(LightGBMError::not_implemented(
            "Feature statistics calculation",
        ))
    }

    /// Detect outliers using IQR method
    pub fn detect_outliers_iqr(_values: &[f32], _iqr_multiplier: f64) -> Vec<bool> {
        // TODO: Implement outlier detection
        vec![]
    }

    /// Impute missing values
    pub fn impute_missing_values(
        _values: &mut [f32],
        _strategy: ImputationStrategy,
        _fill_value: Option<f32>,
    ) -> Result<()> {
        // TODO: Implement missing value imputation
        Err(LightGBMError::not_implemented("Missing value imputation"))
    }
}

/// Feature statistics
#[derive(Debug, Clone)]
pub struct FeatureStats {
    /// Feature index
    pub index: usize,
    /// Feature type
    pub feature_type: FeatureType,
    /// Number of missing values
    pub missing_count: usize,
    /// Number of unique values
    pub unique_count: usize,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Quantiles
    pub quantiles: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocessor_config_default() {
        let config = PreprocessorConfig::default();
        assert_eq!(config.imputation_strategy, ImputationStrategy::Mean);
        assert_eq!(config.encoding_strategy, EncodingStrategy::OneHot);
        assert_eq!(config.scaling_strategy, ScalingStrategy::None);
        assert_eq!(config.outlier_handling, OutlierHandling::None);
    }

    #[test]
    fn test_preprocessor_creation() {
        let config = PreprocessorConfig::default();
        let preprocessor = Preprocessor::new(config);
        assert!(!preprocessor.is_fitted());
    }

    #[test]
    fn test_feature_selection_config_default() {
        let config = FeatureSelectionConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.method, FeatureSelectionMethod::VarianceThreshold);
        assert_eq!(config.correlation_threshold, Some(0.95));
    }

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert!(config.enabled);
        assert!(config.check_missing);
        assert!(config.check_infinite);
        assert!(config.check_types);
        assert!(config.check_ranges);
        assert!(!config.check_duplicates);
    }
}
