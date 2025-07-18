//! Missing value handling for Pure Rust LightGBM.
//!
//! This module provides comprehensive missing value detection, analysis, and imputation
//! strategies optimized for gradient boosting algorithms.

use crate::core::error::{LightGBMError, Result};
use crate::dataset::Dataset;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Missing value imputation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImputationStrategy {
    /// Remove samples with missing values
    Remove,
    /// Fill with constant value
    Constant(i32),
    /// Fill with mean value (for numerical features)
    Mean,
    /// Fill with median value (for numerical features)
    Median,
    /// Fill with mode (most frequent value)
    Mode,
    /// Forward fill (use previous valid value)
    Forward,
    /// Backward fill (use next valid value)
    Backward,
    /// Linear interpolation
    LinearInterpolation,
    /// K-nearest neighbors imputation
    KNN(usize), // k parameter
    /// Random sampling from valid values
    Random,
    /// Zero imputation
    Zero,
}

impl Default for ImputationStrategy {
    fn default() -> Self {
        ImputationStrategy::Mean
    }
}

/// Missing value configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MissingValueConfig {
    /// Default imputation strategy
    pub default_strategy: ImputationStrategy,
    /// Per-feature imputation strategies
    pub feature_strategies: HashMap<usize, ImputationStrategy>,
    /// Threshold for considering a feature as having too many missing values
    pub missing_threshold: f64,
    /// Whether to remove features with too many missing values
    pub remove_high_missing_features: bool,
    /// Whether to create missing value indicators
    pub create_missing_indicators: bool,
    /// Custom missing value representations
    pub custom_missing_values: Vec<f32>,
    /// Random seed for random imputation
    pub random_seed: Option<u64>,
}

impl Default for MissingValueConfig {
    fn default() -> Self {
        MissingValueConfig {
            default_strategy: ImputationStrategy::Mean,
            feature_strategies: HashMap::new(),
            missing_threshold: 0.8, // Remove features with >80% missing values
            remove_high_missing_features: false,
            create_missing_indicators: false,
            custom_missing_values: vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            random_seed: None,
        }
    }
}

/// Missing value analysis results
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MissingValueAnalysis {
    /// Total number of missing values
    pub total_missing: usize,
    /// Missing values per feature
    pub missing_per_feature: Vec<usize>,
    /// Missing value ratio per feature
    pub missing_ratio_per_feature: Vec<f64>,
    /// Missing values per sample
    pub missing_per_sample: Vec<usize>,
    /// Features with high missing value ratio
    pub high_missing_features: Vec<usize>,
    /// Samples with high missing value ratio
    pub high_missing_samples: Vec<usize>,
    /// Missing value patterns
    pub missing_patterns: Vec<MissingPattern>,
}

/// Missing value pattern
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MissingPattern {
    /// Feature indices with missing values
    pub features: Vec<usize>,
    /// Number of samples with this pattern
    pub count: usize,
    /// Percentage of total samples
    pub percentage: f64,
}

/// Fitted imputation parameters
#[derive(Debug, Clone)]
pub struct ImputationParameters {
    /// Strategy used for each feature
    pub strategies: HashMap<usize, ImputationStrategy>,
    /// Computed values for imputation
    pub imputation_values: HashMap<usize, f32>,
    /// Statistical parameters for each feature
    pub feature_stats: HashMap<usize, FeatureStatistics>,
    /// KNN models for features using KNN imputation
    pub knn_models: HashMap<usize, KNNModel>,
    /// Features that were removed due to high missing rates
    pub removed_features: Vec<usize>,
    /// Random number generator state
    pub rng_state: Option<u64>,
}

/// Feature statistics for imputation
#[derive(Debug, Clone)]
pub struct FeatureStatistics {
    /// Mean value
    pub mean: f64,
    /// Median value
    pub median: f64,
    /// Mode value
    pub mode: f32,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Valid values for random sampling
    pub valid_values: Vec<f32>,
}

/// Simple KNN model for imputation
#[derive(Debug, Clone)]
pub struct KNNModel {
    /// Reference data points (features only, excluding the target feature)
    pub reference_features: Array2<f32>,
    /// Target values for the reference points
    pub reference_values: Array1<f32>,
    /// Number of neighbors
    pub k: usize,
}

/// Missing value imputer
pub struct MissingValueImputer {
    /// Configuration
    config: MissingValueConfig,
    /// Fitted parameters
    fitted_params: Option<ImputationParameters>,
}

impl MissingValueImputer {
    /// Create a new missing value imputer
    pub fn new(config: MissingValueConfig) -> Self {
        MissingValueImputer {
            config,
            fitted_params: None,
        }
    }

    /// Create imputer with default configuration
    pub fn default() -> Self {
        Self::new(MissingValueConfig::default())
    }

    /// Set imputation strategy for a specific feature
    pub fn set_feature_strategy(&mut self, feature_idx: usize, strategy: ImputationStrategy) {
        self.config.feature_strategies.insert(feature_idx, strategy);
    }

    /// Set missing threshold
    pub fn set_missing_threshold(&mut self, threshold: f64) {
        self.config.missing_threshold = threshold;
    }

    /// Enable missing value indicators
    pub fn enable_missing_indicators(&mut self, enable: bool) {
        self.config.create_missing_indicators = enable;
    }

    /// Analyze missing values in dataset
    pub fn analyze(&self, dataset: &Dataset) -> Result<MissingValueAnalysis> {
        let features = dataset.features();
        let num_samples = dataset.num_data();
        let num_features = dataset.num_features();

        log::info!(
            "Analyzing missing values in dataset with {} samples and {} features",
            num_samples,
            num_features
        );

        // Detect missing values
        let missing_mask = self.detect_missing_values(&features)?;

        // Calculate statistics
        let total_missing = missing_mask.iter().filter(|&&x| x).count();
        let missing_per_feature = self.calculate_missing_per_feature(&missing_mask);
        let missing_ratio_per_feature: Vec<f64> = missing_per_feature
            .iter()
            .map(|&count| count as f64 / num_samples as f64)
            .collect();
        let missing_per_sample = self.calculate_missing_per_sample(&missing_mask);

        // Identify high missing features and samples
        let high_missing_features = missing_ratio_per_feature
            .iter()
            .enumerate()
            .filter(|(_, &ratio)| ratio > self.config.missing_threshold)
            .map(|(idx, _)| idx)
            .collect();

        let high_missing_samples = missing_per_sample
            .iter()
            .enumerate()
            .filter(|(_, &count)| {
                count as f64 / num_features as f64 > self.config.missing_threshold
            })
            .map(|(idx, _)| idx)
            .collect();

        // Analyze missing patterns
        let missing_patterns = self.analyze_missing_patterns(&missing_mask)?;

        Ok(MissingValueAnalysis {
            total_missing,
            missing_per_feature,
            missing_ratio_per_feature,
            missing_per_sample,
            high_missing_features,
            high_missing_samples,
            missing_patterns,
        })
    }

    /// Fit the imputer on training data
    pub fn fit(&mut self, dataset: &Dataset) -> Result<()> {
        log::info!("Fitting missing value imputer on dataset");

        let features = dataset.features();
        let num_features = dataset.num_features();

        // Detect missing values
        let missing_mask = self.detect_missing_values(&features)?;

        // Initialize fitted parameters
        let mut strategies = HashMap::new();
        let mut imputation_values = HashMap::new();
        let mut feature_stats = HashMap::new();
        let mut knn_models = HashMap::new();
        let mut removed_features = Vec::new();

        // Process each feature
        for feature_idx in 0..num_features {
            let feature_data = features.column(feature_idx);
            let feature_missing = missing_mask.column(feature_idx);

            // Check if feature should be removed due to high missing rate
            let missing_ratio =
                feature_missing.iter().filter(|&&x| x).count() as f64 / feature_data.len() as f64;

            if self.config.remove_high_missing_features
                && missing_ratio > self.config.missing_threshold
            {
                log::warn!(
                    "Removing feature {} due to high missing rate: {:.2}%",
                    feature_idx,
                    missing_ratio * 100.0
                );
                removed_features.push(feature_idx);
                continue;
            }

            // Determine strategy for this feature
            let strategy = self
                .config
                .feature_strategies
                .get(&feature_idx)
                .copied()
                .unwrap_or(self.config.default_strategy);

            strategies.insert(feature_idx, strategy);

            // Calculate feature statistics
            let stats = self.calculate_feature_statistics(&feature_data, &feature_missing)?;
            feature_stats.insert(feature_idx, stats.clone());

            // Calculate imputation value based on strategy
            let imputation_value = match strategy {
                ImputationStrategy::Constant(value) => value as f32,
                ImputationStrategy::Mean => stats.mean as f32,
                ImputationStrategy::Median => stats.median as f32,
                ImputationStrategy::Mode => stats.mode,
                ImputationStrategy::Zero => 0.0,
                ImputationStrategy::KNN(k) => {
                    // Fit KNN model
                    let knn_model = self.fit_knn_model(&features, feature_idx, k)?;
                    knn_models.insert(feature_idx, knn_model);
                    0.0 // Placeholder, actual values computed during transform
                }
                _ => stats.mean as f32, // Default to mean for other strategies
            };

            imputation_values.insert(feature_idx, imputation_value);
        }

        self.fitted_params = Some(ImputationParameters {
            strategies,
            imputation_values,
            feature_stats,
            knn_models,
            removed_features,
            rng_state: self.config.random_seed,
        });

        log::info!("Missing value imputer fitting completed");
        Ok(())
    }

    /// Transform dataset by imputing missing values
    pub fn transform(&self, dataset: &Dataset) -> Result<Dataset> {
        let fitted_params = self
            .fitted_params
            .as_ref()
            .ok_or_else(|| LightGBMError::internal("Imputer not fitted. Call fit() first."))?;

        log::info!("Transforming dataset with missing value imputation");

        let features = dataset.features();
        let mut imputed_features = features.to_owned();

        // Detect missing values
        let missing_mask = self.detect_missing_values(&features)?;

        // Apply imputation for each feature
        for (feature_idx, &strategy) in &fitted_params.strategies {
            self.impute_feature(
                &mut imputed_features,
                &missing_mask,
                *feature_idx,
                strategy,
                fitted_params,
            )?;
        }

        // Remove high missing features if configured
        let final_features = if !fitted_params.removed_features.is_empty() {
            self.remove_features(&imputed_features, &fitted_params.removed_features)?
        } else {
            imputed_features
        };

        // Create missing indicators if configured
        let final_features_with_indicators = if self.config.create_missing_indicators {
            self.add_missing_indicators(&final_features, &missing_mask)?
        } else {
            final_features
        };

        // Update feature names
        let new_feature_names = self.update_feature_names(dataset, fitted_params)?;

        // Create transformed dataset
        Dataset::new(
            final_features_with_indicators,
            dataset.labels().to_owned(),
            dataset.weights().map(|w| w.to_owned()),
            dataset.groups().map(|g| g.to_owned()),
            Some(new_feature_names),
            None, // feature_types will be inferred
        )
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, dataset: &Dataset) -> Result<Dataset> {
        self.fit(dataset)?;
        self.transform(dataset)
    }

    /// Check if imputer is fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted_params.is_some()
    }

    /// Get configuration
    pub fn config(&self) -> &MissingValueConfig {
        &self.config
    }

    /// Get fitted parameters
    pub fn fitted_params(&self) -> Option<&ImputationParameters> {
        self.fitted_params.as_ref()
    }

    /// Detect missing values in feature matrix
    fn detect_missing_values(&self, features: &ArrayView2<f32>) -> Result<Array2<bool>> {
        let (num_samples, num_features) = features.dim();
        let mut missing_mask = Array2::<bool>::from_elem((num_samples, num_features), false);

        for ((i, j), &value) in features.indexed_iter() {
            missing_mask[[i, j]] = self.is_missing_value(value);
        }

        Ok(missing_mask)
    }

    /// Check if a value should be considered missing
    fn is_missing_value(&self, value: f32) -> bool {
        value.is_nan() || value.is_infinite() || self.config.custom_missing_values.contains(&value)
    }

    /// Calculate missing values per feature
    fn calculate_missing_per_feature(&self, missing_mask: &Array2<bool>) -> Vec<usize> {
        missing_mask
            .axis_iter(Axis(0))
            .map(|row| row.iter().filter(|&&x| x).count())
            .collect()
    }

    /// Calculate missing values per sample
    fn calculate_missing_per_sample(&self, missing_mask: &Array2<bool>) -> Vec<usize> {
        missing_mask
            .axis_iter(Axis(1))
            .map(|col| col.iter().filter(|&&x| x).count())
            .collect()
    }

    /// Analyze missing value patterns
    fn analyze_missing_patterns(&self, missing_mask: &Array2<bool>) -> Result<Vec<MissingPattern>> {
        let num_samples = missing_mask.nrows();
        let mut pattern_counts: HashMap<Vec<usize>, usize> = HashMap::new();

        // Count patterns
        for sample_idx in 0..num_samples {
            let sample_missing = missing_mask.row(sample_idx);
            let missing_features: Vec<usize> = sample_missing
                .iter()
                .enumerate()
                .filter(|(_, &is_missing)| is_missing)
                .map(|(feature_idx, _)| feature_idx)
                .collect();

            *pattern_counts.entry(missing_features).or_insert(0) += 1;
        }

        // Convert to patterns with percentages
        let mut patterns = Vec::new();
        for (features, count) in pattern_counts {
            let percentage = count as f64 / num_samples as f64 * 100.0;
            patterns.push(MissingPattern {
                features,
                count,
                percentage,
            });
        }

        // Sort by frequency
        patterns.sort_by(|a, b| b.count.cmp(&a.count));

        Ok(patterns)
    }

    /// Calculate feature statistics
    fn calculate_feature_statistics(
        &self,
        feature_data: &ArrayView1<f32>,
        missing_mask: &ArrayView1<bool>,
    ) -> Result<FeatureStatistics> {
        let valid_values: Vec<f32> = feature_data
            .iter()
            .zip(missing_mask.iter())
            .filter(|(_, &is_missing)| !is_missing)
            .map(|(&value, _)| value)
            .collect();

        if valid_values.is_empty() {
            return Ok(FeatureStatistics {
                mean: 0.0,
                median: 0.0,
                mode: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                valid_values: vec![0.0],
            });
        }

        // Calculate mean
        let mean = valid_values.iter().sum::<f32>() as f64 / valid_values.len() as f64;

        // Calculate median
        let mut sorted_values = valid_values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted_values.len() % 2 == 0 {
            let mid1 = sorted_values[sorted_values.len() / 2 - 1];
            let mid2 = sorted_values[sorted_values.len() / 2];
            (mid1 + mid2) as f64 / 2.0
        } else {
            sorted_values[sorted_values.len() / 2] as f64
        };

        // Calculate mode (most frequent value)
        let mut value_counts = HashMap::new();
        for &value in &valid_values {
            *value_counts.entry(value as i32).or_insert(0) += 1;
        }
        let mode = value_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&value, _)| value as f32)
            .unwrap_or(0.0);

        // Calculate standard deviation
        let variance = valid_values
            .iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            / valid_values.len() as f64;
        let std_dev = variance.sqrt();

        // Calculate min and max
        let min = valid_values.iter().fold(f32::INFINITY, |a, &b| a.min(b)) as f64;
        let max = valid_values
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;

        Ok(FeatureStatistics {
            mean,
            median,
            mode,
            std_dev,
            min,
            max,
            valid_values,
        })
    }

    /// Fit KNN model for a feature
    fn fit_knn_model(
        &self,
        features: &ArrayView2<f32>,
        target_feature: usize,
        k: usize,
    ) -> Result<KNNModel> {
        let num_samples = features.nrows();
        let num_features = features.ncols();

        // Get indices of other features (excluding target feature)
        let other_features: Vec<usize> =
            (0..num_features).filter(|&i| i != target_feature).collect();

        // Extract reference data for samples without missing values in target feature
        let mut reference_features_vec = Vec::new();
        let mut reference_values_vec = Vec::new();

        for sample_idx in 0..num_samples {
            let target_value = features[[sample_idx, target_feature]];
            if !self.is_missing_value(target_value) {
                // Check if other features are also valid
                let other_values: Vec<f32> = other_features
                    .iter()
                    .map(|&feat_idx| features[[sample_idx, feat_idx]])
                    .collect();

                if other_values.iter().all(|&val| !self.is_missing_value(val)) {
                    reference_features_vec.extend(other_values);
                    reference_values_vec.push(target_value);
                }
            }
        }

        if reference_values_vec.is_empty() {
            return Err(LightGBMError::internal(
                "No valid reference data for KNN imputation",
            ));
        }

        let num_reference_samples = reference_values_vec.len();
        let num_reference_features = other_features.len();

        let reference_features = Array2::from_shape_vec(
            (num_reference_samples, num_reference_features),
            reference_features_vec,
        )
        .map_err(|e| {
            LightGBMError::internal(format!("Failed to create KNN reference features: {}", e))
        })?;

        let reference_values = Array1::from_vec(reference_values_vec);

        Ok(KNNModel {
            reference_features,
            reference_values,
            k,
        })
    }

    /// Impute missing values for a specific feature
    fn impute_feature(
        &self,
        features: &mut Array2<f32>,
        missing_mask: &Array2<bool>,
        feature_idx: usize,
        strategy: ImputationStrategy,
        fitted_params: &ImputationParameters,
    ) -> Result<()> {
        let num_samples = features.nrows();

        match strategy {
            ImputationStrategy::Remove => {
                // Skip - removal should be handled at dataset level
                return Ok(());
            }
            ImputationStrategy::Constant(_)
            | ImputationStrategy::Mean
            | ImputationStrategy::Median
            | ImputationStrategy::Mode
            | ImputationStrategy::Zero => {
                // Simple value imputation
                let imputation_value = fitted_params
                    .imputation_values
                    .get(&feature_idx)
                    .copied()
                    .unwrap_or(0.0);

                for sample_idx in 0..num_samples {
                    if missing_mask[[sample_idx, feature_idx]] {
                        features[[sample_idx, feature_idx]] = imputation_value;
                    }
                }
            }
            ImputationStrategy::Forward => {
                self.apply_forward_fill(features, missing_mask, feature_idx)?;
            }
            ImputationStrategy::Backward => {
                self.apply_backward_fill(features, missing_mask, feature_idx)?;
            }
            ImputationStrategy::LinearInterpolation => {
                self.apply_linear_interpolation(features, missing_mask, feature_idx)?;
            }
            ImputationStrategy::KNN(_) => {
                if let Some(knn_model) = fitted_params.knn_models.get(&feature_idx) {
                    self.apply_knn_imputation(features, missing_mask, feature_idx, knn_model)?;
                }
            }
            ImputationStrategy::Random => {
                self.apply_random_imputation(features, missing_mask, feature_idx, fitted_params)?;
            }
        }

        Ok(())
    }

    /// Apply forward fill imputation
    fn apply_forward_fill(
        &self,
        features: &mut Array2<f32>,
        missing_mask: &Array2<bool>,
        feature_idx: usize,
    ) -> Result<()> {
        let num_samples = features.nrows();
        let mut last_valid_value = 0.0f32;

        for sample_idx in 0..num_samples {
            if missing_mask[[sample_idx, feature_idx]] {
                features[[sample_idx, feature_idx]] = last_valid_value;
            } else {
                last_valid_value = features[[sample_idx, feature_idx]];
            }
        }

        Ok(())
    }

    /// Apply backward fill imputation
    fn apply_backward_fill(
        &self,
        features: &mut Array2<f32>,
        missing_mask: &Array2<bool>,
        feature_idx: usize,
    ) -> Result<()> {
        let num_samples = features.nrows();
        let mut next_valid_value = 0.0f32;

        for sample_idx in (0..num_samples).rev() {
            if missing_mask[[sample_idx, feature_idx]] {
                features[[sample_idx, feature_idx]] = next_valid_value;
            } else {
                next_valid_value = features[[sample_idx, feature_idx]];
            }
        }

        Ok(())
    }

    /// Apply linear interpolation
    fn apply_linear_interpolation(
        &self,
        features: &mut Array2<f32>,
        missing_mask: &Array2<bool>,
        feature_idx: usize,
    ) -> Result<()> {
        let num_samples = features.nrows();

        for sample_idx in 0..num_samples {
            if !missing_mask[[sample_idx, feature_idx]] {
                continue;
            }

            // Find previous valid value
            let mut prev_idx = None;
            let mut prev_value = 0.0f32;
            for i in (0..sample_idx).rev() {
                if !missing_mask[[i, feature_idx]] {
                    prev_idx = Some(i);
                    prev_value = features[[i, feature_idx]];
                    break;
                }
            }

            // Find next valid value
            let mut next_idx = None;
            let mut next_value = 0.0f32;
            for i in (sample_idx + 1)..num_samples {
                if !missing_mask[[i, feature_idx]] {
                    next_idx = Some(i);
                    next_value = features[[i, feature_idx]];
                    break;
                }
            }

            // Interpolate
            let interpolated_value = match (prev_idx, next_idx) {
                (Some(p_idx), Some(n_idx)) => {
                    let ratio = (sample_idx - p_idx) as f32 / (n_idx - p_idx) as f32;
                    prev_value + ratio * (next_value - prev_value)
                }
                (Some(_), None) => prev_value,
                (None, Some(_)) => next_value,
                (None, None) => 0.0,
            };

            features[[sample_idx, feature_idx]] = interpolated_value;
        }

        Ok(())
    }

    /// Apply KNN imputation
    fn apply_knn_imputation(
        &self,
        features: &mut Array2<f32>,
        missing_mask: &Array2<bool>,
        feature_idx: usize,
        knn_model: &KNNModel,
    ) -> Result<()> {
        let num_samples = features.nrows();
        let num_features = features.ncols();

        // Get indices of other features (excluding target feature)
        let other_features: Vec<usize> = (0..num_features).filter(|&i| i != feature_idx).collect();

        for sample_idx in 0..num_samples {
            if !missing_mask[[sample_idx, feature_idx]] {
                continue;
            }

            // Extract features for this sample
            let sample_features: Vec<f32> = other_features
                .iter()
                .map(|&feat_idx| features[[sample_idx, feat_idx]])
                .collect();

            // Skip if other features are also missing
            if sample_features
                .iter()
                .any(|&val| self.is_missing_value(val))
            {
                continue;
            }

            // Find k nearest neighbors
            let mut distances: Vec<(f32, f32)> = Vec::new();

            for ref_idx in 0..knn_model.reference_features.nrows() {
                let ref_features = knn_model.reference_features.row(ref_idx);
                let distance =
                    self.calculate_euclidean_distance(&sample_features, &ref_features.to_vec());
                let ref_value = knn_model.reference_values[ref_idx];
                distances.push((distance, ref_value));
            }

            // Sort by distance and take k nearest
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let k_nearest = distances.into_iter().take(knn_model.k).collect::<Vec<_>>();

            // Calculate weighted average (inverse distance weighting)
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for (distance, value) in k_nearest {
                let weight = if distance == 0.0 { 1e6 } else { 1.0 / distance };
                weighted_sum += weight * value;
                weight_sum += weight;
            }

            let imputed_value = if weight_sum > 0.0 {
                weighted_sum / weight_sum
            } else {
                0.0
            };

            features[[sample_idx, feature_idx]] = imputed_value;
        }

        Ok(())
    }

    /// Calculate Euclidean distance between two feature vectors
    fn calculate_euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Apply random imputation
    fn apply_random_imputation(
        &self,
        features: &mut Array2<f32>,
        missing_mask: &Array2<bool>,
        feature_idx: usize,
        fitted_params: &ImputationParameters,
    ) -> Result<()> {
        use rand::{Rng, SeedableRng};

        let mut rng = match fitted_params.rng_state {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };

        if let Some(stats) = fitted_params.feature_stats.get(&feature_idx) {
            let num_samples = features.nrows();

            for sample_idx in 0..num_samples {
                if missing_mask[[sample_idx, feature_idx]] {
                    if !stats.valid_values.is_empty() {
                        let random_idx = rng.gen_range(0..stats.valid_values.len());
                        features[[sample_idx, feature_idx]] = stats.valid_values[random_idx];
                    }
                }
            }
        }

        Ok(())
    }

    /// Remove features with high missing rates
    fn remove_features(
        &self,
        features: &Array2<f32>,
        removed_features: &[usize],
    ) -> Result<Array2<f32>> {
        if removed_features.is_empty() {
            return Ok(features.clone());
        }

        let num_samples = features.nrows();
        let num_features = features.ncols();
        let remaining_features: Vec<usize> = (0..num_features)
            .filter(|idx| !removed_features.contains(idx))
            .collect();

        let new_num_features = remaining_features.len();
        let mut new_features = Array2::<f32>::zeros((num_samples, new_num_features));

        for (new_idx, &old_idx) in remaining_features.iter().enumerate() {
            for sample_idx in 0..num_samples {
                new_features[[sample_idx, new_idx]] = features[[sample_idx, old_idx]];
            }
        }

        Ok(new_features)
    }

    /// Add missing value indicators
    fn add_missing_indicators(
        &self,
        features: &Array2<f32>,
        missing_mask: &Array2<bool>,
    ) -> Result<Array2<f32>> {
        let num_samples = features.nrows();
        let num_features = features.ncols();
        let new_num_features = num_features * 2; // Original + indicators

        let mut features_with_indicators = Array2::<f32>::zeros((num_samples, new_num_features));

        // Copy original features
        for sample_idx in 0..num_samples {
            for feature_idx in 0..num_features {
                features_with_indicators[[sample_idx, feature_idx]] =
                    features[[sample_idx, feature_idx]];
            }
        }

        // Add missing indicators
        for sample_idx in 0..num_samples {
            for feature_idx in 0..num_features {
                let indicator_idx = num_features + feature_idx;
                features_with_indicators[[sample_idx, indicator_idx]] =
                    if missing_mask[[sample_idx, feature_idx]] {
                        1.0
                    } else {
                        0.0
                    };
            }
        }

        Ok(features_with_indicators)
    }

    /// Update feature names after transformation
    fn update_feature_names(
        &self,
        dataset: &Dataset,
        fitted_params: &ImputationParameters,
    ) -> Result<Vec<String>> {
        let default_names = (0..dataset.num_features())
            .map(|i| format!("feature_{}", i))
            .collect::<Vec<_>>();
        let original_names = dataset.feature_names().unwrap_or(&default_names);

        // Remove high missing features
        let mut new_names = Vec::new();
        for (idx, name) in original_names.iter().enumerate() {
            if !fitted_params.removed_features.contains(&idx) {
                new_names.push(name.clone());
            }
        }

        // Add missing indicator names if configured
        if self.config.create_missing_indicators {
            let indicator_names: Vec<String> = new_names
                .iter()
                .map(|name| format!("{}_missing_indicator", name))
                .collect();
            new_names.extend(indicator_names);
        }

        Ok(new_names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_missing_value_config_default() {
        let config = MissingValueConfig::default();
        assert_eq!(config.default_strategy, ImputationStrategy::Mean);
        assert_eq!(config.missing_threshold, 0.8);
        assert!(!config.remove_high_missing_features);
        assert!(!config.create_missing_indicators);
    }

    #[test]
    fn test_imputation_strategy_constant() {
        let strategy = ImputationStrategy::Constant(42);
        match strategy {
            ImputationStrategy::Constant(value) => assert_eq!(value, 42),
            _ => panic!("Expected Constant strategy"),
        }
    }

    #[test]
    fn test_missing_value_detection() {
        let imputer = MissingValueImputer::default();

        assert!(imputer.is_missing_value(f32::NAN));
        assert!(imputer.is_missing_value(f32::INFINITY));
        assert!(imputer.is_missing_value(f32::NEG_INFINITY));
        assert!(!imputer.is_missing_value(1.0));
        assert!(!imputer.is_missing_value(0.0));
        assert!(!imputer.is_missing_value(-1.0));
    }

    #[test]
    fn test_feature_statistics_calculation() -> Result<()> {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, f32::NAN]);
        let missing = Array1::from_vec(vec![false, false, false, false, false, true]);

        let imputer = MissingValueImputer::default();
        let stats = imputer.calculate_feature_statistics(&data.view(), &missing.view())?;

        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.valid_values.len(), 5);

        Ok(())
    }

    #[test]
    fn test_euclidean_distance() {
        let imputer = MissingValueImputer::default();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 8.0];

        let distance = imputer.calculate_euclidean_distance(&a, &b);
        let expected = ((3.0_f32).powi(2) + (4.0_f32).powi(2) + (5.0_f32).powi(2)).sqrt();

        assert!((distance - expected).abs() < 1e-6);
    }
}
