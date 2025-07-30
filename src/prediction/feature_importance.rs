//! Feature importance calculation module for Pure Rust LightGBM.
//!
//! This module provides various methods for calculating feature importance
//! including split-based, gain-based, and SHAP-based importance metrics.

use crate::core::error::{LightGBMError, Result};
use crate::core::types::*;
use ndarray::{Array1, ArrayView2};

// Re-export ImportanceType from core::types for convenience
pub use crate::core::types::ImportanceType;

/// Feature importance calculator
#[derive(Debug)]
pub struct FeatureImportanceCalculator {
    model: Option<crate::boosting::GBDT>,
}

impl FeatureImportanceCalculator {
    /// Create a new feature importance calculator
    pub fn new(model: crate::boosting::GBDT) -> Self {
        Self { model: Some(model) }
    }

    /// Create calculator without model (for testing)
    pub fn new_without_model() -> Self {
        Self { model: None }
    }

    /// Calculate feature importance using specified method
    pub fn calculate_importance(&self, importance_type: &ImportanceType) -> Result<Array1<f64>> {
        match &self.model {
            Some(model) => {
                match importance_type {
                    ImportanceType::Split | ImportanceType::Gain | ImportanceType::Coverage | ImportanceType::TotalGain => {
                        // Use the model's built-in feature importance calculation
                        model.feature_importance(importance_type)
                    }
                    ImportanceType::Permutation => {
                        Err(LightGBMError::not_implemented(
                            "Permutation importance requires data samples - use calculate_permutation_importance instead"
                        ))
                    }
                }
            }
            None => Err(LightGBMError::prediction(
                "No model available for importance calculation",
            )),
        }
    }

    /// Calculate SHAP-based feature importance from data
    pub fn calculate_shap_importance(&self, features: &ArrayView2<'_, f32>) -> Result<Array1<f64>> {
        match &self.model {
            Some(model) => {
                let features_owned = features.to_owned();
                let shap_values = model.predict_contrib(&features_owned)?;

                // Calculate mean absolute SHAP value for each feature
                let num_features = shap_values.ncols();
                let mut importance = Array1::zeros(num_features);

                for feature_idx in 0..num_features {
                    let feature_column = shap_values.column(feature_idx);
                    let mean_abs_shap = feature_column.iter().map(|&x| x.abs()).sum::<f64>()
                        / feature_column.len() as f64;
                    importance[feature_idx] = mean_abs_shap;
                }

                Ok(importance)
            }
            None => Err(LightGBMError::prediction(
                "No model available for SHAP importance calculation",
            )),
        }
    }

    /// Calculate permutation-based feature importance
    pub fn calculate_permutation_importance(
        &self,
        features: &ArrayView2<'_, f32>,
        targets: &Array1<f32>,
        metric: PermutationMetric,
        n_repeats: usize,
    ) -> Result<Array1<f64>> {
        match &self.model {
            Some(model) => {
                let features_owned = features.to_owned();
                let baseline_predictions = model.predict(&features_owned)?;
                let baseline_score =
                    self.calculate_metric_score(&baseline_predictions, targets, &metric)?;

                let num_features = features.ncols();
                let mut importance_scores = Array1::zeros(num_features);

                for feature_idx in 0..num_features {
                    let mut permutation_scores = Vec::new();

                    for _ in 0..n_repeats {
                        // Create a copy of features with one column permuted
                        let mut permuted_features = features_owned.clone();
                        let mut feature_column = permuted_features.column_mut(feature_idx);

                        // Shuffle the feature column
                        self.shuffle_column(&mut feature_column);

                        // Make predictions with permuted feature
                        let permuted_predictions = model.predict(&permuted_features)?;
                        let permuted_score =
                            self.calculate_metric_score(&permuted_predictions, targets, &metric)?;

                        // Importance is the decrease in performance
                        let importance = baseline_score - permuted_score;
                        permutation_scores.push(importance);
                    }

                    // Average importance across repeats
                    importance_scores[feature_idx] =
                        permutation_scores.iter().sum::<f64>() / n_repeats as f64;
                }

                Ok(importance_scores)
            }
            None => Err(LightGBMError::prediction(
                "No model available for permutation importance calculation",
            )),
        }
    }

    /// Get feature importance ranking
    pub fn get_importance_ranking(
        &self,
        importance_type: &ImportanceType,
        feature_names: Option<&[String]>,
    ) -> Result<Vec<(usize, f64, Option<String>)>> {
        let importance = self.calculate_importance(importance_type)?;

        let mut ranking: Vec<(usize, f64, Option<String>)> = importance
            .iter()
            .enumerate()
            .map(|(idx, &score)| {
                let name = feature_names.and_then(|names| names.get(idx).cloned());
                (idx, score, name)
            })
            .collect();

        // Sort by importance score (descending)
        ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(ranking)
    }

    /// Calculate feature importance statistics
    pub fn calculate_importance_stats(
        &self,
        importance_type: &ImportanceType,
    ) -> Result<ImportanceStats> {
        let importance = self.calculate_importance(importance_type)?;

        if importance.is_empty() {
            return Ok(ImportanceStats::default());
        }

        let total_importance: f64 = importance.iter().sum();
        let max_importance = importance.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_importance = importance.iter().copied().fold(f64::INFINITY, f64::min);
        let mean_importance = total_importance / importance.len() as f64;

        // Calculate relative importance (normalized to sum to 1.0)
        let relative_importance = if total_importance > 0.0 {
            importance.mapv(|x| x / total_importance)
        } else {
            Array1::zeros(importance.len())
        };

        // Find top features
        let mut indexed_importance: Vec<(usize, f64)> = importance
            .iter()
            .enumerate()
            .map(|(idx, &score)| (idx, score))
            .collect();
        indexed_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_features: Vec<usize> = indexed_importance
            .iter()
            .take(10) // Top 10 features
            .map(|(idx, _)| *idx)
            .collect();

        Ok(ImportanceStats {
            total_importance,
            max_importance,
            min_importance,
            mean_importance,
            relative_importance,
            top_features,
        })
    }

    /// Helper method to shuffle a column (for permutation importance)
    fn shuffle_column(&self, column: &mut ndarray::ArrayViewMut1<'_, f32>) {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut values: Vec<f32> = column.iter().copied().collect();
        values.shuffle(&mut thread_rng());

        for (i, &value) in values.iter().enumerate() {
            column[i] = value;
        }
    }

    /// Helper method to calculate metric score
    fn calculate_metric_score(
        &self,
        predictions: &Array1<Score>,
        targets: &Array1<f32>,
        metric: &PermutationMetric,
    ) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(LightGBMError::invalid_parameter(
                "predictions",
                format!(
                    "length {} != targets length {}",
                    predictions.len(),
                    targets.len()
                ),
                "Predictions and targets must have same length",
            ));
        }

        match metric {
            PermutationMetric::MeanSquaredError => {
                let mse = predictions
                    .iter()
                    .zip(targets.iter())
                    .map(|(&pred, &target)| (pred - target).powi(2))
                    .sum::<f32>()
                    / predictions.len() as f32;
                Ok(-(mse as f64)) // Negative because we want higher scores to be better
            }
            PermutationMetric::MeanAbsoluteError => {
                let mae = predictions
                    .iter()
                    .zip(targets.iter())
                    .map(|(&pred, &target)| (pred - target).abs())
                    .sum::<f32>()
                    / predictions.len() as f32;
                Ok(-(mae as f64)) // Negative because we want higher scores to be better
            }
            PermutationMetric::Accuracy => {
                // For classification - assume predictions are probabilities and threshold at 0.5
                let correct = predictions
                    .iter()
                    .zip(targets.iter())
                    .map(|(&pred, &target)| {
                        let predicted_class = if pred > 0.5 { 1.0 } else { 0.0 };
                        if (predicted_class - target).abs() < 0.5 {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .sum::<f32>();
                Ok(correct as f64 / predictions.len() as f64)
            }
        }
    }
}

/// Metrics for permutation importance calculation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermutationMetric {
    MeanSquaredError,
    MeanAbsoluteError,
    Accuracy,
}

/// Feature importance statistics
#[derive(Debug, Clone)]
pub struct ImportanceStats {
    /// Total importance across all features
    pub total_importance: f64,
    /// Maximum importance score
    pub max_importance: f64,
    /// Minimum importance score
    pub min_importance: f64,
    /// Mean importance score
    pub mean_importance: f64,
    /// Relative importance (normalized to sum to 1.0)
    pub relative_importance: Array1<f64>,
    /// Indices of top features by importance
    pub top_features: Vec<usize>,
}

impl Default for ImportanceStats {
    fn default() -> Self {
        Self {
            total_importance: 0.0,
            max_importance: 0.0,
            min_importance: 0.0,
            mean_importance: 0.0,
            relative_importance: Array1::zeros(0),
            top_features: Vec::new(),
        }
    }
}

impl ImportanceStats {
    /// Get summary string of importance statistics
    pub fn summary(&self) -> String {
        format!(
            "Feature Importance Statistics:\n\
             - Total: {:.6}\n\
             - Max: {:.6}\n\
             - Min: {:.6}\n\
             - Mean: {:.6}\n\
             - Top features: {:?}",
            self.total_importance,
            self.max_importance,
            self.min_importance,
            self.mean_importance,
            self.top_features
        )
    }
}
