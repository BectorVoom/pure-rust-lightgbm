//! SHAP value computation module for Pure Rust LightGBM.
//!
//! This module provides comprehensive SHAP (SHapley Additive exPlanations) functionality
//! including TreeSHAP algorithm implementation, SHAP value calculation, and validation.

use crate::core::error::{LightGBMError, Result};

use ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};

/// Configuration for SHAP calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SHAPConfig {
    /// Whether to validate SHAP values sum correctly
    pub validate_shap_sums: bool,
    /// Tolerance for SHAP sum validation
    pub validation_tolerance: f64,
    /// Whether to compute SHAP interactions
    pub compute_interactions: bool,
    /// Number of background samples for expected value calculation
    pub background_samples: Option<usize>,
}

impl SHAPConfig {
    /// Create new SHAP configuration with defaults
    pub fn new() -> Self {
        Self {
            validate_shap_sums: true,
            validation_tolerance: 1e-6,
            compute_interactions: false,
            background_samples: None,
        }
    }

    /// Set SHAP validation enabled
    pub fn with_validation(mut self, enabled: bool) -> Self {
        self.validate_shap_sums = enabled;
        self
    }

    /// Set validation tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.validation_tolerance = tolerance;
        self
    }

    /// Set whether to compute interactions
    pub fn with_interactions(mut self, compute: bool) -> Self {
        self.compute_interactions = compute;
        self
    }

    /// Set number of background samples
    pub fn with_background_samples(mut self, samples: Option<usize>) -> Self {
        self.background_samples = samples;
        self
    }
}

impl Default for SHAPConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// SHAP calculator for feature importance analysis
#[derive(Debug)]
pub struct SHAPCalculator {
    config: SHAPConfig,
    model: Option<crate::boosting::GBDT>,
}

impl SHAPCalculator {
    /// Create a new SHAP calculator
    pub fn new(model: crate::boosting::GBDT, config: SHAPConfig) -> Self {
        Self {
            config,
            model: Some(model),
        }
    }

    /// Calculate SHAP values for a batch of samples
    pub fn calculate_shap_values(&self, features: &ArrayView2<'_, f32>) -> Result<Array2<f64>> {
        match &self.model {
            Some(model) => {
                let features_owned = features.to_owned();
                let shap_values = model.predict_contrib(&features_owned)?;

                if self.config.validate_shap_sums {
                    self.validate_shap_values(&features_owned, &shap_values)?;
                }

                Ok(shap_values)
            }
            None => Err(LightGBMError::prediction(
                "No model available for SHAP calculation",
            )),
        }
    }

    /// Calculate SHAP values for a single sample
    pub fn calculate_shap_single(&self, features: &[f32]) -> Result<Array1<f64>> {
        match &self.model {
            Some(model) => {
                let shap_values = model.predict_contrib_single(features)?;
                Ok(shap_values)
            }
            None => Err(LightGBMError::prediction(
                "No model available for SHAP calculation",
            )),
        }
    }

    /// Calculate SHAP interaction values
    pub fn calculate_shap_interactions(
        &self,
        features: &ArrayView2<'_, f32>,
    ) -> Result<ndarray::Array3<f64>> {
        if !self.config.compute_interactions {
            return Err(LightGBMError::config(
                "SHAP interactions not enabled in configuration",
            ));
        }

        match &self.model {
            Some(model) => {
                let features_owned = features.to_owned();
                let interactions = model.predict_contrib_interactions(&features_owned)?;
                Ok(interactions)
            }
            None => Err(LightGBMError::prediction(
                "No model available for SHAP interaction calculation",
            )),
        }
    }

    /// Get detailed SHAP explanation for a single prediction
    pub fn explain_prediction(&self, features: &[f32]) -> Result<crate::boosting::SHAPExplanation> {
        match &self.model {
            Some(model) => {
                let explanation = model.explain_prediction(features)?;
                Ok(explanation)
            }
            None => Err(LightGBMError::prediction(
                "No model available for SHAP explanation",
            )),
        }
    }

    /// Validate SHAP values sum correctly
    pub fn validate_shap_values(
        &self,
        features: &Array2<f32>,
        _shap_values: &Array2<f64>,
    ) -> Result<()> {
        match &self.model {
            Some(model) => {
                // TODO: Implement actual SHAP validation by comparing sum of SHAP values 
                // with difference between prediction and expected value
                let validation_stats = model.validate_shap_values(features)?;

                if validation_stats.mean_shap_error.abs() > self.config.validation_tolerance {
                    return Err(LightGBMError::data_validation(format!(
                        "SHAP validation failed: mean error {:.6} exceeds tolerance {:.6}",
                        validation_stats.mean_shap_error, self.config.validation_tolerance
                    )));
                }

                if validation_stats.max_shap_error.abs() > self.config.validation_tolerance * 10.0 {
                    return Err(LightGBMError::data_validation(format!(
                        "SHAP validation failed: max error {:.6} exceeds tolerance {:.6}",
                        validation_stats.max_shap_error,
                        self.config.validation_tolerance * 10.0
                    )));
                }

                log::debug!(
                    "SHAP validation passed: mean_error={:.6}, max_error={:.6}",
                    validation_stats.mean_shap_error,
                    validation_stats.max_shap_error
                );
                Ok(())
            }
            None => Err(LightGBMError::prediction(
                "No model available for SHAP validation",
            )),
        }
    }

    /// Get SHAP configuration
    pub fn config(&self) -> &SHAPConfig {
        &self.config
    }

    /// Check if model is available
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    /// Calculate expected value (base value) for SHAP calculations
    pub fn calculate_expected_value(
        &self,
        background_data: Option<&ArrayView2<'_, f32>>,
    ) -> Result<f64> {
        match &self.model {
            Some(model) => {
                if let Some(background) = background_data {
                    // Calculate expected value from background data
                    let background_owned = background.to_owned();
                    let predictions = model.predict(&background_owned)?;
                    let expected_value = predictions.iter().map(|&x| x as f64).sum::<f64>()
                        / predictions.len() as f64;
                    Ok(expected_value)
                } else {
                    // Use model's default base value
                    // This would need to be implemented in the GBDT model
                    Ok(0.0) // Placeholder - in practice, this should come from model training
                }
            }
            None => Err(LightGBMError::prediction(
                "No model available for expected value calculation",
            )),
        }
    }
}

/// Helper functions for SHAP calculations
pub mod helpers {
    use super::*;

    /// Sort features by absolute SHAP value importance
    pub fn sort_features_by_importance(
        shap_values: &Array1<f64>,
        feature_names: Option<&[String]>,
    ) -> Vec<(usize, f64, Option<String>)> {
        let mut feature_importance: Vec<(usize, f64, Option<String>)> = shap_values
            .iter()
            .enumerate()
            .map(|(idx, &value)| {
                let name = feature_names.and_then(|names| names.get(idx).cloned());
                (idx, value, name)
            })
            .collect();

        feature_importance.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        feature_importance
    }

    /// Calculate SHAP summary statistics
    pub fn calculate_shap_summary(shap_values: &Array2<f64>) -> crate::boosting::SHAPSummaryStats {
        let flat_values: Vec<f64> = shap_values.iter().copied().collect();

        if flat_values.is_empty() {
            return crate::boosting::SHAPSummaryStats {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
                total_absolute_contribution: 0.0,
                additivity_error: 0.0,
            };
        }

        let mut sorted_values = flat_values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted_values[0];
        let max = sorted_values[sorted_values.len() - 1];
        let mean = flat_values.iter().sum::<f64>() / flat_values.len() as f64;
        let median = if sorted_values.len() % 2 == 0 {
            (sorted_values[sorted_values.len() / 2 - 1] + sorted_values[sorted_values.len() / 2])
                / 2.0
        } else {
            sorted_values[sorted_values.len() / 2]
        };

        let variance =
            flat_values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / flat_values.len() as f64;
        let std_dev = variance.sqrt();
        let total_absolute_contribution = flat_values.iter().map(|&x| x.abs()).sum::<f64>();

        crate::boosting::SHAPSummaryStats {
            min,
            max,
            mean,
            median,
            std_dev,
            total_absolute_contribution,
            additivity_error: 0.0, // Would need prediction comparison to calculate
        }
    }
}
