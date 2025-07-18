//! Regression metrics for model evaluation.
//!
//! This module provides comprehensive regression metrics including
//! MSE, RMSE, MAE, MAPE, R², and other statistical measures for
//! evaluating regression model performance.

use crate::core::{
    types::*,
    error::{LightGBMError, Result},
    traits::*,
};
use ndarray::{Array1, ArrayView1};
use std::collections::HashMap;

/// Regression metrics calculator.
#[derive(Debug, Clone)]
pub struct RegressionMetrics {
    /// Configuration for metric calculation
    config: RegressionMetricsConfig,
    /// Cached metric values
    cached_metrics: HashMap<String, f64>,
}

/// Configuration for regression metrics calculation.
#[derive(Debug, Clone)]
pub struct RegressionMetricsConfig {
    /// Whether to compute all metrics or just basic ones
    pub compute_all: bool,
    /// Whether to handle missing values
    pub handle_missing: bool,
    /// Tolerance for numerical stability
    pub epsilon: f64,
    /// Whether to use sample weights
    pub use_weights: bool,
    /// Whether to compute confidence intervals
    pub compute_confidence_intervals: bool,
    /// Confidence level for intervals (e.g., 0.95 for 95%)
    pub confidence_level: f64,
}

impl Default for RegressionMetricsConfig {
    fn default() -> Self {
        Self {
            compute_all: true,
            handle_missing: true,
            epsilon: 1e-15,
            use_weights: false,
            compute_confidence_intervals: false,
            confidence_level: 0.95,
        }
    }
}

impl RegressionMetrics {
    /// Create a new regression metrics calculator.
    pub fn new(config: RegressionMetricsConfig) -> Self {
        Self {
            config,
            cached_metrics: HashMap::new(),
        }
    }

    /// Calculate all regression metrics.
    pub fn calculate(
        &mut self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<RegressionMetricsResult> {
        self.validate_inputs(predictions, targets, weights)?;
        
        let mut result = RegressionMetricsResult::new();
        
        // Calculate basic metrics
        result.mse = self.mean_squared_error(predictions, targets, weights)?;
        result.rmse = result.mse.sqrt();
        result.mae = self.mean_absolute_error(predictions, targets, weights)?;
        result.r_squared = self.r_squared(predictions, targets, weights)?;
        
        if self.config.compute_all {
            result.mape = self.mean_absolute_percentage_error(predictions, targets, weights)?;
            result.msle = self.mean_squared_log_error(predictions, targets, weights)?;
            result.explained_variance = self.explained_variance_score(predictions, targets, weights)?;
            result.median_absolute_error = self.median_absolute_error(predictions, targets)?;
            result.mean_absolute_scaled_error = self.mean_absolute_scaled_error(predictions, targets)?;
            result.mean_squared_scaled_error = self.mean_squared_scaled_error(predictions, targets)?;
            result.symmetric_mean_absolute_percentage_error = self.symmetric_mape(predictions, targets, weights)?;
            result.mean_directional_accuracy = self.mean_directional_accuracy(predictions, targets)?;
            result.theil_u_statistic = self.theil_u_statistic(predictions, targets)?;
            result.coefficient_of_determination = self.coefficient_of_determination(predictions, targets, weights)?;
            result.mean_bias_error = self.mean_bias_error(predictions, targets, weights)?;
            result.mean_percentage_error = self.mean_percentage_error(predictions, targets, weights)?;
            result.root_mean_squared_log_error = self.root_mean_squared_log_error(predictions, targets, weights)?;
            result.normalized_root_mean_squared_error = self.normalized_rmse(predictions, targets, weights)?;
            result.mean_gamma_deviance = self.mean_gamma_deviance(predictions, targets, weights)?;
            result.mean_poisson_deviance = self.mean_poisson_deviance(predictions, targets, weights)?;
            result.mean_tweedie_deviance = self.mean_tweedie_deviance(predictions, targets, weights, 1.5)?;
            result.d2_absolute_error_score = self.d2_absolute_error_score(predictions, targets, weights)?;
            result.d2_pinball_score = self.d2_pinball_score(predictions, targets, weights, 0.5)?;
            result.d2_tweedie_score = self.d2_tweedie_score(predictions, targets, weights, 1.5)?;
        }
        
        // Calculate sample size
        result.sample_size = predictions.len();
        
        // Calculate confidence intervals if requested
        if self.config.compute_confidence_intervals {
            result.confidence_intervals = Some(self.calculate_confidence_intervals(predictions, targets, weights)?);
        }
        
        Ok(result)
    }

    /// Validate input arrays.
    fn validate_inputs(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<()> {
        if predictions.len() != targets.len() {
            return Err(LightGBMError::dimension_mismatch(
                format!("predictions: {}", predictions.len()),
                format!("targets: {}", targets.len()),
            ));
        }
        
        if let Some(w) = weights {
            if w.len() != predictions.len() {
                return Err(LightGBMError::dimension_mismatch(
                    format!("weights: {}", w.len()),
                    format!("predictions: {}", predictions.len()),
                ));
            }
        }
        
        if predictions.is_empty() {
            return Err(LightGBMError::config("Empty input arrays"));
        }
        
        Ok(())
    }

    /// Calculate Mean Squared Error.
    pub fn mean_squared_error(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<f64> {
        if let Some(w) = weights {
            let weighted_sum = predictions.iter()
                .zip(targets.iter())
                .zip(w.iter())
                .map(|((&pred, &target), &weight)| weight * (pred - target).powi(2))
                .sum::<f32>();
            let weight_sum = w.sum();
            Ok((weighted_sum / weight_sum) as f64)
        } else {
            let sum = predictions.iter()
                .zip(targets.iter())
                .map(|(&pred, &target)| (pred - target).powi(2))
                .sum::<f32>();
            Ok((sum / predictions.len() as f32) as f64)
        }
    }

    /// Calculate Mean Absolute Error.
    pub fn mean_absolute_error(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<f64> {
        if let Some(w) = weights {
            let weighted_sum = predictions.iter()
                .zip(targets.iter())
                .zip(w.iter())
                .map(|((&pred, &target), &weight)| weight * (pred - target).abs())
                .sum::<f32>();
            let weight_sum = w.sum();
            Ok((weighted_sum / weight_sum) as f64)
        } else {
            let sum = predictions.iter()
                .zip(targets.iter())
                .map(|(&pred, &target)| (pred - target).abs())
                .sum::<f32>();
            Ok((sum / predictions.len() as f32) as f64)
        }
    }

    /// Calculate R-squared (coefficient of determination).
    pub fn r_squared(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<f64> {
        let target_mean = if let Some(w) = weights {
            let weighted_sum = targets.iter().zip(w.iter()).map(|(&t, &w)| w * t).sum::<f32>();
            let weight_sum = w.sum();
            weighted_sum / weight_sum
        } else {
            targets.sum() / targets.len() as f32
        };

        let ss_res = if let Some(w) = weights {
            predictions.iter()
                .zip(targets.iter())
                .zip(w.iter())
                .map(|((&pred, &target), &weight)| weight * (target - pred).powi(2))
                .sum::<f32>()
        } else {
            predictions.iter()
                .zip(targets.iter())
                .map(|(&pred, &target)| (target - pred).powi(2))
                .sum::<f32>()
        };

        let ss_tot = if let Some(w) = weights {
            targets.iter()
                .zip(w.iter())
                .map(|(&target, &weight)| weight * (target - target_mean).powi(2))
                .sum::<f32>()
        } else {
            targets.iter()
                .map(|&target| (target - target_mean).powi(2))
                .sum::<f32>()
        };

        if ss_tot.abs() < self.config.epsilon as f32 {
            Ok(0.0)
        } else {
            Ok((1.0 - ss_res / ss_tot) as f64)
        }
    }

    /// Calculate Mean Absolute Percentage Error.
    pub fn mean_absolute_percentage_error(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<f64> {
        if let Some(w) = weights {
            let weighted_sum = predictions.iter()
                .zip(targets.iter())
                .zip(w.iter())
                .filter(|((&_pred, &target), &_weight)| target.abs() > self.config.epsilon as f32)
                .map(|((&pred, &target), &weight)| weight * (pred - target).abs() / target.abs())
                .sum::<f32>();
            let weight_sum = w.iter()
                .zip(targets.iter())
                .filter(|(&_weight, &target)| target.abs() > self.config.epsilon as f32)
                .map(|(&weight, &_target)| weight)
                .sum::<f32>();
            Ok((weighted_sum / weight_sum * 100.0) as f64)
        } else {
            let sum = predictions.iter()
                .zip(targets.iter())
                .filter(|(&_pred, &target)| target.abs() > self.config.epsilon as f32)
                .map(|(&pred, &target)| (pred - target).abs() / target.abs())
                .sum::<f32>();
            let count = predictions.iter()
                .zip(targets.iter())
                .filter(|(&_pred, &target)| target.abs() > self.config.epsilon as f32)
                .count();
            Ok((sum / count as f32 * 100.0) as f64)
        }
    }

    /// Calculate Mean Squared Log Error.
    pub fn mean_squared_log_error(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<f64> {
        if let Some(w) = weights {
            let weighted_sum = predictions.iter()
                .zip(targets.iter())
                .zip(w.iter())
                .filter(|((&pred, &target), &_weight)| pred > 0.0 && target > 0.0)
                .map(|((&pred, &target), &weight)| weight * (pred.ln() - target.ln()).powi(2))
                .sum::<f32>();
            let weight_sum = w.iter()
                .zip(predictions.iter())
                .zip(targets.iter())
                .filter(|((_, &pred), &target)| pred > 0.0 && target > 0.0)
                .map(|((&weight, &_pred), &_target)| weight)
                .sum::<f32>();
            Ok((weighted_sum / weight_sum) as f64)
        } else {
            let sum = predictions.iter()
                .zip(targets.iter())
                .filter(|(&pred, &target)| pred > 0.0 && target > 0.0)
                .map(|(&pred, &target)| (pred.ln() - target.ln()).powi(2))
                .sum::<f32>();
            let count = predictions.iter()
                .zip(targets.iter())
                .filter(|(&pred, &target)| pred > 0.0 && target > 0.0)
                .count();
            Ok((sum / count as f32) as f64)
        }
    }

    /// Calculate Explained Variance Score.
    pub fn explained_variance_score(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<f64> {
        let target_mean = if let Some(w) = weights {
            let weighted_sum = targets.iter().zip(w.iter()).map(|(&t, &w)| w * t).sum::<f32>();
            let weight_sum = w.sum();
            weighted_sum / weight_sum
        } else {
            targets.sum() / targets.len() as f32
        };

        let residual_variance = if let Some(w) = weights {
            let weighted_sum = predictions.iter()
                .zip(targets.iter())
                .zip(w.iter())
                .map(|((&pred, &target), &weight)| weight * (target - pred).powi(2))
                .sum::<f32>();
            let weight_sum = w.sum();
            weighted_sum / weight_sum
        } else {
            predictions.iter()
                .zip(targets.iter())
                .map(|(&pred, &target)| (target - pred).powi(2))
                .sum::<f32>() / predictions.len() as f32
        };

        let target_variance = if let Some(w) = weights {
            let weighted_sum = targets.iter()
                .zip(w.iter())
                .map(|(&target, &weight)| weight * (target - target_mean).powi(2))
                .sum::<f32>();
            let weight_sum = w.sum();
            weighted_sum / weight_sum
        } else {
            targets.iter()
                .map(|&target| (target - target_mean).powi(2))
                .sum::<f32>() / targets.len() as f32
        };

        if target_variance.abs() < self.config.epsilon as f32 {
            Ok(0.0)
        } else {
            Ok((1.0 - residual_variance / target_variance) as f64)
        }
    }

    /// Calculate Median Absolute Error.
    pub fn median_absolute_error(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
    ) -> Result<f64> {
        let mut errors: Vec<f32> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred - target).abs())
            .collect();
        
        errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let median = if errors.len() % 2 == 0 {
            (errors[errors.len() / 2 - 1] + errors[errors.len() / 2]) / 2.0
        } else {
            errors[errors.len() / 2]
        };
        
        Ok(median as f64)
    }

    /// Calculate Mean Absolute Scaled Error.
    pub fn mean_absolute_scaled_error(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
    ) -> Result<f64> {
        if targets.len() < 2 {
            return Ok(f64::NAN);
        }

        let mae = self.mean_absolute_error(predictions, targets, None)?;
        
        // Calculate naive forecast error (using lag-1 as naive forecast)
        let naive_mae = targets.windows(2)
            .map(|window| (window[1] - window[0]).abs())
            .sum::<f32>() / (targets.len() - 1) as f32;
        
        if naive_mae.abs() < self.config.epsilon as f32 {
            Ok(f64::NAN)
        } else {
            Ok(mae / naive_mae as f64)
        }
    }

    /// Calculate Mean Squared Scaled Error.
    pub fn mean_squared_scaled_error(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
    ) -> Result<f64> {
        if targets.len() < 2 {
            return Ok(f64::NAN);
        }

        let mse = self.mean_squared_error(predictions, targets, None)?;
        
        // Calculate naive forecast error (using lag-1 as naive forecast)
        let naive_mse = targets.windows(2)
            .map(|window| (window[1] - window[0]).powi(2))
            .sum::<f32>() / (targets.len() - 1) as f32;
        
        if naive_mse.abs() < self.config.epsilon as f32 {
            Ok(f64::NAN)
        } else {
            Ok(mse / naive_mse as f64)
        }
    }

    /// Calculate Symmetric Mean Absolute Percentage Error.
    pub fn symmetric_mape(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<f64> {
        if let Some(w) = weights {
            let weighted_sum = predictions.iter()
                .zip(targets.iter())
                .zip(w.iter())
                .map(|((&pred, &target), &weight)| {
                    let denominator = (pred.abs() + target.abs()) / 2.0;
                    if denominator.abs() > self.config.epsilon as f32 {
                        weight * (pred - target).abs() / denominator
                    } else {
                        0.0
                    }
                })
                .sum::<f32>();
            let weight_sum = w.sum();
            Ok((weighted_sum / weight_sum * 100.0) as f64)
        } else {
            let sum = predictions.iter()
                .zip(targets.iter())
                .map(|(&pred, &target)| {
                    let denominator = (pred.abs() + target.abs()) / 2.0;
                    if denominator.abs() > self.config.epsilon as f32 {
                        (pred - target).abs() / denominator
                    } else {
                        0.0
                    }
                })
                .sum::<f32>();
            Ok((sum / predictions.len() as f32 * 100.0) as f64)
        }
    }

    /// Calculate Mean Directional Accuracy.
    pub fn mean_directional_accuracy(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
    ) -> Result<f64> {
        if targets.len() < 2 {
            return Ok(f64::NAN);
        }

        let mut correct_directions = 0;
        let mut total_comparisons = 0;

        for i in 1..targets.len() {
            let actual_direction = targets[i] - targets[i - 1];
            let predicted_direction = predictions[i] - predictions[i - 1];
            
            if actual_direction * predicted_direction > 0.0 {
                correct_directions += 1;
            }
            total_comparisons += 1;
        }

        Ok(correct_directions as f64 / total_comparisons as f64)
    }

    /// Calculate Theil's U statistic.
    pub fn theil_u_statistic(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
    ) -> Result<f64> {
        if targets.len() < 2 {
            return Ok(f64::NAN);
        }

        let mse = self.mean_squared_error(predictions, targets, None)?;
        let rmse = mse.sqrt();
        
        // Calculate naive forecast RMSE (using lag-1 as naive forecast)
        let naive_mse = targets.windows(2)
            .map(|window| (window[1] - window[0]).powi(2))
            .sum::<f32>() / (targets.len() - 1) as f32;
        let naive_rmse = naive_mse.sqrt();
        
        if naive_rmse.abs() < self.config.epsilon as f32 {
            Ok(f64::NAN)
        } else {
            Ok(rmse / naive_rmse as f64)
        }
    }

    /// Calculate Coefficient of Determination (alternative R²).
    pub fn coefficient_of_determination(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<f64> {
        self.r_squared(predictions, targets, weights)
    }

    /// Calculate Mean Bias Error.
    pub fn mean_bias_error(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<f64> {
        if let Some(w) = weights {
            let weighted_sum = predictions.iter()
                .zip(targets.iter())
                .zip(w.iter())
                .map(|((&pred, &target), &weight)| weight * (pred - target))
                .sum::<f32>();
            let weight_sum = w.sum();
            Ok((weighted_sum / weight_sum) as f64)
        } else {
            let sum = predictions.iter()
                .zip(targets.iter())
                .map(|(&pred, &target)| pred - target)
                .sum::<f32>();
            Ok((sum / predictions.len() as f32) as f64)
        }
    }

    /// Calculate Mean Percentage Error.
    pub fn mean_percentage_error(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<f64> {
        if let Some(w) = weights {
            let weighted_sum = predictions.iter()
                .zip(targets.iter())
                .zip(w.iter())
                .filter(|((&_pred, &target), &_weight)| target.abs() > self.config.epsilon as f32)
                .map(|((&pred, &target), &weight)| weight * (pred - target) / target)
                .sum::<f32>();
            let weight_sum = w.iter()
                .zip(targets.iter())
                .filter(|(&_weight, &target)| target.abs() > self.config.epsilon as f32)
                .map(|(&weight, &_target)| weight)
                .sum::<f32>();
            Ok((weighted_sum / weight_sum * 100.0) as f64)
        } else {
            let sum = predictions.iter()
                .zip(targets.iter())
                .filter(|(&_pred, &target)| target.abs() > self.config.epsilon as f32)
                .map(|(&pred, &target)| (pred - target) / target)
                .sum::<f32>();
            let count = predictions.iter()
                .zip(targets.iter())
                .filter(|(&_pred, &target)| target.abs() > self.config.epsilon as f32)
                .count();
            Ok((sum / count as f32 * 100.0) as f64)
        }
    }

    /// Calculate Root Mean Squared Log Error.
    pub fn root_mean_squared_log_error(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<f64> {
        let msle = self.mean_squared_log_error(predictions, targets, weights)?;
        Ok(msle.sqrt())
    }

    /// Calculate Normalized Root Mean Squared Error.
    pub fn normalized_rmse(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<f64> {
        let rmse = self.mean_squared_error(predictions, targets, weights)?.sqrt();
        let target_range = targets.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
            - targets.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        
        if target_range.abs() < self.config.epsilon as f32 {
            Ok(f64::NAN)
        } else {
            Ok(rmse / target_range as f64)
        }
    }

    /// Calculate Mean Gamma Deviance.
    pub fn mean_gamma_deviance(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<f64> {
        if let Some(w) = weights {
            let weighted_sum = predictions.iter()
                .zip(targets.iter())
                .zip(w.iter())
                .filter(|((&pred, &target), &_weight)| pred > 0.0 && target > 0.0)
                .map(|((&pred, &target), &weight)| weight * 2.0 * (target / pred - target.ln() + pred.ln() - 1.0))
                .sum::<f32>();
            let weight_sum = w.iter()
                .zip(predictions.iter())
                .zip(targets.iter())
                .filter(|((_, &pred), &target)| pred > 0.0 && target > 0.0)
                .map(|((&weight, &_pred), &_target)| weight)
                .sum::<f32>();
            Ok((weighted_sum / weight_sum) as f64)
        } else {
            let sum = predictions.iter()
                .zip(targets.iter())
                .filter(|(&pred, &target)| pred > 0.0 && target > 0.0)
                .map(|(&pred, &target)| 2.0 * (target / pred - target.ln() + pred.ln() - 1.0))
                .sum::<f32>();
            let count = predictions.iter()
                .zip(targets.iter())
                .filter(|(&pred, &target)| pred > 0.0 && target > 0.0)
                .count();
            Ok((sum / count as f32) as f64)
        }
    }

    /// Calculate Mean Poisson Deviance.
    pub fn mean_poisson_deviance(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<f64> {
        if let Some(w) = weights {
            let weighted_sum = predictions.iter()
                .zip(targets.iter())
                .zip(w.iter())
                .filter(|((&pred, &target), &_weight)| pred > 0.0 && target >= 0.0)
                .map(|((&pred, &target), &weight)| {
                    weight * 2.0 * (target * (target / pred).ln() - target + pred)
                })
                .sum::<f32>();
            let weight_sum = w.iter()
                .zip(predictions.iter())
                .zip(targets.iter())
                .filter(|((_, &pred), &target)| pred > 0.0 && target >= 0.0)
                .map(|((&weight, &_pred), &_target)| weight)
                .sum::<f32>();
            Ok((weighted_sum / weight_sum) as f64)
        } else {
            let sum = predictions.iter()
                .zip(targets.iter())
                .filter(|(&pred, &target)| pred > 0.0 && target >= 0.0)
                .map(|(&pred, &target)| 2.0 * (target * (target / pred).ln() - target + pred))
                .sum::<f32>();
            let count = predictions.iter()
                .zip(targets.iter())
                .filter(|(&pred, &target)| pred > 0.0 && target >= 0.0)
                .count();
            Ok((sum / count as f32) as f64)
        }
    }

    /// Calculate Mean Tweedie Deviance.
    pub fn mean_tweedie_deviance(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
        power: f64,
    ) -> Result<f64> {
        if power < 1.0 || power > 2.0 {
            return Err(LightGBMError::config("Tweedie power must be between 1 and 2"));
        }

        if let Some(w) = weights {
            let weighted_sum = predictions.iter()
                .zip(targets.iter())
                .zip(w.iter())
                .filter(|((&pred, &target), &_weight)| pred > 0.0 && target >= 0.0)
                .map(|((&pred, &target), &weight)| {
                    let term1 = if power == 1.0 { 
                        target * (target / pred).ln() 
                    } else { 
                        target.powf(2.0 - power as f32) / (2.0 - power as f32) 
                    };
                    let term2 = if power == 2.0 { 
                        pred.ln() 
                    } else { 
                        pred.powf(2.0 - power as f32) / (2.0 - power as f32) 
                    };
                    let term3 = if power == 1.0 { 
                        target.ln() 
                    } else { 
                        target.powf(1.0 - power as f32) / (1.0 - power as f32) 
                    };
                    
                    weight * 2.0 * (term1 - term2 - term3)
                })
                .sum::<f32>();
            let weight_sum = w.iter()
                .zip(predictions.iter())
                .zip(targets.iter())
                .filter(|((_, &pred), &target)| pred > 0.0 && target >= 0.0)
                .map(|((&weight, &_pred), &_target)| weight)
                .sum::<f32>();
            Ok((weighted_sum / weight_sum) as f64)
        } else {
            let sum = predictions.iter()
                .zip(targets.iter())
                .filter(|(&pred, &target)| pred > 0.0 && target >= 0.0)
                .map(|(&pred, &target)| {
                    let term1 = if power == 1.0 { 
                        target * (target / pred).ln() 
                    } else { 
                        target.powf(2.0 - power as f32) / (2.0 - power as f32) 
                    };
                    let term2 = if power == 2.0 { 
                        pred.ln() 
                    } else { 
                        pred.powf(2.0 - power as f32) / (2.0 - power as f32) 
                    };
                    let term3 = if power == 1.0 { 
                        target.ln() 
                    } else { 
                        target.powf(1.0 - power as f32) / (1.0 - power as f32) 
                    };
                    
                    2.0 * (term1 - term2 - term3)
                })
                .sum::<f32>();
            let count = predictions.iter()
                .zip(targets.iter())
                .filter(|(&pred, &target)| pred > 0.0 && target >= 0.0)
                .count();
            Ok((sum / count as f32) as f64)
        }
    }

    /// Calculate D² Absolute Error Score.
    pub fn d2_absolute_error_score(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<f64> {
        let mae = self.mean_absolute_error(predictions, targets, weights)?;
        
        let target_median = {
            let mut sorted_targets = targets.to_vec();
            sorted_targets.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if sorted_targets.len() % 2 == 0 {
                (sorted_targets[sorted_targets.len() / 2 - 1] + sorted_targets[sorted_targets.len() / 2]) / 2.0
            } else {
                sorted_targets[sorted_targets.len() / 2]
            }
        };
        
        let mae_baseline = self.mean_absolute_error(
            &Array1::from_elem(targets.len(), target_median).view(),
            targets,
            weights,
        )?;
        
        if mae_baseline.abs() < self.config.epsilon {
            Ok(0.0)
        } else {
            Ok(1.0 - mae / mae_baseline)
        }
    }

    /// Calculate D² Pinball Score.
    pub fn d2_pinball_score(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
        alpha: f64,
    ) -> Result<f64> {
        let pinball_loss = |pred: f32, target: f32, alpha: f32| {
            let error = target - pred;
            if error > 0.0 {
                alpha * error
            } else {
                (alpha - 1.0) * error
            }
        };

        let loss = if let Some(w) = weights {
            let weighted_sum = predictions.iter()
                .zip(targets.iter())
                .zip(w.iter())
                .map(|((&pred, &target), &weight)| weight * pinball_loss(pred, target, alpha as f32))
                .sum::<f32>();
            let weight_sum = w.sum();
            weighted_sum / weight_sum
        } else {
            predictions.iter()
                .zip(targets.iter())
                .map(|(&pred, &target)| pinball_loss(pred, target, alpha as f32))
                .sum::<f32>() / predictions.len() as f32
        };

        // Calculate baseline (using quantile as prediction)
        let mut sorted_targets = targets.to_vec();
        sorted_targets.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let quantile_idx = ((sorted_targets.len() - 1) as f64 * alpha) as usize;
        let baseline_pred = sorted_targets[quantile_idx];
        
        let baseline_loss = if let Some(w) = weights {
            let weighted_sum = targets.iter()
                .zip(w.iter())
                .map(|(&target, &weight)| weight * pinball_loss(baseline_pred, target, alpha as f32))
                .sum::<f32>();
            let weight_sum = w.sum();
            weighted_sum / weight_sum
        } else {
            targets.iter()
                .map(|&target| pinball_loss(baseline_pred, target, alpha as f32))
                .sum::<f32>() / targets.len() as f32
        };

        if baseline_loss.abs() < self.config.epsilon as f32 {
            Ok(0.0)
        } else {
            Ok((1.0 - loss / baseline_loss) as f64)
        }
    }

    /// Calculate D² Tweedie Score.
    pub fn d2_tweedie_score(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
        power: f64,
    ) -> Result<f64> {
        let tweedie_deviance = self.mean_tweedie_deviance(predictions, targets, weights, power)?;
        
        let target_mean = if let Some(w) = weights {
            let weighted_sum = targets.iter().zip(w.iter()).map(|(&t, &w)| w * t).sum::<f32>();
            let weight_sum = w.sum();
            weighted_sum / weight_sum
        } else {
            targets.sum() / targets.len() as f32
        };
        
        let baseline_deviance = self.mean_tweedie_deviance(
            &Array1::from_elem(targets.len(), target_mean).view(),
            targets,
            weights,
            power,
        )?;
        
        if baseline_deviance.abs() < self.config.epsilon {
            Ok(0.0)
        } else {
            Ok(1.0 - tweedie_deviance / baseline_deviance)
        }
    }

    /// Calculate confidence intervals for metrics.
    fn calculate_confidence_intervals(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<RegressionConfidenceIntervals> {
        // Simple bootstrap confidence intervals
        let n_bootstrap = 1000;
        let n_samples = predictions.len();
        let mut bootstrap_mse = Vec::new();
        let mut bootstrap_mae = Vec::new();
        let mut bootstrap_r2 = Vec::new();
        
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::from_entropy();
        
        for _ in 0..n_bootstrap {
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);
            
            let bootstrap_predictions: Vec<f32> = indices.iter()
                .map(|&i| predictions[i])
                .collect();
            let bootstrap_targets: Vec<f32> = indices.iter()
                .map(|&i| targets[i])
                .collect();
            let bootstrap_weights: Option<Vec<f32>> = weights.map(|w| {
                indices.iter().map(|&i| w[i]).collect()
            });
            
            let bootstrap_pred_view = Array1::from_vec(bootstrap_predictions);
            let bootstrap_target_view = Array1::from_vec(bootstrap_targets);
            let bootstrap_weight_view = bootstrap_weights.as_ref().map(|w| Array1::from_vec(w.clone()));
            
            let mse = self.mean_squared_error(
                &bootstrap_pred_view.view(),
                &bootstrap_target_view.view(),
                bootstrap_weight_view.as_ref().map(|w| w.view()),
            )?;
            let mae = self.mean_absolute_error(
                &bootstrap_pred_view.view(),
                &bootstrap_target_view.view(),
                bootstrap_weight_view.as_ref().map(|w| w.view()),
            )?;
            let r2 = self.r_squared(
                &bootstrap_pred_view.view(),
                &bootstrap_target_view.view(),
                bootstrap_weight_view.as_ref().map(|w| w.view()),
            )?;
            
            bootstrap_mse.push(mse);
            bootstrap_mae.push(mae);
            bootstrap_r2.push(r2);
        }
        
        bootstrap_mse.sort_by(|a, b| a.partial_cmp(b).unwrap());
        bootstrap_mae.sort_by(|a, b| a.partial_cmp(b).unwrap());
        bootstrap_r2.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let alpha = 1.0 - self.config.confidence_level;
        let lower_percentile = (alpha / 2.0 * n_bootstrap as f64) as usize;
        let upper_percentile = ((1.0 - alpha / 2.0) * n_bootstrap as f64) as usize;
        
        Ok(RegressionConfidenceIntervals {
            confidence_level: self.config.confidence_level,
            mse_interval: (bootstrap_mse[lower_percentile], bootstrap_mse[upper_percentile]),
            mae_interval: (bootstrap_mae[lower_percentile], bootstrap_mae[upper_percentile]),
            r2_interval: (bootstrap_r2[lower_percentile], bootstrap_r2[upper_percentile]),
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &RegressionMetricsConfig {
        &self.config
    }

    /// Set the configuration.
    pub fn set_config(&mut self, config: RegressionMetricsConfig) {
        self.config = config;
        self.cached_metrics.clear();
    }
}

/// Result of regression metrics calculation.
#[derive(Debug, Clone)]
pub struct RegressionMetricsResult {
    /// Mean Squared Error
    pub mse: f64,
    /// Root Mean Squared Error
    pub rmse: f64,
    /// Mean Absolute Error
    pub mae: f64,
    /// R-squared (coefficient of determination)
    pub r_squared: f64,
    /// Mean Absolute Percentage Error
    pub mape: f64,
    /// Mean Squared Log Error
    pub msle: f64,
    /// Explained Variance Score
    pub explained_variance: f64,
    /// Median Absolute Error
    pub median_absolute_error: f64,
    /// Mean Absolute Scaled Error
    pub mean_absolute_scaled_error: f64,
    /// Mean Squared Scaled Error
    pub mean_squared_scaled_error: f64,
    /// Symmetric Mean Absolute Percentage Error
    pub symmetric_mean_absolute_percentage_error: f64,
    /// Mean Directional Accuracy
    pub mean_directional_accuracy: f64,
    /// Theil's U statistic
    pub theil_u_statistic: f64,
    /// Coefficient of Determination
    pub coefficient_of_determination: f64,
    /// Mean Bias Error
    pub mean_bias_error: f64,
    /// Mean Percentage Error
    pub mean_percentage_error: f64,
    /// Root Mean Squared Log Error
    pub root_mean_squared_log_error: f64,
    /// Normalized Root Mean Squared Error
    pub normalized_root_mean_squared_error: f64,
    /// Mean Gamma Deviance
    pub mean_gamma_deviance: f64,
    /// Mean Poisson Deviance
    pub mean_poisson_deviance: f64,
    /// Mean Tweedie Deviance
    pub mean_tweedie_deviance: f64,
    /// D² Absolute Error Score
    pub d2_absolute_error_score: f64,
    /// D² Pinball Score
    pub d2_pinball_score: f64,
    /// D² Tweedie Score
    pub d2_tweedie_score: f64,
    /// Sample size
    pub sample_size: usize,
    /// Confidence intervals (if computed)
    pub confidence_intervals: Option<RegressionConfidenceIntervals>,
}

impl RegressionMetricsResult {
    /// Create a new result with default values.
    pub fn new() -> Self {
        Self {
            mse: 0.0,
            rmse: 0.0,
            mae: 0.0,
            r_squared: 0.0,
            mape: 0.0,
            msle: 0.0,
            explained_variance: 0.0,
            median_absolute_error: 0.0,
            mean_absolute_scaled_error: 0.0,
            mean_squared_scaled_error: 0.0,
            symmetric_mean_absolute_percentage_error: 0.0,
            mean_directional_accuracy: 0.0,
            theil_u_statistic: 0.0,
            coefficient_of_determination: 0.0,
            mean_bias_error: 0.0,
            mean_percentage_error: 0.0,
            root_mean_squared_log_error: 0.0,
            normalized_root_mean_squared_error: 0.0,
            mean_gamma_deviance: 0.0,
            mean_poisson_deviance: 0.0,
            mean_tweedie_deviance: 0.0,
            d2_absolute_error_score: 0.0,
            d2_pinball_score: 0.0,
            d2_tweedie_score: 0.0,
            sample_size: 0,
            confidence_intervals: None,
        }
    }

    /// Get a summary of the metrics.
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str(&format!("Regression Metrics Summary (n={}):\n", self.sample_size));
        summary.push_str(&format!("  MSE: {:.6}\n", self.mse));
        summary.push_str(&format!("  RMSE: {:.6}\n", self.rmse));
        summary.push_str(&format!("  MAE: {:.6}\n", self.mae));
        summary.push_str(&format!("  R²: {:.6}\n", self.r_squared));
        summary.push_str(&format!("  MAPE: {:.2}%\n", self.mape));
        summary.push_str(&format!("  Explained Variance: {:.6}\n", self.explained_variance));
        summary.push_str(&format!("  Median AE: {:.6}\n", self.median_absolute_error));
        
        if let Some(ci) = &self.confidence_intervals {
            summary.push_str(&format!("\nConfidence Intervals ({:.0}%):\n", ci.confidence_level * 100.0));
            summary.push_str(&format!("  MSE: [{:.6}, {:.6}]\n", ci.mse_interval.0, ci.mse_interval.1));
            summary.push_str(&format!("  MAE: [{:.6}, {:.6}]\n", ci.mae_interval.0, ci.mae_interval.1));
            summary.push_str(&format!("  R²: [{:.6}, {:.6}]\n", ci.r2_interval.0, ci.r2_interval.1));
        }
        
        summary
    }

    /// Get the primary metric (MSE).
    pub fn primary_metric(&self) -> f64 {
        self.mse
    }

    /// Check if the model performance is good based on R².
    pub fn is_good_performance(&self, threshold: f64) -> bool {
        self.r_squared >= threshold
    }
}

impl Default for RegressionMetricsResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Confidence intervals for regression metrics.
#[derive(Debug, Clone)]
pub struct RegressionConfidenceIntervals {
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// MSE confidence interval
    pub mse_interval: (f64, f64),
    /// MAE confidence interval
    pub mae_interval: (f64, f64),
    /// R² confidence interval
    pub r2_interval: (f64, f64),
}

/// Builder for regression metrics configuration.
#[derive(Debug)]
pub struct RegressionMetricsConfigBuilder {
    config: RegressionMetricsConfig,
}

impl RegressionMetricsConfigBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: RegressionMetricsConfig::default(),
        }
    }

    /// Set whether to compute all metrics.
    pub fn compute_all(mut self, compute_all: bool) -> Self {
        self.config.compute_all = compute_all;
        self
    }

    /// Set whether to handle missing values.
    pub fn handle_missing(mut self, handle_missing: bool) -> Self {
        self.config.handle_missing = handle_missing;
        self
    }

    /// Set the epsilon value.
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.config.epsilon = epsilon;
        self
    }

    /// Set whether to use weights.
    pub fn use_weights(mut self, use_weights: bool) -> Self {
        self.config.use_weights = use_weights;
        self
    }

    /// Set whether to compute confidence intervals.
    pub fn compute_confidence_intervals(mut self, compute_ci: bool) -> Self {
        self.config.compute_confidence_intervals = compute_ci;
        self
    }

    /// Set the confidence level.
    pub fn confidence_level(mut self, level: f64) -> Self {
        self.config.confidence_level = level;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> RegressionMetricsConfig {
        self.config
    }
}

impl Default for RegressionMetricsConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for RegressionMetrics {
    fn default() -> Self {
        Self::new(RegressionMetricsConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_regression_metrics_config_default() {
        let config = RegressionMetricsConfig::default();
        assert!(config.compute_all);
        assert!(config.handle_missing);
        assert_eq!(config.epsilon, 1e-15);
        assert!(!config.use_weights);
        assert!(!config.compute_confidence_intervals);
        assert_eq!(config.confidence_level, 0.95);
    }

    #[test]
    fn test_regression_metrics_config_builder() {
        let config = RegressionMetricsConfigBuilder::new()
            .compute_all(false)
            .handle_missing(false)
            .epsilon(1e-10)
            .use_weights(true)
            .compute_confidence_intervals(true)
            .confidence_level(0.99)
            .build();

        assert!(!config.compute_all);
        assert!(!config.handle_missing);
        assert_eq!(config.epsilon, 1e-10);
        assert!(config.use_weights);
        assert!(config.compute_confidence_intervals);
        assert_eq!(config.confidence_level, 0.99);
    }

    #[test]
    fn test_mean_squared_error() {
        let mut metrics = RegressionMetrics::default();
        let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let targets = Array1::from_vec(vec![1.1, 2.1, 2.9]);
        
        let mse = metrics.mean_squared_error(&predictions.view(), &targets.view(), None).unwrap();
        let expected_mse = ((1.0 - 1.1).powi(2) + (2.0 - 2.1).powi(2) + (3.0 - 2.9).powi(2)) / 3.0;
        assert!((mse - expected_mse as f64).abs() < 1e-10);
    }

    #[test]
    fn test_mean_absolute_error() {
        let mut metrics = RegressionMetrics::default();
        let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let targets = Array1::from_vec(vec![1.1, 2.1, 2.9]);
        
        let mae = metrics.mean_absolute_error(&predictions.view(), &targets.view(), None).unwrap();
        let expected_mae = ((1.0 - 1.1).abs() + (2.0 - 2.1).abs() + (3.0 - 2.9).abs()) / 3.0;
        assert!((mae - expected_mae as f64).abs() < 1e-10);
    }

    #[test]
    fn test_r_squared() {
        let mut metrics = RegressionMetrics::default();
        let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let targets = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        
        let r2 = metrics.r_squared(&predictions.view(), &targets.view(), None).unwrap();
        assert!((r2 - 1.0).abs() < 1e-10); // Perfect prediction should give R² = 1
    }

    #[test]
    fn test_regression_metrics_result_summary() {
        let mut result = RegressionMetricsResult::new();
        result.mse = 0.25;
        result.rmse = 0.5;
        result.mae = 0.4;
        result.r_squared = 0.8;
        result.mape = 10.0;
        result.sample_size = 100;
        
        let summary = result.summary();
        assert!(summary.contains("MSE: 0.250000"));
        assert!(summary.contains("RMSE: 0.500000"));
        assert!(summary.contains("MAE: 0.400000"));
        assert!(summary.contains("R²: 0.800000"));
        assert!(summary.contains("MAPE: 10.00%"));
        assert!(summary.contains("n=100"));
    }

    #[test]
    fn test_weighted_metrics() {
        let mut metrics = RegressionMetrics::default();
        let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let targets = Array1::from_vec(vec![1.1, 2.1, 2.9]);
        let weights = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        
        let mse = metrics.mean_squared_error(&predictions.view(), &targets.view(), Some(&weights.view())).unwrap();
        let expected_mse = (
            1.0 * (1.0 - 1.1).powi(2) + 
            2.0 * (2.0 - 2.1).powi(2) + 
            3.0 * (3.0 - 2.9).powi(2)
        ) / (1.0 + 2.0 + 3.0);
        assert!((mse - expected_mse as f64).abs() < 1e-10);
    }

    #[test]
    fn test_median_absolute_error() {
        let mut metrics = RegressionMetrics::default();
        let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let targets = Array1::from_vec(vec![1.1, 2.1, 2.9, 3.8, 5.2]);
        
        let median_ae = metrics.median_absolute_error(&predictions.view(), &targets.view()).unwrap();
        let expected_median = 0.1; // Median of [0.1, 0.1, 0.1, 0.2, 0.2]
        assert!((median_ae - expected_median).abs() < 1e-10);
    }

    #[test]
    fn test_validation_errors() {
        let mut metrics = RegressionMetrics::default();
        let predictions = Array1::from_vec(vec![1.0, 2.0]);
        let targets = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        
        let result = metrics.mean_squared_error(&predictions.view(), &targets.view(), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_good_performance() {
        let mut result = RegressionMetricsResult::new();
        result.r_squared = 0.85;
        
        assert!(result.is_good_performance(0.8));
        assert!(!result.is_good_performance(0.9));
    }
}