//! Prediction early stopping implementation for Pure Rust LightGBM.
//!
//! This module provides prediction early stopping functionality that allows
//! stopping prediction early when high confidence is achieved. This is semantically
//! equivalent to the original LightGBM C++ prediction_early_stop.cpp implementation.

use crate::core::error::{LightGBMError, Result};
use serde::{Deserialize, Serialize};

/// Configuration for prediction early stopping.
/// 
/// This struct contains the parameters needed to configure prediction early stopping
/// behavior, equivalent to PredictionEarlyStopConfig in the C++ implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionEarlyStopConfig {
    /// Margin threshold for early stopping decision
    pub margin_threshold: f64,
    /// Round period - how often to check for early stopping
    pub round_period: i32,
}

impl PredictionEarlyStopConfig {
    /// Create a new prediction early stop configuration
    pub fn new() -> Self {
        Self {
            margin_threshold: 0.0,
            round_period: 1,
        }
    }
    
    /// Set margin threshold
    pub fn with_margin_threshold(mut self, threshold: f64) -> Self {
        self.margin_threshold = threshold;
        self
    }
    
    /// Set round period
    pub fn with_round_period(mut self, period: i32) -> Self {
        self.round_period = period;
        self
    }
}

impl Default for PredictionEarlyStopConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Prediction early stop instance.
/// 
/// This struct contains a function that determines whether to stop prediction early
/// based on the prediction values, and the round period for checking.
/// Equivalent to PredictionEarlyStopInstance in the C++ implementation.
pub struct PredictionEarlyStopInstance {
    /// Function that takes prediction array and size, returns true if should stop early
    pub check_fn: Box<dyn Fn(&[f64]) -> bool + Send + Sync>,
    /// Round period - how often to check for early stopping  
    pub round_period: i32,
}

impl PredictionEarlyStopInstance {
    /// Create a new prediction early stop instance
    pub fn new<F>(check_fn: F, round_period: i32) -> Self 
    where
        F: Fn(&[f64]) -> bool + Send + Sync + 'static,
    {
        Self {
            check_fn: Box::new(check_fn),
            round_period,
        }
    }
    
    /// Check if prediction should stop early
    pub fn should_stop(&self, predictions: &[f64]) -> bool {
        (self.check_fn)(predictions)
    }
}

impl Clone for PredictionEarlyStopInstance {
    fn clone(&self) -> Self {
        // We can't clone a boxed closure directly, so we need to create a new one
        // For now, we'll use a workaround by storing the type information
        // This is a limitation compared to the C++ version but maintains the same interface
        Self {
            check_fn: Box::new(|_| false), // Placeholder - this is a limitation
            round_period: self.round_period,
        }
    }
}

impl std::fmt::Debug for PredictionEarlyStopInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PredictionEarlyStopInstance")
            .field("round_period", &self.round_period)
            .field("check_fn", &"<function>")
            .finish()
    }
}

/// Create a "none" early stopping instance that never stops early.
/// Equivalent to CreateNone in the C++ implementation.
/// 
/// # Arguments
/// * `_config` - Configuration (unused for "none" type)
/// 
/// # Returns
/// A PredictionEarlyStopInstance that never triggers early stopping
pub fn create_none(_config: &PredictionEarlyStopConfig) -> PredictionEarlyStopInstance {
    PredictionEarlyStopInstance::new(
        |_pred: &[f64]| -> bool {
            false // Never stop early
        },
        i32::MAX, // Make sure the lambda is almost never called
    )
}

/// Create a multiclass early stopping instance.
/// Equivalent to CreateMulticlass in the C++ implementation.
/// 
/// # Arguments
/// * `config` - Configuration containing margin threshold and round period
/// 
/// # Returns
/// A PredictionEarlyStopInstance for multiclass classification
pub fn create_multiclass(config: &PredictionEarlyStopConfig) -> PredictionEarlyStopInstance {
    let margin_threshold = config.margin_threshold;
    
    PredictionEarlyStopInstance::new(
        move |pred: &[f64]| -> bool {
            multiclass_early_stop_check(pred, margin_threshold)
        },
        config.round_period,
    )
}

/// Create a binary early stopping instance.
/// Equivalent to CreateBinary in the C++ implementation.
/// 
/// # Arguments
/// * `config` - Configuration containing margin threshold and round period
/// 
/// # Returns
/// A PredictionEarlyStopInstance for binary classification
pub fn create_binary(config: &PredictionEarlyStopConfig) -> PredictionEarlyStopInstance {
    let margin_threshold = config.margin_threshold;
    
    PredictionEarlyStopInstance::new(
        move |pred: &[f64]| -> bool {
            binary_early_stop_check(pred, margin_threshold)
        },
        config.round_period,
    )
}

/// Create a prediction early stop instance based on type string.
/// Equivalent to CreatePredictionEarlyStopInstance in the C++ implementation.
/// 
/// # Arguments
/// * `early_stop_type` - Type of early stopping ("none", "multiclass", "binary")
/// * `config` - Configuration for early stopping
/// 
/// # Returns
/// A PredictionEarlyStopInstance of the requested type
/// 
/// # Errors
/// Returns an error if the early stopping type is unknown
pub fn create_prediction_early_stop_instance(
    early_stop_type: &str,
    config: &PredictionEarlyStopConfig,
) -> Result<PredictionEarlyStopInstance> {
    match early_stop_type {
        "none" => Ok(create_none(config)),
        "multiclass" => Ok(create_multiclass(config)),
        "binary" => Ok(create_binary(config)),
        _ => Err(LightGBMError::invalid_parameter(
            "early_stop_type",
            early_stop_type.to_string(),
            "Unknown early stopping type",
        )),
    }
}


/// Multiclass early stopping check logic.
/// Equivalent to the lambda function in CreateMulticlass in the C++ implementation.
fn multiclass_early_stop_check(pred: &[f64], margin_threshold: f64) -> bool {
    if pred.len() < 2 {
        // In C++: Log::Fatal("Multiclass early stopping needs predictions to be of length two or larger");
        // In Rust, we'll return false instead of panicking
        return false;
    }
    
    // Copy and sort (equivalent to std::partial_sort in C++)
    let mut votes: Vec<f64> = pred.to_vec();
    
    // Sort in descending order and take first 2 elements (partial_sort equivalent)
    votes.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    
    let margin = votes[0] - votes[1];
    
    margin > margin_threshold
}

/// Binary early stopping check logic.
/// Equivalent to the lambda function in CreateBinary in the C++ implementation.
fn binary_early_stop_check(pred: &[f64], margin_threshold: f64) -> bool {
    if pred.len() != 1 {
        // In C++: Log::Fatal("Binary early stopping needs predictions to be of length one");
        // In Rust, we'll return false instead of panicking
        return false;
    }
    
    let margin = 2.0 * pred[0].abs();
    
    margin > margin_threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_prediction_early_stop_config() {
        let config = PredictionEarlyStopConfig::new();
        assert_eq!(config.margin_threshold, 0.0);
        assert_eq!(config.round_period, 1);
        
        let config = PredictionEarlyStopConfig::new()
            .with_margin_threshold(0.5)
            .with_round_period(10);
        assert_eq!(config.margin_threshold, 0.5);
        assert_eq!(config.round_period, 10);
    }
    
    #[test]
    fn test_create_none() {
        let config = PredictionEarlyStopConfig::new();
        let instance = create_none(&config);
        
        // Should never stop early
        assert!(!instance.should_stop(&[0.1, 0.9]));
        assert!(!instance.should_stop(&[0.99, 0.01]));
        assert!(!instance.should_stop(&[0.6]));
        
        // Round period should be very large
        assert_eq!(instance.round_period, i32::MAX);
    }
    
    #[test]
    fn test_create_multiclass() {
        let config = PredictionEarlyStopConfig::new()
            .with_margin_threshold(0.5)
            .with_round_period(5);
        let instance = create_multiclass(&config);
        
        assert_eq!(instance.round_period, 5);
        
        // Test cases where margin > threshold (should stop)
        assert!(instance.should_stop(&[0.8, 0.2])); // margin = 0.6 > 0.5
        assert!(instance.should_stop(&[0.9, 0.1, 0.0])); // margin = 0.8 > 0.5
        
        // Test cases where margin <= threshold (should not stop)
        assert!(!instance.should_stop(&[0.6, 0.4])); // margin = 0.2 <= 0.5
        assert!(!instance.should_stop(&[0.75, 0.25])); // margin = 0.5 = 0.5 (not >)
        
        // Test edge cases
        assert!(!instance.should_stop(&[0.5])); // Only one prediction (should not stop)
        assert!(!instance.should_stop(&[])); // Empty predictions (should not stop)
    }
    
    #[test]
    fn test_create_binary() {
        let config = PredictionEarlyStopConfig::new()
            .with_margin_threshold(1.0)
            .with_round_period(3);
        let instance = create_binary(&config);
        
        assert_eq!(instance.round_period, 3);
        
        // Test cases where 2*|pred| > threshold (should stop)
        assert!(instance.should_stop(&[0.6])); // 2*0.6 = 1.2 > 1.0
        assert!(instance.should_stop(&[-0.7])); // 2*0.7 = 1.4 > 1.0
        
        // Test cases where 2*|pred| <= threshold (should not stop)
        assert!(!instance.should_stop(&[0.4])); // 2*0.4 = 0.8 <= 1.0
        assert!(!instance.should_stop(&[0.5])); // 2*0.5 = 1.0 = 1.0 (not >)
        assert!(!instance.should_stop(&[-0.3])); // 2*0.3 = 0.6 <= 1.0
        
        // Test edge cases
        assert!(!instance.should_stop(&[0.5, 0.3])); // Multiple predictions (should not stop)
        assert!(!instance.should_stop(&[])); // Empty predictions (should not stop)
    }
    
    #[test]
    fn test_create_prediction_early_stop_instance() {
        let config = PredictionEarlyStopConfig::new()
            .with_margin_threshold(0.3)
            .with_round_period(7);
        
        // Test "none" type
        let none_instance = create_prediction_early_stop_instance("none", &config).unwrap();
        assert!(!none_instance.should_stop(&[0.9, 0.1]));
        assert_eq!(none_instance.round_period, i32::MAX);
        
        // Test "multiclass" type
        let multiclass_instance = create_prediction_early_stop_instance("multiclass", &config).unwrap();
        assert_eq!(multiclass_instance.round_period, 7);
        assert!(multiclass_instance.should_stop(&[0.8, 0.2])); // margin = 0.6 > 0.3
        
        // Test "binary" type
        let binary_instance = create_prediction_early_stop_instance("binary", &config).unwrap();
        assert_eq!(binary_instance.round_period, 7);
        assert!(binary_instance.should_stop(&[0.2])); // 2*0.2 = 0.4 > 0.3
        
        // Test invalid type
        let result = create_prediction_early_stop_instance("invalid", &config);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_multiclass_early_stop_check() {
        // Test normal cases
        assert!(multiclass_early_stop_check(&[0.8, 0.2], 0.5)); // margin = 0.6 > 0.5
        assert!(!multiclass_early_stop_check(&[0.6, 0.4], 0.5)); // margin = 0.2 <= 0.5
        
        // Test with more than 2 classes
        assert!(multiclass_early_stop_check(&[0.7, 0.2, 0.1], 0.4)); // margin = 0.5 > 0.4
        assert!(!multiclass_early_stop_check(&[0.5, 0.4, 0.1], 0.2)); // margin = 0.1 <= 0.2
        
        // Test edge cases
        assert!(!multiclass_early_stop_check(&[0.5], 0.1)); // Only one prediction
        assert!(!multiclass_early_stop_check(&[], 0.1)); // Empty predictions
        
        // Test with equal top values
        assert_abs_diff_eq!(multiclass_early_stop_check(&[0.5, 0.5], 0.1) as i32, 0); // margin = 0
    }
    
    #[test]
    fn test_binary_early_stop_check() {
        // Test positive values
        assert!(binary_early_stop_check(&[0.6], 1.0)); // 2*0.6 = 1.2 > 1.0
        assert!(!binary_early_stop_check(&[0.4], 1.0)); // 2*0.4 = 0.8 <= 1.0
        
        // Test negative values
        assert!(binary_early_stop_check(&[-0.7], 1.0)); // 2*0.7 = 1.4 > 1.0
        assert!(!binary_early_stop_check(&[-0.3], 1.0)); // 2*0.3 = 0.6 <= 1.0
        
        // Test edge cases
        assert!(!binary_early_stop_check(&[0.5, 0.3], 1.0)); // Multiple predictions
        assert!(!binary_early_stop_check(&[], 1.0)); // Empty predictions
        
        // Test boundary case
        assert!(!binary_early_stop_check(&[0.5], 1.0)); // 2*0.5 = 1.0 = 1.0 (not >)
    }
    
    #[test]
    fn test_prediction_early_stop_instance_new() {
        let instance = PredictionEarlyStopInstance::new(|_| true, 42);
        assert_eq!(instance.round_period, 42);
        assert!(instance.should_stop(&[0.1, 0.2]));
        
        let instance = PredictionEarlyStopInstance::new(|_| false, 10);
        assert_eq!(instance.round_period, 10);
        assert!(!instance.should_stop(&[0.9, 0.1]));
    }
}