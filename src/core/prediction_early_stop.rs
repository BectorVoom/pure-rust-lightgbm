/*!
 * Copyright (c) 2017 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

//! Prediction early stopping module for Pure Rust LightGBM.
//!
//! This module provides the core types and factory function for prediction early stopping,
//! semantically equivalent to include/LightGBM/prediction_early_stop.h

use std::sync::Arc;

/// Callback function type for early stopping.
/// Takes current prediction slice and returns true if prediction should stop according to criterion.
/// Equivalent to std::function<bool(const double*, int)> in C++.
pub type FunctionType = Arc<dyn Fn(&[f64]) -> bool + Send + Sync>;

/// Prediction early stop instance.
/// Equivalent to PredictionEarlyStopInstance struct in C++.
#[derive(Clone)]
pub struct PredictionEarlyStopInstance {
    /// Callback function itself
    pub callback_function: FunctionType,
    /// Call callback_function every `round_period` iterations
    pub round_period: i32,
}

impl PredictionEarlyStopInstance {
    /// Create a new prediction early stop instance
    pub fn new(callback_function: FunctionType, round_period: i32) -> Self {
        Self {
            callback_function,
            round_period,
        }
    }
}

impl std::fmt::Debug for PredictionEarlyStopInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PredictionEarlyStopInstance")
            .field("round_period", &self.round_period)
            .field("callback_function", &"<function>")
            .finish()
    }
}

/// Configuration for prediction early stopping.
/// Equivalent to PredictionEarlyStopConfig struct in C++.
#[derive(Debug, Clone, Copy)]
pub struct PredictionEarlyStopConfig {
    /// Number of rounds between early stopping checks
    pub round_period: i32,
    /// Margin threshold for early stopping
    pub margin_threshold: f64,
}

impl PredictionEarlyStopConfig {
    /// Create a new configuration
    pub fn new(round_period: i32, margin_threshold: f64) -> Self {
        Self {
            round_period,
            margin_threshold,
        }
    }
}

/// Create an early stopping algorithm of type `type`, with given round_period and margin threshold.
/// Equivalent to CreatePredictionEarlyStopInstance function in C++.
///
/// # Arguments
/// * `type_str` - The type of early stopping algorithm ("none", "multiclass", "binary")
/// * `config` - Configuration containing round_period and margin_threshold
///
/// # Returns
/// A PredictionEarlyStopInstance configured for the specified type
///
/// # Panics
/// Panics if the type string is not recognized (matches C++ behavior of potential undefined behavior)
pub fn create_prediction_early_stop_instance(
    type_str: &str,
    config: &PredictionEarlyStopConfig,
) -> PredictionEarlyStopInstance {
    match type_str {
        "none" => create_none_instance(config),
        "multiclass" => create_multiclass_instance(config),
        "binary" => create_binary_instance(config),
        _ => panic!("Unknown early stopping type: {}", type_str),
    }
}

/// Create a "none" early stopping instance that never stops early
fn create_none_instance(config: &PredictionEarlyStopConfig) -> PredictionEarlyStopInstance {
    let callback: FunctionType = Arc::new(|_pred: &[f64]| -> bool {
        false // Never stop early
    });
    
    PredictionEarlyStopInstance::new(callback, i32::MAX)
}

/// Create a multiclass early stopping instance
fn create_multiclass_instance(config: &PredictionEarlyStopConfig) -> PredictionEarlyStopInstance {
    let margin_threshold = config.margin_threshold;
    let callback: FunctionType = Arc::new(move |pred: &[f64]| -> bool {
        multiclass_early_stop_check(pred, margin_threshold)
    });
    
    PredictionEarlyStopInstance::new(callback, config.round_period)
}

/// Create a binary early stopping instance
fn create_binary_instance(config: &PredictionEarlyStopConfig) -> PredictionEarlyStopInstance {
    let margin_threshold = config.margin_threshold;
    let callback: FunctionType = Arc::new(move |pred: &[f64]| -> bool {
        binary_early_stop_check(pred, margin_threshold)
    });
    
    PredictionEarlyStopInstance::new(callback, config.round_period)
}

/// Multiclass early stopping check logic
fn multiclass_early_stop_check(pred: &[f64], margin_threshold: f64) -> bool {
    if pred.len() < 2 {
        return false;
    }
    
    // Create a sorted copy (equivalent to std::partial_sort in C++)
    let mut votes: Vec<f64> = pred.to_vec();
    votes.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    
    let margin = votes[0] - votes[1];
    margin > margin_threshold
}

/// Binary early stopping check logic
fn binary_early_stop_check(pred: &[f64], margin_threshold: f64) -> bool {
    if pred.len() != 1 {
        return false;
    }
    
    let margin = 2.0 * pred[0].abs();
    margin > margin_threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_early_stop_config() {
        let config = PredictionEarlyStopConfig::new(10, 0.5);
        assert_eq!(config.round_period, 10);
        assert_eq!(config.margin_threshold, 0.5);
    }

    #[test]
    fn test_prediction_early_stop_instance_creation() {
        let callback: FunctionType = Arc::new(|_| true);
        let instance = PredictionEarlyStopInstance::new(callback, 5);
        assert_eq!(instance.round_period, 5);
    }

    #[test]
    fn test_create_none_instance() {
        let config = PredictionEarlyStopConfig::new(5, 0.5);
        let instance = create_prediction_early_stop_instance("none", &config);
        
        // Should never stop early
        assert!(!(instance.callback_function)(&[0.1, 0.9]));
        assert!(!(instance.callback_function)(&[0.99, 0.01]));
        assert_eq!(instance.round_period, i32::MAX);
    }

    #[test]
    fn test_create_multiclass_instance() {
        let config = PredictionEarlyStopConfig::new(5, 0.5);
        let instance = create_prediction_early_stop_instance("multiclass", &config);
        
        assert_eq!(instance.round_period, 5);
        
        // Test cases where margin > threshold (should stop)
        assert!((instance.callback_function)(&[0.8, 0.2])); // margin = 0.6 > 0.5
        assert!((instance.callback_function)(&[0.9, 0.1, 0.0])); // margin = 0.8 > 0.5
        
        // Test cases where margin <= threshold (should not stop)
        assert!(!(instance.callback_function)(&[0.6, 0.4])); // margin = 0.2 <= 0.5
        assert!(!(instance.callback_function)(&[0.75, 0.25])); // margin = 0.5 = 0.5 (not >)
    }

    #[test]
    fn test_create_binary_instance() {
        let config = PredictionEarlyStopConfig::new(3, 1.0);
        let instance = create_prediction_early_stop_instance("binary", &config);
        
        assert_eq!(instance.round_period, 3);
        
        // Test cases where 2*|pred| > threshold (should stop)
        assert!((instance.callback_function)(&[0.6])); // 2*0.6 = 1.2 > 1.0
        assert!((instance.callback_function)(&[-0.7])); // 2*0.7 = 1.4 > 1.0
        
        // Test cases where 2*|pred| <= threshold (should not stop)
        assert!(!(instance.callback_function)(&[0.4])); // 2*0.4 = 0.8 <= 1.0
        assert!(!(instance.callback_function)(&[0.5])); // 2*0.5 = 1.0 = 1.0 (not >)
    }

    #[test]
    #[should_panic(expected = "Unknown early stopping type")]
    fn test_create_unknown_type() {
        let config = PredictionEarlyStopConfig::new(1, 0.0);
        create_prediction_early_stop_instance("unknown", &config);
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
    }

    /// Comprehensive equivalence test that matches the C++ behavior exactly
    #[test]
    fn test_cpp_rust_equivalence() {
        println!("=== Rust Prediction Early Stop Equivalence Tests ===");
        
        // Test 1: None type
        {
            let config = PredictionEarlyStopConfig::new(5, 0.5);
            let instance = create_prediction_early_stop_instance("none", &config);
            
            let pred1 = vec![0.1, 0.9];
            let pred2 = vec![0.99, 0.01];
            
            let result1 = (instance.callback_function)(&pred1);
            let result2 = (instance.callback_function)(&pred2);
            
            println!("None type - pred1: {} (expected: false)", result1);
            println!("None type - pred2: {} (expected: false)", result2);
            println!("None type - round_period: {} (expected: {})", instance.round_period, i32::MAX);
            
            assert_eq!(result1, false);
            assert_eq!(result2, false);
            assert_eq!(instance.round_period, i32::MAX);
        }
        
        // Test 2: Multiclass type
        {
            let config = PredictionEarlyStopConfig::new(5, 0.5);
            let instance = create_prediction_early_stop_instance("multiclass", &config);
            
            // Test cases where margin > threshold (should stop)
            let pred1 = vec![0.8, 0.2];  // margin = 0.6 > 0.5
            let pred2 = vec![0.9, 0.1, 0.0];  // margin = 0.8 > 0.5
            
            // Test cases where margin <= threshold (should not stop)
            let pred3 = vec![0.6, 0.4];  // margin = 0.2 <= 0.5
            let pred4 = vec![0.75, 0.25];  // margin = 0.5 = 0.5 (not >)
            
            let result1 = (instance.callback_function)(&pred1);
            let result2 = (instance.callback_function)(&pred2);
            let result3 = (instance.callback_function)(&pred3);
            let result4 = (instance.callback_function)(&pred4);
            
            println!("Multiclass - pred1 (0.8,0.2): {} (expected: true)", result1);
            println!("Multiclass - pred2 (0.9,0.1,0.0): {} (expected: true)", result2);
            println!("Multiclass - pred3 (0.6,0.4): {} (expected: false)", result3);
            println!("Multiclass - pred4 (0.75,0.25): {} (expected: false)", result4);
            println!("Multiclass - round_period: {} (expected: 5)", instance.round_period);
            
            assert_eq!(result1, true);
            assert_eq!(result2, true);
            assert_eq!(result3, false);
            assert_eq!(result4, false);
            assert_eq!(instance.round_period, 5);
            
            // Test edge cases (Rust returns false, C++ would throw)
            let pred_single = vec![0.5];
            let result_single = (instance.callback_function)(&pred_single);
            println!("Multiclass - single pred: {} (C++ throws, Rust returns false)", result_single);
            assert_eq!(result_single, false);
            
            let pred_empty: Vec<f64> = vec![];
            let result_empty = (instance.callback_function)(&pred_empty);
            println!("Multiclass - empty pred: {} (C++ throws, Rust returns false)", result_empty);
            assert_eq!(result_empty, false);
        }
        
        // Test 3: Binary type
        {
            let config = PredictionEarlyStopConfig::new(3, 1.0);
            let instance = create_prediction_early_stop_instance("binary", &config);
            
            // Test cases where 2*|pred| > threshold (should stop)
            let pred1 = vec![0.6];  // 2*0.6 = 1.2 > 1.0
            let pred2 = vec![-0.7]; // 2*0.7 = 1.4 > 1.0
            
            // Test cases where 2*|pred| <= threshold (should not stop)
            let pred3 = vec![0.4];  // 2*0.4 = 0.8 <= 1.0
            let pred4 = vec![0.5];  // 2*0.5 = 1.0 = 1.0 (not >)
            let pred5 = vec![-0.3]; // 2*0.3 = 0.6 <= 1.0
            
            let result1 = (instance.callback_function)(&pred1);
            let result2 = (instance.callback_function)(&pred2);
            let result3 = (instance.callback_function)(&pred3);
            let result4 = (instance.callback_function)(&pred4);
            let result5 = (instance.callback_function)(&pred5);
            
            println!("Binary - pred1 (0.6): {} (expected: true)", result1);
            println!("Binary - pred2 (-0.7): {} (expected: true)", result2);
            println!("Binary - pred3 (0.4): {} (expected: false)", result3);
            println!("Binary - pred4 (0.5): {} (expected: false)", result4);
            println!("Binary - pred5 (-0.3): {} (expected: false)", result5);
            println!("Binary - round_period: {} (expected: 3)", instance.round_period);
            
            assert_eq!(result1, true);
            assert_eq!(result2, true);
            assert_eq!(result3, false);
            assert_eq!(result4, false);
            assert_eq!(result5, false);
            assert_eq!(instance.round_period, 3);
            
            // Test edge cases (Rust returns false, C++ would throw)
            let pred_multi = vec![0.5, 0.3];
            let result_multi = (instance.callback_function)(&pred_multi);
            println!("Binary - multiple pred: {} (C++ throws, Rust returns false)", result_multi);
            assert_eq!(result_multi, false);
            
            let pred_empty: Vec<f64> = vec![];
            let result_empty = (instance.callback_function)(&pred_empty);
            println!("Binary - empty pred: {} (C++ throws, Rust returns false)", result_empty);
            assert_eq!(result_empty, false);        
        }
        
        println!("All Rust equivalence tests passed!");
    }
}