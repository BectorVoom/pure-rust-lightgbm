//! Objective function configuration for Pure Rust LightGBM.
//!
//! This module provides configuration structures and utilities for different
//! objective functions supported by LightGBM, including regression, binary
//! classification, multiclass classification, and ranking objectives.

use crate::core::types::*;
use crate::core::error::{Result, LightGBMError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Objective function configuration structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveConfig {
    /// Type of objective function
    pub objective_type: ObjectiveType,
    /// Number of classes for multiclass classification
    pub num_class: usize,
    /// Whether to use unbalanced dataset handling
    pub is_unbalance: bool,
    /// Positive class weight scaling factor
    pub scale_pos_weight: f64,
    /// Sigmoid parameter for binary classification
    pub sigmoid: f64,
    /// Alpha parameter for Tweedie regression
    pub tweedie_variance_power: f64,
    /// Alpha parameter for Focal loss
    pub focal_loss_alpha: f64,
    /// Gamma parameter for Focal loss
    pub focal_loss_gamma: f64,
    /// Poisson maximum delta step
    pub poisson_max_delta_step: f64,
    /// Fair C parameter for Fair regression
    pub fair_c: f64,
    /// Huber delta parameter for Huber regression
    pub huber_delta: f64,
    /// Quantile alpha for quantile regression
    pub quantile_alpha: f64,
    /// Custom objective function parameters
    pub custom_params: HashMap<String, f64>,
}

impl Default for ObjectiveConfig {
    fn default() -> Self {
        ObjectiveConfig {
            objective_type: ObjectiveType::Regression,
            num_class: 1,
            is_unbalance: false,
            scale_pos_weight: 1.0,
            sigmoid: 1.0,
            tweedie_variance_power: 1.5,
            focal_loss_alpha: 0.25,
            focal_loss_gamma: 2.0,
            poisson_max_delta_step: 0.7,
            fair_c: 1.0,
            huber_delta: 1.0,
            quantile_alpha: 0.5,
            custom_params: HashMap::new(),
        }
    }
}

impl ObjectiveConfig {
    /// Create a new objective configuration
    pub fn new(objective_type: ObjectiveType) -> Self {
        let mut config = ObjectiveConfig::default();
        config.objective_type = objective_type;
        config
    }

    /// Create configuration for regression
    pub fn regression() -> Self {
        ObjectiveConfig::new(ObjectiveType::Regression)
    }

    /// Create configuration for binary classification
    pub fn binary() -> Self {
        ObjectiveConfig::new(ObjectiveType::Binary)
    }

    /// Create configuration for multiclass classification
    pub fn multiclass(num_class: usize) -> Self {
        let mut config = ObjectiveConfig::new(ObjectiveType::Multiclass);
        config.num_class = num_class;
        config
    }

    /// Create configuration for ranking
    pub fn ranking() -> Self {
        ObjectiveConfig::new(ObjectiveType::Ranking)
    }

    /// Create configuration for Poisson regression
    pub fn poisson() -> Self {
        ObjectiveConfig::new(ObjectiveType::Poisson)
    }

    /// Create configuration for Gamma regression
    pub fn gamma() -> Self {
        ObjectiveConfig::new(ObjectiveType::Gamma)
    }

    /// Create configuration for Tweedie regression
    pub fn tweedie(variance_power: f64) -> Self {
        let mut config = ObjectiveConfig::new(ObjectiveType::Tweedie);
        config.tweedie_variance_power = variance_power;
        config
    }

    /// Set unbalanced dataset handling
    pub fn with_unbalance(mut self, is_unbalance: bool) -> Self {
        self.is_unbalance = is_unbalance;
        self
    }

    /// Set positive class weight scaling
    pub fn with_scale_pos_weight(mut self, scale_pos_weight: f64) -> Self {
        self.scale_pos_weight = scale_pos_weight;
        self
    }

    /// Set sigmoid parameter
    pub fn with_sigmoid(mut self, sigmoid: f64) -> Self {
        self.sigmoid = sigmoid;
        self
    }

    /// Set focal loss parameters
    pub fn with_focal_loss(mut self, alpha: f64, gamma: f64) -> Self {
        self.focal_loss_alpha = alpha;
        self.focal_loss_gamma = gamma;
        self
    }

    /// Set Poisson max delta step
    pub fn with_poisson_max_delta_step(mut self, max_delta_step: f64) -> Self {
        self.poisson_max_delta_step = max_delta_step;
        self
    }

    /// Set Fair C parameter
    pub fn with_fair_c(mut self, fair_c: f64) -> Self {
        self.fair_c = fair_c;
        self
    }

    /// Set Huber delta parameter
    pub fn with_huber_delta(mut self, huber_delta: f64) -> Self {
        self.huber_delta = huber_delta;
        self
    }

    /// Set quantile alpha parameter
    pub fn with_quantile_alpha(mut self, quantile_alpha: f64) -> Self {
        self.quantile_alpha = quantile_alpha;
        self
    }

    /// Add custom parameter
    pub fn with_custom_param(mut self, key: String, value: f64) -> Self {
        self.custom_params.insert(key, value);
        self
    }

    /// Validate the objective configuration
    pub fn validate(&self) -> Result<()> {
        // Validate number of classes for multiclass
        if self.objective_type == ObjectiveType::Multiclass && self.num_class < 2 {
            return Err(LightGBMError::invalid_parameter(
                "num_class",
                self.num_class.to_string(),
                "must be at least 2 for multiclass objective",
            ));
        }

        // Validate scale_pos_weight
        if self.scale_pos_weight <= 0.0 {
            return Err(LightGBMError::invalid_parameter(
                "scale_pos_weight",
                self.scale_pos_weight.to_string(),
                "must be positive",
            ));
        }

        // Validate sigmoid
        if self.sigmoid <= 0.0 {
            return Err(LightGBMError::invalid_parameter(
                "sigmoid",
                self.sigmoid.to_string(),
                "must be positive",
            ));
        }

        // Validate Tweedie variance power
        if self.objective_type == ObjectiveType::Tweedie {
            if self.tweedie_variance_power < 1.0 || self.tweedie_variance_power > 2.0 {
                return Err(LightGBMError::invalid_parameter(
                    "tweedie_variance_power",
                    self.tweedie_variance_power.to_string(),
                    "must be in range [1.0, 2.0]",
                ));
            }
        }

        // Validate focal loss parameters
        if self.focal_loss_alpha < 0.0 || self.focal_loss_alpha > 1.0 {
            return Err(LightGBMError::invalid_parameter(
                "focal_loss_alpha",
                self.focal_loss_alpha.to_string(),
                "must be in range [0.0, 1.0]",
            ));
        }

        if self.focal_loss_gamma < 0.0 {
            return Err(LightGBMError::invalid_parameter(
                "focal_loss_gamma",
                self.focal_loss_gamma.to_string(),
                "must be non-negative",
            ));
        }

        // Validate Poisson max delta step
        if self.objective_type == ObjectiveType::Poisson {
            if self.poisson_max_delta_step <= 0.0 {
                return Err(LightGBMError::invalid_parameter(
                    "poisson_max_delta_step",
                    self.poisson_max_delta_step.to_string(),
                    "must be positive",
                ));
            }
        }

        // Validate Fair C parameter
        if self.fair_c <= 0.0 {
            return Err(LightGBMError::invalid_parameter(
                "fair_c",
                self.fair_c.to_string(),
                "must be positive",
            ));
        }

        // Validate Huber delta parameter
        if self.huber_delta <= 0.0 {
            return Err(LightGBMError::invalid_parameter(
                "huber_delta",
                self.huber_delta.to_string(),
                "must be positive",
            ));
        }

        // Validate quantile alpha
        if self.quantile_alpha < 0.0 || self.quantile_alpha > 1.0 {
            return Err(LightGBMError::invalid_parameter(
                "quantile_alpha",
                self.quantile_alpha.to_string(),
                "must be in range [0.0, 1.0]",
            ));
        }

        Ok(())
    }

    /// Get the number of model outputs for this objective
    pub fn num_model_outputs(&self) -> usize {
        match self.objective_type {
            ObjectiveType::Regression |
            ObjectiveType::Binary |
            ObjectiveType::Ranking |
            ObjectiveType::Poisson |
            ObjectiveType::Gamma |
            ObjectiveType::Tweedie => 1,
            ObjectiveType::Multiclass => self.num_class,
        }
    }

    /// Check if this objective requires class weights
    pub fn requires_class_weights(&self) -> bool {
        matches!(self.objective_type, ObjectiveType::Binary | ObjectiveType::Multiclass)
    }

    /// Check if this objective supports early stopping
    pub fn supports_early_stopping(&self) -> bool {
        true  // All objectives support early stopping
    }

    /// Get objective-specific parameter map
    pub fn as_parameter_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        
        map.insert("objective".to_string(), self.objective_type.to_string());
        map.insert("num_class".to_string(), self.num_class.to_string());
        map.insert("is_unbalance".to_string(), self.is_unbalance.to_string());
        map.insert("scale_pos_weight".to_string(), self.scale_pos_weight.to_string());
        map.insert("sigmoid".to_string(), self.sigmoid.to_string());
        
        if self.objective_type == ObjectiveType::Tweedie {
            map.insert("tweedie_variance_power".to_string(), self.tweedie_variance_power.to_string());
        }
        
        if self.objective_type == ObjectiveType::Poisson {
            map.insert("poisson_max_delta_step".to_string(), self.poisson_max_delta_step.to_string());
        }
        
        // Add custom parameters
        for (key, value) in &self.custom_params {
            map.insert(key.clone(), value.to_string());
        }
        
        map
    }
}

/// Objective function trait for pluggable objective functions
pub trait ObjectiveFunction: Send + Sync {
    /// Get the objective function name
    fn name(&self) -> &'static str;
    
    /// Get the number of model outputs
    fn num_model_outputs(&self) -> usize;
    
    /// Check if this objective supports early stopping
    fn supports_early_stopping(&self) -> bool {
        true
    }
    
    /// Check if this objective requires class weights
    fn requires_class_weights(&self) -> bool {
        false
    }
    
    /// Validate the objective configuration
    fn validate_config(&self, config: &ObjectiveConfig) -> Result<()>;
    
    /// Get default configuration for this objective
    fn default_config(&self) -> ObjectiveConfig;
    
    /// Transform raw prediction scores to final output (e.g., sigmoid, softmax)
    fn transform_predictions(&self, raw_scores: &[f64]) -> Result<Vec<f64>>;
    
    /// Calculate gradients and hessians for given predictions and labels
    fn calculate_gradients_hessians(
        &self,
        predictions: &[f64],
        labels: &[f64],
        gradients: &mut [f64],
        hessians: &mut [f64],
    ) -> Result<()>;
}

/// Regression objective function
pub struct RegressionObjective;

impl ObjectiveFunction for RegressionObjective {
    fn name(&self) -> &'static str {
        "regression"
    }
    
    fn num_model_outputs(&self) -> usize {
        1
    }
    
    fn validate_config(&self, config: &ObjectiveConfig) -> Result<()> {
        if config.objective_type != ObjectiveType::Regression {
            return Err(LightGBMError::config("Invalid objective type for regression"));
        }
        config.validate()
    }
    
    fn default_config(&self) -> ObjectiveConfig {
        ObjectiveConfig::regression()
    }
    
    fn transform_predictions(&self, raw_scores: &[f64]) -> Result<Vec<f64>> {
        // No transformation for regression
        Ok(raw_scores.to_vec())
    }
    
    fn calculate_gradients_hessians(
        &self,
        predictions: &[f64],
        labels: &[f64],
        gradients: &mut [f64],
        hessians: &mut [f64],
    ) -> Result<()> {
        if predictions.len() != labels.len() || gradients.len() != predictions.len() || hessians.len() != predictions.len() {
            return Err(LightGBMError::dimension_mismatch(
                "predictions, labels, gradients, and hessians must have the same length",
                format!("predictions: {}, labels: {}, gradients: {}, hessians: {}", 
                       predictions.len(), labels.len(), gradients.len(), hessians.len()),
            ));
        }
        
        // L2 loss: gradient = prediction - label, hessian = 1.0
        for i in 0..predictions.len() {
            gradients[i] = predictions[i] - labels[i];
            hessians[i] = 1.0;
        }
        
        Ok(())
    }
}

/// Binary classification objective function
pub struct BinaryObjective {
    config: ObjectiveConfig,
}

impl BinaryObjective {
    pub fn new(config: ObjectiveConfig) -> Result<Self> {
        config.validate()?;
        Ok(BinaryObjective { config })
    }
}

impl ObjectiveFunction for BinaryObjective {
    fn name(&self) -> &'static str {
        "binary"
    }
    
    fn num_model_outputs(&self) -> usize {
        1
    }
    
    fn requires_class_weights(&self) -> bool {
        true
    }
    
    fn validate_config(&self, config: &ObjectiveConfig) -> Result<()> {
        if config.objective_type != ObjectiveType::Binary {
            return Err(LightGBMError::config("Invalid objective type for binary classification"));
        }
        config.validate()
    }
    
    fn default_config(&self) -> ObjectiveConfig {
        ObjectiveConfig::binary()
    }
    
    fn transform_predictions(&self, raw_scores: &[f64]) -> Result<Vec<f64>> {
        // Sigmoid transformation
        let predictions = raw_scores.iter()
            .map(|&score| 1.0 / (1.0 + (-score * self.config.sigmoid).exp()))
            .collect();
        Ok(predictions)
    }
    
    fn calculate_gradients_hessians(
        &self,
        predictions: &[f64],
        labels: &[f64],
        gradients: &mut [f64],
        hessians: &mut [f64],
    ) -> Result<()> {
        if predictions.len() != labels.len() || gradients.len() != predictions.len() || hessians.len() != predictions.len() {
            return Err(LightGBMError::dimension_mismatch(
                "predictions, labels, gradients, and hessians must have the same length",
                format!("predictions: {}, labels: {}, gradients: {}, hessians: {}", 
                       predictions.len(), labels.len(), gradients.len(), hessians.len()),
            ));
        }
        
        // Binary logistic loss
        for i in 0..predictions.len() {
            let prob = 1.0 / (1.0 + (-predictions[i] * self.config.sigmoid).exp());
            gradients[i] = prob - labels[i];
            hessians[i] = prob * (1.0 - prob) * self.config.sigmoid * self.config.sigmoid;
            
            // Apply class weight scaling
            if self.config.is_unbalance {
                let weight = if labels[i] > 0.5 { self.config.scale_pos_weight } else { 1.0 };
                gradients[i] *= weight;
                hessians[i] *= weight;
            }
        }
        
        Ok(())
    }
}

/// Multiclass classification objective function
pub struct MulticlassObjective {
    config: ObjectiveConfig,
}

impl MulticlassObjective {
    pub fn new(config: ObjectiveConfig) -> Result<Self> {
        config.validate()?;
        Ok(MulticlassObjective { config })
    }
}

impl ObjectiveFunction for MulticlassObjective {
    fn name(&self) -> &'static str {
        "multiclass"
    }
    
    fn num_model_outputs(&self) -> usize {
        self.config.num_class
    }
    
    fn requires_class_weights(&self) -> bool {
        true
    }
    
    fn validate_config(&self, config: &ObjectiveConfig) -> Result<()> {
        if config.objective_type != ObjectiveType::Multiclass {
            return Err(LightGBMError::config("Invalid objective type for multiclass classification"));
        }
        config.validate()
    }
    
    fn default_config(&self) -> ObjectiveConfig {
        ObjectiveConfig::multiclass(self.config.num_class)
    }
    
    fn transform_predictions(&self, raw_scores: &[f64]) -> Result<Vec<f64>> {
        let num_samples = raw_scores.len() / self.config.num_class;
        let mut predictions = Vec::with_capacity(raw_scores.len());
        
        // Apply softmax transformation
        for i in 0..num_samples {
            let start_idx = i * self.config.num_class;
            let end_idx = start_idx + self.config.num_class;
            let scores = &raw_scores[start_idx..end_idx];
            
            // Find max score for numerical stability
            let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            
            // Calculate softmax
            let exp_scores: Vec<f64> = scores.iter()
                .map(|&score| (score - max_score).exp())
                .collect();
            let sum_exp: f64 = exp_scores.iter().sum();
            
            for exp_score in exp_scores {
                predictions.push(exp_score / sum_exp);
            }
        }
        
        Ok(predictions)
    }
    
    fn calculate_gradients_hessians(
        &self,
        predictions: &[f64],
        labels: &[f64],
        gradients: &mut [f64],
        hessians: &mut [f64],
    ) -> Result<()> {
        let num_samples = predictions.len() / self.config.num_class;
        
        if labels.len() != num_samples || gradients.len() != predictions.len() || hessians.len() != predictions.len() {
            return Err(LightGBMError::dimension_mismatch(
                "Invalid dimensions for multiclass gradient calculation",
                format!("predictions: {}, labels: {}, gradients: {}, hessians: {}", 
                       predictions.len(), labels.len(), gradients.len(), hessians.len()),
            ));
        }
        
        // Multiclass logistic loss
        for i in 0..num_samples {
            let start_idx = i * self.config.num_class;
            let end_idx = start_idx + self.config.num_class;
            let scores = &predictions[start_idx..end_idx];
            let label = labels[i] as usize;
            
            // Calculate softmax probabilities
            let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = scores.iter()
                .map(|&score| (score - max_score).exp())
                .collect();
            let sum_exp: f64 = exp_scores.iter().sum();
            
            for j in 0..self.config.num_class {
                let prob = exp_scores[j] / sum_exp;
                let target = if j == label { 1.0 } else { 0.0 };
                
                gradients[start_idx + j] = prob - target;
                hessians[start_idx + j] = prob * (1.0 - prob);
            }
        }
        
        Ok(())
    }
}

/// Factory function to create objective functions
pub fn create_objective_function(config: &ObjectiveConfig) -> Result<Box<dyn ObjectiveFunction>> {
    match config.objective_type {
        ObjectiveType::Regression => Ok(Box::new(RegressionObjective)),
        ObjectiveType::Binary => Ok(Box::new(BinaryObjective::new(config.clone())?)),
        ObjectiveType::Multiclass => Ok(Box::new(MulticlassObjective::new(config.clone())?)),
        _ => Err(LightGBMError::not_implemented(format!("Objective type {:?}", config.objective_type))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_objective_config_default() {
        let config = ObjectiveConfig::default();
        assert_eq!(config.objective_type, ObjectiveType::Regression);
        assert_eq!(config.num_class, 1);
        assert!(!config.is_unbalance);
        assert_eq!(config.scale_pos_weight, 1.0);
    }

    #[test]
    fn test_objective_config_builders() {
        let regression = ObjectiveConfig::regression();
        assert_eq!(regression.objective_type, ObjectiveType::Regression);
        
        let binary = ObjectiveConfig::binary();
        assert_eq!(binary.objective_type, ObjectiveType::Binary);
        
        let multiclass = ObjectiveConfig::multiclass(5);
        assert_eq!(multiclass.objective_type, ObjectiveType::Multiclass);
        assert_eq!(multiclass.num_class, 5);
    }

    #[test]
    fn test_objective_config_validation() {
        let mut config = ObjectiveConfig::multiclass(5);
        assert!(config.validate().is_ok());
        
        config.num_class = 1;
        assert!(config.validate().is_err());
        
        config.num_class = 5;
        config.scale_pos_weight = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_regression_objective() {
        let objective = RegressionObjective;
        assert_eq!(objective.name(), "regression");
        assert_eq!(objective.num_model_outputs(), 1);
        assert!(!objective.requires_class_weights());
        
        let predictions = [1.0, 2.0, 3.0];
        let labels = [1.5, 2.5, 2.0];
        let mut gradients = [0.0; 3];
        let mut hessians = [0.0; 3];
        
        objective.calculate_gradients_hessians(&predictions, &labels, &mut gradients, &mut hessians).unwrap();
        
        assert_eq!(gradients[0], -0.5);
        assert_eq!(gradients[1], -0.5);
        assert_eq!(gradients[2], 1.0);
        assert_eq!(hessians[0], 1.0);
        assert_eq!(hessians[1], 1.0);
        assert_eq!(hessians[2], 1.0);
    }

    #[test]
    fn test_binary_objective() {
        let config = ObjectiveConfig::binary();
        let objective = BinaryObjective::new(config).unwrap();
        
        assert_eq!(objective.name(), "binary");
        assert_eq!(objective.num_model_outputs(), 1);
        assert!(objective.requires_class_weights());
        
        let predictions = [0.0, 1.0, -1.0];
        let transformed = objective.transform_predictions(&predictions).unwrap();
        
        assert!((transformed[0] - 0.5).abs() < 1e-10);
        assert!(transformed[1] > 0.5);
        assert!(transformed[2] < 0.5);
    }

    #[test]
    fn test_multiclass_objective() {
        let config = ObjectiveConfig::multiclass(3);
        let objective = MulticlassObjective::new(config).unwrap();
        
        assert_eq!(objective.name(), "multiclass");
        assert_eq!(objective.num_model_outputs(), 3);
        assert!(objective.requires_class_weights());
        
        let predictions = [1.0, 2.0, 0.0];  // One sample with 3 classes
        let transformed = objective.transform_predictions(&predictions).unwrap();
        
        // Check that probabilities sum to 1
        let sum: f64 = transformed.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        
        // Check that all probabilities are positive
        for prob in &transformed {
            assert!(*prob > 0.0);
        }
    }

    #[test]
    fn test_objective_factory() {
        let regression_config = ObjectiveConfig::regression();
        let regression_obj = create_objective_function(&regression_config).unwrap();
        assert_eq!(regression_obj.name(), "regression");
        
        let binary_config = ObjectiveConfig::binary();
        let binary_obj = create_objective_function(&binary_config).unwrap();
        assert_eq!(binary_obj.name(), "binary");
        
        let multiclass_config = ObjectiveConfig::multiclass(5);
        let multiclass_obj = create_objective_function(&multiclass_config).unwrap();
        assert_eq!(multiclass_obj.name(), "multiclass");
    }
}