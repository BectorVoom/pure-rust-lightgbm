//! Core prediction engine for Pure Rust LightGBM.
//!
//! This module provides the main prediction functionality including
//! configuration management and prediction pipeline execution.

use crate::core::error::{LightGBMError, Result};
use crate::core::types::*;
use ndarray::{Array1, ArrayView2};
use serde::{Deserialize, Serialize};

/// Configuration for prediction settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionConfig {
    /// Number of iterations to use for prediction (None = use all)
    pub num_iterations: Option<usize>,
    /// Whether to return raw scores
    pub raw_score: bool,
    /// Whether to predict leaf indices
    pub predict_leaf_index: bool,
    /// Whether to predict feature contributions (SHAP values)
    pub predict_contrib: bool,
    /// Early stopping rounds for prediction
    pub early_stopping_rounds: Option<usize>,
    /// Early stopping margin
    pub early_stopping_margin: f64,
}

impl PredictionConfig {
    /// Create a new prediction configuration with defaults
    pub fn new() -> Self {
        Self {
            num_iterations: None,
            raw_score: false,
            predict_leaf_index: false,
            predict_contrib: false,
            early_stopping_rounds: None,
            early_stopping_margin: 0.0,
        }
    }

    /// Set number of iterations to use for prediction
    pub fn with_num_iterations(mut self, num_iterations: Option<usize>) -> Self {
        self.num_iterations = num_iterations;
        self
    }

    /// Set whether to return raw scores
    pub fn with_raw_score(mut self, raw_score: bool) -> Self {
        self.raw_score = raw_score;
        self
    }

    /// Set whether to predict leaf indices
    pub fn with_predict_leaf_index(mut self, predict_leaf_index: bool) -> Self {
        self.predict_leaf_index = predict_leaf_index;
        self
    }

    /// Set whether to predict feature contributions
    pub fn with_predict_contrib(mut self, predict_contrib: bool) -> Self {
        self.predict_contrib = predict_contrib;
        self
    }

    /// Set early stopping rounds
    pub fn with_early_stopping_rounds(mut self, rounds: Option<usize>) -> Self {
        self.early_stopping_rounds = rounds;
        self
    }

    /// Set early stopping margin
    pub fn with_early_stopping_margin(mut self, margin: f64) -> Self {
        self.early_stopping_margin = margin;
        self
    }
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Predictor trait for making predictions
pub trait PredictorTrait {
    /// Make predictions on features
    fn predict(&self, features: &ArrayView2<'_, f32>) -> Result<Array1<Score>>;

    /// Get prediction configuration
    fn config(&self) -> &PredictionConfig;

    /// Make SHAP predictions if configured
    fn predict_shap(&self, _features: &ArrayView2<'_, f32>) -> Result<Option<ndarray::Array2<f64>>> {
        if self.config().predict_contrib {
            // TODO: Implement actual SHAP prediction calculation using tree traversal
            // This should use the features parameter to compute SHAP values for each sample
            // Default implementation - should be overridden by concrete implementations
            Err(LightGBMError::not_implemented("SHAP prediction in default trait implementation"))
        } else {
            Ok(None)
        }
    }
}

/// Concrete predictor implementation
#[derive(Debug)]
pub struct Predictor {
    config: PredictionConfig,
    model: Option<crate::boosting::GBDT>,
}

impl Predictor {
    /// Create a new predictor with the given model and configuration
    pub fn new(model: crate::boosting::GBDT, config: PredictionConfig) -> Result<Self> {
        Ok(Predictor { 
            config,
            model: Some(model),
        })
    }

    /// Create a new predictor without a model (for testing)
    pub fn new_without_model(config: PredictionConfig) -> Result<Self> {
        Ok(Predictor { 
            config,
            model: None,
        })
    }

    /// Make predictions on features
    pub fn predict(&self, features: &ArrayView2<'_, f32>) -> Result<Array1<Score>> {
        match &self.model {
            Some(model) => {
                // Use the GBDT model's prediction method
                let features_owned = features.to_owned();
                model.predict(&features_owned)
            }
            None => {
                // Placeholder implementation - return zeros for testing
                Ok(Array1::zeros(features.nrows()))
            }
        }
    }

    /// Get prediction configuration
    pub fn config(&self) -> &PredictionConfig {
        &self.config
    }

    /// Make SHAP predictions
    pub fn predict_shap(&self, features: &ArrayView2<'_, f32>) -> Result<Option<ndarray::Array2<f64>>> {
        if !self.config.predict_contrib {
            return Ok(None);
        }

        match &self.model {
            Some(model) => {
                let features_owned = features.to_owned();
                let shap_values = model.predict_contrib(&features_owned)?;
                Ok(Some(shap_values))
            }
            None => {
                Err(LightGBMError::prediction("No trained model available for SHAP calculation"))
            }
        }
    }
}

impl PredictorTrait for Predictor {
    fn predict(&self, features: &ArrayView2<'_, f32>) -> Result<Array1<Score>> {
        self.predict(features)
    }

    fn config(&self) -> &PredictionConfig {
        &self.config
    }

    fn predict_shap(&self, features: &ArrayView2<'_, f32>) -> Result<Option<ndarray::Array2<f64>>> {
        self.predict_shap(features)
    }
}