//! Early stopping mechanisms for prediction pipeline.
//!
//! This module provides early stopping functionality for prediction operations,
//! allowing for more efficient inference when high confidence is achieved early.

use crate::core::types::*;
use ndarray::ArrayView1;
use serde::{Deserialize, Serialize};

/// Configuration for prediction early stopping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionEarlyStopping {
    /// Minimum number of iterations before early stopping can trigger
    pub min_iterations: usize,
    /// Confidence threshold for early stopping
    pub confidence_threshold: f64,
    /// Number of consecutive iterations with high confidence required
    pub patience: usize,
    /// Whether early stopping is enabled
    pub enabled: bool,
}

impl PredictionEarlyStopping {
    /// Create new early stopping configuration
    pub fn new() -> Self {
        Self {
            min_iterations: 10,
            confidence_threshold: 0.95,
            patience: 3,
            enabled: false,
        }
    }

    /// Enable early stopping with given threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold;
        self.enabled = true;
        self
    }

    /// Set minimum iterations
    pub fn with_min_iterations(mut self, min_iter: usize) -> Self {
        self.min_iterations = min_iter;
        self
    }

    /// Set patience (consecutive high-confidence iterations)
    pub fn with_patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    /// Check if early stopping should trigger
    pub fn should_stop(&self, current_iteration: usize, confidence_history: &[f64]) -> bool {
        if !self.enabled || current_iteration < self.min_iterations {
            return false;
        }

        // Check if we have enough confidence history
        if confidence_history.len() < self.patience {
            return false;
        }

        // Implement predictions-based early stopping decision using convergence criteria
        let recent_window = self.patience;
        let total_len = confidence_history.len();
        
        if total_len < recent_window * 2 {
            // Not enough history for convergence analysis
            return false;
        }

        // Calculate prediction stability using confidence variance
        let recent_confidence = &confidence_history[total_len - recent_window..];
        let previous_confidence = &confidence_history[total_len - recent_window * 2..total_len - recent_window];
        
        // Calculate average confidence for both windows
        let recent_avg = recent_confidence.iter().sum::<f64>() / recent_confidence.len() as f64;
        let previous_avg = previous_confidence.iter().sum::<f64>() / previous_confidence.len() as f64;
        
        // Calculate confidence variance for stability assessment
        let recent_variance = recent_confidence
            .iter()
            .map(|&x| (x - recent_avg).powi(2))
            .sum::<f64>() / recent_confidence.len() as f64;
        
        // Convergence criteria:
        // 1. Recent confidence is above threshold
        // 2. Confidence improvement has plateaued (relative change < 1%)
        // 3. Predictions are stable (low variance)
        let confidence_above_threshold = recent_avg >= self.confidence_threshold;
        let relative_improvement = if previous_avg.abs() > 1e-8 {
            (recent_avg - previous_avg).abs() / previous_avg.abs()
        } else {
            (recent_avg - previous_avg).abs()
        };
        let improvement_plateaued = relative_improvement < 0.01; // Less than 1% change
        let predictions_stable = recent_variance < 0.001; // Low variance indicates stability
        
        confidence_above_threshold && improvement_plateaued && predictions_stable
    }

    /// Calculate confidence score from predictions
    pub fn calculate_confidence(&self, predictions: &ArrayView1<'_, Score>) -> f64 {
        if predictions.is_empty() {
            return 0.0;
        }

        // For binary classification, confidence is max(p, 1-p)
        // For multiclass, confidence is max probability
        // For regression, we use a simple heuristic based on prediction stability

        if predictions.len() == 1 {
            // Regression case - use prediction magnitude as proxy for confidence
            let pred = predictions[0] as f64;
            (pred.abs() / (1.0 + pred.abs())).min(1.0)
        } else {
            // Classification case - find max probability
            let max_prob = predictions
                .iter()
                .map(|&x| x as f64)
                .fold(f64::NEG_INFINITY, f64::max);
            let total_prob: f64 = predictions.iter().map(|&x| (x as f64).exp()).sum();
            let normalized_max = max_prob.exp() / total_prob;
            normalized_max.min(1.0)
        }
    }
}

impl Default for PredictionEarlyStopping {
    fn default() -> Self {
        Self::new()
    }
}

/// Early stopping state tracker
#[derive(Debug)]
pub struct EarlyStoppingTracker {
    config: PredictionEarlyStopping,
    confidence_history: Vec<f64>,
    current_iteration: usize,
}

impl EarlyStoppingTracker {
    /// Create new tracker
    pub fn new(config: PredictionEarlyStopping) -> Self {
        Self {
            config,
            confidence_history: Vec::new(),
            current_iteration: 0,
        }
    }

    /// Update tracker with new predictions
    pub fn update(&mut self, predictions: &ArrayView1<'_, Score>) -> bool {
        self.current_iteration += 1;

        let confidence = self.config.calculate_confidence(predictions);
        self.confidence_history.push(confidence);

        // Keep only recent history to avoid unbounded growth
        if self.confidence_history.len() > self.config.patience * 2 {
            self.confidence_history.remove(0);
        }

        self.config.should_stop(
            self.current_iteration,
            &self.confidence_history,
        )
    }

    /// Reset tracker state
    pub fn reset(&mut self) {
        self.confidence_history.clear();
        self.current_iteration = 0;
    }

    /// Get current confidence
    pub fn current_confidence(&self) -> Option<f64> {
        self.confidence_history.last().copied()
    }

    /// Get average confidence over recent iterations
    pub fn average_recent_confidence(&self) -> f64 {
        if self.confidence_history.is_empty() {
            return 0.0;
        }

        let recent_window = self.config.patience.min(self.confidence_history.len());
        let recent_slice =
            &self.confidence_history[self.confidence_history.len() - recent_window..];
        recent_slice.iter().sum::<f64>() / recent_slice.len() as f64
    }
}
