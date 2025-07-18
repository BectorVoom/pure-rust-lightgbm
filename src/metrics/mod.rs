//! Evaluation metrics for LightGBM models.
//!
//! This module provides comprehensive evaluation metrics for different types of
//! machine learning tasks including regression, classification, and ranking.
//! The metrics are designed to be efficient, accurate, and compatible with
//! various model types and evaluation scenarios.
//!
//! # Examples
//!
//! ## Regression Metrics
//!
//! ```rust,no_run
//! use lightgbm_rust::metrics::{RegressionMetrics, RegressionMetricsConfig};
//! use ndarray::Array1;
//!
//! # fn example() -> lightgbm_rust::Result<()> {
//! let config = RegressionMetricsConfig::default();
//! let mut metrics = RegressionMetrics::new(config);
//!
//! let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0]);
//! let targets = Array1::from_vec(vec![1.1, 2.1, 2.9]);
//!
//! let result = metrics.calculate(&predictions.view(), &targets.view(), None)?;
//! println!("MSE: {:.6}", result.mse);
//! println!("R²: {:.6}", result.r_squared);
//! # Ok(())
//! # }
//! ```
//!
//! ## Classification Metrics
//!
//! ```rust,no_run
//! use lightgbm_rust::metrics::{ClassificationMetrics, ClassificationMetricsConfig};
//! use ndarray::Array1;
//!
//! # fn example() -> lightgbm_rust::Result<()> {
//! let config = ClassificationMetricsConfig::default();
//! let mut metrics = ClassificationMetrics::new(config);
//!
//! let predictions = Array1::from_vec(vec![0.8, 0.3, 0.9]);
//! let targets = Array1::from_vec(vec![1.0, 0.0, 1.0]);
//!
//! let result = metrics.calculate(&predictions.view(), &targets.view(), None)?;
//! println!("AUC: {:.6}", result.auc);
//! println!("Log Loss: {:.6}", result.log_loss);
//! # Ok(())
//! # }
//! ```
//!
//! ## Ranking Metrics
//!
//! ```rust,no_run
//! use lightgbm_rust::metrics::{RankingMetrics, RankingMetricsConfig};
//! use ndarray::Array1;
//!
//! # fn example() -> lightgbm_rust::Result<()> {
//! let config = RankingMetricsConfig::default();
//! let mut metrics = RankingMetrics::new(config);
//!
//! let predictions = Array1::from_vec(vec![3.0, 1.0, 2.0]);
//! let targets = Array1::from_vec(vec![2.0, 0.0, 1.0]);
//! let groups = Array1::from_vec(vec![0, 0, 0]); // Single group
//!
//! let result = metrics.calculate(&predictions.view(), &targets.view(), &groups.view(), None)?;
//! println!("NDCG@5: {:.6}", result.ndcg_at_k.get(&5).unwrap_or(&0.0));
//! # Ok(())
//! # }
//! ```
//!
//! ## Custom Metrics
//!
//! ```rust,no_run
//! use lightgbm_rust::metrics::{CustomMetric, MetricDirection};
//! use ndarray::ArrayView1;
//!
//! # fn example() -> lightgbm_rust::Result<()> {
//! let custom_metric = CustomMetric::new(
//!     "custom_mse",
//!     MetricDirection::Minimize,
//!     |predictions: &ArrayView1<f32>, targets: &ArrayView1<f32>, _weights: Option<&ArrayView1<f32>>| {
//!         let mse = predictions.iter()
//!             .zip(targets.iter())
//!             .map(|(&pred, &target)| (pred - target).powi(2))
//!             .sum::<f32>() / predictions.len() as f32;
//!         Ok(mse as f64)
//!     }
//! );
//!
//! let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0]);
//! let targets = Array1::from_vec(vec![1.1, 2.1, 2.9]);
//! let metric_value = custom_metric.calculate(&predictions.view(), &targets.view(), None)?;
//! println!("Custom MSE: {:.6}", metric_value);
//! # Ok(())
//! # }
//! ```

pub mod regression;
pub mod classification;
pub mod ranking;
pub mod custom;

// Re-export main types for convenience
pub use regression::{
    RegressionMetrics, RegressionMetricsConfig, RegressionMetricsConfigBuilder,
    RegressionMetricsResult, RegressionConfidenceIntervals,
};

pub use classification::{
    ClassificationMetrics, ClassificationMetricsConfig, ClassificationMetricsConfigBuilder,
    ClassificationMetricsResult, ClassificationConfidenceIntervals,
    BinaryClassificationMetrics, MulticlassClassificationMetrics,
    ConfusionMatrix, ClassificationThreshold,
};

pub use ranking::{
    RankingMetrics, RankingMetricsConfig, RankingMetricsConfigBuilder,
    RankingMetricsResult, RankingConfidenceIntervals,
    RankingGroup, RankingItem,
};

pub use custom::{
    CustomMetric, CustomMetricFunction, MetricDirection, MetricType,
    CustomMetricRegistry, CustomMetricResult,
};

/// Common trait for all metric calculators.
pub trait MetricCalculator {
    /// The type of configuration used by this metric calculator.
    type Config;
    
    /// The type of result returned by this metric calculator.
    type Result;
    
    /// Calculate metrics with the given configuration.
    fn calculate_with_config(&mut self, config: Self::Config) -> crate::core::Result<Self::Result>;
    
    /// Get the name of this metric calculator.
    fn name(&self) -> &'static str;
    
    /// Get the metric type (regression, classification, ranking, custom).
    fn metric_type(&self) -> MetricType;
    
    /// Whether this metric should be minimized or maximized.
    fn direction(&self) -> MetricDirection;
}

/// Metric evaluation context providing additional information.
#[derive(Debug, Clone)]
pub struct MetricContext {
    /// Number of samples
    pub num_samples: usize,
    /// Number of features
    pub num_features: usize,
    /// Number of classes (for classification)
    pub num_classes: Option<usize>,
    /// Whether sample weights are used
    pub has_weights: bool,
    /// Model type that generated the predictions
    pub model_type: Option<String>,
    /// Training iteration (for tracking during training)
    pub iteration: Option<usize>,
}

impl MetricContext {
    /// Create a new metric context.
    pub fn new(num_samples: usize, num_features: usize) -> Self {
        Self {
            num_samples,
            num_features,
            num_classes: None,
            has_weights: false,
            model_type: None,
            iteration: None,
        }
    }

    /// Set the number of classes.
    pub fn with_num_classes(mut self, num_classes: usize) -> Self {
        self.num_classes = Some(num_classes);
        self
    }

    /// Set whether weights are used.
    pub fn with_weights(mut self, has_weights: bool) -> Self {
        self.has_weights = has_weights;
        self
    }

    /// Set the model type.
    pub fn with_model_type(mut self, model_type: String) -> Self {
        self.model_type = Some(model_type);
        self
    }

    /// Set the iteration number.
    pub fn with_iteration(mut self, iteration: usize) -> Self {
        self.iteration = Some(iteration);
        self
    }
}

/// Utility functions for metric calculations.
pub mod utils {
    use crate::core::types::*;
    use ndarray::ArrayView1;

    /// Calculate the confusion matrix for binary classification.
    pub fn binary_confusion_matrix(
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        threshold: f64,
    ) -> (usize, usize, usize, usize) {
        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_ = 0;

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            let predicted_class = if pred >= threshold as f32 { 1.0 } else { 0.0 };
            
            match (target as i32, predicted_class as i32) {
                (1, 1) => tp += 1,
                (0, 1) => fp += 1,
                (0, 0) => tn += 1,
                (1, 0) => fn_ += 1,
                _ => {} // Handle other cases if needed
            }
        }

        (tp, fp, tn, fn_)
    }

    /// Calculate area under the ROC curve.
    pub fn calculate_auc(
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
    ) -> f64 {
        let mut pairs: Vec<(f32, f32)> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred, target))
            .collect();

        // Sort by predictions in descending order
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut auc = 0.0;
        let mut tp = 0.0;
        let mut fp = 0.0;
        let total_positive = targets.iter().map(|&t| t as f64).sum::<f64>();
        let total_negative = targets.len() as f64 - total_positive;

        for (_, target) in pairs {
            if target == 1.0 {
                tp += 1.0;
            } else {
                fp += 1.0;
                auc += tp;
            }
        }

        if total_positive == 0.0 || total_negative == 0.0 {
            0.5 // Random classifier AUC
        } else {
            auc / (total_positive * total_negative)
        }
    }

    /// Calculate log loss for binary classification.
    pub fn calculate_log_loss(
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        epsilon: f64,
    ) -> f64 {
        let eps = epsilon as f32;
        let log_loss = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let clamped_pred = pred.max(eps).min(1.0 - eps);
                -(target * clamped_pred.ln() + (1.0 - target) * (1.0 - clamped_pred).ln())
            })
            .sum::<f32>();

        (log_loss / predictions.len() as f32) as f64
    }

    /// Calculate normalized discounted cumulative gain (NDCG).
    pub fn calculate_ndcg(
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        k: usize,
    ) -> f64 {
        let mut items: Vec<(f32, f32)> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred, target))
            .collect();

        // Sort by predictions in descending order
        items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let dcg = calculate_dcg(&items, k);
        let idcg = calculate_idcg(targets, k);

        if idcg == 0.0 {
            0.0
        } else {
            dcg / idcg
        }
    }

    /// Calculate Discounted Cumulative Gain (DCG).
    fn calculate_dcg(items: &[(f32, f32)], k: usize) -> f64 {
        items.iter()
            .take(k)
            .enumerate()
            .map(|(i, (_, relevance))| {
                let gain = (2.0_f32.powf(*relevance) - 1.0) as f64;
                let discount = (i as f64 + 2.0).log2();
                gain / discount
            })
            .sum()
    }

    /// Calculate Ideal Discounted Cumulative Gain (IDCG).
    fn calculate_idcg(targets: &ArrayView1<Label>, k: usize) -> f64 {
        let mut sorted_targets: Vec<f32> = targets.iter().copied().collect();
        sorted_targets.sort_by(|a, b| b.partial_cmp(a).unwrap());

        sorted_targets.iter()
            .take(k)
            .enumerate()
            .map(|(i, &relevance)| {
                let gain = (2.0_f32.powf(relevance) - 1.0) as f64;
                let discount = (i as f64 + 2.0).log2();
                gain / discount
            })
            .sum()
    }

    /// Calculate Mean Reciprocal Rank (MRR).
    pub fn calculate_mrr(
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
    ) -> f64 {
        let mut items: Vec<(f32, f32)> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred, target))
            .collect();

        // Sort by predictions in descending order
        items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        for (rank, (_, relevance)) in items.iter().enumerate() {
            if *relevance > 0.0 {
                return 1.0 / (rank as f64 + 1.0);
            }
        }

        0.0
    }

    /// Calculate precision at k.
    pub fn calculate_precision_at_k(
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        k: usize,
    ) -> f64 {
        let mut items: Vec<(f32, f32)> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred, target))
            .collect();

        // Sort by predictions in descending order
        items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let relevant_items = items.iter()
            .take(k)
            .filter(|(_, relevance)| *relevance > 0.0)
            .count();

        relevant_items as f64 / k.min(items.len()) as f64
    }

    /// Calculate recall at k.
    pub fn calculate_recall_at_k(
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        k: usize,
    ) -> f64 {
        let total_relevant = targets.iter().filter(|&&t| t > 0.0).count();
        
        if total_relevant == 0 {
            return 0.0;
        }

        let mut items: Vec<(f32, f32)> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred, target))
            .collect();

        // Sort by predictions in descending order
        items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let relevant_items = items.iter()
            .take(k)
            .filter(|(_, relevance)| *relevance > 0.0)
            .count();

        relevant_items as f64 / total_relevant as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::*;
    use ndarray::Array1;

    #[test]
    fn test_metric_context_creation() {
        let context = MetricContext::new(100, 10);
        assert_eq!(context.num_samples, 100);
        assert_eq!(context.num_features, 10);
        assert_eq!(context.num_classes, None);
        assert!(!context.has_weights);
        assert_eq!(context.model_type, None);
        assert_eq!(context.iteration, None);
    }

    #[test]
    fn test_metric_context_builder() {
        let context = MetricContext::new(100, 10)
            .with_num_classes(3)
            .with_weights(true)
            .with_model_type("lightgbm".to_string())
            .with_iteration(42);

        assert_eq!(context.num_samples, 100);
        assert_eq!(context.num_features, 10);
        assert_eq!(context.num_classes, Some(3));
        assert!(context.has_weights);
        assert_eq!(context.model_type, Some("lightgbm".to_string()));
        assert_eq!(context.iteration, Some(42));
    }

    #[test]
    fn test_binary_confusion_matrix() {
        let predictions = Array1::from_vec(vec![0.8, 0.3, 0.9, 0.1]);
        let targets = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);
        
        let (tp, fp, tn, fn_) = utils::binary_confusion_matrix(&predictions.view(), &targets.view(), 0.5);
        
        assert_eq!(tp, 2); // 0.8 > 0.5 and target=1, 0.9 > 0.5 and target=1
        assert_eq!(fp, 0); // No false positives
        assert_eq!(tn, 2); // 0.3 < 0.5 and target=0, 0.1 < 0.5 and target=0
        assert_eq!(fn_, 0); // No false negatives
    }

    #[test]
    fn test_calculate_auc() {
        let predictions = Array1::from_vec(vec![0.9, 0.8, 0.3, 0.1]);
        let targets = Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0]);
        
        let auc = utils::calculate_auc(&predictions.view(), &targets.view());
        assert!(auc > 0.8); // Should be high for good separation
    }

    #[test]
    fn test_calculate_log_loss() {
        let predictions = Array1::from_vec(vec![0.9, 0.8, 0.2, 0.1]);
        let targets = Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0]);
        
        let log_loss = utils::calculate_log_loss(&predictions.view(), &targets.view(), 1e-15);
        assert!(log_loss > 0.0); // Log loss should be positive
        assert!(log_loss < 1.0); // Should be reasonable for good predictions
    }

    #[test]
    fn test_calculate_ndcg() {
        let predictions = Array1::from_vec(vec![0.9, 0.8, 0.7, 0.6]);
        let targets = Array1::from_vec(vec![3.0, 2.0, 1.0, 0.0]);
        
        let ndcg = utils::calculate_ndcg(&predictions.view(), &targets.view(), 3);
        assert!(ndcg > 0.0);
        assert!(ndcg <= 1.0);
    }

    #[test]
    fn test_calculate_mrr() {
        let predictions = Array1::from_vec(vec![0.9, 0.8, 0.7, 0.6]);
        let targets = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
        
        let mrr = utils::calculate_mrr(&predictions.view(), &targets.view());
        assert!((mrr - 0.5).abs() < 1e-10); // Should be 1/2 = 0.5 (second position)
    }

    #[test]
    fn test_calculate_precision_at_k() {
        let predictions = Array1::from_vec(vec![0.9, 0.8, 0.7, 0.6]);
        let targets = Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0]);
        
        let precision = utils::calculate_precision_at_k(&predictions.view(), &targets.view(), 2);
        assert!((precision - 1.0).abs() < 1e-10); // Should be 1.0 (2/2)
    }

    #[test]
    fn test_calculate_recall_at_k() {
        let predictions = Array1::from_vec(vec![0.9, 0.8, 0.7, 0.6]);
        let targets = Array1::from_vec(vec![1.0, 1.0, 0.0, 1.0]);
        
        let recall = utils::calculate_recall_at_k(&predictions.view(), &targets.view(), 2);
        assert!((recall - 2.0/3.0).abs() < 1e-10); // Should be 2/3 (2 out of 3 relevant retrieved)
    }
}