//! Classification metrics for model evaluation.
//!
//! This module provides comprehensive classification metrics including
//! accuracy, precision, recall, F1-score, AUC, log loss, and other
//! statistical measures for evaluating binary and multiclass classification
//! model performance.

use crate::core::{
    types::*,
    error::{LightGBMError, Result},
    traits::*,
};
use crate::metrics::utils;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::HashMap;

/// Classification metrics calculator for binary and multiclass classification.
#[derive(Debug, Clone)]
pub struct ClassificationMetrics {
    /// Configuration for metric calculation
    config: ClassificationMetricsConfig,
    /// Cached metric values
    cached_metrics: HashMap<String, f64>,
}

/// Configuration for classification metrics calculation.
#[derive(Debug, Clone)]
pub struct ClassificationMetricsConfig {
    /// Number of classes for multiclass classification
    pub num_classes: usize,
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
    /// Threshold for binary classification
    pub threshold: f64,
    /// Whether to compute per-class metrics
    pub per_class_metrics: bool,
    /// Top-k for top-k accuracy
    pub top_k: Vec<usize>,
    /// Whether to compute ROC curve points
    pub compute_roc_curve: bool,
    /// Whether to compute PR curve points
    pub compute_pr_curve: bool,
}

impl Default for ClassificationMetricsConfig {
    fn default() -> Self {
        Self {
            num_classes: 2,
            compute_all: true,
            handle_missing: true,
            epsilon: 1e-15,
            use_weights: false,
            compute_confidence_intervals: false,
            confidence_level: 0.95,
            threshold: 0.5,
            per_class_metrics: false,
            top_k: vec![1, 3, 5],
            compute_roc_curve: false,
            compute_pr_curve: false,
        }
    }
}

impl ClassificationMetrics {
    /// Create a new classification metrics calculator.
    pub fn new(config: ClassificationMetricsConfig) -> Self {
        Self {
            config,
            cached_metrics: HashMap::new(),
        }
    }

    /// Calculate all classification metrics.
    pub fn calculate(
        &mut self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<ClassificationMetricsResult> {
        self.validate_inputs(predictions, targets, weights)?;
        
        let mut result = ClassificationMetricsResult::new(self.config.num_classes);
        
        if self.config.num_classes == 2 {
            // Binary classification
            let binary_result = self.calculate_binary_metrics(predictions, targets, weights)?;
            result.binary_metrics = Some(binary_result);
        } else {
            // Multiclass classification
            let multiclass_result = self.calculate_multiclass_metrics(predictions, targets, weights)?;
            result.multiclass_metrics = Some(multiclass_result);
        }
        
        // Calculate sample size
        result.sample_size = predictions.len();
        
        // Calculate confidence intervals if requested
        if self.config.compute_confidence_intervals {
            result.confidence_intervals = Some(self.calculate_confidence_intervals(predictions, targets, weights)?);
        }
        
        Ok(result)
    }

    /// Calculate metrics for multiclass classification with probability predictions.
    pub fn calculate_multiclass_proba(
        &mut self,
        predictions: &ArrayView2<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<ClassificationMetricsResult> {
        self.validate_multiclass_inputs(predictions, targets, weights)?;
        
        let mut result = ClassificationMetricsResult::new(self.config.num_classes);
        
        // Calculate multiclass metrics with probability predictions
        let multiclass_result = self.calculate_multiclass_proba_metrics(predictions, targets, weights)?;
        result.multiclass_metrics = Some(multiclass_result);
        
        // Calculate sample size
        result.sample_size = predictions.nrows();
        
        // Calculate confidence intervals if requested
        if self.config.compute_confidence_intervals {
            // Convert to single predictions for confidence intervals
            let single_predictions = self.convert_proba_to_single_predictions(predictions);
            result.confidence_intervals = Some(self.calculate_confidence_intervals(
                &single_predictions.view(),
                targets,
                weights,
            )?);
        }
        
        Ok(result)
    }

    /// Validate input arrays for single predictions.
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

    /// Validate input arrays for multiclass probability predictions.
    fn validate_multiclass_inputs(
        &self,
        predictions: &ArrayView2<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<()> {
        if predictions.nrows() != targets.len() {
            return Err(LightGBMError::dimension_mismatch(
                format!("predictions rows: {}", predictions.nrows()),
                format!("targets: {}", targets.len()),
            ));
        }
        
        if predictions.ncols() != self.config.num_classes {
            return Err(LightGBMError::dimension_mismatch(
                format!("predictions cols: {}", predictions.ncols()),
                format!("num_classes: {}", self.config.num_classes),
            ));
        }
        
        if let Some(w) = weights {
            if w.len() != predictions.nrows() {
                return Err(LightGBMError::dimension_mismatch(
                    format!("weights: {}", w.len()),
                    format!("predictions rows: {}", predictions.nrows()),
                ));
            }
        }
        
        if predictions.nrows() == 0 {
            return Err(LightGBMError::config("Empty input arrays"));
        }
        
        Ok(())
    }

    /// Calculate binary classification metrics.
    fn calculate_binary_metrics(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<BinaryClassificationMetrics> {
        let mut metrics = BinaryClassificationMetrics::new();
        
        // Calculate confusion matrix
        let (tp, fp, tn, fn_) = utils::binary_confusion_matrix(predictions, targets, self.config.threshold);
        metrics.confusion_matrix = ConfusionMatrix::new_binary(tp, fp, tn, fn_);
        
        // Calculate basic metrics
        metrics.accuracy = self.calculate_accuracy(predictions, targets, weights)?;
        metrics.precision = self.calculate_precision(tp, fp);
        metrics.recall = self.calculate_recall(tp, fn_);
        metrics.f1_score = self.calculate_f1_score(metrics.precision, metrics.recall);
        metrics.specificity = self.calculate_specificity(tn, fp);
        
        // Calculate probability-based metrics
        metrics.auc = utils::calculate_auc(predictions, targets);
        metrics.log_loss = utils::calculate_log_loss(predictions, targets, self.config.epsilon);
        
        if self.config.compute_all {
            metrics.brier_score = self.calculate_brier_score(predictions, targets, weights)?;
            metrics.matthews_corrcoef = self.calculate_matthews_corrcoef(tp, fp, tn, fn_);
            metrics.balanced_accuracy = self.calculate_balanced_accuracy(metrics.recall, metrics.specificity);
            metrics.cohen_kappa = self.calculate_cohen_kappa(tp, fp, tn, fn_);
            metrics.jaccard_score = self.calculate_jaccard_score(tp, fp, fn_);
            metrics.hamming_loss = self.calculate_hamming_loss(predictions, targets, weights)?;
            metrics.zero_one_loss = self.calculate_zero_one_loss(predictions, targets, weights)?;
            metrics.hinge_loss = self.calculate_hinge_loss(predictions, targets, weights)?;
            metrics.fbeta_score = self.calculate_fbeta_score(metrics.precision, metrics.recall, 0.5);
        }
        
        // Calculate curves if requested
        if self.config.compute_roc_curve {
            metrics.roc_curve = Some(self.calculate_roc_curve(predictions, targets)?);
        }
        
        if self.config.compute_pr_curve {
            metrics.pr_curve = Some(self.calculate_pr_curve(predictions, targets)?);
        }
        
        Ok(metrics)
    }

    /// Calculate multiclass classification metrics.
    fn calculate_multiclass_metrics(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<MulticlassClassificationMetrics> {
        let mut metrics = MulticlassClassificationMetrics::new(self.config.num_classes);
        
        // Calculate confusion matrix
        metrics.confusion_matrix = self.calculate_multiclass_confusion_matrix(predictions, targets)?;
        
        // Calculate basic metrics
        metrics.accuracy = self.calculate_accuracy(predictions, targets, weights)?;
        metrics.macro_precision = self.calculate_macro_precision(&metrics.confusion_matrix);
        metrics.macro_recall = self.calculate_macro_recall(&metrics.confusion_matrix);
        metrics.macro_f1_score = self.calculate_macro_f1_score(metrics.macro_precision, metrics.macro_recall);
        metrics.weighted_precision = self.calculate_weighted_precision(&metrics.confusion_matrix);
        metrics.weighted_recall = self.calculate_weighted_recall(&metrics.confusion_matrix);
        metrics.weighted_f1_score = self.calculate_weighted_f1_score(metrics.weighted_precision, metrics.weighted_recall);
        
        if self.config.compute_all {
            metrics.cohen_kappa = self.calculate_multiclass_cohen_kappa(&metrics.confusion_matrix);
            metrics.hamming_loss = self.calculate_hamming_loss(predictions, targets, weights)?;
            metrics.zero_one_loss = self.calculate_zero_one_loss(predictions, targets, weights)?;
        }
        
        // Calculate per-class metrics if requested
        if self.config.per_class_metrics {
            metrics.per_class_precision = Some(self.calculate_per_class_precision(&metrics.confusion_matrix));
            metrics.per_class_recall = Some(self.calculate_per_class_recall(&metrics.confusion_matrix));
            metrics.per_class_f1_score = Some(self.calculate_per_class_f1_score(&metrics.confusion_matrix));
        }
        
        Ok(metrics)
    }

    /// Calculate multiclass classification metrics with probability predictions.
    fn calculate_multiclass_proba_metrics(
        &self,
        predictions: &ArrayView2<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<MulticlassClassificationMetrics> {
        let mut metrics = MulticlassClassificationMetrics::new(self.config.num_classes);
        
        // Convert probability predictions to class predictions
        let class_predictions = self.convert_proba_to_class_predictions(predictions);
        
        // Calculate confusion matrix
        metrics.confusion_matrix = self.calculate_multiclass_confusion_matrix(&class_predictions.view(), targets)?;
        
        // Calculate basic metrics
        metrics.accuracy = self.calculate_accuracy(&class_predictions.view(), targets, weights)?;
        metrics.macro_precision = self.calculate_macro_precision(&metrics.confusion_matrix);
        metrics.macro_recall = self.calculate_macro_recall(&metrics.confusion_matrix);
        metrics.macro_f1_score = self.calculate_macro_f1_score(metrics.macro_precision, metrics.macro_recall);
        metrics.weighted_precision = self.calculate_weighted_precision(&metrics.confusion_matrix);
        metrics.weighted_recall = self.calculate_weighted_recall(&metrics.confusion_matrix);
        metrics.weighted_f1_score = self.calculate_weighted_f1_score(metrics.weighted_precision, metrics.weighted_recall);
        
        // Calculate probability-based metrics
        metrics.log_loss = Some(self.calculate_multiclass_log_loss(predictions, targets, weights)?);
        
        // Calculate top-k accuracy
        for &k in &self.config.top_k {
            let top_k_acc = self.calculate_top_k_accuracy(predictions, targets, k)?;
            metrics.top_k_accuracy.insert(k, top_k_acc);
        }
        
        if self.config.compute_all {
            metrics.cohen_kappa = self.calculate_multiclass_cohen_kappa(&metrics.confusion_matrix);
            metrics.hamming_loss = self.calculate_hamming_loss(&class_predictions.view(), targets, weights)?;
            metrics.zero_one_loss = self.calculate_zero_one_loss(&class_predictions.view(), targets, weights)?;
        }
        
        // Calculate per-class metrics if requested
        if self.config.per_class_metrics {
            metrics.per_class_precision = Some(self.calculate_per_class_precision(&metrics.confusion_matrix));
            metrics.per_class_recall = Some(self.calculate_per_class_recall(&metrics.confusion_matrix));
            metrics.per_class_f1_score = Some(self.calculate_per_class_f1_score(&metrics.confusion_matrix));
        }
        
        Ok(metrics)
    }

    /// Calculate accuracy.
    fn calculate_accuracy(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<f64> {
        let correct_predictions = if self.config.num_classes == 2 {
            predictions.iter()
                .zip(targets.iter())
                .map(|(&pred, &target)| {
                    let predicted_class = if pred >= self.config.threshold as f32 { 1.0 } else { 0.0 };
                    if predicted_class == target { 1.0 } else { 0.0 }
                })
                .sum::<f32>()
        } else {
            predictions.iter()
                .zip(targets.iter())
                .map(|(&pred, &target)| {
                    if pred == target { 1.0 } else { 0.0 }
                })
                .sum::<f32>()
        };

        if let Some(w) = weights {
            let weighted_correct = predictions.iter()
                .zip(targets.iter())
                .zip(w.iter())
                .map(|((&pred, &target), &weight)| {
                    let predicted_class = if self.config.num_classes == 2 {
                        if pred >= self.config.threshold as f32 { 1.0 } else { 0.0 }
                    } else {
                        pred
                    };
                    if predicted_class == target { weight } else { 0.0 }
                })
                .sum::<f32>();
            let weight_sum = w.sum();
            Ok((weighted_correct / weight_sum) as f64)
        } else {
            Ok((correct_predictions / predictions.len() as f32) as f64)
        }
    }

    /// Calculate precision.
    fn calculate_precision(&self, tp: usize, fp: usize) -> f64 {
        if tp + fp == 0 {
            0.0
        } else {
            tp as f64 / (tp + fp) as f64
        }
    }

    /// Calculate recall.
    fn calculate_recall(&self, tp: usize, fn_: usize) -> f64 {
        if tp + fn_ == 0 {
            0.0
        } else {
            tp as f64 / (tp + fn_) as f64
        }
    }

    /// Calculate F1 score.
    fn calculate_f1_score(&self, precision: f64, recall: f64) -> f64 {
        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }

    /// Calculate specificity.
    fn calculate_specificity(&self, tn: usize, fp: usize) -> f64 {
        if tn + fp == 0 {
            0.0
        } else {
            tn as f64 / (tn + fp) as f64
        }
    }

    /// Calculate Brier score.
    fn calculate_brier_score(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<f64> {
        let brier_sum = if let Some(w) = weights {
            predictions.iter()
                .zip(targets.iter())
                .zip(w.iter())
                .map(|((&pred, &target), &weight)| weight * (pred - target).powi(2))
                .sum::<f32>()
        } else {
            predictions.iter()
                .zip(targets.iter())
                .map(|(&pred, &target)| (pred - target).powi(2))
                .sum::<f32>()
        };

        let total_weight = if let Some(w) = weights {
            w.sum()
        } else {
            predictions.len() as f32
        };

        Ok((brier_sum / total_weight) as f64)
    }

    /// Calculate Matthews correlation coefficient.
    fn calculate_matthews_corrcoef(&self, tp: usize, fp: usize, tn: usize, fn_: usize) -> f64 {
        let numerator = (tp * tn) as f64 - (fp * fn_) as f64;
        let denominator = ((tp + fp) as f64 * (tp + fn_) as f64 * (tn + fp) as f64 * (tn + fn_) as f64).sqrt();
        
        if denominator.abs() < self.config.epsilon {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Calculate balanced accuracy.
    fn calculate_balanced_accuracy(&self, recall: f64, specificity: f64) -> f64 {
        (recall + specificity) / 2.0
    }

    /// Calculate Cohen's kappa.
    fn calculate_cohen_kappa(&self, tp: usize, fp: usize, tn: usize, fn_: usize) -> f64 {
        let total = (tp + fp + tn + fn_) as f64;
        let po = (tp + tn) as f64 / total;
        let pe = ((tp + fp) as f64 * (tp + fn_) as f64 + (tn + fn_) as f64 * (tn + fp) as f64) / (total * total);
        
        if (1.0 - pe).abs() < self.config.epsilon {
            0.0
        } else {
            (po - pe) / (1.0 - pe)
        }
    }

    /// Calculate Jaccard score.
    fn calculate_jaccard_score(&self, tp: usize, fp: usize, fn_: usize) -> f64 {
        let union = tp + fp + fn_;
        if union == 0 {
            0.0
        } else {
            tp as f64 / union as f64
        }
    }

    /// Calculate Hamming loss.
    fn calculate_hamming_loss(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<f64> {
        let incorrect_predictions = if self.config.num_classes == 2 {
            predictions.iter()
                .zip(targets.iter())
                .map(|(&pred, &target)| {
                    let predicted_class = if pred >= self.config.threshold as f32 { 1.0 } else { 0.0 };
                    if predicted_class != target { 1.0 } else { 0.0 }
                })
                .sum::<f32>()
        } else {
            predictions.iter()
                .zip(targets.iter())
                .map(|(&pred, &target)| {
                    if pred != target { 1.0 } else { 0.0 }
                })
                .sum::<f32>()
        };

        if let Some(w) = weights {
            let weighted_incorrect = predictions.iter()
                .zip(targets.iter())
                .zip(w.iter())
                .map(|((&pred, &target), &weight)| {
                    let predicted_class = if self.config.num_classes == 2 {
                        if pred >= self.config.threshold as f32 { 1.0 } else { 0.0 }
                    } else {
                        pred
                    };
                    if predicted_class != target { weight } else { 0.0 }
                })
                .sum::<f32>();
            let weight_sum = w.sum();
            Ok((weighted_incorrect / weight_sum) as f64)
        } else {
            Ok((incorrect_predictions / predictions.len() as f32) as f64)
        }
    }

    /// Calculate zero-one loss.
    fn calculate_zero_one_loss(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<f64> {
        self.calculate_hamming_loss(predictions, targets, weights)
    }

    /// Calculate hinge loss.
    fn calculate_hinge_loss(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<f64> {
        let hinge_sum = if let Some(w) = weights {
            predictions.iter()
                .zip(targets.iter())
                .zip(w.iter())
                .map(|((&pred, &target), &weight)| {
                    let y = if target == 1.0 { 1.0 } else { -1.0 };
                    let loss = (1.0 - y * pred).max(0.0);
                    weight * loss
                })
                .sum::<f32>()
        } else {
            predictions.iter()
                .zip(targets.iter())
                .map(|(&pred, &target)| {
                    let y = if target == 1.0 { 1.0 } else { -1.0 };
                    (1.0 - y * pred).max(0.0)
                })
                .sum::<f32>()
        };

        let total_weight = if let Some(w) = weights {
            w.sum()
        } else {
            predictions.len() as f32
        };

        Ok((hinge_sum / total_weight) as f64)
    }

    /// Calculate F-beta score.
    fn calculate_fbeta_score(&self, precision: f64, recall: f64, beta: f64) -> f64 {
        let beta_squared = beta * beta;
        let denominator = beta_squared * precision + recall;
        
        if denominator == 0.0 {
            0.0
        } else {
            (1.0 + beta_squared) * precision * recall / denominator
        }
    }

    /// Calculate multiclass confusion matrix.
    fn calculate_multiclass_confusion_matrix(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
    ) -> Result<Array2<usize>> {
        let mut matrix = Array2::zeros((self.config.num_classes, self.config.num_classes));
        
        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            let pred_class = pred as usize;
            let target_class = target as usize;
            
            if pred_class < self.config.num_classes && target_class < self.config.num_classes {
                matrix[[target_class, pred_class]] += 1;
            }
        }
        
        Ok(matrix)
    }

    /// Calculate macro precision.
    fn calculate_macro_precision(&self, confusion_matrix: &Array2<usize>) -> f64 {
        let mut precision_sum = 0.0;
        
        for class in 0..self.config.num_classes {
            let tp = confusion_matrix[[class, class]];
            let fp = confusion_matrix.column(class).sum() - tp;
            let precision = self.calculate_precision(tp, fp);
            precision_sum += precision;
        }
        
        precision_sum / self.config.num_classes as f64
    }

    /// Calculate macro recall.
    fn calculate_macro_recall(&self, confusion_matrix: &Array2<usize>) -> f64 {
        let mut recall_sum = 0.0;
        
        for class in 0..self.config.num_classes {
            let tp = confusion_matrix[[class, class]];
            let fn_ = confusion_matrix.row(class).sum() - tp;
            let recall = self.calculate_recall(tp, fn_);
            recall_sum += recall;
        }
        
        recall_sum / self.config.num_classes as f64
    }

    /// Calculate macro F1 score.
    fn calculate_macro_f1_score(&self, macro_precision: f64, macro_recall: f64) -> f64 {
        self.calculate_f1_score(macro_precision, macro_recall)
    }

    /// Calculate weighted precision.
    fn calculate_weighted_precision(&self, confusion_matrix: &Array2<usize>) -> f64 {
        let mut weighted_precision = 0.0;
        let total_samples = confusion_matrix.sum();
        
        for class in 0..self.config.num_classes {
            let tp = confusion_matrix[[class, class]];
            let fp = confusion_matrix.column(class).sum() - tp;
            let precision = self.calculate_precision(tp, fp);
            let class_support = confusion_matrix.row(class).sum();
            weighted_precision += precision * class_support as f64;
        }
        
        if total_samples > 0 {
            weighted_precision / total_samples as f64
        } else {
            0.0
        }
    }

    /// Calculate weighted recall.
    fn calculate_weighted_recall(&self, confusion_matrix: &Array2<usize>) -> f64 {
        let mut weighted_recall = 0.0;
        let total_samples = confusion_matrix.sum();
        
        for class in 0..self.config.num_classes {
            let tp = confusion_matrix[[class, class]];
            let fn_ = confusion_matrix.row(class).sum() - tp;
            let recall = self.calculate_recall(tp, fn_);
            let class_support = confusion_matrix.row(class).sum();
            weighted_recall += recall * class_support as f64;
        }
        
        if total_samples > 0 {
            weighted_recall / total_samples as f64
        } else {
            0.0
        }
    }

    /// Calculate weighted F1 score.
    fn calculate_weighted_f1_score(&self, weighted_precision: f64, weighted_recall: f64) -> f64 {
        self.calculate_f1_score(weighted_precision, weighted_recall)
    }

    /// Calculate multiclass Cohen's kappa.
    fn calculate_multiclass_cohen_kappa(&self, confusion_matrix: &Array2<usize>) -> f64 {
        let total = confusion_matrix.sum() as f64;
        let po = confusion_matrix.diag().iter().map(|&x| x as f64).sum::<f64>() / total;
        
        let mut pe = 0.0;
        for class in 0..self.config.num_classes {
            let row_sum = confusion_matrix.row(class).sum() as f64;
            let col_sum = confusion_matrix.column(class).sum() as f64;
            pe += (row_sum * col_sum) / (total * total);
        }
        
        if (1.0 - pe).abs() < self.config.epsilon {
            0.0
        } else {
            (po - pe) / (1.0 - pe)
        }
    }

    /// Calculate per-class precision.
    fn calculate_per_class_precision(&self, confusion_matrix: &Array2<usize>) -> Vec<f64> {
        (0..self.config.num_classes)
            .map(|class| {
                let tp = confusion_matrix[[class, class]];
                let fp = confusion_matrix.column(class).sum() - tp;
                self.calculate_precision(tp, fp)
            })
            .collect()
    }

    /// Calculate per-class recall.
    fn calculate_per_class_recall(&self, confusion_matrix: &Array2<usize>) -> Vec<f64> {
        (0..self.config.num_classes)
            .map(|class| {
                let tp = confusion_matrix[[class, class]];
                let fn_ = confusion_matrix.row(class).sum() - tp;
                self.calculate_recall(tp, fn_)
            })
            .collect()
    }

    /// Calculate per-class F1 score.
    fn calculate_per_class_f1_score(&self, confusion_matrix: &Array2<usize>) -> Vec<f64> {
        (0..self.config.num_classes)
            .map(|class| {
                let tp = confusion_matrix[[class, class]];
                let fp = confusion_matrix.column(class).sum() - tp;
                let fn_ = confusion_matrix.row(class).sum() - tp;
                let precision = self.calculate_precision(tp, fp);
                let recall = self.calculate_recall(tp, fn_);
                self.calculate_f1_score(precision, recall)
            })
            .collect()
    }

    /// Calculate multiclass log loss.
    fn calculate_multiclass_log_loss(
        &self,
        predictions: &ArrayView2<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<f64> {
        let eps = self.config.epsilon as f32;
        let mut log_loss = 0.0;
        let mut total_weight = 0.0;

        for (sample_idx, &target) in targets.iter().enumerate() {
            let target_class = target as usize;
            if target_class >= self.config.num_classes {
                continue;
            }

            let weight = weights.map(|w| w[sample_idx]).unwrap_or(1.0);
            let pred_prob = predictions[[sample_idx, target_class]];
            let clamped_prob = pred_prob.max(eps).min(1.0 - eps);
            
            log_loss += weight * -clamped_prob.ln();
            total_weight += weight;
        }

        if total_weight > 0.0 {
            Ok((log_loss / total_weight) as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate top-k accuracy.
    fn calculate_top_k_accuracy(
        &self,
        predictions: &ArrayView2<Score>,
        targets: &ArrayView1<Label>,
        k: usize,
    ) -> Result<f64> {
        let mut correct = 0;
        let num_samples = predictions.nrows();

        for (sample_idx, &target) in targets.iter().enumerate() {
            let target_class = target as usize;
            if target_class >= self.config.num_classes {
                continue;
            }

            let sample_predictions = predictions.row(sample_idx);
            let mut indexed_predictions: Vec<(usize, f32)> = sample_predictions
                .iter()
                .enumerate()
                .map(|(idx, &pred)| (idx, pred))
                .collect();

            indexed_predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top_k_classes: Vec<usize> = indexed_predictions
                .iter()
                .take(k)
                .map(|(idx, _)| *idx)
                .collect();

            if top_k_classes.contains(&target_class) {
                correct += 1;
            }
        }

        Ok(correct as f64 / num_samples as f64)
    }

    /// Convert probability predictions to class predictions.
    fn convert_proba_to_class_predictions(&self, predictions: &ArrayView2<Score>) -> Array1<Score> {
        let mut class_predictions = Array1::zeros(predictions.nrows());
        
        for (sample_idx, mut row) in predictions.axis_iter(ndarray::Axis(0)).enumerate() {
            let mut max_prob = row[0];
            let mut max_class = 0;
            
            for (class_idx, &prob) in row.iter().enumerate() {
                if prob > max_prob {
                    max_prob = prob;
                    max_class = class_idx;
                }
            }
            
            class_predictions[sample_idx] = max_class as f32;
        }
        
        class_predictions
    }

    /// Convert probability predictions to single predictions for confidence intervals.
    fn convert_proba_to_single_predictions(&self, predictions: &ArrayView2<Score>) -> Array1<Score> {
        if self.config.num_classes == 2 {
            // For binary classification, return probabilities for positive class
            predictions.column(1).to_owned()
        } else {
            // For multiclass, return max probability
            let mut max_probs = Array1::zeros(predictions.nrows());
            
            for (sample_idx, row) in predictions.axis_iter(ndarray::Axis(0)).enumerate() {
                let max_prob = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                max_probs[sample_idx] = max_prob;
            }
            
            max_probs
        }
    }

    /// Calculate ROC curve points.
    fn calculate_roc_curve(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
    ) -> Result<Vec<(f64, f64)>> {
        let mut pairs: Vec<(f32, f32)> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred, target))
            .collect();

        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut roc_points = Vec::new();
        let mut tp = 0.0;
        let mut fp = 0.0;
        let total_positive = targets.iter().map(|&t| t as f64).sum::<f64>();
        let total_negative = targets.len() as f64 - total_positive;

        roc_points.push((0.0, 0.0)); // Start at origin

        for (_, target) in pairs {
            if target == 1.0 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            
            let tpr = if total_positive > 0.0 { tp / total_positive } else { 0.0 };
            let fpr = if total_negative > 0.0 { fp / total_negative } else { 0.0 };
            roc_points.push((fpr, tpr));
        }

        Ok(roc_points)
    }

    /// Calculate precision-recall curve points.
    fn calculate_pr_curve(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
    ) -> Result<Vec<(f64, f64)>> {
        let mut pairs: Vec<(f32, f32)> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred, target))
            .collect();

        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut pr_points = Vec::new();
        let mut tp = 0.0;
        let mut fp = 0.0;
        let total_positive = targets.iter().map(|&t| t as f64).sum::<f64>();

        for (_, target) in pairs {
            if target == 1.0 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            
            let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let recall = if total_positive > 0.0 { tp / total_positive } else { 0.0 };
            pr_points.push((recall, precision));
        }

        Ok(pr_points)
    }

    /// Calculate confidence intervals for classification metrics.
    fn calculate_confidence_intervals(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<ClassificationConfidenceIntervals> {
        // Simple bootstrap confidence intervals
        let n_bootstrap = 1000;
        let n_samples = predictions.len();
        let mut bootstrap_accuracy = Vec::new();
        
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
            
            let accuracy = self.calculate_accuracy(
                &bootstrap_pred_view.view(),
                &bootstrap_target_view.view(),
                bootstrap_weight_view.as_ref().map(|w| w.view()),
            )?;
            
            bootstrap_accuracy.push(accuracy);
        }
        
        bootstrap_accuracy.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let alpha = 1.0 - self.config.confidence_level;
        let lower_percentile = (alpha / 2.0 * n_bootstrap as f64) as usize;
        let upper_percentile = ((1.0 - alpha / 2.0) * n_bootstrap as f64) as usize;
        
        Ok(ClassificationConfidenceIntervals {
            confidence_level: self.config.confidence_level,
            accuracy_interval: (bootstrap_accuracy[lower_percentile], bootstrap_accuracy[upper_percentile]),
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &ClassificationMetricsConfig {
        &self.config
    }

    /// Set the configuration.
    pub fn set_config(&mut self, config: ClassificationMetricsConfig) {
        self.config = config;
        self.cached_metrics.clear();
    }
}

/// Result of classification metrics calculation.
#[derive(Debug, Clone)]
pub struct ClassificationMetricsResult {
    /// Number of classes
    pub num_classes: usize,
    /// Binary classification metrics (if applicable)
    pub binary_metrics: Option<BinaryClassificationMetrics>,
    /// Multiclass classification metrics (if applicable)
    pub multiclass_metrics: Option<MulticlassClassificationMetrics>,
    /// Sample size
    pub sample_size: usize,
    /// Confidence intervals (if computed)
    pub confidence_intervals: Option<ClassificationConfidenceIntervals>,
}

impl ClassificationMetricsResult {
    /// Create a new result.
    pub fn new(num_classes: usize) -> Self {
        Self {
            num_classes,
            binary_metrics: None,
            multiclass_metrics: None,
            sample_size: 0,
            confidence_intervals: None,
        }
    }

    /// Get the primary metric (accuracy).
    pub fn primary_metric(&self) -> f64 {
        if let Some(ref binary) = self.binary_metrics {
            binary.accuracy
        } else if let Some(ref multiclass) = self.multiclass_metrics {
            multiclass.accuracy
        } else {
            0.0
        }
    }

    /// Get a summary of the metrics.
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        
        if let Some(ref binary) = self.binary_metrics {
            summary.push_str(&format!("Binary Classification Metrics Summary (n={}):\n", self.sample_size));
            summary.push_str(&format!("  Accuracy: {:.6}\n", binary.accuracy));
            summary.push_str(&format!("  Precision: {:.6}\n", binary.precision));
            summary.push_str(&format!("  Recall: {:.6}\n", binary.recall));
            summary.push_str(&format!("  F1 Score: {:.6}\n", binary.f1_score));
            summary.push_str(&format!("  AUC: {:.6}\n", binary.auc));
            summary.push_str(&format!("  Log Loss: {:.6}\n", binary.log_loss));
        }
        
        if let Some(ref multiclass) = self.multiclass_metrics {
            summary.push_str(&format!("Multiclass Classification Metrics Summary (n={}):\n", self.sample_size));
            summary.push_str(&format!("  Accuracy: {:.6}\n", multiclass.accuracy));
            summary.push_str(&format!("  Macro Precision: {:.6}\n", multiclass.macro_precision));
            summary.push_str(&format!("  Macro Recall: {:.6}\n", multiclass.macro_recall));
            summary.push_str(&format!("  Macro F1 Score: {:.6}\n", multiclass.macro_f1_score));
            summary.push_str(&format!("  Weighted Precision: {:.6}\n", multiclass.weighted_precision));
            summary.push_str(&format!("  Weighted Recall: {:.6}\n", multiclass.weighted_recall));
            summary.push_str(&format!("  Weighted F1 Score: {:.6}\n", multiclass.weighted_f1_score));
            
            if let Some(ref log_loss) = multiclass.log_loss {
                summary.push_str(&format!("  Log Loss: {:.6}\n", log_loss));
            }
        }
        
        if let Some(ref ci) = self.confidence_intervals {
            summary.push_str(&format!("\nConfidence Intervals ({:.0}%):\n", ci.confidence_level * 100.0));
            summary.push_str(&format!("  Accuracy: [{:.6}, {:.6}]\n", ci.accuracy_interval.0, ci.accuracy_interval.1));
        }
        
        summary
    }
}

/// Binary classification metrics.
#[derive(Debug, Clone)]
pub struct BinaryClassificationMetrics {
    /// Accuracy
    pub accuracy: f64,
    /// Precision
    pub precision: f64,
    /// Recall (sensitivity)
    pub recall: f64,
    /// F1 score
    pub f1_score: f64,
    /// Specificity
    pub specificity: f64,
    /// Area under the ROC curve
    pub auc: f64,
    /// Log loss
    pub log_loss: f64,
    /// Brier score
    pub brier_score: f64,
    /// Matthews correlation coefficient
    pub matthews_corrcoef: f64,
    /// Balanced accuracy
    pub balanced_accuracy: f64,
    /// Cohen's kappa
    pub cohen_kappa: f64,
    /// Jaccard score
    pub jaccard_score: f64,
    /// Hamming loss
    pub hamming_loss: f64,
    /// Zero-one loss
    pub zero_one_loss: f64,
    /// Hinge loss
    pub hinge_loss: f64,
    /// F-beta score
    pub fbeta_score: f64,
    /// Confusion matrix
    pub confusion_matrix: ConfusionMatrix,
    /// ROC curve points (FPR, TPR)
    pub roc_curve: Option<Vec<(f64, f64)>>,
    /// PR curve points (recall, precision)
    pub pr_curve: Option<Vec<(f64, f64)>>,
}

impl BinaryClassificationMetrics {
    /// Create a new binary classification metrics instance.
    pub fn new() -> Self {
        Self {
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            specificity: 0.0,
            auc: 0.0,
            log_loss: 0.0,
            brier_score: 0.0,
            matthews_corrcoef: 0.0,
            balanced_accuracy: 0.0,
            cohen_kappa: 0.0,
            jaccard_score: 0.0,
            hamming_loss: 0.0,
            zero_one_loss: 0.0,
            hinge_loss: 0.0,
            fbeta_score: 0.0,
            confusion_matrix: ConfusionMatrix::new_binary(0, 0, 0, 0),
            roc_curve: None,
            pr_curve: None,
        }
    }
}

/// Multiclass classification metrics.
#[derive(Debug, Clone)]
pub struct MulticlassClassificationMetrics {
    /// Number of classes
    pub num_classes: usize,
    /// Accuracy
    pub accuracy: f64,
    /// Macro-averaged precision
    pub macro_precision: f64,
    /// Macro-averaged recall
    pub macro_recall: f64,
    /// Macro-averaged F1 score
    pub macro_f1_score: f64,
    /// Weighted precision
    pub weighted_precision: f64,
    /// Weighted recall
    pub weighted_recall: f64,
    /// Weighted F1 score
    pub weighted_f1_score: f64,
    /// Log loss (for probability predictions)
    pub log_loss: Option<f64>,
    /// Cohen's kappa
    pub cohen_kappa: f64,
    /// Hamming loss
    pub hamming_loss: f64,
    /// Zero-one loss
    pub zero_one_loss: f64,
    /// Top-k accuracy
    pub top_k_accuracy: HashMap<usize, f64>,
    /// Confusion matrix
    pub confusion_matrix: Array2<usize>,
    /// Per-class precision
    pub per_class_precision: Option<Vec<f64>>,
    /// Per-class recall
    pub per_class_recall: Option<Vec<f64>>,
    /// Per-class F1 score
    pub per_class_f1_score: Option<Vec<f64>>,
}

impl MulticlassClassificationMetrics {
    /// Create a new multiclass classification metrics instance.
    pub fn new(num_classes: usize) -> Self {
        Self {
            num_classes,
            accuracy: 0.0,
            macro_precision: 0.0,
            macro_recall: 0.0,
            macro_f1_score: 0.0,
            weighted_precision: 0.0,
            weighted_recall: 0.0,
            weighted_f1_score: 0.0,
            log_loss: None,
            cohen_kappa: 0.0,
            hamming_loss: 0.0,
            zero_one_loss: 0.0,
            top_k_accuracy: HashMap::new(),
            confusion_matrix: Array2::zeros((num_classes, num_classes)),
            per_class_precision: None,
            per_class_recall: None,
            per_class_f1_score: None,
        }
    }
}

/// Confusion matrix for classification.
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    /// Matrix values
    pub matrix: Array2<usize>,
    /// True positives (for binary classification)
    pub tp: Option<usize>,
    /// False positives (for binary classification)
    pub fp: Option<usize>,
    /// True negatives (for binary classification)
    pub tn: Option<usize>,
    /// False negatives (for binary classification)
    pub fn_: Option<usize>,
}

impl ConfusionMatrix {
    /// Create a new binary confusion matrix.
    pub fn new_binary(tp: usize, fp: usize, tn: usize, fn_: usize) -> Self {
        let mut matrix = Array2::zeros((2, 2));
        matrix[[0, 0]] = tn;
        matrix[[0, 1]] = fp;
        matrix[[1, 0]] = fn_;
        matrix[[1, 1]] = tp;
        
        Self {
            matrix,
            tp: Some(tp),
            fp: Some(fp),
            tn: Some(tn),
            fn_: Some(fn_),
        }
    }

    /// Create a new multiclass confusion matrix.
    pub fn new_multiclass(matrix: Array2<usize>) -> Self {
        Self {
            matrix,
            tp: None,
            fp: None,
            tn: None,
            fn_: None,
        }
    }
}

/// Classification threshold information.
#[derive(Debug, Clone)]
pub struct ClassificationThreshold {
    /// Threshold value
    pub threshold: f64,
    /// Precision at this threshold
    pub precision: f64,
    /// Recall at this threshold
    pub recall: f64,
    /// F1 score at this threshold
    pub f1_score: f64,
    /// True positive rate
    pub tpr: f64,
    /// False positive rate
    pub fpr: f64,
}

/// Confidence intervals for classification metrics.
#[derive(Debug, Clone)]
pub struct ClassificationConfidenceIntervals {
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Accuracy confidence interval
    pub accuracy_interval: (f64, f64),
}

/// Builder for classification metrics configuration.
#[derive(Debug)]
pub struct ClassificationMetricsConfigBuilder {
    config: ClassificationMetricsConfig,
}

impl ClassificationMetricsConfigBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: ClassificationMetricsConfig::default(),
        }
    }

    /// Set the number of classes.
    pub fn num_classes(mut self, num_classes: usize) -> Self {
        self.config.num_classes = num_classes;
        self
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

    /// Set the classification threshold.
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.config.threshold = threshold;
        self
    }

    /// Set whether to compute per-class metrics.
    pub fn per_class_metrics(mut self, per_class: bool) -> Self {
        self.config.per_class_metrics = per_class;
        self
    }

    /// Set the top-k values.
    pub fn top_k(mut self, top_k: Vec<usize>) -> Self {
        self.config.top_k = top_k;
        self
    }

    /// Set whether to compute ROC curve.
    pub fn compute_roc_curve(mut self, compute_roc: bool) -> Self {
        self.config.compute_roc_curve = compute_roc;
        self
    }

    /// Set whether to compute PR curve.
    pub fn compute_pr_curve(mut self, compute_pr: bool) -> Self {
        self.config.compute_pr_curve = compute_pr;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> ClassificationMetricsConfig {
        self.config
    }
}

impl Default for ClassificationMetricsConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ClassificationMetrics {
    fn default() -> Self {
        Self::new(ClassificationMetricsConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_classification_metrics_config_default() {
        let config = ClassificationMetricsConfig::default();
        assert_eq!(config.num_classes, 2);
        assert!(config.compute_all);
        assert!(config.handle_missing);
        assert_eq!(config.epsilon, 1e-15);
        assert!(!config.use_weights);
        assert!(!config.compute_confidence_intervals);
        assert_eq!(config.confidence_level, 0.95);
        assert_eq!(config.threshold, 0.5);
        assert!(!config.per_class_metrics);
        assert_eq!(config.top_k, vec![1, 3, 5]);
        assert!(!config.compute_roc_curve);
        assert!(!config.compute_pr_curve);
    }

    #[test]
    fn test_classification_metrics_config_builder() {
        let config = ClassificationMetricsConfigBuilder::new()
            .num_classes(3)
            .compute_all(false)
            .handle_missing(false)
            .epsilon(1e-10)
            .use_weights(true)
            .compute_confidence_intervals(true)
            .confidence_level(0.99)
            .threshold(0.6)
            .per_class_metrics(true)
            .top_k(vec![1, 2])
            .compute_roc_curve(true)
            .compute_pr_curve(true)
            .build();

        assert_eq!(config.num_classes, 3);
        assert!(!config.compute_all);
        assert!(!config.handle_missing);
        assert_eq!(config.epsilon, 1e-10);
        assert!(config.use_weights);
        assert!(config.compute_confidence_intervals);
        assert_eq!(config.confidence_level, 0.99);
        assert_eq!(config.threshold, 0.6);
        assert!(config.per_class_metrics);
        assert_eq!(config.top_k, vec![1, 2]);
        assert!(config.compute_roc_curve);
        assert!(config.compute_pr_curve);
    }

    #[test]
    fn test_binary_classification_accuracy() {
        let mut metrics = ClassificationMetrics::default();
        let predictions = Array1::from_vec(vec![0.8, 0.3, 0.9, 0.1]);
        let targets = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);
        
        let accuracy = metrics.calculate_accuracy(&predictions.view(), &targets.view(), None).unwrap();
        assert_eq!(accuracy, 1.0); // Perfect accuracy
    }

    #[test]
    fn test_binary_classification_metrics() {
        let mut metrics = ClassificationMetrics::default();
        let predictions = Array1::from_vec(vec![0.8, 0.3, 0.9, 0.1]);
        let targets = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);
        
        let result = metrics.calculate(&predictions.view(), &targets.view(), None).unwrap();
        assert!(result.binary_metrics.is_some());
        
        let binary_metrics = result.binary_metrics.unwrap();
        assert_eq!(binary_metrics.accuracy, 1.0);
        assert_eq!(binary_metrics.precision, 1.0);
        assert_eq!(binary_metrics.recall, 1.0);
        assert_eq!(binary_metrics.f1_score, 1.0);
        assert_eq!(binary_metrics.specificity, 1.0);
    }

    #[test]
    fn test_multiclass_confusion_matrix() {
        let config = ClassificationMetricsConfigBuilder::new()
            .num_classes(3)
            .build();
        let mut metrics = ClassificationMetrics::new(config);
        
        let predictions = Array1::from_vec(vec![0.0, 1.0, 2.0, 0.0]);
        let targets = Array1::from_vec(vec![0.0, 1.0, 2.0, 1.0]);
        
        let matrix = metrics.calculate_multiclass_confusion_matrix(&predictions.view(), &targets.view()).unwrap();
        assert_eq!(matrix[[0, 0]], 1); // True class 0, predicted class 0
        assert_eq!(matrix[[1, 1]], 1); // True class 1, predicted class 1
        assert_eq!(matrix[[2, 2]], 1); // True class 2, predicted class 2
        assert_eq!(matrix[[1, 0]], 1); // True class 1, predicted class 0
    }

    #[test]
    fn test_multiclass_classification_metrics() {
        let config = ClassificationMetricsConfigBuilder::new()
            .num_classes(3)
            .build();
        let mut metrics = ClassificationMetrics::new(config);
        
        let predictions = Array1::from_vec(vec![0.0, 1.0, 2.0, 0.0]);
        let targets = Array1::from_vec(vec![0.0, 1.0, 2.0, 1.0]);
        
        let result = metrics.calculate(&predictions.view(), &targets.view(), None).unwrap();
        assert!(result.multiclass_metrics.is_some());
        
        let multiclass_metrics = result.multiclass_metrics.unwrap();
        assert_eq!(multiclass_metrics.accuracy, 0.75); // 3 out of 4 correct
    }

    #[test]
    fn test_validation_errors() {
        let mut metrics = ClassificationMetrics::default();
        let predictions = Array1::from_vec(vec![0.8, 0.3]);
        let targets = Array1::from_vec(vec![1.0, 0.0, 1.0]);
        
        let result = metrics.calculate(&predictions.view(), &targets.view(), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_confusion_matrix_binary() {
        let matrix = ConfusionMatrix::new_binary(10, 5, 20, 3);
        assert_eq!(matrix.tp, Some(10));
        assert_eq!(matrix.fp, Some(5));
        assert_eq!(matrix.tn, Some(20));
        assert_eq!(matrix.fn_, Some(3));
        assert_eq!(matrix.matrix[[1, 1]], 10); // TP
        assert_eq!(matrix.matrix[[0, 1]], 5);  // FP
        assert_eq!(matrix.matrix[[0, 0]], 20); // TN
        assert_eq!(matrix.matrix[[1, 0]], 3);  // FN
    }

    #[test]
    fn test_classification_result_summary() {
        let mut result = ClassificationMetricsResult::new(2);
        result.sample_size = 100;
        result.binary_metrics = Some(BinaryClassificationMetrics {
            accuracy: 0.85,
            precision: 0.80,
            recall: 0.90,
            f1_score: 0.85,
            auc: 0.92,
            log_loss: 0.35,
            ..BinaryClassificationMetrics::new()
        });
        
        let summary = result.summary();
        assert!(summary.contains("Binary Classification"));
        assert!(summary.contains("Accuracy: 0.850000"));
        assert!(summary.contains("Precision: 0.800000"));
        assert!(summary.contains("Recall: 0.900000"));
        assert!(summary.contains("F1 Score: 0.850000"));
        assert!(summary.contains("AUC: 0.920000"));
        assert!(summary.contains("Log Loss: 0.350000"));
        assert!(summary.contains("n=100"));
    }

    #[test]
    fn test_weighted_metrics() {
        let mut metrics = ClassificationMetrics::default();
        let predictions = Array1::from_vec(vec![0.8, 0.3, 0.9, 0.1]);
        let targets = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);
        let weights = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        
        let accuracy = metrics.calculate_accuracy(&predictions.view(), &targets.view(), Some(&weights.view())).unwrap();
        assert_eq!(accuracy, 1.0); // Perfect accuracy regardless of weights
    }

    #[test]
    fn test_multiclass_proba_predictions() {
        let config = ClassificationMetricsConfigBuilder::new()
            .num_classes(3)
            .top_k(vec![1, 2])
            .build();
        let mut metrics = ClassificationMetrics::new(config);
        
        let predictions = Array2::from_shape_vec((4, 3), vec![
            0.7, 0.2, 0.1,  // Class 0
            0.1, 0.8, 0.1,  // Class 1
            0.1, 0.1, 0.8,  // Class 2
            0.4, 0.5, 0.1,  // Class 1
        ]).unwrap();
        let targets = Array1::from_vec(vec![0.0, 1.0, 2.0, 1.0]);
        
        let result = metrics.calculate_multiclass_proba(&predictions.view(), &targets.view(), None).unwrap();
        assert!(result.multiclass_metrics.is_some());
        
        let multiclass_metrics = result.multiclass_metrics.unwrap();
        assert_eq!(multiclass_metrics.accuracy, 1.0); // Perfect accuracy
        assert!(multiclass_metrics.log_loss.is_some());
        assert!(multiclass_metrics.top_k_accuracy.contains_key(&1));
        assert!(multiclass_metrics.top_k_accuracy.contains_key(&2));
    }
}