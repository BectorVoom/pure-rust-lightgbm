//! Metrics evaluation module for Pure Rust LightGBM.
//!
//! This module provides evaluation metrics for regression and classification tasks,
//! including standard metrics like RMSE, MAE, accuracy, precision, recall, F1-score, etc.

use ndarray::{ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

/// Regression evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionMetrics {
    /// Root Mean Square Error
    pub rmse: f64,
    /// Mean Absolute Error
    pub mae: f64,
    /// R-squared (coefficient of determination)
    pub r2: f64,
    /// Mean Squared Error
    pub mse: f64,
    /// Mean Absolute Percentage Error
    pub mape: f64,
}

/// Binary classification evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationMetrics {
    /// Accuracy
    pub accuracy: f64,
    /// Precision
    pub precision: f64,
    /// Recall
    pub recall: f64,
    /// F1-score
    pub f1_score: f64,
    /// Area Under ROC Curve
    pub auc: f64,
    /// Log loss
    pub log_loss: f64,
}

/// Multiclass classification evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MulticlassMetrics {
    /// Overall accuracy
    pub accuracy: f64,
    /// Macro-averaged F1-score
    pub macro_f1: f64,
    /// Weighted F1-score
    pub weighted_f1: f64,
    /// Per-class precision
    pub per_class_precision: Vec<f64>,
    /// Per-class recall
    pub per_class_recall: Vec<f64>,
    /// Per-class F1-score
    pub per_class_f1: Vec<f64>,
    /// Multiclass log-loss (cross-entropy loss)
    pub log_loss: f64,
    /// Per-class AUC-ROC scores (one-vs-rest)
    pub per_class_auc: Vec<f64>,
    /// Macro-averaged AUC
    pub macro_auc: f64,
}

/// Evaluate regression metrics
pub fn evaluate_regression(predictions: &ArrayView1<'_, f32>, true_values: &ArrayView1<'_, f32>) -> RegressionMetrics {
    let n = predictions.len() as f64;
    
    // Calculate basic metrics
    let mut sum_squared_error = 0.0;
    let mut sum_absolute_error = 0.0;
    let mut sum_percentage_error = 0.0;
    
    for (&pred, &true_val) in predictions.iter().zip(true_values.iter()) {
        let error = pred - true_val;
        sum_squared_error += error * error;
        sum_absolute_error += error.abs();
        
        if true_val != 0.0 {
            sum_percentage_error += ((error / true_val).abs() * 100.0) as f64;
        }
    }
    
    let mse = sum_squared_error as f64 / n;
    let rmse = mse.sqrt();
    let mae = sum_absolute_error as f64 / n;
    let mape = sum_percentage_error / n;
    
    // Calculate R-squared
    let true_mean = true_values.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mut total_sum_squares = 0.0;
    
    for &true_val in true_values.iter() {
        let diff = true_val as f64 - true_mean;
        total_sum_squares += diff * diff;
    }
    
    let r2 = if total_sum_squares > 0.0 {
        1.0 - (sum_squared_error as f64 / total_sum_squares)
    } else {
        0.0
    };
    
    RegressionMetrics {
        rmse,
        mae,
        r2,
        mse,
        mape,
    }
}

/// Evaluate binary classification metrics
pub fn evaluate_binary_classification(
    class_predictions: &ArrayView1<'_, f32>,
    prob_predictions: &ArrayView2<'_, f32>,
    true_labels: &ArrayView1<'_, f32>,
) -> ClassificationMetrics {
    let n = class_predictions.len();
    
    // Calculate confusion matrix components
    let mut tp = 0.0; // True positives
    let mut fp = 0.0; // False positives
    let mut tn = 0.0; // True negatives
    let mut fn_ = 0.0; // False negatives
    
    for i in 0..n {
        let pred = class_predictions[i];
        let true_label = true_labels[i];
        
        match (pred > 0.5, true_label > 0.5) {
            (true, true) => tp += 1.0,
            (true, false) => fp += 1.0,
            (false, false) => tn += 1.0,
            (false, true) => fn_ += 1.0,
        }
    }
    
    // Calculate metrics
    let accuracy = (tp + tn) / (tp + fp + tn + fn_);
    let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
    let f1_score = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    
    // Calculate AUC (simplified implementation)
    let auc = calculate_auc_roc(prob_predictions, true_labels);
    
    // Calculate log loss
    let log_loss = calculate_log_loss(prob_predictions, true_labels);
    
    ClassificationMetrics {
        accuracy,
        precision,
        recall,
        f1_score,
        auc,
        log_loss,
    }
}

/// Evaluate multiclass classification metrics
pub fn evaluate_multiclass_classification(
    class_predictions: &ArrayView1<'_, f32>,
    prob_predictions: &ArrayView2<'_, f32>,
    true_labels: &ArrayView1<'_, f32>,
    num_classes: usize,
) -> MulticlassMetrics {
    let n = class_predictions.len();
    
    // Calculate overall accuracy
    let mut correct = 0;
    for i in 0..n {
        if (class_predictions[i] as usize) == (true_labels[i] as usize) {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / n as f64;
    
    // Calculate per-class metrics
    let mut per_class_precision = vec![0.0; num_classes];
    let mut per_class_recall = vec![0.0; num_classes];
    let mut per_class_f1 = vec![0.0; num_classes];
    
    for class in 0..num_classes {
        let mut tp = 0.0;
        let mut fp = 0.0;
        let mut fn_ = 0.0;
        
        for i in 0..n {
            let pred_class = class_predictions[i] as usize;
            let true_class = true_labels[i] as usize;
            
            if pred_class == class && true_class == class {
                tp += 1.0;
            } else if pred_class == class && true_class != class {
                fp += 1.0;
            } else if pred_class != class && true_class == class {
                fn_ += 1.0;
            }
        }
        
        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        
        per_class_precision[class] = precision;
        per_class_recall[class] = recall;
        per_class_f1[class] = f1;
    }
    
    // Calculate macro and weighted averages
    let macro_f1 = per_class_f1.iter().sum::<f64>() / num_classes as f64;
    
    // For weighted F1, we need class frequencies
    let mut class_counts = vec![0.0; num_classes];
    for &label in true_labels.iter() {
        if (label as usize) < num_classes {
            class_counts[label as usize] += 1.0;
        }
    }
    
    let mut weighted_f1 = 0.0;
    for class in 0..num_classes {
        let weight = class_counts[class] / n as f64;
        weighted_f1 += per_class_f1[class] * weight;
    }
    
    // Calculate probability-based metrics using prob_predictions
    let log_loss = calculate_multiclass_log_loss(prob_predictions, true_labels, num_classes);
    let per_class_auc = calculate_multiclass_auc(prob_predictions, true_labels, num_classes);
    let macro_auc = per_class_auc.iter().sum::<f64>() / num_classes as f64;
    
    MulticlassMetrics {
        accuracy,
        macro_f1,
        weighted_f1,
        per_class_precision,
        per_class_recall,
        per_class_f1,
        log_loss,
        per_class_auc,
        macro_auc,
    }
}

/// Calculate AUC-ROC (simplified implementation)
fn calculate_auc_roc(prob_predictions: &ArrayView2<'_, f32>, true_labels: &ArrayView1<'_, f32>) -> f64 {
    let n = prob_predictions.nrows();
    if n == 0 || prob_predictions.ncols() < 2 {
        return 0.5; // Random classifier
    }
    
    // Get positive class probabilities
    let mut pairs: Vec<(f32, f32)> = Vec::new();
    for i in 0..n {
        let prob = prob_predictions[[i, 1]]; // Probability of positive class
        let label = true_labels[i];
        pairs.push((prob, label));
    }
    
    // Sort by probability (descending)
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    
    // Calculate AUC using trapezoidal rule
    let mut auc = 0.0;
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut prev_fpr = 0.0;
    
    let total_pos = true_labels.iter().filter(|&&x| x > 0.5).count() as f64;
    let total_neg = n as f64 - total_pos;
    
    if total_pos == 0.0 || total_neg == 0.0 {
        return 0.5;
    }
    
    for (_, label) in pairs {
        if label > 0.5 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        
        let tpr = tp / total_pos;
        let fpr = fp / total_neg;
        
        auc += (fpr - prev_fpr) * tpr;
        prev_fpr = fpr;
    }
    
    auc
}

/// Calculate log loss
fn calculate_log_loss(prob_predictions: &ArrayView2<'_, f32>, true_labels: &ArrayView1<'_, f32>) -> f64 {
    let n = prob_predictions.nrows();
    if n == 0 || prob_predictions.ncols() < 2 {
        return 0.0;
    }
    
    let mut log_loss = 0.0;
    for i in 0..n {
        let true_label = true_labels[i];
        let pred_prob = if true_label > 0.5 {
            prob_predictions[[i, 1]] // Positive class probability
        } else {
            prob_predictions[[i, 0]] // Negative class probability
        };
        
        // Clamp probability to avoid log(0)
        let clamped_prob = pred_prob.clamp(1e-15, 1.0 - 1e-15);
        log_loss -= clamped_prob.ln() as f64;
    }
    
    log_loss / n as f64
}

/// Calculate multiclass log-loss (cross-entropy loss)
fn calculate_multiclass_log_loss(prob_predictions: &ArrayView2<'_, f32>, true_labels: &ArrayView1<'_, f32>, num_classes: usize) -> f64 {
    let n = prob_predictions.nrows();
    if n == 0 || prob_predictions.ncols() != num_classes {
        return 0.0;
    }
    
    let mut log_loss = 0.0;
    for i in 0..n {
        let true_class = true_labels[i] as usize;
        if true_class < num_classes {
            let pred_prob = prob_predictions[[i, true_class]];
            // Clamp probabilities to avoid log(0)
            let clamped_prob = pred_prob.clamp(1e-15_f32, 1.0 - 1e-15_f32);
            log_loss -= (clamped_prob as f64).ln();
        }
    }
    
    log_loss / n as f64
}

/// Calculate per-class AUC using one-vs-rest approach
fn calculate_multiclass_auc(prob_predictions: &ArrayView2<'_, f32>, true_labels: &ArrayView1<'_, f32>, num_classes: usize) -> Vec<f64> {
    let n = prob_predictions.nrows();
    let mut per_class_auc = vec![0.5; num_classes]; // Default to random classifier
    
    if n == 0 || prob_predictions.ncols() != num_classes {
        return per_class_auc;
    }
    
    for class in 0..num_classes {
        // Create binary labels: 1 if sample belongs to this class, 0 otherwise
        let mut binary_labels = vec![0.0; n];
        let mut class_probs = vec![0.0; n];
        
        for i in 0..n {
            binary_labels[i] = if (true_labels[i] as usize) == class { 1.0 } else { 0.0 };
            class_probs[i] = prob_predictions[[i, class]] as f64;
        }
        
        // Calculate AUC for this class using binary classification approach
        per_class_auc[class] = calculate_binary_auc(&class_probs, &binary_labels);
    }
    
    per_class_auc
}

/// Calculate binary AUC-ROC using trapezoidal rule
fn calculate_binary_auc(probabilities: &[f64], labels: &[f64]) -> f64 {
    let n = probabilities.len();
    if n == 0 {
        return 0.5;
    }
    
    // Create (probability, label) pairs and sort by probability descending
    let mut pairs: Vec<(f64, f64)> = probabilities.iter().zip(labels.iter()).map(|(&p, &l)| (p, l)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    
    let total_pos = labels.iter().sum::<f64>();
    let total_neg = n as f64 - total_pos;
    
    if total_pos == 0.0 || total_neg == 0.0 {
        return 0.5; // Random classifier when all samples are same class
    }
    
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut auc = 0.0;
    let mut prev_fpr = 0.0;
    
    for (_, label) in pairs {
        if label > 0.5 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        
        let tpr = tp / total_pos;
        let fpr = fp / total_neg;
        
        // Trapezoidal rule: area = (fpr - prev_fpr) * tpr
        auc += (fpr - prev_fpr) * tpr;
        prev_fpr = fpr;
    }
    
    auc
}

/// Calculate custom metrics
pub fn calculate_custom_metric(
    predictions: &ArrayView1<'_, f32>,
    true_values: &ArrayView1<'_, f32>,
    metric_name: &str,
) -> f64 {
    match metric_name.to_lowercase().as_str() {
        "mape" => {
            let mut sum = 0.0;
            let mut count = 0;
            for (&pred, &true_val) in predictions.iter().zip(true_values.iter()) {
                if true_val != 0.0 {
                    sum += ((pred - true_val) / true_val).abs() as f64;
                    count += 1;
                }
            }
            if count > 0 { sum / count as f64 * 100.0 } else { 0.0 }
        },
        "smape" => {
            let mut sum = 0.0;
            let n = predictions.len();
            for (&pred, &true_val) in predictions.iter().zip(true_values.iter()) {
                let denominator = (pred.abs() + true_val.abs()) / 2.0;
                if denominator != 0.0 {
                    sum += ((pred - true_val).abs() / denominator) as f64;
                }
            }
            sum / n as f64 * 100.0
        },
        _ => 0.0, // Unknown metric
    }
}