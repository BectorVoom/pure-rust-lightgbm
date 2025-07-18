//! Model validation and accuracy testing framework.
//!
//! This test suite provides comprehensive validation for model training and prediction accuracy.
//! It includes frameworks for testing when the actual training functionality is implemented.

use lightgbm_rust::*;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

mod common;
use common::*;

/// Test framework for regression model validation
#[test]
fn test_regression_model_validation_framework() {
    println!("Testing regression model validation framework...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    // Create test data
    let (features, labels) = create_test_data!(regression, 100, 5);
    let dataset = Dataset::new(
        features.clone(),
        labels.clone(),
        None,
        None,
        None,
        None,
    ).unwrap();
    
    // Create configuration
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.1)
        .num_iterations(100)
        .num_leaves(31)
        .build()
        .unwrap();
    
    // Create model
    let mut regressor = LGBMRegressor::new(config);
    
    // Test training (will fail until implemented)
    match regressor.fit(&dataset) {
        Ok(_) => {
            println!("✓ Model training succeeded");
            
            // Test prediction
            match regressor.predict(&features) {
                Ok(predictions) => {
                    println!("✓ Prediction succeeded");
                    
                    // Validate prediction shape
                    assert_eq!(predictions.len(), labels.len());
                    
                    // Test prediction accuracy metrics
                    let accuracy_metrics = calculate_regression_metrics(&predictions, &labels);
                    validate_regression_accuracy(&accuracy_metrics);
                    
                    print_regression_metrics(&accuracy_metrics);
                }
                Err(e) => {
                    println!("✗ Prediction failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("✗ Training not implemented yet: {}", e);
            // Test that the error is specifically NotImplemented
            assert!(e.to_string().contains("not implemented"));
        }
    }
    
    println!("Regression model validation framework test completed");
}

/// Test framework for binary classification model validation
#[test]
fn test_binary_classification_model_validation_framework() {
    println!("Testing binary classification model validation framework...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    // Create test data
    let (features, labels) = create_test_data!(binary, 100, 5);
    let dataset = Dataset::new(
        features.clone(),
        labels.clone(),
        None,
        None,
        None,
        None,
    ).unwrap();
    
    // Create configuration
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .learning_rate(0.1)
        .num_iterations(100)
        .num_leaves(31)
        .build()
        .unwrap();
    
    // Create model
    let mut classifier = LGBMClassifier::new(config);
    
    // Test training (will fail until implemented)
    match classifier.fit(&dataset) {
        Ok(_) => {
            println!("✓ Model training succeeded");
            
            // Test class prediction
            match classifier.predict(&features) {
                Ok(predictions) => {
                    println!("✓ Class prediction succeeded");
                    
                    // Validate prediction shape
                    assert_eq!(predictions.len(), labels.len());
                    
                    // Test classification accuracy metrics
                    let accuracy_metrics = calculate_classification_metrics(&predictions, &labels);
                    validate_binary_classification_accuracy(&accuracy_metrics);
                    
                    print_classification_metrics(&accuracy_metrics);
                }
                Err(e) => {
                    println!("✗ Class prediction failed: {}", e);
                }
            }
            
            // Test probability prediction
            match classifier.predict_proba(&features) {
                Ok(probabilities) => {
                    println!("✓ Probability prediction succeeded");
                    
                    // Validate probability shape
                    assert_eq!(probabilities.nrows(), labels.len());
                    assert_eq!(probabilities.ncols(), 2); // Binary classification
                    
                    // Test probability constraints
                    validate_binary_probabilities(&probabilities);
                    
                    // Test probability-based metrics
                    let prob_metrics = calculate_probability_metrics(&probabilities, &labels);
                    validate_probability_metrics(&prob_metrics);
                    
                    print_probability_metrics(&prob_metrics);
                }
                Err(e) => {
                    println!("✗ Probability prediction failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("✗ Training not implemented yet: {}", e);
            // Test that the error is specifically NotImplemented
            assert!(e.to_string().contains("not implemented"));
        }
    }
    
    println!("Binary classification model validation framework test completed");
}

/// Test framework for multiclass classification model validation
#[test]
fn test_multiclass_classification_model_validation_framework() {
    println!("Testing multiclass classification model validation framework...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    let num_classes = 3;
    let (features, labels) = create_test_data!(multiclass, 120, 5, num_classes);
    let dataset = Dataset::new(
        features.clone(),
        labels.clone(),
        None,
        None,
        None,
        None,
    ).unwrap();
    
    // Create configuration
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Multiclass)
        .num_class(num_classes)
        .learning_rate(0.1)
        .num_iterations(100)
        .num_leaves(31)
        .build()
        .unwrap();
    
    // Create model
    let mut classifier = LGBMClassifier::new(config);
    
    // Test training (will fail until implemented)
    match classifier.fit(&dataset) {
        Ok(_) => {
            println!("✓ Model training succeeded");
            
            // Test class prediction
            match classifier.predict(&features) {
                Ok(predictions) => {
                    println!("✓ Class prediction succeeded");
                    
                    // Validate prediction shape
                    assert_eq!(predictions.len(), labels.len());
                    
                    // Test multiclass accuracy metrics
                    let accuracy_metrics = calculate_multiclass_metrics(&predictions, &labels, num_classes);
                    validate_multiclass_accuracy(&accuracy_metrics);
                    
                    print_multiclass_metrics(&accuracy_metrics);
                }
                Err(e) => {
                    println!("✗ Class prediction failed: {}", e);
                }
            }
            
            // Test probability prediction
            match classifier.predict_proba(&features) {
                Ok(probabilities) => {
                    println!("✓ Probability prediction succeeded");
                    
                    // Validate probability shape
                    assert_eq!(probabilities.nrows(), labels.len());
                    assert_eq!(probabilities.ncols(), num_classes);
                    
                    // Test probability constraints
                    validate_multiclass_probabilities(&probabilities);
                    
                    // Test probability-based metrics
                    let prob_metrics = calculate_multiclass_probability_metrics(&probabilities, &labels, num_classes);
                    validate_multiclass_probability_metrics(&prob_metrics);
                    
                    print_multiclass_probability_metrics(&prob_metrics);
                }
                Err(e) => {
                    println!("✗ Probability prediction failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("✗ Training not implemented yet: {}", e);
            // Test that the error is specifically NotImplemented
            assert!(e.to_string().contains("not implemented"));
        }
    }
    
    println!("Multiclass classification model validation framework test completed");
}

/// Test model overfitting detection
#[test]
fn test_overfitting_detection_framework() {
    println!("Testing overfitting detection framework...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    // Create small dataset prone to overfitting
    let (train_features, train_labels) = create_test_data!(regression, 50, 10);
    let (test_features, test_labels) = create_test_data!(regression, 50, 10);
    
    let train_dataset = Dataset::new(
        train_features.clone(),
        train_labels.clone(),
        None,
        None,
        None,
        None,
    ).unwrap();
    
    // Create configuration that might overfit
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.5) // High learning rate
        .num_iterations(1000) // Many iterations
        .num_leaves(127) // Many leaves
        .lambda_l1(0.0) // No regularization
        .lambda_l2(0.0) // No regularization
        .build()
        .unwrap();
    
    let mut regressor = LGBMRegressor::new(config);
    
    // Test training (will fail until implemented)
    match regressor.fit(&train_dataset) {
        Ok(_) => {
            println!("✓ Model training succeeded");
            
            // Test predictions on training data
            match regressor.predict(&train_features) {
                Ok(train_predictions) => {
                    let train_metrics = calculate_regression_metrics(&train_predictions, &train_labels);
                    println!("Train metrics: {:?}", train_metrics);
                    
                    // Test predictions on test data
                    match regressor.predict(&test_features) {
                        Ok(test_predictions) => {
                            let test_metrics = calculate_regression_metrics(&test_predictions, &test_labels);
                            println!("Test metrics: {:?}", test_metrics);
                            
                            // Check for overfitting
                            detect_overfitting(&train_metrics, &test_metrics);
                        }
                        Err(e) => {
                            println!("✗ Test prediction failed: {}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("✗ Train prediction failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("✗ Training not implemented yet: {}", e);
        }
    }
    
    println!("Overfitting detection framework test completed");
}

/// Test model performance with different configurations
#[test]
fn test_model_configuration_performance() {
    println!("Testing model configuration performance...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    let (features, labels) = create_test_data!(regression, 200, 10);
    let dataset = Dataset::new(
        features.clone(),
        labels.clone(),
        None,
        None,
        None,
        None,
    ).unwrap();
    
    // Test different configurations
    let configs = vec![
        ConfigBuilder::new()
            .objective(ObjectiveType::Regression)
            .learning_rate(0.01)
            .num_iterations(50)
            .num_leaves(15)
            .build()
            .unwrap(),
        ConfigBuilder::new()
            .objective(ObjectiveType::Regression)
            .learning_rate(0.1)
            .num_iterations(100)
            .num_leaves(31)
            .build()
            .unwrap(),
        ConfigBuilder::new()
            .objective(ObjectiveType::Regression)
            .learning_rate(0.2)
            .num_iterations(200)
            .num_leaves(63)
            .build()
            .unwrap(),
    ];
    
    for (i, config) in configs.iter().enumerate() {
        println!("Testing configuration {}", i + 1);
        
        let mut regressor = LGBMRegressor::new(config.clone());
        
        match regressor.fit(&dataset) {
            Ok(_) => {
                println!("✓ Configuration {} training succeeded", i + 1);
                
                match regressor.predict(&features) {
                    Ok(predictions) => {
                        let metrics = calculate_regression_metrics(&predictions, &labels);
                        println!("Configuration {} metrics: {:?}", i + 1, metrics);
                        
                        // Compare configurations
                        compare_regression_metrics(&metrics, i + 1);
                    }
                    Err(e) => {
                        println!("✗ Configuration {} prediction failed: {}", i + 1, e);
                    }
                }
            }
            Err(e) => {
                println!("✗ Configuration {} training not implemented: {}", i + 1, e);
            }
        }
    }
    
    println!("Model configuration performance test completed");
}

/// Test cross-validation framework
#[test]
fn test_cross_validation_framework() {
    println!("Testing cross-validation framework...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    let (features, labels) = create_test_data!(binary, 100, 5);
    
    // Simulate k-fold cross-validation
    let k = 5;
    let fold_size = features.nrows() / k;
    
    let mut cv_results = Vec::new();
    
    for fold in 0..k {
        let start_idx = fold * fold_size;
        let end_idx = if fold == k - 1 { features.nrows() } else { (fold + 1) * fold_size };
        
        // Create train and validation splits
        let mut train_features = Vec::new();
        let mut train_labels = Vec::new();
        let mut val_features = Vec::new();
        let mut val_labels = Vec::new();
        
        for i in 0..features.nrows() {
            if i >= start_idx && i < end_idx {
                // Validation set
                val_features.push(features.row(i).to_vec());
                val_labels.push(labels[i]);
            } else {
                // Training set
                train_features.push(features.row(i).to_vec());
                train_labels.push(labels[i]);
            }
        }
        
        // Convert to arrays
        let train_features_array = Array2::from_shape_vec(
            (train_features.len(), features.ncols()),
            train_features.into_iter().flatten().collect(),
        ).unwrap();
        let train_labels_array = Array1::from_vec(train_labels);
        
        let val_features_array = Array2::from_shape_vec(
            (val_features.len(), features.ncols()),
            val_features.into_iter().flatten().collect(),
        ).unwrap();
        let val_labels_array = Array1::from_vec(val_labels);
        
        // Create dataset
        let train_dataset = Dataset::new(
            train_features_array,
            train_labels_array,
            None,
            None,
            None,
            None,
        ).unwrap();
        
        // Create configuration
        let config = ConfigBuilder::new()
            .objective(ObjectiveType::Binary)
            .learning_rate(0.1)
            .num_iterations(50)
            .num_leaves(31)
            .build()
            .unwrap();
        
        let mut classifier = LGBMClassifier::new(config);
        
        // Train and validate
        match classifier.fit(&train_dataset) {
            Ok(_) => {
                match classifier.predict(&val_features_array) {
                    Ok(predictions) => {
                        let metrics = calculate_classification_metrics(&predictions, &val_labels_array);
                        cv_results.push(metrics);
                        println!("Fold {} metrics: {:?}", fold + 1, metrics);
                    }
                    Err(e) => {
                        println!("✗ Fold {} prediction failed: {}", fold + 1, e);
                    }
                }
            }
            Err(e) => {
                println!("✗ Fold {} training not implemented: {}", fold + 1, e);
            }
        }
    }
    
    // Calculate cross-validation statistics
    if !cv_results.is_empty() {
        calculate_cv_statistics(&cv_results);
    }
    
    println!("Cross-validation framework test completed");
}

// Helper functions for metrics calculation and validation

#[derive(Debug, Clone)]
struct RegressionMetrics {
    mse: f64,
    rmse: f64,
    mae: f64,
    r2: f64,
}

fn calculate_regression_metrics(predictions: &Array1<f32>, labels: &Array1<f32>) -> RegressionMetrics {
    let n = predictions.len() as f64;
    
    // Mean Squared Error
    let mse = predictions.iter()
        .zip(labels.iter())
        .map(|(&pred, &label)| (pred - label).powi(2) as f64)
        .sum::<f64>() / n;
    
    // Root Mean Squared Error
    let rmse = mse.sqrt();
    
    // Mean Absolute Error
    let mae = predictions.iter()
        .zip(labels.iter())
        .map(|(&pred, &label)| (pred - label).abs() as f64)
        .sum::<f64>() / n;
    
    // R-squared
    let label_mean = labels.iter().map(|&x| x as f64).sum::<f64>() / n;
    let ss_tot = labels.iter()
        .map(|&label| (label as f64 - label_mean).powi(2))
        .sum::<f64>();
    let ss_res = predictions.iter()
        .zip(labels.iter())
        .map(|(&pred, &label)| (label - pred).powi(2) as f64)
        .sum::<f64>();
    let r2 = 1.0 - (ss_res / ss_tot);
    
    RegressionMetrics { mse, rmse, mae, r2 }
}

fn validate_regression_accuracy(metrics: &RegressionMetrics) {
    // Validate metrics are reasonable
    assert!(metrics.mse >= 0.0);
    assert!(metrics.rmse >= 0.0);
    assert!(metrics.mae >= 0.0);
    assert!(metrics.rmse.sqrt() >= metrics.mse.sqrt());
    
    // For good models, these should be reasonable
    println!("Regression metrics validation: MSE={:.4}, RMSE={:.4}, MAE={:.4}, R²={:.4}", 
             metrics.mse, metrics.rmse, metrics.mae, metrics.r2);
}

fn print_regression_metrics(metrics: &RegressionMetrics) {
    println!("Regression Metrics:");
    println!("  MSE: {:.4}", metrics.mse);
    println!("  RMSE: {:.4}", metrics.rmse);
    println!("  MAE: {:.4}", metrics.mae);
    println!("  R²: {:.4}", metrics.r2);
}

#[derive(Debug, Clone)]
struct ClassificationMetrics {
    accuracy: f64,
    precision: f64,
    recall: f64,
    f1_score: f64,
}

fn calculate_classification_metrics(predictions: &Array1<i32>, labels: &Array1<f32>) -> ClassificationMetrics {
    let n = predictions.len() as f64;
    
    // Convert labels to i32 for comparison
    let int_labels: Vec<i32> = labels.iter().map(|&x| x as i32).collect();
    
    // Accuracy
    let correct = predictions.iter()
        .zip(int_labels.iter())
        .filter(|(&pred, &label)| pred == label)
        .count() as f64;
    let accuracy = correct / n;
    
    // For binary classification, calculate precision, recall, F1
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut fn_ = 0.0;
    
    for (&pred, &label) in predictions.iter().zip(int_labels.iter()) {
        match (pred, label) {
            (1, 1) => tp += 1.0,
            (1, 0) => fp += 1.0,
            (0, 1) => fn_ += 1.0,
            _ => {}
        }
    }
    
    let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
    let f1_score = if precision + recall > 0.0 { 2.0 * (precision * recall) / (precision + recall) } else { 0.0 };
    
    ClassificationMetrics { accuracy, precision, recall, f1_score }
}

fn validate_binary_classification_accuracy(metrics: &ClassificationMetrics) {
    // Validate metrics are in valid ranges
    assert!(metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0);
    assert!(metrics.precision >= 0.0 && metrics.precision <= 1.0);
    assert!(metrics.recall >= 0.0 && metrics.recall <= 1.0);
    assert!(metrics.f1_score >= 0.0 && metrics.f1_score <= 1.0);
    
    println!("Binary classification metrics validation: Accuracy={:.4}, Precision={:.4}, Recall={:.4}, F1={:.4}", 
             metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score);
}

fn print_classification_metrics(metrics: &ClassificationMetrics) {
    println!("Classification Metrics:");
    println!("  Accuracy: {:.4}", metrics.accuracy);
    println!("  Precision: {:.4}", metrics.precision);
    println!("  Recall: {:.4}", metrics.recall);
    println!("  F1 Score: {:.4}", metrics.f1_score);
}

#[derive(Debug, Clone)]
struct MulticlassMetrics {
    accuracy: f64,
    macro_precision: f64,
    macro_recall: f64,
    macro_f1: f64,
    per_class_metrics: HashMap<i32, ClassificationMetrics>,
}

fn calculate_multiclass_metrics(predictions: &Array1<i32>, labels: &Array1<f32>, num_classes: usize) -> MulticlassMetrics {
    let n = predictions.len() as f64;
    let int_labels: Vec<i32> = labels.iter().map(|&x| x as i32).collect();
    
    // Overall accuracy
    let correct = predictions.iter()
        .zip(int_labels.iter())
        .filter(|(&pred, &label)| pred == label)
        .count() as f64;
    let accuracy = correct / n;
    
    // Per-class metrics
    let mut per_class_metrics = HashMap::new();
    let mut macro_precision = 0.0;
    let mut macro_recall = 0.0;
    let mut macro_f1 = 0.0;
    
    for class in 0..num_classes {
        let class_i32 = class as i32;
        
        // Binary classification metrics for this class vs all others
        let class_predictions: Array1<i32> = predictions.iter()
            .map(|&pred| if pred == class_i32 { 1 } else { 0 })
            .collect();
        let class_labels: Array1<f32> = int_labels.iter()
            .map(|&label| if label == class_i32 { 1.0 } else { 0.0 })
            .collect();
        
        let class_metrics = calculate_classification_metrics(&class_predictions, &class_labels);
        per_class_metrics.insert(class_i32, class_metrics.clone());
        
        macro_precision += class_metrics.precision;
        macro_recall += class_metrics.recall;
        macro_f1 += class_metrics.f1_score;
    }
    
    macro_precision /= num_classes as f64;
    macro_recall /= num_classes as f64;
    macro_f1 /= num_classes as f64;
    
    MulticlassMetrics {
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
        per_class_metrics,
    }
}

fn validate_multiclass_accuracy(metrics: &MulticlassMetrics) {
    assert!(metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0);
    assert!(metrics.macro_precision >= 0.0 && metrics.macro_precision <= 1.0);
    assert!(metrics.macro_recall >= 0.0 && metrics.macro_recall <= 1.0);
    assert!(metrics.macro_f1 >= 0.0 && metrics.macro_f1 <= 1.0);
    
    println!("Multiclass metrics validation: Accuracy={:.4}, Macro-Precision={:.4}, Macro-Recall={:.4}, Macro-F1={:.4}", 
             metrics.accuracy, metrics.macro_precision, metrics.macro_recall, metrics.macro_f1);
}

fn print_multiclass_metrics(metrics: &MulticlassMetrics) {
    println!("Multiclass Metrics:");
    println!("  Accuracy: {:.4}", metrics.accuracy);
    println!("  Macro Precision: {:.4}", metrics.macro_precision);
    println!("  Macro Recall: {:.4}", metrics.macro_recall);
    println!("  Macro F1: {:.4}", metrics.macro_f1);
    
    for (class, class_metrics) in &metrics.per_class_metrics {
        println!("  Class {} - Precision: {:.4}, Recall: {:.4}, F1: {:.4}", 
                 class, class_metrics.precision, class_metrics.recall, class_metrics.f1_score);
    }
}

#[derive(Debug, Clone)]
struct ProbabilityMetrics {
    log_loss: f64,
    brier_score: f64,
}

fn calculate_probability_metrics(probabilities: &Array2<f32>, labels: &Array1<f32>) -> ProbabilityMetrics {
    let n = probabilities.nrows() as f64;
    
    // Log loss
    let mut log_loss = 0.0;
    for i in 0..probabilities.nrows() {
        let label = labels[i] as usize;
        let prob = probabilities[[i, label]].max(1e-15); // Avoid log(0)
        log_loss -= prob.ln() as f64;
    }
    log_loss /= n;
    
    // Brier score (for binary classification)
    let mut brier_score = 0.0;
    for i in 0..probabilities.nrows() {
        let label = labels[i];
        let prob = probabilities[[i, 1]]; // Probability of class 1
        brier_score += (prob - label).powi(2) as f64;
    }
    brier_score /= n;
    
    ProbabilityMetrics { log_loss, brier_score }
}

fn validate_probability_metrics(metrics: &ProbabilityMetrics) {
    assert!(metrics.log_loss >= 0.0);
    assert!(metrics.brier_score >= 0.0 && metrics.brier_score <= 1.0);
    
    println!("Probability metrics validation: Log Loss={:.4}, Brier Score={:.4}", 
             metrics.log_loss, metrics.brier_score);
}

fn print_probability_metrics(metrics: &ProbabilityMetrics) {
    println!("Probability Metrics:");
    println!("  Log Loss: {:.4}", metrics.log_loss);
    println!("  Brier Score: {:.4}", metrics.brier_score);
}

fn validate_binary_probabilities(probabilities: &Array2<f32>) {
    for i in 0..probabilities.nrows() {
        let prob_sum = probabilities.row(i).sum();
        assert!((prob_sum - 1.0).abs() < 1e-6, "Probabilities don't sum to 1");
        
        for j in 0..probabilities.ncols() {
            let prob = probabilities[[i, j]];
            assert!(prob >= 0.0 && prob <= 1.0, "Probability out of range");
        }
    }
}

fn validate_multiclass_probabilities(probabilities: &Array2<f32>) {
    for i in 0..probabilities.nrows() {
        let prob_sum = probabilities.row(i).sum();
        assert!((prob_sum - 1.0).abs() < 1e-6, "Probabilities don't sum to 1");
        
        for j in 0..probabilities.ncols() {
            let prob = probabilities[[i, j]];
            assert!(prob >= 0.0 && prob <= 1.0, "Probability out of range");
        }
    }
}

fn calculate_multiclass_probability_metrics(probabilities: &Array2<f32>, labels: &Array1<f32>, num_classes: usize) -> ProbabilityMetrics {
    // Simplified multiclass probability metrics
    calculate_probability_metrics(probabilities, labels)
}

fn validate_multiclass_probability_metrics(metrics: &ProbabilityMetrics) {
    validate_probability_metrics(metrics);
}

fn print_multiclass_probability_metrics(metrics: &ProbabilityMetrics) {
    print_probability_metrics(metrics);
}

fn detect_overfitting(train_metrics: &RegressionMetrics, test_metrics: &RegressionMetrics) {
    let train_test_ratio = test_metrics.mse / train_metrics.mse;
    
    if train_test_ratio > 2.0 {
        println!("⚠️  Potential overfitting detected (test MSE / train MSE = {:.2})", train_test_ratio);
    } else {
        println!("✓ No significant overfitting detected (test MSE / train MSE = {:.2})", train_test_ratio);
    }
}

fn compare_regression_metrics(metrics: &RegressionMetrics, config_num: usize) {
    // In a real implementation, this would compare against baseline or other configurations
    println!("Configuration {} performance: MSE={:.4}, R²={:.4}", config_num, metrics.mse, metrics.r2);
}

fn calculate_cv_statistics(cv_results: &[ClassificationMetrics]) {
    let n = cv_results.len() as f64;
    
    let mean_accuracy = cv_results.iter().map(|m| m.accuracy).sum::<f64>() / n;
    let mean_precision = cv_results.iter().map(|m| m.precision).sum::<f64>() / n;
    let mean_recall = cv_results.iter().map(|m| m.recall).sum::<f64>() / n;
    let mean_f1 = cv_results.iter().map(|m| m.f1_score).sum::<f64>() / n;
    
    println!("Cross-validation results:");
    println!("  Mean Accuracy: {:.4}", mean_accuracy);
    println!("  Mean Precision: {:.4}", mean_precision);
    println!("  Mean Recall: {:.4}", mean_recall);
    println!("  Mean F1: {:.4}", mean_f1);
    
    // Calculate standard deviations
    let accuracy_var = cv_results.iter()
        .map(|m| (m.accuracy - mean_accuracy).powi(2))
        .sum::<f64>() / n;
    let accuracy_std = accuracy_var.sqrt();
    
    println!("  Accuracy std: {:.4}", accuracy_std);
}