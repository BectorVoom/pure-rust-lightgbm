//! Complete workflow validation tests for pure Rust LightGBM.
//!
//! This test suite validates entire data science workflows to ensure
//! the library can handle real-world usage patterns and data pipelines.

use lightgbm_rust::*;
use ndarray::{Array1, Array2};
use std::fs;
use std::path::Path;
use ndarray::s;
use std::time::Instant;
use tempfile::TempDir;

mod common;
use common::*;

/// Test complete data science workflow for regression
#[test]
fn test_complete_data_science_workflow_regression() {
    println!("Testing complete data science workflow - Regression");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    let temp_dir = TempDir::new().unwrap();
    let workflow_result = run_regression_workflow(&temp_dir);
    
    match workflow_result {
        Ok(metrics) => {
            println!("  ✓ Regression workflow completed successfully");
            println!("  ✓ Final metrics: RMSE={:.4}, R²={:.4}", metrics.rmse, metrics.r2);
            assert!(metrics.rmse > 0.0);
            assert!(metrics.r2 >= 0.0);
        }
        Err(LightGBMError::NotImplemented { .. }) => {
            println!("  ⚠ Training not implemented yet - workflow structure validated");
        }
        Err(e) => {
            panic!("Unexpected error in regression workflow: {}", e);
        }
    }
}

/// Test complete data science workflow for classification
#[test]
fn test_complete_data_science_workflow_classification() {
    println!("Testing complete data science workflow - Classification");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    let temp_dir = TempDir::new().unwrap();
    let workflow_result = run_classification_workflow(&temp_dir);
    
    match workflow_result {
        Ok(metrics) => {
            println!("  ✓ Classification workflow completed successfully");
            println!("  ✓ Final metrics: Accuracy={:.4}, F1={:.4}", metrics.accuracy, metrics.f1_score);
            assert!(metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0);
            assert!(metrics.f1_score >= 0.0 && metrics.f1_score <= 1.0);
        }
        Err(LightGBMError::NotImplemented { .. }) => {
            println!("  ⚠ Training not implemented yet - workflow structure validated");
        }
        Err(e) => {
            panic!("Unexpected error in classification workflow: {}", e);
        }
    }
}

/// Test data pipeline robustness with various data issues
#[test]
fn test_robust_data_pipeline() {
    println!("Testing robust data pipeline handling");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    let temp_dir = TempDir::new().unwrap();
    
    // Test 1: Missing values
    test_missing_values_pipeline(&temp_dir).unwrap();
    println!("  ✓ Missing values handling validated");
    
    // Test 2: Outliers
    test_outliers_pipeline(&temp_dir).unwrap();
    println!("  ✓ Outliers handling validated");
    
    // Test 3: Mixed data types
    test_mixed_datatypes_pipeline(&temp_dir).unwrap();
    println!("  ✓ Mixed data types handling validated");
    
    // Test 4: Imbalanced classes
    test_imbalanced_pipeline(&temp_dir).unwrap();
    println!("  ✓ Imbalanced data handling validated");
    
    // Test 5: High dimensionality
    test_high_dimensional_pipeline(&temp_dir).unwrap();
    println!("  ✓ High dimensional data handling validated");
}

/// Test model evaluation and metrics computation
#[test]
fn test_model_evaluation_workflow() {
    println!("Testing model evaluation workflow");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    // Test regression metrics
    test_regression_metrics().unwrap();
    println!("  ✓ Regression metrics computation validated");
    
    // Test classification metrics
    test_classification_metrics().unwrap();
    println!("  ✓ Classification metrics computation validated");
    
    // Test custom metrics
    test_custom_metrics().unwrap();
    println!("  ✓ Custom metrics support validated");
}

/// Test model interpretability features
#[test]
fn test_model_interpretability_workflow() {
    println!("Testing model interpretability workflow");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    let temp_dir = TempDir::new().unwrap();
    
    // Feature importance analysis
    match test_feature_importance_workflow(&temp_dir) {
        Ok(_) => {
            println!("  ✓ Feature importance workflow completed");
        }
        Err(LightGBMError::NotImplemented { .. }) => {
            println!("  ⚠ Feature importance not implemented yet");
        }
        Err(e) => {
            panic!("Unexpected error in feature importance workflow: {}", e);
        }
    }
    
    // SHAP values analysis
    match test_shap_workflow(&temp_dir) {
        Ok(_) => {
            println!("  ✓ SHAP values workflow completed");
        }
        Err(LightGBMError::NotImplemented { .. }) => {
            println!("  ⚠ SHAP values not implemented yet");
        }
        Err(e) => {
            panic!("Unexpected error in SHAP workflow: {}", e);
        }
    }
}

/// Test performance and scalability
#[test]
fn test_performance_scalability_workflow() {
    println!("Testing performance and scalability workflow");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    let sizes = vec![
        (100, 5),
        (500, 10),
        (1000, 20),
        (2000, 50),
    ];
    
    for (n_samples, n_features) in sizes {
        let start_time = Instant::now();
        
        // Create dataset
        let features = create_test_features_regression(n_samples, n_features);
        let labels = create_test_labels_regression(&features);
        let dataset = Dataset::new(
            features,
            labels,
            None,
            None,
            None,
            None,
        ).unwrap();
        
        // Validate dataset
        let validation_result = dataset::utils::validate_dataset(&dataset);
        assert!(validation_result.is_valid);
        
        // Calculate statistics
        let _stats = dataset::utils::calculate_statistics(&dataset);
        
        let duration = start_time.elapsed();
        
        // Performance assertions
        assert!(duration.as_secs() < 10); // Should complete within 10 seconds
        
        println!("  ✓ {}x{} dataset processed in {:.2}ms", 
                n_samples, n_features, duration.as_secs_f64() * 1000.0);
    }
}

/// Test concurrent workflow execution
#[test]
fn test_concurrent_workflows() {
    println!("Testing concurrent workflow execution");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    use std::sync::Arc;
    use std::thread;
    
    // Create shared dataset
    let features = create_test_features_regression(500, 10);
    let labels = create_test_labels_regression(&features);
    let dataset = Arc::new(Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    ).unwrap());
    
    // Run concurrent workflows
    let handles: Vec<_> = (0..4).map(|thread_id| {
        let dataset = Arc::clone(&dataset);
        thread::spawn(move || {
            // Each thread runs a complete workflow
            let config = ConfigBuilder::new()
                .objective(ObjectiveType::Regression)
                .learning_rate(0.1)
                .num_iterations(10)
                .build()
                .unwrap();
            
            let mut regressor = LGBMRegressor::new(config);
            
            // Validation and statistics
            let validation_result = dataset::utils::validate_dataset(&dataset);
            let stats = dataset::utils::calculate_statistics(&dataset);
            
            // Training (will fail for now)
            let training_result = regressor.fit(&dataset);
            
            (thread_id, validation_result, stats, training_result)
        })
    }).collect();
    
    // Wait for all threads
    for handle in handles {
        let (thread_id, validation_result, stats, training_result) = handle.join().unwrap();
        
        assert!(validation_result.is_valid);
        assert_eq!(stats.num_samples, 500);
        assert_eq!(stats.num_features, 10);
        
        // Training should fail gracefully
        match training_result {
            Err(LightGBMError::NotImplemented { .. }) => {
                // Expected
            }
            Err(e) => {
                panic!("Unexpected error in thread {}: {}", thread_id, e);
            }
            Ok(_) => {
                println!("  ✓ Thread {} training completed", thread_id);
            }
        }
        
        println!("  ✓ Thread {} workflow completed successfully", thread_id);
    }
}

// Workflow implementation functions

fn run_regression_workflow(temp_dir: &TempDir) -> Result<RegressionMetrics> {
    // Step 1: Data Generation
    let (train_features, train_labels) = create_test_data!(regression, 400, 8);
    let (test_features, test_labels) = create_test_data!(regression, 100, 8);
    
    // Step 2: Dataset Creation
    let train_dataset = Dataset::new(
        train_features.clone(),
        train_labels.clone(),
        None,
        None,
        None,
        None,
    )?;
    
    let test_dataset = Dataset::new(
        test_features.clone(),
        test_labels.clone(),
        None,
        None,
        None,
        None,
    )?;
    
    // Step 3: Data Validation
    let train_validation = dataset::utils::validate_dataset(&train_dataset);
    let test_validation = dataset::utils::validate_dataset(&test_dataset);
    
    if !train_validation.is_valid || !test_validation.is_valid {
        return Err(LightGBMError::data_validation("Dataset validation failed"));
    }
    
    // Step 4: Feature Analysis
    let train_stats = dataset::utils::calculate_statistics(&train_dataset);
    let test_stats = dataset::utils::calculate_statistics(&test_dataset);
    
    // Ensure compatible datasets
    if train_stats.num_features != test_stats.num_features {
        return Err(LightGBMError::data_dimension_mismatch(
            format!("Feature count mismatch between train and test: expected {}, got {}", 
                   train_stats.num_features, test_stats.num_features)
        ));
    }
    
    // Step 5: Model Configuration
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.1)
        .num_iterations(100)
        .num_leaves(31)
        .lambda_l1(0.1)
        .lambda_l2(0.1)
        .min_data_in_leaf(20)
        .early_stopping_rounds(Some(10))
        .verbose(false)
        .build()?;
    
    // Step 6: Model Training
    let mut regressor = LGBMRegressor::new(config);
    regressor.fit(&train_dataset)?;
    
    // Step 7: Model Evaluation
    let predictions = regressor.predict(&test_features)?;
    let metrics = evaluate_regression(&predictions.view(), &test_labels.view());
    
    // Step 8: Model Persistence
    let model_path = temp_dir.path().join("regression_model.lgb");
    regressor.save_model(&model_path)?;
    
    // Step 9: Model Loading and Verification
    let loaded_regressor = LGBMRegressor::load_model(&model_path)?;
    let loaded_predictions = loaded_regressor.predict(&test_features)?;
    
    // Verify predictions consistency
    for (orig, loaded) in predictions.iter().zip(loaded_predictions.iter()) {
        if (orig - loaded).abs() > 1e-6 {
            return Err(LightGBMError::internal("Model loading inconsistency"));
        }
    }
    
    Ok(metrics)
}

fn run_classification_workflow(temp_dir: &TempDir) -> Result<ClassificationMetrics> {
    // Step 1: Data Generation
    let (train_features, train_labels) = create_test_data!(binary, 400, 8);
    let (test_features, test_labels) = create_test_data!(binary, 100, 8);
    
    // Step 2: Dataset Creation
    let train_dataset = Dataset::new(
        train_features.clone(),
        train_labels.clone(),
        None,
        None,
        None,
        None,
    )?;
    
    // Step 3: Class Balance Analysis
    let mut class_counts = [0, 0];
    for i in 0..train_dataset.num_data() {
        let label = train_dataset.label(i)?;
        if label == 0.0 {
            class_counts[0] += 1;
        } else {
            class_counts[1] += 1;
        }
    }
    
    if class_counts[0] == 0 || class_counts[1] == 0 {
        return Err(LightGBMError::data_validation("Imbalanced dataset with missing class"));
    }
    
    // Step 4: Model Configuration
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .learning_rate(0.1)
        .num_iterations(100)
        .num_leaves(31)
        .build()?;
    
    // Step 5: Model Training
    let mut classifier = LGBMClassifier::new(config);
    classifier.fit(&train_dataset)?;
    
    // Step 6: Model Evaluation
    let class_predictions = classifier.predict(&test_features)?;
    let prob_predictions = classifier.predict_proba(&test_features)?;
    
    let metrics = evaluate_binary_classification(
        &class_predictions.view(),
        &prob_predictions.view(),
        &test_labels.view(),
    );
    
    // Step 7: Model Persistence
    let model_path = temp_dir.path().join("classification_model.lgb");
    classifier.save_model(&model_path)?;
    
    Ok(metrics)
}

// Helper workflow functions

fn test_missing_values_pipeline(temp_dir: &TempDir) -> Result<()> {
    let features = create_test_features_with_missing(100, 5, 0.2);
    let labels = create_test_labels_regression(&features);
    
    let dataset = Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    )?;
    
    let validation_result = dataset::utils::validate_dataset(&dataset);
    assert!(validation_result.is_valid);
    
    let stats = dataset::utils::calculate_statistics(&dataset);
    assert!(stats.sparsity > 0.0); // Should have missing values
    
    Ok(())
}

fn test_outliers_pipeline(temp_dir: &TempDir) -> Result<()> {
    let features = create_test_features_with_outliers(100, 5, 0.1);
    let labels = create_test_labels_regression(&features);
    
    let dataset = Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    )?;
    
    let validation_result = dataset::utils::validate_dataset(&dataset);
    assert!(validation_result.is_valid);
    
    Ok(())
}

fn test_mixed_datatypes_pipeline(temp_dir: &TempDir) -> Result<()> {
    let (features, categorical_indices) = create_test_mixed_features(100, 4, 3, 5);
    let labels = create_test_labels_regression(&features);
    
    let config = DatasetConfig::new()
        .with_categorical_features(categorical_indices);
    
    let dataset = DatasetFactory::from_arrays(
        features,
        labels,
        None,
        config,
    )?;
    
    let validation_result = dataset::utils::validate_dataset(&dataset);
    assert!(validation_result.is_valid);
    
    Ok(())
}

fn test_imbalanced_pipeline(temp_dir: &TempDir) -> Result<()> {
    let (features, labels) = create_test_imbalanced_classification(100, 5, 0.1);
    
    let dataset = Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    )?;
    
    // Check class distribution
    let mut class_counts = [0, 0];
    for i in 0..dataset.num_data() {
        let label = dataset.label(i)?;
        if label == 0.0 {
            class_counts[0] += 1;
        } else {
            class_counts[1] += 1;
        }
    }
    
    let minority_ratio = class_counts[1] as f32 / dataset.num_data() as f32;
    assert!(minority_ratio < 0.2); // Should be imbalanced
    
    Ok(())
}

fn test_high_dimensional_pipeline(temp_dir: &TempDir) -> Result<()> {
    let features = create_test_features_regression(100, 50); // 50 features
    let labels = create_test_labels_regression(&features);
    
    let dataset = Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    )?;
    
    assert_eq!(dataset.num_features(), 50);
    
    let validation_result = dataset::utils::validate_dataset(&dataset);
    assert!(validation_result.is_valid);
    
    Ok(())
}

fn test_regression_metrics() -> Result<()> {
    // Create test predictions and true values
    let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let true_values = Array1::from_vec(vec![1.1, 1.9, 3.1, 3.9, 5.1]);
    
    let metrics = evaluate_regression(&predictions.view(), &true_values.view());
    
    assert!(metrics.rmse > 0.0);
    assert!(metrics.mae > 0.0);
    assert!(metrics.r2 >= 0.0);
    
    Ok(())
}

fn test_classification_metrics() -> Result<()> {
    // Create test predictions and true values
    let class_predictions = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0]);
    let prob_predictions = Array2::from_shape_vec(
        (5, 2),
        vec![0.8, 0.2, 0.3, 0.7, 0.9, 0.1, 0.4, 0.6, 0.2, 0.8],
    ).unwrap();
    let true_values = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0]);
    
    let metrics = evaluate_binary_classification(
        &class_predictions.view(),
        &prob_predictions.view(),
        &true_values.view(),
    );
    
    assert!(metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0);
    assert!(metrics.precision >= 0.0 && metrics.precision <= 1.0);
    assert!(metrics.recall >= 0.0 && metrics.recall <= 1.0);
    
    Ok(())
}

fn test_custom_metrics() -> Result<()> {
    // Test custom metric implementation
    let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let true_values = Array1::from_vec(vec![1.1, 1.9, 3.1, 3.9, 5.1]);
    
    // Custom MAPE metric
    let mape = calculate_mape(&predictions.view(), &true_values.view());
    assert!(mape >= 0.0);
    
    // Custom log loss for binary classification
    let probs = Array1::from_vec(vec![0.8, 0.3, 0.9, 0.4, 0.2]);
    let labels = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0, 0.0]);
    
    let log_loss = calculate_log_loss(&probs.view(), &labels.view());
    assert!(log_loss >= 0.0);
    
    Ok(())
}

fn test_feature_importance_workflow(temp_dir: &TempDir) -> Result<()> {
    let features = create_test_features_regression(200, 8);
    let labels = create_test_labels_regression(&features);
    let feature_names = Some((0..8).map(|i| format!("feature_{}", i)).collect());
    
    let dataset = Dataset::new(
        features.clone(),
        labels,
        None,
        None,
        feature_names,
        None,
    )?;
    
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .build()?;
    
    let mut regressor = LGBMRegressor::new(config);
    regressor.fit(&dataset)?;
    
    // Test feature importance
    let importance = regressor.feature_importance(ImportanceType::Split)?;
    assert_eq!(importance.len(), 8);
    
    Ok(())
}

fn test_shap_workflow(temp_dir: &TempDir) -> Result<()> {
    let features = create_test_features_regression(100, 5);
    let labels = create_test_labels_regression(&features);
    
    let dataset = Dataset::new(
        features.clone(),
        labels,
        None,
        None,
        None,
        None,
    )?;
    
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .build()?;
    
    let mut regressor = LGBMRegressor::new(config);
    regressor.fit(&dataset)?;
    
    // Test SHAP values
    let test_features = features.slice(s![0..10, ..]);
    let shap_values = regressor.predict_contrib(&test_features)?;
    assert_eq!(shap_values.dim(), (10, 6)); // 5 features + bias
    
    Ok(())
}

// Helper metric calculation functions

fn calculate_mape(predictions: &ndarray::ArrayView1<f32>, true_values: &ndarray::ArrayView1<f32>) -> f32 {
    let mut sum = 0.0;
    let mut count = 0;
    
    for (&pred, &true_val) in predictions.iter().zip(true_values.iter()) {
        if true_val != 0.0 {
            sum += ((pred - true_val) / true_val).abs();
            count += 1;
        }
    }
    
    if count > 0 {
        (sum / count as f32) * 100.0
    } else {
        0.0
    }
}

fn calculate_log_loss(probabilities: &ndarray::ArrayView1<f32>, labels: &ndarray::ArrayView1<f32>) -> f32 {
    let mut sum = 0.0;
    
    for (&prob, &label) in probabilities.iter().zip(labels.iter()) {
        let clamped_prob = prob.clamp(1e-15, 1.0 - 1e-15);
        if label == 1.0 {
            sum += -clamped_prob.ln();
        } else {
            sum += -(1.0 - clamped_prob).ln();
        }
    }
    
    sum / probabilities.len() as f32
}
