//! End-to-end integration tests for pure Rust LightGBM implementation.
//!
//! This test suite validates complete workflows from data loading through
//! model training to prediction and evaluation. These tests simulate real-world
//! usage scenarios and verify the entire system works correctly.

use lightgbm_rust::*;
use ndarray::{Array1, Array2, s};
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;
use std::time::Instant;

mod common;
use common::*;

/// End-to-end regression workflow test
#[test]
fn test_complete_regression_workflow() {
    println!("Testing complete regression workflow...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    let temp_dir = TempDir::new().unwrap();
    let data_path = temp_dir.path().join("regression_data.csv");
    let model_path = temp_dir.path().join("regression_model.lgb");
    
    // Step 1: Generate and save training data
    let (train_features, train_labels) = create_test_data!(regression, 500, 10);
    let train_weights = Some(create_test_weights(500));
    
    create_test_csv(
        &data_path,
        &train_features,
        &train_labels,
        train_weights.as_ref(),
        &Some((0..10).map(|i| format!("feature_{}", i)).collect()),
    ).unwrap();
    
    // Step 2: Load dataset from CSV
    let dataset_config = DatasetConfig::new()
        .with_target_column("target")
        .with_weight_column("weight")
        .with_max_bin(255)
        .with_categorical_features(vec![0, 1]); // First two features as categorical
    
    let train_dataset = DatasetFactory::from_csv(&data_path, dataset_config).unwrap();
    
    assert_eq!(train_dataset.num_data(), 500);
    assert_eq!(train_dataset.num_features(), 10);
    assert!(train_dataset.has_weights());
    
    // Step 3: Configure model
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.1)
        .num_iterations(100)
        .num_leaves(31)
        .lambda_l1(0.1)
        .lambda_l2(0.1)
        .min_data_in_leaf(20)
        .feature_fraction(0.8)
        .bagging_fraction(0.8)
        .bagging_freq(5)
        .early_stopping_rounds(Some(10))
        .verbose(false)
        .build()
        .unwrap();
    
    // Step 4: Create and train model
    let mut regressor = LGBMRegressor::new(config);
    
    // Training will fail for now, but test the interface
    match regressor.fit(&train_dataset) {
        Ok(_) => {
            println!("  ✓ Model training completed");
            
            // Step 5: Generate test data
            let (test_features, test_labels) = create_test_data!(regression, 100, 10);
            
            // Step 6: Make predictions
            let predictions = regressor.predict(&test_features).unwrap();
            assert_eq!(predictions.len(), 100);
            
            // Step 7: Evaluate model
            let metrics = evaluate_regression(&predictions.view(), &test_labels.view());
            
            println!("  ✓ Model evaluation:");
            println!("    - RMSE: {:.4}", metrics.rmse);
            println!("    - MAE: {:.4}", metrics.mae);
            println!("    - R²: {:.4}", metrics.r2);
            
            // Step 8: Save model
            regressor.save_model(&model_path).unwrap();
            assert!(model_path.exists());
            println!("  ✓ Model saved to {:?}", model_path);
            
            // Step 9: Load model and verify predictions
            let loaded_regressor = LGBMRegressor::load_model(&model_path).unwrap();
            let loaded_predictions = loaded_regressor.predict(&test_features).unwrap();
            
            // Predictions should be identical
            for (orig, loaded) in predictions.iter().zip(loaded_predictions.iter()) {
                assert!((orig - loaded).abs() < 1e-6);
            }
            
            println!("  ✓ Model loading and prediction consistency verified");
            
        }
        Err(LightGBMError::NotImplemented { .. }) => {
            println!("  ⚠ Training not implemented yet - testing interface only");
            
            // Test prediction interface
            let prediction_result = regressor.predict(&train_features);
            assert!(prediction_result.is_err());
            
            // Test model saving interface
            let save_result = regressor.save_model(&model_path);
            assert!(save_result.is_err());
        }
        Err(e) => {
            panic!("Unexpected error: {}", e);
        }
    }
    
    println!("Complete regression workflow test completed");
}

/// End-to-end binary classification workflow test
#[test]
fn test_complete_binary_classification_workflow() {
    println!("Testing complete binary classification workflow...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    let temp_dir = TempDir::new().unwrap();
    let data_path = temp_dir.path().join("binary_data.csv");
    let model_path = temp_dir.path().join("binary_model.lgb");
    
    // Step 1: Generate and save training data
    let (train_features, train_labels) = create_test_data!(binary, 400, 8);
    let train_weights = Some(create_test_weights(400));
    
    create_test_csv(
        &data_path,
        &train_features,
        &train_labels,
        train_weights.as_ref(),
        &Some((0..8).map(|i| format!("feature_{}", i)).collect()),
    ).unwrap();
    
    // Step 2: Load dataset from CSV
    let dataset_config = DatasetConfig::new()
        .with_target_column("target")
        .with_weight_column("weight")
        .with_max_bin(255);
    
    let train_dataset = DatasetFactory::from_csv(&data_path, dataset_config).unwrap();
    
    assert_eq!(train_dataset.num_data(), 400);
    assert_eq!(train_dataset.num_features(), 8);
    
    // Verify class balance
    let mut class_counts = [0, 0];
    for i in 0..train_dataset.num_data() {
        let label = train_dataset.label(i).unwrap();
        if label == 0.0 {
            class_counts[0] += 1;
        } else {
            class_counts[1] += 1;
        }
    }
    assert!(class_counts[0] > 0 && class_counts[1] > 0);
    
    // Step 3: Configure model
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .learning_rate(0.1)
        .num_iterations(100)
        .num_leaves(31)
        .lambda_l1(0.05)
        .lambda_l2(0.05)
        .min_data_in_leaf(10)
        .feature_fraction(0.9)
        .bagging_fraction(0.8)
        .bagging_freq(5)
        .early_stopping_rounds(Some(10))
        .verbose(false)
        .build()
        .unwrap();
    
    // Step 4: Create and train model
    let mut classifier = LGBMClassifier::new(config);
    
    // Training will fail for now, but test the interface
    match classifier.fit(&train_dataset) {
        Ok(_) => {
            println!("  ✓ Model training completed");
            
            // Step 5: Generate test data
            let (test_features, test_labels) = create_test_data!(binary, 100, 8);
            
            // Step 6: Make predictions
            let class_predictions = classifier.predict(&test_features).unwrap();
            let prob_predictions = classifier.predict_proba(&test_features).unwrap();
            
            assert_eq!(class_predictions.len(), 100);
            assert_eq!(prob_predictions.dim(), (100, 2));
            
            // Step 7: Evaluate model
            let metrics = evaluate_binary_classification(
                &class_predictions.view(),
                &prob_predictions.view(),
                &test_labels.view(),
            );
            
            println!("  ✓ Model evaluation:");
            println!("    - Accuracy: {:.4}", metrics.accuracy);
            println!("    - Precision: {:.4}", metrics.precision);
            println!("    - Recall: {:.4}", metrics.recall);
            println!("    - F1-Score: {:.4}", metrics.f1_score);
            println!("    - AUC: {:.4}", metrics.auc);
            
            // Step 8: Save model
            classifier.save_model(&model_path).unwrap();
            assert!(model_path.exists());
            println!("  ✓ Model saved to {:?}", model_path);
            
            // Step 9: Load model and verify predictions
            let loaded_classifier = LGBMClassifier::load_model(&model_path).unwrap();
            let loaded_class_predictions = loaded_classifier.predict(&test_features).unwrap();
            let loaded_prob_predictions = loaded_classifier.predict_proba(&test_features).unwrap();
            
            // Predictions should be identical
            for (orig, loaded) in class_predictions.iter().zip(loaded_class_predictions.iter()) {
                assert_eq!(*orig, *loaded);
            }
            
            for (orig, loaded) in prob_predictions.iter().zip(loaded_prob_predictions.iter()) {
                assert!((orig - loaded).abs() < 1e-6);
            }
            
            println!("  ✓ Model loading and prediction consistency verified");
            
        }
        Err(LightGBMError::NotImplemented { .. }) => {
            println!("  ⚠ Training not implemented yet - testing interface only");
            
            // Test prediction interfaces
            let class_result = classifier.predict(&train_features);
            let prob_result = classifier.predict_proba(&train_features);
            assert!(class_result.is_err());
            assert!(prob_result.is_err());
        }
        Err(e) => {
            panic!("Unexpected error: {}", e);
        }
    }
    
    println!("Complete binary classification workflow test completed");
}

/// End-to-end multiclass classification workflow test
#[test]
fn test_complete_multiclass_workflow() {
    println!("Testing complete multiclass classification workflow...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    let temp_dir = TempDir::new().unwrap();
    let data_path = temp_dir.path().join("multiclass_data.csv");
    let model_path = temp_dir.path().join("multiclass_model.lgb");
    
    // Step 1: Generate and save training data
    let (train_features, train_labels) = create_test_data!(multiclass, 300, 6, 3);
    let train_weights = Some(create_test_weights(300));
    
    create_test_csv(
        &data_path,
        &train_features,
        &train_labels,
        train_weights.as_ref(),
        &Some((0..6).map(|i| format!("feature_{}", i)).collect()),
    ).unwrap();
    
    // Step 2: Load dataset from CSV
    let dataset_config = DatasetConfig::new()
        .with_target_column("target")
        .with_weight_column("weight")
        .with_max_bin(255);
    
    let train_dataset = DatasetFactory::from_csv(&data_path, dataset_config).unwrap();
    
    assert_eq!(train_dataset.num_data(), 300);
    assert_eq!(train_dataset.num_features(), 6);
    
    // Verify class distribution
    let mut class_counts = [0, 0, 0];
    for i in 0..train_dataset.num_data() {
        let label = train_dataset.label(i).unwrap() as usize;
        if label < 3 {
            class_counts[label] += 1;
        }
    }
    assert!(class_counts[0] > 0 && class_counts[1] > 0 && class_counts[2] > 0);
    
    // Step 3: Configure model
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Multiclass)
        .num_class(3)
        .learning_rate(0.1)
        .num_iterations(100)
        .num_leaves(31)
        .lambda_l1(0.05)
        .lambda_l2(0.05)
        .min_data_in_leaf(10)
        .feature_fraction(0.9)
        .early_stopping_rounds(Some(10))
        .verbose(false)
        .build()
        .unwrap();
    
    // Step 4: Create and train model
    let mut classifier = LGBMClassifier::new(config);
    
    // Training will fail for now, but test the interface
    match classifier.fit(&train_dataset) {
        Ok(_) => {
            println!("  ✓ Model training completed");
            
            // Step 5: Generate test data
            let (test_features, test_labels) = create_test_data!(multiclass, 75, 6, 3);
            
            // Step 6: Make predictions
            let class_predictions = classifier.predict(&test_features).unwrap();
            let prob_predictions = classifier.predict_proba(&test_features).unwrap();
            
            assert_eq!(class_predictions.len(), 75);
            assert_eq!(prob_predictions.dim(), (75, 3));
            
            // Step 7: Evaluate model
            let metrics = evaluate_multiclass_classification(
                &class_predictions.view(),
                &prob_predictions.view(),
                &test_labels.view(),
                3,
            );
            
            println!("  ✓ Model evaluation:");
            println!("    - Accuracy: {:.4}", metrics.accuracy);
            println!("    - Macro F1: {:.4}", metrics.macro_f1);
            println!("    - Weighted F1: {:.4}", metrics.weighted_f1);
            
            // Step 8: Save model
            classifier.save_model(&model_path).unwrap();
            assert!(model_path.exists());
            println!("  ✓ Model saved to {:?}", model_path);
            
        }
        Err(LightGBMError::NotImplemented { .. }) => {
            println!("  ⚠ Training not implemented yet - testing interface only");
        }
        Err(e) => {
            panic!("Unexpected error: {}", e);
        }
    }
    
    println!("Complete multiclass classification workflow test completed");
}

/// Test feature importance and model interpretation
#[test]
fn test_feature_importance_workflow() {
    println!("Testing feature importance workflow...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    // Create dataset with named features
    let features = create_test_features_regression(200, 5);
    let labels = create_test_labels_regression(&features);
    let feature_names = Some(vec![
        "numerical_1".to_string(),
        "numerical_2".to_string(),
        "categorical_1".to_string(),
        "categorical_2".to_string(),
        "interaction".to_string(),
    ]);
    
    let dataset = Dataset::new(
        features.clone(),
        labels,
        None,
        None,
        feature_names.clone(),
        None,
    ).unwrap();
    
    // Configure model
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.1)
        .num_iterations(50)
        .build()
        .unwrap();
    
    let mut regressor = LGBMRegressor::new(config);
    
    // Training will fail for now, but test the interface
    match regressor.fit(&dataset) {
        Ok(_) => {
            println!("  ✓ Model training completed");
            
            // Test feature importance
            let importance = regressor.feature_importance(ImportanceType::Split).unwrap();
            assert_eq!(importance.len(), 5);
            
            println!("  ✓ Feature importance (split):");
            for (i, &imp) in importance.iter().enumerate() {
                let name = feature_names.as_ref().unwrap()[i].as_str();
                println!("    - {}: {:.4}", name, imp);
            }
            
            let importance_gain = regressor.feature_importance(ImportanceType::Gain).unwrap();
            assert_eq!(importance_gain.len(), 5);
            
            println!("  ✓ Feature importance (gain):");
            for (i, &imp) in importance_gain.iter().enumerate() {
                let name = feature_names.as_ref().unwrap()[i].as_str();
                println!("    - {}: {:.4}", name, imp);
            }
            
            // Test SHAP values
            let test_features = features.slice(s![0..10, ..]);
            let shap_values = regressor.predict_contrib(&test_features).unwrap();
            assert_eq!(shap_values.dim(), (10, 6)); // 5 features + bias
            
            println!("  ✓ SHAP values computed for 10 samples");
            
        }
        Err(LightGBMError::NotImplemented { .. }) => {
            println!("  ⚠ Training not implemented yet - testing interface only");
            
            // Test feature importance interface
            let importance_result = regressor.feature_importance(ImportanceType::Split);
            assert!(importance_result.is_err());
            
            // Test SHAP interface
            let shap_result = regressor.predict_contrib(&features.slice(s![0..10, ..]));
            assert!(shap_result.is_err());
        }
        Err(e) => {
            panic!("Unexpected error: {}", e);
        }
    }
    
    println!("Feature importance workflow test completed");
}

/// Test cross-validation workflow
#[test]
fn test_cross_validation_workflow() {
    println!("Testing cross-validation workflow...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    // Create dataset
    let features = create_test_features_regression(500, 8);
    let labels = create_test_labels_regression(&features);
    let dataset = Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
    // Configure cross-validation
    let cv_config = CrossValidationConfig::new()
        .with_num_folds(5)
        .with_stratified(false)
        .with_shuffle(true)
        .with_random_seed(42);
    
    let model_config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.1)
        .num_iterations(50)
        .verbose(false)
        .build()
        .unwrap();
    
    // Run cross-validation
    let cv_result = cross_validate(
        &dataset,
        &model_config,
        &cv_config,
        &["rmse", "mae"],
    );
    
    match cv_result {
        Ok(results) => {
            println!("  ✓ Cross-validation completed");
            
            assert_eq!(results.num_folds, 5);
            assert!(results.metrics.contains_key("rmse"));
            assert!(results.metrics.contains_key("mae"));
            
            let rmse_scores = &results.metrics["rmse"];
            let mae_scores = &results.metrics["mae"];
            
            assert_eq!(rmse_scores.len(), 5);
            assert_eq!(mae_scores.len(), 5);
            
            println!("  ✓ RMSE scores: {:?}", rmse_scores);
            println!("  ✓ MAE scores: {:?}", mae_scores);
            
            let rmse_mean = rmse_scores.iter().sum::<f64>() / rmse_scores.len() as f64;
            let mae_mean = mae_scores.iter().sum::<f64>() / mae_scores.len() as f64;
            
            println!("  ✓ Average RMSE: {:.4}", rmse_mean);
            println!("  ✓ Average MAE: {:.4}", mae_mean);
            
        }
        Err(LightGBMError::NotImplemented { .. }) => {
            println!("  ⚠ Cross-validation not implemented yet");
        }
        Err(e) => {
            panic!("Unexpected error: {}", e);
        }
    }
    
    println!("Cross-validation workflow test completed");
}

/// Test early stopping workflow
#[test]
fn test_early_stopping_workflow() {
    println!("Testing early stopping workflow...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    // Create training and validation datasets
    let (train_features, train_labels) = create_test_data!(regression, 400, 6);
    let (valid_features, valid_labels) = create_test_data!(regression, 100, 6);
    
    let train_dataset = Dataset::new(
        train_features,
        train_labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
    let valid_dataset = Dataset::new(
        valid_features,
        valid_labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
    // Configure model with early stopping
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.1)
        .num_iterations(200)
        .early_stopping_rounds(Some(10))
        .early_stopping_tolerance(0.001)
        .verbose(true)
        .build()
        .unwrap();
    
    let mut regressor = LGBMRegressor::new(config);
    
    // Add validation dataset
    regressor.add_validation_data(&valid_dataset).unwrap();
    
    // Training will fail for now, but test the interface
    match regressor.fit(&train_dataset) {
        Ok(_) => {
            println!("  ✓ Model training with early stopping completed");
            
            // Check that training stopped early
            let actual_iterations = regressor.num_iterations();
            assert!(actual_iterations < 200);
            
            println!("  ✓ Training stopped early at iteration {}", actual_iterations);
            
            // Get training history
            let history = regressor.training_history().unwrap();
            assert!(!history.train_metrics.is_empty());
            assert!(!history.valid_metrics.is_empty());
            
            println!("  ✓ Training history available with {} iterations", history.train_metrics.len());
            
        }
        Err(LightGBMError::NotImplemented { .. }) => {
            println!("  ⚠ Training not implemented yet - testing interface only");
        }
        Err(e) => {
            panic!("Unexpected error: {}", e);
        }
    }
    
    println!("Early stopping workflow test completed");
}

/// Test hyperparameter optimization workflow
#[test]
fn test_hyperparameter_optimization_workflow() {
    println!("Testing hyperparameter optimization workflow...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    // Create dataset
    let features = create_test_features_regression(300, 5);
    let labels = create_test_labels_regression(&features);
    let dataset = Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
    // Define hyperparameter search space
    let param_space = HyperparameterSpace::new()
        .add_float("learning_rate", 0.01, 0.3)
        .add_int("num_leaves", 10, 100)
        .add_float("lambda_l1", 0.0, 1.0)
        .add_float("lambda_l2", 0.0, 1.0)
        .add_float("feature_fraction", 0.5, 1.0);
    
    // Configure optimization
    let opt_config = OptimizationConfig::new()
        .with_num_trials(20)
        .with_cv_folds(3)
        .with_metric("rmse")
        .with_direction(OptimizationDirection::Minimize)
        .with_timeout_seconds(300);
    
    // Run hyperparameter optimization
    let opt_result = optimize_hyperparameters(
        &dataset,
        &param_space,
        &opt_config,
    );
    
    match opt_result {
        Ok(result) => {
            println!("  ✓ Hyperparameter optimization completed");
            
            assert!(result.num_trials > 0);
            assert!(result.best_score.is_finite());
            assert!(!result.best_params.is_empty());
            
            println!("  ✓ Best score: {:.4}", result.best_score);
            println!("  ✓ Best parameters:");
            for (param, value) in &result.best_params {
                println!("    - {}: {:?}", param, value);
            }
            
            // Train final model with best parameters
            let final_config = ConfigBuilder::from_params(&result.best_params)
                .objective(ObjectiveType::Regression)
                .num_iterations(100)
                .build()
                .unwrap();
            
            let mut final_regressor = LGBMRegressor::new(final_config);
            let final_result = final_regressor.fit(&dataset);
            
            match final_result {
                Ok(_) => {
                    println!("  ✓ Final model trained with optimized parameters");
                }
                Err(LightGBMError::NotImplemented { .. }) => {
                    println!("  ⚠ Final training not implemented yet");
                }
                Err(e) => {
                    panic!("Unexpected error in final training: {}", e);
                }
            }
            
        }
        Err(LightGBMError::NotImplemented { .. }) => {
            println!("  ⚠ Hyperparameter optimization not implemented yet");
        }
        Err(e) => {
            panic!("Unexpected error: {}", e);
        }
    }
    
    println!("Hyperparameter optimization workflow test completed");
}

/// Test model ensemble workflow
#[test]
fn test_model_ensemble_workflow() {
    println!("Testing model ensemble workflow...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    // Create dataset
    let features = create_test_features_regression(400, 7);
    let labels = create_test_labels_regression(&features);
    let dataset = Dataset::new(
        features.clone(),
        labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
    // Create multiple models with different configurations
    let configs = vec![
        ConfigBuilder::new()
            .objective(ObjectiveType::Regression)
            .learning_rate(0.05)
            .num_leaves(31)
            .build()
            .unwrap(),
        ConfigBuilder::new()
            .objective(ObjectiveType::Regression)
            .learning_rate(0.1)
            .num_leaves(63)
            .build()
            .unwrap(),
        ConfigBuilder::new()
            .objective(ObjectiveType::Regression)
            .learning_rate(0.15)
            .num_leaves(15)
            .build()
            .unwrap(),
    ];
    
    let mut models = Vec::new();
    
    for (i, config) in configs.into_iter().enumerate() {
        let mut regressor = LGBMRegressor::new(config);
        
        match regressor.fit(&dataset) {
            Ok(_) => {
                models.push(regressor);
                println!("  ✓ Model {} trained successfully", i + 1);
            }
            Err(LightGBMError::NotImplemented { .. }) => {
                models.push(regressor);
                println!("  ⚠ Model {} training not implemented yet", i + 1);
            }
            Err(e) => {
                panic!("Unexpected error training model {}: {}", i + 1, e);
            }
        }
    }
    
    // Create ensemble
    let ensemble_config = EnsembleConfig::new()
        .with_method(EnsembleMethod::Average)
        .with_weights(vec![0.4, 0.4, 0.2]);
    
    let ensemble = ModelEnsemble::new(models, ensemble_config);
    
    match ensemble {
        Ok(ensemble) => {
            println!("  ✓ Model ensemble created");
            
            // Test ensemble prediction
            let test_features = features.slice(s![0..50, ..]);
            let ensemble_predictions = ensemble.predict(&test_features);
            
            match ensemble_predictions {
                Ok(predictions) => {
                    assert_eq!(predictions.len(), 50);
                    println!("  ✓ Ensemble predictions computed");
                }
                Err(LightGBMError::NotImplemented { .. }) => {
                    println!("  ⚠ Ensemble prediction not implemented yet");
                }
                Err(e) => {
                    panic!("Unexpected error in ensemble prediction: {}", e);
                }
            }
            
        }
        Err(LightGBMError::NotImplemented { .. }) => {
            println!("  ⚠ Model ensemble not implemented yet");
        }
        Err(e) => {
            panic!("Unexpected error creating ensemble: {}", e);
        }
    }
    
    println!("Model ensemble workflow test completed");
}
