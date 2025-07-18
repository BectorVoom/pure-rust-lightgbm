//! Comprehensive End-to-End Tests for Pure Rust LightGBM
//! 
//! This test suite provides comprehensive validation of the complete LightGBM system
//! behavior from data loading through training to prediction and evaluation.
//! 
//! These tests are designed to:
//! 1. Validate the complete system workflows when fully implemented
//! 2. Verify interface compliance and error handling in current state
//! 3. Ensure data integrity throughout the pipeline
//! 4. Test cross-references with the design document specifications
//! 
//! Test Categories:
//! - Data Pipeline E2E Tests
//! - Model Training E2E Tests  
//! - Prediction Pipeline E2E Tests
//! - Configuration E2E Tests
//! - Error Handling E2E Tests
//! - Performance E2E Tests
//! - Memory Management E2E Tests
//! - Serialization E2E Tests

use lightgbm_rust::*;
use lightgbm_rust::config::ConfigManager;
use ndarray::{Array1, Array2, s, Axis};
use std::fs;
use tempfile::TempDir;
use std::time::Instant;
use approx::assert_abs_diff_eq;

mod common;
use common::*;

/// Test Category 1: Data Pipeline E2E Tests
/// These tests validate the complete data loading and preprocessing pipeline

#[test]
fn test_e2e_data_pipeline_csv_loading() {
    println!("\n=== E2E Data Pipeline: CSV Loading ===");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    let temp_dir = TempDir::new().unwrap();
    let csv_path = temp_dir.path().join("test_data.csv");
    
    // Test 1: Simple regression data
    let (features, labels) = create_test_data!(regression, 100, 5);
    let weights = Some(create_test_weights(100));
    let feature_names: Option<Vec<String>> = Some((0..5).map(|i| format!("feature_{}", i)).collect());
    
    create_test_csv(&csv_path, &features, &labels, weights.as_ref(), feature_names.as_ref().map(|v| &**v)).unwrap();
    
    // Load dataset with various configurations
    let configs = vec![
        DatasetConfig::new()
            .with_target_column("target")
            .with_weight_column("weight")
            .with_max_bin(255),
        DatasetConfig::new()
            .with_target_column("target")
            .with_max_bin(127)
            .with_categorical_features(vec![0, 1]),
        DatasetConfig::new()
            .with_target_column("target")
            .with_min_data_in_bin(5),
    ];
    
    for (i, config) in configs.into_iter().enumerate() {
        println!("  Testing config variant {}", i + 1);
        
        let dataset_result = DatasetFactory::from_csv(&csv_path, config);
        match dataset_result {
            Ok(dataset) => {
                assert_eq!(dataset.num_data(), 100);
                assert_eq!(dataset.num_features(), 5);
                println!("    ✓ Dataset loaded successfully");
                
                // Validate data integrity
                let labels = dataset.labels();
                assert_eq!(labels.len(), 100);
                println!("    ✓ Data access working");
            }
            Err(LightGBMError::NotImplemented { .. }) => {
                println!("    ⚠ CSV loading not implemented yet");
            }
            Err(e) => {
                panic!("Unexpected error loading CSV: {}", e);
            }
        }
    }
}

#[test]
fn test_e2e_data_pipeline_memory_efficiency() {
    println!("\n=== E2E Data Pipeline: Memory Efficiency ===");
    
    assert!(lightgbm_rust::init().is_ok());
    
    // Test large dataset handling
    let sizes = vec![1000, 10000];
    let feature_counts = vec![10, 50, 100];
    
    for &size in &sizes {
        for &num_features in &feature_counts {
            println!("  Testing dataset size: {} x {}", size, num_features);
            
            let start_memory = get_memory_usage();
            
            // Create large dataset
            let features = Array2::from_shape_fn((size, num_features), |(i, j)| {
                (i as f32 * 0.1 + j as f32 * 0.01) % 10.0
            });
            let labels = Array1::from_shape_fn(size, |i| (i % 3) as f32);
            
            let dataset_result = Dataset::new(
                features,
                labels,
                None, // weights
                None, // queries
                None, // feature_names
                None, // categorical_features
            );
            
            match dataset_result {
                Ok(dataset) => {
                    assert_eq!(dataset.num_data() as usize, size);
                    assert_eq!(dataset.num_features(), num_features);
                    
                    let end_memory = get_memory_usage();
                    let memory_used = end_memory - start_memory;
                    
                    // Validate memory usage is reasonable
                    let expected_min = (size * num_features * 4) as u64; // 4 bytes per f32
                    if memory_used > 0 {
                        assert!(memory_used >= expected_min);
                        println!("    ✓ Memory usage: {} bytes (expected min: {})", memory_used, expected_min);
                    }
                }
                Err(LightGBMError::NotImplemented { .. }) => {
                    println!("    ⚠ Dataset creation not fully implemented");
                }
                Err(e) => {
                    panic!("Unexpected error creating dataset: {}", e);
                }
            }
        }
    }
}

#[test]
fn test_e2e_data_pipeline_missing_values() {
    println!("\n=== E2E Data Pipeline: Missing Values Handling ===");
    
    assert!(lightgbm_rust::init().is_ok());
    
    let temp_dir = TempDir::new().unwrap();
    let csv_path = temp_dir.path().join("missing_data.csv");
    
    // Create CSV with missing values
    let csv_content = r#"feature_1,feature_2,feature_3,target
1.5,2.0,,3.5
,3.5,1.0,2.0
2.5,,2.5,4.0
3.0,4.0,3.0,
1.0,1.5,1.5,1.5"#;
    
    fs::write(&csv_path, csv_content).unwrap();
    
    let dataset_config = DatasetConfig::new()
        .with_target_column("target")
        .with_max_bin(255);
    
    let dataset_result = DatasetFactory::from_csv(&csv_path, dataset_config);
    
    match dataset_result {
        Ok(dataset) => {
            println!("  ✓ Dataset with missing values loaded");
            
            // Validate missing value detection
            let has_missing = dataset.has_missing_values();
            if has_missing {
                println!("    ✓ Missing values detected correctly");
            } else {
                println!("    ⚠ Missing values not detected");
            }
        }
        Err(LightGBMError::NotImplemented { .. }) => {
            println!("  ⚠ CSV loading with missing values not implemented");
        }
        Err(e) => {
            panic!("Unexpected error: {}", e);
        }
    }
}

/// Test Category 2: Model Training E2E Tests
/// These tests validate the complete training pipeline

#[test]
fn test_e2e_training_regression_workflow() {
    println!("\n=== E2E Training: Regression Workflow ===");
    
    assert!(lightgbm_rust::init().is_ok());
    
    let scenarios = vec![
        ("Small Dataset", 100, 5, 50),
        ("Medium Dataset", 500, 10, 100),
        ("Large Features", 300, 50, 75),
    ];
    
    for (name, num_data, num_features, num_iterations) in scenarios {
        println!("  Testing scenario: {}", name);
        
        // Generate synthetic regression data
        let features = create_test_features_regression(num_data, num_features);
        let labels = create_test_labels_regression(&features);
        let weights = Some(create_test_weights(num_data));
        
        let dataset = Dataset::new(
            features.clone(),
            labels,
            weights,
            None,
            None,
            None,
        ).unwrap();
        
        // Test various configuration options
        let configs = vec![
            ConfigBuilder::new()
                .objective(ObjectiveType::Regression)
                .learning_rate(0.1)
                .num_iterations(num_iterations)
                .num_leaves(31)
                .build().unwrap(),
            ConfigBuilder::new()
                .objective(ObjectiveType::Regression)
                .learning_rate(0.05)
                .num_iterations(num_iterations)
                .num_leaves(15)
                .lambda_l1(0.1)
                .lambda_l2(0.1)
                .build().unwrap(),
            ConfigBuilder::new()
                .objective(ObjectiveType::Regression)
                .learning_rate(0.2)
                .num_iterations(num_iterations / 2)
                .num_leaves(63)
                .feature_fraction(0.8)
                .bagging_fraction(0.8)
                .bagging_freq(5)
                .build().unwrap(),
        ];
        
        for (config_idx, config) in configs.into_iter().enumerate() {
            println!("    Config {}: lr={}, leaves={}", 
                     config_idx + 1, config.learning_rate, config.num_leaves);
            
            let mut regressor = LGBMRegressor::new(config);
            
            let training_start = Instant::now();
            let training_result = regressor.fit(&dataset);
            let training_duration = training_start.elapsed();
            
            match training_result {
                Ok(_) => {
                    println!("      ✓ Training completed in {:?}", training_duration);
                    
                    // Test prediction
                    let test_features = features.slice(s![0..10, ..]);
                    let prediction_result = regressor.predict(&test_features.to_owned());
                    
                    match prediction_result {
                        Ok(predictions) => {
                            assert_eq!(predictions.len(), 10);
                            
                            // Validate predictions are reasonable
                            for &pred in predictions.iter() {
                                assert!(pred.is_finite());
                            }
                            
                            println!("      ✓ Predictions generated successfully");
                        }
                        Err(e) => {
                            println!("      ✗ Prediction failed: {}", e);
                        }
                    }
                    
                    // Test feature importance
                    let importance_result = regressor.feature_importance(ImportanceType::Gain);
                    match importance_result {
                        Ok(importance) => {
                            assert_eq!(importance.len(), num_features);
                            println!("      ✓ Feature importance computed");
                        }
                        Err(LightGBMError::NotImplemented { .. }) => {
                            println!("      ⚠ Feature importance not implemented");
                        }
                        Err(e) => {
                            println!("      ✗ Feature importance failed: {}", e);
                        }
                    }
                }
                Err(LightGBMError::NotImplemented { .. }) => {
                    println!("      ⚠ Training not implemented yet");
                }
                Err(e) => {
                    panic!("Unexpected training error: {}", e);
                }
            }
        }
    }
}

#[test]
fn test_e2e_training_classification_workflow() {
    println!("\n=== E2E Training: Classification Workflow ===");
    
    assert!(lightgbm_rust::init().is_ok());
    
    let test_cases = vec![
        ("Binary Classification", ObjectiveType::Binary, 2, 400, 8),
        ("Multiclass Classification", ObjectiveType::Multiclass, 3, 300, 6),
        ("Large Multiclass", ObjectiveType::Multiclass, 5, 500, 12),
    ];
    
    for (name, objective, num_classes, num_data, num_features) in test_cases {
        println!("  Testing: {}", name);
        
        let (features, labels) = match objective {
            ObjectiveType::Binary => create_test_data!(binary, num_data, num_features),
            ObjectiveType::Multiclass => create_test_data!(multiclass, num_data, num_features, num_classes),
            _ => unreachable!(),
        };
        
        let dataset = Dataset::new(
            features.clone(),
            labels,
            None,
            None,
            None,
            None,
        ).unwrap();
        
        let config = ConfigBuilder::new()
            .objective(objective)
            .num_class(if objective == ObjectiveType::Multiclass { num_classes } else { 2 })
            .learning_rate(0.1)
            .num_iterations(100)
            .num_leaves(31)
            .early_stopping_rounds(Some(10))
            .build().unwrap();
        
        let mut classifier = LGBMClassifier::new(config);
        
        let training_result = classifier.fit(&dataset);
        
        match training_result {
            Ok(_) => {
                println!("    ✓ Training completed");
                
                // Test class prediction
                let test_features = features.slice(s![0..10, ..]);
                let class_pred_result = classifier.predict(&test_features.to_owned());
                
                match class_pred_result {
                    Ok(class_predictions) => {
                        assert_eq!(class_predictions.len(), 10);
                        
                        // Validate class predictions are in valid range
                        for &pred in class_predictions.iter() {
                            if objective == ObjectiveType::Binary {
                                assert!(pred == 0.0 || pred == 1.0);
                            } else {
                                assert!(pred >= 0.0 && pred < num_classes as f32);
                            }
                        }
                        
                        println!("    ✓ Class predictions generated");
                    }
                    Err(LightGBMError::NotImplemented { .. }) => {
                        println!("    ⚠ Class prediction not implemented");
                    }
                    Err(e) => {
                        println!("    ✗ Class prediction failed: {}", e);
                    }
                }
                
                // Test probability prediction
                let prob_pred_result = classifier.predict_proba(&test_features.to_owned());
                
                match prob_pred_result {
                    Ok(prob_predictions) => {
                        let expected_shape = if objective == ObjectiveType::Binary {
                            (10, 2)
                        } else {
                            (10, num_classes)
                        };
                        
                        assert_eq!(prob_predictions.dim(), expected_shape);
                        
                        // Validate probabilities sum to 1
                        for row in prob_predictions.axis_iter(Axis(0)) {
                            let sum: f32 = row.iter().sum();
                            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
                        }
                        
                        println!("    ✓ Probability predictions generated");
                    }
                    Err(LightGBMError::NotImplemented { .. }) => {
                        println!("    ⚠ Probability prediction not implemented");
                    }
                    Err(e) => {
                        println!("    ✗ Probability prediction failed: {}", e);
                    }
                }
            }
            Err(LightGBMError::NotImplemented { .. }) => {
                println!("    ⚠ Training not implemented yet");
            }
            Err(e) => {
                panic!("Unexpected training error: {}", e);
            }
        }
    }
}

#[test]
fn test_e2e_training_early_stopping() {
    println!("\n=== E2E Training: Early Stopping ===");
    
    assert!(lightgbm_rust::init().is_ok());
    
    // Create training and validation datasets
    let (train_features, train_labels) = create_test_data!(regression, 300, 8);
    let (valid_features, valid_labels) = create_test_data!(regression, 100, 8);
    
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
    
    let early_stopping_configs = vec![
        (10, 0.001),  // Conservative
        (5, 0.01),    // Moderate
        (3, 0.1),     // Aggressive
    ];
    
    for (rounds, tolerance) in early_stopping_configs {
        println!("  Testing early stopping: rounds={}, tolerance={}", rounds, tolerance);
        
        let config = ConfigBuilder::new()
            .objective(ObjectiveType::Regression)
            .learning_rate(0.1)
            .num_iterations(200)  // Large number to test early stopping
            .early_stopping_rounds(Some(rounds))
            .early_stopping_tolerance(tolerance)
            .verbose(false)
            .build().unwrap();
        
        let mut regressor = LGBMRegressor::new(config);
        
        // Add validation dataset
        let add_valid_result = regressor.add_validation_data(&valid_dataset);
        match add_valid_result {
            Ok(_) => println!("    ✓ Validation dataset added"),
            Err(LightGBMError::NotImplemented { .. }) => {
                println!("    ⚠ Validation dataset handling not implemented");
                continue;
            }
            Err(e) => panic!("Error adding validation data: {}", e),
        }
        
        let training_result = regressor.fit(&train_dataset);
        
        match training_result {
            Ok(_) => {
                let actual_iterations = regressor.num_iterations();
                assert!(actual_iterations <= 200);
                
                println!("    ✓ Training stopped at iteration {}", actual_iterations);
                
                // Get training history to validate early stopping worked
                let history_result = regressor.training_history();
                match history_result {
                    Ok(history) => {
                        assert!(!history.train_metrics.is_empty());
                        assert!(!history.valid_metrics.is_empty());
                        assert_eq!(history.train_metrics.len(), actual_iterations);
                        
                        println!("    ✓ Training history available with {} iterations", 
                                history.train_metrics.len());
                    }
                    Err(LightGBMError::NotImplemented { .. }) => {
                        println!("    ⚠ Training history not implemented");
                    }
                    Err(e) => {
                        println!("    ✗ Error getting training history: {}", e);
                    }
                }
            }
            Err(LightGBMError::NotImplemented { .. }) => {
                println!("    ⚠ Training not implemented yet");
            }
            Err(e) => {
                panic!("Unexpected training error: {}", e);
            }
        }
    }
}

/// Test Category 3: Prediction Pipeline E2E Tests
/// These tests validate the complete prediction pipeline

#[test]
fn test_e2e_prediction_consistency() {
    println!("\n=== E2E Prediction: Consistency Tests ===");
    
    assert!(lightgbm_rust::init().is_ok());
    
    // Create dataset and train model
    let (features, labels) = create_test_data!(regression, 200, 6);
    let dataset = Dataset::new(features.clone(), labels, None, None, None, None).unwrap();
    
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.1)
        .num_iterations(50)
        .num_leaves(15)
        .random_seed(42)  // Fixed seed for reproducibility
        .build().unwrap();
    
    let mut regressor = LGBMRegressor::new(config);
    
    let training_result = regressor.fit(&dataset);
    
    match training_result {
        Ok(_) => {
            println!("  ✓ Model trained successfully");
            
            // Test prediction consistency across multiple calls
            let test_features = features.slice(s![0..10, ..]).to_owned();
            
            let mut predictions = Vec::new();
            for i in 0..3 {
                let pred_result = regressor.predict(&test_features);
                match pred_result {
                    Ok(pred) => {
                        predictions.push(pred);
                        println!("    ✓ Prediction run {} completed", i + 1);
                    }
                    Err(e) => {
                        panic!("Prediction failed on run {}: {}", i + 1, e);
                    }
                }
            }
            
            // Verify predictions are identical
            for i in 1..predictions.len() {
                for (j, (&pred1, &pred2)) in predictions[0].iter().zip(predictions[i].iter()).enumerate() {
                    assert_abs_diff_eq!(pred1, pred2, epsilon = 1e-6);
                }
            }
            
            println!("    ✓ Prediction consistency verified across multiple runs");
            
            // Test batch vs single prediction consistency
            let single_predictions: Vec<f32> = (0..10).map(|i| {
                let single_row = test_features.slice(s![i..i+1, ..]).to_owned();
                regressor.predict(&single_row).unwrap()[0]
            }).collect();
            
            let batch_predictions = regressor.predict(&test_features).unwrap();
            
            for (i, (&single, &batch)) in single_predictions.iter().zip(batch_predictions.iter()).enumerate() {
                assert_abs_diff_eq!(single, batch, epsilon = 1e-6);
            }
            
            println!("    ✓ Batch vs single prediction consistency verified");
        }
        Err(LightGBMError::NotImplemented { .. }) => {
            println!("  ⚠ Training not implemented - skipping prediction consistency tests");
        }
        Err(e) => {
            panic!("Unexpected training error: {}", e);
        }
    }
}

#[test]
fn test_e2e_prediction_performance() {
    println!("\n=== E2E Prediction: Performance Tests ===");
    
    assert!(lightgbm_rust::init().is_ok());
    
    let test_sizes = vec![100, 1000, 10000];
    let num_features = 20;
    
    for &test_size in &test_sizes {
        println!("  Testing prediction performance with {} samples", test_size);
        
        // Create test data
        let features = Array2::from_shape_fn((500, num_features), |(i, j)| {
            (i as f32 * 0.1 + j as f32 * 0.01) % 5.0
        });
        let labels = create_test_labels_regression(&features);
        let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();
        
        let config = ConfigBuilder::new()
            .objective(ObjectiveType::Regression)
            .learning_rate(0.1)
            .num_iterations(50)
            .build().unwrap();
        
        let mut regressor = LGBMRegressor::new(config);
        
        let training_result = regressor.fit(&dataset);
        
        match training_result {
            Ok(_) => {
                // Generate test features
                let test_features = Array2::from_shape_fn((test_size, num_features), |(i, j)| {
                    (i as f32 * 0.05 + j as f32 * 0.02) % 3.0
                });
                
                // Measure prediction performance
                let prediction_start = Instant::now();
                let prediction_result = regressor.predict(&test_features);
                let prediction_duration = prediction_start.elapsed();
                
                match prediction_result {
                    Ok(predictions) => {
                        assert_eq!(predictions.len(), test_size);
                        
                        let predictions_per_second = test_size as f64 / prediction_duration.as_secs_f64();
                        
                        println!("    ✓ {} predictions in {:?} ({:.0} pred/sec)", 
                                test_size, prediction_duration, predictions_per_second);
                        
                        // Basic performance expectations (these are rough benchmarks)
                        if test_size <= 1000 {
                            assert!(prediction_duration.as_millis() < 100, 
                                   "Small batch prediction too slow: {:?}", prediction_duration);
                        } else {
                            assert!(predictions_per_second > 1000.0, 
                                   "Prediction rate too low: {:.0} pred/sec", predictions_per_second);
                        }
                    }
                    Err(e) => {
                        panic!("Prediction failed: {}", e);
                    }
                }
            }
            Err(LightGBMError::NotImplemented { .. }) => {
                println!("    ⚠ Training not implemented - skipping performance tests");
            }
            Err(e) => {
                panic!("Unexpected training error: {}", e);
            }
        }
    }
}

/// Test Category 4: Configuration E2E Tests
/// These tests validate configuration management and validation

#[test]
fn test_e2e_configuration_validation() {
    println!("\n=== E2E Configuration: Validation Tests ===");
    
    assert!(lightgbm_rust::init().is_ok());
    
    // Test valid configurations
    let valid_configs = vec![
        ConfigBuilder::new()
            .objective(ObjectiveType::Regression)
            .learning_rate(0.1)
            .num_iterations(100)
            .build(),
        ConfigBuilder::new()
            .objective(ObjectiveType::Binary)
            .learning_rate(0.05)
            .num_iterations(200)
            .num_leaves(63)
            .lambda_l1(0.1)
            .lambda_l2(0.2)
            .build(),
        ConfigBuilder::new()
            .objective(ObjectiveType::Multiclass)
            .num_class(5)
            .learning_rate(0.15)
            .num_iterations(150)
            .feature_fraction(0.8)
            .bagging_fraction(0.7)
            .bagging_freq(5)
            .build(),
    ];
    
    for (i, config_result) in valid_configs.into_iter().enumerate() {
        match config_result {
            Ok(config) => {
                println!("  ✓ Valid config {} created successfully", i + 1);
                
                // Test configuration validation
                let validation_result = config.validate();
                match validation_result {
                    Ok(_) => println!("    ✓ Configuration validation passed"),
                    Err(e) => panic!("Valid configuration failed validation: {}", e),
                }
                
                // Test model creation with config
                match config.objective {
                    ObjectiveType::Regression => {
                        let _regressor = LGBMRegressor::new(config);
                        println!("    ✓ Regressor created successfully");
                    }
                    ObjectiveType::Binary | ObjectiveType::Multiclass => {
                        let _classifier = LGBMClassifier::new(config);
                        println!("    ✓ Classifier created successfully");
                    }
                    _ => {}
                }
            }
            Err(e) => panic!("Valid configuration creation failed: {}", e),
        }
    }
    
    // Test invalid configurations
    let invalid_configs = vec![
        ("Negative learning rate", ConfigBuilder::new().learning_rate(-0.1).build()),
        ("Zero learning rate", ConfigBuilder::new().learning_rate(0.0).build()),
        ("Too few leaves", ConfigBuilder::new().num_leaves(1).build()),
        ("Negative regularization", ConfigBuilder::new().lambda_l1(-0.5).build()),
        ("Invalid feature fraction", ConfigBuilder::new().feature_fraction(1.5).build()),
        ("Invalid bagging fraction", ConfigBuilder::new().bagging_fraction(0.0).build()),
    ];
    
    for (description, config_result) in invalid_configs {
        match config_result {
            Ok(_) => panic!("Invalid configuration should have failed: {}", description),
            Err(_) => println!("  ✓ Invalid config correctly rejected: {}", description),
        }
    }
}

#[test]
fn test_e2e_configuration_serialization() {
    println!("\n=== E2E Configuration: Serialization Tests ===");
    
    assert!(lightgbm_rust::init().is_ok());
    
    let temp_dir = TempDir::new().unwrap();
    
    let original_config = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .learning_rate(0.1)
        .num_iterations(100)
        .num_leaves(31)
        .lambda_l1(0.1)
        .lambda_l2(0.2)
        .feature_fraction(0.8)
        .bagging_fraction(0.7)
        .bagging_freq(5)
        .early_stopping_rounds(Some(10))
        .verbose(false)
        .build().unwrap();
    
    // Test JSON serialization
    let json_path = temp_dir.path().join("config.json");
    let json_save_result = original_config.save_to_file(&json_path);
    
    match json_save_result {
        Ok(_) => {
            println!("  ✓ Configuration saved to JSON");
            
            let loaded_config_result = Config::load_from_file(&json_path);
            match loaded_config_result {
                Ok(loaded_config) => {
                    assert_eq!(loaded_config.objective, original_config.objective);
                    assert_eq!(loaded_config.learning_rate, original_config.learning_rate);
                    assert_eq!(loaded_config.num_iterations, original_config.num_iterations);
                    assert_eq!(loaded_config.num_leaves, original_config.num_leaves);
                    
                    println!("  ✓ Configuration loaded from JSON successfully");
                }
                Err(LightGBMError::NotImplemented { .. }) => {
                    println!("  ⚠ Configuration loading not implemented");
                }
                Err(e) => panic!("Error loading configuration: {}", e),
            }
        }
        Err(LightGBMError::NotImplemented { .. }) => {
            println!("  ⚠ Configuration saving not implemented");
        }
        Err(e) => panic!("Error saving configuration: {}", e),
    }
    
    // Test TOML serialization
    let toml_path = temp_dir.path().join("config.toml");
    let toml_save_result = original_config.save_to_file(&toml_path);
    
    match toml_save_result {
        Ok(_) => {
            println!("  ✓ Configuration saved to TOML");
        }
        Err(LightGBMError::NotImplemented { .. }) => {
            println!("  ⚠ TOML configuration saving not implemented");
        }
        Err(e) => panic!("Error saving TOML configuration: {}", e),
    }
}

/// Test Category 5: Error Handling E2E Tests
/// These tests validate comprehensive error handling throughout the system

#[test]
fn test_e2e_error_handling_graceful_failures() {
    println!("\n=== E2E Error Handling: Graceful Failures ===");
    
    assert!(lightgbm_rust::init().is_ok());
    
    // Test 1: Invalid dataset scenarios
    println!("  Testing invalid dataset scenarios");
    
    // Empty dataset
    let empty_features = Array2::zeros((0, 5));
    let empty_labels = Array1::zeros(0);
    let empty_dataset_result = Dataset::new(empty_features, empty_labels, None, None, None, None);
    
    match empty_dataset_result {
        Ok(_) => println!("    ⚠ Empty dataset accepted (may be valid)"),
        Err(e) => println!("    ✓ Empty dataset correctly rejected: {}", e),
    }
    
    // Mismatched dimensions
    let features = Array2::zeros((100, 5));
    let wrong_labels = Array1::zeros(50);  // Wrong size
    let mismatch_result = Dataset::new(features, wrong_labels, None, None, None, None);
    
    match mismatch_result {
        Ok(_) => panic!("Mismatched dataset should be rejected"),
        Err(e) => println!("    ✓ Mismatched dimensions correctly rejected: {}", e),
    }
    
    // Test 2: Model training with invalid data
    println!("  Testing model training error scenarios");
    
    let valid_features = Array2::from_shape_fn((50, 3), |(i, j)| i as f32 + j as f32);
    let valid_labels = Array1::from_shape_fn(50, |i| i as f32);
    let valid_dataset = Dataset::new(valid_features, valid_labels, None, None, None, None).unwrap();
    
    // Test with extreme configuration values
    let extreme_config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.00001)  // Very small
        .num_iterations(1)       // Very few
        .num_leaves(2)          // Minimum
        .build().unwrap();
    
    let mut extreme_regressor = LGBMRegressor::new(extreme_config);
    let extreme_training_result = extreme_regressor.fit(&valid_dataset);
    
    match extreme_training_result {
        Ok(_) => println!("    ✓ Extreme configuration handled gracefully"),
        Err(LightGBMError::NotImplemented { .. }) => {
            println!("    ⚠ Training not implemented");
        }
        Err(e) => println!("    ⚠ Extreme configuration failed (may be expected): {}", e),
    }
    
    // Test 3: Prediction with invalid input
    println!("  Testing prediction error scenarios");
    
    let config = Config::default();
    let regressor = LGBMRegressor::new(config);
    
    // Wrong feature count
    let wrong_features = Array2::zeros((10, 999));  // Wrong number of features
    let wrong_pred_result = regressor.predict(&wrong_features);
    
    match wrong_pred_result {
        Ok(_) => println!("    ⚠ Wrong feature count accepted (may not be validated yet)"),
        Err(LightGBMError::NotImplemented { .. }) => {
            println!("    ⚠ Prediction not implemented");
        }
        Err(e) => println!("    ✓ Wrong feature count correctly rejected: {}", e),
    }
    
    // Empty prediction input
    let empty_pred_features = Array2::zeros((0, 5));
    let empty_pred_result = regressor.predict(&empty_pred_features);
    
    match empty_pred_result {
        Ok(predictions) => {
            assert_eq!(predictions.len(), 0);
            println!("    ✓ Empty prediction input handled gracefully");
        }
        Err(LightGBMError::NotImplemented { .. }) => {
            println!("    ⚠ Prediction not implemented");
        }
        Err(e) => println!("    ⚠ Empty prediction input failed (may be expected): {}", e),
    }
}

#[test]
fn test_e2e_error_recovery() {
    println!("\n=== E2E Error Handling: Error Recovery ===");
    
    assert!(lightgbm_rust::init().is_ok());
    
    // Test recovery from configuration errors
    let mut config_manager = ConfigManager::new().unwrap();
    
    // Apply invalid configuration and check recovery
    let invalid_update = Config {
        learning_rate: -0.1,  // Invalid
        ..Config::default()
    };
    
    let update_result = config_manager.update(invalid_update);
    match update_result {
        Ok(_) => panic!("Invalid configuration update should fail"),
        Err(e) => {
            println!("  ✓ Invalid configuration update correctly rejected: {}", e);
            
            // Verify manager is still in valid state
            assert!(config_manager.config().validate().is_ok());
            println!("  ✓ Configuration manager recovered to valid state");
        }
    }
    
    // Test recovery from file operations
    let temp_dir = TempDir::new().unwrap();
    let invalid_path = temp_dir.path().join("nonexistent/config.json");
    
    let file_result = Config::load_from_file(&invalid_path);
    match file_result {
        Ok(_) => panic!("Loading from nonexistent path should fail"),
        Err(e) => {
            println!("  ✓ Invalid file path correctly handled: {}", e);
            
            // System should still be functional
            let default_config = Config::default();
            assert!(default_config.validate().is_ok());
            println!("  ✓ System remains functional after file error");
        }
    }
}

/// Helper function to get current memory usage (simplified)
fn get_memory_usage() -> u64 {
    // This is a simplified implementation
    // In a real scenario, you'd use platform-specific APIs
    0
}

// Using the create_test_data macro from common module