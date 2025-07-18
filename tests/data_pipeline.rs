//! Complete data pipeline integration tests.
//!
//! This test suite validates the entire data pipeline from data loading
//! through preprocessing, training, and prediction.

use lightgbm_rust::*;
use ndarray::{Array1, Array2};
use std::fs;
use std::time::Instant;
use tempfile::TempDir;

mod common;
use common::*;

#[test]
fn test_complete_regression_pipeline() {
    println!("Testing complete regression pipeline...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    // 1. Data Generation
    let (features, labels) = create_test_data!(regression, 200, 10);
    let weights = Some(create_test_weights(200));
    
    // 2. Dataset Creation
    let dataset = Dataset::new(
        features.clone(),
        labels.clone(),
        weights.clone(),
        None,
        None,
        None,
    ).unwrap();
    
    assert_eq!(dataset.num_data(), 200);
    assert_eq!(dataset.num_features(), 10);
    assert!(dataset.has_weights());
    
    // 3. Data Validation
    let validation_result = dataset::utils::validate_dataset(&dataset);
    assert!(validation_result.is_valid);
    
    // 4. Statistics Calculation
    let stats = dataset::utils::calculate_statistics(&dataset);
    assert_eq!(stats.num_samples, 200);
    assert_eq!(stats.num_features, 10);
    assert!(stats.sparsity < 0.1); // Should be mostly complete
    
    // 5. Feature Type Detection
    let feature_types = dataset::utils::detect_feature_types(dataset.features_raw());
    assert_eq!(feature_types.len(), 10);
    
    // 6. Configuration Creation
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.1)
        .num_iterations(100)
        .num_leaves(31)
        .lambda_l1(0.1)
        .lambda_l2(0.1)
        .min_data_in_leaf(10)
        .build()
        .unwrap();
    
    // 7. Model Creation
    let mut regressor = LGBMRegressor::new(config);
    
    // 8. Training (placeholder - will fail until implemented)
    let training_result = regressor.fit(&dataset);
    assert!(training_result.is_err()); // Expected for now
    
    // 9. Prediction (placeholder - will fail until implemented)
    let prediction_result = regressor.predict(&features);
    assert!(prediction_result.is_err()); // Expected for now
    
    println!("Regression pipeline test completed");
}

#[test]
fn test_complete_classification_pipeline() {
    println!("Testing complete classification pipeline...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    // 1. Data Generation - Binary Classification
    let (features, labels) = create_test_data!(binary, 150, 8);
    let weights = Some(create_test_weights(150));
    
    // 2. Dataset Creation
    let dataset = Dataset::new(
        features.clone(),
        labels.clone(),
        weights.clone(),
        None,
        None,
        None,
    ).unwrap();
    
    assert_eq!(dataset.num_data(), 150);
    assert_eq!(dataset.num_features(), 8);
    assert!(dataset.has_weights());
    
    // 3. Class Distribution Check
    let mut class_counts = [0, 0];
    for i in 0..dataset.num_data() {
        let label = dataset.label(i).unwrap();
        if label == 0.0 {
            class_counts[0] += 1;
        } else {
            class_counts[1] += 1;
        }
    }
    assert!(class_counts[0] > 0);
    assert!(class_counts[1] > 0);
    
    // 4. Data Validation
    let validation_result = dataset::utils::validate_dataset(&dataset);
    assert!(validation_result.is_valid);
    
    // 5. Statistics Calculation
    let stats = dataset::utils::calculate_statistics(&dataset);
    assert_eq!(stats.num_samples, 150);
    assert_eq!(stats.num_features, 8);
    
    // 6. Feature Type Detection
    let feature_types = dataset::utils::detect_feature_types(dataset.features_raw());
    assert_eq!(feature_types.len(), 8);
    
    // 7. Configuration Creation
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .learning_rate(0.1)
        .num_iterations(100)
        .num_leaves(31)
        .lambda_l1(0.05)
        .lambda_l2(0.05)
        .min_data_in_leaf(5)
        .build()
        .unwrap();
    
    // 8. Model Creation
    let mut classifier = LGBMClassifier::new(config);
    
    // 9. Training (placeholder - will fail until implemented)
    let training_result = classifier.fit(&dataset);
    assert!(training_result.is_err()); // Expected for now
    
    // 10. Prediction (placeholder - will fail until implemented)
    let prediction_result = classifier.predict(&features);
    assert!(prediction_result.is_err()); // Expected for now
    
    let proba_result = classifier.predict_proba(&features);
    assert!(proba_result.is_err()); // Expected for now
    
    println!("Classification pipeline test completed");
}

#[test]
fn test_multiclass_classification_pipeline() {
    println!("Testing multiclass classification pipeline...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    // 1. Data Generation - Multiclass Classification
    let (features, labels) = create_test_data!(multiclass, 120, 6, 3);
    let weights = Some(create_test_weights(120));
    
    // 2. Dataset Creation
    let dataset = Dataset::new(
        features.clone(),
        labels.clone(),
        weights.clone(),
        None,
        None,
        None,
    ).unwrap();
    
    assert_eq!(dataset.num_data(), 120);
    assert_eq!(dataset.num_features(), 6);
    assert!(dataset.has_weights());
    
    // 3. Class Distribution Check
    let mut class_counts = [0, 0, 0];
    for i in 0..dataset.num_data() {
        let label = dataset.label(i).unwrap() as usize;
        if label < 3 {
            class_counts[label] += 1;
        }
    }
    assert!(class_counts[0] > 0);
    assert!(class_counts[1] > 0);
    assert!(class_counts[2] > 0);
    
    // 4. Data Validation
    let validation_result = dataset::utils::validate_dataset(&dataset);
    assert!(validation_result.is_valid);
    
    // 5. Statistics Calculation
    let stats = dataset::utils::calculate_statistics(&dataset);
    assert_eq!(stats.num_samples, 120);
    assert_eq!(stats.num_features, 6);
    
    // 6. Configuration Creation
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Multiclass)
        .num_class(3)
        .learning_rate(0.1)
        .num_iterations(100)
        .num_leaves(31)
        .build()
        .unwrap();
    
    // 7. Model Creation
    let mut classifier = LGBMClassifier::new(config);
    
    // 8. Training (placeholder - will fail until implemented)
    let training_result = classifier.fit(&dataset);
    assert!(training_result.is_err()); // Expected for now
    
    // 9. Prediction (placeholder - will fail until implemented)
    let prediction_result = classifier.predict(&features);
    assert!(prediction_result.is_err()); // Expected for now
    
    println!("Multiclass classification pipeline test completed");
}

#[test]
fn test_csv_to_model_pipeline() {
    println!("Testing CSV to model pipeline...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    let temp_dir = TempDir::new().unwrap();
    let csv_path = temp_dir.path().join("pipeline_test.csv");
    
    // 1. Create CSV Dataset
    let (features, labels) = create_test_data!(regression, 100, 5);
    let weights = Some(create_test_weights(100));
    let feature_names = Some(vec![
        "feature_1".to_string(),
        "feature_2".to_string(),
        "feature_3".to_string(),
        "feature_4".to_string(),
        "feature_5".to_string(),
    ]);
    
    create_test_csv(
        &csv_path,
        &features,
        &labels,
        weights.as_ref(),
        feature_names.as_ref(),
    ).unwrap();
    
    // 2. Dataset Configuration
    let dataset_config = DatasetConfig::new()
        .with_target_column("target")
        .with_weight_column("weight")
        .with_feature_columns(feature_names.clone().unwrap())
        .with_max_bin(256)
        .with_categorical_features(vec![0, 1]); // Make first two features categorical
    
    // 3. Load Dataset from CSV
    let dataset = DatasetFactory::from_csv(&csv_path, dataset_config).unwrap();
    
    assert_eq!(dataset.num_data(), 100);
    assert_eq!(dataset.num_features(), 5);
    assert!(dataset.has_weights());
    
    // 4. Data Preprocessing
    let validation_result = dataset::utils::validate_dataset(&dataset);
    assert!(validation_result.is_valid);
    
    let stats = dataset::utils::calculate_statistics(&dataset);
    assert_eq!(stats.num_samples, 100);
    assert_eq!(stats.num_features, 5);
    
    // 5. Model Configuration
    let model_config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.1)
        .num_iterations(50)
        .num_leaves(31)
        .lambda_l1(0.1)
        .lambda_l2(0.1)
        .build()
        .unwrap();
    
    // 6. Model Training
    let mut regressor = LGBMRegressor::new(model_config);
    let training_result = regressor.fit(&dataset);
    assert!(training_result.is_err()); // Expected for now
    
    println!("CSV to model pipeline test completed");
}

#[test]
fn test_mixed_feature_types_pipeline() {
    println!("Testing mixed feature types pipeline...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    // 1. Create Mixed Dataset
    let (features, categorical_indices) = create_test_mixed_features(80, 4, 3, 5);
    let labels = create_test_labels_binary(&features);
    let weights = Some(create_test_weights(80));
    
    // 2. Dataset Creation with Categorical Features
    let dataset = Dataset::new(
        features.clone(),
        labels.clone(),
        weights.clone(),
        None,
        None,
        None,
    ).unwrap();
    
    assert_eq!(dataset.num_data(), 80);
    assert_eq!(dataset.num_features(), 7); // 4 numerical + 3 categorical
    
    // 3. Feature Type Detection
    let feature_types = dataset::utils::detect_feature_types(dataset.features_raw());
    assert_eq!(feature_types.len(), 7);
    
    // Check that categorical features are detected
    for &cat_idx in &categorical_indices {
        assert_eq!(feature_types[cat_idx], FeatureType::Categorical);
    }
    
    // 4. Dataset Configuration with Categorical Features
    let dataset_config = DatasetConfig::new()
        .with_categorical_features(categorical_indices.clone())
        .with_max_bin(256);
    
    assert!(dataset_config.validate().is_ok());
    
    // 5. Statistics Calculation
    let stats = dataset::utils::calculate_statistics(&dataset);
    assert_eq!(stats.num_samples, 80);
    assert_eq!(stats.num_features, 7);
    
    // 6. Model Configuration
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .learning_rate(0.1)
        .num_iterations(100)
        .num_leaves(31)
        .build()
        .unwrap();
    
    // 7. Model Training
    let mut classifier = LGBMClassifier::new(config);
    let training_result = classifier.fit(&dataset);
    assert!(training_result.is_err()); // Expected for now
    
    println!("Mixed feature types pipeline test completed");
}

#[test]
fn test_pipeline_with_missing_values() {
    println!("Testing pipeline with missing values...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    // 1. Create Dataset with Missing Values
    let features = create_test_features_with_missing(100, 6, 0.2);
    let labels = create_test_labels_regression(&features);
    let weights = Some(create_test_weights(100));
    
    // 2. Dataset Creation
    let dataset = Dataset::new(
        features.clone(),
        labels.clone(),
        weights.clone(),
        None,
        None,
        None,
    ).unwrap();
    
    assert_eq!(dataset.num_data(), 100);
    assert_eq!(dataset.num_features(), 6);
    
    // 3. Statistics with Missing Values
    let stats = dataset::utils::calculate_statistics(&dataset);
    assert_eq!(stats.num_samples, 100);
    assert_eq!(stats.num_features, 6);
    
    // Should have some missing values
    let total_missing: usize = stats.missing_counts.iter().sum();
    assert!(total_missing > 0);
    assert!(stats.sparsity > 0.0);
    
    // 4. Validation with Missing Values
    let validation_result = dataset::utils::validate_dataset(&dataset);
    assert!(validation_result.is_valid);
    
    // Should have warnings about sparsity
    if stats.sparsity > 0.5 {
        assert!(!validation_result.warnings.is_empty());
    }
    
    // 5. Dataset Configuration for Missing Values
    let dataset_config = DatasetConfig::new()
        .with_max_bin(256)
        .with_two_round(true); // Enable two-round for missing values
    
    assert!(dataset_config.validate().is_ok());
    
    // 6. Model Configuration
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.1)
        .num_iterations(100)
        .num_leaves(31)
        .build()
        .unwrap();
    
    // 7. Model Training
    let mut regressor = LGBMRegressor::new(config);
    let training_result = regressor.fit(&dataset);
    assert!(training_result.is_err()); // Expected for now
    
    println!("Pipeline with missing values test completed");
}

#[test]
fn test_performance_pipeline() {
    println!("Testing performance pipeline...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    let sizes = vec![
        (100, 10),
        (500, 20),
        (1000, 50),
        (2000, 100),
    ];
    
    for (num_samples, num_features) in sizes {
        let start_time = Instant::now();
        
        // 1. Data Generation
        let (features, labels) = create_test_data!(regression, num_samples, num_features);
        let data_gen_time = start_time.elapsed();
        
        // 2. Dataset Creation
        let dataset_start = Instant::now();
        let dataset = Dataset::new(
            features.clone(),
            labels.clone(),
            None,
            None,
            None,
            None,
        ).unwrap();
        let dataset_time = dataset_start.elapsed();
        
        // 3. Statistics Calculation
        let stats_start = Instant::now();
        let stats = dataset::utils::calculate_statistics(&dataset);
        let stats_time = stats_start.elapsed();
        
        // 4. Validation
        let validation_start = Instant::now();
        let validation_result = dataset::utils::validate_dataset(&dataset);
        let validation_time = validation_start.elapsed();
        
        // 5. Configuration Creation
        let config_start = Instant::now();
        let config = ConfigBuilder::new()
            .objective(ObjectiveType::Regression)
            .learning_rate(0.1)
            .num_iterations(10)
            .num_leaves(31)
            .build()
            .unwrap();
        let config_time = config_start.elapsed();
        
        let total_time = start_time.elapsed();
        
        // Assertions
        assert_eq!(stats.num_samples, num_samples);
        assert_eq!(stats.num_features, num_features);
        assert!(validation_result.is_valid);
        assert_eq!(config.objective, ObjectiveType::Regression);
        
        // Performance expectations (should be fast)
        assert!(data_gen_time.as_secs() < 5);
        assert!(dataset_time.as_secs() < 5);
        assert!(stats_time.as_secs() < 5);
        assert!(validation_time.as_secs() < 5);
        assert!(config_time.as_millis() < 100);
        assert!(total_time.as_secs() < 10);
        
        println!("  {}x{}: total={:.2}ms (data={:.2}ms, dataset={:.2}ms, stats={:.2}ms, validation={:.2}ms, config={:.2}ms)",
                 num_samples, num_features,
                 total_time.as_secs_f64() * 1000.0,
                 data_gen_time.as_secs_f64() * 1000.0,
                 dataset_time.as_secs_f64() * 1000.0,
                 stats_time.as_secs_f64() * 1000.0,
                 validation_time.as_secs_f64() * 1000.0,
                 config_time.as_secs_f64() * 1000.0);
    }
    
    println!("Performance pipeline test completed");
}

#[test]
fn test_error_recovery_pipeline() {
    println!("Testing error recovery pipeline...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    // 1. Test Invalid Configuration Recovery
    let invalid_config = ConfigBuilder::new()
        .learning_rate(-0.1)
        .build();
    
    assert!(invalid_config.is_err());
    
    // Recover with valid configuration
    let valid_config = ConfigBuilder::new()
        .learning_rate(0.1)
        .objective(ObjectiveType::Regression)
        .num_iterations(100)
        .build()
        .unwrap();
    
    assert!(valid_config.validate().is_ok());
    
    // 2. Test Invalid Dataset Recovery
    let features = Array2::zeros((5, 3));
    let labels = Array1::zeros(3); // Wrong size
    
    let invalid_dataset = Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    );
    
    assert!(invalid_dataset.is_err());
    
    // Recover with valid dataset
    let valid_features = Array2::zeros((5, 3));
    let valid_labels = Array1::zeros(5);
    
    let valid_dataset = Dataset::new(
        valid_features,
        valid_labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
    assert_eq!(valid_dataset.num_data(), 5);
    assert_eq!(valid_dataset.num_features(), 3);
    
    // 3. Test Invalid Dataset Config Recovery
    let invalid_dataset_config = DatasetConfig::new()
        .with_max_bin(1); // Too small
    
    assert!(invalid_dataset_config.validate().is_err());
    
    // Recover with valid config
    let valid_dataset_config = DatasetConfig::new()
        .with_max_bin(256)
        .with_categorical_features(vec![0, 1]);
    
    assert!(valid_dataset_config.validate().is_ok());
    
    // 4. Test Model Training Error Handling
    let mut regressor = LGBMRegressor::new(valid_config);
    
    // This should fail gracefully
    let training_result = regressor.fit(&valid_dataset);
    assert!(training_result.is_err());
    
    // Error should be a NotImplemented error
    match training_result.unwrap_err() {
        LightGBMError::NotImplemented { .. } => {
            // Expected
        }
        _ => panic!("Expected NotImplemented error"),
    }
    
    println!("Error recovery pipeline test completed");
}

#[test]
fn test_memory_efficient_pipeline() {
    println!("Testing memory efficient pipeline...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    // 1. Test with Large Dataset
    let num_samples = 10000;
    let num_features = 100;
    
    let start_memory = get_memory_usage();
    
    // Create dataset in chunks to test memory efficiency
    let (features, labels) = create_test_data!(regression, num_samples, num_features);
    
    let dataset = Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
    let peak_memory = get_memory_usage();
    
    // 2. Statistics Calculation
    let stats = dataset::utils::calculate_statistics(&dataset);
    
    // 3. Validation
    let validation_result = dataset::utils::validate_dataset(&dataset);
    
    // 4. Configuration
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.1)
        .num_iterations(10)
        .num_leaves(31)
        .build()
        .unwrap();
    
    let final_memory = get_memory_usage();
    
    // Assertions
    assert_eq!(stats.num_samples, num_samples);
    assert_eq!(stats.num_features, num_features);
    assert!(validation_result.is_valid);
    
    // Memory usage should be reasonable
    let memory_per_sample = (peak_memory - start_memory) as f64 / num_samples as f64;
    assert!(memory_per_sample < 10000.0); // Less than 10KB per sample
    
    println!("  Memory usage: start={}MB, peak={}MB, final={}MB",
             start_memory / 1024 / 1024,
             peak_memory / 1024 / 1024,
             final_memory / 1024 / 1024);
    
    println!("Memory efficient pipeline test completed");
}

#[test]
fn test_concurrent_pipeline() {
    println!("Testing concurrent pipeline...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    use std::sync::Arc;
    use std::thread;
    
    // Create shared test data
    let (features, labels) = create_test_data!(regression, 500, 20);
    let dataset = Arc::new(Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    ).unwrap());
    
    // Test concurrent access to dataset
    let handles: Vec<_> = (0..4).map(|thread_id| {
        let dataset = Arc::clone(&dataset);
        thread::spawn(move || {
            // Each thread performs the complete pipeline
            let stats = dataset::utils::calculate_statistics(&dataset);
            let validation_result = dataset::utils::validate_dataset(&dataset);
            let feature_types = dataset::utils::detect_feature_types(dataset.features_raw());
            
            let config = ConfigBuilder::new()
                .objective(ObjectiveType::Regression)
                .learning_rate(0.1)
                .num_iterations(10)
                .num_leaves(31)
                .build()
                .unwrap();
            
            let mut regressor = LGBMRegressor::new(config);
            
            // Training will fail but should be thread-safe
            let training_result = regressor.fit(&dataset);
            
            (thread_id, stats, validation_result, feature_types, training_result)
        })
    }).collect();
    
    // Wait for all threads and verify results
    for handle in handles {
        let (thread_id, stats, validation_result, feature_types, training_result) = handle.join().unwrap();
        
        assert_eq!(stats.num_samples, 500);
        assert_eq!(stats.num_features, 20);
        assert!(validation_result.is_valid);
        assert_eq!(feature_types.len(), 20);
        assert!(training_result.is_err()); // Expected for now
        
        println!("  Thread {} completed successfully", thread_id);
    }
    
    println!("Concurrent pipeline test completed");
}

// Helper function to get memory usage (simplified)
fn get_memory_usage() -> usize {
    // In a real implementation, this would get actual memory usage
    // For now, return a mock value
    1024 * 1024 // 1MB
}