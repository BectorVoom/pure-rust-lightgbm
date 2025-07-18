//! Integration tests for the pure Rust LightGBM implementation.
//!
//! This test suite validates that all core modules interact correctly through the public API
//! and simulates complete data pipelines from data loading to model training/prediction.

use lightgbm_rust::*;
use ndarray::{Array1, Array2};
use std::fs;
use std::path::Path;
use tempfile::TempDir;

mod common;
use common::*;

/// Test core module initialization and capabilities
#[test]
fn test_core_module_integration() {
    // Test library initialization
    assert!(lightgbm_rust::init().is_ok());
    assert!(lightgbm_rust::is_initialized());
    
    // Test capabilities
    let caps = lightgbm_rust::capabilities();
    assert!(caps.simd_aligned_memory);
    assert!(caps.thread_safe_memory);
    assert!(caps.rich_error_types);
    assert!(caps.trait_abstractions);
    assert!(caps.serialization);
    
    // Test version information
    assert!(!lightgbm_rust::VERSION.is_empty());
}

/// Test configuration system integration
#[test]
fn test_configuration_integration() {
    // Test configuration builder
    let config = ConfigBuilder::new()
        .learning_rate(0.05)
        .num_iterations(100)
        .num_leaves(31)
        .objective(ObjectiveType::Regression)
        .device_type(DeviceType::CPU)
        .num_threads(4)
        .lambda_l1(0.1)
        .lambda_l2(0.2)
        .min_data_in_leaf(20)
        .build();
    
    assert!(config.is_ok());
    let config = config.unwrap();
    
    // Validate configuration properties
    assert_eq!(config.learning_rate, 0.05);
    assert_eq!(config.num_iterations, 100);
    assert_eq!(config.num_leaves, 31);
    assert_eq!(config.objective, ObjectiveType::Regression);
    assert_eq!(config.device_type, DeviceType::CPU);
    assert_eq!(config.num_threads, 4);
    assert_eq!(config.lambda_l1, 0.1);
    assert_eq!(config.lambda_l2, 0.2);
    assert_eq!(config.min_data_in_leaf, 20);
    
    // Test configuration validation
    assert!(config.validate().is_ok());
    
    // Test invalid configuration
    let invalid_config = ConfigBuilder::new()
        .learning_rate(-0.1)  // Invalid learning rate
        .build();
    
    assert!(invalid_config.is_err());
    
    // Test configuration with different objectives
    let binary_config = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .build()
        .unwrap();
    assert_eq!(binary_config.objective, ObjectiveType::Binary);
    
    let multiclass_config = ConfigBuilder::new()
        .objective(ObjectiveType::Multiclass)
        .num_class(5)
        .build()
        .unwrap();
    assert_eq!(multiclass_config.objective, ObjectiveType::Multiclass);
    assert_eq!(multiclass_config.num_class, 5);
}

/// Test dataset management integration
#[test]
fn test_dataset_integration() {
    // Test dataset creation from arrays
    let features = Array2::from_shape_vec(
        (6, 3),
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
            13.0, 14.0, 15.0,
            16.0, 17.0, 18.0,
        ]
    ).unwrap();
    
    let labels = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    let weights = Some(Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
    
    let dataset = Dataset::new(
        features.clone(),
        labels.clone(),
        weights.clone(),
        None,
        None,
        None,
    );
    
    assert!(dataset.is_ok());
    let dataset = dataset.unwrap();
    
    // Test dataset properties
    assert_eq!(dataset.num_data(), 6);
    assert_eq!(dataset.num_features(), 3);
    assert!(dataset.has_weights());
    
    // Test dataset configuration
    let config = DatasetConfig::new()
        .with_target_column("target")
        .with_max_bin(256)
        .with_categorical_features(vec![0, 1]);
    
    assert!(config.validate().is_ok());
    assert_eq!(config.target_column, Some("target".to_string()));
    assert_eq!(config.max_bin, 256);
    assert_eq!(config.categorical_features, Some(vec![0, 1]));
}

/// Test CSV dataset loading integration
#[test]
fn test_csv_dataset_integration() {
    let temp_dir = TempDir::new().unwrap();
    let csv_path = temp_dir.path().join("test_data.csv");
    
    // Create test CSV file
    let csv_content = r#"feature1,feature2,feature3,target
1.0,2.0,3.0,0.5
4.0,5.0,6.0,1.5
7.0,8.0,9.0,2.5
10.0,11.0,12.0,3.5
13.0,14.0,15.0,4.5
16.0,17.0,18.0,5.5"#;
    
    fs::write(&csv_path, csv_content).unwrap();
    
    // Test CSV loading
    let config = DatasetConfig::new()
        .with_target_column("target")
        .with_max_bin(256);
    
    let dataset = DatasetFactory::from_csv(&csv_path, config);
    assert!(dataset.is_ok());
    
    let dataset = dataset.unwrap();
    assert_eq!(dataset.num_data(), 6);
    assert_eq!(dataset.num_features(), 3);
}

/// Test memory management integration
#[test]
fn test_memory_management_integration() {
    // Test aligned buffer creation
    let buffer: AlignedBuffer<f32> = AlignedBuffer::new(1000).unwrap();
    assert_eq!(buffer.len(), 0);
    assert_eq!(buffer.capacity(), 1000);
    assert!(buffer.is_aligned());
    
    // Test memory pool functionality
    let pool = MemoryPool::new(1024);
    assert!(pool.is_ok());
    
    // Test memory statistics
    let stats = pool.unwrap().stats();
    assert_eq!(stats.alignment, ALIGNED_SIZE);
    assert!(stats.allocated_bytes >= 0);
}

/// Test error handling integration
#[test]
fn test_error_handling_integration() {
    // Test various error types
    let config_error = LightGBMError::config("Invalid configuration");
    assert_eq!(config_error.category(), "config");
    assert!(!config_error.is_recoverable());
    
    let param_error = LightGBMError::invalid_parameter("learning_rate", "-0.1", "must be positive");
    assert_eq!(param_error.category(), "invalid_parameter");
    
    let not_impl_error = LightGBMError::not_implemented("Feature not implemented");
    assert_eq!(not_impl_error.category(), "not_implemented");
    
    // Test error propagation
    let invalid_config = ConfigBuilder::new()
        .learning_rate(-0.1)
        .build();
    
    assert!(invalid_config.is_err());
    match invalid_config.unwrap_err() {
        LightGBMError::InvalidParameter { .. } => (),
        _ => panic!("Expected InvalidParameter error"),
    }
}

/// Test data pipeline integration
#[test]
fn test_data_pipeline_integration() {
    // Initialize library
    lightgbm_rust::init().unwrap();
    
    // Create dataset
    let features = create_test_features_regression(100, 5);
    let labels = create_test_labels_regression(&features);
    let weights = Some(Array1::ones(100));
    
    let dataset = Dataset::new(
        features,
        labels,
        weights,
        None,
        None,
        None,
    ).unwrap();
    
    // Validate dataset
    let validation_result = dataset::utils::validate_dataset(&dataset);
    assert!(validation_result.is_valid);
    
    // Calculate statistics
    let stats = dataset::utils::calculate_statistics(&dataset);
    assert_eq!(stats.num_samples, 100);
    assert_eq!(stats.num_features, 5);
    assert_eq!(stats.feature_stats.len(), 5);
    
    // Test feature type detection
    let feature_types = dataset::utils::detect_feature_types(&dataset.features().to_owned());
    assert_eq!(feature_types.len(), 5);
    
    // Create configuration
    let config = ConfigBuilder::new()
        .num_iterations(10)
        .learning_rate(0.1)
        .objective(ObjectiveType::Regression)
        .build()
        .unwrap();
    
    // Test model creation (will be placeholder until boosting is implemented)
    let regressor = LGBMRegressor::new(config);
    assert_eq!(regressor.config().objective, ObjectiveType::Regression);
}

/// Test module interaction between config and dataset
#[test]
fn test_config_dataset_interaction() {
    // Test that dataset configuration affects dataset creation
    let features = Array2::from_shape_vec(
        (4, 2),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    ).unwrap();
    let labels = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
    
    // Test with categorical features
    let config = DatasetConfig::new()
        .with_categorical_features(vec![0])
        .with_max_bin(64);
    
    let dataset = DatasetFactory::from_arrays(
        features.clone(),
        labels.clone(),
        None,
        config,
    );
    
    assert!(dataset.is_ok());
    let dataset = dataset.unwrap();
    
    // Test dataset properties are affected by configuration
    assert_eq!(dataset.num_data(), 4);
    assert_eq!(dataset.num_features(), 2);
}

/// Test serialization integration
#[test]
fn test_serialization_integration() {
    // Test configuration serialization
    let config = ConfigBuilder::new()
        .learning_rate(0.05)
        .num_iterations(100)
        .objective(ObjectiveType::Binary)
        .build()
        .unwrap();
    
    // Test JSON serialization
    let json_str = serde_json::to_string(&config).unwrap();
    assert!(!json_str.is_empty());
    
    let deserialized_config: Config = serde_json::from_str(&json_str).unwrap();
    assert_eq!(deserialized_config.learning_rate, config.learning_rate);
    assert_eq!(deserialized_config.num_iterations, config.num_iterations);
    assert_eq!(deserialized_config.objective, config.objective);
    
    // Test dataset config serialization
    let dataset_config = DatasetConfig::new()
        .with_target_column("target")
        .with_max_bin(256);
    
    let dataset_json = serde_json::to_string(&dataset_config).unwrap();
    let deserialized_dataset_config: DatasetConfig = serde_json::from_str(&dataset_json).unwrap();
    assert_eq!(deserialized_dataset_config.target_column, dataset_config.target_column);
    assert_eq!(deserialized_dataset_config.max_bin, dataset_config.max_bin);
}

/// Test multi-threaded safety integration
#[test]
fn test_thread_safety_integration() {
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    // Initialize library
    lightgbm_rust::init().unwrap();
    
    // Create shared dataset
    let features = create_test_features_regression(200, 10);
    let labels = create_test_labels_regression(&features);
    let dataset = Arc::new(Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    ).unwrap());
    
    // Test concurrent access to dataset
    let handles: Vec<_> = (0..4).map(|i| {
        let dataset = Arc::clone(&dataset);
        thread::spawn(move || {
            // Each thread performs operations on the dataset
            let stats = dataset::utils::calculate_statistics(&dataset);
            assert_eq!(stats.num_samples, 200);
            assert_eq!(stats.num_features, 10);
            
            // Test dataset validation
            let validation_result = dataset::utils::validate_dataset(&dataset);
            assert!(validation_result.is_valid);
            
            i
        })
    }).collect();
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test performance characteristics
#[test]
fn test_performance_characteristics() {
    use std::time::Instant;
    
    // Test large dataset creation performance
    let start = Instant::now();
    let features = create_test_features_regression(10000, 100);
    let labels = create_test_labels_regression(&features);
    let dataset = Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    let creation_time = start.elapsed();
    
    println!("Dataset creation time: {:?}", creation_time);
    assert!(creation_time.as_secs() < 10); // Should complete within 10 seconds
    
    // Test statistics calculation performance
    let start = Instant::now();
    let stats = dataset::utils::calculate_statistics(&dataset);
    let stats_time = start.elapsed();
    
    println!("Statistics calculation time: {:?}", stats_time);
    assert!(stats_time.as_secs() < 5); // Should complete within 5 seconds
    
    // Verify results
    assert_eq!(stats.num_samples, 10000);
    assert_eq!(stats.num_features, 100);
    assert_eq!(stats.feature_stats.len(), 100);
}

/// Test edge cases and error conditions
#[test]
fn test_edge_cases() {
    // Test empty dataset
    let empty_features = Array2::zeros((0, 0));
    let empty_labels = Array1::zeros(0);
    let empty_dataset = Dataset::new(
        empty_features,
        empty_labels,
        None,
        None,
        None,
        None,
    );
    assert!(empty_dataset.is_ok());
    
    // Test mismatched dimensions
    let features = Array2::zeros((5, 3));
    let labels = Array1::zeros(3); // Wrong size
    let mismatched_dataset = Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    );
    assert!(mismatched_dataset.is_err());
    
    // Test invalid configuration values
    let invalid_config = DatasetConfig::new()
        .with_max_bin(1); // Too small
    assert!(invalid_config.validate().is_err());
    
    // Test extremely large max_bin
    let invalid_config2 = DatasetConfig::new()
        .with_max_bin(100000); // Too large
    assert!(invalid_config2.validate().is_err());
}

/// Test compatibility with different data types
#[test]
fn test_data_type_compatibility() {
    // Test with different array shapes
    let small_features = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
    let small_labels = Array1::from_vec(vec![0.0, 1.0]);
    
    let small_dataset = Dataset::new(
        small_features,
        small_labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
    assert_eq!(small_dataset.num_data(), 2);
    assert_eq!(small_dataset.num_features(), 1);
    
    // Test with larger arrays
    let large_features = Array2::zeros((1000, 50));
    let large_labels = Array1::zeros(1000);
    
    let large_dataset = Dataset::new(
        large_features,
        large_labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
    assert_eq!(large_dataset.num_data(), 1000);
    assert_eq!(large_dataset.num_features(), 50);
}

/// Test model placeholder functionality
#[test]
fn test_model_placeholder_integration() {
    // Test regressor placeholder
    let regressor = LGBMRegressor::default();
    assert_eq!(regressor.config().objective, ObjectiveType::Regression);
    
    // Test classifier placeholder
    let classifier = LGBMClassifier::default();
    assert_eq!(classifier.config().objective, ObjectiveType::Binary);
    
    // Test that fit/predict methods return NotImplemented errors
    let features = Array2::zeros((10, 5));
    let labels = Array1::zeros(10);
    let dataset = Dataset::new(
        features.clone(),
        labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
    let mut regressor = LGBMRegressor::default();
    let fit_result = regressor.fit(&dataset);
    assert!(fit_result.is_err());
    
    let predict_result = regressor.predict(&features);
    assert!(predict_result.is_err());
}