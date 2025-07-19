//! Advanced integration tests for pure Rust LightGBM implementation.
//!
//! This test suite focuses on advanced scenarios, edge cases, and cross-module
//! interactions that test the robustness of the complete system.

use lightgbm_rust::*;
use ndarray::{Array1, Array2};
use std::fs;
use std::path::Path;
use tempfile::TempDir;
use std::time::Instant;

mod common;
use common::*;

/// Test gradient boosting engine integration with tree learners
#[test]
fn test_gbdt_tree_learner_integration() {
    println!("Testing GBDT-TreeLearner integration...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    // Create dataset
    let features = create_test_features_regression(100, 5);
    let labels = create_test_labels_regression(&features);
    let dataset = Dataset::new(
        features.clone(),
        labels.clone(),
        None,
        None,
        None,
        None,
    ).unwrap();
    
    // Test configuration for different tree learners
    let serial_config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.1)
        .num_iterations(10)
        .tree_learner(TreeLearnerType::Serial)
        .num_threads(1)
        .build()
        .unwrap();
    
    let parallel_config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.1)
        .num_iterations(10)
        .tree_learner(TreeLearnerType::Parallel)
        .num_threads(4)
        .build()
        .unwrap();
    
    // Test GBDT creation with different learners
    let serial_gbdt = GBDT::new(serial_config, dataset.clone());
    let parallel_gbdt = GBDT::new(parallel_config, dataset.clone());
    
    // Both should be created successfully
    assert!(serial_gbdt.is_ok());
    assert!(parallel_gbdt.is_ok());
    
    let serial_gbdt = serial_gbdt.unwrap();
    let parallel_gbdt = parallel_gbdt.unwrap();
    
    // Verify configurations
    assert_eq!(serial_gbdt.config().tree_learner, TreeLearnerType::Serial);
    assert_eq!(parallel_gbdt.config().tree_learner, TreeLearnerType::Parallel);
    
    // Test training interface - should work successfully
    let mut serial_gbdt = serial_gbdt;
    let training_result = serial_gbdt.train();
    assert!(training_result.is_ok());
    
    println!("GBDT-TreeLearner integration test completed");
}

/// Test objective function integration with gradient computation
#[test]
fn test_objective_function_integration() {
    println!("Testing objective function integration...");
    
    // Test different objective functions
    let objectives = vec![
        ObjectiveType::Regression,
        ObjectiveType::Binary,
        ObjectiveType::Multiclass,
    ];
    
    for objective in objectives {
        let config = match objective {
            ObjectiveType::Multiclass => {
                ConfigBuilder::new()
                    .objective(objective)
                    .num_class(3)
                    .build()
                    .unwrap()
            }
            _ => {
                ConfigBuilder::new()
                    .objective(objective)
                    .build()
                    .unwrap()
            }
        };
        
        // Test objective function creation
        let obj_func = create_objective_function(&config);
        assert!(obj_func.is_ok());
        
        let obj_func = obj_func.unwrap();
        
        // Test gradient computation interface
        let num_data = 10;
        let num_tree_per_iteration = match objective {
            ObjectiveType::Multiclass => 3,
            _ => 1,
        };
        
        let scores = Array1::zeros(num_data * num_tree_per_iteration);
        let labels = match objective {
            ObjectiveType::Regression => {
                Array1::from_vec((0..num_data).map(|i| i as f32 * 0.5).collect())
            }
            ObjectiveType::Binary => {
                Array1::from_vec((0..num_data).map(|i| (i % 2) as f32).collect())
            }
            ObjectiveType::Multiclass => {
                Array1::from_vec((0..num_data).map(|i| (i % 3) as f32).collect())
            }
            _ => Array1::zeros(num_data),
        };
        
        let mut gradients = Array1::zeros(num_data * num_tree_per_iteration);
        let mut hessians = Array1::zeros(num_data * num_tree_per_iteration);
        
        // Test gradient computation
        let result = obj_func.get_gradients(
            &scores.view(),
            &labels.view(),
            None, // weights
            &mut gradients.view_mut(),
            &mut hessians.view_mut(),
        );
        
        assert!(result.is_ok());
        
        // Verify gradient and hessian properties
        assert!(!gradients.iter().all(|&x| x == 0.0)); // Should have non-zero gradients
        assert!(!hessians.iter().all(|&x| x == 0.0));  // Should have non-zero hessians
        
        println!("  {:?} objective function test passed", objective);
    }
    
    println!("Objective function integration test completed");
}

/// Test histogram construction and split finding integration
#[test]
fn test_histogram_split_integration() {
    println!("Testing histogram-split integration...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    // Create test dataset
    let features = create_test_features_regression(200, 8);
    let labels = create_test_labels_regression(&features);
    let dataset = Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
    // Create configuration
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .max_bin(64)
        .num_leaves(15)
        .build()
        .unwrap();
    
    // Test histogram pool creation
    let hist_pool = HistogramPool::new(&config);
    assert!(hist_pool.is_ok());
    
    let mut hist_pool = hist_pool.unwrap();
    
    // Test histogram allocation
    let hist_index = hist_pool.get_histogram();
    assert!(hist_index.is_ok());
    
    let hist_index = hist_index.unwrap();
    
    // Test histogram construction
    let gradients = Array1::ones(dataset.num_data() as usize);
    let hessians = Array1::ones(dataset.num_data() as usize);
    let data_indices: Vec<i32> = (0..dataset.num_data()).map(|i| i as i32).collect();
    
    let result = hist_pool.construct_histogram(
        hist_index,
        &dataset,
        &gradients.view(),
        &hessians.view(),
        &data_indices,
        0, // feature index
    );
    
    assert!(result.is_ok());
    
    // Test split finder integration
    let split_finder = SplitFinder::new(&config);
    assert!(split_finder.is_ok());
    
    let split_finder = split_finder.unwrap();
    
    // Test split evaluation
    let split_info = split_finder.find_best_split(
        hist_index,
        &hist_pool,
        0, // feature index
        &data_indices,
        &gradients.view(),
        &hessians.view(),
    );
    
    assert!(split_info.is_ok());
    
    // Release histogram
    hist_pool.release_histogram(hist_index);
    
    println!("Histogram-split integration test completed");
}

/// Test feature binning and dataset preprocessing integration
#[test]
fn test_binning_preprocessing_integration() {
    println!("Testing binning-preprocessing integration...");
    
    // Create mixed-type dataset
    let num_samples = 150;
    let num_numerical = 5;
    let num_categorical = 3;
    
    let (features, categorical_indices) = create_test_mixed_features(
        num_samples,
        num_numerical,
        num_categorical,
        10, // max_categorical_value
    );
    
    let labels = create_test_labels_regression(&features);
    
    // Test dataset configuration with binning
    let config = DatasetConfig::new()
        .with_categorical_features(categorical_indices.clone())
        .with_max_bin(32)
        .with_min_data_in_bin(3);
    
    assert!(config.validate().is_ok());
    
    // Create dataset with preprocessing
    let dataset = DatasetFactory::from_arrays(
        features.clone(),
        labels.clone(),
        None,
        config,
    );
    
    assert!(dataset.is_ok());
    let dataset = dataset.unwrap();
    
    // Verify binning was applied
    assert!(dataset.has_bin_mappers());
    assert_eq!(dataset.num_bin_mappers(), features.ncols());
    
    // Test bin mapping for different feature types
    for feature_idx in 0..dataset.num_features() {
        let bin_mapper = dataset.bin_mapper(feature_idx).unwrap();
        
        if categorical_indices.contains(&feature_idx) {
            assert_eq!(bin_mapper.bin_type, BinType::Categorical);
        } else {
            assert_eq!(bin_mapper.bin_type, BinType::Numerical);
        }
        
        // Test value to bin conversion
        let test_value = features[[0, feature_idx]];
        let bin = bin_mapper.value_to_bin(test_value);
        assert!(bin < 32 as u32); // config.max_bin was set to 32
    }
    
    // Test preprocessing statistics
    let stats = dataset.preprocessing_stats();
    assert_eq!(stats.num_features, num_numerical + num_categorical);
    assert_eq!(stats.num_samples, num_samples);
    assert!(stats.memory_usage >= 0);
    
    println!("Binning-preprocessing integration test completed");
}

/// Test memory pool and aligned buffer integration
#[test]
fn test_memory_management_integration() {
    println!("Testing memory management integration...");
    
    // Test aligned buffer allocation
    let buffer_sizes = vec![1024, 4096, 16384, 65536];
    
    for size in buffer_sizes {
        let buffer: AlignedBuffer<f32> = AlignedBuffer::new(size).unwrap();
        
        // Verify alignment
        assert!(buffer.is_aligned());
        assert_eq!(buffer.alignment(), ALIGNED_SIZE);
        assert_eq!(buffer.capacity(), size);
        
        // Test buffer operations
        let mut buffer = buffer;
        buffer.resize(size / 2, 0.0).unwrap();
        assert_eq!(buffer.len(), size / 2);
        
        // Fill with test data
        for i in 0..buffer.len() {
            buffer[i] = i as f32;
        }
        
        // Verify data integrity
        for i in 0..buffer.len() {
            assert_eq!(buffer[i], i as f32);
        }
    }
    
    // Test memory pool
    let mut pool = MemoryPool::<f32>::new(1024 * 1024, 10); // 1MB pool
    
    // Test pool allocation
    let allocations = vec![1024, 2048, 4096];
    let mut handles = Vec::new();
    
    for size in allocations {
        let handle = pool.allocate::<f32>(size);
        assert!(handle.is_ok());
        handles.push(handle.unwrap());
    }
    
    // Test pool statistics
    let stats = pool.stats();
    assert!(stats.allocated_bytes > 0);
    assert!(stats.free_bytes > 0);
    assert!(stats.num_allocations > 0);
    
    // Test deallocation
    for handle in handles {
        pool.deallocate(handle).unwrap();
    }
    
    let final_stats = pool.stats();
    assert_eq!(final_stats.num_allocations, 0);
    
    println!("Memory management integration test completed");
}

/// Test prediction pipeline with different modes
#[test]
fn test_prediction_pipeline_integration() {
    println!("Testing prediction pipeline integration...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    // Create test dataset
    let features = create_test_features_regression(50, 4);
    let labels = create_test_labels_regression(&features);
    let dataset = Dataset::new(
        features.clone(),
        labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
    // Create configuration
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.1)
        .num_iterations(10)
        .build()
        .unwrap();
    
    // Test prediction configuration
    let pred_configs = vec![
        PredictionConfig::new()
            .with_num_iterations(Some(5))
            .with_raw_score(true),
        PredictionConfig::new()
            .with_raw_score(false)
            .with_predict_leaf_index(true),
        PredictionConfig::new()
            .with_predict_contrib(true),
        PredictionConfig::new()
            .with_early_stopping_rounds(Some(3))
            .with_early_stopping_margin(0.01),
    ];
    
    for pred_config in pred_configs {
        // Create predictor
        let mut regressor = LGBMRegressor::new(config.clone());
        
        // Training will fail for now, but test predictor creation
        let predictor = Predictor::new(regressor, pred_config.clone());
        assert!(predictor.is_ok());
        
        let predictor = predictor.unwrap();
        
        // Test prediction interface (will fail until training is implemented)
        let prediction_result = predictor.predict(&features.view());
        assert!(prediction_result.is_err());
        
        // Verify configuration
        assert_eq!(predictor.config().raw_score, pred_config.raw_score);
        assert_eq!(predictor.config().predict_leaf_index, pred_config.predict_leaf_index);
        assert_eq!(predictor.config().predict_contrib, pred_config.predict_contrib);
    }
    
    println!("Prediction pipeline integration test completed");
}

/// Test serialization integration across modules
#[test]
fn test_serialization_integration() {
    println!("Testing serialization integration...");
    
    // Test configuration serialization
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .learning_rate(0.05)
        .num_iterations(200)
        .num_leaves(63)
        .lambda_l1(0.1)
        .lambda_l2(0.2)
        .max_bin(128)
        .min_data_in_leaf(10)
        .build()
        .unwrap();
    
    // Test multiple serialization formats
    
    // JSON serialization
    let json_str = serde_json::to_string_pretty(&config).unwrap();
    assert!(!json_str.is_empty());
    
    let deserialized_config: Config = serde_json::from_str(&json_str).unwrap();
    assert_configs_equal(&config, &deserialized_config);
    
    // Bincode serialization
    let bincode_data = bincode::serialize(&config).unwrap();
    assert!(!bincode_data.is_empty());
    
    let deserialized_config: Config = bincode::deserialize(&bincode_data).unwrap();
    assert_configs_equal(&config, &deserialized_config);
    
    // TOML serialization
    let toml_str = toml::to_string(&config).unwrap();
    assert!(!toml_str.is_empty());
    
    let deserialized_config: Config = toml::from_str(&toml_str).unwrap();
    assert_configs_equal(&config, &deserialized_config);
    
    // Test dataset configuration serialization
    let dataset_config = DatasetConfig::new()
        .with_target_column("target")
        .with_feature_columns(vec!["f1".to_string(), "f2".to_string()])
        .with_categorical_features(vec![0, 1])
        .with_max_bin(256);
    
    let json_str = serde_json::to_string(&dataset_config).unwrap();
    let deserialized: DatasetConfig = serde_json::from_str(&json_str).unwrap();
    assert_eq!(deserialized.target_column, dataset_config.target_column);
    assert_eq!(deserialized.feature_columns, dataset_config.feature_columns);
    assert_eq!(deserialized.categorical_features, dataset_config.categorical_features);
    
    println!("Serialization integration test completed");
}

/// Test cross-module error propagation
#[test]
fn test_error_propagation_integration() {
    println!("Testing error propagation integration...");
    
    // Test error propagation from config to dataset
    let invalid_config = ConfigBuilder::new()
        .learning_rate(-0.5)
        .num_iterations(0)
        .build();
    
    assert!(invalid_config.is_err());
    
    match invalid_config.unwrap_err() {
        LightGBMError::InvalidParameter { parameter, .. } => {
            assert!(parameter == "learning_rate" || parameter == "num_iterations");
        }
        _ => panic!("Expected InvalidParameter error"),
    }
    
    // Test error propagation from dataset to model
    let features = Array2::zeros((5, 3));
    let labels = Array1::zeros(3); // Mismatched dimensions
    
    let dataset_result = Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    );
    
    assert!(dataset_result.is_err());
    
    match dataset_result.unwrap_err() {
        LightGBMError::DimensionMismatch { .. } => {
            // Expected
        }
        _ => panic!("Expected DataDimensionMismatch error"),
    }
    
    // Test error propagation in training pipeline
    let valid_features = Array2::zeros((5, 3));
    let valid_labels = Array1::zeros(5);
    
    let dataset = Dataset::new(
        valid_features.clone(),
        valid_labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .build()
        .unwrap();
    
    let mut regressor = LGBMRegressor::new(config);
    
    // Training should fail with NotImplemented error
    let training_result = regressor.fit(&dataset);
    assert!(training_result.is_err());
    
    match training_result.unwrap_err() {
        LightGBMError::NotImplemented { .. } => {
            // Expected
        }
        _ => panic!("Expected NotImplemented error"),
    }
    
    println!("Error propagation integration test completed");
}

/// Test resource cleanup and lifecycle management
#[test]
fn test_resource_lifecycle_integration() {
    println!("Testing resource lifecycle integration...");
    
    // Initialize library
    assert!(lightgbm_rust::init().is_ok());
    
    // Test multiple initialization calls
    assert!(lightgbm_rust::init().is_ok());
    assert!(lightgbm_rust::is_initialized());
    
    // Create resources
    let features = create_test_features_regression(100, 5);
    let labels = create_test_labels_regression(&features);
    
    let dataset = Dataset::new(
        features.clone(),
        labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .build()
        .unwrap();
    
    // Test resource cleanup with drop
    {
        let temp_dataset = dataset.clone();
        let temp_regressor = LGBMRegressor::new(config.clone());
        
        // Resources should be cleaned up when going out of scope
    }
    
    // Original resources should still be valid
    assert_eq!(dataset.num_data(), 100);
    assert_eq!(dataset.num_features(), 5);
    
    // Test memory pool lifecycle
    {
        let mut pool = MemoryPool::<f32>::new(1024, 10);
        let _handle = pool.allocate::<f32>(256).unwrap();
        
        // Pool and handle should be cleaned up when going out of scope
    }
    
    // Test histogram pool lifecycle
    {
        let mut hist_pool = HistogramPool::new(&config).unwrap();
        let hist_index = hist_pool.get_histogram().unwrap();
        hist_pool.release_histogram(hist_index);
        
        // Pool should be cleaned up when going out of scope
    }
    
    println!("Resource lifecycle integration test completed");
}

// Helper function to compare configurations
fn assert_configs_equal(config1: &Config, config2: &Config) {
    assert_eq!(config1.learning_rate, config2.learning_rate);
    assert_eq!(config1.num_iterations, config2.num_iterations);
    assert_eq!(config1.num_leaves, config2.num_leaves);
    assert_eq!(config1.objective, config2.objective);
    assert_eq!(config1.lambda_l1, config2.lambda_l1);
    assert_eq!(config1.lambda_l2, config2.lambda_l2);
    assert_eq!(config1.max_bin, config2.max_bin);
    assert_eq!(config1.min_data_in_leaf, config2.min_data_in_leaf);
}
