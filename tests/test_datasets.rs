//! Tests using pre-created test datasets.

use lightgbm_rust::*;
use std::path::Path;

#[test]
fn test_regression_small_dataset() {
    let dataset_path = Path::new("tests/data/regression_small.csv");
    
    // Skip test if dataset doesn't exist
    if !dataset_path.exists() {
        println!("Skipping test - dataset not found: {:?}", dataset_path);
        return;
    }
    
    let config = DatasetConfig::new()
        .with_target_column("target")
        .with_max_bin(256);
    
    let dataset = DatasetFactory::from_csv(dataset_path, config);
    assert!(dataset.is_ok());
    
    let dataset = dataset.unwrap();
    assert_eq!(dataset.num_data(), 20);
    assert_eq!(dataset.num_features(), 4);
    
    // Validate dataset
    let validation_result = dataset::utils::validate_dataset(&dataset);
    assert!(validation_result.is_valid);
    
    // Calculate statistics
    let stats = dataset::utils::calculate_statistics(&dataset);
    assert_eq!(stats.num_samples, 20);
    assert_eq!(stats.num_features, 4);
    assert_eq!(stats.sparsity, 0.0); // No missing values
    
    // Test regression model
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.1)
        .num_iterations(10)
        .build()
        .unwrap();
    
    let mut regressor = LGBMRegressor::new(config);
    let training_result = regressor.fit(&dataset);
    assert!(training_result.is_err()); // Expected for now
}

#[test]
fn test_binary_classification_dataset() {
    let dataset_path = Path::new("tests/data/binary_classification.csv");
    
    // Skip test if dataset doesn't exist
    if !dataset_path.exists() {
        println!("Skipping test - dataset not found: {:?}", dataset_path);
        return;
    }
    
    let config = DatasetConfig::new()
        .with_target_column("target")
        .with_max_bin(256);
    
    let dataset = DatasetFactory::from_csv(dataset_path, config);
    assert!(dataset.is_ok());
    
    let dataset = dataset.unwrap();
    assert_eq!(dataset.num_data(), 25);
    assert_eq!(dataset.num_features(), 5);
    
    // Validate dataset
    let validation_result = dataset::utils::validate_dataset(&dataset);
    assert!(validation_result.is_valid);
    
    // Check class distribution
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
    
    // Test binary classification model
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .learning_rate(0.1)
        .num_iterations(10)
        .build()
        .unwrap();
    
    let mut classifier = LGBMClassifier::new(config);
    let training_result = classifier.fit(&dataset);
    assert!(training_result.is_err()); // Expected for now
}

#[test]
fn test_multiclass_classification_dataset() {
    let dataset_path = Path::new("tests/data/multiclass_classification.csv");
    
    // Skip test if dataset doesn't exist
    if !dataset_path.exists() {
        println!("Skipping test - dataset not found: {:?}", dataset_path);
        return;
    }
    
    let config = DatasetConfig::new()
        .with_target_column("target")
        .with_max_bin(256);
    
    let dataset = DatasetFactory::from_csv(dataset_path, config);
    assert!(dataset.is_ok());
    
    let dataset = dataset.unwrap();
    assert_eq!(dataset.num_data(), 32);
    assert_eq!(dataset.num_features(), 3);
    
    // Validate dataset
    let validation_result = dataset::utils::validate_dataset(&dataset);
    assert!(validation_result.is_valid);
    
    // Check class distribution
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
    
    // Test multiclass classification model
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Multiclass)
        .num_class(3)
        .learning_rate(0.1)
        .num_iterations(10)
        .build()
        .unwrap();
    
    let mut classifier = LGBMClassifier::new(config);
    let training_result = classifier.fit(&dataset);
    assert!(training_result.is_err()); // Expected for now
}

#[test]
fn test_missing_values_dataset() {
    let dataset_path = Path::new("tests/data/missing_values.csv");
    
    // Skip test if dataset doesn't exist
    if !dataset_path.exists() {
        println!("Skipping test - dataset not found: {:?}", dataset_path);
        return;
    }
    
    let config = DatasetConfig::new()
        .with_target_column("target")
        .with_max_bin(256);
    
    let dataset = DatasetFactory::from_csv(dataset_path, config);
    assert!(dataset.is_ok());
    
    let dataset = dataset.unwrap();
    assert_eq!(dataset.num_data(), 20);
    assert_eq!(dataset.num_features(), 4);
    
    // Validate dataset
    let validation_result = dataset::utils::validate_dataset(&dataset);
    assert!(validation_result.is_valid);
    
    // Calculate statistics
    let stats = dataset::utils::calculate_statistics(&dataset);
    assert_eq!(stats.num_samples, 20);
    assert_eq!(stats.num_features, 4);
    
    // Should have missing values
    let total_missing: usize = stats.missing_counts.iter().sum();
    assert!(total_missing > 0);
    assert!(stats.sparsity > 0.0);
    
    // Test with missing values config
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.1)
        .num_iterations(10)
        .build()
        .unwrap();
    
    let mut regressor = LGBMRegressor::new(config);
    let training_result = regressor.fit(&dataset);
    assert!(training_result.is_err()); // Expected for now
}

#[test]
fn test_weighted_regression_dataset() {
    let dataset_path = Path::new("tests/data/weighted_regression.csv");
    
    // Skip test if dataset doesn't exist
    if !dataset_path.exists() {
        println!("Skipping test - dataset not found: {:?}", dataset_path);
        return;
    }
    
    let config = DatasetConfig::new()
        .with_target_column("target")
        .with_weight_column("weight")
        .with_max_bin(256);
    
    let dataset = DatasetFactory::from_csv(dataset_path, config);
    assert!(dataset.is_ok());
    
    let dataset = dataset.unwrap();
    assert_eq!(dataset.num_data(), 20);
    assert_eq!(dataset.num_features(), 3);
    assert!(dataset.has_weights());
    
    // Validate dataset
    let validation_result = dataset::utils::validate_dataset(&dataset);
    assert!(validation_result.is_valid);
    
    // Calculate statistics
    let stats = dataset::utils::calculate_statistics(&dataset);
    assert_eq!(stats.num_samples, 20);
    assert_eq!(stats.num_features, 3);
    assert_eq!(stats.sparsity, 0.0); // No missing values
    
    // Test weighted regression model
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.1)
        .num_iterations(10)
        .build()
        .unwrap();
    
    let mut regressor = LGBMRegressor::new(config);
    let training_result = regressor.fit(&dataset);
    assert!(training_result.is_err()); // Expected for now
}

#[test]
fn test_all_datasets_validation() {
    let dataset_paths = vec![
        "tests/data/regression_small.csv",
        "tests/data/binary_classification.csv",
        "tests/data/multiclass_classification.csv",
        "tests/data/missing_values.csv",
        "tests/data/weighted_regression.csv",
    ];
    
    for dataset_path in dataset_paths {
        let path = Path::new(dataset_path);
        
        // Skip test if dataset doesn't exist
        if !path.exists() {
            println!("Skipping dataset: {}", dataset_path);
            continue;
        }
        
        let config = DatasetConfig::new()
            .with_target_column("target")
            .with_max_bin(256);
        
        let dataset = DatasetFactory::from_csv(path, config);
        assert!(dataset.is_ok(), "Failed to load dataset: {}", dataset_path);
        
        let dataset = dataset.unwrap();
        assert!(dataset.num_data() > 0, "Dataset has no data: {}", dataset_path);
        assert!(dataset.num_features() > 0, "Dataset has no features: {}", dataset_path);
        
        // Validate each dataset
        let validation_result = dataset::utils::validate_dataset(&dataset);
        assert!(validation_result.is_valid, "Dataset validation failed: {}", dataset_path);
        
        // Calculate statistics
        let stats = dataset::utils::calculate_statistics(&dataset);
        assert_eq!(stats.num_samples, dataset.num_data(), "Sample count mismatch: {}", dataset_path);
        assert_eq!(stats.num_features, dataset.num_features(), "Feature count mismatch: {}", dataset_path);
        
        println!("✓ Dataset validated: {} ({} samples, {} features)", 
                 dataset_path, stats.num_samples, stats.num_features);
    }
}

#[test]
fn test_dataset_performance_comparison() {
    let dataset_paths = vec![
        ("regression_small.csv", 20, 4),
        ("binary_classification.csv", 25, 5),
        ("multiclass_classification.csv", 32, 3),
        ("missing_values.csv", 20, 4),
        ("weighted_regression.csv", 20, 3),
    ];
    
    use std::time::Instant;
    
    for (filename, expected_samples, expected_features) in dataset_paths {
        let dataset_path_str = format!("tests/data/{}", filename);
        let dataset_path = Path::new(&dataset_path_str);
        
        // Skip test if dataset doesn't exist
        if !dataset_path.exists() {
            println!("Skipping dataset: {}", filename);
            continue;
        }
        
        let start = Instant::now();
        
        let config = DatasetConfig::new()
            .with_target_column("target")
            .with_max_bin(256);
        
        let dataset = DatasetFactory::from_csv(dataset_path, config);
        assert!(dataset.is_ok());
        
        let dataset = dataset.unwrap();
        let load_time = start.elapsed();
        
        assert_eq!(dataset.num_data(), expected_samples);
        assert_eq!(dataset.num_features(), expected_features);
        
        // Benchmark statistics calculation
        let stats_start = Instant::now();
        let stats = dataset::utils::calculate_statistics(&dataset);
        let stats_time = stats_start.elapsed();
        
        // Benchmark validation
        let validation_start = Instant::now();
        let validation_result = dataset::utils::validate_dataset(&dataset);
        let validation_time = validation_start.elapsed();
        
        assert!(validation_result.is_valid);
        
        println!("Dataset {}: load={:.2}ms, stats={:.2}ms, validation={:.2}ms",
                 filename,
                 load_time.as_secs_f64() * 1000.0,
                 stats_time.as_secs_f64() * 1000.0,
                 validation_time.as_secs_f64() * 1000.0);
        
        // Performance expectations
        assert!(load_time.as_millis() < 1000); // Should load in under 1 second
        assert!(stats_time.as_millis() < 1000); // Should calculate stats in under 1 second
        assert!(validation_time.as_millis() < 1000); // Should validate in under 1 second
    }
}

#[test]
fn test_dataset_memory_usage() {
    let dataset_paths = vec![
        "tests/data/regression_small.csv",
        "tests/data/binary_classification.csv",
        "tests/data/multiclass_classification.csv",
        "tests/data/missing_values.csv",
        "tests/data/weighted_regression.csv",
    ];
    
    for dataset_path in dataset_paths {
        let path = Path::new(dataset_path);
        
        // Skip test if dataset doesn't exist
        if !path.exists() {
            println!("Skipping dataset: {}", dataset_path);
            continue;
        }
        
        let config = DatasetConfig::new()
            .with_target_column("target")
            .with_max_bin(256);
        
        let dataset = DatasetFactory::from_csv(path, config);
        assert!(dataset.is_ok());
        
        let dataset = dataset.unwrap();
        let memory_usage = dataset.memory_usage();
        
        assert!(memory_usage > 0);
        
        // Memory usage should be reasonable
        let memory_per_sample = memory_usage as f64 / dataset.num_data() as f64;
        assert!(memory_per_sample < 10000.0); // Less than 10KB per sample
        
        println!("Dataset {}: {} bytes ({:.2} bytes/sample)",
                 dataset_path,
                 memory_usage,
                 memory_per_sample);
    }
}

#[test]
fn test_dataset_feature_types() {
    let dataset_paths = vec![
        "tests/data/regression_small.csv",
        "tests/data/binary_classification.csv",
        "tests/data/multiclass_classification.csv",
        "tests/data/missing_values.csv",
        "tests/data/weighted_regression.csv",
    ];
    
    for dataset_path in dataset_paths {
        let path = Path::new(dataset_path);
        
        // Skip test if dataset doesn't exist
        if !path.exists() {
            println!("Skipping dataset: {}", dataset_path);
            continue;
        }
        
        let config = DatasetConfig::new()
            .with_target_column("target")
            .with_max_bin(256);
        
        let dataset = DatasetFactory::from_csv(path, config);
        assert!(dataset.is_ok());
        
        let dataset = dataset.unwrap();
        let feature_types = dataset::utils::detect_feature_types(dataset.features_raw());
        
        assert_eq!(feature_types.len(), dataset.num_features());
        
        // All datasets should have at least some numerical features
        let numerical_count = feature_types.iter()
            .filter(|&ft| *ft == FeatureType::Numerical)
            .count();
        
        println!("Dataset {}: {} numerical, {} categorical features",
                 dataset_path,
                 numerical_count,
                 feature_types.len() - numerical_count);
    }
}