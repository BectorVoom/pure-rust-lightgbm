//! Regression-specific integration tests.

use lightgbm_rust::*;

use tempfile::TempDir;

mod common;
use common::*;

#[test]
fn test_regression_configuration() {
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.1)
        .num_iterations(100)
        .num_leaves(31)
        .lambda_l1(0.1)
        .lambda_l2(0.1)
        .min_data_in_leaf(20)
        .build();

    assert!(config.is_ok());
    let config = config.unwrap();
    assert_eq!(config.objective, ObjectiveType::Regression);
    assert_eq!(config.learning_rate, 0.1);
    assert_eq!(config.num_iterations, 100);
    assert_eq!(config.num_leaves, 31);
    assert_eq!(config.lambda_l1, 0.1);
    assert_eq!(config.lambda_l2, 0.1);
    assert_eq!(config.min_data_in_leaf, 20);
}

#[test]
fn test_regression_dataset_creation() {
    let (features, labels) = create_test_data!(regression, 100, 5);
    let weights = Some(create_test_weights(100));

    let dataset = Dataset::new(features, labels, weights, None, None, None);

    assert!(dataset.is_ok());
    let dataset = dataset.unwrap();
    assert_eq!(dataset.num_data(), 100);
    assert_eq!(dataset.num_features(), 5);
    assert!(dataset.has_weights());
}

#[test]
fn test_regression_dataset_validation() {
    let (features, labels) = create_test_data!(regression, 50, 3);
    let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();

    let validation_result = dataset::utils::validate_dataset(&dataset);
    assert!(validation_result.is_valid);
    assert!(validation_result.errors.is_empty());
}

#[test]
fn test_regression_statistics() {
    let (features, labels) = create_test_data!(regression, 200, 10);
    let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();

    let stats = dataset::utils::calculate_statistics(&dataset);
    assert_eq!(stats.num_samples, 200);
    assert_eq!(stats.num_features, 10);
    assert_eq!(stats.feature_stats.len(), 10);

    // Check that all features have statistics
    for feature_stat in &stats.feature_stats {
        assert!(feature_stat.min_value.is_finite());
        assert!(feature_stat.max_value.is_finite());
        assert!(feature_stat.mean_value.is_finite());
        assert!(feature_stat.std_dev.is_finite());
        assert!(feature_stat.std_dev >= 0.0);
    }
}

#[test]
fn test_regression_with_missing_values() {
    let features = create_test_features_with_missing(100, 5, 0.1);
    let labels = create_test_labels_regression(&features);

    let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();

    let stats = dataset::utils::calculate_statistics(&dataset);
    assert_eq!(stats.num_samples, 100);
    assert_eq!(stats.num_features, 5);

    // Check that missing values are handled
    let total_missing: usize = stats.missing_counts.iter().sum();
    assert!(total_missing > 0);
    assert!(stats.sparsity > 0.0);
}

#[test]
fn test_regression_csv_loading() {
    let temp_dir = TempDir::new().unwrap();
    let csv_path = temp_dir.path().join("regression_test.csv");

    let (features, labels) = create_test_data!(regression, 50, 4);
    let weights = Some(create_test_weights(50));

    create_test_csv(&csv_path, &features, &labels, weights.as_ref(), None).unwrap();

    let config = DatasetConfig::new()
        .with_target_column("target")
        .with_weight_column("weight")
        .with_max_bin(256);

    let dataset = DatasetFactory::from_csv(&csv_path, config);
    assert!(dataset.is_ok());

    let dataset = dataset.unwrap();
    assert_eq!(dataset.num_data(), 50);
    assert_eq!(dataset.num_features(), 4);
    assert!(dataset.has_weights());
}

#[test]
fn test_regression_model_placeholder() {
    let (features, labels) = create_test_data!(regression, 30, 3);
    let dataset = Dataset::new(features.clone(), labels, None, None, None, None).unwrap();

    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.1)
        .num_iterations(10)
        .build()
        .unwrap();

    let mut regressor = LGBMRegressor::new(config);

    // Test that fit returns NotImplemented error
    let fit_result = regressor.fit(&dataset);
    assert!(fit_result.is_err());

    // Test that predict returns NotImplemented error
    let predict_result = regressor.predict(&features);
    assert!(predict_result.is_err());
}

#[test]
fn test_regression_feature_types() {
    let (features, labels) = create_test_data!(regression, 100, 6);
    let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();

    let feature_types = dataset::utils::detect_feature_types(&dataset.features().to_owned());
    assert_eq!(feature_types.len(), 6);

    // Most features should be detected as numerical for regression
    let numerical_count = feature_types
        .iter()
        .filter(|&ft| *ft == FeatureType::Numerical)
        .count();
    assert!(numerical_count > 0);
}

#[test]
fn test_regression_large_dataset() {
    let (features, labels) = create_test_data!(regression, 1000, 20);
    let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();

    assert_eq!(dataset.num_data(), 1000);
    assert_eq!(dataset.num_features(), 20);

    let stats = dataset::utils::calculate_statistics(&dataset);
    assert_eq!(stats.num_samples, 1000);
    assert_eq!(stats.num_features, 20);
}

#[test]
fn test_regression_edge_cases() {
    // Test with single feature
    let (features, labels) = create_test_data!(regression, 10, 1);
    let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();

    assert_eq!(dataset.num_features(), 1);

    // Test with many features
    let (features, labels) = create_test_data!(regression, 50, 100);
    let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();

    assert_eq!(dataset.num_features(), 100);

    // Test validation
    let validation_result = dataset::utils::validate_dataset(&dataset);
    assert!(validation_result.is_valid);
}

#[test]
fn test_regression_config_validation() {
    // Test valid configuration
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.1)
        .num_iterations(100)
        .build()
        .unwrap();

    assert!(config.validate().is_ok());

    // Test invalid learning rate
    let mut invalid_config = config.clone();
    invalid_config.learning_rate = -0.1;
    assert!(invalid_config.validate().is_err());

    // Test invalid number of iterations
    let mut invalid_config2 = config.clone();
    invalid_config2.num_iterations = 0;
    assert!(invalid_config2.validate().is_err());
}

#[test]
fn test_regression_serialization() {
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.05)
        .num_iterations(200)
        .num_leaves(31)
        .lambda_l1(0.1)
        .lambda_l2(0.2)
        .build()
        .unwrap();

    // Test JSON serialization
    let json_str = serde_json::to_string(&config).unwrap();
    assert!(!json_str.is_empty());

    let deserialized: Config = serde_json::from_str(&json_str).unwrap();
    assert_eq!(deserialized.objective, ObjectiveType::Regression);
    assert_eq!(deserialized.learning_rate, 0.05);
    assert_eq!(deserialized.num_iterations, 200);
    assert_eq!(deserialized.num_leaves, 31);
    assert_eq!(deserialized.lambda_l1, 0.1);
    assert_eq!(deserialized.lambda_l2, 0.2);
}

#[test]
fn test_regression_memory_efficiency() {
    use std::time::Instant;

    // Test that large datasets can be created efficiently
    let start = Instant::now();
    let (features, labels) = create_test_data!(regression, 5000, 50);
    let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();
    let duration = start.elapsed();

    assert_eq!(dataset.num_data(), 5000);
    assert_eq!(dataset.num_features(), 50);
    assert!(duration.as_secs() < 5); // Should be fast

    // Test memory usage
    let memory_usage = dataset.memory_usage();
    assert!(memory_usage > 0);
}

#[test]
fn test_regression_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let (features, labels) = create_test_data!(regression, 200, 10);
    let dataset = Arc::new(Dataset::new(features, labels, None, None, None, None).unwrap());

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let dataset = Arc::clone(&dataset);
            thread::spawn(move || {
                let stats = dataset::utils::calculate_statistics(&dataset);
                assert_eq!(stats.num_samples, 200);
                assert_eq!(stats.num_features, 10);
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}
