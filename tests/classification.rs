//! Classification-specific integration tests.

use lightgbm_rust::*;
use ndarray::{Array1, Array2};
use tempfile::TempDir;
use std::fs;

mod common;
use common::*;

#[test]
fn test_binary_classification_configuration() {
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .learning_rate(0.1)
        .num_iterations(100)
        .num_leaves(31)
        .lambda_l1(0.1)
        .lambda_l2(0.1)
        .min_data_in_leaf(20)
        .build();
    
    assert!(config.is_ok());
    let config = config.unwrap();
    assert_eq!(config.objective, ObjectiveType::Binary);
    assert_eq!(config.learning_rate, 0.1);
    assert_eq!(config.num_iterations, 100);
    assert_eq!(config.num_leaves, 31);
    assert_eq!(config.lambda_l1, 0.1);
    assert_eq!(config.lambda_l2, 0.1);
    assert_eq!(config.min_data_in_leaf, 20);
}

#[test]
fn test_multiclass_classification_configuration() {
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Multiclass)
        .num_class(5)
        .learning_rate(0.1)
        .num_iterations(100)
        .num_leaves(31)
        .build();
    
    assert!(config.is_ok());
    let config = config.unwrap();
    assert_eq!(config.objective, ObjectiveType::Multiclass);
    assert_eq!(config.num_class, 5);
    assert_eq!(config.learning_rate, 0.1);
    assert_eq!(config.num_iterations, 100);
    assert_eq!(config.num_leaves, 31);
}

#[test]
fn test_binary_classification_dataset() {
    let (features, labels) = create_test_data!(binary, 100, 5);
    let weights = Some(create_test_weights(100));
    
    let dataset = Dataset::new(
        features,
        labels,
        weights,
        None,
        None,
        None,
    );
    
    assert!(dataset.is_ok());
    let dataset = dataset.unwrap();
    assert_eq!(dataset.num_data(), 100);
    assert_eq!(dataset.num_features(), 5);
    assert!(dataset.has_weights());
    
    // Verify labels are binary
    for i in 0..dataset.num_data() {
        let label = dataset.label(i).unwrap();
        assert!(label == 0.0 || label == 1.0);
    }
}

#[test]
fn test_multiclass_classification_dataset() {
    let (features, labels) = create_test_data!(multiclass, 100, 5, 3);
    let weights = Some(create_test_weights(100));
    
    let dataset = Dataset::new(
        features,
        labels,
        weights,
        None,
        None,
        None,
    );
    
    assert!(dataset.is_ok());
    let dataset = dataset.unwrap();
    assert_eq!(dataset.num_data(), 100);
    assert_eq!(dataset.num_features(), 5);
    assert!(dataset.has_weights());
    
    // Verify labels are in range [0, 3)
    for i in 0..dataset.num_data() {
        let label = dataset.label(i).unwrap();
        assert!(label >= 0.0 && label < 3.0);
    }
}

#[test]
fn test_classification_dataset_validation() {
    let (features, labels) = create_test_data!(binary, 50, 3);
    let dataset = Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
    let validation_result = dataset::utils::validate_dataset(&dataset);
    assert!(validation_result.is_valid);
    assert!(validation_result.errors.is_empty());
}

#[test]
fn test_classification_statistics() {
    let (features, labels) = create_test_data!(binary, 200, 10);
    let dataset = Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
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
fn test_classification_with_categorical_features() {
    let (features, categorical_indices) = create_test_mixed_features(100, 3, 2, 5);
    let labels = create_test_labels_binary(&features);
    
    let dataset = Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
    assert_eq!(dataset.num_data(), 100);
    assert_eq!(dataset.num_features(), 5);
    
    // Test dataset config with categorical features
    let config = DatasetConfig::new()
        .with_categorical_features(categorical_indices)
        .with_max_bin(256);
    
    assert!(config.validate().is_ok());
}

#[test]
fn test_classification_csv_loading() {
    let temp_dir = TempDir::new().unwrap();
    let csv_path = temp_dir.path().join("classification_test.csv");
    
    let (features, labels) = create_test_data!(binary, 50, 4);
    let weights = Some(create_test_weights(50));
    
    create_test_csv(
        &csv_path,
        &features,
        &labels,
        weights.as_ref(),
        None,
    ).unwrap();
    
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
fn test_binary_classifier_placeholder() {
    let (features, labels) = create_test_data!(binary, 30, 3);
    let dataset = Dataset::new(
        features.clone(),
        labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .learning_rate(0.1)
        .num_iterations(10)
        .build()
        .unwrap();
    
    let mut classifier = LGBMClassifier::new(config);
    
    // Test that fit returns NotImplemented error
    let fit_result = classifier.fit(&dataset);
    assert!(fit_result.is_err());
    
    // Test that predict returns NotImplemented error
    let predict_result = classifier.predict(&features);
    assert!(predict_result.is_err());
    
    // Test that predict_proba returns NotImplemented error
    let predict_proba_result = classifier.predict_proba(&features);
    assert!(predict_proba_result.is_err());
}

#[test]
fn test_multiclass_classifier_placeholder() {
    let (features, labels) = create_test_data!(multiclass, 30, 3, 4);
    let dataset = Dataset::new(
        features.clone(),
        labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Multiclass)
        .num_class(4)
        .learning_rate(0.1)
        .num_iterations(10)
        .build()
        .unwrap();
    
    let mut classifier = LGBMClassifier::new(config);
    
    // Test that fit returns NotImplemented error
    let fit_result = classifier.fit(&dataset);
    assert!(fit_result.is_err());
    
    // Test that predict returns NotImplemented error
    let predict_result = classifier.predict(&features);
    assert!(predict_result.is_err());
    
    // Test that predict_proba returns NotImplemented error
    let predict_proba_result = classifier.predict_proba(&features);
    assert!(predict_proba_result.is_err());
}

#[test]
fn test_classification_feature_types() {
    let (features, labels) = create_test_data!(binary, 100, 6);
    let dataset = Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
    let feature_types = dataset::utils::detect_feature_types(&dataset.features().to_owned());
    assert_eq!(feature_types.len(), 6);
    
    // Most features should be detected as numerical
    let numerical_count = feature_types.iter()
        .filter(|&ft| *ft == FeatureType::Numerical)
        .count();
    assert!(numerical_count > 0);
}

#[test]
fn test_classification_with_missing_values() {
    let features = create_test_features_with_missing(100, 5, 0.1);
    let labels = create_test_labels_binary(&features);
    
    let dataset = Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
    let stats = dataset::utils::calculate_statistics(&dataset);
    assert_eq!(stats.num_samples, 100);
    assert_eq!(stats.num_features, 5);
    
    // Check that missing values are handled
    let total_missing: usize = stats.missing_counts.iter().sum();
    assert!(total_missing > 0);
    assert!(stats.sparsity > 0.0);
}

#[test]
fn test_classification_config_validation() {
    // Test valid binary configuration
    let binary_config = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .learning_rate(0.1)
        .num_iterations(100)
        .build()
        .unwrap();
    
    assert!(binary_config.validate().is_ok());
    
    // Test valid multiclass configuration
    let multiclass_config = ConfigBuilder::new()
        .objective(ObjectiveType::Multiclass)
        .num_class(3)
        .learning_rate(0.1)
        .num_iterations(100)
        .build()
        .unwrap();
    
    assert!(multiclass_config.validate().is_ok());
    
    // Test invalid multiclass configuration (missing num_class)
    let invalid_multiclass = ConfigBuilder::new()
        .objective(ObjectiveType::Multiclass)
        .learning_rate(0.1)
        .num_iterations(100)
        .build();
    
    assert!(invalid_multiclass.is_err());
}

#[test]
fn test_classification_serialization() {
    let binary_config = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .learning_rate(0.05)
        .num_iterations(200)
        .num_leaves(31)
        .lambda_l1(0.1)
        .lambda_l2(0.2)
        .build()
        .unwrap();
    
    // Test JSON serialization for binary
    let json_str = serde_json::to_string(&binary_config).unwrap();
    assert!(!json_str.is_empty());
    
    let deserialized: Config = serde_json::from_str(&json_str).unwrap();
    assert_eq!(deserialized.objective, ObjectiveType::Binary);
    assert_eq!(deserialized.learning_rate, 0.05);
    assert_eq!(deserialized.num_iterations, 200);
    
    // Test multiclass serialization
    let multiclass_config = ConfigBuilder::new()
        .objective(ObjectiveType::Multiclass)
        .num_class(5)
        .learning_rate(0.1)
        .num_iterations(100)
        .build()
        .unwrap();
    
    let json_str = serde_json::to_string(&multiclass_config).unwrap();
    let deserialized: Config = serde_json::from_str(&json_str).unwrap();
    assert_eq!(deserialized.objective, ObjectiveType::Multiclass);
    assert_eq!(deserialized.num_class, 5);
}

#[test]
fn test_classification_large_dataset() {
    let (features, labels) = create_test_data!(binary, 1000, 20);
    let dataset = Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
    assert_eq!(dataset.num_data(), 1000);
    assert_eq!(dataset.num_features(), 20);
    
    let stats = dataset::utils::calculate_statistics(&dataset);
    assert_eq!(stats.num_samples, 1000);
    assert_eq!(stats.num_features, 20);
    
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
    
    // Should have both classes represented
    assert!(class_counts[0] > 0);
    assert!(class_counts[1] > 0);
}

#[test]
fn test_classification_edge_cases() {
    // Test with minimal binary dataset
    let features = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let labels = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
    
    let dataset = Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    
    assert_eq!(dataset.num_data(), 4);
    assert_eq!(dataset.num_features(), 2);
    
    // Test validation
    let validation_result = dataset::utils::validate_dataset(&dataset);
    assert!(validation_result.is_valid);
}

#[test]
fn test_classification_thread_safety() {
    use std::sync::Arc;
    use std::thread;
    
    let (features, labels) = create_test_data!(binary, 200, 10);
    let dataset = Arc::new(Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    ).unwrap());
    
    let handles: Vec<_> = (0..4).map(|_| {
        let dataset = Arc::clone(&dataset);
        thread::spawn(move || {
            let stats = dataset::utils::calculate_statistics(&dataset);
            assert_eq!(stats.num_samples, 200);
            assert_eq!(stats.num_features, 10);
            
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
        })
    }).collect();
    
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_classification_default_models() {
    let regressor = LGBMRegressor::default();
    assert_eq!(regressor.config().objective, ObjectiveType::Regression);
    
    let classifier = LGBMClassifier::default();
    assert_eq!(classifier.config().objective, ObjectiveType::Binary);
}

#[test]
fn test_classification_memory_efficiency() {
    use std::time::Instant;
    
    // Test that large classification datasets can be created efficiently
    let start = Instant::now();
    let (features, labels) = create_test_data!(binary, 5000, 50);
    let dataset = Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    ).unwrap();
    let duration = start.elapsed();
    
    assert_eq!(dataset.num_data(), 5000);
    assert_eq!(dataset.num_features(), 50);
    assert!(duration.as_secs() < 5); // Should be fast
    
    // Test memory usage
    let memory_usage = dataset.memory_usage();
    assert!(memory_usage > 0);
    
    // Verify classes are represented
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
}