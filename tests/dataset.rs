//! Dataset-specific integration tests.

use lightgbm_rust::*;
use ndarray::{Array1, Array2};
use tempfile::TempDir;

mod common;
use common::*;

#[test]
fn test_dataset_creation_from_arrays() {
    let features = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ],
    )
    .unwrap();

    let labels = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0]);
    let weights = Some(Array1::from_vec(vec![1.0, 2.0, 1.0, 2.0, 1.0]));

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

    assert_eq!(dataset.num_data(), 5);
    assert_eq!(dataset.num_features(), 3);
    assert!(dataset.has_weights());
}

#[test]
fn test_dataset_with_feature_names() {
    let features = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let labels = Array1::from_vec(vec![0.0, 1.0, 0.0]);
    let feature_names = Some(vec!["feature_a".to_string(), "feature_b".to_string()]);

    let dataset = Dataset::new(features, labels, None, None, feature_names, None).unwrap();

    assert_eq!(dataset.num_data(), 3);
    assert_eq!(dataset.num_features(), 2);

    // Test feature names
    assert_eq!(dataset.feature_name(0), Some("feature_a"));
    assert_eq!(dataset.feature_name(1), Some("feature_b"));
    assert_eq!(dataset.feature_name(2), None);
}

#[test]
fn test_dataset_dimension_validation() {
    let features = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let labels = Array1::from_vec(vec![0.0, 1.0]); // Wrong size

    let dataset = Dataset::new(features, labels, None, None, None, None);

    assert!(dataset.is_err());
}

#[test]
fn test_dataset_with_weights_validation() {
    let features = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let labels = Array1::from_vec(vec![0.0, 1.0, 0.0]);
    let weights = Some(Array1::from_vec(vec![1.0, 2.0])); // Wrong size

    let dataset = Dataset::new(features, labels, weights, None, None, None);

    assert!(dataset.is_err());
}

#[test]
fn test_dataset_statistics_calculation() {
    let features = Array2::from_shape_vec(
        (6, 3),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0,
        ],
    )
    .unwrap();

    let labels = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();

    let stats = dataset::utils::calculate_statistics(&dataset);

    assert_eq!(stats.num_samples, 6);
    assert_eq!(stats.num_features, 3);
    assert_eq!(stats.feature_stats.len(), 3);

    // Check first feature statistics
    let feature_0_stats = &stats.feature_stats[0];
    assert_eq!(feature_0_stats.min_value, 1.0);
    assert_eq!(feature_0_stats.max_value, 16.0);
    assert_eq!(feature_0_stats.missing_count, 0);
    assert!(feature_0_stats.std_dev > 0.0);
}

#[test]
fn test_dataset_with_missing_values() {
    let features = Array2::from_shape_vec(
        (4, 2),
        vec![1.0, 2.0, f32::NAN, 4.0, 5.0, f32::NAN, 7.0, 8.0],
    )
    .unwrap();

    let labels = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);

    let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();

    let stats = dataset::utils::calculate_statistics(&dataset);

    assert_eq!(stats.num_samples, 4);
    assert_eq!(stats.num_features, 2);

    // Check missing value counts
    assert_eq!(stats.missing_counts[0], 1); // First feature has 1 missing
    assert_eq!(stats.missing_counts[1], 1); // Second feature has 1 missing

    // Check sparsity
    assert!(stats.sparsity > 0.0);
    assert!(stats.sparsity < 1.0);
}

#[test]
fn test_dataset_validation_pass() {
    let features = Array2::from_shape_vec((50, 5), (0..250).map(|i| i as f32).collect()).unwrap();
    let labels = Array1::from_vec((0..50).map(|i| (i % 2) as f32).collect());

    let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();

    let validation_result = dataset::utils::validate_dataset(&dataset);

    assert!(validation_result.is_valid);
    assert!(validation_result.errors.is_empty());
}

#[test]
fn test_dataset_validation_fail_too_few_samples() {
    let features = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ],
    )
    .unwrap();
    let labels = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0]);

    let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();

    let validation_result = dataset::utils::validate_dataset(&dataset);

    assert!(!validation_result.is_valid);
    assert!(!validation_result.errors.is_empty());
    assert!(validation_result
        .errors
        .iter()
        .any(|e| e.contains("fewer than 10 samples")));
}

#[test]
fn test_dataset_validation_warnings() {
    // Create dataset with high sparsity
    let features = Array2::from_shape_vec(
        (20, 5),
        (0..100)
            .map(|i| if i % 10 == 0 { i as f32 } else { f32::NAN })
            .collect(),
    )
    .unwrap();
    let labels = Array1::from_vec((0..20).map(|i| (i % 2) as f32).collect());

    let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();

    let validation_result = dataset::utils::validate_dataset(&dataset);

    assert!(validation_result.is_valid); // Should still be valid
    assert!(!validation_result.warnings.is_empty());
    assert!(validation_result
        .warnings
        .iter()
        .any(|w| w.contains("sparsity")));
    assert!(!validation_result.suggestions.is_empty());
}

#[test]
fn test_dataset_factory_from_arrays() {
    let features = Array2::from_shape_vec((10, 3), (0..30).map(|i| i as f32).collect()).unwrap();
    let labels = Array1::from_vec((0..10).map(|i| (i % 2) as f32).collect());
    let weights = Some(Array1::from_vec(
        (0..10).map(|i| 1.0 + i as f32 / 10.0).collect(),
    ));

    let config = DatasetConfig::new()
        .with_max_bin(256)
        .with_categorical_features(vec![0, 1]);

    let dataset = DatasetFactory::from_arrays(features, labels, weights, config);

    assert!(dataset.is_ok());
    let dataset = dataset.unwrap();

    assert_eq!(dataset.num_data(), 10);
    assert_eq!(dataset.num_features(), 3);
    assert!(dataset.has_weights());
}

#[test]
fn test_dataset_factory_csv_loading() {
    let temp_dir = TempDir::new().unwrap();
    let csv_path = temp_dir.path().join("test.csv");

    // Create test CSV
    let features = Array2::from_shape_vec((8, 3), (0..24).map(|i| i as f32).collect()).unwrap();
    let labels = Array1::from_vec((0..8).map(|i| (i % 2) as f32).collect());
    let feature_names = Some(vec!["f1".to_string(), "f2".to_string(), "f3".to_string()]);

    create_test_csv(
        &csv_path,
        &features,
        &labels,
        None,
        feature_names.as_deref(),
    )
    .unwrap();

    let config = DatasetConfig::new()
        .with_target_column("target")
        .with_max_bin(256);

    let dataset = DatasetFactory::from_csv(&csv_path, config);

    assert!(dataset.is_ok());
    let dataset = dataset.unwrap();

    assert_eq!(dataset.num_data(), 8);
    assert_eq!(dataset.num_features(), 3);
}

#[test]
fn test_dataset_config_validation() {
    // Valid config
    let config = DatasetConfig::new()
        .with_max_bin(256)
        .with_categorical_features(vec![0, 1]);

    assert!(config.validate().is_ok());

    // Invalid max_bin (too small)
    let mut invalid_config = config.clone();
    invalid_config.max_bin = 1;
    assert!(invalid_config.validate().is_err());

    // Invalid max_bin (too large)
    let mut invalid_config2 = config.clone();
    invalid_config2.max_bin = 100000;
    assert!(invalid_config2.validate().is_err());

    // Invalid min_data_per_bin
    let mut invalid_config3 = config.clone();
    invalid_config3.min_data_per_bin = 0;
    assert!(invalid_config3.validate().is_err());

    // Conflicting force options
    let mut invalid_config4 = config.clone();
    invalid_config4.force_col_wise = true;
    invalid_config4.force_row_wise = true;
    assert!(invalid_config4.validate().is_err());
}

#[test]
fn test_dataset_config_builder() {
    let config = DatasetConfig::new()
        .with_target_column("target")
        .with_feature_columns(vec!["f1".to_string(), "f2".to_string()])
        .with_weight_column("weight")
        .with_max_bin(512)
        .with_categorical_features(vec![0, 1])
        .with_two_round(true)
        .with_memory_limit(1024);

    assert_eq!(config.target_column, Some("target".to_string()));
    assert_eq!(
        config.feature_columns,
        Some(vec!["f1".to_string(), "f2".to_string()])
    );
    assert_eq!(config.weight_column, Some("weight".to_string()));
    assert_eq!(config.max_bin, 512);
    assert_eq!(config.categorical_features, Some(vec![0, 1]));
    assert!(config.two_round);
    assert_eq!(config.memory_limit_mb, Some(1024));
}

#[test]
fn test_feature_type_detection() {
    // Create mixed features
    let features = Array2::from_shape_vec(
        (10, 4),
        vec![
            // Numerical columns (0, 1)
            1.5, 2.3, 0.0, 1.0, 2.7, 3.1, 0.0, 1.0, 3.2, 4.8, 0.0, 2.0, 4.1, 5.2, 1.0, 2.0, 5.9,
            6.7, 1.0, 3.0, 6.3, 7.1, 1.0, 3.0, 7.8, 8.9, 2.0, 4.0, 8.2, 9.4, 2.0, 4.0, 9.6, 10.1,
            2.0, 5.0, 10.3, 11.8, 3.0, 5.0,
        ],
    )
    .unwrap();

    let feature_types = dataset::utils::detect_feature_types(&features);

    assert_eq!(feature_types.len(), 4);

    // First two columns should be numerical (non-integer values)
    assert_eq!(feature_types[0], FeatureType::Numerical);
    assert_eq!(feature_types[1], FeatureType::Numerical);

    // Last two columns should be categorical (few unique integer values)
    assert_eq!(feature_types[2], FeatureType::Categorical);
    assert_eq!(feature_types[3], FeatureType::Categorical);
}

#[test]
fn test_dataset_memory_usage() {
    let sizes = vec![(100, 10), (1000, 10), (100, 100), (1000, 100)];

    for (num_samples, num_features) in sizes {
        let features = Array2::zeros((num_samples, num_features));
        let labels = Array1::zeros(num_samples);

        let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();

        let memory_usage = dataset.memory_usage();

        // Memory usage should be proportional to data size
        assert!(memory_usage > 0);

        // Basic sanity check: memory should be at least the size of the arrays
        let expected_min_memory = (num_samples * num_features * 4) + (num_samples * 4); // f32 is 4 bytes
        assert!(memory_usage >= expected_min_memory);
    }
}

#[test]
fn test_dataset_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let features =
        Array2::from_shape_vec((100, 10), (0..1000).map(|i| i as f32).collect()).unwrap();
    let labels = Array1::from_vec((0..100).map(|i| (i % 2) as f32).collect());

    let dataset = Arc::new(Dataset::new(features, labels, None, None, None, None).unwrap());

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let dataset = Arc::clone(&dataset);
            thread::spawn(move || {
                // Concurrent access to dataset properties
                let num_data = dataset.num_data();
                let num_features = dataset.num_features();
                let has_weights = dataset.has_weights();
                let memory_usage = dataset.memory_usage();

                // Concurrent statistics calculation
                let stats = dataset::utils::calculate_statistics(&dataset);

                // Concurrent validation
                let validation = dataset::utils::validate_dataset(&dataset);

                (
                    num_data,
                    num_features,
                    has_weights,
                    memory_usage,
                    stats,
                    validation,
                )
            })
        })
        .collect();

    // Wait for all threads and check results
    for handle in handles {
        let (num_data, num_features, has_weights, memory_usage, stats, validation) =
            handle.join().unwrap();

        assert_eq!(num_data, 100);
        assert_eq!(num_features, 10);
        assert!(!has_weights);
        assert!(memory_usage > 0);
        assert_eq!(stats.num_samples, 100);
        assert_eq!(stats.num_features, 10);
        assert!(validation.is_valid);
    }
}

#[test]
fn test_dataset_edge_cases() {
    // Test with single sample
    let features = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
    let labels = Array1::from_vec(vec![1.0]);

    let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();

    assert_eq!(dataset.num_data(), 1);
    assert_eq!(dataset.num_features(), 3);

    // Test with single feature
    let features = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let labels = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0]);

    let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();

    assert_eq!(dataset.num_data(), 5);
    assert_eq!(dataset.num_features(), 1);

    // Test with all missing values in one feature
    let features =
        Array2::from_shape_vec((3, 2), vec![1.0, f32::NAN, 2.0, f32::NAN, 3.0, f32::NAN]).unwrap();
    let labels = Array1::from_vec(vec![0.0, 1.0, 0.0]);

    let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();

    let stats = dataset::utils::calculate_statistics(&dataset);
    assert_eq!(stats.missing_counts[1], 3); // All values missing in second feature
}

#[test]
fn test_dataset_serialization() {
    let config = DatasetConfig::new()
        .with_target_column("target")
        .with_max_bin(512)
        .with_categorical_features(vec![0, 1]);

    // Test JSON serialization
    let json_str = serde_json::to_string(&config).unwrap();
    assert!(!json_str.is_empty());

    let deserialized: DatasetConfig = serde_json::from_str(&json_str).unwrap();
    assert_eq!(deserialized.target_column, config.target_column);
    assert_eq!(deserialized.max_bin, config.max_bin);
    assert_eq!(
        deserialized.categorical_features,
        config.categorical_features
    );
}

#[test]
fn test_dataset_error_handling() {
    // Test empty arrays
    let empty_features = Array2::zeros((0, 0));
    let empty_labels = Array1::zeros(0);

    let dataset = Dataset::new(empty_features, empty_labels, None, None, None, None);

    // Should succeed with empty dataset
    assert!(dataset.is_ok());

    // Test non-finite values
    let features = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, f32::INFINITY, 4.0]).unwrap();
    let labels = Array1::from_vec(vec![0.0, 1.0]);

    let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();

    // Should handle infinite values gracefully
    let stats = dataset::utils::calculate_statistics(&dataset);
    assert_eq!(stats.num_samples, 2);
    assert_eq!(stats.num_features, 2);
}
