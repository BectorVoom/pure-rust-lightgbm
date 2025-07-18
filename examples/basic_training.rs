//! Basic training example for pure Rust LightGBM.
//!
//! This example demonstrates how to use the LightGBM library for basic
//! regression and classification tasks. It shows the complete workflow
//! from data loading to model training and prediction.
//!
//! Run with: `cargo run --example basic_training`

use lightgbm_rust::*;
use ndarray::{Array1, Array2};
use std::fs;
use std::path::Path;

fn main() -> Result<()> {
    // Initialize the library
    lightgbm_rust::init()?;
    
    println!("Pure Rust LightGBM - Basic Training Example");
    println!("==========================================");
    
    // Display capabilities
    let caps = lightgbm_rust::capabilities();
    println!("Library capabilities: {}", caps.summary());
    println!();
    
    // Run regression example
    run_regression_example()?;
    
    // Run binary classification example
    run_binary_classification_example()?;
    
    // Run multiclass classification example
    run_multiclass_classification_example()?;
    
    // Run CSV loading example
    run_csv_loading_example()?;
    
    println!("All examples completed successfully!");
    
    Ok(())
}

fn run_regression_example() -> Result<()> {
    println!("1. Regression Example");
    println!("--------------------");
    
    // Create synthetic regression data
    let features = Array2::from_shape_vec(
        (10, 3),
        vec![
            1.0, 2.0, 3.0,
            2.0, 3.0, 4.0,
            3.0, 4.0, 5.0,
            4.0, 5.0, 6.0,
            5.0, 6.0, 7.0,
            6.0, 7.0, 8.0,
            7.0, 8.0, 9.0,
            8.0, 9.0, 10.0,
            9.0, 10.0, 11.0,
            10.0, 11.0, 12.0,
        ]
    ).unwrap();
    
    let labels = Array1::from_vec(vec![
        6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0
    ]);
    
    println!("  Features shape: {:?}", features.dim());
    println!("  Labels shape: {:?}", labels.dim());
    
    // Create dataset
    let dataset = Dataset::new(
        features.clone(),
        labels.clone(),
        None,
        None,
        Some(vec!["feature_1".to_string(), "feature_2".to_string(), "feature_3".to_string()]),
        None,
    )?;
    
    println!("  Dataset created with {} samples and {} features", 
             dataset.num_data(), dataset.num_features());
    
    // Validate dataset
    let validation_result = dataset::utils::validate_dataset(&dataset);
    println!("  Dataset validation: {}", if validation_result.is_valid { "PASSED" } else { "FAILED" });
    
    // Calculate statistics
    let stats = dataset::utils::calculate_statistics(&dataset);
    println!("  Dataset statistics:");
    println!("    - Samples: {}", stats.num_samples);
    println!("    - Features: {}", stats.num_features);
    println!("    - Sparsity: {:.4}", stats.sparsity);
    
    // Create regression configuration
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .learning_rate(0.1)
        .num_iterations(100)
        .num_leaves(31)
        .lambda_l1(0.0)
        .lambda_l2(0.0)
        .min_data_in_leaf(1)
        .build()?;
    
    println!("  Configuration created:");
    println!("    - Objective: {:?}", config.objective);
    println!("    - Learning rate: {}", config.learning_rate);
    println!("    - Iterations: {}", config.num_iterations);
    println!("    - Leaves: {}", config.num_leaves);
    
    // Create and train model
    let mut regressor = LGBMRegressor::new(config);
    
    // Note: Training is not implemented yet, so this will return an error
    match regressor.fit(&dataset) {
        Ok(_) => {
            println!("  Model training: COMPLETED");
            
            // Make predictions
            match regressor.predict(&features) {
                Ok(predictions) => {
                    println!("  Predictions: {:?}", predictions);
                }
                Err(e) => {
                    println!("  Prediction error: {}", e);
                }
            }
        }
        Err(e) => {
            println!("  Training not implemented yet: {}", e);
        }
    }
    
    println!();
    Ok(())
}

fn run_binary_classification_example() -> Result<()> {
    println!("2. Binary Classification Example");
    println!("-------------------------------");
    
    // Create synthetic binary classification data
    let features = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 2.0,
            2.0, 3.0,
            3.0, 1.0,
            4.0, 2.0,
            5.0, 6.0,
            6.0, 7.0,
            7.0, 5.0,
            8.0, 6.0,
        ]
    ).unwrap();
    
    let labels = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    
    println!("  Features shape: {:?}", features.dim());
    println!("  Labels shape: {:?}", labels.dim());
    
    // Create dataset
    let dataset = Dataset::new(
        features.clone(),
        labels.clone(),
        None,
        None,
        Some(vec!["feature_1".to_string(), "feature_2".to_string()]),
        None,
    )?;
    
    println!("  Dataset created with {} samples and {} features", 
             dataset.num_data(), dataset.num_features());
    
    // Validate dataset
    let validation_result = dataset::utils::validate_dataset(&dataset);
    println!("  Dataset validation: {}", if validation_result.is_valid { "PASSED" } else { "FAILED" });
    
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
    println!("  Class distribution: Class 0: {}, Class 1: {}", class_counts[0], class_counts[1]);
    
    // Create binary classification configuration
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .learning_rate(0.1)
        .num_iterations(100)
        .num_leaves(31)
        .build()?;
    
    println!("  Configuration created:");
    println!("    - Objective: {:?}", config.objective);
    println!("    - Learning rate: {}", config.learning_rate);
    println!("    - Iterations: {}", config.num_iterations);
    
    // Create and train model
    let mut classifier = LGBMClassifier::new(config);
    
    // Note: Training is not implemented yet, so this will return an error
    match classifier.fit(&dataset) {
        Ok(_) => {
            println!("  Model training: COMPLETED");
            
            // Make predictions
            match classifier.predict(&features) {
                Ok(predictions) => {
                    println!("  Class predictions: {:?}", predictions);
                }
                Err(e) => {
                    println!("  Prediction error: {}", e);
                }
            }
            
            // Make probability predictions
            match classifier.predict_proba(&features) {
                Ok(probabilities) => {
                    println!("  Probability predictions: {:?}", probabilities);
                }
                Err(e) => {
                    println!("  Probability prediction error: {}", e);
                }
            }
        }
        Err(e) => {
            println!("  Training not implemented yet: {}", e);
        }
    }
    
    println!();
    Ok(())
}

fn run_multiclass_classification_example() -> Result<()> {
    println!("3. Multiclass Classification Example");
    println!("-----------------------------------");
    
    // Create synthetic multiclass classification data
    let features = Array2::from_shape_vec(
        (9, 2),
        vec![
            1.0, 1.0,
            2.0, 1.0,
            1.0, 2.0,
            4.0, 4.0,
            5.0, 4.0,
            4.0, 5.0,
            8.0, 8.0,
            9.0, 8.0,
            8.0, 9.0,
        ]
    ).unwrap();
    
    let labels = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    
    println!("  Features shape: {:?}", features.dim());
    println!("  Labels shape: {:?}", labels.dim());
    
    // Create dataset
    let dataset = Dataset::new(
        features.clone(),
        labels.clone(),
        None,
        None,
        Some(vec!["feature_1".to_string(), "feature_2".to_string()]),
        None,
    )?;
    
    println!("  Dataset created with {} samples and {} features", 
             dataset.num_data(), dataset.num_features());
    
    // Validate dataset
    let validation_result = dataset::utils::validate_dataset(&dataset);
    println!("  Dataset validation: {}", if validation_result.is_valid { "PASSED" } else { "FAILED" });
    
    // Check class distribution
    let mut class_counts = [0, 0, 0];
    for i in 0..dataset.num_data() {
        let label = dataset.label(i).unwrap() as usize;
        if label < 3 {
            class_counts[label] += 1;
        }
    }
    println!("  Class distribution: Class 0: {}, Class 1: {}, Class 2: {}", 
             class_counts[0], class_counts[1], class_counts[2]);
    
    // Create multiclass classification configuration
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Multiclass)
        .num_class(3)
        .learning_rate(0.1)
        .num_iterations(100)
        .num_leaves(31)
        .build()?;
    
    println!("  Configuration created:");
    println!("    - Objective: {:?}", config.objective);
    println!("    - Number of classes: {}", config.num_class);
    println!("    - Learning rate: {}", config.learning_rate);
    println!("    - Iterations: {}", config.num_iterations);
    
    // Create and train model
    let mut classifier = LGBMClassifier::new(config);
    
    // Note: Training is not implemented yet, so this will return an error
    match classifier.fit(&dataset) {
        Ok(_) => {
            println!("  Model training: COMPLETED");
            
            // Make predictions
            match classifier.predict(&features) {
                Ok(predictions) => {
                    println!("  Class predictions: {:?}", predictions);
                }
                Err(e) => {
                    println!("  Prediction error: {}", e);
                }
            }
            
            // Make probability predictions
            match classifier.predict_proba(&features) {
                Ok(probabilities) => {
                    println!("  Probability predictions shape: {:?}", probabilities.dim());
                }
                Err(e) => {
                    println!("  Probability prediction error: {}", e);
                }
            }
        }
        Err(e) => {
            println!("  Training not implemented yet: {}", e);
        }
    }
    
    println!();
    Ok(())
}

fn run_csv_loading_example() -> Result<()> {
    println!("4. CSV Loading Example");
    println!("---------------------");
    
    // Create a temporary CSV file
    let csv_content = r#"feature1,feature2,feature3,target
1.0,2.0,3.0,6.0
2.0,3.0,4.0,9.0
3.0,4.0,5.0,12.0
4.0,5.0,6.0,15.0
5.0,6.0,7.0,18.0
6.0,7.0,8.0,21.0
7.0,8.0,9.0,24.0
8.0,9.0,10.0,27.0"#;
    
    let csv_path = "example_data.csv";
    fs::write(csv_path, csv_content)?;
    
    println!("  Created CSV file: {}", csv_path);
    
    // Create dataset configuration
    let config = DatasetConfig::new()
        .with_target_column("target")
        .with_feature_columns(vec![
            "feature1".to_string(),
            "feature2".to_string(),
            "feature3".to_string(),
        ])
        .with_max_bin(256);
    
    println!("  Dataset configuration:");
    println!("    - Target column: {:?}", config.target_column);
    println!("    - Feature columns: {:?}", config.feature_columns);
    println!("    - Max bins: {}", config.max_bin);
    
    // Load dataset from CSV
    match DatasetFactory::from_csv(csv_path, config) {
        Ok(dataset) => {
            println!("  CSV loading: SUCCESS");
            println!("  Dataset loaded with {} samples and {} features", 
                     dataset.num_data(), dataset.num_features());
            
            // Calculate statistics
            let stats = dataset::utils::calculate_statistics(&dataset);
            println!("  Dataset statistics:");
            println!("    - Samples: {}", stats.num_samples);
            println!("    - Features: {}", stats.num_features);
            println!("    - Memory usage: {} bytes", stats.memory_usage);
            
            // Display feature statistics
            for (i, feature_stat) in stats.feature_stats.iter().enumerate() {
                println!("    - Feature {}: min={:.2}, max={:.2}, mean={:.2}, std={:.2}", 
                         i, feature_stat.min_value, feature_stat.max_value, 
                         feature_stat.mean_value, feature_stat.std_dev);
            }
        }
        Err(e) => {
            println!("  CSV loading error: {}", e);
        }
    }
    
    // Clean up
    if Path::new(csv_path).exists() {
        fs::remove_file(csv_path)?;
        println!("  Cleaned up CSV file");
    }
    
    println!();
    Ok(())
}

// Helper function to demonstrate configuration serialization
fn _demonstrate_config_serialization() -> Result<()> {
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .learning_rate(0.05)
        .num_iterations(200)
        .num_leaves(31)
        .lambda_l1(0.1)
        .lambda_l2(0.2)
        .build()?;
    
    // Serialize to JSON
    let json_str = serde_json::to_string_pretty(&config)?;
    println!("Configuration JSON:");
    println!("{}", json_str);
    
    // Deserialize from JSON
    let _deserialized: Config = serde_json::from_str(&json_str)?;
    println!("Configuration deserialized successfully");
    
    Ok(())
}

// Helper function to demonstrate memory usage
fn _demonstrate_memory_usage() -> Result<()> {
    use std::time::Instant;
    
    println!("Memory Usage Demonstration");
    println!("-------------------------");
    
    let start = Instant::now();
    
    // Create a moderately large dataset
    let num_samples = 1000;
    let num_features = 50;
    
    let features = Array2::zeros((num_samples, num_features));
    let labels = Array1::zeros(num_samples);
    
    let dataset = Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    )?;
    
    let duration = start.elapsed();
    
    println!("Dataset creation time: {:?}", duration);
    println!("Dataset memory usage: {} bytes", dataset.memory_usage());
    
    Ok(())
}