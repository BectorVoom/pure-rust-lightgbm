//! Debug test for split finding issue in GBDT training
//! 
//! This test helps diagnose why all predictions remain constant
//! and training loss increases instead of decreasing.

use lightgbm_rust::*;
use ndarray::{Array1, Array2};

fn main() -> lightgbm_rust::Result<()> {
    // Initialize logger with debug level
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Debug)
        .init();

    println!("=== Debugging GBDT Split Finding Issue ===");

    // Create a simple synthetic dataset with clear patterns
    let num_samples = 100;
    let num_features = 2;
    
    // Generate features with clear separable patterns
    let mut features = Array2::<f32>::zeros((num_samples, num_features));
    let mut labels = Array1::<f32>::zeros(num_samples);
    
    for i in 0..num_samples {
        let x1 = (i as f32 / num_samples as f32) * 10.0; // 0 to 10
        let x2 = ((i * 7) % 37) as f32 / 10.0; // Some variation
        
        features[[i, 0]] = x1;
        features[[i, 1]] = x2;
        
        // Simple linear relationship: y = 2*x1 + x2 + noise
        labels[i] = 2.0 * x1 + x2 + (i as f32 % 3.0 - 1.0); // Small noise
    }
    
    println!("Created dataset: {} samples, {} features", num_samples, num_features);
    println!("Feature ranges: x1=[{:.2}, {:.2}], x2=[{:.2}, {:.2}]", 
             features.column(0).iter().fold(f32::INFINITY, |a, &b| a.min(b)),
             features.column(0).iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
             features.column(1).iter().fold(f32::INFINITY, |a, &b| a.min(b)),
             features.column(1).iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    println!("Label range: [{:.2}, {:.2}]", 
             labels.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
             labels.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));

    // Create dataset
    let dataset_config = dataset::DatasetConfig::new()
        .with_max_bin(32);

    let dataset = dataset::DatasetFactory::from_arrays(
        features, labels, None, dataset_config
    )?;

    println!("Dataset created successfully");

    // Create GBDT configuration with very permissive settings
    let mut config = config::Config::new();
    config.objective = ObjectiveType::Regression;
    config.num_iterations = 5; // Just a few iterations for debugging
    config.learning_rate = 0.1;
    config.max_depth = 3;
    config.num_leaves = 8;
    config.min_data_in_leaf = 1; // Very small to allow splits
    config.min_gain_to_split = 0.0; // No minimum gain requirement
    config.lambda_l2 = 0.0; // No L2 regularization
    config.feature_fraction = 1.0; // Use all features
    config.bagging_fraction = 1.0; // Use all samples

    println!("Config: max_depth={}, num_leaves={}, min_data_in_leaf={}, min_gain_to_split={}", 
             config.max_depth, config.num_leaves, config.min_data_in_leaf, config.min_gain_to_split);

    // Create and train GBDT
    let mut gbdt = boosting::GBDT::new(config, dataset)?;
    
    println!("\n=== Starting GBDT Training ===");
    gbdt.train()?;

    println!("\n=== Training Complete ===");
    
    // Test predictions
    let test_features = Array2::from_shape_vec((5, 2), vec![
        0.0, 0.0,
        2.5, 1.0,
        5.0, 2.0,
        7.5, 3.0,
        10.0, 4.0,
    ]).map_err(|e| LightGBMError::data_loading(format!("Failed to create test features: {}", e)))?;

    let predictions = gbdt.predict(&test_features)?;
    println!("Test predictions: {:?}", predictions);
    
    // Check if predictions vary
    let min_pred = predictions.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_pred = predictions.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let pred_range = max_pred - min_pred;
    
    println!("Prediction range: [{:.6}, {:.6}] (variation: {:.6})", min_pred, max_pred, pred_range);
    
    if pred_range < 1e-6 {
        println!("❌ ISSUE CONFIRMED: All predictions are essentially the same!");
        println!("This indicates that the trees are not learning meaningful splits.");
    } else {
        println!("✅ Predictions show variation - split finding appears to be working.");
    }

    Ok(())
}