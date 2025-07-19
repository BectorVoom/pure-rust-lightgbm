//! Test the core GBDT training algorithm implementation
//!
//! This example creates a simple synthetic dataset and trains a GBDT model
//! to verify that the mathematical formulas from formula.md are working correctly.

use lightgbm_rust::*;
use ndarray::{Array1, Array2};
use std::time::Instant;

fn main() -> Result<()> {
    // Initialize the library
    lightgbm_rust::init()?;
    
    println!("Testing GBDT Training Algorithm");
    println!("==============================");
    
    // Create a simple synthetic regression dataset
    let features = Array2::from_shape_vec(
        (20, 3),
        vec![
            // Feature values designed to have a clear pattern: y = 2*x1 + 3*x2 - x3 + noise
            1.0, 2.0, 0.5,  2.0, 1.0, 1.0,  3.0, 0.5, 2.0,  0.5, 3.0, 0.2,
            2.5, 1.5, 1.2,  1.5, 2.5, 0.8,  3.2, 0.8, 1.8,  0.8, 2.8, 0.5,
            2.0, 2.0, 1.0,  1.0, 1.0, 1.5,  3.5, 1.2, 2.2,  1.2, 3.2, 0.3,
            2.8, 1.8, 1.5,  1.8, 2.2, 1.1,  3.0, 1.0, 2.0,  1.0, 3.0, 0.8,
            2.2, 2.3, 1.3,  2.3, 1.7, 0.9,  3.3, 0.7, 1.9,  0.7, 2.7, 0.4,
        ]
    ).map_err(|e| LightGBMError::dataset(format!("Failed to create feature array: {}", e)))?;
    
    // Calculate target values: y = 2*x1 + 3*x2 - x3 + small_noise
    let labels = Array1::from_vec(
        (0..20).map(|i| {
            let x1 = features[[i, 0]];
            let x2 = features[[i, 1]];
            let x3 = features[[i, 2]];
            2.0 * x1 + 3.0 * x2 - x3 + (i as f32 * 0.01) // Small noise term
        }).collect()
    );
    
    println!("Dataset created: {} samples, {} features", features.nrows(), features.ncols());
    println!("Target range: [{:.2}, {:.2}]", 
             labels.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
             labels.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    
    // Create dataset
    let dataset = Dataset::new(features.clone(), labels.clone(), None, None, None, None)?;
    
    // Configure GBDT model with small learning rate and few iterations for testing
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .num_iterations(10)          // Small number for quick testing
        .learning_rate(0.1)          // Conservative learning rate
        .num_leaves(7)               // Small trees
        .min_data_in_leaf(2)         // Allow small leaves for this tiny dataset
        .lambda_l2(1.0)              // Some regularization
        .build()?;
    
    println!("\nConfiguration:");
    println!("- Objective: {:?}", config.objective);
    println!("- Iterations: {}", config.num_iterations);
    println!("- Learning rate: {}", config.learning_rate);
    println!("- Max leaves: {}", config.num_leaves);
    println!("- L2 regularization: {}", config.lambda_l2);
    
    // Create and train GBDT model
    let mut gbdt = GBDT::new(config, dataset)?;
    
    println!("\nStarting training...");
    let start_time = Instant::now();
    
    gbdt.train()?;
    
    let training_time = start_time.elapsed();
    println!("Training completed in {:.2}ms", training_time.as_secs_f64() * 1000.0);
    
    // Test predictions on training data
    println!("\nTesting predictions on training data:");
    let predictions = gbdt.predict(&features)?;
    
    // Calculate training error
    let mut total_error = 0.0f32;
    let mut max_error = 0.0f32;
    for i in 0..labels.len() {
        let error = (predictions[i] - labels[i]).abs();
        total_error += error;
        max_error = max_error.max(error);
        
        if i < 5 {  // Show first 5 predictions vs actual
            println!("  Sample {}: predicted={:.3}, actual={:.3}, error={:.3}", 
                     i, predictions[i], labels[i], error);
        }
    }
    
    let mean_abs_error = total_error / labels.len() as f32;
    println!("  ...");
    println!("Training Mean Absolute Error: {:.4}", mean_abs_error);
    println!("Training Max Error: {:.4}", max_error);
    
    // Test on a few new samples to verify generalization
    println!("\nTesting on new samples:");
    let test_features = Array2::from_shape_vec(
        (3, 3),
        vec![
            1.5, 2.5, 1.0,   // Expected: 2*1.5 + 3*2.5 - 1.0 = 9.5
            2.0, 1.0, 0.5,   // Expected: 2*2.0 + 3*1.0 - 0.5 = 6.5  
            0.5, 3.5, 2.0,   // Expected: 2*0.5 + 3*3.5 - 2.0 = 9.5
        ]
    ).map_err(|e| LightGBMError::dataset(format!("Failed to create test feature array: {}", e)))?;
    
    let test_predictions = gbdt.predict(&test_features)?;
    let expected_values = vec![9.5, 6.5, 9.5];
    
    for i in 0..3 {
        let prediction_error = (test_predictions[i] - expected_values[i]).abs();
        println!("  Test {}: predicted={:.3}, expected={:.3}, error={:.3}", 
                 i, test_predictions[i], expected_values[i], prediction_error);
    }
    
    // Show model info
    println!("\nModel Information:");
    println!("- Number of trees: {}", gbdt.num_iterations());
    println!("- Training history available: {}", !gbdt.training_history().train_loss.is_empty());
    
    if !gbdt.training_history().train_loss.is_empty() {
        let final_loss = gbdt.training_history().train_loss.last().unwrap();
        let initial_loss = gbdt.training_history().train_loss.first().unwrap();
        println!("- Initial training loss: {:.6}", initial_loss);
        println!("- Final training loss: {:.6}", final_loss);
        println!("- Loss improvement: {:.6}", initial_loss - final_loss);
    }
    
    println!("\n✅ GBDT training algorithm test completed successfully!");
    
    if mean_abs_error < 1.0 {
        println!("✅ Model achieved good accuracy (MAE < 1.0)");
    } else {
        println!("⚠️  Model accuracy could be improved (MAE = {:.4})", mean_abs_error);
    }
    
    Ok(())
}