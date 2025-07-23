//! Direct comparison test between Rust GBM and Python LightGBM
//! 
//! This test runs the same datasets through both implementations and compares
//! the prediction outputs to verify they are within 10^-10 tolerance.

use lightgbm_rust::*;
use ndarray::{Array1, Array2};
use serde_json;
use std::collections::HashMap;

fn create_test_datasets() -> Result<HashMap<&'static str, (Array2<f32>, Array1<f32>)>> {
    let mut datasets = HashMap::new();
    
    // Dataset 1: Simple linear relationship (same as original test)
    let features1 = Array2::from_shape_vec(
        (20, 3),
        vec![
            1.0, 2.0, 0.5,  2.0, 1.0, 1.0,  3.0, 0.5, 2.0,  0.5, 3.0, 0.2,
            2.5, 1.5, 1.2,  1.5, 2.5, 0.8,  3.2, 0.8, 1.8,  0.8, 2.8, 0.5,
            2.0, 2.0, 1.0,  1.0, 1.0, 1.5,  3.5, 1.2, 2.2,  1.2, 3.2, 0.3,
            2.8, 1.8, 1.5,  1.8, 2.2, 1.1,  3.0, 1.0, 2.0,  1.0, 3.0, 0.8,
            2.2, 2.3, 1.3,  2.3, 1.7, 0.9,  3.3, 0.7, 1.9,  0.7, 2.7, 0.4,
        ]
    ).map_err(|e| LightGBMError::dataset(format!("Failed to create feature array: {}", e)))?;
    
    let labels1 = Array1::from_vec(
        (0..20).map(|i| {
            let x1 = features1[[i, 0]];
            let x2 = features1[[i, 1]];
            let x3 = features1[[i, 2]];
            2.0 * x1 + 3.0 * x2 - x3 + (i as f32 * 0.01)
        }).collect()
    );
    
    datasets.insert("linear", (features1, labels1));
    
    // Dataset 2: Different scale values
    let features2 = Array2::from_shape_vec(
        (20, 3),
        vec![
            10.0, 20.0, 5.0,  20.0, 10.0, 10.0,  30.0, 5.0, 20.0,  5.0, 30.0, 2.0,
            25.0, 15.0, 12.0,  15.0, 25.0, 8.0,  32.0, 8.0, 18.0,  8.0, 28.0, 5.0,
            20.0, 20.0, 10.0,  10.0, 10.0, 15.0,  35.0, 12.0, 22.0,  12.0, 32.0, 3.0,
            28.0, 18.0, 15.0,  18.0, 22.0, 11.0,  30.0, 10.0, 20.0,  10.0, 30.0, 8.0,
            22.0, 23.0, 13.0,  23.0, 17.0, 9.0,  33.0, 7.0, 19.0,  7.0, 27.0, 4.0,
        ]
    ).map_err(|e| LightGBMError::dataset(format!("Failed to create feature array: {}", e)))?;
    
    let labels2 = Array1::from_vec(
        (0..20).map(|i| {
            let x1 = features2[[i, 0]];
            let x2 = features2[[i, 1]];
            let x3 = features2[[i, 2]];
            0.2 * x1 + 0.3 * x2 - 0.1 * x3 + (i as f32 * 0.001)
        }).collect()
    );
    
    datasets.insert("scaled", (features2, labels2));
    
    // Dataset 3: Small values near zero
    let features3 = Array2::from_shape_vec(
        (20, 3),
        vec![
            0.1, 0.2, 0.05,  0.2, 0.1, 0.1,  0.3, 0.05, 0.2,  0.05, 0.3, 0.02,
            0.25, 0.15, 0.12,  0.15, 0.25, 0.08,  0.32, 0.08, 0.18,  0.08, 0.28, 0.05,
            0.2, 0.2, 0.1,  0.1, 0.1, 0.15,  0.35, 0.12, 0.22,  0.12, 0.32, 0.03,
            0.28, 0.18, 0.15,  0.18, 0.22, 0.11,  0.3, 0.1, 0.2,  0.1, 0.3, 0.08,
            0.22, 0.23, 0.13,  0.23, 0.17, 0.09,  0.33, 0.07, 0.19,  0.07, 0.27, 0.04,
        ]
    ).map_err(|e| LightGBMError::dataset(format!("Failed to create feature array: {}", e)))?;
    
    let labels3 = Array1::from_vec(
        (0..20).map(|i| {
            let x1 = features3[[i, 0]];
            let x2 = features3[[i, 1]];
            let x3 = features3[[i, 2]];
            2.0 * x1 + 3.0 * x2 - x3 + (i as f32 * 0.0001)
        }).collect()
    );
    
    datasets.insert("small_values", (features3, labels3));
    
    Ok(datasets)
}

fn train_rust_model(features: &Array2<f32>, labels: &Array1<f32>) -> Result<Vec<f32>> {
    // Create dataset
    let dataset = Dataset::new(features.clone(), labels.clone(), None, None, None, None)?;
    
    // Configure model with same parameters as Python
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Regression)
        .num_iterations(10)
        .learning_rate(0.1)
        .num_leaves(7)
        .min_data_in_leaf(2)
        .lambda_l2(1.0)
        .build()?;
    
    // Create and train GBDT model
    let mut gbdt = GBDT::new(config, dataset)?;
    gbdt.train()?;
    
    // Get predictions on training data
    let training_predictions = gbdt.predict(features)?;
    
    // Test on specific test cases (same as Python)
    let test_features = Array2::from_shape_vec(
        (3, 3),
        vec![
            1.5, 2.5, 1.0,   // Test case 1
            2.0, 1.0, 0.5,   // Test case 2  
            0.5, 3.5, 2.0,   // Test case 3
        ]
    ).map_err(|e| LightGBMError::dataset(format!("Failed to create test feature array: {}", e)))?;
    
    let test_predictions = gbdt.predict(&test_features)?;
    
    // Combine training and test predictions for comparison
    let mut all_predictions = training_predictions.to_vec();
    all_predictions.extend(test_predictions.to_vec());
    
    Ok(all_predictions)
}

fn main() -> Result<()> {
    // Initialize the library
    lightgbm_rust::init()?;
    
    println!("Direct Comparison: Rust GBM vs Python LightGBM");
    println!("================================================");
    
    // Create test datasets
    let datasets = create_test_datasets()?;
    
    // Python results from the previous run (we'll store these for comparison)
    let python_results = serde_json::json!({
        "linear": {
            "training_predictions": [8.345406940300016, 6.686353572493509, 7.334650353874479, 9.117180895043166, 8.477641583205978, 7.838102267371894, 7.976336618390043, 7.59789626531162, 7.976336618390043, 6.686353572493509, 8.707876535659288, 8.1179209396434, 8.477641583205978, 7.59789626531162, 7.976336618390043, 7.838102267371894, 8.345875465793104, 7.59789626531162, 8.117180895043166, 7.838102267371894],
            "test_predictions": [9.279181739703443, 6.686353572493509, 9.152445346613726]
        },
        "scaled": {
            "training_predictions": [8.297019947866598, 6.617893820433392, 7.248541568347387, 9.147563633002304, 8.430265501002456, 7.781222748396607, 7.918977542098728, 7.5502423607297975, 7.918977542098728, 6.617893820433392, 8.659510279101816, 8.06980718905367, 8.430265501002456, 7.5502423607297975, 7.918977542098728, 7.781222748396607, 8.298253159814644, 7.5502423607297975, 8.069807189053671, 7.781222748396607],
            "test_predictions": [6.617893820433392, 6.617893820433392, 6.617893820433392]
        },
        "small_values": {
            "training_predictions": [0.8297019720760485, 0.6617893812557063, 0.7248541527241468, 0.9147563622900845, 0.8430265509686705, 0.7781222757696063, 0.7918977524478174, 0.7550242359042168, 0.7918977524478174, 0.6617893812557063, 0.8659510288581276, 0.8069807190167904, 0.8430265509686705, 0.7550242359042168, 0.7918977524478174, 0.7781222757696063, 0.8298253141295982, 0.7550242359042168, 0.8069807190167904, 0.7781222757696063],
            "test_predictions": [0.9406194731800092, 0.9406194731800092, 0.9406194731800092]
        }
    });
    
    let tolerance = 1e-10f32;
    let mut all_within_tolerance = true;
    
    // Test each dataset
    for (dataset_name, (features, labels)) in &datasets {
        println!("\nTesting {} dataset:", dataset_name);
        println!("Features shape: {:?}", features.dim());
        println!("Labels range: [{:.6}, {:.6}]", 
                 labels.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                 labels.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
        
        // Train Rust model
        let rust_predictions = train_rust_model(features, labels)?;
        
        // Get corresponding Python results
        if let Some(python_data) = python_results.get(dataset_name) {
            let python_train = python_data["training_predictions"].as_array().unwrap();
            let python_test = python_data["test_predictions"].as_array().unwrap();
            
            // Compare training predictions (first 20 values)
            println!("Comparing training predictions:");
            for i in 0..20.min(rust_predictions.len()) {
                let rust_val = rust_predictions[i];
                let python_val = python_train[i].as_f64().unwrap() as f32;
                let diff = (rust_val - python_val).abs();
                
                if i < 3 {  // Show first 3 comparisons
                    println!("  Sample {}: Rust={:.6}, Python={:.6}, diff={:.2e}", 
                             i, rust_val, python_val, diff);
                }
                
                if diff > tolerance {
                    println!("  ❌ Sample {} exceeds tolerance: diff = {:.2e}", i, diff);
                    all_within_tolerance = false;
                }
            }
            
            // Compare test predictions (last 3 values)
            println!("Comparing test predictions:");
            for i in 0..3 {
                let rust_idx = 20 + i;
                if rust_idx < rust_predictions.len() {
                    let rust_val = rust_predictions[rust_idx];
                    let python_val = python_test[i].as_f64().unwrap() as f32;
                    let diff = (rust_val - python_val).abs();
                    
                    println!("  Test {}: Rust={:.6}, Python={:.6}, diff={:.2e}", 
                             i, rust_val, python_val, diff);
                    
                    if diff > tolerance {
                        println!("  ❌ Test {} exceeds tolerance: diff = {:.2e}", i, diff);
                        all_within_tolerance = false;
                    } else {
                        println!("  ✅ Test {} within tolerance", i);
                    }
                }
            }
        }
    }
    
    println!("\n{}", "=".repeat(50));
    if all_within_tolerance {
        println!("✅ All predictions are within 10^-10 tolerance!");
    } else {
        println!("❌ Some predictions exceed 10^-10 tolerance");
    }
    
    println!("Comparison completed successfully!");
    Ok(())
}