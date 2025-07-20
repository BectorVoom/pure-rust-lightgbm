//! Complete workflow demonstration for Pure Rust LightGBM
//! 
//! This test demonstrates:
//! 1. Creating a CSV dataset for binary classification
//! 2. Loading the dataset from CSV file  
//! 3. Training a binary classification model
//! 4. Saving the trained model to a .bin file
//! 5. Loading the model back from the .bin file
//! 6. Executing prediction tests and reporting results

use lightgbm_rust::*;
use ndarray::{Array1, Array2};
use std::io::Write;

fn main() -> Result<()> {
    // Initialize the library
    lightgbm_rust::init()?;
    
    println!("=== Pure Rust LightGBM Complete Workflow Demo ===\n");
    
    // Step 1: Create a CSV dataset for binary classification
    println!("Step 1: Creating CSV dataset for binary classification...");
    let csv_data = create_binary_classification_csv()?;
    println!("✓ Created dataset with {} samples", csv_data.num_samples);
    
    // Step 2: Load the dataset from CSV file
    println!("\nStep 2: Loading dataset from CSV file...");
    let dataset = load_dataset_from_csv(&csv_data.file_path)?;
    println!("✓ Loaded dataset: {} samples, {} features", 
             dataset.num_data(), dataset.num_features());
    
    // Step 3: Train a binary classification model
    println!("\nStep 3: Training binary classification model...");
    let trained_model = train_binary_classifier(dataset)?;
    println!("✓ Training completed successfully");
    
    // Step 4: Save the trained model to .bin file
    println!("\nStep 4: Saving trained model to .bin file...");
    let model_path = save_model_to_file(&trained_model)?;
    println!("✓ Model saved to: {}", model_path.display());
    
    // Step 5: Load the model back from .bin file
    println!("\nStep 5: Loading model back from .bin file...");
    let loaded_model = load_model_from_file(&model_path)?;
    println!("✓ Model loaded successfully");
    
    // Step 6: Execute prediction tests and report results
    println!("\nStep 6: Executing prediction tests...");
    run_prediction_tests(&loaded_model, &csv_data.test_features, &csv_data.test_labels)?;
    
    println!("\n=== Workflow completed successfully! ===");
    Ok(())
}

struct BinaryClassificationData {
    file_path: std::path::PathBuf,
    test_features: Array2<f32>,
    test_labels: Array1<f32>,
    num_samples: usize,
}

fn create_binary_classification_csv() -> Result<BinaryClassificationData> {
    use rand::{Rng, SeedableRng};
    
    // Create deterministic random data
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    
    // Generate synthetic binary classification dataset
    let n_samples = 1000;
    let n_features = 4;
    let mut features = Vec::new();
    let mut labels = Vec::new();
    
    // Generate features and create linearly separable classes
    for _ in 0..n_samples {
        let x1: f32 = rng.gen_range(-2.0..2.0);
        let x2: f32 = rng.gen_range(-2.0..2.0);
        let x3: f32 = rng.gen_range(-1.0..1.0);
        let x4: f32 = rng.gen_range(-1.0..1.0);
        
        // Create a decision boundary: class depends on linear combination
        let decision_value = 0.5 * x1 + 0.3 * x2 + 0.1 * x3 - 0.2 * x4;
        let noise: f32 = rng.gen_range(-0.1..0.1);
        let label = if decision_value + noise > 0.0 { 1.0 } else { 0.0 };
        
        features.push(vec![x1, x2, x3, x4]);
        labels.push(label);
    }
    
    // Split into train/test
    let train_size = (n_samples as f32 * 0.8) as usize;
    let train_features = &features[..train_size];
    let train_labels = &labels[..train_size];
    let test_features = &features[train_size..];
    let test_labels = &labels[train_size..];
    
    // Create temporary CSV file
    let temp_file = std::env::temp_dir().join("binary_classification_data.csv");
    let mut file = std::fs::File::create(&temp_file)
        .map_err(|e| LightGBMError::data_loading(format!("Failed to create temp file: {}", e)))?;
    
    // Write CSV header
    writeln!(file, "feature1,feature2,feature3,feature4,target")
        .map_err(|e| LightGBMError::data_loading(format!("Failed to write CSV header: {}", e)))?;
    
    // Write training data to CSV
    for (features, &label) in train_features.iter().zip(train_labels.iter()) {
        writeln!(file, "{},{},{},{},{}", 
                features[0], features[1], features[2], features[3], label)
            .map_err(|e| LightGBMError::data_loading(format!("Failed to write CSV data: {}", e)))?;
    }
    
    file.flush()
        .map_err(|e| LightGBMError::data_loading(format!("Failed to flush CSV file: {}", e)))?;
    
    // Convert test data to arrays
    let test_features_flat: Vec<f32> = test_features.iter().flatten().copied().collect();
    let test_features_array = Array2::from_shape_vec(
        (test_features.len(), n_features), 
        test_features_flat
    ).map_err(|e| LightGBMError::data_loading(format!("Failed to create test features array: {}", e)))?;
    
    let test_labels_array = Array1::from_vec(test_labels.to_vec());
    
    Ok(BinaryClassificationData {
        file_path: temp_file,
        test_features: test_features_array,
        test_labels: test_labels_array,
        num_samples: train_size,
    })
}

fn load_dataset_from_csv(csv_path: &std::path::Path) -> Result<Dataset> {
    // Use the polars loader to load CSV
    use dataset::DatasetConfig;
    use polars::datatypes::PlSmallStr;
    
    let config = DatasetConfig::new()
        .with_target_column(PlSmallStr::from_static("target"));
    
    // Create Polars CSV loader
    let loader = dataset::loader::PolarsLoader::new(config)?;
    loader.load_csv(csv_path)
}

fn train_binary_classifier(dataset: Dataset) -> Result<boosting::GBDT> {
    // Create binary classification configuration using ConfigBuilder
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .num_iterations(50)
        .learning_rate(0.1)
        .num_leaves(15)
        .max_depth(5)
        .min_data_in_leaf(20)
        .build()?;
    
    // Create and train GBDT model
    let mut model = boosting::GBDT::new(config, dataset)?;
    model.train()?;
    
    println!("   - Number of iterations: {}", model.num_iterations());
    let history = model.training_history();
    if let Some(last_loss) = history.train_loss.last() {
        println!("   - Training loss: {:.6}", last_loss);
    }
    
    Ok(model)
}

fn save_model_to_file(model: &boosting::GBDT) -> Result<std::path::PathBuf> {
    use serde_json;
    
    // Create temporary file for model
    let model_path = std::env::temp_dir().join("trained_model.bin");
    
    // For now, save as JSON since the IO module is disabled
    // In a complete implementation, we would use the bincode serialization
    let model_json = serde_json::to_string_pretty(&model)
        .map_err(|e| LightGBMError::serialization(format!("Failed to serialize model: {}", e)))?;
    
    std::fs::write(&model_path, model_json)
        .map_err(|e| LightGBMError::serialization(format!("Failed to write model file: {}", e)))?;
    
    Ok(model_path)
}

fn load_model_from_file(model_path: &std::path::Path) -> Result<boosting::GBDT> {
    use serde_json;
    
    // Load JSON model file
    let model_json = std::fs::read_to_string(model_path)
        .map_err(|e| LightGBMError::serialization(format!("Failed to read model file: {}", e)))?;
    
    let model: boosting::GBDT = serde_json::from_str(&model_json)
        .map_err(|e| LightGBMError::serialization(format!("Failed to deserialize model: {}", e)))?;
    
    Ok(model)
}

fn run_prediction_tests(
    model: &boosting::GBDT, 
    test_features: &Array2<f32>, 
    test_labels: &Array1<f32>
) -> Result<()> {
    println!("   Running predictions on {} test samples...", test_features.nrows());
    
    // Make predictions
    let predictions = model.predict(test_features)?;
    
    // Convert predictions to binary class (threshold at 0.5)
    let predicted_classes: Vec<f32> = predictions.iter()
        .map(|&p| if p > 0.5 { 1.0 } else { 0.0 })
        .collect();
    
    // Calculate accuracy
    let correct_predictions = predicted_classes.iter()
        .zip(test_labels.iter())
        .filter(|(&pred, &true_label)| pred == true_label)
        .count();
    
    let accuracy = correct_predictions as f64 / test_labels.len() as f64;
    
    // Calculate additional metrics
    let mut tp = 0; // true positives
    let mut fp = 0; // false positives  
    let mut tn = 0; // true negatives
    let mut fn_ = 0; // false negatives
    
    for (&pred, &true_label) in predicted_classes.iter().zip(test_labels.iter()) {
        match (pred as i32, true_label as i32) {
            (1, 1) => tp += 1,
            (1, 0) => fp += 1,
            (0, 0) => tn += 1,
            (0, 1) => fn_ += 1,
            _ => {}
        }
    }
    
    let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
    let recall = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
    let f1_score = if precision + recall > 0.0 { 
        2.0 * precision * recall / (precision + recall) 
    } else { 
        0.0 
    };
    
    // Report results
    println!("\n   === Prediction Results ===");
    println!("   Test samples: {}", test_labels.len());
    println!("   Correct predictions: {}", correct_predictions);
    println!("   Accuracy: {:.4} ({:.2}%)", accuracy, accuracy * 100.0);
    println!("   Precision: {:.4}", precision);
    println!("   Recall: {:.4}", recall);
    println!("   F1-Score: {:.4}", f1_score);
    
    println!("\n   Confusion Matrix:");
    println!("              Predicted");
    println!("              0    1");
    println!("   Actual  0  {}   {}", tn, fp);
    println!("           1  {}   {}", fn_, tp);
    
    // Show some example predictions
    println!("\n   Sample predictions:");
    for i in 0..std::cmp::min(10, predictions.len()) {
        println!("   Sample {}: Features=[{:.2}, {:.2}, {:.2}, {:.2}] -> Prediction={:.4}, True={:.0}", 
                i + 1,
                test_features[[i, 0]], test_features[[i, 1]], 
                test_features[[i, 2]], test_features[[i, 3]],
                predictions[i], test_labels[i]);
    }
    
    if accuracy > 0.7 {
        println!("\n   ✓ Model performance is good (accuracy > 70%)");
    } else {
        println!("\n   ⚠ Model performance could be improved (accuracy < 70%)");
    }
    
    Ok(())
}