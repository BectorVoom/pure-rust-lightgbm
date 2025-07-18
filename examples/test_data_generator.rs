//! Test data generator for pure Rust LightGBM.
//!
//! This example generates various types of test datasets for validating
//! the LightGBM implementation across different scenarios and data types.
//!
//! Run with: `cargo run --example test_data_generator`

use lightgbm_rust::*;
use ndarray::{Array1, Array2};
use std::fs;
use std::path::{Path, PathBuf};
use rand::prelude::*;
use rand::distributions::{Distribution, Standard, Uniform};

fn main() -> Result<()> {
    println!("Pure Rust LightGBM - Test Data Generator");
    println!("========================================");
    println!();
    
    let output_dir = Path::new("tests/data");
    
    // Ensure output directory exists
    if !output_dir.exists() {
        fs::create_dir_all(output_dir)?;
    }
    
    // Generate different types of test datasets
    generate_regression_datasets(output_dir)?;
    generate_classification_datasets(output_dir)?;
    generate_special_case_datasets(output_dir)?;
    
    println!("\nAll test datasets generated successfully!");
    println!("Datasets available in: {:?}", output_dir);
    
    Ok(())
}

fn generate_regression_datasets(output_dir: &Path) -> Result<()> {
    println!("Generating regression datasets...");
    
    // Small regression dataset
    generate_simple_regression(output_dir, "simple_regression.csv", 100, 5)?;
    
    // Medium regression dataset with noise
    generate_noisy_regression(output_dir, "noisy_regression.csv", 500, 10, 0.1)?;
    
    // Large regression dataset
    generate_complex_regression(output_dir, "complex_regression.csv", 2000, 20)?;
    
    // Regression with missing values
    generate_regression_with_missing(output_dir, "regression_missing.csv", 300, 8, 0.15)?;
    
    // Regression with categorical features
    generate_mixed_regression(output_dir, "mixed_regression.csv", 400, 6, 3)?;
    
    Ok(())
}

fn generate_classification_datasets(output_dir: &Path) -> Result<()> {
    println!("Generating classification datasets...");
    
    // Binary classification
    generate_binary_classification(output_dir, "binary_classification_large.csv", 800, 12)?;
    
    // Imbalanced binary classification
    generate_imbalanced_binary(output_dir, "imbalanced_binary.csv", 600, 8, 0.2)?;
    
    // Multiclass classification
    generate_multiclass_classification(output_dir, "multiclass_large.csv", 900, 10, 4)?;
    
    // Multiclass with many classes
    generate_multiclass_classification(output_dir, "multiclass_many.csv", 1000, 15, 8)?;
    
    // Classification with categorical features
    generate_mixed_classification(output_dir, "mixed_classification.csv", 700, 8, 4, 3)?;
    
    Ok(())
}

fn generate_special_case_datasets(output_dir: &Path) -> Result<()> {
    println!("Generating special case datasets...");
    
    // Tiny dataset
    generate_tiny_dataset(output_dir, "tiny_dataset.csv", 5, 2)?;
    
    // Single feature dataset
    generate_single_feature(output_dir, "single_feature.csv", 100)?;
    
    // High dimensional dataset
    generate_high_dimensional(output_dir, "high_dimensional.csv", 200, 100)?;
    
    // Dataset with extreme values
    generate_extreme_values(output_dir, "extreme_values.csv", 150, 6)?;
    
    // Dataset with constant features
    generate_constant_features(output_dir, "constant_features.csv", 200, 5, 2)?;
    
    // Highly correlated features
    generate_correlated_features(output_dir, "correlated_features.csv", 300, 8)?;
    
    Ok(())
}

// Regression dataset generators

fn generate_simple_regression(output_dir: &Path, filename: &str, n_samples: usize, n_features: usize) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(42);
    
    let mut features = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);
    
    // Generate features
    for i in 0..n_samples {
        for j in 0..n_features {
            features[[i, j]] = rng.gen::<f32>() * 10.0;
        }
    }
    
    // Generate labels with linear relationship
    for i in 0..n_samples {
        let mut sum = 0.0;
        for j in 0..n_features {
            sum += features[[i, j]] * (j + 1) as f32;
        }
        labels[i] = sum / n_features as f32 + rng.gen::<f32>() * 2.0 - 1.0; // Add small noise
    }
    
    save_regression_csv(output_dir, filename, &features, &labels, None, n_features)?;
    println!("  Generated: {}", filename);
    Ok(())
}

fn generate_noisy_regression(output_dir: &Path, filename: &str, n_samples: usize, n_features: usize, noise_level: f64) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(123);
    let normal = Normal::new(0.0, noise_level).unwrap();
    
    let mut features = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);
    
    // Generate features
    for i in 0..n_samples {
        for j in 0..n_features {
            features[[i, j]] = rng.gen::<f32>() * 20.0 - 10.0; // Range [-10, 10]
        }
    }
    
    // Generate labels with quadratic relationship + noise
    for i in 0..n_samples {
        let mut target = 0.0;
        for j in 0..n_features {
            let x = features[[i, j]];
            target += x * x * 0.1 + x * (j + 1) as f32 * 0.5;
        }
        
        // Add noise
        let noise = normal.sample(&mut rng) as f32;
        labels[i] = target + noise;
    }
    
    save_regression_csv(output_dir, filename, &features, &labels, None, n_features)?;
    println!("  Generated: {} (noise level: {})", filename, noise_level);
    Ok(())
}

fn generate_complex_regression(output_dir: &Path, filename: &str, n_samples: usize, n_features: usize) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(456);
    
    let mut features = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);
    let weights = Array1::from_fn(n_samples, |_| rng.gen::<f32>() * 0.5 + 0.5); // Weights between 0.5 and 1.0
    
    // Generate features with different distributions
    for i in 0..n_samples {
        for j in 0..n_features {
            if j % 3 == 0 {
                // Normal distribution
                features[[i, j]] = Normal::new(0.0, 2.0).unwrap().sample(&mut rng) as f32;
            } else if j % 3 == 1 {
                // Uniform distribution
                features[[i, j]] = rng.gen::<f32>() * 50.0 - 25.0;
            } else {
                // Exponential-like distribution
                features[[i, j]] = (-rng.gen::<f32>().ln()) * 5.0;
            }
        }
    }
    
    // Generate complex target with interactions
    for i in 0..n_samples {
        let mut target = 10.0; // Base value
        
        // Linear terms
        for j in 0..n_features {
            target += features[[i, j]] * (j + 1) as f32 * 0.3;
        }
        
        // Interaction terms
        for j in 0..(n_features - 1) {
            target += features[[i, j]] * features[[i, j + 1]] * 0.1;
        }
        
        // Non-linear terms
        if n_features > 2 {
            target += features[[i, 0]].sin() * 5.0;
            target += (features[[i, 1]] * features[[i, 2]]).cos() * 3.0;
        }
        
        labels[i] = target;
    }
    
    save_regression_csv(output_dir, filename, &features, &labels, Some(&weights), n_features)?;
    println!("  Generated: {} (complex with interactions)", filename);
    Ok(())
}

fn generate_regression_with_missing(output_dir: &Path, filename: &str, n_samples: usize, n_features: usize, missing_rate: f64) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(789);
    
    let mut features = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);
    
    // Generate features
    for i in 0..n_samples {
        for j in 0..n_features {
            if rng.gen::<f64>() < missing_rate {
                features[[i, j]] = f32::NAN; // Missing value
            } else {
                features[[i, j]] = rng.gen::<f32>() * 10.0;
            }
        }
    }
    
    // Generate labels (handle missing values)
    for i in 0..n_samples {
        let mut sum = 0.0;
        let mut count = 0;
        
        for j in 0..n_features {
            if !features[[i, j]].is_nan() {
                sum += features[[i, j]] * (j + 1) as f32;
                count += 1;
            }
        }
        
        labels[i] = if count > 0 { sum / count as f32 } else { 0.0 };
    }
    
    save_regression_csv_with_missing(output_dir, filename, &features, &labels, n_features)?;
    println!("  Generated: {} ({}% missing values)", filename, missing_rate * 100.0);
    Ok(())
}

fn generate_mixed_regression(output_dir: &Path, filename: &str, n_samples: usize, n_numerical: usize, n_categorical: usize) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(101112);
    let n_features = n_numerical + n_categorical;
    
    let mut features = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);
    
    // Generate features
    for i in 0..n_samples {
        // Numerical features
        for j in 0..n_numerical {
            features[[i, j]] = rng.gen::<f32>() * 20.0 - 10.0;
        }
        
        // Categorical features
        for j in n_numerical..n_features {
            features[[i, j]] = rng.gen_range(0..5) as f32; // 5 categories
        }
    }
    
    // Generate labels considering both numerical and categorical
    for i in 0..n_samples {
        let mut target = 0.0;
        
        // Numerical contribution
        for j in 0..n_numerical {
            target += features[[i, j]] * 0.5;
        }
        
        // Categorical contribution
        for j in n_numerical..n_features {
            let cat_value = features[[i, j]] as usize;
            target += match cat_value {
                0 => -2.0,
                1 => -1.0,
                2 => 0.0,
                3 => 1.0,
                4 => 2.0,
                _ => 0.0,
            };
        }
        
        labels[i] = target + rng.gen::<f32>() * 2.0 - 1.0;
    }
    
    save_mixed_regression_csv(output_dir, filename, &features, &labels, n_numerical, n_categorical)?;
    println!("  Generated: {} ({} numerical, {} categorical)", filename, n_numerical, n_categorical);
    Ok(())
}

// Classification dataset generators

fn generate_binary_classification(output_dir: &Path, filename: &str, n_samples: usize, n_features: usize) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(131415);
    
    let mut features = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);
    
    // Generate features
    for i in 0..n_samples {
        for j in 0..n_features {
            features[[i, j]] = Normal::new(0.0, 1.0).unwrap().sample(&mut rng) as f32;
        }
    }
    
    // Generate labels using logistic function
    for i in 0..n_samples {
        let mut linear_combination = 0.0;
        for j in 0..n_features {
            linear_combination += features[[i, j]] * (if j % 2 == 0 { 1.0 } else { -0.5 });
        }
        
        let probability = 1.0 / (1.0 + (-linear_combination as f64).exp());
        labels[i] = if rng.gen::<f32>() < probability { 1.0 } else { 0.0 };
    }
    
    save_classification_csv(output_dir, filename, &features, &labels, None, n_features)?;
    println!("  Generated: {} (binary classification)", filename);
    Ok(())
}

fn generate_imbalanced_binary(output_dir: &Path, filename: &str, n_samples: usize, n_features: usize, minority_ratio: f64) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(161718);
    
    let mut features = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);
    
    let n_minority = (n_samples as f64 * minority_ratio) as usize;
    
    // Generate features and labels
    for i in 0..n_samples {
        let is_minority = i < n_minority;
        
        for j in 0..n_features {
            let mean = if is_minority { 2.0 } else { -1.0 };
            // Simple approximation of normal distribution using Box-Muller transform
            let u1: f32 = rng.gen_range(0.0001..0.9999);
            let u2: f32 = rng.gen_range(0.0001..0.9999);
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            features[[i, j]] = mean + z0;
        }
        
        labels[i] = if is_minority { 1.0 } else { 0.0 };
    }
    
    // Shuffle the data
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);
    
    let mut shuffled_features = Array2::zeros((n_samples, n_features));
    let mut shuffled_labels = Array1::zeros(n_samples);
    
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        for j in 0..n_features {
            shuffled_features[[new_idx, j]] = features[[old_idx, j]];
        }
        shuffled_labels[new_idx] = labels[old_idx];
    }
    
    save_classification_csv(output_dir, filename, &shuffled_features, &shuffled_labels, None, n_features)?;
    println!("  Generated: {} (imbalanced: {:.1}% minority)", filename, minority_ratio * 100.0);
    Ok(())
}

fn generate_multiclass_classification(output_dir: &Path, filename: &str, n_samples: usize, n_features: usize, n_classes: usize) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(192021);
    
    let mut features = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);
    
    // Generate cluster centers
    let mut centers = Array2::zeros((n_classes, n_features));
    for i in 0..n_classes {
        for j in 0..n_features {
            centers[[i, j]] = Normal::new(0.0, 3.0).unwrap().sample(&mut rng) as f32;
        }
    }
    
    // Generate samples around cluster centers
    for i in 0..n_samples {
        let class = i % n_classes;
        labels[i] = class as f32;
        
        for j in 0..n_features {
            let noise = Normal::new(0.0, 1.0).unwrap().sample(&mut rng) as f32;
            features[[i, j]] = centers[[class, j]] + noise;
        }
    }
    
    // Shuffle the data
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);
    
    let mut shuffled_features = Array2::zeros((n_samples, n_features));
    let mut shuffled_labels = Array1::zeros(n_samples);
    
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        for j in 0..n_features {
            shuffled_features[[new_idx, j]] = features[[old_idx, j]];
        }
        shuffled_labels[new_idx] = labels[old_idx];
    }
    
    save_classification_csv(output_dir, filename, &shuffled_features, &shuffled_labels, None, n_features)?;
    println!("  Generated: {} ({} classes)", filename, n_classes);
    Ok(())
}

fn generate_mixed_classification(output_dir: &Path, filename: &str, n_samples: usize, n_numerical: usize, n_categorical: usize, n_classes: usize) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(222324);
    let n_features = n_numerical + n_categorical;
    
    let mut features = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);
    
    // Generate samples
    for i in 0..n_samples {
        let class = i % n_classes;
        labels[i] = class as f32;
        
        // Numerical features
        for j in 0..n_numerical {
            let class_offset = class as f32 * 2.0;
            features[[i, j]] = Normal::new(class_offset as f64, 1.0).unwrap().sample(&mut rng) as f32;
        }
        
        // Categorical features
        for j in n_numerical..n_features {
            // Make categorical features somewhat predictive of class
            let preferred_category = (class + j) % 4;
            features[[i, j]] = if rng.gen::<f32>() < 0.7 {
                preferred_category as f32
            } else {
                rng.gen_range(0..4) as f32
            };
        }
    }
    
    // Shuffle the data
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);
    
    let mut shuffled_features = Array2::zeros((n_samples, n_features));
    let mut shuffled_labels = Array1::zeros(n_samples);
    
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        for j in 0..n_features {
            shuffled_features[[new_idx, j]] = features[[old_idx, j]];
        }
        shuffled_labels[new_idx] = labels[old_idx];
    }
    
    save_mixed_classification_csv(output_dir, filename, &shuffled_features, &shuffled_labels, n_numerical, n_categorical, n_classes)?;
    println!("  Generated: {} ({} classes, {} numerical, {} categorical)", filename, n_classes, n_numerical, n_categorical);
    Ok(())
}

// Special case dataset generators

fn generate_tiny_dataset(output_dir: &Path, filename: &str, n_samples: usize, n_features: usize) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(252627);
    
    let mut features = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);
    
    for i in 0..n_samples {
        for j in 0..n_features {
            features[[i, j]] = rng.gen::<f32>() * 10.0;
        }
        labels[i] = features.row(i).sum();
    }
    
    save_regression_csv(output_dir, filename, &features, &labels, None, n_features)?;
    println!("  Generated: {} (tiny dataset)", filename);
    Ok(())
}

fn generate_single_feature(output_dir: &Path, filename: &str, n_samples: usize) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(282930);
    
    let mut features = Array2::zeros((n_samples, 1));
    let mut labels = Array1::zeros(n_samples);
    
    for i in 0..n_samples {
        features[[i, 0]] = rng.gen::<f32>() * 20.0 - 10.0;
        labels[i] = features[[i, 0]] * 2.0 + rng.gen::<f32>() * 2.0 - 1.0;
    }
    
    save_regression_csv(output_dir, filename, &features, &labels, None, 1)?;
    println!("  Generated: {} (single feature)", filename);
    Ok(())
}

fn generate_high_dimensional(output_dir: &Path, filename: &str, n_samples: usize, n_features: usize) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(313233);
    
    let mut features = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);
    
    // Only first 10 features are truly predictive
    let n_informative = std::cmp::min(10, n_features);
    
    for i in 0..n_samples {
        let mut target = 0.0;
        
        for j in 0..n_features {
            features[[i, j]] = Normal::new(0.0, 1.0).unwrap().sample(&mut rng) as f32;
            
            if j < n_informative {
                target += features[[i, j]] * (j + 1) as f32 * 0.5;
            }
        }
        
        labels[i] = target;
    }
    
    save_regression_csv(output_dir, filename, &features, &labels, None, n_features)?;
    println!("  Generated: {} (high dimensional: {} features, {} informative)", filename, n_features, n_informative);
    Ok(())
}

fn generate_extreme_values(output_dir: &Path, filename: &str, n_samples: usize, n_features: usize) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(343536);
    
    let mut features = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);
    
    for i in 0..n_samples {
        for j in 0..n_features {
            if rng.gen::<f32>() < 0.1 { // 10% extreme values
                features[[i, j]] = if rng.gen::<bool>() { 1e6 } else { -1e6 };
            } else {
                features[[i, j]] = Normal::new(0.0, 1.0).unwrap().sample(&mut rng) as f32;
            }
        }
        
        // Robust target calculation
        let mut sum = 0.0;
        for j in 0..n_features {
            sum += features[[i, j]].clamp(-100.0, 100.0); // Clamp extreme values
        }
        labels[i] = sum / n_features as f32;
    }
    
    save_regression_csv(output_dir, filename, &features, &labels, None, n_features)?;
    println!("  Generated: {} (with extreme values)", filename);
    Ok(())
}

fn generate_constant_features(output_dir: &Path, filename: &str, n_samples: usize, n_features: usize, n_constant: usize) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(373839);
    
    let mut features = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);
    
    for i in 0..n_samples {
        for j in 0..n_features {
            if j < n_constant {
                features[[i, j]] = 5.0; // Constant value
            } else {
                features[[i, j]] = rng.gen::<f32>() * 10.0;
            }
        }
        
        // Target based only on non-constant features
        let mut sum = 0.0;
        for j in n_constant..n_features {
            sum += features[[i, j]];
        }
        labels[i] = sum / (n_features - n_constant) as f32;
    }
    
    save_regression_csv(output_dir, filename, &features, &labels, None, n_features)?;
    println!("  Generated: {} ({} constant features)", filename, n_constant);
    Ok(())
}

fn generate_correlated_features(output_dir: &Path, filename: &str, n_samples: usize, n_features: usize) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(404142);
    
    let mut features = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);
    
    // Generate base features
    for i in 0..n_samples {
        // First feature is independent
        features[[i, 0]] = Normal::new(0.0, 1.0).unwrap().sample(&mut rng) as f32;
        
        // Other features are correlated with the first
        for j in 1..n_features {
            let correlation = 0.8;
            let noise = Normal::new(0.0, (1.0 - correlation * correlation).sqrt()).unwrap().sample(&mut rng) as f32;
            features[[i, j]] = features[[i, 0]] * correlation + noise;
        }
        
        // Target depends mainly on the first feature
        labels[i] = features[[i, 0]] * 2.0 + rng.gen::<f32>() * 0.5;
    }
    
    save_regression_csv(output_dir, filename, &features, &labels, None, n_features)?;
    println!("  Generated: {} (highly correlated features)", filename);
    Ok(())
}

// CSV saving functions

fn save_regression_csv(output_dir: &Path, filename: &str, features: &Array2<f32>, labels: &Array1<f32>, weights: Option<&Array1<f32>>, n_features: usize) -> Result<()> {
    let path = output_dir.join(filename);
    let mut content = String::new();
    
    // Header
    for i in 0..n_features {
        content.push_str(&format!("feature{},", i + 1));
    }
    if weights.is_some() {
        content.push_str("weight,");
    }
    content.push_str("target\n");
    
    // Data
    for i in 0..features.nrows() {
        for j in 0..n_features {
            content.push_str(&format!("{},", features[[i, j]]));
        }
        if let Some(w) = weights {
            content.push_str(&format!("{},", w[i]));
        }
        content.push_str(&format!("{}\n", labels[i]));
    }
    
    fs::write(path, content)?;
    Ok(())
}

fn save_regression_csv_with_missing(output_dir: &Path, filename: &str, features: &Array2<f32>, labels: &Array1<f32>, n_features: usize) -> Result<()> {
    let path = output_dir.join(filename);
    let mut content = String::new();
    
    // Header
    for i in 0..n_features {
        content.push_str(&format!("feature{},", i + 1));
    }
    content.push_str("target\n");
    
    // Data
    for i in 0..features.nrows() {
        for j in 0..n_features {
            if features[[i, j]].is_nan() {
                content.push_str(","); // Empty for missing
            } else {
                content.push_str(&format!("{},", features[[i, j]]));
            }
        }
        content.push_str(&format!("{}\n", labels[i]));
    }
    
    fs::write(path, content)?;
    Ok(())
}

fn save_mixed_regression_csv(output_dir: &Path, filename: &str, features: &Array2<f32>, labels: &Array1<f32>, n_numerical: usize, n_categorical: usize) -> Result<()> {
    let path = output_dir.join(filename);
    let mut content = String::new();
    
    // Header
    for i in 0..n_numerical {
        content.push_str(&format!("numerical{},", i + 1));
    }
    for i in 0..n_categorical {
        content.push_str(&format!("categorical{},", i + 1));
    }
    content.push_str("target\n");
    
    // Data
    for i in 0..features.nrows() {
        for j in 0..features.ncols() {
            content.push_str(&format!("{},", features[[i, j]]));
        }
        content.push_str(&format!("{}\n", labels[i]));
    }
    
    fs::write(path, content)?;
    Ok(())
}

fn save_classification_csv(output_dir: &Path, filename: &str, features: &Array2<f32>, labels: &Array1<f32>, weights: Option<&Array1<f32>>, n_features: usize) -> Result<()> {
    let path = output_dir.join(filename);
    let mut content = String::new();
    
    // Header
    for i in 0..n_features {
        content.push_str(&format!("feature{},", i + 1));
    }
    if weights.is_some() {
        content.push_str("weight,");
    }
    content.push_str("class\n");
    
    // Data
    for i in 0..features.nrows() {
        for j in 0..n_features {
            content.push_str(&format!("{},", features[[i, j]]));
        }
        if let Some(w) = weights {
            content.push_str(&format!("{},", w[i]));
        }
        content.push_str(&format!("{}\n", labels[i] as i32));
    }
    
    fs::write(path, content)?;
    Ok(())
}

fn save_mixed_classification_csv(output_dir: &Path, filename: &str, features: &Array2<f32>, labels: &Array1<f32>, n_numerical: usize, n_categorical: usize, n_classes: usize) -> Result<()> {
    let path = output_dir.join(filename);
    let mut content = String::new();
    
    // Header
    for i in 0..n_numerical {
        content.push_str(&format!("numerical{},", i + 1));
    }
    for i in 0..n_categorical {
        content.push_str(&format!("categorical{},", i + 1));
    }
    content.push_str("class\n");
    
    // Data
    for i in 0..features.nrows() {
        for j in 0..features.ncols() {
            content.push_str(&format!("{},", features[[i, j]]));
        }
        content.push_str(&format!("{}\n", labels[i] as i32));
    }
    
    fs::write(path, content)?;
    Ok(())
}
