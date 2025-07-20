use lightgbm_rust::dataset::{DatasetConfig, loader::PolarsLoader};
use lightgbm_rust::core::error::Result;
use lightgbm_rust::core::types::ImportanceType;
use lightgbm_rust::{LGBMRegressor, ConfigBuilder};
use std::fs;

fn main() -> Result<()> {
    println!("Testing Feature Importance Calculation...");
    
    // Create a test CSV with clearly identifiable features
    let temp_path = std::env::temp_dir().join("test_feature_importance.csv");
    let csv_content = r#"feature_0,feature_1,feature_2,target
1.0,2.0,3.0,10.0
2.0,4.0,6.0,20.0
3.0,6.0,9.0,30.0
4.0,8.0,12.0,40.0
5.0,10.0,15.0,50.0
6.0,12.0,18.0,60.0
7.0,14.0,21.0,70.0
8.0,16.0,24.0,80.0
9.0,18.0,27.0,90.0
10.0,20.0,30.0,100.0"#;
    
    fs::write(&temp_path, csv_content)?;
    
    println!("Created test CSV with 3 features and strong linear relationship");
    
    // Load dataset using PolarsLoader
    let dataset_config = DatasetConfig::new()
        .with_target_column("target");
    let loader = PolarsLoader::new(dataset_config)?;
    let dataset = loader.load_csv(&temp_path)?;
    
    println!("Dataset loaded: {} samples, {} features", dataset.num_data(), dataset.num_features());
    
    // Create and train a regressor
    let config = ConfigBuilder::new()
        .num_iterations(10)
        .learning_rate(0.1)
        .num_leaves(31)
        .build()?;
    
    let mut regressor = LGBMRegressor::new(config);
    
    println!("Training model...");
    regressor.fit(&dataset)?;
    
    println!("Model trained successfully!");
    
    // Test Split-based importance
    println!("\n🧪 Testing Split-based Feature Importance:");
    match regressor.feature_importance(ImportanceType::Split) {
        Ok(importance) => {
            println!("Split importance shape: {:?}", importance.dim());
            println!("Split importance values: {:?}", importance.as_slice().unwrap());
            
            // Check if we have importance values for all features
            if importance.len() == dataset.num_features() {
                println!("✅ Correct number of importance values");
            } else {
                println!("❌ Wrong number of importance values");
            }
            
            // Check if importance values are non-negative
            let all_non_negative = importance.iter().all(|&x| x >= 0.0);
            if all_non_negative {
                println!("✅ All importance values are non-negative");
            } else {
                println!("❌ Some importance values are negative");
            }
        }
        Err(e) => {
            println!("❌ Error calculating split importance: {}", e);
        }
    }
    
    // Test Gain-based importance
    println!("\n🧪 Testing Gain-based Feature Importance:");
    match regressor.feature_importance(ImportanceType::Gain) {
        Ok(importance) => {
            println!("Gain importance shape: {:?}", importance.dim());
            println!("Gain importance values: {:?}", importance.as_slice().unwrap());
            
            // Check if we have importance values for all features
            if importance.len() == dataset.num_features() {
                println!("✅ Correct number of importance values");
            } else {
                println!("❌ Wrong number of importance values");
            }
            
            // Check if importance values are non-negative
            let all_non_negative = importance.iter().all(|&x| x >= 0.0);
            if all_non_negative {
                println!("✅ All importance values are non-negative");
            } else {
                println!("❌ Some importance values are negative");
            }
        }
        Err(e) => {
            println!("❌ Error calculating gain importance: {}", e);
        }
    }
    
    // Test with untrained model
    println!("\n🧪 Testing Feature Importance with Untrained Model:");
    let untrained_regressor = LGBMRegressor::new(ConfigBuilder::new().build()?);
    
    match untrained_regressor.feature_importance(ImportanceType::Split) {
        Ok(_) => {
            println!("❌ Should have failed with untrained model");
        }
        Err(e) => {
            println!("✅ Correctly failed with untrained model: {}", e);
        }
    }
    
    println!("\n🎉 Feature Importance Testing Complete!");
    
    Ok(())
}