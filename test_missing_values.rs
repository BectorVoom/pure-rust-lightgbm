use lightgbm_rust::dataset::{DatasetConfig, DatasetFactory};
use lightgbm_rust::core::error::Result;
use std::fs;

fn main() -> Result<()> {
    // Enable debug logging
    std::env::set_var("RUST_LOG", "debug");
    env_logger::init();
    
    println!("Testing missing value detection...");
    
    // Create a CSV with missing values (empty cells)
    let temp_path = std::env::temp_dir().join("test_missing_values.csv");
    let csv_content = r#"feature_1,feature_2,feature_3,target
1.5,2.0,,3.5
,3.5,1.0,2.0
2.5,,2.5,4.0
3.0,4.0,3.0,1.0
1.0,1.5,1.5,1.5"#;
    
    fs::write(&temp_path, csv_content)?;
    
    println!("Created CSV with missing values (empty cells)");
    
    let config = DatasetConfig::new()
        .with_target_column("target");
    
    let dataset = DatasetFactory::from_csv(&temp_path, config)?;
    
    println!("Dataset loaded successfully!");
    println!("Number of data points: {}", dataset.num_data());
    println!("Number of features: {}", dataset.num_features());
    
    // Check if missing values are detected
    let has_missing = dataset.has_missing_values();
    println!("Has missing values: {}", has_missing);
    
    if has_missing {
        println!("✅ Missing values detected correctly!");
        
        // Get the missing values matrix
        if let Some(missing_matrix) = dataset.missing_values() {
            println!("Missing values matrix shape: {:?}", missing_matrix.dim());
            
            // Count total missing values
            let total_missing = missing_matrix.iter().filter(|&&x| x).count();
            println!("Total missing values: {}", total_missing);
            
            // Print missing values pattern
            for (i, row) in missing_matrix.outer_iter().enumerate() {
                for (j, &is_missing) in row.iter().enumerate() {
                    if is_missing {
                        println!("Missing value at row {}, feature {}", i, j);
                    }
                }
            }
        }
    } else {
        println!("❌ Missing values not detected");
    }
    
    Ok(())
}