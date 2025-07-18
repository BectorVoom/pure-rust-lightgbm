use lightgbm_rust::*;
use ndarray::{Array1, Array2};
use std::fs;
use tempfile::NamedTempFile;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a test CSV with 5 features + 1 target + 1 weight = 7 total columns
    let temp_file = NamedTempFile::new()?;
    
    // Write CSV content with header: 5 features + target + weight = 7 columns
    let csv_content = "feature_0,feature_1,feature_2,feature_3,feature_4,target,weight\n\
                      1.0,2.0,3.0,4.0,5.0,0,1.0\n\
                      6.0,7.0,8.0,9.0,10.0,1,1.0\n\
                      11.0,12.0,13.0,14.0,15.0,0,1.0\n";
    
    fs::write(temp_file.path(), csv_content)?;
    
    // Configure dataset to use "target" as target column and "weight" as weight column
    let config = DatasetConfig::new()
        .with_target_column("target")
        .with_weight_column("weight");
    
    // Load the dataset
    let dataset = DatasetFactory::from_csv(temp_file.path(), config)?;
    
    println!("CSV has 7 total columns: 5 features + 1 target + 1 weight");
    println!("Dataset num_data: {}", dataset.num_data());
    println!("Dataset num_features: {}", dataset.num_features());
    println!("Expected num_features: 5 (excluding target and weight)");
    
    if dataset.num_features() == 5 {
        println!("✓ PASS: Feature count is correct");
    } else {
        println!("✗ FAIL: Expected 5 features, got {}", dataset.num_features());
    }
    
    Ok(())
}