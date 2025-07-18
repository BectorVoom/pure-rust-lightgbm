use std::fs;
use std::io::Write;
use tempfile::NamedTempFile;

fn main() {
    // Create a test CSV file with 5 features + 1 target + 1 weight
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "feature_0,feature_1,feature_2,feature_3,feature_4,target,weight").unwrap();
    writeln!(temp_file, "1.0,2.0,3.0,4.0,5.0,0.0,1.0").unwrap();
    writeln!(temp_file, "6.0,7.0,8.0,9.0,10.0,1.0,1.0").unwrap();
    writeln!(temp_file, "11.0,12.0,13.0,14.0,15.0,0.0,1.0").unwrap();
    
    // Try to load with the CSV loader
    use lightgbm_rust::dataset::{DatasetConfig, DatasetFactory};
    
    let config = DatasetConfig::new()
        .with_target_column("target")
        .with_weight_column("weight");
    
    match DatasetFactory::from_csv(temp_file.path(), config) {
        Ok(dataset) => {
            println!("Dataset loaded successfully!");
            println!("Number of data points: {}", dataset.num_data());
            println!("Number of features: {}", dataset.num_features());
            println!("Expected features: 5");
            
            if dataset.num_features() == 5 {
                println!("✅ Bug is fixed!");
            } else {
                println!("❌ Bug still exists - got {} features instead of 5", dataset.num_features());
            }
        }
        Err(e) => {
            println!("Failed to load dataset: {}", e);
        }
    }
}