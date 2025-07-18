#[test]
fn debug_csv_feature_count() {
    use std::fs;
    use tempfile::NamedTempFile;
    use lightgbm_rust::*;
    
    // Create a test CSV with exactly 5 features + 1 target + 1 weight = 7 total columns
    let temp_file = NamedTempFile::new().unwrap();
    let csv_content = "feature_0,feature_1,feature_2,feature_3,feature_4,target,weight\n\
                      1.0,2.0,3.0,4.0,5.0,0,1.0\n\
                      6.0,7.0,8.0,9.0,10.0,1,1.0\n\
                      11.0,12.0,13.0,14.0,15.0,0,1.0\n";
    
    fs::write(temp_file.path(), csv_content).unwrap();
    
    // Test with target and weight specified
    let config = DatasetConfig::new()
        .with_target_column("target")
        .with_weight_column("weight");
    
    match DatasetFactory::from_csv(temp_file.path(), config) {
        Ok(dataset) => {
            println!("CSV parsing successful!");
            println!("num_data: {}", dataset.num_data());
            println!("num_features: {}", dataset.num_features());
            println!("Expected num_features: 5");
            
            if let Some(feature_names) = dataset.feature_names() {
                println!("Feature names: {:?}", feature_names);
                println!("Feature count from names: {}", feature_names.len());
            }
            
            let features = dataset.features();
            println!("Features shape: {:?}", features.shape());
            
            if dataset.num_features() != 5 {
                panic!("Expected 5 features, got {}", dataset.num_features());
            }
        }
        Err(e) => {
            panic!("CSV loading failed: {}", e);
        }
    }
}