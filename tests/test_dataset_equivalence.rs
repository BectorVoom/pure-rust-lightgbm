use lightgbm_rust::io::dataset::{DatasetIO, one_feature_per_group, get_conflict_count, mark_used};
use std::fs;

#[test]
fn test_dataset_creation() {
    println!("Testing DatasetIO creation...");
    
    let dataset = DatasetIO::new();
    assert_eq!(dataset.data_filename(), "noname");
    assert_eq!(dataset.num_data(), 0);
    assert!(!dataset.is_finish_load());
    
    println!("DatasetIO creation test passed!");
}

#[test]
fn test_dataset_with_num_data() {
    println!("Testing DatasetIO with num_data...");
    
    let dataset = DatasetIO::with_num_data(100).unwrap();
    assert_eq!(dataset.num_data(), 100);
    assert_eq!(dataset.metadata().num_data, 100);
    
    println!("DatasetIO with num_data test passed!");
}

#[test]
fn test_invalid_num_data() {
    let result = DatasetIO::with_num_data(0);
    assert!(result.is_err());
}

#[test]
fn test_set_float_field() {
    println!("Testing set_float_field...");
    
    let mut dataset = DatasetIO::with_num_data(3).unwrap();
    let labels = vec![1.0, 0.0, 1.0];
    
    let result = dataset.set_float_field("label", &labels).unwrap();
    assert!(result);
    
    let retrieved_labels = dataset.get_float_field("label").unwrap().unwrap();
    assert_eq!(retrieved_labels, &labels);
    
    println!("set_float_field test passed!");
}

#[test]
fn test_set_int_field() {
    println!("Testing set_int_field...");
    
    let mut dataset = DatasetIO::with_num_data(3).unwrap();
    let groups = vec![0, 0, 1];
    
    let result = dataset.set_int_field("group", &groups).unwrap();
    assert!(result);
    
    let retrieved_groups = dataset.get_int_field("group").unwrap().unwrap();
    assert_eq!(retrieved_groups, groups);
    
    println!("set_int_field test passed!");
}

#[test]
fn test_helper_functions() {
    println!("Testing helper functions...");
    
    // Test one_feature_per_group
    let features = vec![0, 1, 2];
    let groups = one_feature_per_group(&features);
    assert_eq!(groups.len(), 3);
    assert_eq!(groups[0], vec![0]);
    assert_eq!(groups[1], vec![1]);
    assert_eq!(groups[2], vec![2]);
    
    // Test get_conflict_count
    let mark = vec![true, false, true, false];
    let indices = vec![0, 2];
    let count = get_conflict_count(&mark, &indices, 10);
    assert_eq!(count, Some(2));
    
    // Test mark_used
    let mut mark2 = vec![false; 4];
    let indices2 = vec![0, 2];
    mark_used(&mut mark2, &indices2);
    assert!(mark2[0]);
    assert!(!mark2[1]);
    assert!(mark2[2]);
    assert!(!mark2[3]);
    
    println!("Helper functions test passed!");
}

#[test]
fn test_binary_serialization() {
    println!("Testing binary serialization...");
    
    let mut dataset = DatasetIO::with_num_data(3).unwrap();
    let labels = vec![1.0, 0.0, 1.0];
    let weights = vec![1.0, 1.0, 1.0];
    
    dataset.set_float_field("label", &labels).unwrap();
    dataset.set_float_field("weight", &weights).unwrap();
    dataset.finish_load().unwrap();
    
    // This will create a test binary file
    dataset.save_binary_file(Some("rust_test_output.bin")).unwrap();
    
    // Verify file was created
    assert!(fs::metadata("rust_test_output.bin").is_ok());
    
    // Clean up
    let _ = fs::remove_file("rust_test_output.bin");
    
    println!("Binary serialization test passed!");
}

#[test]
fn test_finish_load() {
    println!("Testing finish_load...");
    
    let mut dataset = DatasetIO::with_num_data(3).unwrap();
    assert!(!dataset.is_finish_load());
    
    dataset.finish_load().unwrap();
    assert!(dataset.is_finish_load());
    
    // Calling finish_load again should be safe
    dataset.finish_load().unwrap();
    assert!(dataset.is_finish_load());
    
    println!("finish_load test passed!");
}

#[test]
fn test_metadata_operations() {
    println!("Testing metadata operations...");
    
    let mut dataset = DatasetIO::with_num_data(5).unwrap();
    
    // Test setting labels
    let labels = vec![1.0, 0.0, 1.0, 0.0, 1.0];
    dataset.set_float_field("label", &labels).unwrap();
    
    // Test setting weights
    let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    dataset.set_float_field("weight", &weights).unwrap();
    
    // Test setting groups
    let groups = vec![0, 0, 1, 1, 2];
    dataset.set_int_field("group", &groups).unwrap();
    
    // Verify all fields are set correctly
    let retrieved_labels = dataset.get_float_field("label").unwrap().unwrap();
    assert_eq!(retrieved_labels, &labels);
    
    let retrieved_weights = dataset.get_float_field("weight").unwrap().unwrap();
    assert_eq!(retrieved_weights, &weights);
    
    let retrieved_groups = dataset.get_int_field("group").unwrap().unwrap();
    assert_eq!(retrieved_groups, groups);
    
    println!("Metadata operations test passed!");
}

#[test]
fn test_dataset_resize() {
    println!("Testing dataset resize...");
    
    let mut dataset = DatasetIO::with_num_data(3).unwrap();
    
    // Set initial data
    let labels = vec![1.0, 0.0, 1.0];
    dataset.set_float_field("label", &labels).unwrap();
    
    // Resize to larger
    dataset.resize(5).unwrap();
    assert_eq!(dataset.num_data(), 5);
    assert_eq!(dataset.metadata().num_data, 5);
    assert_eq!(dataset.metadata().labels.len(), 5);
    
    println!("Dataset resize test passed!");
}

#[test]
fn test_error_conditions() {
    println!("Testing error conditions...");
    
    let mut dataset = DatasetIO::with_num_data(3).unwrap();
    
    // Test dimension mismatch for labels
    let wrong_size_labels = vec![1.0, 0.0]; // Size 2 instead of 3
    let result = dataset.set_float_field("label", &wrong_size_labels);
    assert!(result.is_err());
    
    // Test dimension mismatch for weights
    let wrong_size_weights = vec![1.0, 1.0, 1.0, 1.0]; // Size 4 instead of 3
    let result = dataset.set_float_field("weight", &wrong_size_weights);
    assert!(result.is_err());
    
    // Test dimension mismatch for groups
    let wrong_size_groups = vec![0, 1]; // Size 2 instead of 3
    let result = dataset.set_int_field("group", &wrong_size_groups);
    assert!(result.is_err());
    
    println!("Error conditions test passed!");
}

#[test]
fn test_unknown_fields() {
    println!("Testing unknown fields...");
    
    let mut dataset = DatasetIO::with_num_data(3).unwrap();
    
    // Test unknown float field
    let data = vec![1.0, 2.0, 3.0];
    let result = dataset.set_float_field("unknown_field", &data).unwrap();
    assert!(!result); // Should return false for unknown field
    
    // Test unknown int field
    let int_data = vec![1, 2, 3];
    let result = dataset.set_int_field("unknown_field", &int_data).unwrap();
    assert!(!result); // Should return false for unknown field
    
    // Test getting unknown fields
    let result = dataset.get_float_field("unknown_field").unwrap();
    assert!(result.is_none());
    
    let result = dataset.get_int_field("unknown_field").unwrap();
    assert!(result.is_none());
    
    println!("Unknown fields test passed!");
}

/// Run all equivalence tests
#[cfg(test)]
mod equivalence_tests {
    use super::*;
    
    #[test]
    fn run_comprehensive_equivalence_test() {
        println!("Running comprehensive Rust Dataset equivalence tests...");
        
        // All individual tests are run automatically by cargo test
        // This is a placeholder for any comprehensive testing
        
        println!("All Rust tests completed!");
    }
}