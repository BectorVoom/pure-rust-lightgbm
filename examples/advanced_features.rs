//! Advanced features example for pure Rust LightGBM.
//!
//! This example demonstrates advanced features like:
//! - Custom configurations
//! - Feature importance
//! - Model serialization
//! - Memory management
//! - Dataset preprocessing
//!
//! Run with: `cargo run --example advanced_features`

use lightgbm_rust::*;
use ndarray::{Array1, Array2};
use std::fs;
use std::time::Instant;

fn main() -> Result<()> {
    // Initialize the library
    lightgbm_rust::init()?;
    
    println!("Pure Rust LightGBM - Advanced Features Example");
    println!("==============================================");
    
    // Display system capabilities
    demonstrate_capabilities();
    
    // Demonstrate advanced configuration
    demonstrate_advanced_configuration()?;
    
    // Demonstrate dataset preprocessing
    demonstrate_dataset_preprocessing()?;
    
    // Demonstrate memory management
    demonstrate_memory_management()?;
    
    // Demonstrate serialization
    demonstrate_serialization()?;
    
    // Demonstrate performance benchmarking
    demonstrate_performance_benchmarking()?;
    
    // Demonstrate error handling
    demonstrate_error_handling();
    
    println!("All advanced features demonstrated successfully!");
    
    Ok(())
}

fn demonstrate_capabilities() {
    println!("1. System Capabilities");
    println!("---------------------");
    
    let caps = lightgbm_rust::capabilities();
    println!("  SIMD aligned memory: {}", caps.simd_aligned_memory);
    println!("  Thread safe memory: {}", caps.thread_safe_memory);
    println!("  Rich error types: {}", caps.rich_error_types);
    println!("  Trait abstractions: {}", caps.trait_abstractions);
    println!("  Serialization: {}", caps.serialization);
    println!("  GPU ready: {}", caps.gpu_ready);
    println!("  All capabilities available: {}", caps.all_available());
    println!();
}

fn demonstrate_advanced_configuration() -> Result<()> {
    println!("2. Advanced Configuration");
    println!("------------------------");
    
    // Create a comprehensive configuration
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .learning_rate(0.05)
        .num_iterations(1000)
        .num_leaves(127)
        .max_depth(7)
        .lambda_l1(0.1)
        .lambda_l2(0.2)
        .min_data_in_leaf(50)
        // .min_sum_hessian_in_leaf(0.001)  // Method not available in ConfigBuilder yet
        .feature_fraction(0.8)
        .bagging_fraction(0.9)
        .bagging_freq(5)
        .device_type(DeviceType::CPU)
        .num_threads(4)
        .early_stopping_rounds(Some(100))
        .early_stopping_tolerance(0.01)
        .build()?;
    
    println!("  Configuration created:");
    println!("    - Objective: {:?}", config.objective);
    println!("    - Learning rate: {}", config.learning_rate);
    println!("    - Iterations: {}", config.num_iterations);
    println!("    - Leaves: {}", config.num_leaves);
    println!("    - Max depth: {}", config.max_depth);
    println!("    - L1 regularization: {}", config.lambda_l1);
    println!("    - L2 regularization: {}", config.lambda_l2);
    println!("    - Min data in leaf: {}", config.min_data_in_leaf);
    println!("    - Feature fraction: {}", config.feature_fraction);
    println!("    - Bagging fraction: {}", config.bagging_fraction);
    println!("    - Bagging frequency: {}", config.bagging_freq);
    println!("    - Device type: {:?}", config.device_type);
    println!("    - Number of threads: {}", config.num_threads);
    println!("    - Early stopping rounds: {:?}", config.early_stopping_rounds);
    println!("    - Early stopping tolerance: {}", config.early_stopping_tolerance);
    
    // Validate configuration
    match config.validate() {
        Ok(_) => println!("  Configuration validation: PASSED"),
        Err(e) => println!("  Configuration validation: FAILED - {}", e),
    }
    
    println!();
    Ok(())
}

fn demonstrate_dataset_preprocessing() -> Result<()> {
    println!("3. Dataset Preprocessing");
    println!("-----------------------");
    
    // Create a dataset with mixed features
    let features = Array2::from_shape_vec(
        (20, 5),
        (0..100).map(|i| i as f32 / 10.0).collect()
    ).unwrap();
    
    let labels = Array1::from_vec(
        (0..20).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect()
    );
    
    let weights = Some(Array1::from_vec(
        (0..20).map(|i| 1.0 + (i as f32 / 20.0)).collect()
    ));
    
    let feature_names = Some(vec![
        "numerical_1".to_string(),
        "numerical_2".to_string(),
        "categorical_1".to_string(),
        "categorical_2".to_string(),
        "mixed_feature".to_string(),
    ]);
    
    let dataset = Dataset::new(
        features,
        labels,
        weights,
        None,
        feature_names,
        None,
    )?;
    
    println!("  Dataset created:");
    println!("    - Samples: {}", dataset.num_data());
    println!("    - Features: {}", dataset.num_features());
    println!("    - Has weights: {}", dataset.has_weights());
    println!("    - Memory usage: {} bytes", dataset.memory_usage());
    
    // Calculate comprehensive statistics
    let stats = dataset::utils::calculate_statistics(&dataset);
    println!("  Dataset statistics:");
    println!("    - Sparsity: {:.4}", stats.sparsity);
    println!("    - Total missing values: {}", stats.missing_counts.iter().sum::<usize>());
    
    // Feature type detection
    let feature_types = dataset::utils::detect_feature_types(&dataset.features().to_owned());
    println!("  Feature types:");
    for (i, feature_type) in feature_types.iter().enumerate() {
        println!("    - Feature {}: {:?}", i, feature_type);
    }
    
    // Detailed feature statistics
    println!("  Feature statistics:");
    for (i, feature_stat) in stats.feature_stats.iter().enumerate() {
        println!("    - {}: min={:.2}, max={:.2}, mean={:.2}, std={:.2}, unique={}", 
                 feature_stat.name, feature_stat.min_value, feature_stat.max_value,
                 feature_stat.mean_value, feature_stat.std_dev, feature_stat.num_unique);
    }
    
    // Dataset validation
    let validation_result = dataset::utils::validate_dataset(&dataset);
    println!("  Dataset validation:");
    println!("    - Valid: {}", validation_result.is_valid);
    println!("    - Errors: {}", validation_result.errors.len());
    println!("    - Warnings: {}", validation_result.warnings.len());
    
    if !validation_result.warnings.is_empty() {
        println!("  Warnings:");
        for warning in &validation_result.warnings {
            println!("    - {}", warning);
        }
    }
    
    if !validation_result.suggestions.is_empty() {
        println!("  Suggestions:");
        for suggestion in &validation_result.suggestions {
            println!("    - {}", suggestion);
        }
    }
    
    println!();
    Ok(())
}

fn demonstrate_memory_management() -> Result<()> {
    println!("4. Memory Management");
    println!("-------------------");
    
    // Test aligned buffer
    let buffer: AlignedBuffer<f32> = AlignedBuffer::new(1000)?;
    println!("  Aligned buffer created:");
    println!("    - Capacity: {}", buffer.capacity());
    println!("    - Length: {}", buffer.len());
    println!("    - Is aligned: {}", buffer.is_aligned());
    println!("    - Alignment: {} bytes", ALIGNED_SIZE);
    
    // Test memory pool
    let pool = MemoryPool::new(4096, 5);
    let stats = pool.stats();
    println!("  Memory pool created:");
    println!("    - Allocated bytes: {}", stats.allocated_bytes);
    println!("    - Used bytes: {}", stats.used_bytes);
    println!("    - Alignment: {} bytes", stats.alignment);
    
    // Test memory efficiency with different sizes
    let sizes = vec![100, 1000, 10000, 100000];
    println!("  Memory efficiency test:");
    
    for size in sizes {
        let start = Instant::now();
        let features = Array2::zeros((size, 10));
        let labels = Array1::zeros(size);
        
        let dataset = Dataset::new(
            features,
            labels,
            None,
            None,
            None,
            None,
        )?;
        
        let duration = start.elapsed();
        let memory_usage = dataset.memory_usage();
        
        println!("    - Size {}: {:.2}ms, {} bytes, {:.2} bytes/sample", 
                 size, duration.as_secs_f64() * 1000.0, memory_usage, 
                 memory_usage as f64 / size as f64);
    }
    
    println!();
    Ok(())
}

fn demonstrate_serialization() -> Result<()> {
    println!("5. Serialization");
    println!("---------------");
    
    // Create configuration for serialization
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Multiclass)
        .num_class(3)
        .learning_rate(0.1)
        .num_iterations(100)
        .num_leaves(31)
        .lambda_l1(0.05)
        .lambda_l2(0.1)
        .build()?;
    
    // JSON serialization
    let json_str = serde_json::to_string_pretty(&config)?;
    println!("  JSON serialization:");
    println!("    - Size: {} bytes", json_str.len());
    println!("    - Preview: {}", &json_str[..json_str.len().min(200)]);
    if json_str.len() > 200 {
        println!("    - ... (truncated)");
    }
    
    // Deserialize from JSON
    let deserialized_config: Config = serde_json::from_str(&json_str)?;
    println!("  JSON deserialization: SUCCESS");
    println!("    - Objective matches: {}", deserialized_config.objective == config.objective);
    println!("    - Learning rate matches: {}", deserialized_config.learning_rate == config.learning_rate);
    println!("    - Num class matches: {}", deserialized_config.num_class == config.num_class);
    
    // Binary serialization
    let binary_data = bincode::serialize(&config)?;
    println!("  Binary serialization:");
    println!("    - Size: {} bytes", binary_data.len());
    println!("    - Compression ratio: {:.2}x", json_str.len() as f64 / binary_data.len() as f64);
    
    // Deserialize from binary
    let deserialized_binary: Config = bincode::deserialize(&binary_data)?;
    println!("  Binary deserialization: SUCCESS");
    println!("    - Configurations match: {}", deserialized_binary.objective == config.objective);
    
    // Dataset configuration serialization
    let dataset_config = DatasetConfig::new()
        .with_target_column("target")
        .with_feature_columns(vec!["f1".to_string(), "f2".to_string()])
        .with_categorical_features(vec![0, 1])
        .with_max_bin(512)
        .with_two_round(true);
    
    let dataset_json = serde_json::to_string_pretty(&dataset_config)?;
    println!("  Dataset config serialization:");
    println!("    - Size: {} bytes", dataset_json.len());
    
    // Save and load from file
    let config_file = "config.json";
    fs::write(config_file, &json_str)?;
    let loaded_json = fs::read_to_string(config_file)?;
    let loaded_config: Config = serde_json::from_str(&loaded_json)?;
    
    println!("  File I/O:");
    println!("    - Saved to: {}", config_file);
    println!("    - Loaded successfully: {}", loaded_config.objective == config.objective);
    
    // Clean up
    fs::remove_file(config_file)?;
    
    println!();
    Ok(())
}

fn demonstrate_performance_benchmarking() -> Result<()> {
    println!("6. Performance Benchmarking");
    println!("---------------------------");
    
    let test_sizes = vec![
        (100, 10),
        (1000, 10),
        (10000, 10),
        (1000, 100),
        (10000, 100),
    ];
    
    println!("  Dataset creation benchmarks:");
    for (samples, features) in test_sizes {
        let start = Instant::now();
        
        let features_array = Array2::zeros((samples, features));
        let labels_array = Array1::zeros(samples);
        
        let dataset = Dataset::new(
            features_array,
            labels_array,
            None,
            None,
            None,
            None,
        )?;
        
        let creation_time = start.elapsed();
        
        // Benchmark statistics calculation
        let stats_start = Instant::now();
        let stats = dataset::utils::calculate_statistics(&dataset);
        let stats_time = stats_start.elapsed();
        
        // Benchmark validation
        let validation_start = Instant::now();
        let _validation = dataset::utils::validate_dataset(&dataset);
        let validation_time = validation_start.elapsed();
        
        println!("    - {}x{}: create={:.2}ms, stats={:.2}ms, validate={:.2}ms, memory={}KB",
                 samples, features,
                 creation_time.as_secs_f64() * 1000.0,
                 stats_time.as_secs_f64() * 1000.0,
                 validation_time.as_secs_f64() * 1000.0,
                 stats.memory_usage / 1024);
    }
    
    // Configuration creation benchmark
    let config_start = Instant::now();
    for _ in 0..1000 {
        let _config = ConfigBuilder::new()
            .objective(ObjectiveType::Binary)
            .learning_rate(0.1)
            .num_iterations(100)
            .build()?;
    }
    let config_time = config_start.elapsed();
    
    println!("  Configuration creation benchmark:");
    println!("    - 1000 configs: {:.2}ms ({:.2}μs per config)",
             config_time.as_secs_f64() * 1000.0,
             config_time.as_secs_f64() * 1000000.0 / 1000.0);
    
    println!();
    Ok(())
}

fn demonstrate_error_handling() {
    println!("7. Error Handling");
    println!("----------------");
    
    // Test various error scenarios
    
    // Invalid configuration
    let invalid_config = ConfigBuilder::new()
        .learning_rate(-0.1)
        .build();
    
    match invalid_config {
        Ok(_) => println!("  Invalid config error: FAILED TO DETECT"),
        Err(e) => {
            println!("  Invalid config error: DETECTED");
            println!("    - Error: {}", e);
            println!("    - Category: {}", e.category());
            println!("    - Recoverable: {}", e.is_recoverable());
        }
    }
    
    // Invalid dataset dimensions
    let features = Array2::zeros((10, 5));
    let labels = Array1::zeros(5); // Wrong size
    
    let invalid_dataset = Dataset::new(
        features,
        labels,
        None,
        None,
        None,
        None,
    );
    
    match invalid_dataset {
        Ok(_) => println!("  Invalid dataset error: FAILED TO DETECT"),
        Err(e) => {
            println!("  Invalid dataset error: DETECTED");
            println!("    - Error: {}", e);
            println!("    - Category: {}", e.category());
        }
    }
    
    // Invalid dataset config
    let invalid_dataset_config = DatasetConfig::new()
        .with_max_bin(1); // Too small
    
    match invalid_dataset_config.validate() {
        Ok(_) => println!("  Invalid dataset config error: FAILED TO DETECT"),
        Err(e) => {
            println!("  Invalid dataset config error: DETECTED");
            println!("    - Error: {}", e);
            println!("    - Category: {}", e.category());
        }
    }
    
    // Not implemented errors
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .build()
        .unwrap();
    
    let features = Array2::zeros((10, 5));
    let labels = Array1::zeros(10);
    let dataset = Dataset::new(features.clone(), labels, None, None, None, None).unwrap();
    
    let mut classifier = LGBMClassifier::new(config);
    
    match classifier.fit(&dataset) {
        Ok(_) => println!("  Not implemented error: FAILED TO DETECT"),
        Err(e) => {
            println!("  Not implemented error: DETECTED");
            println!("    - Error: {}", e);
            println!("    - Category: {}", e.category());
        }
    }
    
    // Memory allocation errors (simulated)
    match AlignedBuffer::<f32>::new(0) {
        Ok(_) => println!("  Memory allocation error: FAILED TO DETECT"),
        Err(e) => {
            println!("  Memory allocation error: DETECTED");
            println!("    - Error: {}", e);
            println!("    - Category: {}", e.category());
        }
    }
    
    println!();
}

// Helper function to create test data with specific characteristics
fn _create_test_data_with_properties(
    num_samples: usize, 
    num_features: usize, 
    missing_rate: f32, 
    categorical_indices: Vec<usize>
) -> Result<Dataset> {
    let mut features = Array2::zeros((num_samples, num_features));
    
    // Fill with test data
    for i in 0..num_samples {
        for j in 0..num_features {
            if categorical_indices.contains(&j) {
                features[[i, j]] = (i % 5) as f32; // Categorical with 5 categories
            } else {
                features[[i, j]] = (i * j) as f32 / 10.0; // Numerical
            }
            
            // Add missing values
            if rand::random::<f32>() < missing_rate {
                features[[i, j]] = f32::NAN;
            }
        }
    }
    
    let labels = Array1::from_vec(
        (0..num_samples).map(|i| (i % 2) as f32).collect()
    );
    
    Dataset::new(features, labels, None, None, None, None)
}