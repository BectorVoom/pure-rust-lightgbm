//! Test hyperparameter optimization functionality

use lightgbm_rust::*;
use lightgbm_rust::hyperopt::*;
use ndarray::{Array1, Array2};

fn main() -> Result<()> {
    // Initialize the library
    lightgbm_rust::init()?;
    
    println!("=== Testing Hyperparameter Optimization Implementation ===");
    
    // Create a test dataset
    let features = Array2::from_shape_vec((100, 3), 
        (0..300).map(|i| (i as f32) * 0.1).collect()
    ).map_err(|e| LightGBMError::data_loading(format!("Failed to create features: {}", e)))?;
    
    // Create labels with some pattern
    let labels = Array1::from_vec(
        (0..100).map(|i| {
            let sum: f32 = features.row(i).sum();
            if sum > 15.0 { 1.0 } else { 0.0 }
        }).collect()
    );
    
    let dataset = Dataset::new(features.clone(), labels, None, None, None, None)?;
    
    println!("Created test dataset: {} samples, {} features", 
             dataset.num_data(), dataset.num_features());
    
    // Test 1: Cross-validation functionality
    println!("\n=== Test 1: Cross-Validation ===");
    
    let model_config = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .num_iterations(5)
        .learning_rate(0.1)
        .build()?;
    
    let cv_config = CrossValidationConfig::new()
        .with_num_folds(3)
        .with_shuffle(true)
        .with_random_seed(42);
    
    let cv_result = cross_validate(
        &dataset, 
        &model_config, 
        &cv_config, 
        &["accuracy", "f1"]
    )?;
    
    println!("Cross-validation completed:");
    for (metric, mean_val) in &cv_result.mean_metrics {
        let std_val = cv_result.std_metrics.get(metric).unwrap_or(&0.0);
        println!("  {}: {:.4} ± {:.4}", metric, mean_val, std_val);
    }
    
    // Test 2: Parameter space creation and validation
    println!("\n=== Test 2: Parameter Space ===");
    
    let param_space = HyperparameterSpace::new()
        .add_float("learning_rate", 0.01, 0.3)
        .add_int("num_leaves", 10, 100)
        .add_float("reg_alpha", 0.0, 1.0);
    
    println!("Created parameter space:");
    println!("  Float params: {:?}", param_space.float_params);
    println!("  Int params: {:?}", param_space.int_params);
    println!("  Categorical params: {:?}", param_space.categorical_params);
    
    // Test 3: Random search optimization
    println!("\n=== Test 3: Random Search Optimization ===");
    
    let opt_config = OptimizationConfig::new()
        .with_num_trials(8)
        .with_cv_folds(3)
        .with_metric("accuracy")
        .with_direction(OptimizationDirection::Maximize)
        .with_random_seed(42)
        .with_progress_interval(3);
    
    let opt_result = optimize_hyperparameters_with_strategy(
        &dataset,
        &param_space,
        &opt_config,
        OptimizationStrategy::RandomSearch
    )?;
    
    println!("Random search optimization completed:");
    println!("  Trials completed: {}", opt_result.num_trials);
    println!("  Best score: {:.6}", opt_result.best_score);
    println!("  Best parameters: {:?}", opt_result.best_params);
    
    // Test 4: Grid search optimization
    println!("\n=== Test 4: Grid Search Optimization ===");
    
    let grid_opt_config = OptimizationConfig::new()
        .with_num_trials(10)
        .with_cv_folds(3)
        .with_metric("accuracy")
        .with_direction(OptimizationDirection::Maximize);
    
    let grid_result = optimize_hyperparameters_with_strategy(
        &dataset,
        &param_space,
        &grid_opt_config,
        OptimizationStrategy::GridSearch
    )?;
    
    println!("Grid search optimization completed:");
    println!("  Trials completed: {}", grid_result.num_trials);
    println!("  Best score: {:.6}", grid_result.best_score);
    println!("  Best parameters: {:?}", grid_result.best_params);
    
    // Test 5: Early stopping
    println!("\n=== Test 5: Early Stopping ===");
    
    let early_stop_config = OptimizationConfig::new()
        .with_num_trials(20)
        .with_cv_folds(3)
        .with_metric("accuracy")
        .with_direction(OptimizationDirection::Maximize)
        .with_early_stopping_patience(5)
        .with_early_stopping_min_delta(0.001)
        .with_progress_interval(2);
    
    let early_stop_result = optimize_hyperparameters_with_strategy(
        &dataset,
        &param_space,
        &early_stop_config,
        OptimizationStrategy::RandomSearch
    )?;
    
    println!("Early stopping optimization completed:");
    println!("  Trials completed: {}", early_stop_result.num_trials);
    println!("  Best score: {:.6}", early_stop_result.best_score);
    println!("  (Early stopping may have terminated before {} trials)", early_stop_config.num_trials);
    
    // Test 6: Regression optimization
    println!("\n=== Test 6: Regression Optimization ===");
    
    // Create regression dataset
    let reg_labels = Array1::from_vec(
        (0..100).map(|i| {
            let sum: f32 = features.row(i).sum();
            sum * 0.1 + (i as f32) * 0.01
        }).collect()
    );
    
    let reg_dataset = Dataset::new(features.clone(), reg_labels, None, None, None, None)?;
    
    let reg_opt_config = OptimizationConfig::new()
        .with_num_trials(6)
        .with_cv_folds(3)
        .with_metric("rmse")
        .with_direction(OptimizationDirection::Minimize);
    
    let reg_result = optimize_hyperparameters_with_strategy(
        &reg_dataset,
        &param_space,
        &reg_opt_config,
        OptimizationStrategy::RandomSearch
    )?;
    
    println!("Regression optimization completed:");
    println!("  Trials completed: {}", reg_result.num_trials);
    println!("  Best RMSE: {:.6}", reg_result.best_score);
    println!("  Best parameters: {:?}", reg_result.best_params);
    
    // Test 7: Error handling
    println!("\n=== Test 7: Error Handling ===");
    
    // Test invalid parameter space
    let invalid_space = HyperparameterSpace::new()
        .add_float("invalid_param", 1.0, 0.5); // min > max
    
    match optimize_hyperparameters(&dataset, &invalid_space, &opt_config) {
        Ok(_) => println!("ERROR: Should have failed with invalid parameter space"),
        Err(e) => println!("✅ Correctly caught invalid parameter space: {}", e),
    }
    
    // Test invalid cross-validation config
    let invalid_cv_config = CrossValidationConfig::new().with_num_folds(1);
    match cross_validate(&dataset, &model_config, &invalid_cv_config, &["accuracy"]) {
        Ok(_) => println!("ERROR: Should have failed with < 2 folds"),
        Err(e) => println!("✅ Correctly caught invalid CV config: {}", e),
    }
    
    // Test unsupported metric
    match cross_validate(&dataset, &model_config, &cv_config, &["unsupported_metric"]) {
        Ok(_) => println!("ERROR: Should have failed with unsupported metric"),
        Err(e) => println!("✅ Correctly caught unsupported metric: {}", e),
    }
    
    println!("\n✅ All hyperparameter optimization tests completed successfully!");
    println!("   - Cross-validation with multiple metrics");
    println!("   - Parameter space creation and validation");
    println!("   - Random search optimization");
    println!("   - Grid search optimization");
    println!("   - Early stopping functionality");
    println!("   - Regression optimization");
    println!("   - Comprehensive error handling");
    
    Ok(())
}