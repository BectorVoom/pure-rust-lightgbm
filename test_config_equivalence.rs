use std::collections::HashMap;
use lightgbm_rust::core::config::*;

fn test_default_config() {
    println!("Testing default configuration values...");
    
    let config = Config::default();
    
    // Test default values match expectations
    assert_eq!(config.objective, "regression");
    assert_eq!(config.num_iterations, 100);
    assert_eq!((config.learning_rate - 0.1).abs() < 1e-9, true);
    assert_eq!(config.num_leaves, 31);
    assert_eq!(config.max_depth, -1);
    assert_eq!(config.boosting, "gbdt");
    assert_eq!(config.tree_learner, "serial");
    assert_eq!(config.device_type, "cpu");
    assert_eq!(config.num_threads, 0);
    assert_eq!(config.seed, 0);
    assert_eq!(config.deterministic, false);
    
    println!("✓ Default configuration test passed");
}

fn test_parameter_parsing() {
    println!("Testing parameter parsing...");
    
    let mut params = HashMap::new();
    
    // Test get_string
    params.insert("test_string".to_string(), "value".to_string());
    assert_eq!(Config::get_string(&params, "test_string"), Some("value".to_string()));
    assert_eq!(Config::get_string(&params, "nonexistent"), None);
    
    // Test get_int
    params.insert("test_int".to_string(), "42".to_string());
    assert_eq!(Config::get_int(&params, "test_int").unwrap(), Some(42));
    
    // Test get_double
    params.insert("test_double".to_string(), "3.14".to_string());
    let result = Config::get_double(&params, "test_double").unwrap().unwrap();
    assert!((result - 3.14).abs() < 1e-9);
    
    // Test get_bool
    params.insert("test_bool_true".to_string(), "true".to_string());
    params.insert("test_bool_false".to_string(), "false".to_string());
    assert_eq!(Config::get_bool(&params, "test_bool_true").unwrap(), Some(true));
    assert_eq!(Config::get_bool(&params, "test_bool_false").unwrap(), Some(false));
    
    println!("✓ Parameter parsing test passed");
}

fn test_string_to_map() {
    println!("Testing str_to_map function...");
    
    let param_str = "num_leaves=63 learning_rate=0.05 objective=binary";
    let params = Config::str_to_map(param_str);
    
    assert_eq!(params.get("num_leaves"), Some(&"63".to_string()));
    assert_eq!(params.get("learning_rate"), Some(&"0.05".to_string()));
    assert_eq!(params.get("objective"), Some(&"binary".to_string()));
    
    println!("✓ str_to_map test passed");
}

fn test_config_from_params() {
    println!("Testing Config construction from parameters...");
    
    let mut params = HashMap::new();
    params.insert("objective".to_string(), "binary".to_string());
    params.insert("num_iterations".to_string(), "500".to_string());
    params.insert("learning_rate".to_string(), "0.05".to_string());
    params.insert("num_leaves".to_string(), "63".to_string());
    params.insert("max_depth".to_string(), "6".to_string());
    params.insert("device_type".to_string(), "cpu".to_string());
    params.insert("deterministic".to_string(), "true".to_string());
    
    let config = Config::from_params(params);
    
    assert_eq!(config.objective, "binary");
    assert_eq!(config.num_iterations, 500);
    assert!((config.learning_rate - 0.05).abs() < 1e-9);
    assert_eq!(config.num_leaves, 63);
    assert_eq!(config.max_depth, 6);
    assert_eq!(config.device_type, "cpu");
    assert_eq!(config.deterministic, true);
    
    println!("✓ Config from parameters test passed");
}

fn test_alias_handling() {
    println!("Testing parameter aliases...");
    
    let mut params = HashMap::new();
    params.insert("num_iteration".to_string(), "200".to_string());  // alias for num_iterations
    params.insert("eta".to_string(), "0.2".to_string());           // alias for learning_rate
    params.insert("num_leaf".to_string(), "127".to_string());      // alias for num_leaves
    
    let config = Config::from_params(params);
    
    assert_eq!(config.num_iterations, 200);
    assert!((config.learning_rate - 0.2).abs() < 1e-9);
    assert_eq!(config.num_leaves, 127);
    
    println!("✓ Alias handling test passed");
}

fn test_task_types() {
    println!("Testing task types...");
    
    let config = Config::default();
    
    // Test default task
    assert_eq!(config.task, TaskType::Train);
    
    // Test different task types through string parameters
    let mut params = HashMap::new();
    params.insert("task".to_string(), "predict".to_string());
    let predict_config = Config::from_params(params);
    assert_eq!(predict_config.task, TaskType::Predict);
    
    println!("✓ Task types test passed");
}

fn test_objective_aliases() {
    println!("Testing objective aliases...");
    
    // Test various objective aliases resolve correctly
    assert_eq!(Config::parse_objective_alias("regression"), "regression");
    assert_eq!(Config::parse_objective_alias("mse"), "regression");
    assert_eq!(Config::parse_objective_alias("l2"), "regression");
    assert_eq!(Config::parse_objective_alias("rmse"), "regression");
    assert_eq!(Config::parse_objective_alias("mae"), "regression_l1");
    assert_eq!(Config::parse_objective_alias("l1"), "regression_l1");
    
    println!("✓ Objective aliases test passed");
}

fn test_metric_aliases() {
    println!("Testing metric aliases...");
    
    // Test various metric aliases resolve correctly
    assert_eq!(Config::parse_metric_alias("mse"), "l2");
    assert_eq!(Config::parse_metric_alias("mae"), "l1");
    assert_eq!(Config::parse_metric_alias("rmse"), "rmse");
    assert_eq!(Config::parse_metric_alias("binary"), "binary_logloss");
    assert_eq!(Config::parse_metric_alias("multiclass"), "multi_logloss");
    
    println!("✓ Metric aliases test passed");
}

fn test_config_builder() {
    println!("Testing ConfigBuilder...");
    
    let result = ConfigBuilder::new()
        .objective("binary")
        .num_iterations(500)
        .learning_rate(0.05)
        .num_leaves(63)
        .device_type("cpu")
        .build();
        
    match result {
        Ok(config) => {
            assert_eq!(config.objective, "binary");
            assert_eq!(config.num_iterations, 500);
            assert!((config.learning_rate - 0.05).abs() < 1e-9);
            assert_eq!(config.num_leaves, 63);
            assert_eq!(config.device_type, "cpu");
            println!("✓ ConfigBuilder test passed");
        }
        Err(e) => panic!("ConfigBuilder test failed: {}", e),
    }
}

fn main() {
    println!("Running Rust Config equivalence tests...");
    
    test_default_config();
    test_parameter_parsing();
    test_string_to_map();
    test_config_from_params();
    test_alias_handling();
    test_task_types();
    test_objective_aliases();
    test_metric_aliases();
    test_config_builder();
    
    println!("\\n✅ All Rust tests passed!");
    
    // Output key values for comparison with C++ version
    println!("\\n=== Rust Reference Values ===");
    let config = Config::default();
    println!("default_num_leaves: {}", config.num_leaves);
    println!("default_learning_rate: {}", config.learning_rate);
    println!("default_num_iterations: {}", config.num_iterations);
    println!("default_objective: {}", config.objective);
    println!("default_boosting: {}", config.boosting);
    println!("default_device_type: {}", config.device_type);
}