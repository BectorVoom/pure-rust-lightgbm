//! Configuration management system for Pure Rust LightGBM.
//!
//! This module provides comprehensive configuration management with type-safe
//! parameter validation, device configuration, and objective function settings.
//! It serves as the foundation for all LightGBM model configuration.

pub mod core;
pub mod device;
pub mod objective;
pub mod validation;

// Re-export commonly used configuration types
pub use core::{Config, ConfigBuilder, ConfigError};
pub use device::{DeviceConfig, DeviceCapabilities};
pub use objective::{ObjectiveConfig, ObjectiveFunction};
pub use validation::{ConfigValidator, ValidationResult, ValidationError};

use crate::core::types::*;
use crate::core::error::{Result, LightGBMError};
use serde::{Deserialize, Serialize};

/// Configuration constants for sensible defaults
pub const DEFAULT_CONFIG_FILE: &str = "lightgbm.toml";
pub const DEFAULT_MODEL_FILE: &str = "model.bin";

/// Configuration format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfigFormat {
    /// TOML configuration format
    Toml,
    /// JSON configuration format
    Json,
    /// YAML configuration format  
    Yaml,
    /// Environment variables
    Environment,
}

impl Default for ConfigFormat {
    fn default() -> Self {
        ConfigFormat::Toml
    }
}

/// Configuration source enumeration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigSource {
    /// Default configuration values
    Default,
    /// Configuration from file
    File(String),
    /// Configuration from environment variables
    Environment,
    /// Configuration from command line arguments
    CommandLine,
    /// Configuration from programmatic API
    Programmatic,
}

/// Configuration management utilities
pub struct ConfigManager {
    /// Current configuration
    config: Config,
    /// Configuration source
    source: ConfigSource,
    /// Validation results
    validation_results: Vec<ValidationResult>,
}

impl ConfigManager {
    /// Create a new configuration manager with default configuration
    pub fn new() -> Result<Self> {
        let config = Config::default();
        let mut manager = ConfigManager {
            config,
            source: ConfigSource::Default,
            validation_results: Vec::new(),
        };
        
        // Validate the default configuration
        manager.validate()?;
        
        Ok(manager)
    }
    
    /// Load configuration from file
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let config = Config::load_from_file(path)?;
        let mut manager = ConfigManager {
            config,
            source: ConfigSource::File(path.to_string_lossy().to_string()),
            validation_results: Vec::new(),
        };
        
        manager.validate()?;
        Ok(manager)
    }
    
    /// Load configuration from environment variables
    pub fn from_environment() -> Result<Self> {
        let config = Config::load_from_environment()?;
        let mut manager = ConfigManager {
            config,
            source: ConfigSource::Environment,
            validation_results: Vec::new(),
        };
        
        manager.validate()?;
        Ok(manager)
    }
    
    /// Get the current configuration
    pub fn config(&self) -> &Config {
        &self.config
    }
    
    /// Get a mutable reference to the current configuration
    pub fn config_mut(&mut self) -> &mut Config {
        &mut self.config
    }
    
    /// Get the configuration source
    pub fn source(&self) -> &ConfigSource {
        &self.source
    }
    
    /// Get validation results
    pub fn validation_results(&self) -> &[ValidationResult] {
        &self.validation_results
    }
    
    /// Validate the current configuration
    pub fn validate(&mut self) -> Result<()> {
        let validator = ConfigValidator::new();
        self.validation_results = validator.validate(&self.config)?;
        
        // Check for any validation errors
        for result in &self.validation_results {
            if let ValidationResult::Error(ref err) = result {
                return Err(LightGBMError::config(format!("Configuration validation failed: {}", err)));
            }
        }
        
        Ok(())
    }
    
    /// Save configuration to file
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        self.config.save_to_file(path)
    }
    
    /// Apply configuration overrides from environment variables
    pub fn apply_environment_overrides(&mut self) -> Result<()> {
        self.config.apply_environment_overrides()?;
        self.validate()
    }
    
    /// Update configuration with new values
    pub fn update(&mut self, new_config: Config) -> Result<()> {
        // Validate the new configuration first, before updating state
        new_config.validate()?;
        
        // Only update state if validation passes
        self.config = new_config;
        self.source = ConfigSource::Programmatic;
        self.validation_results.clear(); // Clear previous validation results
        
        Ok(())
    }
    
    /// Merge configuration with another configuration
    pub fn merge(&mut self, other: &Config) -> Result<()> {
        self.config.merge(other)?;
        self.validate()
    }
    
    /// Reset configuration to defaults
    pub fn reset_to_defaults(&mut self) -> Result<()> {
        self.config = Config::default();
        self.source = ConfigSource::Default;
        self.validate()
    }
    
    /// Get configuration summary for debugging
    pub fn summary(&self) -> String {
        format!(
            "Configuration Summary:\n\
             Source: {:?}\n\
             Objective: {}\n\
             Device: {}\n\
             Iterations: {}\n\
             Learning Rate: {}\n\
             Num Leaves: {}\n\
             Validation Issues: {}",
            self.source,
            self.config.objective,
            self.config.device_type,
            self.config.num_iterations,
            self.config.learning_rate,
            self.config.num_leaves,
            self.validation_results.len()
        )
    }
}

impl Default for ConfigManager {
    fn default() -> Self {
        Self::new().expect("Failed to create default configuration manager")
    }
}

/// Configuration builder pattern implementation
pub struct ConfigBuilderExt {
    config: Config,
    errors: Vec<String>,
}

impl ConfigBuilderExt {
    /// Create a new configuration builder
    pub fn new() -> Self {
        ConfigBuilderExt {
            config: Config::default(),
            errors: Vec::new(),
        }
    }
    
    /// Set all core training parameters at once
    pub fn training_params(
        mut self,
        num_iterations: usize,
        learning_rate: f64,
        num_leaves: usize,
        max_depth: i32,
    ) -> Self {
        self.config.num_iterations = num_iterations;
        self.config.learning_rate = learning_rate;
        self.config.num_leaves = num_leaves;
        self.config.max_depth = max_depth;
        
        // Validate parameters
        if learning_rate <= 0.0 || learning_rate > 1.0 {
            self.errors.push("learning_rate must be in range (0.0, 1.0]".to_string());
        }
        if num_leaves < 2 {
            self.errors.push("num_leaves must be at least 2".to_string());
        }
        
        self
    }
    
    /// Set regularization parameters
    pub fn regularization(mut self, lambda_l1: f64, lambda_l2: f64) -> Self {
        self.config.lambda_l1 = lambda_l1;
        self.config.lambda_l2 = lambda_l2;
        
        if lambda_l1 < 0.0 {
            self.errors.push("lambda_l1 must be non-negative".to_string());
        }
        if lambda_l2 < 0.0 {
            self.errors.push("lambda_l2 must be non-negative".to_string());
        }
        
        self
    }
    
    /// Set sampling parameters
    pub fn sampling(mut self, feature_fraction: f64, bagging_fraction: f64, bagging_freq: usize) -> Self {
        self.config.feature_fraction = feature_fraction;
        self.config.bagging_fraction = bagging_fraction;
        self.config.bagging_freq = bagging_freq;
        
        if feature_fraction <= 0.0 || feature_fraction > 1.0 {
            self.errors.push("feature_fraction must be in range (0.0, 1.0]".to_string());
        }
        if bagging_fraction <= 0.0 || bagging_fraction > 1.0 {
            self.errors.push("bagging_fraction must be in range (0.0, 1.0]".to_string());
        }
        
        self
    }
    
    /// Set device configuration
    pub fn device(mut self, device_type: DeviceType) -> Self {
        self.config.device_type = device_type;
        self
    }
    
    /// Set objective function
    pub fn objective(mut self, objective: ObjectiveType) -> Self {
        self.config.objective = objective;
        self
    }
    
    /// Set early stopping parameters
    pub fn early_stopping(mut self, rounds: usize, tolerance: f64) -> Self {
        self.config.early_stopping_rounds = Some(rounds);
        self.config.early_stopping_tolerance = tolerance;
        self
    }
    
    /// Build the configuration
    pub fn build(self) -> Result<Config> {
        if !self.errors.is_empty() {
            return Err(LightGBMError::config(format!(
                "Configuration validation failed: {}",
                self.errors.join(", ")
            )));
        }
        
        // Final validation
        self.config.validate()?;
        
        Ok(self.config)
    }
    
    /// Build the configuration without validation (for testing)
    pub fn build_unchecked(self) -> Config {
        self.config
    }
}

impl Default for ConfigBuilderExt {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for configuration management
pub mod utils {
    use super::*;
    use std::collections::HashMap;
    
    /// Parse configuration from key-value pairs
    pub fn parse_config_from_map(map: HashMap<String, String>) -> Result<Config> {
        let mut builder = ConfigBuilderExt::new();
        
        for (key, value) in map {
            match key.as_str() {
                "num_iterations" => {
                    let val = value.parse::<usize>()
                        .map_err(|_| LightGBMError::config(format!("Invalid num_iterations: {}", value)))?;
                    builder.config.num_iterations = val;
                }
                "learning_rate" => {
                    let val = value.parse::<f64>()
                        .map_err(|_| LightGBMError::config(format!("Invalid learning_rate: {}", value)))?;
                    builder.config.learning_rate = val;
                }
                "num_leaves" => {
                    let val = value.parse::<usize>()
                        .map_err(|_| LightGBMError::config(format!("Invalid num_leaves: {}", value)))?;
                    builder.config.num_leaves = val;
                }
                "objective" => {
                    let val = match value.as_str() {
                        "regression" => ObjectiveType::Regression,
                        "binary" => ObjectiveType::Binary,
                        "multiclass" => ObjectiveType::Multiclass,
                        "ranking" => ObjectiveType::Ranking,
                        _ => return Err(LightGBMError::config(format!("Invalid objective: {}", value))),
                    };
                    builder.config.objective = val;
                }
                "device_type" => {
                    let val = match value.as_str() {
                        "cpu" => DeviceType::CPU,
                        "gpu" => DeviceType::GPU,
                        _ => return Err(LightGBMError::config(format!("Invalid device_type: {}", value))),
                    };
                    builder.config.device_type = val;
                }
                _ => {
                    log::warn!("Unknown configuration parameter: {}", key);
                }
            }
        }
        
        builder.build()
    }
    
    /// Convert configuration to key-value pairs
    pub fn config_to_map(config: &Config) -> HashMap<String, String> {
        let mut map = HashMap::new();
        
        map.insert("num_iterations".to_string(), config.num_iterations.to_string());
        map.insert("learning_rate".to_string(), config.learning_rate.to_string());
        map.insert("num_leaves".to_string(), config.num_leaves.to_string());
        map.insert("max_depth".to_string(), config.max_depth.to_string());
        map.insert("lambda_l1".to_string(), config.lambda_l1.to_string());
        map.insert("lambda_l2".to_string(), config.lambda_l2.to_string());
        map.insert("feature_fraction".to_string(), config.feature_fraction.to_string());
        map.insert("bagging_fraction".to_string(), config.bagging_fraction.to_string());
        map.insert("bagging_freq".to_string(), config.bagging_freq.to_string());
        map.insert("objective".to_string(), config.objective.to_string());
        map.insert("device_type".to_string(), config.device_type.to_string());
        map.insert("num_threads".to_string(), config.num_threads.to_string());
        map.insert("random_seed".to_string(), config.random_seed.to_string());
        
        if let Some(rounds) = config.early_stopping_rounds {
            map.insert("early_stopping_rounds".to_string(), rounds.to_string());
        }
        map.insert("early_stopping_tolerance".to_string(), config.early_stopping_tolerance.to_string());
        
        map
    }
    
    /// Get environment variable with fallback
    pub fn get_env_var_or_default(key: &str, default: &str) -> String {
        std::env::var(key).unwrap_or_else(|_| default.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    
    #[test]
    fn test_config_manager_creation() {
        let manager = ConfigManager::new().unwrap();
        assert!(matches!(manager.source(), ConfigSource::Default));
        assert_eq!(manager.config().objective, ObjectiveType::Regression);
    }
    
    #[test]
    fn test_config_builder_ext() {
        let config = ConfigBuilderExt::new()
            .training_params(200, 0.05, 63, 10)
            .regularization(0.1, 0.2)
            .sampling(0.8, 0.9, 5)
            .device(DeviceType::CPU)
            .objective(ObjectiveType::Binary)
            .early_stopping(20, 1e-6)
            .build()
            .unwrap();
        
        assert_eq!(config.num_iterations, 200);
        assert_eq!(config.learning_rate, 0.05);
        assert_eq!(config.num_leaves, 63);
        assert_eq!(config.max_depth, 10);
        assert_eq!(config.lambda_l1, 0.1);
        assert_eq!(config.lambda_l2, 0.2);
        assert_eq!(config.feature_fraction, 0.8);
        assert_eq!(config.bagging_fraction, 0.9);
        assert_eq!(config.bagging_freq, 5);
        assert_eq!(config.device_type, DeviceType::CPU);
        assert_eq!(config.objective, ObjectiveType::Binary);
        assert_eq!(config.early_stopping_rounds, Some(20));
        assert_eq!(config.early_stopping_tolerance, 1e-6);
    }
    
    #[test]
    fn test_config_builder_validation() {
        let result = ConfigBuilderExt::new()
            .training_params(100, -0.1, 1, 5)  // Invalid learning_rate and num_leaves
            .build();
        
        assert!(result.is_err());
    }
    
    #[test]
    fn test_config_from_map() {
        let mut map = HashMap::new();
        map.insert("num_iterations".to_string(), "500".to_string());
        map.insert("learning_rate".to_string(), "0.05".to_string());
        map.insert("objective".to_string(), "binary".to_string());
        
        let config = utils::parse_config_from_map(map).unwrap();
        assert_eq!(config.num_iterations, 500);
        assert_eq!(config.learning_rate, 0.05);
        assert_eq!(config.objective, ObjectiveType::Binary);
    }
    
    #[test]
    fn test_config_to_map() {
        let config = Config::default();
        let map = utils::config_to_map(&config);
        
        assert!(map.contains_key("num_iterations"));
        assert!(map.contains_key("learning_rate"));
        assert!(map.contains_key("objective"));
        assert_eq!(map.get("objective").unwrap(), "regression");
    }
    
    #[test]
    fn test_config_manager_summary() {
        let manager = ConfigManager::new().unwrap();
        let summary = manager.summary();
        
        assert!(summary.contains("Configuration Summary"));
        assert!(summary.contains("Source: Default"));
        assert!(summary.contains("Objective: regression"));
    }
}