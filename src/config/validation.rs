//! Configuration validation system for Pure Rust LightGBM.
//!
//! This module provides comprehensive validation for LightGBM configuration
//! parameters, ensuring that all settings are valid and consistent before
//! training begins.

use crate::core::error::Result;
use crate::core::types::*;

use crate::config::core::Config;

use serde::{Deserialize, Serialize};

/// Validation result enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidationResult {
    /// Parameter is valid
    Valid,
    /// Parameter has a warning (non-fatal)
    Warning(ValidationWarning),
    /// Parameter has an error (fatal)
    Error(ValidationError),
}

/// Validation warning structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationWarning {
    /// Parameter name
    pub parameter: String,
    /// Parameter value
    pub value: String,
    /// Warning message
    pub message: String,
    /// Suggested fix
    pub suggestion: Option<String>,
}

/// Validation error structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationError {
    /// Parameter name
    pub parameter: String,
    /// Parameter value
    pub value: String,
    /// Error message
    pub message: String,
    /// Valid range or options
    pub valid_range: Option<String>,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Parameter '{}' = '{}': {}",
            self.parameter, self.value, self.message
        )?;
        if let Some(ref range) = self.valid_range {
            write!(f, " (valid range: {})", range)?;
        }
        Ok(())
    }
}

impl std::error::Error for ValidationError {}

/// Configuration validator
pub struct ConfigValidator {
    /// Enable strict validation
    strict_mode: bool,
    /// Custom validation rules
    custom_rules: Vec<Box<dyn ValidationRule>>,
    /// Validation context
    context: ValidationContext,
}

/// Validation context
#[derive(Debug, Clone, Default)]
pub struct ValidationContext {
    /// Dataset information (if available)
    pub dataset_info: Option<DatasetInfo>,
    /// System capabilities
    pub system_capabilities: Option<SystemCapabilities>,
    /// Performance requirements
    pub performance_requirements: Option<PerformanceRequirements>,
}

/// Dataset information for validation
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    /// Number of samples
    pub num_samples: usize,
    /// Number of features
    pub num_features: usize,
    /// Number of classes (for classification)
    pub num_classes: Option<usize>,
    /// Feature types
    pub feature_types: Vec<FeatureType>,
    /// Has missing values
    pub has_missing_values: bool,
    /// Memory size estimate in MB
    pub memory_size_mb: usize,
}

/// System capabilities for validation
#[derive(Debug, Clone)]
pub struct SystemCapabilities {
    /// Available memory in MB
    pub available_memory_mb: usize,
    /// Number of CPU cores
    pub num_cpu_cores: usize,
    /// GPU available
    pub gpu_available: bool,
    /// GPU memory in MB
    pub gpu_memory_mb: Option<usize>,
}

/// Performance requirements for validation
#[derive(Debug, Clone)]
pub struct PerformanceRequirements {
    /// Maximum training time in seconds
    pub max_training_time_seconds: Option<f64>,
    /// Maximum memory usage in MB
    pub max_memory_usage_mb: Option<usize>,
    /// Target accuracy
    pub target_accuracy: Option<f64>,
    /// Inference speed requirements
    pub inference_speed_requirements: Option<InferenceSpeedRequirements>,
}

/// Inference speed requirements
#[derive(Debug, Clone)]
pub struct InferenceSpeedRequirements {
    /// Maximum prediction time per sample in microseconds
    pub max_prediction_time_us: f64,
    /// Batch size for prediction
    pub batch_size: usize,
}

/// Validation rule trait
pub trait ValidationRule: Send + Sync {
    /// Rule name
    fn name(&self) -> &'static str;

    /// Validate configuration
    fn validate(&self, config: &Config, context: &ValidationContext) -> Vec<ValidationResult>;
}

impl Default for ConfigValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigValidator {
    /// Create a new configuration validator
    pub fn new() -> Self {
        let mut validator = ConfigValidator {
            strict_mode: false,
            custom_rules: Vec::new(),
            context: ValidationContext::default(),
        };

        // Add default validation rules
        validator.add_default_rules();

        validator
    }

    /// Create a strict validator
    pub fn strict() -> Self {
        let mut validator = Self::new();
        validator.strict_mode = true;
        validator
    }

    /// Set validation context
    pub fn with_context(mut self, context: ValidationContext) -> Self {
        self.context = context;
        self
    }

    /// Add custom validation rule
    pub fn add_rule(mut self, rule: Box<dyn ValidationRule>) -> Self {
        self.custom_rules.push(rule);
        self
    }

    /// Validate configuration
    pub fn validate(&self, config: &Config) -> Result<Vec<ValidationResult>> {
        let mut results = Vec::new();

        // Core parameter validation
        results.extend(self.validate_core_parameters(config));

        // Objective-specific validation
        results.extend(self.validate_objective_parameters(config));

        // Device-specific validation
        results.extend(self.validate_device_parameters(config));

        // Feature-specific validation
        results.extend(self.validate_feature_parameters(config));

        // Performance validation
        results.extend(self.validate_performance_parameters(config));

        // Consistency validation
        results.extend(self.validate_consistency(config));

        // Custom rule validation
        for rule in &self.custom_rules {
            results.extend(rule.validate(config, &self.context));
        }

        Ok(results)
    }

    /// Add default validation rules
    fn add_default_rules(&mut self) {
        // Add built-in validation rules
        self.custom_rules.push(Box::new(LearningRateRule));
        self.custom_rules.push(Box::new(NumLeavesRule));
        self.custom_rules.push(Box::new(RegularizationRule));
        self.custom_rules.push(Box::new(SamplingRule));
        self.custom_rules.push(Box::new(EarlyStoppingRule));
        self.custom_rules.push(Box::new(MemoryUsageRule));
        self.custom_rules.push(Box::new(PerformanceRule));
    }

    /// Validate core training parameters
    fn validate_core_parameters(&self, config: &Config) -> Vec<ValidationResult> {
        let mut results = Vec::new();

        // Validate num_iterations
        if config.num_iterations == 0 {
            results.push(ValidationResult::Error(ValidationError {
                parameter: "num_iterations".to_string(),
                value: config.num_iterations.to_string(),
                message: "Must be positive".to_string(),
                valid_range: Some("1 to 100000".to_string()),
            }));
        } else if config.num_iterations > 10000 {
            results.push(ValidationResult::Warning(ValidationWarning {
                parameter: "num_iterations".to_string(),
                value: config.num_iterations.to_string(),
                message: "Very large number of iterations may cause overfitting".to_string(),
                suggestion: Some("Consider using early stopping".to_string()),
            }));
        }

        // Validate learning_rate
        if config.learning_rate <= 0.0 || config.learning_rate > 1.0 {
            results.push(ValidationResult::Error(ValidationError {
                parameter: "learning_rate".to_string(),
                value: config.learning_rate.to_string(),
                message: "Must be in range (0.0, 1.0]".to_string(),
                valid_range: Some("(0.0, 1.0]".to_string()),
            }));
        } else if config.learning_rate > 0.3 {
            results.push(ValidationResult::Warning(ValidationWarning {
                parameter: "learning_rate".to_string(),
                value: config.learning_rate.to_string(),
                message: "High learning rate may cause training instability".to_string(),
                suggestion: Some("Consider using a lower learning rate (0.01-0.1)".to_string()),
            }));
        }

        // Validate num_leaves
        if config.num_leaves < 2 {
            results.push(ValidationResult::Error(ValidationError {
                parameter: "num_leaves".to_string(),
                value: config.num_leaves.to_string(),
                message: "Must be at least 2".to_string(),
                valid_range: Some("2 to 131072".to_string()),
            }));
        } else if config.num_leaves > 10000 {
            results.push(ValidationResult::Warning(ValidationWarning {
                parameter: "num_leaves".to_string(),
                value: config.num_leaves.to_string(),
                message: "Very large number of leaves may cause overfitting".to_string(),
                suggestion: Some(
                    "Consider using fewer leaves or increase regularization".to_string(),
                ),
            }));
        }

        // Validate max_depth
        if config.max_depth == 0 {
            results.push(ValidationResult::Error(ValidationError {
                parameter: "max_depth".to_string(),
                value: config.max_depth.to_string(),
                message: "Must be positive or -1 for unlimited".to_string(),
                valid_range: Some("-1 or 1 to 100".to_string()),
            }));
        } else if config.max_depth > 50 {
            results.push(ValidationResult::Warning(ValidationWarning {
                parameter: "max_depth".to_string(),
                value: config.max_depth.to_string(),
                message: "Very deep trees may cause overfitting".to_string(),
                suggestion: Some("Consider using shallower trees or regularization".to_string()),
            }));
        }

        results
    }

    /// Validate objective-specific parameters
    fn validate_objective_parameters(&self, config: &Config) -> Vec<ValidationResult> {
        let mut results = Vec::new();

        // Validate num_class for multiclass
        if config.objective == ObjectiveType::Multiclass {
            if config.num_class < 2 {
                results.push(ValidationResult::Error(ValidationError {
                    parameter: "num_class".to_string(),
                    value: config.num_class.to_string(),
                    message: "Must be at least 2 for multiclass objective".to_string(),
                    valid_range: Some("2 to 10000".to_string()),
                }));
            } else if config.num_class > 1000 {
                results.push(ValidationResult::Warning(ValidationWarning {
                    parameter: "num_class".to_string(),
                    value: config.num_class.to_string(),
                    message: "Very large number of classes may impact performance".to_string(),
                    suggestion: Some(
                        "Consider hierarchical classification or feature engineering".to_string(),
                    ),
                }));
            }
        } else if config.num_class > 1 {
            results.push(ValidationResult::Warning(ValidationWarning {
                parameter: "num_class".to_string(),
                value: config.num_class.to_string(),
                message: "num_class > 1 is only relevant for multiclass objective".to_string(),
                suggestion: Some(
                    "Set num_class to 1 for regression/binary classification".to_string(),
                ),
            }));
        }

        // Validate scale_pos_weight
        if config.scale_pos_weight <= 0.0 {
            results.push(ValidationResult::Error(ValidationError {
                parameter: "scale_pos_weight".to_string(),
                value: config.scale_pos_weight.to_string(),
                message: "Must be positive".to_string(),
                valid_range: Some("(0.0, inf)".to_string()),
            }));
        } else if config.scale_pos_weight > 100.0 {
            results.push(ValidationResult::Warning(ValidationWarning {
                parameter: "scale_pos_weight".to_string(),
                value: config.scale_pos_weight.to_string(),
                message: "Very high positive class weight may cause training instability"
                    .to_string(),
                suggestion: Some("Consider using is_unbalance=true instead".to_string()),
            }));
        }

        results
    }

    /// Validate device-specific parameters
    fn validate_device_parameters(&self, config: &Config) -> Vec<ValidationResult> {
        let mut results = Vec::new();

        // Validate num_threads
        if config.num_threads > 0 {
            let num_cpus = num_cpus::get();
            if config.num_threads > num_cpus * 2 {
                results.push(ValidationResult::Warning(ValidationWarning {
                    parameter: "num_threads".to_string(),
                    value: config.num_threads.to_string(),
                    message: format!("More threads than recommended for {} CPU cores", num_cpus),
                    suggestion: Some(format!("Consider using {} threads", num_cpus)),
                }));
            }
        }

        // Validate GPU parameters
        if config.device_type == DeviceType::GPU {
            if config.gpu_platform_id < -1 {
                results.push(ValidationResult::Error(ValidationError {
                    parameter: "gpu_platform_id".to_string(),
                    value: config.gpu_platform_id.to_string(),
                    message: "Must be >= -1".to_string(),
                    valid_range: Some("-1 or 0 to platform_count-1".to_string()),
                }));
            }

            if config.gpu_device_id < -1 {
                results.push(ValidationResult::Error(ValidationError {
                    parameter: "gpu_device_id".to_string(),
                    value: config.gpu_device_id.to_string(),
                    message: "Must be >= -1".to_string(),
                    valid_range: Some("-1 or 0 to device_count-1".to_string()),
                }));
            }

            // Check if GPU is actually available
            if let Some(ref sys_caps) = self.context.system_capabilities {
                if !sys_caps.gpu_available {
                    results.push(ValidationResult::Error(ValidationError {
                        parameter: "device_type".to_string(),
                        value: "GPU".to_string(),
                        message: "GPU not available on this system".to_string(),
                        valid_range: Some("CPU".to_string()),
                    }));
                }
            }
        }

        results
    }

    /// Validate feature-specific parameters
    fn validate_feature_parameters(&self, config: &Config) -> Vec<ValidationResult> {
        let mut results = Vec::new();

        // Validate max_bin
        if config.max_bin < 2 {
            results.push(ValidationResult::Error(ValidationError {
                parameter: "max_bin".to_string(),
                value: config.max_bin.to_string(),
                message: "Must be at least 2".to_string(),
                valid_range: Some("2 to 65535".to_string()),
            }));
        } else if config.max_bin > 65535 {
            results.push(ValidationResult::Error(ValidationError {
                parameter: "max_bin".to_string(),
                value: config.max_bin.to_string(),
                message: "Cannot exceed 65535".to_string(),
                valid_range: Some("2 to 65535".to_string()),
            }));
        } else if config.max_bin > 1000 {
            results.push(ValidationResult::Warning(ValidationWarning {
                parameter: "max_bin".to_string(),
                value: config.max_bin.to_string(),
                message: "Very large number of bins may increase memory usage".to_string(),
                suggestion: Some(
                    "Consider using fewer bins (64-255) for better performance".to_string(),
                ),
            }));
        }

        // Validate feature_fraction
        if config.feature_fraction <= 0.0 || config.feature_fraction > 1.0 {
            results.push(ValidationResult::Error(ValidationError {
                parameter: "feature_fraction".to_string(),
                value: config.feature_fraction.to_string(),
                message: "Must be in range (0.0, 1.0]".to_string(),
                valid_range: Some("(0.0, 1.0]".to_string()),
            }));
        } else if config.feature_fraction < 0.1 {
            results.push(ValidationResult::Warning(ValidationWarning {
                parameter: "feature_fraction".to_string(),
                value: config.feature_fraction.to_string(),
                message: "Very low feature fraction may reduce model quality".to_string(),
                suggestion: Some("Consider using a higher feature fraction (0.5-1.0)".to_string()),
            }));
        }

        // Validate bagging_fraction
        if config.bagging_fraction <= 0.0 || config.bagging_fraction > 1.0 {
            results.push(ValidationResult::Error(ValidationError {
                parameter: "bagging_fraction".to_string(),
                value: config.bagging_fraction.to_string(),
                message: "Must be in range (0.0, 1.0]".to_string(),
                valid_range: Some("(0.0, 1.0]".to_string()),
            }));
        } else if config.bagging_fraction < 0.1 {
            results.push(ValidationResult::Warning(ValidationWarning {
                parameter: "bagging_fraction".to_string(),
                value: config.bagging_fraction.to_string(),
                message: "Very low bagging fraction may reduce model quality".to_string(),
                suggestion: Some("Consider using a higher bagging fraction (0.5-1.0)".to_string()),
            }));
        }

        results
    }

    /// Validate performance-related parameters
    fn validate_performance_parameters(&self, config: &Config) -> Vec<ValidationResult> {
        let mut results = Vec::new();

        // Check dataset size vs configuration
        if let Some(ref dataset_info) = self.context.dataset_info {
            // Check min_data_in_leaf vs dataset size
            if config.min_data_in_leaf as usize > dataset_info.num_samples / 10 {
                results.push(ValidationResult::Warning(ValidationWarning {
                    parameter: "min_data_in_leaf".to_string(),
                    value: config.min_data_in_leaf.to_string(),
                    message: "min_data_in_leaf is very large relative to dataset size".to_string(),
                    suggestion: Some("Consider using a smaller value".to_string()),
                }));
            }

            // Check memory requirements
            let estimated_memory_mb = self.estimate_memory_usage(config, dataset_info);
            if let Some(ref sys_caps) = self.context.system_capabilities {
                if estimated_memory_mb > sys_caps.available_memory_mb {
                    results.push(ValidationResult::Error(ValidationError {
                        parameter: "configuration".to_string(),
                        value: "memory_usage".to_string(),
                        message: format!(
                            "Estimated memory usage ({} MB) exceeds available memory ({} MB)",
                            estimated_memory_mb, sys_caps.available_memory_mb
                        ),
                        valid_range: Some(
                            "Reduce max_bin, num_leaves, or use less data".to_string(),
                        ),
                    }));
                }
            }
        }

        results
    }

    /// Validate parameter consistency
    fn validate_consistency(&self, config: &Config) -> Vec<ValidationResult> {
        let mut results = Vec::new();

        // Check conflicting histogram construction modes
        if config.force_col_wise && config.force_row_wise {
            results.push(ValidationResult::Error(ValidationError {
                parameter: "force_col_wise/force_row_wise".to_string(),
                value: "both true".to_string(),
                message: "Cannot force both column-wise and row-wise histogram construction"
                    .to_string(),
                valid_range: Some("Set only one to true".to_string()),
            }));
        }

        // Check early stopping consistency
        if config.early_stopping_rounds.is_some() && config.metric.is_empty() {
            results.push(ValidationResult::Warning(ValidationWarning {
                parameter: "early_stopping_rounds".to_string(),
                value: config.early_stopping_rounds.unwrap().to_string(),
                message: "Early stopping enabled but no metric specified".to_string(),
                suggestion: Some("Specify a metric for early stopping".to_string()),
            }));
        }

        // Check DART-specific parameters
        if config.boosting_type == BoostingType::DART {
            if config.drop_rate <= 0.0 || config.drop_rate > 1.0 {
                results.push(ValidationResult::Error(ValidationError {
                    parameter: "drop_rate".to_string(),
                    value: config.drop_rate.to_string(),
                    message: "Must be in range (0.0, 1.0] for DART boosting".to_string(),
                    valid_range: Some("(0.0, 1.0]".to_string()),
                }));
            }

            if config.skip_drop < 0.0 || config.skip_drop > 1.0 {
                results.push(ValidationResult::Error(ValidationError {
                    parameter: "skip_drop".to_string(),
                    value: config.skip_drop.to_string(),
                    message: "Must be in range [0.0, 1.0] for DART boosting".to_string(),
                    valid_range: Some("[0.0, 1.0]".to_string()),
                }));
            }
        }

        results
    }

    /// Estimate memory usage for the configuration
    fn estimate_memory_usage(&self, config: &Config, dataset_info: &DatasetInfo) -> usize {
        let base_memory = dataset_info.memory_size_mb;
        let histogram_memory = (config.max_bin * dataset_info.num_features * 8) / (1024 * 1024); // 8 bytes per bin
        let tree_memory = (config.num_leaves * config.num_iterations * 64) / (1024 * 1024); // 64 bytes per leaf
        let additional_memory = base_memory / 2; // Buffers and temporary data

        base_memory + histogram_memory + tree_memory + additional_memory
    }
}

/// Learning rate validation rule
struct LearningRateRule;

impl ValidationRule for LearningRateRule {
    fn name(&self) -> &'static str {
        "learning_rate"
    }

    fn validate(&self, config: &Config, _context: &ValidationContext) -> Vec<ValidationResult> {
        let mut results = Vec::new();

        if config.learning_rate <= 0.0 || config.learning_rate > 1.0 {
            results.push(ValidationResult::Error(ValidationError {
                parameter: "learning_rate".to_string(),
                value: config.learning_rate.to_string(),
                message: "Must be in range (0.0, 1.0]".to_string(),
                valid_range: Some("(0.0, 1.0]".to_string()),
            }));
        }

        results
    }
}

/// Number of leaves validation rule
struct NumLeavesRule;

impl ValidationRule for NumLeavesRule {
    fn name(&self) -> &'static str {
        "num_leaves"
    }

    fn validate(&self, config: &Config, _context: &ValidationContext) -> Vec<ValidationResult> {
        let mut results = Vec::new();

        if config.num_leaves < 2 {
            results.push(ValidationResult::Error(ValidationError {
                parameter: "num_leaves".to_string(),
                value: config.num_leaves.to_string(),
                message: "Must be at least 2".to_string(),
                valid_range: Some("2 to 131072".to_string()),
            }));
        }

        results
    }
}

/// Regularization validation rule
struct RegularizationRule;

impl ValidationRule for RegularizationRule {
    fn name(&self) -> &'static str {
        "regularization"
    }

    fn validate(&self, config: &Config, _context: &ValidationContext) -> Vec<ValidationResult> {
        let mut results = Vec::new();

        if config.lambda_l1 < 0.0 {
            results.push(ValidationResult::Error(ValidationError {
                parameter: "lambda_l1".to_string(),
                value: config.lambda_l1.to_string(),
                message: "Must be non-negative".to_string(),
                valid_range: Some("[0.0, inf)".to_string()),
            }));
        }

        if config.lambda_l2 < 0.0 {
            results.push(ValidationResult::Error(ValidationError {
                parameter: "lambda_l2".to_string(),
                value: config.lambda_l2.to_string(),
                message: "Must be non-negative".to_string(),
                valid_range: Some("[0.0, inf)".to_string()),
            }));
        }

        results
    }
}

/// Sampling validation rule
struct SamplingRule;

impl ValidationRule for SamplingRule {
    fn name(&self) -> &'static str {
        "sampling"
    }

    fn validate(&self, config: &Config, _context: &ValidationContext) -> Vec<ValidationResult> {
        let mut results = Vec::new();

        if config.feature_fraction <= 0.0 || config.feature_fraction > 1.0 {
            results.push(ValidationResult::Error(ValidationError {
                parameter: "feature_fraction".to_string(),
                value: config.feature_fraction.to_string(),
                message: "Must be in range (0.0, 1.0]".to_string(),
                valid_range: Some("(0.0, 1.0]".to_string()),
            }));
        }

        if config.bagging_fraction <= 0.0 || config.bagging_fraction > 1.0 {
            results.push(ValidationResult::Error(ValidationError {
                parameter: "bagging_fraction".to_string(),
                value: config.bagging_fraction.to_string(),
                message: "Must be in range (0.0, 1.0]".to_string(),
                valid_range: Some("(0.0, 1.0]".to_string()),
            }));
        }

        results
    }
}

/// Early stopping validation rule
struct EarlyStoppingRule;

impl ValidationRule for EarlyStoppingRule {
    fn name(&self) -> &'static str {
        "early_stopping"
    }

    fn validate(&self, config: &Config, _context: &ValidationContext) -> Vec<ValidationResult> {
        let mut results = Vec::new();

        if let Some(rounds) = config.early_stopping_rounds {
            if rounds == 0 {
                results.push(ValidationResult::Error(ValidationError {
                    parameter: "early_stopping_rounds".to_string(),
                    value: rounds.to_string(),
                    message: "Must be positive when specified".to_string(),
                    valid_range: Some("1 to 10000".to_string()),
                }));
            }
        }

        results
    }
}

/// Memory usage validation rule
struct MemoryUsageRule;

impl ValidationRule for MemoryUsageRule {
    fn name(&self) -> &'static str {
        "memory_usage"
    }

    fn validate(&self, _config: &Config, _context: &ValidationContext) -> Vec<ValidationResult> {
        // Memory usage validation would require dataset information
        Vec::new()
    }
}

/// Performance validation rule
struct PerformanceRule;

impl ValidationRule for PerformanceRule {
    fn name(&self) -> &'static str {
        "performance"
    }

    fn validate(&self, _config: &Config, _context: &ValidationContext) -> Vec<ValidationResult> {
        // Performance validation would require benchmarking
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::core::Config;

    #[test]
    fn test_validation_result_types() {
        let warning = ValidationWarning {
            parameter: "learning_rate".to_string(),
            value: "0.5".to_string(),
            message: "High learning rate".to_string(),
            suggestion: Some("Use lower rate".to_string()),
        };

        let error = ValidationError {
            parameter: "num_leaves".to_string(),
            value: "1".to_string(),
            message: "Too few leaves".to_string(),
            valid_range: Some("2 to 131072".to_string()),
        };

        assert_eq!(warning.parameter, "learning_rate");
        assert_eq!(error.parameter, "num_leaves");
        assert!(error.to_string().contains("Too few leaves"));
    }

    #[test]
    fn test_config_validator_creation() {
        let validator = ConfigValidator::new();
        assert!(!validator.strict_mode);
        assert!(!validator.custom_rules.is_empty());

        let strict_validator = ConfigValidator::strict();
        assert!(strict_validator.strict_mode);
    }

    #[test]
    fn test_config_validation_valid() {
        let config = Config::default();
        let validator = ConfigValidator::new();

        let results = validator.validate(&config).unwrap();

        // Should have no errors, might have warnings
        let has_errors = results
            .iter()
            .any(|r| matches!(r, ValidationResult::Error(_)));
        assert!(!has_errors);
    }

    #[test]
    fn test_config_validation_invalid() {
        let mut config = Config::default();
        config.learning_rate = -0.1;
        config.num_leaves = 1;

        let validator = ConfigValidator::new();
        let results = validator.validate(&config).unwrap();

        // Should have errors
        let has_errors = results
            .iter()
            .any(|r| matches!(r, ValidationResult::Error(_)));
        assert!(has_errors);
    }

    #[test]
    fn test_validation_rules() {
        let config = Config::default();
        let context = ValidationContext::default();

        let lr_rule = LearningRateRule;
        let results = lr_rule.validate(&config, &context);
        assert!(!results
            .iter()
            .any(|r| matches!(r, ValidationResult::Error(_))));

        let leaves_rule = NumLeavesRule;
        let results = leaves_rule.validate(&config, &context);
        assert!(!results
            .iter()
            .any(|r| matches!(r, ValidationResult::Error(_))));
    }

    #[test]
    fn test_validation_with_context() {
        let config = Config::default();
        let context = ValidationContext {
            dataset_info: Some(DatasetInfo {
                num_samples: 1000,
                num_features: 10,
                num_classes: Some(2),
                feature_types: vec![FeatureType::Numerical; 10],
                has_missing_values: false,
                memory_size_mb: 10,
            }),
            system_capabilities: Some(SystemCapabilities {
                available_memory_mb: 8192,
                num_cpu_cores: 8,
                gpu_available: false,
                gpu_memory_mb: None,
            }),
            performance_requirements: None,
        };

        let validator = ConfigValidator::new().with_context(context);
        let results = validator.validate(&config).unwrap();

        // Should validate successfully with context
        let has_errors = results
            .iter()
            .any(|r| matches!(r, ValidationResult::Error(_)));
        assert!(!has_errors);
    }

    #[test]
    fn test_multiclass_validation() {
        let mut config = Config::default();
        config.objective = ObjectiveType::Multiclass;
        config.num_class = 1; // Invalid for multiclass

        let validator = ConfigValidator::new();
        let results = validator.validate(&config).unwrap();

        // Should have an error for invalid num_class
        let has_errors = results
            .iter()
            .any(|r| matches!(r, ValidationResult::Error(_)));
        assert!(has_errors);
    }
}
