//! Core configuration structures and implementation for Pure Rust LightGBM.
//!
//! This module provides the main configuration structure and builder pattern
//! for setting up LightGBM training parameters, device settings, and other
//! configuration options.

use crate::core::constants::*;
use crate::core::error::{LightGBMError, Result};

use crate::core::types::*;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Main configuration structure for LightGBM training and prediction.
///
/// This structure contains all parameters needed to configure a LightGBM
/// model, including training parameters, regularization settings, device
/// configuration, and other options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    // Core training parameters
    /// Objective function type (regression, classification, etc.)
    pub objective: ObjectiveType,
    /// Number of boosting iterations
    pub num_iterations: usize,
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Maximum number of leaves in one tree
    pub num_leaves: usize,
    /// Maximum depth of tree (-1 for unlimited)
    pub max_depth: i32,
    /// Boosting algorithm type
    pub boosting_type: BoostingType,
    /// Tree learner algorithm type
    pub tree_learner: TreeLearnerType,

    // Regularization parameters
    /// L1 regularization term
    pub lambda_l1: f64,
    /// L2 regularization term
    pub lambda_l2: f64,
    /// Minimum number of data points in a leaf
    pub min_data_in_leaf: DataSize,
    /// Minimum sum of hessian values in a leaf
    pub min_sum_hessian_in_leaf: f64,
    /// Minimum gain required to make a split
    pub min_gain_to_split: f64,
    /// Maximum delta step for updates
    pub max_delta_step: f64,

    // Sampling parameters
    /// Fraction of features to use for each tree
    pub feature_fraction: f64,
    /// Fraction of features to use for each node
    pub feature_fraction_bynode: f64,
    /// Random seed for feature sampling
    pub feature_fraction_seed: u64,
    /// Fraction of data to use for bagging
    pub bagging_fraction: f64,
    /// Frequency of bagging (0 = disabled)
    pub bagging_freq: usize,
    /// Random seed for bagging
    pub bagging_seed: u64,
    /// Fraction of positive samples to use for bagging
    pub pos_bagging_fraction: f64,
    /// Fraction of negative samples to use for bagging
    pub neg_bagging_fraction: f64,
    /// Whether to use query-based bagging for ranking tasks
    pub bagging_by_query: bool,
    /// Data sampling strategy (bagging, goss, etc.)
    pub data_sample_strategy: String,
    /// Top rate for GOSS sampling (fraction of large gradient data to keep)
    pub top_rate: f64,
    /// Other rate for GOSS sampling (fraction of small gradient data to keep)
    pub other_rate: f64,

    // Feature parameters
    /// Maximum number of bins for feature discretization
    pub max_bin: usize,
    /// Minimum data per categorical group
    pub min_data_per_group: usize,
    /// Smoothing parameter for categorical features
    pub cat_smooth: f64,
    /// L2 regularization for categorical features
    pub cat_l2: f64,
    /// Maximum categories to use one-hot encoding
    pub max_cat_to_onehot: usize,
    /// Top-k categories to consider
    pub top_k: usize,

    // Device configuration
    /// Device type (CPU, GPU, etc.)
    pub device_type: DeviceType,
    /// Number of threads for parallel processing
    pub num_threads: usize,
    /// GPU platform ID
    pub gpu_platform_id: i32,
    /// GPU device ID
    pub gpu_device_id: i32,
    /// Use double precision on GPU
    pub gpu_use_dp: bool,

    // Multiclass parameters
    /// Number of classes for multiclass classification
    pub num_class: usize,
    /// Whether dataset is unbalanced
    pub is_unbalance: bool,
    /// Weight for positive class in binary classification
    pub scale_pos_weight: f64,

    // Early stopping
    /// Early stopping rounds (None = disabled)
    pub early_stopping_rounds: Option<usize>,
    /// Tolerance for early stopping
    pub early_stopping_tolerance: f64,
    /// Use only first metric for early stopping
    pub first_metric_only: bool,

    // Validation and metrics
    /// Metrics to evaluate during training
    pub metric: Vec<MetricType>,
    /// Frequency of metric evaluation
    pub metric_freq: usize,
    /// Whether to compute training metrics
    pub is_training_metric: bool,

    // Output control
    /// Verbosity level for logging
    pub verbosity: VerbosityLevel,
    /// Output model file path
    pub output_model: Option<String>,
    /// Input model file path for continuing training
    pub input_model: Option<String>,
    /// Save model in binary format
    pub save_binary: bool,

    // Reproducibility
    /// Random seed for reproducible results
    pub random_seed: u64,
    /// Force deterministic behavior
    pub deterministic: bool,

    // Advanced parameters
    /// Force column-wise histogram building
    pub force_col_wise: bool,
    /// Force row-wise histogram building
    pub force_row_wise: bool,
    /// Maximum bins for each feature (if None, use max_bin for all)
    pub max_bin_by_feature: Option<Vec<usize>>,
    /// Monotone constraints for features (-1, 0, 1 for decreasing, none, increasing)
    pub monotone_constraints: Option<Vec<i8>>,
    /// Method for monotone constraints enforcement
    pub monotone_constraints_method: String,
    /// Penalty for violating monotone constraints
    pub monotone_penalty: f64,
    /// Interaction constraints between features
    pub interaction_constraints: Option<Vec<Vec<usize>>>,
    /// Filename containing forced splits information
    pub forcedsplits_filename: Option<String>,
    /// Decay rate for refitting leaf values
    pub refit_decay_rate: f64,
    /// Path smoothing parameter
    pub path_smooth: f64,

    // Network parameters (for distributed training)
    /// Number of machines for distributed training
    pub num_machines: usize,
    /// Local port for listening in distributed training
    pub local_listen_port: u16,
    /// Timeout for network operations in seconds
    pub time_out: u64,
    /// Filename containing machine list for distributed training
    pub machine_list_filename: Option<String>,

    // Additional parameters
    /// Extra random seed for additional randomness
    pub extra_seed: u64,
    /// Enable extremely randomized trees
    pub extra_trees: bool,
    /// Probability of skipping dropout
    pub skip_drop: f64,
    /// Dropout rate for DART boosting
    pub drop_rate: f64,
    /// Maximum number of dropped trees in DART
    pub max_drop: usize,
    /// Use uniform dropout in DART
    pub uniform_drop: bool,
    /// Enable XGBoost-style DART mode
    pub xgboost_dart_mode: bool,
    /// Random seed for dropout
    pub drop_seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            // Core training parameters
            objective: ObjectiveType::Regression,
            num_iterations: DEFAULT_NUM_ITERATIONS,
            learning_rate: DEFAULT_LEARNING_RATE,
            num_leaves: DEFAULT_NUM_LEAVES,
            max_depth: DEFAULT_MAX_DEPTH,
            boosting_type: BoostingType::GBDT,
            tree_learner: TreeLearnerType::Serial,

            // Regularization parameters
            lambda_l1: DEFAULT_LAMBDA_L1,
            lambda_l2: DEFAULT_LAMBDA_L2,
            min_data_in_leaf: DEFAULT_MIN_DATA_IN_LEAF,
            min_sum_hessian_in_leaf: DEFAULT_MIN_SUM_HESSIAN_IN_LEAF,
            min_gain_to_split: 0.0,
            max_delta_step: 0.0,

            // Sampling parameters
            feature_fraction: DEFAULT_FEATURE_FRACTION,
            feature_fraction_bynode: 1.0,
            feature_fraction_seed: DEFAULT_RANDOM_SEED,
            bagging_fraction: DEFAULT_BAGGING_FRACTION,
            bagging_freq: DEFAULT_BAGGING_FREQ,
            bagging_seed: DEFAULT_RANDOM_SEED,
            pos_bagging_fraction: 1.0,
            neg_bagging_fraction: 1.0,
            bagging_by_query: false,
            data_sample_strategy: "bagging".to_string(),
            top_rate: 0.2,
            other_rate: 0.1,

            // Feature parameters
            max_bin: DEFAULT_MAX_BIN,
            min_data_per_group: 100,
            cat_smooth: 10.0,
            cat_l2: 10.0,
            max_cat_to_onehot: 4,
            top_k: 20,

            // Device configuration
            device_type: DeviceType::CPU,
            num_threads: DEFAULT_NUM_THREADS,
            gpu_platform_id: -1,
            gpu_device_id: -1,
            gpu_use_dp: false,

            // Multiclass parameters
            num_class: DEFAULT_NUM_CLASS,
            is_unbalance: false,
            scale_pos_weight: 1.0,

            // Early stopping
            early_stopping_rounds: Some(DEFAULT_EARLY_STOPPING_ROUNDS),
            early_stopping_tolerance: DEFAULT_EARLY_STOPPING_TOLERANCE,
            first_metric_only: false,

            // Validation and metrics
            metric: vec![MetricType::None],
            metric_freq: 1,
            is_training_metric: false,

            // Output control
            verbosity: DEFAULT_VERBOSITY,
            output_model: None,
            input_model: None,
            save_binary: false,

            // Reproducibility
            random_seed: DEFAULT_RANDOM_SEED,
            deterministic: false,

            // Advanced parameters
            force_col_wise: false,
            force_row_wise: false,
            max_bin_by_feature: None,
            monotone_constraints: None,
            monotone_constraints_method: "basic".to_string(),
            monotone_penalty: 0.0,
            interaction_constraints: None,
            forcedsplits_filename: None,
            refit_decay_rate: 0.9,
            path_smooth: 0.0,

            // Network parameters
            num_machines: 1,
            local_listen_port: 12400,
            time_out: 120,
            machine_list_filename: None,

            // Additional parameters
            extra_seed: DEFAULT_RANDOM_SEED,
            extra_trees: false,
            skip_drop: 0.5,
            drop_rate: 0.1,
            max_drop: 50,
            uniform_drop: false,
            xgboost_dart_mode: false,
            drop_seed: DEFAULT_RANDOM_SEED,
        }
    }
}

impl Config {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate the configuration parameters
    pub fn validate(&self) -> Result<()> {
        // Validate learning rate
        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            return Err(LightGBMError::invalid_parameter(
                "learning_rate",
                self.learning_rate.to_string(),
                "must be in range (0.0, 1.0]",
            ));
        }

        // Validate num_leaves
        if self.num_leaves < 2 {
            return Err(LightGBMError::invalid_parameter(
                "num_leaves",
                self.num_leaves.to_string(),
                "must be at least 2",
            ));
        }

        // Validate feature_fraction
        if self.feature_fraction <= 0.0 || self.feature_fraction > 1.0 {
            return Err(LightGBMError::invalid_parameter(
                "feature_fraction",
                self.feature_fraction.to_string(),
                "must be in range (0.0, 1.0]",
            ));
        }

        // Validate bagging_fraction
        if self.bagging_fraction <= 0.0 || self.bagging_fraction > 1.0 {
            return Err(LightGBMError::invalid_parameter(
                "bagging_fraction",
                self.bagging_fraction.to_string(),
                "must be in range (0.0, 1.0]",
            ));
        }

        // Validate regularization parameters
        if self.lambda_l1 < 0.0 {
            return Err(LightGBMError::invalid_parameter(
                "lambda_l1",
                self.lambda_l1.to_string(),
                "must be non-negative",
            ));
        }

        if self.lambda_l2 < 0.0 {
            return Err(LightGBMError::invalid_parameter(
                "lambda_l2",
                self.lambda_l2.to_string(),
                "must be non-negative",
            ));
        }

        // Validate min_data_in_leaf
        if self.min_data_in_leaf < 1 {
            return Err(LightGBMError::invalid_parameter(
                "min_data_in_leaf",
                self.min_data_in_leaf.to_string(),
                "must be at least 1",
            ));
        }

        // Validate min_sum_hessian_in_leaf
        if self.min_sum_hessian_in_leaf < 0.0 {
            return Err(LightGBMError::invalid_parameter(
                "min_sum_hessian_in_leaf",
                self.min_sum_hessian_in_leaf.to_string(),
                "must be non-negative",
            ));
        }

        // Validate max_bin
        if self.max_bin < 2 || self.max_bin > 65535 {
            return Err(LightGBMError::invalid_parameter(
                "max_bin",
                self.max_bin.to_string(),
                "must be in range [2, 65535]",
            ));
        }

        // Validate num_class for multiclass
        if self.objective == ObjectiveType::Multiclass && self.num_class < 2 {
            return Err(LightGBMError::invalid_parameter(
                "num_class",
                self.num_class.to_string(),
                "must be at least 2 for multiclass objective",
            ));
        }

        // Validate num_threads
        if self.num_threads == 0 {
            // 0 means use all available cores, which is valid
        } else if self.num_threads > num_cpus::get() * 2 {
            log::warn!(
                "num_threads ({}) is much larger than available cores ({})",
                self.num_threads,
                num_cpus::get()
            );
        }

        // Validate GPU parameters
        if self.device_type == DeviceType::GPU {
            if self.gpu_platform_id < -1 {
                return Err(LightGBMError::invalid_parameter(
                    "gpu_platform_id",
                    self.gpu_platform_id.to_string(),
                    "must be >= -1",
                ));
            }
            if self.gpu_device_id < -1 {
                return Err(LightGBMError::invalid_parameter(
                    "gpu_device_id",
                    self.gpu_device_id.to_string(),
                    "must be >= -1",
                ));
            }
        }

        // Validate monotone constraints
        if let Some(ref constraints) = self.monotone_constraints {
            for (i, &constraint) in constraints.iter().enumerate() {
                if constraint < -1 || constraint > 1 {
                    return Err(LightGBMError::invalid_parameter(
                        &format!("monotone_constraints[{}]", i),
                        constraint.to_string(),
                        "must be in range [-1, 1]",
                    ));
                }
            }
        }

        // Validate early stopping
        if let Some(rounds) = self.early_stopping_rounds {
            if rounds == 0 {
                return Err(LightGBMError::invalid_parameter(
                    "early_stopping_rounds",
                    "0".to_string(),
                    "must be positive when specified",
                ));
            }
        }

        // Validate DART parameters
        if self.boosting_type == BoostingType::DART {
            if self.drop_rate < 0.0 || self.drop_rate > 1.0 {
                return Err(LightGBMError::invalid_parameter(
                    "drop_rate",
                    self.drop_rate.to_string(),
                    "must be in range [0.0, 1.0]",
                ));
            }
            if self.skip_drop < 0.0 || self.skip_drop > 1.0 {
                return Err(LightGBMError::invalid_parameter(
                    "skip_drop",
                    self.skip_drop.to_string(),
                    "must be in range [0.0, 1.0]",
                ));
            }
        }

        Ok(())
    }

    /// Load configuration from a file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .map_err(|e| LightGBMError::config(format!("Failed to read config file: {}", e)))?;

        let config = if path.extension().and_then(|s| s.to_str()) == Some("json") {
            serde_json::from_str(&content)
                .map_err(|e| LightGBMError::config(format!("Failed to parse JSON config: {}", e)))?
        } else if path.extension().and_then(|s| s.to_str()) == Some("toml") {
            toml::from_str(&content)
                .map_err(|e| LightGBMError::config(format!("Failed to parse TOML config: {}", e)))?
        } else {
            return Err(LightGBMError::config(
                "Unsupported config file format. Use .json or .toml",
            ));
        };

        Ok(config)
    }

    /// Save configuration to a file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let content = if path.extension().and_then(|s| s.to_str()) == Some("json") {
            serde_json::to_string_pretty(self)
                .map_err(|e| LightGBMError::config(format!("Failed to serialize to JSON: {}", e)))?
        } else if path.extension().and_then(|s| s.to_str()) == Some("toml") {
            toml::to_string_pretty(self)
                .map_err(|e| LightGBMError::config(format!("Failed to serialize to TOML: {}", e)))?
        } else {
            return Err(LightGBMError::config(
                "Unsupported config file format. Use .json or .toml",
            ));
        };

        std::fs::write(path, content)
            .map_err(|e| LightGBMError::config(format!("Failed to write config file: {}", e)))?;

        Ok(())
    }

    /// Load configuration from environment variables
    pub fn load_from_environment() -> Result<Self> {
        let mut config = Config::default();

        // Load common parameters from environment
        if let Ok(val) = std::env::var("LIGHTGBM_NUM_ITERATIONS") {
            config.num_iterations = val
                .parse()
                .map_err(|_| LightGBMError::config("Invalid LIGHTGBM_NUM_ITERATIONS"))?;
        }

        if let Ok(val) = std::env::var("LIGHTGBM_LEARNING_RATE") {
            config.learning_rate = val
                .parse()
                .map_err(|_| LightGBMError::config("Invalid LIGHTGBM_LEARNING_RATE"))?;
        }

        if let Ok(val) = std::env::var("LIGHTGBM_NUM_LEAVES") {
            config.num_leaves = val
                .parse()
                .map_err(|_| LightGBMError::config("Invalid LIGHTGBM_NUM_LEAVES"))?;
        }

        if let Ok(val) = std::env::var("LIGHTGBM_OBJECTIVE") {
            config.objective = match val.as_str() {
                "regression" => ObjectiveType::Regression,
                "binary" => ObjectiveType::Binary,
                "multiclass" => ObjectiveType::Multiclass,
                "ranking" => ObjectiveType::Ranking,
                _ => return Err(LightGBMError::config("Invalid LIGHTGBM_OBJECTIVE")),
            };
        }

        if let Ok(val) = std::env::var("LIGHTGBM_DEVICE_TYPE") {
            config.device_type = match val.as_str() {
                "cpu" => DeviceType::CPU,
                "gpu" => DeviceType::GPU,
                _ => return Err(LightGBMError::config("Invalid LIGHTGBM_DEVICE_TYPE")),
            };
        }

        if let Ok(val) = std::env::var("LIGHTGBM_NUM_THREADS") {
            config.num_threads = val
                .parse()
                .map_err(|_| LightGBMError::config("Invalid LIGHTGBM_NUM_THREADS"))?;
        }

        if let Ok(val) = std::env::var("LIGHTGBM_RANDOM_SEED") {
            config.random_seed = val
                .parse()
                .map_err(|_| LightGBMError::config("Invalid LIGHTGBM_RANDOM_SEED"))?;
        }

        config.validate()?;
        Ok(config)
    }

    /// Apply environment variable overrides to existing configuration
    pub fn apply_environment_overrides(&mut self) -> Result<()> {
        let env_config = Self::load_from_environment()?;

        // Only override non-default values
        if env_config.num_iterations != DEFAULT_NUM_ITERATIONS {
            self.num_iterations = env_config.num_iterations;
        }
        if env_config.learning_rate != DEFAULT_LEARNING_RATE {
            self.learning_rate = env_config.learning_rate;
        }
        if env_config.num_leaves != DEFAULT_NUM_LEAVES {
            self.num_leaves = env_config.num_leaves;
        }
        if env_config.objective != ObjectiveType::Regression {
            self.objective = env_config.objective;
        }
        if env_config.device_type != DeviceType::CPU {
            self.device_type = env_config.device_type;
        }
        if env_config.num_threads != DEFAULT_NUM_THREADS {
            self.num_threads = env_config.num_threads;
        }
        if env_config.random_seed != DEFAULT_RANDOM_SEED {
            self.random_seed = env_config.random_seed;
        }

        self.validate()
    }

    /// Merge this configuration with another configuration
    pub fn merge(&mut self, other: &Config) -> Result<()> {
        // Merge all parameters (other takes precedence)
        self.objective = other.objective;
        self.num_iterations = other.num_iterations;
        self.learning_rate = other.learning_rate;
        self.num_leaves = other.num_leaves;
        self.max_depth = other.max_depth;
        self.boosting_type = other.boosting_type;
        self.tree_learner = other.tree_learner;

        self.lambda_l1 = other.lambda_l1;
        self.lambda_l2 = other.lambda_l2;
        self.min_data_in_leaf = other.min_data_in_leaf;
        self.min_sum_hessian_in_leaf = other.min_sum_hessian_in_leaf;
        self.min_gain_to_split = other.min_gain_to_split;
        self.max_delta_step = other.max_delta_step;

        self.feature_fraction = other.feature_fraction;
        self.feature_fraction_bynode = other.feature_fraction_bynode;
        self.feature_fraction_seed = other.feature_fraction_seed;
        self.bagging_fraction = other.bagging_fraction;
        self.bagging_freq = other.bagging_freq;
        self.bagging_seed = other.bagging_seed;
        self.pos_bagging_fraction = other.pos_bagging_fraction;
        self.neg_bagging_fraction = other.neg_bagging_fraction;
        self.bagging_by_query = other.bagging_by_query;

        self.max_bin = other.max_bin;
        self.min_data_per_group = other.min_data_per_group;
        self.cat_smooth = other.cat_smooth;
        self.cat_l2 = other.cat_l2;
        self.max_cat_to_onehot = other.max_cat_to_onehot;
        self.top_k = other.top_k;

        self.device_type = other.device_type;
        self.num_threads = other.num_threads;
        self.gpu_platform_id = other.gpu_platform_id;
        self.gpu_device_id = other.gpu_device_id;
        self.gpu_use_dp = other.gpu_use_dp;

        self.num_class = other.num_class;
        self.is_unbalance = other.is_unbalance;
        self.scale_pos_weight = other.scale_pos_weight;

        self.early_stopping_rounds = other.early_stopping_rounds;
        self.early_stopping_tolerance = other.early_stopping_tolerance;
        self.first_metric_only = other.first_metric_only;

        self.metric = other.metric.clone();
        self.metric_freq = other.metric_freq;
        self.is_training_metric = other.is_training_metric;

        self.verbosity = other.verbosity;
        self.output_model = other.output_model.clone();
        self.input_model = other.input_model.clone();
        self.save_binary = other.save_binary;

        self.random_seed = other.random_seed;
        self.deterministic = other.deterministic;

        // Merge optional fields
        if other.max_bin_by_feature.is_some() {
            self.max_bin_by_feature = other.max_bin_by_feature.clone();
        }
        if other.monotone_constraints.is_some() {
            self.monotone_constraints = other.monotone_constraints.clone();
        }
        if other.interaction_constraints.is_some() {
            self.interaction_constraints = other.interaction_constraints.clone();
        }
        if other.forcedsplits_filename.is_some() {
            self.forcedsplits_filename = other.forcedsplits_filename.clone();
        }
        if other.machine_list_filename.is_some() {
            self.machine_list_filename = other.machine_list_filename.clone();
        }

        self.validate()
    }

    /// Get the effective number of threads (0 means use all available cores)
    pub fn effective_num_threads(&self) -> usize {
        if self.num_threads == 0 {
            num_cpus::get()
        } else {
            self.num_threads
        }
    }

    /// Check if GPU is enabled and available
    pub fn is_gpu_enabled(&self) -> bool {
        self.device_type == DeviceType::GPU
    }

    /// Get the number of trees per iteration (depends on objective)
    pub fn num_trees_per_iteration(&self) -> usize {
        match self.objective {
            ObjectiveType::Regression | ObjectiveType::Binary | ObjectiveType::Ranking => 1,
            ObjectiveType::Multiclass => self.num_class,
            _ => 1,
        }
    }

    /// Get total number of trees that will be trained
    pub fn total_num_trees(&self) -> usize {
        self.num_iterations * self.num_trees_per_iteration()
    }

    /// Check if early stopping is enabled
    pub fn is_early_stopping_enabled(&self) -> bool {
        self.early_stopping_rounds.is_some()
    }

    /// Get configuration as a parameter map (for debugging/serialization)
    pub fn as_parameter_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();

        map.insert("objective".to_string(), self.objective.to_string());
        map.insert(
            "num_iterations".to_string(),
            self.num_iterations.to_string(),
        );
        map.insert("learning_rate".to_string(), self.learning_rate.to_string());
        map.insert("num_leaves".to_string(), self.num_leaves.to_string());
        map.insert("max_depth".to_string(), self.max_depth.to_string());
        map.insert("boosting_type".to_string(), self.boosting_type.to_string());
        map.insert("tree_learner".to_string(), self.tree_learner.to_string());

        map.insert("lambda_l1".to_string(), self.lambda_l1.to_string());
        map.insert("lambda_l2".to_string(), self.lambda_l2.to_string());
        map.insert(
            "min_data_in_leaf".to_string(),
            self.min_data_in_leaf.to_string(),
        );
        map.insert(
            "min_sum_hessian_in_leaf".to_string(),
            self.min_sum_hessian_in_leaf.to_string(),
        );

        map.insert(
            "feature_fraction".to_string(),
            self.feature_fraction.to_string(),
        );
        map.insert(
            "bagging_fraction".to_string(),
            self.bagging_fraction.to_string(),
        );
        map.insert("bagging_freq".to_string(), self.bagging_freq.to_string());
        map.insert(
            "pos_bagging_fraction".to_string(),
            self.pos_bagging_fraction.to_string(),
        );
        map.insert(
            "neg_bagging_fraction".to_string(),
            self.neg_bagging_fraction.to_string(),
        );
        map.insert(
            "bagging_by_query".to_string(),
            self.bagging_by_query.to_string(),
        );

        map.insert("max_bin".to_string(), self.max_bin.to_string());
        map.insert("device_type".to_string(), self.device_type.to_string());
        map.insert("num_threads".to_string(), self.num_threads.to_string());
        map.insert("num_class".to_string(), self.num_class.to_string());
        map.insert("random_seed".to_string(), self.random_seed.to_string());

        if let Some(rounds) = self.early_stopping_rounds {
            map.insert("early_stopping_rounds".to_string(), rounds.to_string());
        }

        map
    }
}

/// Configuration builder for fluent configuration creation
#[derive(Debug, Clone)]
pub struct ConfigBuilder {
    config: Config,
    validation_errors: Vec<String>,
}

impl ConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        ConfigBuilder {
            config: Config::default(),
            validation_errors: Vec::new(),
        }
    }

    /// Set the objective function
    pub fn objective(mut self, objective: ObjectiveType) -> Self {
        self.config.objective = objective;
        self
    }

    /// Set the number of boosting iterations
    pub fn num_iterations(mut self, iterations: usize) -> Self {
        self.config.num_iterations = iterations;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, rate: f64) -> Self {
        if rate <= 0.0 || rate > 1.0 {
            self.validation_errors
                .push("learning_rate must be in range (0.0, 1.0]".to_string());
        }
        self.config.learning_rate = rate;
        self
    }

    /// Set the number of leaves
    pub fn num_leaves(mut self, leaves: usize) -> Self {
        if leaves < 2 {
            self.validation_errors
                .push("num_leaves must be at least 2".to_string());
        }
        self.config.num_leaves = leaves;
        self
    }

    /// Set the maximum tree depth
    pub fn max_depth(mut self, depth: i32) -> Self {
        self.config.max_depth = depth;
        self
    }

    /// Set L1 regularization parameter
    pub fn lambda_l1(mut self, lambda: f64) -> Self {
        if lambda < 0.0 {
            self.validation_errors
                .push("lambda_l1 must be non-negative".to_string());
        }
        self.config.lambda_l1 = lambda;
        self
    }

    /// Set L2 regularization parameter
    pub fn lambda_l2(mut self, lambda: f64) -> Self {
        if lambda < 0.0 {
            self.validation_errors
                .push("lambda_l2 must be non-negative".to_string());
        }
        self.config.lambda_l2 = lambda;
        self
    }

    /// Set device type
    pub fn device_type(mut self, device: DeviceType) -> Self {
        self.config.device_type = device;
        self
    }

    /// Set number of threads
    pub fn num_threads(mut self, threads: usize) -> Self {
        self.config.num_threads = threads;
        self
    }

    /// Set random seed
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = seed;
        self
    }

    /// Set minimum data points per leaf
    pub fn min_data_in_leaf(mut self, min_data: DataSize) -> Self {
        if min_data < 1 {
            self.validation_errors
                .push("min_data_in_leaf must be at least 1".to_string());
        }
        self.config.min_data_in_leaf = min_data;
        self
    }

    /// Set feature fraction for sampling
    pub fn feature_fraction(mut self, fraction: f64) -> Self {
        if fraction <= 0.0 || fraction > 1.0 {
            self.validation_errors
                .push("feature_fraction must be in range (0.0, 1.0]".to_string());
        }
        self.config.feature_fraction = fraction;
        self
    }

    /// Set bagging fraction for sampling
    pub fn bagging_fraction(mut self, fraction: f64) -> Self {
        if fraction <= 0.0 || fraction > 1.0 {
            self.validation_errors
                .push("bagging_fraction must be in range (0.0, 1.0]".to_string());
        }
        self.config.bagging_fraction = fraction;
        self
    }

    /// Set bagging frequency
    pub fn bagging_freq(mut self, freq: usize) -> Self {
        self.config.bagging_freq = freq;
        self
    }

    /// Set bagging seed
    pub fn bagging_seed(mut self, seed: u64) -> Self {
        self.config.bagging_seed = seed;
        self
    }

    /// Set maximum number of bins
    pub fn max_bin(mut self, max_bin: usize) -> Self {
        if max_bin < 2 {
            self.validation_errors
                .push("max_bin must be at least 2".to_string());
        }
        self.config.max_bin = max_bin;
        self
    }

    /// Set number of classes for multiclass classification
    pub fn num_class(mut self, num_class: usize) -> Self {
        if num_class < 2 {
            self.validation_errors
                .push("num_class must be at least 2".to_string());
        }
        self.config.num_class = num_class;
        self
    }

    /// Set early stopping rounds
    pub fn early_stopping_rounds(mut self, rounds: Option<usize>) -> Self {
        self.config.early_stopping_rounds = rounds;
        self
    }

    /// Set early stopping tolerance
    pub fn early_stopping_tolerance(mut self, tolerance: f64) -> Self {
        if tolerance < 0.0 {
            self.validation_errors
                .push("early_stopping_tolerance must be non-negative".to_string());
        }
        self.config.early_stopping_tolerance = tolerance;
        self
    }

    /// Set tree learner type
    pub fn tree_learner(mut self, learner: TreeLearnerType) -> Self {
        self.config.tree_learner = learner;
        self
    }

    /// Set verbosity level
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbosity = if verbose {
            VerbosityLevel::Info
        } else {
            VerbosityLevel::Silent
        };
        self
    }

    /// Set data sample strategy
    pub fn data_sample_strategy(mut self, strategy: String) -> Self {
        self.config.data_sample_strategy = strategy;
        self
    }

    /// Set GOSS top rate
    pub fn top_rate(mut self, rate: f64) -> Self {
        if rate <= 0.0 || rate > 1.0 {
            self.validation_errors
                .push("top_rate must be in range (0.0, 1.0]".to_string());
        }
        self.config.top_rate = rate;
        self
    }

    /// Set GOSS other rate
    pub fn other_rate(mut self, rate: f64) -> Self {
        if rate <= 0.0 || rate > 1.0 {
            self.validation_errors
                .push("other_rate must be in range (0.0, 1.0]".to_string());
        }
        self.config.other_rate = rate;
        self
    }

    /// Create configuration from parameter map (for hyperparameter optimization)
    pub fn from_params(params: &std::collections::HashMap<String, f64>) -> Self {
        let mut builder = Self::new();

        for (param, &value) in params {
            match param.as_str() {
                "learning_rate" => builder = builder.learning_rate(value),
                "num_leaves" => builder = builder.num_leaves(value as usize),
                "lambda_l1" => builder = builder.lambda_l1(value),
                "lambda_l2" => builder = builder.lambda_l2(value),
                "feature_fraction" => builder = builder.feature_fraction(value),
                "bagging_fraction" => builder = builder.bagging_fraction(value),
                "min_data_in_leaf" => builder = builder.min_data_in_leaf(value as DataSize),
                "max_bin" => builder = builder.max_bin(value as usize),
                _ => {} // Ignore unknown parameters
            }
        }

        builder
    }

    /// Build the configuration
    pub fn build(self) -> Result<Config> {
        if !self.validation_errors.is_empty() {
            return Err(LightGBMError::config(format!(
                "Configuration validation failed: {}",
                self.validation_errors.join(", ")
            )));
        }

        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration error type
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Invalid configuration parameter value
    #[error("Invalid parameter: {parameter} = {value}, {reason}")]
    InvalidParameter {
        /// Name of the invalid parameter
        parameter: String,
        /// Value that was provided
        value: String,
        /// Reason why the parameter is invalid
        reason: String,
    },
    /// Required configuration parameter is missing
    #[error("Missing required parameter: {parameter}")]
    MissingParameter {
        /// Name of the missing parameter
        parameter: String,
    },
    /// Configuration file processing error
    #[error("Configuration file error: {message}")]
    FileError {
        /// Error message describing the file issue
        message: String,
    },
    /// Configuration serialization/deserialization error
    #[error("Serialization error: {message}")]
    SerializationError {
        /// Error message describing the serialization issue
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.objective, ObjectiveType::Regression);
        assert_eq!(config.num_iterations, DEFAULT_NUM_ITERATIONS);
        assert_eq!(config.learning_rate, DEFAULT_LEARNING_RATE);
        assert_eq!(config.num_leaves, DEFAULT_NUM_LEAVES);
        assert_eq!(config.device_type, DeviceType::CPU);
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();

        // Valid configuration should pass
        assert!(config.validate().is_ok());

        // Invalid learning rate should fail
        config.learning_rate = -0.1;
        assert!(config.validate().is_err());

        // Invalid num_leaves should fail
        config.learning_rate = 0.1;
        config.num_leaves = 1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .objective(ObjectiveType::Binary)
            .num_iterations(500)
            .learning_rate(0.05)
            .num_leaves(63)
            .device_type(DeviceType::CPU)
            .build()
            .unwrap();

        assert_eq!(config.objective, ObjectiveType::Binary);
        assert_eq!(config.num_iterations, 500);
        assert_eq!(config.learning_rate, 0.05);
        assert_eq!(config.num_leaves, 63);
        assert_eq!(config.device_type, DeviceType::CPU);
    }

    #[test]
    fn test_config_builder_validation() {
        let result = ConfigBuilder::new()
            .learning_rate(-0.1)
            .num_leaves(1)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_config_parameter_map() {
        let config = Config::default();
        let map = config.as_parameter_map();

        assert!(map.contains_key("objective"));
        assert!(map.contains_key("num_iterations"));
        assert!(map.contains_key("learning_rate"));
        assert_eq!(map.get("objective").unwrap(), "regression");
    }

    #[test]
    fn test_config_helper_methods() {
        let config = Config::default();

        assert_eq!(config.effective_num_threads(), num_cpus::get());
        assert!(!config.is_gpu_enabled());
        assert_eq!(config.num_trees_per_iteration(), 1);
        assert_eq!(config.total_num_trees(), config.num_iterations);
        assert!(config.is_early_stopping_enabled());
    }

    #[test]
    fn test_config_merge() {
        let mut config1 = Config::default();
        let mut config2 = Config::default();

        config2.learning_rate = 0.05;
        config2.num_leaves = 63;
        config2.objective = ObjectiveType::Binary;

        config1.merge(&config2).unwrap();

        assert_eq!(config1.learning_rate, 0.05);
        assert_eq!(config1.num_leaves, 63);
        assert_eq!(config1.objective, ObjectiveType::Binary);
    }

    #[test]
    fn test_multiclass_config() {
        let mut config = Config::default();
        config.objective = ObjectiveType::Multiclass;
        config.num_class = 5;

        assert!(config.validate().is_ok());
        assert_eq!(config.num_trees_per_iteration(), 5);
        assert_eq!(config.total_num_trees(), config.num_iterations * 5);
    }
}
