//! Core configuration structures and implementation for Pure Rust LightGBM.
//!
//! This module provides the main configuration structure and builder pattern
//! for setting up LightGBM training parameters, device settings, and other
//! configuration options.

use crate::core::constants::*;
use crate::core::error::{LightGBMError, Result};
use crate::core::types::*;
use crate::core::utils::common::Common;
use crate::core::utils::log::Log;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Default number of leaves constant (equivalent to kDefaultNumLeaves in C++)
const K_DEFAULT_NUM_LEAVES: usize = 31;

/// Types of tasks (equivalent to TaskType in C++)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    /// Training task
    Train,
    /// Prediction task
    Predict,
    /// Convert model task
    ConvertModel,
    /// Refit tree task
    RefitTree,
    /// Save binary task
    SaveBinary,
}

impl Default for TaskType {
    fn default() -> Self {
        TaskType::Train
    }
}

impl std::fmt::Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskType::Train => write!(f, "train"),
            TaskType::Predict => write!(f, "predict"),
            TaskType::ConvertModel => write!(f, "convert_model"),
            TaskType::RefitTree => write!(f, "refit"),
            TaskType::SaveBinary => write!(f, "save_binary"),
        }
    }
}

/// Main configuration structure for LightGBM training and prediction.
///
/// This structure contains all parameters needed to configure a LightGBM
/// model, including training parameters, regularization settings, device
/// configuration, and other options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    // Task configuration
    /// Task type
    pub task: TaskType,
    /// Path of config file
    pub config: String,
    // Core training parameters
    /// Objective function type (regression, classification, etc.)
    pub objective: String,
    /// Number of boosting iterations
    pub num_iterations: usize,
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Maximum number of leaves in one tree
    pub num_leaves: usize,
    /// Maximum depth of tree (-1 for unlimited)
    pub max_depth: i32,
    /// Boosting algorithm type
    pub boosting: String,
    /// Data sample strategy
    pub data_sample_strategy: String,
    /// Training data path
    pub data: String,
    /// Validation data paths
    pub valid: Vec<String>,
    /// Tree learner algorithm type
    pub tree_learner: String,

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
    /// Top rate for GOSS sampling (fraction of large gradient data to keep)
    pub top_rate: f64,
    /// Other rate for GOSS sampling (fraction of small gradient data to keep)
    pub other_rate: f64,

    // Feature parameters
    /// Random seed
    pub seed: u64,
    /// Force deterministic behavior
    pub deterministic: bool,

    // Learning Control Parameters
    /// Force column-wise histogram building
    pub force_col_wise: bool,
    /// Force row-wise histogram building
    pub force_row_wise: bool,
    /// Histogram pool size
    pub histogram_pool_size: f64,

    // Dataset Parameters
    /// Linear tree flag
    pub linear_tree: bool,
    /// Maximum number of bins for feature discretization
    pub max_bin: usize,
    /// Maximum bins by feature
    pub max_bin_by_feature: Vec<i32>,
    /// Minimum data in bin
    pub min_data_in_bin: usize,
    /// Bin construct sample count
    pub bin_construct_sample_cnt: usize,
    /// Data random seed
    pub data_random_seed: u64,
    /// Enable sparse optimization
    pub is_enable_sparse: bool,
    /// Enable bundle
    pub enable_bundle: bool,
    /// Use missing values
    pub use_missing: bool,
    /// Zero as missing
    pub zero_as_missing: bool,
    /// Feature pre-filter
    pub feature_pre_filter: bool,
    /// Pre-partition flag
    pub pre_partition: bool,
    /// Two round loading
    pub two_round: bool,
    /// Has header
    pub header: bool,
    /// Label column
    pub label_column: String,
    /// Weight column
    pub weight_column: String,
    /// Group column
    pub group_column: String,
    /// Ignore column
    pub ignore_column: String,
    /// Categorical feature
    pub categorical_feature: String,
    /// Forced bins filename
    pub forcedbins_filename: String,
    /// Save binary flag
    pub save_binary: bool,
    /// Precise float parser
    pub precise_float_parser: bool,
    /// Parser config file
    pub parser_config_file: String,
    /// Minimum data per categorical group
    pub min_data_per_group: usize,
    /// Smoothing parameter for categorical features
    pub cat_smooth: f64,
    /// L2 regularization for categorical features
    pub cat_l2: f64,
    /// Maximum categories to use one-hot encoding
    pub max_cat_to_onehot: usize,
    /// Maximum categorical threshold
    pub max_cat_threshold: usize,
    /// Top-k categories to consider
    pub top_k: usize,

    // Device configuration
    /// Device type (CPU, GPU, etc.)
    pub device_type: String,
    /// Number of threads for parallel processing
    pub num_threads: usize,
    /// GPU platform ID
    pub gpu_platform_id: i32,
    /// GPU device ID
    pub gpu_device_id: i32,
    /// Use double precision on GPU
    pub gpu_use_dp: bool,
    /// Number of GPUs
    pub num_gpu: usize,

    // Multiclass parameters
    /// Number of classes for multiclass classification
    pub num_class: usize,
    /// Whether dataset is unbalanced
    pub is_unbalance: bool,
    /// Weight for positive class in binary classification
    pub scale_pos_weight: f64,

    // Early stopping
    /// Early stopping rounds
    pub early_stopping_round: usize,
    /// Early stopping minimum delta
    pub early_stopping_min_delta: f64,
    /// Use only first metric for early stopping
    pub first_metric_only: bool,

    // Validation and metrics
    /// Metrics to evaluate during training
    pub metric: Vec<String>,
    /// Frequency of metric evaluation
    pub metric_freq: usize,
    /// Whether to compute training metrics
    pub is_provide_training_metric: bool,

    // Output control
    /// Verbosity level for logging
    pub verbosity: i32,
    /// Output model file path
    pub output_model: String,
    /// Input model file path for continuing training
    pub input_model: String,
    /// Saved feature importance type
    pub saved_feature_importance_type: usize,
    /// Snapshot frequency
    pub snapshot_freq: i32,

    // Gradient quantization
    /// Use quantized gradients
    pub use_quantized_grad: bool,
    /// Number of gradient quantization bins
    pub num_grad_quant_bins: usize,
    /// Quantized training renew leaf
    pub quant_train_renew_leaf: bool,
    /// Stochastic rounding
    pub stochastic_rounding: bool,

    // Advanced parameters
    /// Monotone constraints for features (-1, 0, 1 for decreasing, none, increasing)
    pub monotone_constraints: Vec<i8>,
    /// Method for monotone constraints enforcement
    pub monotone_constraints_method: String,
    /// Penalty for violating monotone constraints
    pub monotone_penalty: f64,
    /// Feature contributions
    pub feature_contri: Vec<f64>,
    /// Forced splits filename
    pub forcedsplits_filename: String,
    /// Decay rate for refitting leaf values
    pub refit_decay_rate: f64,
    /// Cost-effective gradient boosting tradeoff
    pub cegb_tradeoff: f64,
    /// CEGB penalty split
    pub cegb_penalty_split: f64,
    /// CEGB penalty feature lazy
    pub cegb_penalty_feature_lazy: Vec<f64>,
    /// CEGB penalty feature coupled
    pub cegb_penalty_feature_coupled: Vec<f64>,
    /// Path smoothing parameter
    pub path_smooth: f64,
    /// Interaction constraints
    pub interaction_constraints: String,

    // Network parameters (for distributed training)
    /// Number of machines for distributed training
    pub num_machines: usize,
    /// Local port for listening in distributed training
    pub local_listen_port: u16,
    /// Timeout for network operations in seconds
    pub time_out: u64,
    /// Filename containing machine list for distributed training
    pub machine_list_filename: String,
    /// Machines list
    pub machines: String,

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
    pub max_drop: i32,
    /// Use uniform dropout in DART
    pub uniform_drop: bool,
    /// Enable XGBoost-style DART mode
    pub xgboost_dart_mode: bool,
    /// Random seed for dropout
    pub drop_seed: u64,

    // Prediction parameters
    /// Start iteration for prediction
    pub start_iteration_predict: usize,
    /// Number of iterations for prediction
    pub num_iteration_predict: i32,
    /// Predict raw score
    pub predict_raw_score: bool,
    /// Predict leaf index
    pub predict_leaf_index: bool,
    /// Predict contributions
    pub predict_contrib: bool,
    /// Predict disable shape check
    pub predict_disable_shape_check: bool,
    /// Prediction early stop
    pub pred_early_stop: bool,
    /// Prediction early stop frequency
    pub pred_early_stop_freq: usize,
    /// Prediction early stop margin
    pub pred_early_stop_margin: f64,
    /// Output result filename
    pub output_result: String,

    // Convert parameters
    /// Convert model language
    pub convert_model_language: String,
    /// Convert model filename
    pub convert_model: String,

    // Objective parameters
    /// Objective seed
    pub objective_seed: u64,
    /// Sigmoid parameter
    pub sigmoid: f64,
    /// Boost from average
    pub boost_from_average: bool,
    /// Regression sqrt
    pub reg_sqrt: bool,
    /// Alpha parameter
    pub alpha: f64,
    /// Fair C parameter
    pub fair_c: f64,
    /// Poisson max delta step
    pub poisson_max_delta_step: f64,
    /// Tweedie variance power
    pub tweedie_variance_power: f64,
    /// Lambda rank truncation level
    pub lambdarank_truncation_level: usize,
    /// Lambda rank normalization
    pub lambdarank_norm: bool,
    /// Label gain
    pub label_gain: Vec<f64>,
    /// Lambda rank position bias regularization
    pub lambdarank_position_bias_regularization: f64,

    // Metric parameters
    /// Evaluation positions
    pub eval_at: Vec<usize>,
    /// Multi-error top k
    pub multi_error_top_k: usize,
    /// AUC mu weights
    pub auc_mu_weights: Vec<f64>,

    // Internal state
    /// File load progress interval bytes
    pub file_load_progress_interval_bytes: usize,
    /// Is parallel
    pub is_parallel: bool,
    /// Is data based parallel
    pub is_data_based_parallel: bool,
    /// AUC mu weights matrix
    pub auc_mu_weights_matrix: Vec<Vec<f64>>,
    /// Interaction constraints vector
    pub interaction_constraints_vector: Vec<Vec<usize>>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            // Task configuration
            task: TaskType::Train,
            config: String::new(),

            // Core training parameters
            objective: "regression".to_string(),
            num_iterations: DEFAULT_NUM_ITERATIONS,
            learning_rate: DEFAULT_LEARNING_RATE,
            num_leaves: K_DEFAULT_NUM_LEAVES,
            max_depth: DEFAULT_MAX_DEPTH,
            boosting: "gbdt".to_string(),
            data_sample_strategy: "bagging".to_string(),
            data: String::new(),
            valid: Vec::new(),
            tree_learner: "serial".to_string(),
            seed: 0,
            deterministic: false,

            // Learning Control Parameters
            force_col_wise: false,
            force_row_wise: false,
            histogram_pool_size: -1.0,

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
            feature_fraction_seed: 2,
            bagging_fraction: DEFAULT_BAGGING_FRACTION,
            bagging_freq: DEFAULT_BAGGING_FREQ,
            bagging_seed: 3,
            pos_bagging_fraction: 1.0,
            neg_bagging_fraction: 1.0,
            bagging_by_query: false,
            top_rate: 0.2,
            other_rate: 0.1,

            // Feature parameters
            max_bin: DEFAULT_MAX_BIN,
            max_bin_by_feature: Vec::new(),
            min_data_in_bin: 3,
            bin_construct_sample_cnt: 200000,
            data_random_seed: 1,
            is_enable_sparse: true,
            enable_bundle: true,
            use_missing: true,
            zero_as_missing: false,
            feature_pre_filter: true,
            pre_partition: false,
            two_round: false,
            header: false,
            label_column: String::new(),
            weight_column: String::new(),
            group_column: String::new(),
            ignore_column: String::new(),
            categorical_feature: String::new(),
            forcedbins_filename: String::new(),
            save_binary: false,
            precise_float_parser: false,
            parser_config_file: String::new(),
            linear_tree: false,
            min_data_per_group: 100,
            cat_smooth: 10.0,
            cat_l2: 10.0,
            max_cat_to_onehot: 4,
            max_cat_threshold: 32,
            top_k: 20,

            // Device configuration
            device_type: "cpu".to_string(),
            num_threads: DEFAULT_NUM_THREADS,
            gpu_platform_id: -1,
            gpu_device_id: -1,
            gpu_use_dp: false,
            num_gpu: 1,

            // Multiclass parameters
            num_class: DEFAULT_NUM_CLASS,
            is_unbalance: false,
            scale_pos_weight: 1.0,

            // Early stopping
            early_stopping_round: 0,
            early_stopping_min_delta: 0.0,
            first_metric_only: false,

            // Validation and metrics
            metric: Vec::new(),
            metric_freq: 1,
            is_provide_training_metric: false,

            // Output control
            verbosity: 1,
            output_model: "LightGBM_model.txt".to_string(),
            input_model: String::new(),
            saved_feature_importance_type: 0,
            snapshot_freq: -1,

            // Gradient quantization
            use_quantized_grad: false,
            num_grad_quant_bins: 4,
            quant_train_renew_leaf: false,
            stochastic_rounding: true,

            // Advanced parameters
            monotone_constraints: Vec::new(),
            monotone_constraints_method: "basic".to_string(),
            monotone_penalty: 0.0,
            feature_contri: Vec::new(),
            forcedsplits_filename: String::new(),
            refit_decay_rate: 0.9,
            cegb_tradeoff: 1.0,
            cegb_penalty_split: 0.0,
            cegb_penalty_feature_lazy: Vec::new(),
            cegb_penalty_feature_coupled: Vec::new(),
            path_smooth: 0.0,
            interaction_constraints: String::new(),

            // Network parameters
            num_machines: 1,
            local_listen_port: 12400,
            time_out: 120,
            machine_list_filename: String::new(),
            machines: String::new(),

            // Additional parameters
            extra_seed: 6,
            extra_trees: false,
            skip_drop: 0.5,
            drop_rate: 0.1,
            max_drop: 50,
            uniform_drop: false,
            xgboost_dart_mode: false,
            drop_seed: 4,

            // Prediction parameters
            start_iteration_predict: 0,
            num_iteration_predict: -1,
            predict_raw_score: false,
            predict_leaf_index: false,
            predict_contrib: false,
            predict_disable_shape_check: false,
            pred_early_stop: false,
            pred_early_stop_freq: 10,
            pred_early_stop_margin: 10.0,
            output_result: "LightGBM_predict_result.txt".to_string(),

            // Convert parameters
            convert_model_language: String::new(),
            convert_model: "gbdt_prediction.cpp".to_string(),

            // Objective parameters
            objective_seed: 5,
            sigmoid: 1.0,
            boost_from_average: true,
            reg_sqrt: false,
            alpha: 0.9,
            fair_c: 1.0,
            poisson_max_delta_step: 0.7,
            tweedie_variance_power: 1.5,
            lambdarank_truncation_level: 30,
            lambdarank_norm: true,
            label_gain: Vec::new(),
            lambdarank_position_bias_regularization: 0.0,

            // Metric parameters
            eval_at: vec![1, 2, 3, 4, 5],
            multi_error_top_k: 1,
            auc_mu_weights: Vec::new(),

            // Internal state
            file_load_progress_interval_bytes: 10 * 1024 * 1024 * 1024,
            is_parallel: false,
            is_data_based_parallel: false,
            auc_mu_weights_matrix: Vec::new(),
            interaction_constraints_vector: Vec::new(),
        }
    }
}

impl Config {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Constructor from parameter map (equivalent to Config constructor in C++)
    pub fn from_params(params: HashMap<String, String>) -> Self {
        let mut config = Self::default();
        config.set(params);
        config
    }

    /// Get string value by specific name of key (equivalent to GetString in C++)
    pub fn get_string(params: &HashMap<String, String>, name: &str) -> Option<String> {
        params.get(name).filter(|s| !s.is_empty()).cloned()
    }

    /// Get int value by specific name of key (equivalent to GetInt in C++)
    pub fn get_int(params: &HashMap<String, String>, name: &str) -> Result<Option<i32>> {
        if let Some(value) = params.get(name).filter(|s| !s.is_empty()) {
            match Common::atoi_and_check(value) {
                Some(parsed) => Ok(Some(parsed)),
                None => {
                    return Err(LightGBMError::config(format!(
                        "Parameter {} should be of type int, got \"{}\"",
                        name, value
                    )));
                }
            }
        } else {
            Ok(None)
        }
    }

    /// Get double value by specific name of key (equivalent to GetDouble in C++)
    pub fn get_double(params: &HashMap<String, String>, name: &str) -> Result<Option<f64>> {
        if let Some(value) = params.get(name).filter(|s| !s.is_empty()) {
            match Common::atof_and_check(value) {
                Some(parsed) => Ok(Some(parsed)),
                None => {
                    return Err(LightGBMError::config(format!(
                        "Parameter {} should be of type double, got \"{}\"",
                        name, value
                    )));
                }
            }
        } else {
            Ok(None)
        }
    }

    /// Get bool value by specific name of key (equivalent to GetBool in C++)
    pub fn get_bool(params: &HashMap<String, String>, name: &str) -> Result<Option<bool>> {
        if let Some(value) = params.get(name).filter(|s| !s.is_empty()) {
            let lower_value = value.to_lowercase();
            if lower_value == "false" || lower_value == "-" {
                Ok(Some(false))
            } else if lower_value == "true" || lower_value == "+" {
                Ok(Some(true))
            } else {
                return Err(LightGBMError::config(format!(
                    "Parameter {} should be \"true\"/\"+\" or \"false\"/\"-\", got \"{}\"",
                    name, value
                )));
            }
        } else {
            Ok(None)
        }
    }

    /// Sort aliases by length and then alphabetically (equivalent to SortAlias in C++)
    pub fn sort_alias(x: &str, y: &str) -> bool {
        x.len() < y.len() || (x.len() == y.len() && x < y)
    }

    /// Keep first values from parameter map (equivalent to KeepFirstValues in C++)
    pub fn keep_first_values(
        params: &HashMap<String, Vec<String>>,
        out: &mut HashMap<String, String>,
    ) {
        for (key, values) in params {
            if !values.is_empty() {
                out.insert(key.clone(), values[0].clone());
            }
        }
    }

    /// Parse key-value pair (equivalent to KV2Map in C++)
    pub fn kv_to_map(params: &mut HashMap<String, Vec<String>>, kv: &str) {
        if let Some(equal_pos) = kv.find('=') {
            let key = kv[..equal_pos].trim().to_string();
            let value = kv[equal_pos + 1..].trim().to_string();
            params.entry(key).or_insert_with(Vec::new).push(value);
        }
    }

    /// Set verbosity from parameters (equivalent to SetVerbosity in C++)
    pub fn set_verbosity(params: &HashMap<String, Vec<String>>) {
        if let Some(verbosity_values) = params.get("verbosity") {
            if let Some(verbosity_str) = verbosity_values.first() {
                if let Ok(verbosity) = verbosity_str.parse::<i32>() {
                    // Set the global verbosity level
                    Log::set_verbosity(verbosity);
                }
            }
        }
    }

    /// Parse string to parameter map (equivalent to Str2Map in C++)
    pub fn str_to_map(parameters: &str) -> HashMap<String, String> {
        let mut params_multi: HashMap<String, Vec<String>> = HashMap::new();

        // Split by whitespace and commas, parse each key-value pair
        for part in parameters.split_whitespace() {
            for kv in part.split(',') {
                let kv = kv.trim();
                if !kv.is_empty() {
                    Self::kv_to_map(&mut params_multi, kv);
                }
            }
        }

        // Set verbosity early
        Self::set_verbosity(&params_multi);

        // Convert to single-value map
        let mut result = HashMap::new();
        Self::keep_first_values(&params_multi, &mut result);

        result
    }

    /// Set configuration from parameter map (equivalent to Set in C++)
    pub fn set(&mut self, params: HashMap<String, String>) {
        // Process parameter aliases
        let processed_params = Self::process_aliases(params);

        // Check for parameter conflicts
        self.check_param_conflict(&processed_params);

        // Extract members from the processed parameters
        self.get_members_from_string(&processed_params);

        // Additional processing
        self.get_auc_mu_weights();
        self.get_interaction_constraints();
    }

    /// Convert configuration to string representation (equivalent to ToString in C++)
    pub fn to_string(&self) -> String {
        self.save_members_to_string()
    }

    /// Process parameter aliases (equivalent to ParameterAlias::KeyAliasTransform)
    fn process_aliases(params: HashMap<String, String>) -> HashMap<String, String> {
        // This is a simplified version - in full implementation, you'd have a complete alias table
        let mut alias_map = HashMap::new();

        // Some common aliases (simplified)
        alias_map.insert("num_iteration".to_string(), "num_iterations".to_string());
        alias_map.insert("n_iter".to_string(), "num_iterations".to_string());
        alias_map.insert("num_tree".to_string(), "num_iterations".to_string());
        alias_map.insert("num_trees".to_string(), "num_iterations".to_string());
        alias_map.insert("num_round".to_string(), "num_iterations".to_string());
        alias_map.insert("num_rounds".to_string(), "num_iterations".to_string());
        alias_map.insert("nrounds".to_string(), "num_iterations".to_string());
        alias_map.insert("num_boost_round".to_string(), "num_iterations".to_string());
        alias_map.insert("n_estimators".to_string(), "num_iterations".to_string());
        alias_map.insert("max_iter".to_string(), "num_iterations".to_string());

        alias_map.insert("shrinkage_rate".to_string(), "learning_rate".to_string());
        alias_map.insert("eta".to_string(), "learning_rate".to_string());

        alias_map.insert("num_leaf".to_string(), "num_leaves".to_string());
        alias_map.insert("max_leaves".to_string(), "num_leaves".to_string());
        alias_map.insert("max_leaf".to_string(), "num_leaves".to_string());
        alias_map.insert("max_leaf_nodes".to_string(), "num_leaves".to_string());

        // Process aliases
        let mut result = HashMap::new();
        for (key, value) in params {
            let actual_key = alias_map.get(&key).unwrap_or(&key);
            result.insert(actual_key.clone(), value);
        }

        result
    }

    /// Get alias table (equivalent to alias_table in C++)
    pub fn alias_table() -> &'static HashMap<String, String> {
        // This would be a static HashMap in a real implementation
        // For now, we'll use a simplified approach
        static ALIAS_TABLE: std::sync::OnceLock<HashMap<String, String>> =
            std::sync::OnceLock::new();
        ALIAS_TABLE.get_or_init(|| {
            let mut map = HashMap::new();
            // Add all the aliases here...
            map
        })
    }

    /// Get parameter to aliases mapping (equivalent to parameter2aliases in C++)
    pub fn parameter_to_aliases() -> &'static HashMap<String, Vec<String>> {
        static PARAM_ALIASES: std::sync::OnceLock<HashMap<String, Vec<String>>> =
            std::sync::OnceLock::new();
        PARAM_ALIASES.get_or_init(|| {
            let mut map = HashMap::new();
            // Add parameter to aliases mappings here...
            map
        })
    }

    /// Get parameter set (equivalent to parameter_set in C++)
    pub fn parameter_set() -> &'static HashSet<String> {
        static PARAM_SET: std::sync::OnceLock<HashSet<String>> = std::sync::OnceLock::new();
        PARAM_SET.get_or_init(|| {
            let mut set = HashSet::new();
            // Add all valid parameter names here...
            set
        })
    }

    /// Get parameter types (equivalent to ParameterTypes in C++)
    pub fn parameter_types() -> &'static HashMap<String, String> {
        static PARAM_TYPES: std::sync::OnceLock<HashMap<String, String>> =
            std::sync::OnceLock::new();
        PARAM_TYPES.get_or_init(|| {
            let mut map = HashMap::new();
            // Add parameter type mappings here...
            map
        })
    }

    /// Dump aliases (equivalent to DumpAliases in C++)
    pub fn dump_aliases() -> String {
        // Implementation to dump all aliases
        String::new()
    }

    /// Check for parameter conflicts (equivalent to CheckParamConflict in C++)
    fn check_param_conflict(&self, _params: &HashMap<String, String>) {
        // Implementation for parameter conflict checking
        // This would check for conflicting parameter combinations
    }

    /// Extract members from string parameters (equivalent to GetMembersFromString in C++)
    fn get_members_from_string(&mut self, params: &HashMap<String, String>) {
        // Extract all parameters from the map and set them on the config
        for (key, value) in params {
            match key.as_str() {
                "objective" => self.objective = value.clone(),
                "num_iterations" => {
                    if let Ok(Some(val)) = Self::get_int(params, "num_iterations") {
                        self.num_iterations = val as usize;
                    }
                }
                "learning_rate" => {
                    if let Ok(Some(val)) = Self::get_double(params, "learning_rate") {
                        self.learning_rate = val;
                    }
                }
                "num_leaves" => {
                    if let Ok(Some(val)) = Self::get_int(params, "num_leaves") {
                        self.num_leaves = val as usize;
                    }
                }
                "max_depth" => {
                    if let Ok(Some(val)) = Self::get_int(params, "max_depth") {
                        self.max_depth = val;
                    }
                }
                "boosting" => self.boosting = value.clone(),
                "device_type" => self.device_type = value.clone(),
                "num_threads" => {
                    if let Ok(Some(val)) = Self::get_int(params, "num_threads") {
                        self.num_threads = val as usize;
                    }
                }
                "seed" => {
                    if let Ok(Some(val)) = Self::get_int(params, "seed") {
                        self.seed = val as u64;
                    }
                }
                "deterministic" => {
                    if let Ok(Some(val)) = Self::get_bool(params, "deterministic") {
                        self.deterministic = val;
                    }
                }
                // Add more parameter extractions here...
                _ => {} // Ignore unknown parameters with warning
            }
        }
    }

    /// Save members to string (equivalent to SaveMembersToString in C++)
    fn save_members_to_string(&self) -> String {
        let mut result = String::new();

        result.push_str(&format!("objective = {}\n", self.objective));
        result.push_str(&format!("num_iterations = {}\n", self.num_iterations));
        result.push_str(&format!("learning_rate = {}\n", self.learning_rate));
        result.push_str(&format!("num_leaves = {}\n", self.num_leaves));
        result.push_str(&format!("max_depth = {}\n", self.max_depth));
        result.push_str(&format!("boosting = {}\n", self.boosting));
        result.push_str(&format!("device_type = {}\n", self.device_type));
        result.push_str(&format!("num_threads = {}\n", self.num_threads));
        result.push_str(&format!("seed = {}\n", self.seed));
        result.push_str(&format!("deterministic = {}\n", self.deterministic));

        // Add more parameters as needed...

        result
    }

    /// Get AUC mu weights (equivalent to GetAucMuWeights in C++)
    fn get_auc_mu_weights(&mut self) {
        // Process AUC mu weights into matrix form
        // This would convert the linear auc_mu_weights vector into a matrix
    }

    /// Get interaction constraints (equivalent to GetInteractionConstraints in C++)
    fn get_interaction_constraints(&mut self) {
        // Parse interaction constraints string into vector of vectors
        // This would parse a string like "[0,1,2],[2,3]" into Vec<Vec<usize>>
    }

    /// Validate the configuration parameters
    pub fn validate(&self) -> Result<()> {
        // Validate learning rate
        if self.learning_rate <= 0.0 {
            return Err(LightGBMError::invalid_parameter(
                "learning_rate",
                self.learning_rate.to_string(),
                "must be positive",
            ));
        }

        // Validate num_leaves
        if self.num_leaves <= 1 || self.num_leaves > 131072 {
            return Err(LightGBMError::invalid_parameter(
                "num_leaves",
                self.num_leaves.to_string(),
                "must be in range (1, 131072]",
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
        if self.min_data_in_leaf < 0 {
            return Err(LightGBMError::invalid_parameter(
                "min_data_in_leaf",
                self.min_data_in_leaf.to_string(),
                "must be non-negative",
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
        if self.max_bin <= 1 {
            return Err(LightGBMError::invalid_parameter(
                "max_bin",
                self.max_bin.to_string(),
                "must be greater than 1",
            ));
        }

        // Validate num_class for multiclass
        if self.objective == "multiclass" && self.num_class <= 0 {
            return Err(LightGBMError::invalid_parameter(
                "num_class",
                self.num_class.to_string(),
                "must be positive for multiclass objective",
            ));
        }

        // Validate GPU parameters
        if self.device_type == "gpu" {
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
        for (i, &constraint) in self.monotone_constraints.iter().enumerate() {
            if constraint < -1 || constraint > 1 {
                return Err(LightGBMError::invalid_parameter(
                    &format!("monotone_constraints[{}]", i),
                    constraint.to_string(),
                    "must be in range [-1, 1]",
                ));
            }
        }

        // Validate early stopping
        if self.early_stopping_round < 0 {
            return Err(LightGBMError::invalid_parameter(
                "early_stopping_round",
                self.early_stopping_round.to_string(),
                "must be non-negative",
            ));
        }

        // Validate DART parameters
        if self.boosting == "dart" {
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
            config.objective = val;
        }

        if let Ok(val) = std::env::var("LIGHTGBM_DEVICE_TYPE") {
            config.device_type = val;
        }

        if let Ok(val) = std::env::var("LIGHTGBM_NUM_THREADS") {
            config.num_threads = val
                .parse()
                .map_err(|_| LightGBMError::config("Invalid LIGHTGBM_NUM_THREADS"))?;
        }

        if let Ok(val) = std::env::var("LIGHTGBM_RANDOM_SEED") {
            config.seed = val
                .parse()
                .map_err(|_| LightGBMError::config("Invalid LIGHTGBM_RANDOM_SEED"))?;
        }

        config.validate()?;
        Ok(config)
    }

    // TODO: Implement apply_environment_overrides - temporarily commented out due to compilation issues
    // /// Apply environment variable overrides to existing configuration
    // pub fn apply_environment_overrides(&mut self) -> Result<()> {
    //     // Implementation would apply environment overrides
    //     self.validate()
    // }

    // TODO: Implement merge method - temporarily commented out due to compilation issues
    // /// Merge this configuration with another configuration
    // pub fn merge(&mut self, other: &Config) -> Result<()> {
    //     // Implementation would merge all parameters (other takes precedence)
    //     self.validate()
    // }

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
        self.device_type == "gpu"
    }

    /// Get the number of trees per iteration (depends on objective)
    pub fn num_trees_per_iteration(&self) -> usize {
        match self.objective.as_str() {
            "regression" | "binary" | "lambdarank" | "rank_xendcg" => 1,
            "multiclass" | "multiclassova" => self.num_class,
            _ => 1,
        }
    }

    /// Get total number of trees that will be trained
    pub fn total_num_trees(&self) -> usize {
        self.num_iterations * self.num_trees_per_iteration()
    }

    /// Check if early stopping is enabled
    pub fn is_early_stopping_enabled(&self) -> bool {
        self.early_stopping_round > 0
    }

    /// Get configuration as a parameter map (for debugging/serialization)
    pub fn as_parameter_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();

        map.insert("objective".to_string(), self.objective.clone());
        map.insert(
            "num_iterations".to_string(),
            self.num_iterations.to_string(),
        );
        map.insert("learning_rate".to_string(), self.learning_rate.to_string());
        map.insert("num_leaves".to_string(), self.num_leaves.to_string());
        map.insert("max_depth".to_string(), self.max_depth.to_string());
        map.insert("boosting".to_string(), self.boosting.clone());
        map.insert("tree_learner".to_string(), self.tree_learner.clone());

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
        map.insert("device_type".to_string(), self.device_type.clone());
        map.insert("num_threads".to_string(), self.num_threads.to_string());
        map.insert("num_class".to_string(), self.num_class.to_string());
        map.insert("seed".to_string(), self.seed.to_string());
        map.insert("deterministic".to_string(), self.deterministic.to_string());

        if self.early_stopping_round > 0 {
            map.insert(
                "early_stopping_round".to_string(),
                self.early_stopping_round.to_string(),
            );
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

/// Parse objective alias to canonical form (equivalent to ParseObjectiveAlias in C++)
pub fn parse_objective_alias(objective_type: &str) -> String {
    match objective_type {
        "regression"
        | "regression_l2"
        | "mean_squared_error"
        | "mse"
        | "l2"
        | "l2_root"
        | "root_mean_squared_error"
        | "rmse" => "regression".to_string(),
        "regression_l1" | "mean_absolute_error" | "l1" | "mae" => "regression_l1".to_string(),
        "multiclass" | "softmax" => "multiclass".to_string(),
        "multiclassova" | "multiclass_ova" | "ova" | "ovr" => "multiclassova".to_string(),
        "xentropy" | "cross_entropy" => "cross_entropy".to_string(),
        "xentlambda" | "cross_entropy_lambda" => "cross_entropy_lambda".to_string(),
        "mean_absolute_percentage_error" | "mape" => "mape".to_string(),
        "rank_xendcg" | "xendcg" | "xe_ndcg" | "xe_ndcg_mart" | "xendcg_mart" => {
            "rank_xendcg".to_string()
        }
        "none" | "null" | "custom" | "na" => "custom".to_string(),
        _ => objective_type.to_string(),
    }
}

/// Parse metric alias to canonical form (equivalent to ParseMetricAlias in C++)
pub fn parse_metric_alias(metric_type: &str) -> String {
    match metric_type {
        "regression" | "regression_l2" | "l2" | "mean_squared_error" | "mse" => "l2".to_string(),
        "l2_root" | "root_mean_squared_error" | "rmse" => "rmse".to_string(),
        "regression_l1" | "l1" | "mean_absolute_error" | "mae" => "l1".to_string(),
        "binary_logloss" | "binary" => "binary_logloss".to_string(),
        "ndcg" | "lambdarank" | "rank_xendcg" | "xendcg" | "xe_ndcg" | "xe_ndcg_mart"
        | "xendcg_mart" => "ndcg".to_string(),
        "map" | "mean_average_precision" => "map".to_string(),
        "multi_logloss" | "multiclass" | "softmax" | "multiclassova" | "multiclass_ova" | "ova"
        | "ovr" => "multi_logloss".to_string(),
        "xentropy" | "cross_entropy" => "cross_entropy".to_string(),
        "xentlambda" | "cross_entropy_lambda" => "cross_entropy_lambda".to_string(),
        "kldiv" | "kullback_leibler" => "kullback_leibler".to_string(),
        "mean_absolute_percentage_error" | "mape" => "mape".to_string(),
        "none" | "null" | "custom" | "na" => "custom".to_string(),
        _ => metric_type.to_string(),
    }
}

/// Parameter alias structure for handling parameter transformations
#[derive(Debug)]
pub struct ParameterAlias;

impl ParameterAlias {
    // TODO: Implement key_alias_transform - temporarily commented out due to compilation issues
    // /// Transform parameter keys using alias table (equivalent to KeyAliasTransform in C++)
    // pub fn key_alias_transform(params: &mut HashMap<String, String>) {
    //     // Implementation would transform parameter keys using alias table
    // }
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
    pub fn objective(mut self, objective: &str) -> Self {
        self.config.objective = objective.to_string();
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
    pub fn device_type(mut self, device: &str) -> Self {
        self.config.device_type = device.to_string();
        self
    }

    /// Set number of threads
    pub fn num_threads(mut self, threads: usize) -> Self {
        self.config.num_threads = threads;
        self
    }

    /// Set random seed
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
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
    pub fn early_stopping_rounds(mut self, rounds: usize) -> Self {
        self.config.early_stopping_round = rounds;
        self
    }

    /// Set early stopping tolerance
    pub fn early_stopping_tolerance(mut self, tolerance: f64) -> Self {
        if tolerance < 0.0 {
            self.validation_errors
                .push("early_stopping_tolerance must be non-negative".to_string());
        }
        self.config.early_stopping_min_delta = tolerance;
        self
    }

    /// Set tree learner type
    pub fn tree_learner(mut self, learner: &str) -> Self {
        self.config.tree_learner = learner.to_string();
        self
    }

    /// Set verbosity level
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbosity = if verbose { 1 } else { -1 };
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
        assert_eq!(config.objective, "regression");
        assert_eq!(config.num_iterations, DEFAULT_NUM_ITERATIONS);
        assert_eq!(config.learning_rate, DEFAULT_LEARNING_RATE);
        assert_eq!(config.num_leaves, K_DEFAULT_NUM_LEAVES);
        assert_eq!(config.device_type, "cpu");
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
            .objective("binary")
            .num_iterations(500)
            .learning_rate(0.05)
            .num_leaves(63)
            .device_type("cpu")
            .build()
            .unwrap();

        assert_eq!(config.objective, "binary");
        assert_eq!(config.num_iterations, 500);
        assert_eq!(config.learning_rate, 0.05);
        assert_eq!(config.num_leaves, 63);
        assert_eq!(config.device_type, "cpu");
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
        assert!(!config.is_early_stopping_enabled()); // Default early_stopping_round is 0
    }

    // TODO: Re-implement test_config_merge once merge method is fixed
    // #[test]
    // fn test_config_merge() {
    //     // Test implementation pending merge method fix
    // }

    #[test]
    fn test_multiclass_config() {
        let mut config = Config::default();
        config.objective = "multiclass".to_string();
        config.num_class = 5;

        assert!(config.validate().is_ok());
        assert_eq!(config.num_trees_per_iteration(), 5);
        assert_eq!(config.total_num_trees(), config.num_iterations * 5);
    }
}
