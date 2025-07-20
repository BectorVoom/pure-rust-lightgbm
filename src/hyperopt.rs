//! Hyperparameter optimization module for Pure Rust LightGBM.
//!
//! This module provides placeholder implementations for hyperparameter optimization
//! functionality including parameter space definition, optimization configuration,
//! and optimization algorithms.

use crate::config::Config;
use crate::core::error::{LightGBMError, Result};
use crate::dataset::Dataset;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Hyperparameter space definition for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterSpace {
    /// Float parameters with their ranges
    pub float_params: HashMap<String, (f64, f64)>,
    /// Integer parameters with their ranges
    pub int_params: HashMap<String, (i32, i32)>,
    /// Categorical parameters with their options
    pub categorical_params: HashMap<String, Vec<String>>,
}

impl HyperparameterSpace {
    /// Create a new empty hyperparameter space
    pub fn new() -> Self {
        Self {
            float_params: HashMap::new(),
            int_params: HashMap::new(),
            categorical_params: HashMap::new(),
        }
    }

    /// Add a float parameter with range
    pub fn add_float(mut self, name: &str, min: f64, max: f64) -> Self {
        self.float_params.insert(name.to_string(), (min, max));
        self
    }

    /// Add an integer parameter with range
    pub fn add_int(mut self, name: &str, min: i32, max: i32) -> Self {
        self.int_params.insert(name.to_string(), (min, max));
        self
    }

    /// Add a categorical parameter with options
    pub fn add_categorical(mut self, name: &str, options: Vec<String>) -> Self {
        self.categorical_params.insert(name.to_string(), options);
        self
    }
}

impl Default for HyperparameterSpace {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization direction for hyperparameter search
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationDirection {
    /// Minimize the objective
    Minimize,
    /// Maximize the objective
    Maximize,
}

impl Default for OptimizationDirection {
    fn default() -> Self {
        OptimizationDirection::Minimize
    }
}

/// Configuration for hyperparameter optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Number of trials to run
    pub num_trials: usize,
    /// Number of cross-validation folds
    pub cv_folds: usize,
    /// Metric to optimize
    pub metric: String,
    /// Optimization direction
    pub direction: OptimizationDirection,
    /// Timeout in seconds
    pub timeout_seconds: Option<u64>,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Early stopping patience (stop if no improvement for N trials)
    pub early_stopping_patience: Option<usize>,
    /// Minimum improvement threshold for early stopping
    pub early_stopping_min_delta: Option<f64>,
    /// Progress reporting interval (report every N trials)
    pub progress_interval: usize,
}

impl OptimizationConfig {
    /// Create a new optimization configuration
    pub fn new() -> Self {
        Self {
            num_trials: 100,
            cv_folds: 5,
            metric: "rmse".to_string(),
            direction: OptimizationDirection::Minimize,
            timeout_seconds: None,
            random_seed: None,
            early_stopping_patience: None,
            early_stopping_min_delta: None,
            progress_interval: 10,
        }
    }

    /// Set the number of trials
    pub fn with_num_trials(mut self, num_trials: usize) -> Self {
        self.num_trials = num_trials;
        self
    }

    /// Set the number of CV folds
    pub fn with_cv_folds(mut self, cv_folds: usize) -> Self {
        self.cv_folds = cv_folds;
        self
    }

    /// Set the metric to optimize
    pub fn with_metric(mut self, metric: &str) -> Self {
        self.metric = metric.to_string();
        self
    }

    /// Set the optimization direction
    pub fn with_direction(mut self, direction: OptimizationDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Set the timeout in seconds
    pub fn with_timeout_seconds(mut self, timeout: u64) -> Self {
        self.timeout_seconds = Some(timeout);
        self
    }

    /// Set early stopping patience
    pub fn with_early_stopping_patience(mut self, patience: usize) -> Self {
        self.early_stopping_patience = Some(patience);
        self
    }

    /// Set early stopping minimum delta
    pub fn with_early_stopping_min_delta(mut self, min_delta: f64) -> Self {
        self.early_stopping_min_delta = Some(min_delta);
        self
    }

    /// Set progress reporting interval
    pub fn with_progress_interval(mut self, interval: usize) -> Self {
        self.progress_interval = interval;
        self
    }

    /// Set random seed
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Results from hyperparameter optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Number of trials completed
    pub num_trials: usize,
    /// Best objective value found
    pub best_score: f64,
    /// Best hyperparameters found
    pub best_params: HashMap<String, f64>,
    /// All trial results
    pub trials: Vec<TrialResult>,
}

/// Result from a single optimization trial
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    /// Trial number
    pub trial_id: usize,
    /// Parameters used in this trial
    pub params: HashMap<String, f64>,
    /// Objective value achieved
    pub score: f64,
    /// Cross-validation scores
    pub cv_scores: Vec<f64>,
}

/// Cross-validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationConfig {
    /// Number of folds
    pub num_folds: usize,
    /// Whether to use stratified sampling
    pub stratified: bool,
    /// Whether to shuffle data
    pub shuffle: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl CrossValidationConfig {
    /// Create a new cross-validation configuration
    pub fn new() -> Self {
        Self {
            num_folds: 5,
            stratified: false,
            shuffle: true,
            random_seed: None,
        }
    }

    /// Set the number of folds
    pub fn with_num_folds(mut self, num_folds: usize) -> Self {
        self.num_folds = num_folds;
        self
    }

    /// Set whether to use stratified sampling
    pub fn with_stratified(mut self, stratified: bool) -> Self {
        self.stratified = stratified;
        self
    }

    /// Set whether to shuffle data
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set random seed
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }
}

impl Default for CrossValidationConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Cross-validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationResult {
    /// Number of folds used
    pub num_folds: usize,
    /// Metrics by fold and metric name
    pub metrics: HashMap<String, Vec<f64>>,
    /// Mean metrics
    pub mean_metrics: HashMap<String, f64>,
    /// Standard deviation of metrics
    pub std_metrics: HashMap<String, f64>,
}

/// Hyperparameter optimization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    /// Random search optimization
    RandomSearch,
    /// Grid search optimization
    GridSearch,
    /// Bayesian optimization (future implementation)
    Bayesian,
}

impl Default for OptimizationStrategy {
    fn default() -> Self {
        OptimizationStrategy::RandomSearch
    }
}

/// Optimize hyperparameters using the specified strategy
pub fn optimize_hyperparameters(
    dataset: &Dataset,
    param_space: &HyperparameterSpace,
    config: &OptimizationConfig,
) -> Result<OptimizationResult> {
    optimize_hyperparameters_with_strategy(
        dataset,
        param_space,
        config,
        OptimizationStrategy::default(),
    )
}

/// Optimize hyperparameters using a specific optimization strategy
pub fn optimize_hyperparameters_with_strategy(
    dataset: &Dataset,
    param_space: &HyperparameterSpace,
    config: &OptimizationConfig,
    strategy: OptimizationStrategy,
) -> Result<OptimizationResult> {
    use rand::SeedableRng;
    use std::time::{Duration, Instant};

    log::info!(
        "Starting hyperparameter optimization with strategy: {:?}",
        strategy
    );
    log::info!(
        "Parameters: {} trials, {} CV folds, metric: {}",
        config.num_trials,
        config.cv_folds,
        config.metric
    );

    // Validate parameter space
    validate_parameter_space(param_space)?;

    // Initialize random number generator
    let mut rng = match config.random_seed {
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
        None => rand::rngs::StdRng::from_entropy(),
    };

    // Track optimization state
    let start_time = Instant::now();
    let mut trials = Vec::new();
    let mut best_score = match config.direction {
        OptimizationDirection::Minimize => f64::INFINITY,
        OptimizationDirection::Maximize => f64::NEG_INFINITY,
    };
    let mut best_params = HashMap::new();
    let mut trials_without_improvement = 0;
    let mut last_improvement_trial = 0;

    // Create cross-validation configuration
    let mut cv_config = CrossValidationConfig::new()
        .with_num_folds(config.cv_folds)
        .with_shuffle(true);

    if let Some(seed) = config.random_seed {
        cv_config = cv_config.with_random_seed(seed);
    }

    // Generate parameter configurations based on strategy
    let param_configs = match strategy {
        OptimizationStrategy::RandomSearch => {
            generate_random_search_configs(param_space, config.num_trials, &mut rng)?
        }
        OptimizationStrategy::GridSearch => {
            generate_grid_search_configs(param_space, config.num_trials)?
        }
        OptimizationStrategy::Bayesian => {
            return Err(LightGBMError::not_implemented("Bayesian optimization"));
        }
    };

    log::info!("Generated {} parameter configurations", param_configs.len());

    // Run trials
    for (trial_id, params) in param_configs.into_iter().enumerate() {
        // Check timeout
        if let Some(timeout) = config.timeout_seconds {
            if start_time.elapsed() > Duration::from_secs(timeout) {
                log::warn!("Optimization timeout reached after {} trials", trial_id);
                break;
            }
        }

        log::debug!("Trial {}/{}: {:?}", trial_id + 1, config.num_trials, params);

        // Create model configuration with current parameters
        let model_config = create_model_config_from_params(&params)?;

        // Perform cross-validation
        let cv_result = cross_validate(dataset, &model_config, &cv_config, &[&config.metric])?;

        // Extract the target metric score
        let score = cv_result.mean_metrics.get(&config.metric).ok_or_else(|| {
            LightGBMError::config(format!(
                "Metric '{}' not found in cross-validation results",
                config.metric
            ))
        })?;

        let cv_scores = cv_result
            .metrics
            .get(&config.metric)
            .unwrap_or(&vec![])
            .clone();

        // Check if this is the best score so far
        let is_improvement = match config.direction {
            OptimizationDirection::Minimize => {
                *score < best_score - config.early_stopping_min_delta.unwrap_or(0.0)
            }
            OptimizationDirection::Maximize => {
                *score > best_score + config.early_stopping_min_delta.unwrap_or(0.0)
            }
        };

        if is_improvement {
            best_score = *score;
            best_params = params.clone();
            trials_without_improvement = 0;
            last_improvement_trial = trial_id;
            log::info!("New best score: {:.6} (trial {})", best_score, trial_id + 1);
        } else {
            trials_without_improvement += 1;
        }

        // Progress reporting
        if (trial_id + 1) % config.progress_interval == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let trials_per_sec = (trial_id + 1) as f64 / elapsed;
            log::info!(
                "Progress: {}/{} trials, best score: {:.6}, {:.1} trials/sec",
                trial_id + 1,
                config.num_trials,
                best_score,
                trials_per_sec
            );
        }

        // Early stopping check
        if let Some(patience) = config.early_stopping_patience {
            if trials_without_improvement >= patience {
                log::info!(
                    "Early stopping: no improvement for {} trials (last improvement at trial {})",
                    trials_without_improvement,
                    last_improvement_trial + 1
                );
                break;
            }
        }

        // Store trial result
        trials.push(TrialResult {
            trial_id,
            params,
            score: *score,
            cv_scores,
        });
    }

    let total_time = start_time.elapsed();
    log::info!(
        "Hyperparameter optimization completed in {:.2}s",
        total_time.as_secs_f64()
    );
    log::info!("Best score: {:.6}", best_score);
    log::info!("Best parameters: {:?}", best_params);

    Ok(OptimizationResult {
        num_trials: trials.len(),
        best_score,
        best_params,
        trials,
    })
}

/// Validate parameter space configuration
fn validate_parameter_space(param_space: &HyperparameterSpace) -> Result<()> {
    if param_space.float_params.is_empty()
        && param_space.int_params.is_empty()
        && param_space.categorical_params.is_empty()
    {
        return Err(LightGBMError::config("Parameter space is empty"));
    }

    // Validate float parameters
    for (name, (min, max)) in &param_space.float_params {
        if min >= max {
            return Err(LightGBMError::config(format!(
                "Float parameter '{}': min ({}) must be less than max ({})",
                name, min, max
            )));
        }
        if !min.is_finite() || !max.is_finite() {
            return Err(LightGBMError::config(format!(
                "Float parameter '{}': bounds must be finite",
                name
            )));
        }
    }

    // Validate integer parameters
    for (name, (min, max)) in &param_space.int_params {
        if min >= max {
            return Err(LightGBMError::config(format!(
                "Integer parameter '{}': min ({}) must be less than max ({})",
                name, min, max
            )));
        }
    }

    // Validate categorical parameters
    for (name, options) in &param_space.categorical_params {
        if options.is_empty() {
            return Err(LightGBMError::config(format!(
                "Categorical parameter '{}': must have at least one option",
                name
            )));
        }
    }

    Ok(())
}

/// Generate random search parameter configurations
fn generate_random_search_configs(
    param_space: &HyperparameterSpace,
    num_trials: usize,
    rng: &mut impl rand::Rng,
) -> Result<Vec<HashMap<String, f64>>> {
    let mut configs = Vec::with_capacity(num_trials);

    for _ in 0..num_trials {
        let mut params = HashMap::new();

        // Sample float parameters
        for (name, (min, max)) in &param_space.float_params {
            let value = rng.gen_range(*min..*max);
            params.insert(name.clone(), value);
        }

        // Sample integer parameters
        for (name, (min, max)) in &param_space.int_params {
            let value = rng.gen_range(*min..*max) as f64;
            params.insert(name.clone(), value);
        }

        // Sample categorical parameters
        for (name, options) in &param_space.categorical_params {
            let idx = rng.gen_range(0..options.len());
            // For categorical parameters, we'll use the index as the value
            // In practice, this would need better handling
            params.insert(name.clone(), idx as f64);
        }

        configs.push(params);
    }

    Ok(configs)
}

/// Generate grid search parameter configurations
fn generate_grid_search_configs(
    param_space: &HyperparameterSpace,
    max_trials: usize,
) -> Result<Vec<HashMap<String, f64>>> {
    let mut configs = Vec::new();

    // Calculate grid points for each parameter type
    let grid_points_per_param = 3; // Start with 3 points per parameter for feasibility

    // Collect all parameter specifications
    let mut float_grids: Vec<(String, Vec<f64>)> = Vec::new();
    let mut int_grids: Vec<(String, Vec<f64>)> = Vec::new();
    let mut categorical_grids: Vec<(String, Vec<f64>)> = Vec::new();

    // Generate grid points for float parameters
    for (name, (min, max)) in &param_space.float_params {
        let mut grid_points = Vec::new();
        for i in 0..grid_points_per_param {
            let ratio = i as f64 / (grid_points_per_param - 1) as f64;
            let value = min + ratio * (max - min);
            grid_points.push(value);
        }
        float_grids.push((name.clone(), grid_points));
    }

    // Generate grid points for integer parameters
    for (name, (min, max)) in &param_space.int_params {
        let mut grid_points = Vec::new();
        let range = max - min;
        let step = std::cmp::max(1, range / (grid_points_per_param - 1) as i32);

        let mut current = *min;
        while current <= *max && grid_points.len() < grid_points_per_param {
            grid_points.push(current as f64);
            current += step;
        }

        // Ensure we include the maximum value
        if grid_points
            .last()
            .map_or(true, |&last| (last as i32) < *max)
        {
            grid_points.push(*max as f64);
        }

        int_grids.push((name.clone(), grid_points));
    }

    // Generate grid points for categorical parameters
    for (name, options) in &param_space.categorical_params {
        let grid_points: Vec<f64> = (0..options.len()).map(|i| i as f64).collect();
        categorical_grids.push((name.clone(), grid_points));
    }

    // Generate all combinations using cartesian product
    if float_grids.is_empty() && int_grids.is_empty() && categorical_grids.is_empty() {
        return Ok(configs);
    }

    // Start with an empty configuration
    let mut partial_configs = vec![HashMap::new()];

    // Add float parameters
    for (name, values) in float_grids {
        let mut new_configs = Vec::new();
        for config in partial_configs {
            for &value in &values {
                let mut new_config = config.clone();
                new_config.insert(name.clone(), value);
                new_configs.push(new_config);
            }
        }
        partial_configs = new_configs;

        // Limit early if we're getting too many combinations
        if partial_configs.len() > max_trials {
            partial_configs.truncate(max_trials);
            break;
        }
    }

    // Add integer parameters
    for (name, values) in int_grids {
        let mut new_configs = Vec::new();
        for config in partial_configs {
            for &value in &values {
                let mut new_config = config.clone();
                new_config.insert(name.clone(), value);
                new_configs.push(new_config);
            }
        }
        partial_configs = new_configs;

        // Limit early if we're getting too many combinations
        if partial_configs.len() > max_trials {
            partial_configs.truncate(max_trials);
            break;
        }
    }

    // Add categorical parameters
    for (name, values) in categorical_grids {
        let mut new_configs = Vec::new();
        for config in partial_configs {
            for &value in &values {
                let mut new_config = config.clone();
                new_config.insert(name.clone(), value);
                new_configs.push(new_config);
            }
        }
        partial_configs = new_configs;

        // Limit early if we're getting too many combinations
        if partial_configs.len() > max_trials {
            partial_configs.truncate(max_trials);
            break;
        }
    }

    configs = partial_configs;
    configs.truncate(max_trials);

    Ok(configs)
}

/// Create model configuration from parameter map
fn create_model_config_from_params(params: &HashMap<String, f64>) -> Result<Config> {
    use crate::config::ConfigBuilder;

    let mut builder = ConfigBuilder::new();

    // Map common hyperparameters
    if let Some(&lr) = params.get("learning_rate") {
        builder = builder.learning_rate(lr);
    }

    if let Some(&leaves) = params.get("num_leaves") {
        builder = builder.num_leaves(leaves as usize);
    }

    if let Some(&depth) = params.get("max_depth") {
        builder = builder.max_depth(depth as i32);
    }

    if let Some(&samples) = params.get("min_data_in_leaf") {
        builder = builder.min_data_in_leaf(samples as i32);
    }

    if let Some(&subsample) = params.get("feature_fraction") {
        builder = builder.feature_fraction(subsample);
    }

    if let Some(&bagging) = params.get("bagging_fraction") {
        builder = builder.bagging_fraction(bagging);
    }

    // Note: reg_alpha and reg_lambda methods are not available in ConfigBuilder
    // These would need to be added to the Config struct if needed
    // For now, we'll skip these parameters

    builder.build()
}

/// Perform k-fold cross-validation on a dataset
pub fn cross_validate(
    dataset: &Dataset,
    model_config: &Config,
    cv_config: &CrossValidationConfig,
    metrics: &[&str],
) -> Result<CrossValidationResult> {
    use crate::metrics_eval::{evaluate_binary_classification, evaluate_regression};
    use crate::{LGBMClassifier, LGBMRegressor};
    use rand::SeedableRng;
    use std::collections::HashMap;

    log::info!(
        "Starting {}-fold cross-validation with {} metrics",
        cv_config.num_folds,
        metrics.len()
    );

    if cv_config.num_folds < 2 {
        return Err(LightGBMError::config("Number of folds must be at least 2"));
    }

    let num_samples = dataset.num_data();
    if num_samples < cv_config.num_folds {
        return Err(LightGBMError::config(format!(
            "Number of samples ({}) must be at least equal to number of folds ({})",
            num_samples, cv_config.num_folds
        )));
    }

    // Create indices for samples
    let mut indices: Vec<usize> = (0..num_samples).collect();

    // Shuffle indices if requested
    if cv_config.shuffle {
        if let Some(seed) = cv_config.random_seed {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rng);
        } else {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }
    }

    // Split indices into folds
    let fold_size = num_samples / cv_config.num_folds;
    let mut folds: Vec<Vec<usize>> = Vec::new();

    for fold_idx in 0..cv_config.num_folds {
        let start = fold_idx * fold_size;
        let end = if fold_idx == cv_config.num_folds - 1 {
            num_samples // Last fold gets remaining samples
        } else {
            start + fold_size
        };
        folds.push(indices[start..end].to_vec());
    }

    // Initialize result storage
    let mut fold_metrics: HashMap<String, Vec<f64>> = HashMap::new();
    for metric in metrics {
        fold_metrics.insert(metric.to_string(), Vec::new());
    }

    // Perform cross-validation
    for (fold_idx, test_indices) in folds.iter().enumerate() {
        log::debug!("Processing fold {}/{}", fold_idx + 1, cv_config.num_folds);

        // Create training indices (all except current fold)
        let mut train_indices = Vec::new();
        for (i, fold) in folds.iter().enumerate() {
            if i != fold_idx {
                train_indices.extend(fold);
            }
        }

        // Create training and test datasets
        let train_dataset = create_subset_dataset(dataset, &train_indices)?;
        let test_dataset = create_subset_dataset(dataset, test_indices)?;

        // Train model based on objective type
        let (test_predictions, test_probabilities) = match model_config.objective {
            crate::ObjectiveType::Regression => {
                let mut regressor = LGBMRegressor::new(model_config.clone());
                regressor.fit(&train_dataset)?;
                let predictions = regressor.predict(&test_dataset.features().to_owned())?;
                (predictions, None)
            }
            crate::ObjectiveType::Binary => {
                let mut classifier = LGBMClassifier::new(model_config.clone());
                classifier.fit(&train_dataset)?;
                let class_predictions = classifier.predict(&test_dataset.features().to_owned())?;
                let prob_predictions =
                    classifier.predict_proba(&test_dataset.features().to_owned())?;
                (class_predictions, Some(prob_predictions))
            }
            _ => {
                return Err(LightGBMError::not_implemented(
                    "Cross-validation for multiclass classification",
                ));
            }
        };

        // Calculate metrics for this fold
        let true_labels = test_dataset.labels();

        for metric in metrics {
            let metric_value = match (metric.to_lowercase().as_str(), &model_config.objective) {
                ("rmse", crate::ObjectiveType::Regression) => {
                    let reg_metrics =
                        evaluate_regression(&test_predictions.view(), &true_labels.view());
                    reg_metrics.rmse
                }
                ("mae", crate::ObjectiveType::Regression) => {
                    let reg_metrics =
                        evaluate_regression(&test_predictions.view(), &true_labels.view());
                    reg_metrics.mae
                }
                ("r2", crate::ObjectiveType::Regression) => {
                    let reg_metrics =
                        evaluate_regression(&test_predictions.view(), &true_labels.view());
                    reg_metrics.r2
                }
                ("accuracy", crate::ObjectiveType::Binary) => {
                    if let Some(ref probs) = test_probabilities {
                        let class_metrics = evaluate_binary_classification(
                            &test_predictions.view(),
                            &probs.view(),
                            &true_labels.view(),
                        );
                        class_metrics.accuracy
                    } else {
                        return Err(LightGBMError::prediction(
                            "Missing probability predictions for classification",
                        ));
                    }
                }
                ("f1", crate::ObjectiveType::Binary) => {
                    if let Some(ref probs) = test_probabilities {
                        let class_metrics = evaluate_binary_classification(
                            &test_predictions.view(),
                            &probs.view(),
                            &true_labels.view(),
                        );
                        class_metrics.f1_score
                    } else {
                        return Err(LightGBMError::prediction(
                            "Missing probability predictions for classification",
                        ));
                    }
                }
                ("auc", crate::ObjectiveType::Binary) => {
                    if let Some(ref probs) = test_probabilities {
                        let class_metrics = evaluate_binary_classification(
                            &test_predictions.view(),
                            &probs.view(),
                            &true_labels.view(),
                        );
                        class_metrics.auc
                    } else {
                        return Err(LightGBMError::prediction(
                            "Missing probability predictions for classification",
                        ));
                    }
                }
                _ => {
                    return Err(LightGBMError::config(format!(
                        "Unsupported metric '{}' for objective '{:?}'",
                        metric, model_config.objective
                    )));
                }
            };

            fold_metrics
                .get_mut(&metric.to_string())
                .unwrap()
                .push(metric_value);
        }
    }

    // Calculate mean and standard deviation for each metric
    let mut mean_metrics = HashMap::new();
    let mut std_metrics = HashMap::new();

    for (metric_name, values) in &fold_metrics {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        mean_metrics.insert(metric_name.clone(), mean);
        std_metrics.insert(metric_name.clone(), std_dev);
    }

    log::info!("Cross-validation completed successfully");
    for (metric, mean_val) in &mean_metrics {
        let std_val = std_metrics.get(metric).unwrap_or(&0.0);
        log::info!("  {}: {:.4} ± {:.4}", metric, mean_val, std_val);
    }

    Ok(CrossValidationResult {
        num_folds: cv_config.num_folds,
        metrics: fold_metrics,
        mean_metrics,
        std_metrics,
    })
}

/// Create a subset dataset from indices
fn create_subset_dataset(dataset: &Dataset, indices: &[usize]) -> Result<Dataset> {
    use ndarray::{Array1, Array2};

    let num_samples = indices.len();
    let num_features = dataset.num_features();

    // Extract features for selected indices
    let mut features_data = Vec::with_capacity(num_samples * num_features);
    let mut labels_data = Vec::with_capacity(num_samples);

    for &idx in indices {
        if idx >= dataset.num_data() {
            return Err(LightGBMError::data_loading(format!(
                "Index {} out of bounds for dataset with {} samples",
                idx,
                dataset.num_data()
            )));
        }

        // Extract features for this sample
        for feature_idx in 0..num_features {
            features_data.push(dataset.features()[[idx, feature_idx]]);
        }

        // Extract label for this sample
        labels_data.push(dataset.labels()[idx]);
    }

    // Create arrays
    let features =
        Array2::from_shape_vec((num_samples, num_features), features_data).map_err(|e| {
            LightGBMError::data_loading(format!("Failed to create subset features: {}", e))
        })?;
    let labels = Array1::from_vec(labels_data);

    // Create subset dataset
    Dataset::new(
        features,
        labels,
        None, // weights
        None, // groups
        dataset.feature_names().map(|names| names.to_vec()),
        None, // feature_types
    )
}
