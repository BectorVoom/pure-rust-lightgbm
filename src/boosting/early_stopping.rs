//! Early stopping mechanism for the Pure Rust LightGBM framework.
//!
//! This module provides early stopping functionality to prevent overfitting
//! by monitoring validation metrics and stopping training when improvements plateau.

use crate::core::types::IterationIndex;
use std::collections::VecDeque;

/// Configuration for early stopping behavior.
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Number of iterations to wait for improvement before stopping
    pub patience: usize,
    /// Minimum improvement required to reset patience counter
    pub min_delta: f64,
    /// Whether lower metric values are better (e.g., loss) or higher (e.g., accuracy)
    pub minimize: bool,
    /// Percentage of training data to use for early stopping validation
    pub validation_fraction: f64,
    /// Whether to use relative improvement instead of absolute
    pub use_relative_delta: bool,
    /// Smoothing window size for metric averaging
    pub smoothing_window: usize,
    /// Whether to restore best weights when stopping
    pub restore_best_weights: bool,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        EarlyStoppingConfig {
            patience: 10,
            min_delta: 1e-4,
            minimize: true,
            validation_fraction: 0.1,
            use_relative_delta: false,
            smoothing_window: 1,
            restore_best_weights: true,
        }
    }
}

/// Tracks validation metrics and determines when to stop training.
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    config: EarlyStoppingConfig,
    best_metric: f64,
    best_iteration: IterationIndex,
    patience_counter: usize,
    metric_history: VecDeque<f64>,
    smoothed_history: VecDeque<f64>,
    stopped: bool,
    improvement_history: Vec<f64>,
}

impl EarlyStopping {
    /// Creates a new early stopping monitor with the given configuration.
    pub fn new(config: EarlyStoppingConfig) -> Self {
        let initial_metric = if config.minimize {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };

        EarlyStopping {
            config,
            best_metric: initial_metric,
            best_iteration: 0,
            patience_counter: 0,
            metric_history: VecDeque::new(),
            smoothed_history: VecDeque::new(),
            stopped: false,
            improvement_history: Vec::new(),
        }
    }

    /// Updates the early stopping monitor with a new metric value.
    pub fn update(&mut self, metric: f64, iteration: IterationIndex) -> bool {
        if self.stopped {
            return true;
        }

        // Add to history
        self.metric_history.push_back(metric);

        // Apply smoothing if configured
        let smoothed_metric = self.apply_smoothing(metric);
        self.smoothed_history.push_back(smoothed_metric);

        // Limit history size to prevent unbounded growth
        if self.metric_history.len() > 1000 {
            self.metric_history.pop_front();
        }
        if self.smoothed_history.len() > 1000 {
            self.smoothed_history.pop_front();
        }

        // Check for improvement
        let has_improved = self.check_improvement(smoothed_metric, iteration);

        if has_improved {
            self.best_metric = smoothed_metric;
            self.best_iteration = iteration;
            self.patience_counter = 0;

            // Record improvement
            let improvement = self.calculate_improvement(smoothed_metric);
            self.improvement_history.push(improvement);
        } else {
            self.patience_counter += 1;
            self.improvement_history.push(0.0);
        }

        // Check if we should stop
        if self.patience_counter >= self.config.patience {
            self.stopped = true;
            log::info!(
                "Early stopping triggered at iteration {} (best was {} at iteration {})",
                iteration,
                self.best_metric,
                self.best_iteration
            );
        }

        self.stopped
    }

    /// Applies smoothing to the metric using a rolling window.
    fn apply_smoothing(&self, current_metric: f64) -> f64 {
        if self.config.smoothing_window <= 1 {
            return current_metric;
        }

        let window_size = self
            .config
            .smoothing_window
            .min(self.metric_history.len() + 1);
        let mut sum = current_metric;
        let mut count = 1;

        // Sum the last (window_size - 1) metrics
        for &metric in self.metric_history.iter().rev().take(window_size - 1) {
            sum += metric;
            count += 1;
        }

        sum / count as f64
    }

    /// Checks if the current metric represents an improvement.
    fn check_improvement(&self, metric: f64, _iteration: IterationIndex) -> bool {
        if self.metric_history.len() == 1 {
            // First metric is always considered an improvement
            return true;
        }

        let improvement = if self.config.minimize {
            self.best_metric - metric
        } else {
            metric - self.best_metric
        };

        if self.config.use_relative_delta && self.best_metric.abs() > 1e-8 {
            let relative_improvement = improvement / self.best_metric.abs();
            relative_improvement >= self.config.min_delta
        } else {
            improvement >= self.config.min_delta
        }
    }

    /// Calculates the improvement magnitude for the current metric.
    fn calculate_improvement(&self, metric: f64) -> f64 {
        if self.config.minimize {
            self.best_metric - metric
        } else {
            metric - self.best_metric
        }
    }

    /// Returns true if early stopping has been triggered.
    pub fn should_stop(&self) -> bool {
        self.stopped
    }

    /// Returns the best metric value observed so far.
    pub fn best_metric(&self) -> f64 {
        self.best_metric
    }

    /// Returns the iteration where the best metric was observed.
    pub fn best_iteration(&self) -> IterationIndex {
        self.best_iteration
    }

    /// Returns the current patience counter value.
    pub fn patience_counter(&self) -> usize {
        self.patience_counter
    }

    /// Returns the metric history.
    pub fn metric_history(&self) -> &VecDeque<f64> {
        &self.metric_history
    }

    /// Returns the smoothed metric history.
    pub fn smoothed_history(&self) -> &VecDeque<f64> {
        &self.smoothed_history
    }

    /// Returns the improvement history.
    pub fn improvement_history(&self) -> &[f64] {
        &self.improvement_history
    }

    /// Resets the early stopping state.
    pub fn reset(&mut self) {
        self.best_metric = if self.config.minimize {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
        self.best_iteration = 0;
        self.patience_counter = 0;
        self.metric_history.clear();
        self.smoothed_history.clear();
        self.stopped = false;
        self.improvement_history.clear();
    }

    /// Updates the configuration.
    pub fn update_config(&mut self, config: EarlyStoppingConfig) {
        self.config = config;
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &EarlyStoppingConfig {
        &self.config
    }

    /// Estimates remaining iterations based on improvement trend.
    pub fn estimate_remaining_iterations(&self) -> Option<usize> {
        if self.improvement_history.len() < 5 {
            return None;
        }

        // Calculate recent improvement trend
        let recent_window = 5.min(self.improvement_history.len());
        let recent_improvements: Vec<f64> = self
            .improvement_history
            .iter()
            .rev()
            .take(recent_window)
            .copied()
            .collect();

        let avg_recent_improvement =
            recent_improvements.iter().sum::<f64>() / recent_improvements.len() as f64;

        // If improvement is consistently near zero, estimate based on patience
        if avg_recent_improvement < self.config.min_delta / 10.0 {
            Some(self.config.patience - self.patience_counter)
        } else {
            // Use simple linear extrapolation
            let remaining_patience = self.config.patience - self.patience_counter;
            Some(remaining_patience * 2) // Conservative estimate
        }
    }

    /// Returns statistics about the early stopping behavior.
    pub fn statistics(&self) -> EarlyStoppingStatistics {
        let total_iterations = self.metric_history.len();
        let improvements = self
            .improvement_history
            .iter()
            .filter(|&&x| x > 0.0)
            .count();
        let avg_improvement = if improvements > 0 {
            self.improvement_history
                .iter()
                .filter(|&&x| x > 0.0)
                .sum::<f64>()
                / improvements as f64
        } else {
            0.0
        };

        let stability_score = if total_iterations > 1 {
            let variance = self.calculate_metric_variance();
            1.0 / (1.0 + variance)
        } else {
            0.0
        };

        EarlyStoppingStatistics {
            total_iterations,
            best_iteration: self.best_iteration,
            best_metric: self.best_metric,
            improvements_count: improvements,
            average_improvement: avg_improvement,
            patience_used: self.patience_counter,
            stability_score,
            is_stopped: self.stopped,
        }
    }

    /// Calculates the variance of the smoothed metrics.
    fn calculate_metric_variance(&self) -> f64 {
        if self.smoothed_history.len() < 2 {
            return 0.0;
        }

        let mean = self.smoothed_history.iter().sum::<f64>() / self.smoothed_history.len() as f64;
        let variance = self
            .smoothed_history
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / self.smoothed_history.len() as f64;

        variance
    }
}

/// Statistics about early stopping behavior.
#[derive(Debug, Clone)]
pub struct EarlyStoppingStatistics {
    /// Total number of iterations monitored
    pub total_iterations: usize,
    /// Iteration with the best metric
    pub best_iteration: IterationIndex,
    /// Best metric value achieved
    pub best_metric: f64,
    /// Number of iterations that showed improvement
    pub improvements_count: usize,
    /// Average improvement when improvement occurred
    pub average_improvement: f64,
    /// Current patience counter value
    pub patience_used: usize,
    /// Stability score (0.0 to 1.0, higher is more stable)
    pub stability_score: f64,
    /// Whether early stopping has been triggered
    pub is_stopped: bool,
}

impl std::fmt::Display for EarlyStoppingStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "EarlyStoppingStats(iterations={}, best@{}={:.6}, improvements={}, avg_improvement={:.6}, patience={}, stability={:.3}, stopped={})",
            self.total_iterations,
            self.best_iteration,
            self.best_metric,
            self.improvements_count,
            self.average_improvement,
            self.patience_used,
            self.stability_score,
            self.is_stopped
        )
    }
}

/// Multi-metric early stopping for monitoring multiple validation metrics.
#[derive(Debug, Clone)]
pub struct MultiMetricEarlyStopping {
    monitors: Vec<(String, EarlyStopping)>,
    combination_strategy: CombinationStrategy,
    stopped: bool,
}

/// Strategy for combining multiple metrics in early stopping decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombinationStrategy {
    /// Stop when ANY metric triggers early stopping
    Any,
    /// Stop when ALL metrics trigger early stopping
    All,
    /// Stop when the MAJORITY of metrics trigger early stopping
    Majority,
    /// Use a PRIMARY metric for decisions, others are informational
    Primary,
}

impl MultiMetricEarlyStopping {
    /// Creates a new multi-metric early stopping monitor.
    pub fn new(strategy: CombinationStrategy) -> Self {
        MultiMetricEarlyStopping {
            monitors: Vec::new(),
            combination_strategy: strategy,
            stopped: false,
        }
    }

    /// Adds a metric to monitor.
    pub fn add_metric(&mut self, name: String, config: EarlyStoppingConfig) {
        let monitor = EarlyStopping::new(config);
        self.monitors.push((name, monitor));
    }

    /// Updates all metrics and returns whether training should stop.
    pub fn update(&mut self, metrics: &[(String, f64)], iteration: IterationIndex) -> bool {
        if self.stopped {
            return true;
        }

        // Update each monitor
        for (metric_name, metric_value) in metrics {
            if let Some((_, monitor)) = self
                .monitors
                .iter_mut()
                .find(|(name, _)| name == metric_name)
            {
                monitor.update(*metric_value, iteration);
            }
        }

        // Determine if we should stop based on combination strategy
        self.stopped = match self.combination_strategy {
            CombinationStrategy::Any => self
                .monitors
                .iter()
                .any(|(_, monitor)| monitor.should_stop()),
            CombinationStrategy::All => {
                !self.monitors.is_empty()
                    && self
                        .monitors
                        .iter()
                        .all(|(_, monitor)| monitor.should_stop())
            }
            CombinationStrategy::Majority => {
                let stopped_count = self
                    .monitors
                    .iter()
                    .filter(|(_, monitor)| monitor.should_stop())
                    .count();
                stopped_count > self.monitors.len() / 2
            }
            CombinationStrategy::Primary => {
                // Use first metric as primary
                self.monitors
                    .first()
                    .map_or(false, |(_, monitor)| monitor.should_stop())
            }
        };

        if self.stopped {
            log::info!(
                "Multi-metric early stopping triggered at iteration {}",
                iteration
            );
            for (name, monitor) in &self.monitors {
                if monitor.should_stop() {
                    log::info!(
                        "  {} stopped (best: {:.6} at iteration {})",
                        name,
                        monitor.best_metric(),
                        monitor.best_iteration()
                    );
                }
            }
        }

        self.stopped
    }

    /// Returns whether early stopping has been triggered.
    pub fn should_stop(&self) -> bool {
        self.stopped
    }

    /// Returns statistics for all monitored metrics.
    pub fn statistics(&self) -> Vec<(String, EarlyStoppingStatistics)> {
        self.monitors
            .iter()
            .map(|(name, monitor)| (name.clone(), monitor.statistics()))
            .collect()
    }

    /// Returns the best iteration considering all metrics.
    pub fn best_iteration(&self) -> IterationIndex {
        match self.combination_strategy {
            CombinationStrategy::Primary => self
                .monitors
                .first()
                .map_or(0, |(_, monitor)| monitor.best_iteration()),
            _ => {
                // Return the most common best iteration
                let iterations: Vec<IterationIndex> = self
                    .monitors
                    .iter()
                    .map(|(_, monitor)| monitor.best_iteration())
                    .collect();

                if iterations.is_empty() {
                    0
                } else {
                    // Simple approach: return the median
                    let mut sorted_iterations = iterations;
                    sorted_iterations.sort_unstable();
                    sorted_iterations[sorted_iterations.len() / 2]
                }
            }
        }
    }

    /// Resets all monitors.
    pub fn reset(&mut self) {
        for (_, monitor) in &mut self.monitors {
            monitor.reset();
        }
        self.stopped = false;
    }

    /// Returns a reference to a specific monitor.
    pub fn get_monitor(&self, name: &str) -> Option<&EarlyStopping> {
        self.monitors
            .iter()
            .find(|(monitor_name, _)| monitor_name == name)
            .map(|(_, monitor)| monitor)
    }

    /// Returns a mutable reference to a specific monitor.
    pub fn get_monitor_mut(&mut self, name: &str) -> Option<&mut EarlyStopping> {
        self.monitors
            .iter_mut()
            .find(|(monitor_name, _)| monitor_name == name)
            .map(|(_, monitor)| monitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_stopping_creation() {
        let config = EarlyStoppingConfig::default();
        let early_stopping = EarlyStopping::new(config);

        assert!(!early_stopping.should_stop());
        assert_eq!(early_stopping.patience_counter(), 0);
        assert_eq!(early_stopping.best_metric(), f64::INFINITY);
    }

    #[test]
    fn test_early_stopping_improvement() {
        let config = EarlyStoppingConfig {
            patience: 3,
            min_delta: 0.01,
            minimize: true,
            ..Default::default()
        };
        let mut early_stopping = EarlyStopping::new(config);

        // First update should always be improvement
        assert!(!early_stopping.update(1.0, 0));
        assert_eq!(early_stopping.best_metric(), 1.0);
        assert_eq!(early_stopping.patience_counter(), 0);

        // Improvement should reset patience
        assert!(!early_stopping.update(0.5, 1));
        assert_eq!(early_stopping.best_metric(), 0.5);
        assert_eq!(early_stopping.patience_counter(), 0);

        // No improvement should increase patience
        assert!(!early_stopping.update(0.55, 2));
        assert_eq!(early_stopping.patience_counter(), 1);

        assert!(!early_stopping.update(0.6, 3));
        assert_eq!(early_stopping.patience_counter(), 2);

        // Should trigger early stopping
        assert!(early_stopping.update(0.65, 4));
        assert!(early_stopping.should_stop());
    }

    #[test]
    fn test_early_stopping_minimize_vs_maximize() {
        // Test minimization (default)
        let config_min = EarlyStoppingConfig {
            patience: 2,
            min_delta: 0.01,
            minimize: true,
            ..Default::default()
        };
        let mut early_stopping_min = EarlyStopping::new(config_min);

        early_stopping_min.update(1.0, 0);
        assert!(!early_stopping_min.update(0.5, 1)); // Improvement (lower is better)
        assert!(early_stopping_min.update(0.6, 2)); // No improvement
        assert!(early_stopping_min.update(0.7, 3)); // Should stop

        // Test maximization
        let config_max = EarlyStoppingConfig {
            patience: 2,
            min_delta: 0.01,
            minimize: false,
            ..Default::default()
        };
        let mut early_stopping_max = EarlyStopping::new(config_max);

        early_stopping_max.update(0.5, 0);
        assert!(!early_stopping_max.update(0.8, 1)); // Improvement (higher is better)
        assert!(early_stopping_max.update(0.7, 2)); // No improvement
        assert!(early_stopping_max.update(0.6, 3)); // Should stop
    }

    #[test]
    fn test_smoothing() {
        let config = EarlyStoppingConfig {
            patience: 5,
            min_delta: 0.01,
            minimize: true,
            smoothing_window: 3,
            ..Default::default()
        };
        let mut early_stopping = EarlyStopping::new(config);

        early_stopping.update(1.0, 0);
        early_stopping.update(1.2, 1);
        early_stopping.update(0.8, 2); // Average should be (1.0 + 1.2 + 0.8) / 3 = 1.0

        let smoothed = early_stopping.smoothed_history();
        assert_eq!(smoothed.len(), 3);
        assert!((smoothed[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_relative_delta() {
        let config = EarlyStoppingConfig {
            patience: 2,
            min_delta: 0.1, // 10% relative improvement required
            minimize: true,
            use_relative_delta: true,
            ..Default::default()
        };
        let mut early_stopping = EarlyStopping::new(config);

        early_stopping.update(1.0, 0);
        assert!(!early_stopping.update(0.8, 1)); // 20% improvement - should be good
        assert!(early_stopping.update(0.79, 2)); // 1.25% improvement - not enough
        assert!(early_stopping.update(0.78, 3)); // Should stop
    }

    #[test]
    fn test_multi_metric_early_stopping() {
        let mut multi_es = MultiMetricEarlyStopping::new(CombinationStrategy::Any);

        let config1 = EarlyStoppingConfig {
            patience: 2,
            min_delta: 0.01,
            minimize: true,
            ..Default::default()
        };

        let config2 = EarlyStoppingConfig {
            patience: 3,
            min_delta: 0.01,
            minimize: false,
            ..Default::default()
        };

        multi_es.add_metric("loss".to_string(), config1);
        multi_es.add_metric("accuracy".to_string(), config2);

        // Update with improvements
        let metrics = vec![("loss".to_string(), 1.0), ("accuracy".to_string(), 0.8)];
        assert!(!multi_es.update(&metrics, 0));

        let metrics = vec![("loss".to_string(), 0.5), ("accuracy".to_string(), 0.9)];
        assert!(!multi_es.update(&metrics, 1));

        // Loss stops improving, should trigger early stopping with "Any" strategy
        let metrics = vec![("loss".to_string(), 0.6), ("accuracy".to_string(), 0.95)];
        assert!(!multi_es.update(&metrics, 2));

        let metrics = vec![("loss".to_string(), 0.7), ("accuracy".to_string(), 0.96)];
        assert!(multi_es.update(&metrics, 3)); // Loss triggers stop
    }

    #[test]
    fn test_statistics() {
        let config = EarlyStoppingConfig::default();
        let mut early_stopping = EarlyStopping::new(config);

        early_stopping.update(1.0, 0);
        early_stopping.update(0.8, 1);
        early_stopping.update(0.9, 2);

        let stats = early_stopping.statistics();
        assert_eq!(stats.total_iterations, 3);
        assert_eq!(stats.best_iteration, 1);
        assert_eq!(stats.best_metric, 0.8);
        assert_eq!(stats.improvements_count, 2);
        assert!(!stats.is_stopped);
    }

    #[test]
    fn test_reset() {
        let config = EarlyStoppingConfig::default();
        let mut early_stopping = EarlyStopping::new(config);

        early_stopping.update(1.0, 0);
        early_stopping.update(0.8, 1);
        assert_eq!(early_stopping.metric_history().len(), 2);

        early_stopping.reset();
        assert_eq!(early_stopping.metric_history().len(), 0);
        assert_eq!(early_stopping.best_metric(), f64::INFINITY);
        assert_eq!(early_stopping.patience_counter(), 0);
        assert!(!early_stopping.should_stop());
    }

    #[test]
    fn test_estimate_remaining_iterations() {
        let config = EarlyStoppingConfig {
            patience: 10,
            ..Default::default()
        };
        let mut early_stopping = EarlyStopping::new(config);

        // Not enough history
        early_stopping.update(1.0, 0);
        assert!(early_stopping.estimate_remaining_iterations().is_none());

        // Add more history
        for i in 1..8 {
            early_stopping.update(1.0 - i as f64 * 0.01, i);
        }

        let estimate = early_stopping.estimate_remaining_iterations();
        assert!(estimate.is_some());
        assert!(estimate.unwrap() > 0);
    }
}
