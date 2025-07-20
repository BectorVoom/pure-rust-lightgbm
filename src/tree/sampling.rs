//! Feature sampling utilities for the Pure Rust LightGBM framework.
//!
//! This module provides feature sampling strategies for regularization
//! and performance optimization during tree construction.

use crate::core::types::FeatureIndex;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::HashSet;

/// Configuration for feature sampling.
#[derive(Debug, Clone)]
pub struct FeatureSamplingConfig {
    /// Fraction of features to sample (0.0 to 1.0)
    pub feature_fraction: f64,
    /// Fraction of features to sample for column (feature) subsampling
    pub feature_fraction_bynode: f64,
    /// Random seed for reproducible sampling
    pub seed: u64,
    /// Whether to sample features per tree or per node
    pub sample_per_node: bool,
    /// Minimum number of features to always sample
    pub min_features: usize,
    /// Maximum number of features to sample
    pub max_features: Option<usize>,
    /// Whether to use deterministic sampling
    pub deterministic: bool,
}

impl Default for FeatureSamplingConfig {
    fn default() -> Self {
        FeatureSamplingConfig {
            feature_fraction: 1.0,
            feature_fraction_bynode: 1.0,
            seed: 42,
            sample_per_node: false,
            min_features: 1,
            max_features: None,
            deterministic: false,
        }
    }
}

/// Feature sampling strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Sample features uniformly at random
    Uniform,
    /// Sample features with probability proportional to their importance
    Importance,
    /// Sample features using stratified sampling
    Stratified,
    /// Sample features using systematic sampling
    Systematic,
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        SamplingStrategy::Uniform
    }
}

/// Feature sampler for tree construction.
pub struct FeatureSampler {
    config: FeatureSamplingConfig,
    rng: Xoshiro256PlusPlus,
    strategy: SamplingStrategy,
    feature_importance: Option<Vec<f64>>,
    cached_samples: Option<Vec<FeatureIndex>>,
    last_sample_iteration: Option<usize>,
}

impl FeatureSampler {
    /// Creates a new feature sampler with the given configuration.
    pub fn new(config: FeatureSamplingConfig) -> Self {
        let rng = Xoshiro256PlusPlus::seed_from_u64(config.seed);

        FeatureSampler {
            config,
            rng,
            strategy: SamplingStrategy::default(),
            feature_importance: None,
            cached_samples: None,
            last_sample_iteration: None,
        }
    }

    /// Creates a new feature sampler with a specific strategy.
    pub fn with_strategy(config: FeatureSamplingConfig, strategy: SamplingStrategy) -> Self {
        let mut sampler = Self::new(config);
        sampler.strategy = strategy;
        sampler
    }

    /// Samples features for tree construction.
    pub fn sample_features(
        &mut self,
        num_features: usize,
        iteration: usize,
    ) -> anyhow::Result<Vec<FeatureIndex>> {
        if num_features == 0 {
            return Ok(Vec::new());
        }

        // Check if we can use cached samples
        if let Some(cached) = &self.cached_samples {
            if !self.config.sample_per_node && self.last_sample_iteration == Some(iteration) {
                return Ok(cached.clone());
            }
        }

        let fraction = if self.config.sample_per_node {
            self.config.feature_fraction_bynode
        } else {
            self.config.feature_fraction
        };

        let num_to_sample = self.calculate_sample_size(num_features, fraction)?;

        let sampled_features = match self.strategy {
            SamplingStrategy::Uniform => self.sample_uniform(num_features, num_to_sample)?,
            SamplingStrategy::Importance => {
                self.sample_by_importance(num_features, num_to_sample)?
            }
            SamplingStrategy::Stratified => self.sample_stratified(num_features, num_to_sample)?,
            SamplingStrategy::Systematic => self.sample_systematic(num_features, num_to_sample)?,
        };

        // Cache the result if sampling per tree
        if !self.config.sample_per_node {
            self.cached_samples = Some(sampled_features.clone());
            self.last_sample_iteration = Some(iteration);
        }

        Ok(sampled_features)
    }

    /// Samples features for a specific node.
    pub fn sample_features_for_node(
        &mut self,
        num_features: usize,
        node_depth: usize,
        _node_id: usize,
    ) -> anyhow::Result<Vec<FeatureIndex>> {
        // Adjust sampling fraction based on node depth for regularization
        let depth_factor = 1.0 - (node_depth as f64 * 0.1).min(0.5);
        let adjusted_fraction = self.config.feature_fraction_bynode * depth_factor;

        let num_to_sample = self.calculate_sample_size(num_features, adjusted_fraction)?;

        match self.strategy {
            SamplingStrategy::Uniform => self.sample_uniform(num_features, num_to_sample),
            SamplingStrategy::Importance => self.sample_by_importance(num_features, num_to_sample),
            SamplingStrategy::Stratified => self.sample_stratified(num_features, num_to_sample),
            SamplingStrategy::Systematic => self.sample_systematic(num_features, num_to_sample),
        }
    }

    /// Updates feature importance scores for importance-based sampling.
    pub fn update_feature_importance(&mut self, importance: Vec<f64>) {
        if importance.is_empty() {
            self.feature_importance = None;
        } else {
            self.feature_importance = Some(importance);
        }
    }

    /// Uniform random sampling.
    fn sample_uniform(
        &mut self,
        num_features: usize,
        num_to_sample: usize,
    ) -> anyhow::Result<Vec<FeatureIndex>> {
        if self.config.deterministic {
            // Deterministic sampling using fixed intervals
            let mut features = Vec::new();
            let step = num_features as f64 / num_to_sample as f64;

            for i in 0..num_to_sample {
                let feature_idx = ((i as f64 * step) as usize).min(num_features - 1);
                features.push(feature_idx);
            }

            Ok(features)
        } else {
            // Random sampling without replacement
            let mut features: Vec<FeatureIndex> = (0..num_features).collect();
            features.shuffle(&mut self.rng);
            features.truncate(num_to_sample);
            features.sort_unstable();
            Ok(features)
        }
    }

    /// Importance-based sampling using feature importance scores.
    fn sample_by_importance(
        &mut self,
        num_features: usize,
        num_to_sample: usize,
    ) -> anyhow::Result<Vec<FeatureIndex>> {
        let importance = match &self.feature_importance {
            Some(imp) if imp.len() == num_features => imp,
            _ => {
                // Fall back to uniform sampling if no importance available
                return self.sample_uniform(num_features, num_to_sample);
            }
        };

        // Calculate sampling probabilities based on importance
        let total_importance: f64 = importance.iter().sum();
        if total_importance <= 0.0 {
            return self.sample_uniform(num_features, num_to_sample);
        }

        let mut probabilities: Vec<f64> = importance
            .iter()
            .map(|&imp| imp / total_importance)
            .collect();

        // Add small epsilon to ensure all features have some probability
        let epsilon = 1e-6;
        for prob in &mut probabilities {
            *prob += epsilon;
        }

        let total_prob: f64 = probabilities.iter().sum();
        for prob in &mut probabilities {
            *prob /= total_prob;
        }

        // Sample features based on probabilities
        let mut sampled = HashSet::new();
        let mut features = Vec::new();

        while features.len() < num_to_sample && sampled.len() < num_features {
            let random_val: f64 = self.rng.gen();
            let mut cumulative_prob = 0.0;

            for (feature_idx, &prob) in probabilities.iter().enumerate() {
                cumulative_prob += prob;
                if random_val <= cumulative_prob && !sampled.contains(&feature_idx) {
                    sampled.insert(feature_idx);
                    features.push(feature_idx);
                    break;
                }
            }

            // Prevent infinite loop
            if features.len() == sampled.len() && sampled.len() < num_to_sample {
                // Add remaining features randomly
                for feature_idx in 0..num_features {
                    if !sampled.contains(&feature_idx) && features.len() < num_to_sample {
                        features.push(feature_idx);
                    }
                }
            }
        }

        features.sort_unstable();
        Ok(features)
    }

    /// Stratified sampling dividing features into strata.
    fn sample_stratified(
        &mut self,
        num_features: usize,
        num_to_sample: usize,
    ) -> anyhow::Result<Vec<FeatureIndex>> {
        let num_strata = (num_features as f64).sqrt().ceil() as usize;
        let strata_size = (num_features + num_strata - 1) / num_strata; // Ceiling division
        let samples_per_stratum = num_to_sample / num_strata;
        let extra_samples = num_to_sample % num_strata;

        let mut features = Vec::new();

        for stratum_idx in 0..num_strata {
            let start_idx = stratum_idx * strata_size;
            let end_idx = ((stratum_idx + 1) * strata_size).min(num_features);

            if start_idx >= num_features {
                break;
            }

            let current_stratum_size = end_idx - start_idx;
            let mut samples_from_stratum = samples_per_stratum;

            // Distribute extra samples to first strata
            if stratum_idx < extra_samples {
                samples_from_stratum += 1;
            }

            samples_from_stratum = samples_from_stratum.min(current_stratum_size);

            // Sample from this stratum
            let mut stratum_features: Vec<FeatureIndex> = (start_idx..end_idx).collect();
            stratum_features.shuffle(&mut self.rng);
            stratum_features.truncate(samples_from_stratum);

            features.extend(stratum_features);
        }

        features.sort_unstable();
        Ok(features)
    }

    /// Systematic sampling with fixed intervals.
    fn sample_systematic(
        &mut self,
        num_features: usize,
        num_to_sample: usize,
    ) -> anyhow::Result<Vec<FeatureIndex>> {
        if num_to_sample >= num_features {
            return Ok((0..num_features).collect());
        }

        let interval = num_features as f64 / num_to_sample as f64;
        let start_offset: f64 = self.rng.gen_range(0.0..interval);

        let mut features = Vec::new();
        let mut current_pos = start_offset;

        while features.len() < num_to_sample {
            let feature_idx = (current_pos as usize).min(num_features - 1);
            features.push(feature_idx);
            current_pos += interval;

            if current_pos >= num_features as f64 {
                break;
            }
        }

        // Remove duplicates and sort
        features.sort_unstable();
        features.dedup();

        // If we don't have enough features due to rounding, fill with remaining
        while features.len() < num_to_sample {
            for candidate in 0..num_features {
                if !features.contains(&candidate) {
                    features.push(candidate);
                    if features.len() >= num_to_sample {
                        break;
                    }
                }
            }
            break; // Prevent infinite loop
        }

        features.sort_unstable();
        Ok(features)
    }

    /// Calculates the number of features to sample based on fraction and constraints.
    fn calculate_sample_size(&self, num_features: usize, fraction: f64) -> anyhow::Result<usize> {
        if fraction <= 0.0 || fraction > 1.0 {
            return Err(anyhow::anyhow!(
                "Feature fraction must be between 0.0 and 1.0"
            ));
        }

        let mut num_to_sample = (num_features as f64 * fraction).round() as usize;

        // Apply constraints
        num_to_sample = num_to_sample.max(self.config.min_features);

        if let Some(max_features) = self.config.max_features {
            num_to_sample = num_to_sample.min(max_features);
        }

        num_to_sample = num_to_sample.min(num_features);

        Ok(num_to_sample)
    }

    /// Resets the sampler state and clears cached samples.
    pub fn reset(&mut self) {
        self.cached_samples = None;
        self.last_sample_iteration = None;
        self.rng = Xoshiro256PlusPlus::seed_from_u64(self.config.seed);
    }

    /// Updates the sampling configuration.
    pub fn update_config(&mut self, config: FeatureSamplingConfig) {
        let seed_changed = self.config.seed != config.seed;
        self.config = config;

        if seed_changed {
            self.rng = Xoshiro256PlusPlus::seed_from_u64(self.config.seed);
        }

        // Clear cache when config changes
        self.cached_samples = None;
        self.last_sample_iteration = None;
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &FeatureSamplingConfig {
        &self.config
    }

    /// Returns the current sampling strategy.
    pub fn strategy(&self) -> SamplingStrategy {
        self.strategy
    }

    /// Sets the sampling strategy.
    pub fn set_strategy(&mut self, strategy: SamplingStrategy) {
        self.strategy = strategy;
        // Clear cache when strategy changes
        self.cached_samples = None;
        self.last_sample_iteration = None;
    }

    /// Validates that sampled features are within valid range.
    pub fn validate_features(&self, features: &[FeatureIndex], num_features: usize) -> bool {
        if features.is_empty() {
            return false;
        }

        // Check all features are within range
        for &feature in features {
            if feature >= num_features {
                return false;
            }
        }

        // Check for duplicates
        let mut sorted_features = features.to_vec();
        sorted_features.sort_unstable();

        for window in sorted_features.windows(2) {
            if window[0] == window[1] {
                return false; // Duplicate found
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_sampler_creation() {
        let config = FeatureSamplingConfig::default();
        let sampler = FeatureSampler::new(config);
        assert_eq!(sampler.config.feature_fraction, 1.0);
        assert_eq!(sampler.strategy, SamplingStrategy::Uniform);
    }

    #[test]
    fn test_uniform_sampling() {
        let config = FeatureSamplingConfig {
            feature_fraction: 0.5,
            seed: 42,
            ..Default::default()
        };
        let mut sampler = FeatureSampler::new(config);

        let features = sampler.sample_features(10, 0).unwrap();
        assert_eq!(features.len(), 5); // 50% of 10 features
        assert!(sampler.validate_features(&features, 10));
    }

    #[test]
    fn test_deterministic_sampling() {
        let config = FeatureSamplingConfig {
            feature_fraction: 0.5,
            seed: 42,
            deterministic: true,
            ..Default::default()
        };
        let mut sampler = FeatureSampler::new(config);

        let features1 = sampler.sample_features(10, 0).unwrap();
        let features2 = sampler.sample_features(10, 0).unwrap();

        assert_eq!(features1, features2); // Should be identical
    }

    #[test]
    fn test_importance_based_sampling() {
        let config = FeatureSamplingConfig {
            feature_fraction: 0.5,
            seed: 42,
            ..Default::default()
        };
        let mut sampler = FeatureSampler::with_strategy(config, SamplingStrategy::Importance);

        // Set importance scores (higher importance for later features)
        let importance = (0..10).map(|i| i as f64).collect();
        sampler.update_feature_importance(importance);

        let features = sampler.sample_features(10, 0).unwrap();
        assert_eq!(features.len(), 5);
        assert!(sampler.validate_features(&features, 10));

        // Higher-indexed features should be more likely to be selected
        // (though this is probabilistic, so we won't assert specific values)
    }

    #[test]
    fn test_stratified_sampling() {
        let config = FeatureSamplingConfig {
            feature_fraction: 0.6,
            seed: 42,
            ..Default::default()
        };
        let mut sampler = FeatureSampler::with_strategy(config, SamplingStrategy::Stratified);

        let features = sampler.sample_features(10, 0).unwrap();
        assert_eq!(features.len(), 6); // 60% of 10 features
        assert!(sampler.validate_features(&features, 10));
    }

    #[test]
    fn test_systematic_sampling() {
        let config = FeatureSamplingConfig {
            feature_fraction: 0.4,
            seed: 42,
            ..Default::default()
        };
        let mut sampler = FeatureSampler::with_strategy(config, SamplingStrategy::Systematic);

        let features = sampler.sample_features(10, 0).unwrap();
        assert_eq!(features.len(), 4); // 40% of 10 features
        assert!(sampler.validate_features(&features, 10));
    }

    #[test]
    fn test_sample_size_calculation() {
        let mut config = FeatureSamplingConfig::default();
        config.min_features = 2;
        config.max_features = Some(8);

        let sampler = FeatureSampler::new(config);

        // Test normal case
        assert_eq!(sampler.calculate_sample_size(10, 0.5).unwrap(), 5);

        // Test minimum constraint
        assert_eq!(sampler.calculate_sample_size(10, 0.1).unwrap(), 2);

        // Test maximum constraint
        assert_eq!(sampler.calculate_sample_size(10, 0.9).unwrap(), 8);

        // Test edge case
        assert_eq!(sampler.calculate_sample_size(10, 1.0).unwrap(), 8);
    }

    #[test]
    fn test_feature_validation() {
        let config = FeatureSamplingConfig::default();
        let sampler = FeatureSampler::new(config);

        // Valid features
        assert!(sampler.validate_features(&[0, 2, 4, 6], 10));

        // Empty features (invalid)
        assert!(!sampler.validate_features(&[], 10));

        // Out of range feature (invalid)
        assert!(!sampler.validate_features(&[0, 2, 10], 10));

        // Duplicate features (invalid)
        assert!(!sampler.validate_features(&[0, 2, 2, 4], 10));
    }

    #[test]
    fn test_caching_behavior() {
        let config = FeatureSamplingConfig {
            feature_fraction: 0.5,
            sample_per_node: false, // Enable caching
            seed: 42,
            ..Default::default()
        };
        let mut sampler = FeatureSampler::new(config);

        let features1 = sampler.sample_features(10, 0).unwrap();
        let features2 = sampler.sample_features(10, 0).unwrap(); // Same iteration

        assert_eq!(features1, features2); // Should use cached result

        let features3 = sampler.sample_features(10, 1).unwrap(); // Different iteration
                                                                 // features3 might be different due to new sampling
    }

    #[test]
    fn test_node_depth_adjustment() {
        let config = FeatureSamplingConfig {
            feature_fraction_bynode: 0.8,
            seed: 42,
            ..Default::default()
        };
        let mut sampler = FeatureSampler::new(config);

        let features_depth_0 = sampler.sample_features_for_node(10, 0, 0).unwrap();
        let features_depth_5 = sampler.sample_features_for_node(10, 5, 1).unwrap();

        // Deeper nodes should sample fewer features due to depth adjustment
        assert!(features_depth_5.len() <= features_depth_0.len());
    }

    #[test]
    fn test_config_update() {
        let config = FeatureSamplingConfig {
            feature_fraction: 0.5,
            seed: 42,
            ..Default::default()
        };
        let mut sampler = FeatureSampler::new(config);

        let features1 = sampler.sample_features(10, 0).unwrap();

        // Update config
        let new_config = FeatureSamplingConfig {
            feature_fraction: 0.3,
            seed: 123,
            ..Default::default()
        };
        sampler.update_config(new_config);

        let features2 = sampler.sample_features(10, 0).unwrap();

        assert_eq!(features2.len(), 3); // New fraction applied
                                        // Results should be different due to new seed
    }
}
