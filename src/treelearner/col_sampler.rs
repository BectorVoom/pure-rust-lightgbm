//! Column sampler for feature selection in tree learning.
//!
//! This module provides column sampling functionality for LightGBM's tree learning
//! process, implementing both per-tree and per-node feature selection with support
//! for interaction constraints.

use crate::config::Config;
use crate::dataset::Dataset;
use crate::core::error::{LightGBMError, Result};
use crate::io::tree::Tree;
use std::collections::HashSet;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand::rngs::StdRng;

/// Random number generator wrapper for reproducible sampling
pub struct Random {
    rng: StdRng,
}

impl Random {
    /// Create a new Random instance with the given seed
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Sample k indices from the range [0, n) without replacement
    pub fn sample(&mut self, n: i32, k: i32) -> Vec<i32> {
        if k <= 0 {
            return Vec::new();
        }
        if k >= n {
            return (0..n).collect();
        }

        let mut indices: Vec<i32> = (0..n).collect();
        indices.partial_shuffle(&mut self.rng, k as usize);
        indices.into_iter().take(k as usize).collect()
    }
}

/// Column sampler for feature selection in tree learning
pub struct ColSampler {
    /// Training dataset reference
    train_data: Option<*const Dataset>,
    /// Fraction of features to use per tree
    fraction_bytree: f64,
    /// Fraction of features to use per node
    fraction_bynode: f64,
    /// Whether to reset feature sampling per tree
    need_reset_bytree: bool,
    /// Number of features to use per tree
    used_cnt_bytree: i32,
    /// Random seed for feature sampling
    seed: u64,
    /// Random number generator
    random: Random,
    /// Feature usage flags (indexed by inner feature index)
    is_feature_used: Vec<i8>,
    /// Indices of features used in current tree
    used_feature_indices: Vec<i32>,
    /// Valid feature indices from dataset
    valid_feature_indices: Vec<i32>,
    /// Interaction constraints (sets of features that can interact)
    interaction_constraints: Vec<HashSet<i32>>,
}

impl ColSampler {
    /// Create a new ColSampler from configuration
    pub fn new(config: &Config) -> Self {
        let interaction_constraints = Vec::new();
        // Note: interaction_constraints_vector equivalent would need to be added to Config
        // For now, leaving empty as in the C++ default constructor
        
        Self {
            train_data: None,
            fraction_bytree: config.feature_fraction,
            fraction_bynode: config.feature_fraction_bynode,
            need_reset_bytree: true,
            used_cnt_bytree: 0,
            seed: config.feature_fraction_seed,
            random: Random::new(config.feature_fraction_seed),
            is_feature_used: Vec::new(),
            used_feature_indices: Vec::new(),
            valid_feature_indices: Vec::new(),
            interaction_constraints,
        }
    }

    /// Calculate the number of features to use given total count and fraction
    pub fn get_cnt(total_cnt: usize, fraction: f64) -> i32 {
        let min_val = std::cmp::min(1, total_cnt as i32);
        let used_feature_cnt = (total_cnt as f64 * fraction).round() as i32;
        std::cmp::max(used_feature_cnt, min_val)
    }

    /// Set the training data for the sampler
    pub fn set_training_data(&mut self, train_data: &Dataset) -> Result<()> {
        self.train_data = Some(train_data as *const Dataset);
        self.is_feature_used.resize(train_data.num_features(), 1);
        self.valid_feature_indices = train_data.valid_feature_indices();

        if self.fraction_bytree >= 1.0 {
            self.need_reset_bytree = false;
            self.used_cnt_bytree = self.valid_feature_indices.len() as i32;
        } else {
            self.need_reset_bytree = true;
            self.used_cnt_bytree = Self::get_cnt(self.valid_feature_indices.len(), self.fraction_bytree);
        }

        self.reset_by_tree()?;
        Ok(())
    }

    /// Update configuration
    pub fn set_config(&mut self, config: &Config) -> Result<()> {
        self.fraction_bytree = config.feature_fraction;
        self.fraction_bynode = config.feature_fraction_bynode;

        if let Some(train_data) = self.get_train_data() {
            self.is_feature_used.resize(train_data.num_features(), 1);
        }

        // Update seed if changed
        if self.seed != config.feature_fraction_seed {
            self.seed = config.feature_fraction_seed;
            self.random = Random::new(self.seed);
        }

        if self.fraction_bytree >= 1.0 {
            self.need_reset_bytree = false;
            self.used_cnt_bytree = self.valid_feature_indices.len() as i32;
        } else {
            self.need_reset_bytree = true;
            self.used_cnt_bytree = Self::get_cnt(self.valid_feature_indices.len(), self.fraction_bytree);
        }

        self.reset_by_tree()?;
        Ok(())
    }

    /// Reset feature sampling for a new tree
    pub fn reset_by_tree(&mut self) -> Result<()> {
        if self.need_reset_bytree {
            // Clear all feature usage flags
            self.is_feature_used.fill(0);

            // Sample features for this tree
            self.used_feature_indices = self.random.sample(
                self.valid_feature_indices.len() as i32,
                self.used_cnt_bytree
            );

            // Get train data info we need
            let train_data_ptr = self.train_data
                .ok_or_else(|| LightGBMError::dataset("Training data not set"))?;
            
            // Set selected features as used
            for &idx in &self.used_feature_indices {
                let used_feature = self.valid_feature_indices[idx as usize];
                let inner_feature_index = unsafe { (*train_data_ptr).inner_feature_index(used_feature) };
                if inner_feature_index >= 0 {
                    self.is_feature_used[inner_feature_index as usize] = 1;
                }
            }
        }
        Ok(())
    }

    /// Get feature usage mask for a specific node
    pub fn get_by_node(&mut self, tree: Option<&Tree>, leaf: i32) -> Result<Vec<i8>> {
        let train_data_ptr = self.train_data
            .ok_or_else(|| LightGBMError::dataset("Training data not set"))?;
        
        let num_features = unsafe { (*train_data_ptr).num_features() };

        // Get interaction constraints for current branch
        let mut allowed_features = HashSet::new();
        if !self.interaction_constraints.is_empty() {
            let branch_features = if let Some(tree) = tree {
                tree.branch_features(leaf)
            } else {
                Vec::new()
            };
            
            allowed_features.extend(branch_features.iter());

            for constraint in &self.interaction_constraints {
                let mut num_feat_found = 0;
                if branch_features.is_empty() {
                    allowed_features.extend(constraint.iter());
                }
                for &feat in &branch_features {
                    if !constraint.contains(&feat) {
                        break;
                    }
                    num_feat_found += 1;
                    if num_feat_found == branch_features.len() {
                        allowed_features.extend(constraint.iter());
                        break;
                    }
                }
            }
        }

        let mut ret = vec![0i8; num_features];

        if self.fraction_bynode >= 1.0 {
            if self.interaction_constraints.is_empty() {
                return Ok(vec![1i8; num_features]);
            } else {
                for &feat in &allowed_features {
                    let inner_feat = unsafe { (*train_data_ptr).inner_feature_index(feat) };
                    if inner_feat >= 0 {
                        ret[inner_feat as usize] = 1;
                    }
                }
                return Ok(ret);
            }
        }

        if self.need_reset_bytree {
            let mut used_feature_cnt = Self::get_cnt(self.used_feature_indices.len(), self.fraction_bynode);
            let allowed_used_feature_indices: &Vec<i32>;
            let filtered_feature_indices: Vec<i32>;

            if self.interaction_constraints.is_empty() {
                allowed_used_feature_indices = &self.used_feature_indices;
            } else {
                filtered_feature_indices = self.used_feature_indices
                    .iter()
                    .filter(|&&feat_ind| allowed_features.contains(&self.valid_feature_indices[feat_ind as usize]))
                    .copied()
                    .collect();
                used_feature_cnt = std::cmp::min(used_feature_cnt, filtered_feature_indices.len() as i32);
                allowed_used_feature_indices = &filtered_feature_indices;
            }

            let sampled_indices = self.random.sample(
                allowed_used_feature_indices.len() as i32,
                used_feature_cnt
            );

            for &i in &sampled_indices {
                let used_feature = self.valid_feature_indices[allowed_used_feature_indices[i as usize] as usize];
                let inner_feature_index = unsafe { (*train_data_ptr).inner_feature_index(used_feature) };
                if inner_feature_index >= 0 {
                    ret[inner_feature_index as usize] = 1;
                }
            }
        } else {
            let mut used_feature_cnt = Self::get_cnt(self.valid_feature_indices.len(), self.fraction_bynode);
            let allowed_valid_feature_indices: &Vec<i32>;
            let filtered_feature_indices: Vec<i32>;

            if self.interaction_constraints.is_empty() {
                allowed_valid_feature_indices = &self.valid_feature_indices;
            } else {
                filtered_feature_indices = self.valid_feature_indices
                    .iter()
                    .filter(|&&feat| allowed_features.contains(&feat))
                    .copied()
                    .collect();
                allowed_valid_feature_indices = &filtered_feature_indices;
                used_feature_cnt = std::cmp::min(used_feature_cnt, filtered_feature_indices.len() as i32);
            }

            let sampled_indices = self.random.sample(
                allowed_valid_feature_indices.len() as i32,
                used_feature_cnt
            );

            for &i in &sampled_indices {
                let used_feature = allowed_valid_feature_indices[i as usize];
                let inner_feature_index = unsafe { (*train_data_ptr).inner_feature_index(used_feature) };
                if inner_feature_index >= 0 {
                    ret[inner_feature_index as usize] = 1;
                }
            }
        }

        Ok(ret)
    }

    /// Get feature usage flags for tree-level sampling
    pub fn is_feature_used_bytree(&self) -> &Vec<i8> {
        &self.is_feature_used
    }

    /// Set feature usage flag for a specific feature
    pub fn set_is_feature_used_by_tree(&mut self, fid: usize, val: bool) {
        if fid < self.is_feature_used.len() {
            self.is_feature_used[fid] = if val { 1 } else { 0 };
        }
    }

    /// Get training data reference (unsafe, for internal use only)
    fn get_train_data(&self) -> Option<&Dataset> {
        self.train_data.map(|ptr| unsafe { &*ptr })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ConfigBuilder;

    #[test]
    fn test_get_cnt() {
        assert_eq!(ColSampler::get_cnt(100, 0.5), 50);
        assert_eq!(ColSampler::get_cnt(100, 1.0), 100);
        assert_eq!(ColSampler::get_cnt(0, 0.5), 0);
        assert_eq!(ColSampler::get_cnt(1, 0.5), 1);
    }

    #[test]
    fn test_random_sampling() {
        let mut random = Random::new(42);
        let samples = random.sample(10, 5);
        assert_eq!(samples.len(), 5);

        // Check all samples are in valid range
        for &sample in &samples {
            assert!(sample >= 0 && sample < 10);
        }

        // Check no duplicates
        let mut sorted_samples = samples.clone();
        sorted_samples.sort();
        sorted_samples.dedup();
        assert_eq!(sorted_samples.len(), samples.len());
    }

    #[test]
    fn test_col_sampler_creation() {
        let config = ConfigBuilder::new().build().unwrap();
        let sampler = ColSampler::new(&config);
        assert_eq!(sampler.fraction_bytree, config.feature_fraction);
        assert_eq!(sampler.fraction_bynode, config.feature_fraction_bynode);
        assert_eq!(sampler.seed, config.feature_fraction_seed);
    }
}