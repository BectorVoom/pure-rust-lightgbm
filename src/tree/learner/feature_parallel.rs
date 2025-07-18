//! Feature-parallel tree learner implementation for the Pure Rust LightGBM framework.
//!
//! This module provides a feature-parallel tree learning algorithm that distributes
//! histogram construction and split finding across features rather than data points.

use crate::core::types::{DataSize, FeatureIndex, NodeIndex, Score};
use crate::tree::histogram::{
    HistogramBuilder, HistogramBuilderConfig, HistogramPool, HistogramPoolConfig,
    BinMapper, FeatureType,
};
use crate::tree::learner::serial::{Dataset, SerialTreeLearnerConfig};
use crate::tree::node::TreeNode;
use crate::tree::sampling::{FeatureSampler, FeatureSamplingConfig, SamplingStrategy};
use crate::tree::split::{
    ConstraintManager, SplitEvaluator, SplitEvaluatorConfig, SplitFinder, 
    SplitFinderConfig, SplitInfo,
};
use crate::tree::tree::Tree;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};

/// Configuration for the feature-parallel tree learner.
#[derive(Debug, Clone)]
pub struct FeatureParallelTreeLearnerConfig {
    /// Base configuration from serial learner
    pub base_config: SerialTreeLearnerConfig,
    /// Number of threads for feature parallelism
    pub num_feature_threads: usize,
    /// Number of features per thread
    pub features_per_thread: usize,
    /// Whether to use histogram communication optimization
    pub use_histogram_communication: bool,
    /// Communication buffer size for histogram aggregation
    pub communication_buffer_size: usize,
    /// Feature grouping strategy
    pub feature_grouping: FeatureGroupingStrategy,
}

impl Default for FeatureParallelTreeLearnerConfig {
    fn default() -> Self {
        FeatureParallelTreeLearnerConfig {
            base_config: SerialTreeLearnerConfig::default(),
            num_feature_threads: num_cpus::get(),
            features_per_thread: 10,
            use_histogram_communication: true,
            communication_buffer_size: 1024,
            feature_grouping: FeatureGroupingStrategy::RoundRobin,
        }
    }
}

/// Strategies for grouping features across threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureGroupingStrategy {
    /// Round-robin assignment of features to threads
    RoundRobin,
    /// Block assignment of consecutive features to threads
    Block,
    /// Load-balanced assignment based on feature complexity
    LoadBalanced,
    /// Random assignment of features to threads
    Random,
}

/// Information about a feature group assigned to a thread.
#[derive(Debug, Clone)]
struct FeatureGroup {
    /// Thread ID
    thread_id: usize,
    /// Feature indices assigned to this group
    feature_indices: Vec<FeatureIndex>,
    /// Estimated computational cost for load balancing
    estimated_cost: f64,
}

impl FeatureGroup {
    fn new(thread_id: usize, feature_indices: Vec<FeatureIndex>) -> Self {
        let estimated_cost = feature_indices.len() as f64;
        FeatureGroup {
            thread_id,
            feature_indices,
            estimated_cost,
        }
    }

    fn with_cost(thread_id: usize, feature_indices: Vec<FeatureIndex>, estimated_cost: f64) -> Self {
        FeatureGroup {
            thread_id,
            feature_indices,
            estimated_cost,
        }
    }
}

/// Node information for feature-parallel processing.
#[derive(Debug, Clone)]
struct FeatureParallelNodeInfo {
    /// Node index in the tree
    node_index: NodeIndex,
    /// Data indices belonging to this node
    data_indices: Vec<DataSize>,
    /// Node depth
    depth: usize,
    /// Sum of gradients in this node
    sum_gradients: f64,
    /// Sum of hessians in this node
    sum_hessians: f64,
    /// Features used in the path to this node
    path_features: Vec<FeatureIndex>,
}

impl FeatureParallelNodeInfo {
    fn new(
        node_index: NodeIndex,
        data_indices: Vec<DataSize>,
        depth: usize,
        sum_gradients: f64,
        sum_hessians: f64,
    ) -> Self {
        FeatureParallelNodeInfo {
            node_index,
            data_indices,
            depth,
            sum_gradients,
            sum_hessians,
            path_features: Vec::new(),
        }
    }
}

/// Communication buffer for histogram aggregation.
#[derive(Debug, Clone)]
struct HistogramCommunicationBuffer {
    /// Buffers for each thread
    thread_buffers: Vec<Vec<Array1<f64>>>,
    /// Global aggregated histograms
    global_histograms: Vec<Array1<f64>>,
    /// Feature mapping for communication
    feature_to_thread: HashMap<FeatureIndex, usize>,
}

impl HistogramCommunicationBuffer {
    fn new(num_threads: usize, feature_groups: &[FeatureGroup]) -> Self {
        let mut feature_to_thread = HashMap::new();
        
        for group in feature_groups {
            for &feature_idx in &group.feature_indices {
                feature_to_thread.insert(feature_idx, group.thread_id);
            }
        }

        HistogramCommunicationBuffer {
            thread_buffers: vec![Vec::new(); num_threads],
            global_histograms: Vec::new(),
            feature_to_thread,
        }
    }

    fn add_histogram(&mut self, thread_id: usize, histogram: Array1<f64>) {
        if thread_id < self.thread_buffers.len() {
            self.thread_buffers[thread_id].push(histogram);
        }
    }

    fn aggregate_histograms(&mut self) {
        self.global_histograms.clear();
        
        let max_histograms = self.thread_buffers.iter()
            .map(|buffers| buffers.len())
            .max()
            .unwrap_or(0);

        for hist_idx in 0..max_histograms {
            let mut aggregated = None;
            
            for thread_buffers in &self.thread_buffers {
                if let Some(histogram) = thread_buffers.get(hist_idx) {
                    match &aggregated {
                        None => aggregated = Some(histogram.clone()),
                        Some(agg) => {
                            // Element-wise addition
                            if let Some(mut agg_hist) = aggregated.take() {
                                for (agg_val, &hist_val) in agg_hist.iter_mut().zip(histogram.iter()) {
                                    *agg_val += hist_val;
                                }
                                aggregated = Some(agg_hist);
                            }
                        }
                    }
                }
            }
            
            if let Some(agg_hist) = aggregated {
                self.global_histograms.push(agg_hist);
            }
        }
    }

    fn clear(&mut self) {
        for buffer in &mut self.thread_buffers {
            buffer.clear();
        }
        self.global_histograms.clear();
    }
}

/// Feature-parallel tree learner implementing feature-level parallelism.
pub struct FeatureParallelTreeLearner {
    config: FeatureParallelTreeLearnerConfig,
    feature_sampler: FeatureSampler,
    histogram_builder: Arc<HistogramBuilder>,
    split_finder: SplitFinder,
    split_evaluator: SplitEvaluator,
    constraint_manager: ConstraintManager,
    feature_groups: Vec<FeatureGroup>,
    communication_buffer: HistogramCommunicationBuffer,
}

impl FeatureParallelTreeLearner {
    /// Creates a new feature-parallel tree learner with the given configuration.
    pub fn new(config: FeatureParallelTreeLearnerConfig) -> anyhow::Result<Self> {
        // Create feature sampler
        let feature_sampler = FeatureSampler::with_strategy(
            config.base_config.feature_sampling.clone(),
            SamplingStrategy::Uniform,
        );

        // Create histogram pool and builder
        let histogram_pool_config = HistogramPoolConfig {
            max_bin: config.base_config.max_bin,
            num_features: 0,
            max_pool_size: config.num_feature_threads * 10,
            initial_pool_size: config.num_feature_threads * 2,
            use_double_precision: true,
        };
        let histogram_pool = HistogramPool::new(histogram_pool_config);
        let histogram_builder = Arc::new(HistogramBuilder::new(
            config.base_config.histogram_config.clone(),
            histogram_pool,
        ));

        // Create split finder
        let split_finder_config = SplitFinderConfig {
            min_data_in_leaf: config.base_config.min_data_in_leaf,
            min_sum_hessian_in_leaf: config.base_config.min_sum_hessian_in_leaf,
            lambda_l1: config.base_config.lambda_l1,
            lambda_l2: config.base_config.lambda_l2,
            min_split_gain: config.base_config.min_split_gain,
            max_bin: config.base_config.max_bin,
        };
        let split_finder = SplitFinder::new(split_finder_config);

        // Create split evaluator and constraint manager
        let split_evaluator = SplitEvaluator::new(SplitEvaluatorConfig::default());
        let constraint_manager = ConstraintManager::new();

        // Initialize empty feature groups (will be set when we know the dataset)
        let feature_groups = Vec::new();
        let communication_buffer = HistogramCommunicationBuffer::new(0, &feature_groups);

        Ok(FeatureParallelTreeLearner {
            config,
            feature_sampler,
            histogram_builder,
            split_finder,
            split_evaluator,
            constraint_manager,
            feature_groups,
            communication_buffer,
        })
    }

    /// Trains a decision tree using feature-parallel processing.
    pub fn train(
        &mut self,
        dataset: &Dataset,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
        _iteration: usize,
    ) -> anyhow::Result<Tree> {
        if dataset.num_data == 0 {
            return Err(anyhow::anyhow!("Dataset is empty"));
        }

        if gradients.len() != dataset.num_data as usize || 
           hessians.len() != dataset.num_data as usize {
            return Err(anyhow::anyhow!("Gradient and hessian arrays must match dataset size"));
        }

        // Initialize feature groups for this dataset
        self.initialize_feature_groups(dataset.num_features)?;

        // Initialize tree with root node
        let mut tree = Tree::with_capacity(
            self.config.base_config.max_leaves, 
            self.config.base_config.learning_rate
        );

        // Initialize root node statistics
        let total_gradients: f64 = gradients.iter().map(|&g| g as f64).sum();
        let total_hessians: f64 = hessians.iter().map(|&h| h as f64).sum();
        let all_data_indices: Vec<DataSize> = (0..dataset.num_data).collect();

        // Update root node
        if let Some(root_node) = tree.node_mut(0) {
            root_node.update_statistics(total_gradients, total_hessians, dataset.num_data);
            let root_output = root_node.calculate_leaf_output(
                self.config.base_config.lambda_l1,
                self.config.base_config.lambda_l2,
            );
            root_node.set_leaf_output(root_output);
        }

        // Initialize queue with root node
        let mut node_queue = VecDeque::new();
        node_queue.push_back(FeatureParallelNodeInfo::new(
            0,
            all_data_indices,
            0,
            total_gradients,
            total_hessians,
        ));

        // Build tree iteratively
        while let Some(node_info) = node_queue.pop_front() {
            if tree.num_leaves() >= self.config.base_config.max_leaves {
                break;
            }

            if node_info.depth >= self.config.base_config.max_depth {
                continue;
            }

            if let Some(split_result) = self.find_best_split_feature_parallel(
                dataset,
                gradients,
                hessians,
                &node_info,
            )? {
                // Apply the split
                let (left_child_info, right_child_info) = self.apply_split(
                    &mut tree,
                    dataset,
                    gradients,
                    hessians,
                    &node_info,
                    &split_result,
                )?;

                // Add children to queue
                node_queue.push_back(left_child_info);
                node_queue.push_back(right_child_info);
            }
        }

        // Finalize tree outputs
        self.finalize_tree_outputs(&mut tree)?;

        Ok(tree)
    }

    /// Initializes feature groups based on the grouping strategy.
    fn initialize_feature_groups(&mut self, num_features: usize) -> anyhow::Result<()> {
        if num_features == 0 {
            return Ok(());
        }

        self.feature_groups.clear();
        
        match self.config.feature_grouping {
            FeatureGroupingStrategy::RoundRobin => {
                self.create_round_robin_groups(num_features)?;
            }
            FeatureGroupingStrategy::Block => {
                self.create_block_groups(num_features)?;
            }
            FeatureGroupingStrategy::LoadBalanced => {
                self.create_load_balanced_groups(num_features)?;
            }
            FeatureGroupingStrategy::Random => {
                self.create_random_groups(num_features)?;
            }
        }

        // Update communication buffer
        self.communication_buffer = HistogramCommunicationBuffer::new(
            self.config.num_feature_threads,
            &self.feature_groups,
        );

        Ok(())
    }

    /// Creates round-robin feature groups.
    fn create_round_robin_groups(&mut self, num_features: usize) -> anyhow::Result<()> {
        let mut thread_features: Vec<Vec<FeatureIndex>> = 
            vec![Vec::new(); self.config.num_feature_threads];

        for feature_idx in 0..num_features {
            let thread_id = feature_idx % self.config.num_feature_threads;
            thread_features[thread_id].push(feature_idx);
        }

        for (thread_id, features) in thread_features.into_iter().enumerate() {
            if !features.is_empty() {
                self.feature_groups.push(FeatureGroup::new(thread_id, features));
            }
        }

        Ok(())
    }

    /// Creates block feature groups.
    fn create_block_groups(&mut self, num_features: usize) -> anyhow::Result<()> {
        let features_per_thread = (num_features + self.config.num_feature_threads - 1) 
            / self.config.num_feature_threads;

        for thread_id in 0..self.config.num_feature_threads {
            let start_idx = thread_id * features_per_thread;
            let end_idx = ((thread_id + 1) * features_per_thread).min(num_features);
            
            if start_idx < num_features {
                let features: Vec<FeatureIndex> = (start_idx..end_idx).collect();
                self.feature_groups.push(FeatureGroup::new(thread_id, features));
            }
        }

        Ok(())
    }

    /// Creates load-balanced feature groups.
    fn create_load_balanced_groups(&mut self, num_features: usize) -> anyhow::Result<()> {
        // For now, use block grouping as load balancing requires feature cost estimation
        // In a real implementation, this would analyze feature complexity
        self.create_block_groups(num_features)
    }

    /// Creates random feature groups.
    fn create_random_groups(&mut self, num_features: usize) -> anyhow::Result<()> {
        use rand::prelude::*;
        let mut rng = rand::thread_rng();
        
        let mut features: Vec<FeatureIndex> = (0..num_features).collect();
        features.shuffle(&mut rng);

        let features_per_thread = (num_features + self.config.num_feature_threads - 1) 
            / self.config.num_feature_threads;

        for thread_id in 0..self.config.num_feature_threads {
            let start_idx = thread_id * features_per_thread;
            let end_idx = ((thread_id + 1) * features_per_thread).min(num_features);
            
            if start_idx < features.len() {
                let thread_features = features[start_idx..end_idx.min(features.len())].to_vec();
                self.feature_groups.push(FeatureGroup::new(thread_id, thread_features));
            }
        }

        Ok(())
    }

    /// Finds the best split using feature-parallel processing.
    fn find_best_split_feature_parallel(
        &mut self,
        dataset: &Dataset,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
        node_info: &FeatureParallelNodeInfo,
    ) -> anyhow::Result<Option<SplitInfo>> {
        // Check if node can be split
        if node_info.data_indices.len() < (2 * self.config.base_config.min_data_in_leaf) as usize {
            return Ok(None);
        }

        if node_info.sum_hessians < 2.0 * self.config.base_config.min_sum_hessian_in_leaf {
            return Ok(None);
        }

        // Sample features
        let sampled_features = self.feature_sampler.sample_features(dataset.num_features, 0)?;
        if sampled_features.is_empty() {
            return Ok(None);
        }

        // Filter features based on constraints
        let allowed_features = self.constraint_manager.filter_candidate_features(
            &sampled_features,
            &node_info.path_features,
        );

        if allowed_features.is_empty() {
            return Ok(None);
        }

        // Group allowed features by thread assignment
        let mut thread_feature_groups: HashMap<usize, Vec<FeatureIndex>> = HashMap::new();
        
        for &feature_idx in &allowed_features {
            if let Some(&thread_id) = self.communication_buffer.feature_to_thread.get(&feature_idx) {
                thread_feature_groups.entry(thread_id).or_insert_with(Vec::new).push(feature_idx);
            }
        }

        // Clear communication buffer
        self.communication_buffer.clear();

        // Construct histograms in parallel by feature groups
        let split_candidates: Vec<Option<SplitInfo>> = thread_feature_groups
            .par_iter()
            .map(|(&thread_id, features)| {
                self.construct_histograms_and_find_splits(
                    dataset,
                    gradients,
                    hessians,
                    node_info,
                    features,
                    thread_id,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Find the best split among all candidates
        Ok(split_candidates
            .into_iter()
            .flatten()
            .max_by(|a, b| a.gain.partial_cmp(&b.gain).unwrap_or(std::cmp::Ordering::Equal)))
    }

    /// Constructs histograms and finds splits for a group of features.
    fn construct_histograms_and_find_splits(
        &self,
        dataset: &Dataset,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
        node_info: &FeatureParallelNodeInfo,
        feature_indices: &[FeatureIndex],
        _thread_id: usize,
    ) -> anyhow::Result<Option<SplitInfo>> {
        let mut best_split: Option<SplitInfo> = None;

        for &feature_idx in feature_indices {
            // Construct histogram for this feature
            let feature_column = dataset.features.column(feature_idx);
            let bin_mapper = dataset.bin_mapper(feature_idx).unwrap();

            let histogram = self.histogram_builder.construct_feature_histogram(
                &feature_column,
                gradients,
                hessians,
                &node_info.data_indices,
                bin_mapper,
            )?;

            // Find best split for this feature
            let split_candidate = self.split_finder.find_best_split_for_feature(
                feature_idx,
                &histogram.view(),
                node_info.sum_gradients,
                node_info.sum_hessians,
                node_info.data_indices.len() as DataSize,
                bin_mapper.bin_upper_bounds(),
            );

            if let Some(split) = split_candidate {
                // Validate split against constraints
                let validation_result = self.constraint_manager.validate_split(
                    &split,
                    node_info.depth,
                    0.0,
                    &node_info.path_features,
                );

                if validation_result.is_valid() {
                    match &best_split {
                        None => best_split = Some(split),
                        Some(current_best) => {
                            if split.gain > current_best.gain {
                                best_split = Some(split);
                            }
                        }
                    }
                }
            }
        }

        Ok(best_split)
    }

    /// Applies a split to the tree and returns information about the child nodes.
    fn apply_split(
        &self,
        tree: &mut Tree,
        dataset: &Dataset,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
        parent_info: &FeatureParallelNodeInfo,
        split: &SplitInfo,
    ) -> anyhow::Result<(FeatureParallelNodeInfo, FeatureParallelNodeInfo)> {
        // Partition data based on split
        let (left_indices, right_indices) = self.partition_data(
            dataset,
            &parent_info.data_indices,
            split.feature,
            split.threshold_value,
        )?;

        // Calculate child node statistics
        let left_sum_gradients = left_indices.iter()
            .map(|&idx| gradients[idx as usize] as f64)
            .sum();
        let left_sum_hessians = left_indices.iter()
            .map(|&idx| hessians[idx as usize] as f64)
            .sum();

        let right_sum_gradients = right_indices.iter()
            .map(|&idx| gradients[idx as usize] as f64)
            .sum();
        let right_sum_hessians = right_indices.iter()
            .map(|&idx| hessians[idx as usize] as f64)
            .sum();

        // Create child nodes in tree
        let (left_child_idx, right_child_idx) = tree.split_node(
            parent_info.node_index,
            split.feature,
            split.threshold_value,
            split.threshold_bin,
            split.gain,
            left_sum_gradients,
            left_sum_hessians,
            left_indices.len() as DataSize,
            right_sum_gradients,
            right_sum_hessians,
            right_indices.len() as DataSize,
            split.default_left,
        )?;

        // Create child node information
        let mut left_path_features = parent_info.path_features.clone();
        left_path_features.push(split.feature);

        let mut right_path_features = parent_info.path_features.clone();
        right_path_features.push(split.feature);

        let left_child_info = FeatureParallelNodeInfo {
            node_index: left_child_idx,
            data_indices: left_indices,
            depth: parent_info.depth + 1,
            sum_gradients: left_sum_gradients,
            sum_hessians: left_sum_hessians,
            path_features: left_path_features,
        };

        let right_child_info = FeatureParallelNodeInfo {
            node_index: right_child_idx,
            data_indices: right_indices,
            depth: parent_info.depth + 1,
            sum_gradients: right_sum_gradients,
            sum_hessians: right_sum_hessians,
            path_features: right_path_features,
        };

        Ok((left_child_info, right_child_info))
    }

    /// Partitions data indices based on a split condition.
    fn partition_data(
        &self,
        dataset: &Dataset,
        data_indices: &[DataSize],
        split_feature: FeatureIndex,
        threshold: f64,
    ) -> anyhow::Result<(Vec<DataSize>, Vec<DataSize>)> {
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for &data_idx in data_indices {
            let feature_value = dataset.features[[data_idx as usize, split_feature]];
            
            if feature_value.is_nan() {
                left_indices.push(data_idx);
            } else if (feature_value as f64) <= threshold {
                left_indices.push(data_idx);
            } else {
                right_indices.push(data_idx);
            }
        }

        Ok((left_indices, right_indices))
    }

    /// Finalizes leaf outputs for all leaf nodes in the tree.
    fn finalize_tree_outputs(&self, tree: &mut Tree) -> anyhow::Result<()> {
        let leaf_indices = tree.leaf_indices();

        for leaf_idx in leaf_indices {
            if let Some(node) = tree.node_mut(leaf_idx) {
                if node.is_leaf() && node.leaf_output().is_none() {
                    let output = node.calculate_leaf_output(
                        self.config.base_config.lambda_l1,
                        self.config.base_config.lambda_l2,
                    );
                    node.set_leaf_output(output);
                }
            }
        }

        Ok(())
    }

    /// Returns information about the current feature groups.
    pub fn feature_groups(&self) -> &[FeatureGroup] {
        &self.feature_groups
    }

    /// Returns the number of feature threads.
    pub fn num_feature_threads(&self) -> usize {
        self.config.num_feature_threads
    }

    /// Updates the learner configuration.
    pub fn update_config(&mut self, config: FeatureParallelTreeLearnerConfig) {
        self.config = config.clone();
        self.feature_sampler.update_config(config.base_config.feature_sampling);
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &FeatureParallelTreeLearnerConfig {
        &self.config
    }

    /// Resets the learner state.
    pub fn reset(&mut self) {
        self.feature_sampler.reset();
        self.feature_groups.clear();
        self.communication_buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_feature_parallel_tree_learner_creation() {
        let config = FeatureParallelTreeLearnerConfig::default();
        let learner = FeatureParallelTreeLearner::new(config);
        assert!(learner.is_ok());
    }

    #[test]
    fn test_feature_grouping_strategies() {
        for strategy in [
            FeatureGroupingStrategy::RoundRobin,
            FeatureGroupingStrategy::Block,
            FeatureGroupingStrategy::LoadBalanced,
            FeatureGroupingStrategy::Random,
        ] {
            let config = FeatureParallelTreeLearnerConfig {
                feature_grouping: strategy,
                num_feature_threads: 3,
                ..Default::default()
            };
            
            let mut learner = FeatureParallelTreeLearner::new(config).unwrap();
            let result = learner.initialize_feature_groups(10);
            assert!(result.is_ok());
            
            // Check that features are distributed across threads
            let total_features: usize = learner.feature_groups()
                .iter()
                .map(|group| group.feature_indices.len())
                .sum();
            assert_eq!(total_features, 10);
        }
    }

    #[test]
    fn test_round_robin_grouping() {
        let config = FeatureParallelTreeLearnerConfig {
            feature_grouping: FeatureGroupingStrategy::RoundRobin,
            num_feature_threads: 3,
            ..Default::default()
        };
        
        let mut learner = FeatureParallelTreeLearner::new(config).unwrap();
        learner.initialize_feature_groups(7).unwrap();
        
        // Check round-robin distribution
        let groups = learner.feature_groups();
        assert_eq!(groups.len(), 3);
        
        // Thread 0 should have features [0, 3, 6]
        // Thread 1 should have features [1, 4]
        // Thread 2 should have features [2, 5]
        assert_eq!(groups[0].feature_indices, vec![0, 3, 6]);
        assert_eq!(groups[1].feature_indices, vec![1, 4]);
        assert_eq!(groups[2].feature_indices, vec![2, 5]);
    }

    #[test]
    fn test_block_grouping() {
        let config = FeatureParallelTreeLearnerConfig {
            feature_grouping: FeatureGroupingStrategy::Block,
            num_feature_threads: 3,
            ..Default::default()
        };
        
        let mut learner = FeatureParallelTreeLearner::new(config).unwrap();
        learner.initialize_feature_groups(9).unwrap();
        
        let groups = learner.feature_groups();
        assert_eq!(groups.len(), 3);
        
        // Each thread should get 3 consecutive features
        assert_eq!(groups[0].feature_indices, vec![0, 1, 2]);
        assert_eq!(groups[1].feature_indices, vec![3, 4, 5]);
        assert_eq!(groups[2].feature_indices, vec![6, 7, 8]);
    }

    #[test]
    fn test_feature_parallel_training() {
        let config = FeatureParallelTreeLearnerConfig {
            base_config: SerialTreeLearnerConfig {
                max_leaves: 3,
                max_depth: 2,
                min_data_in_leaf: 1,
                min_sum_hessian_in_leaf: 0.1,
                lambda_l1: 0.0,
                lambda_l2: 0.1,
                ..Default::default()
            },
            num_feature_threads: 2,
            ..Default::default()
        };

        let mut learner = FeatureParallelTreeLearner::new(config).unwrap();

        // Create simple dataset
        let features = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 2.0, 3.0,
                2.0, 3.0, 1.0,
                3.0, 1.0, 2.0,
                1.0, 3.0, 2.0,
                2.0, 1.0, 3.0,
                3.0, 2.0, 1.0,
            ],
        ).unwrap();
        let dataset = Dataset::new(features, 10).unwrap();

        // Create gradients and hessians
        let gradients = Array1::from(vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]);
        let hessians = Array1::from(vec![1.0; 6]);

        let result = learner.train(&dataset, &gradients.view(), &hessians.view(), 0);
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert!(tree.num_nodes() >= 1);
        assert!(tree.num_leaves() >= 1);
    }

    #[test]
    fn test_histogram_communication_buffer() {
        let feature_groups = vec![
            FeatureGroup::new(0, vec![0, 1]),
            FeatureGroup::new(1, vec![2, 3]),
        ];
        
        let mut buffer = HistogramCommunicationBuffer::new(2, &feature_groups);
        
        // Add histograms from different threads
        buffer.add_histogram(0, Array1::from(vec![1.0, 2.0, 3.0, 4.0]));
        buffer.add_histogram(1, Array1::from(vec![0.5, 1.0, 1.5, 2.0]));
        
        // Aggregate histograms
        buffer.aggregate_histograms();
        
        assert_eq!(buffer.global_histograms.len(), 1);
        assert_eq!(buffer.global_histograms[0], Array1::from(vec![1.5, 3.0, 4.5, 6.0]));
    }

    #[test]
    fn test_feature_group_creation() {
        let group = FeatureGroup::new(0, vec![1, 3, 5]);
        assert_eq!(group.thread_id, 0);
        assert_eq!(group.feature_indices, vec![1, 3, 5]);
        assert_eq!(group.estimated_cost, 3.0);
        
        let group_with_cost = FeatureGroup::with_cost(1, vec![2, 4], 10.5);
        assert_eq!(group_with_cost.thread_id, 1);
        assert_eq!(group_with_cost.estimated_cost, 10.5);
    }

    #[test]
    fn test_config_update() {
        let config = FeatureParallelTreeLearnerConfig::default();
        let mut learner = FeatureParallelTreeLearner::new(config).unwrap();

        let new_config = FeatureParallelTreeLearnerConfig {
            base_config: SerialTreeLearnerConfig {
                max_leaves: 15,
                lambda_l1: 0.1,
                ..Default::default()
            },
            num_feature_threads: 4,
            ..Default::default()
        };

        learner.update_config(new_config.clone());
        assert_eq!(learner.config().base_config.max_leaves, 15);
        assert_eq!(learner.config().num_feature_threads, 4);
    }
}