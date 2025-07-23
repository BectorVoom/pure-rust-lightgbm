//! Serial tree learner implementation for the Pure Rust LightGBM framework.
//!
//! This module provides the core serial tree learning algorithm that constructs
//! decision trees using histogram-based split finding and gradient boosting.

use crate::core::types::{DataSize, FeatureIndex, NodeIndex, Score};
use crate::tree::histogram::{
    HistogramBuilder, HistogramBuilderConfig, HistogramPool, HistogramPoolConfig,
    BinMapper, FeatureType,
};
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
use std::sync::Mutex;

/// Configuration for the serial tree learner.
#[derive(Debug, Clone)]
pub struct SerialTreeLearnerConfig {
    /// Maximum number of leaves in the tree
    pub max_leaves: usize,
    /// Maximum tree depth
    pub max_depth: usize,
    /// Minimum number of data points required in each leaf
    pub min_data_in_leaf: DataSize,
    /// Minimum sum of hessians required in each leaf
    pub min_sum_hessian_in_leaf: f64,
    /// L1 regularization parameter
    pub lambda_l1: f64,
    /// L2 regularization parameter
    pub lambda_l2: f64,
    /// Minimum gain required for a split
    pub min_split_gain: f64,
    /// Maximum number of bins for histogram construction
    pub max_bin: usize,
    /// Feature sampling configuration
    pub feature_sampling: FeatureSamplingConfig,
    /// Histogram builder configuration
    pub histogram_config: HistogramBuilderConfig,
    /// Whether to use histogram subtraction optimization
    pub use_histogram_subtraction: bool,
    /// Learning rate (shrinkage) for the tree
    pub learning_rate: f64,
}

impl Default for SerialTreeLearnerConfig {
    fn default() -> Self {
        SerialTreeLearnerConfig {
            max_leaves: 31,
            max_depth: 6,
            min_data_in_leaf: 20,
            min_sum_hessian_in_leaf: 1e-3,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_split_gain: 0.0,
            max_bin: 255,
            feature_sampling: FeatureSamplingConfig::default(),
            histogram_config: HistogramBuilderConfig::default(),
            use_histogram_subtraction: true,
            learning_rate: 0.1,
        }
    }
}

/// Data structure representing a dataset for tree learning.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Feature matrix (num_data × num_features)
    pub features: Array2<f32>,
    /// Number of data points
    pub num_data: DataSize,
    /// Number of features
    pub num_features: usize,
    /// Bin mappers for each feature
    pub bin_mappers: Vec<BinMapper>,
}

impl Dataset {
    /// Creates a new dataset from feature matrix.
    pub fn new(features: Array2<f32>, max_bin: usize) -> anyhow::Result<Self> {
        let (num_data, num_features) = features.dim();
        let mut bin_mappers = Vec::with_capacity(num_features);

        // Create bin mappers for each feature
        for feature_idx in 0..num_features {
            let feature_column = features.column(feature_idx);
            let feature_values: Vec<f32> = feature_column.to_vec();
            let bin_mapper = BinMapper::new_numerical(&feature_values, max_bin);
            bin_mappers.push(bin_mapper);
        }

        Ok(Dataset {
            features,
            num_data: num_data as DataSize,
            num_features,
            bin_mappers,
        })
    }

    /// Returns a view of the feature matrix.
    pub fn features(&self) -> ArrayView2<f32> {
        self.features.view()
    }

    /// Returns the bin mapper for a specific feature.
    pub fn bin_mapper(&self, feature_idx: FeatureIndex) -> Option<&BinMapper> {
        self.bin_mappers.get(feature_idx)
    }
}

/// Information about a tree node being processed.
#[derive(Debug, Clone)]
struct NodeInfo {
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

impl NodeInfo {
    fn new(
        node_index: NodeIndex,
        data_indices: Vec<DataSize>,
        depth: usize,
        sum_gradients: f64,
        sum_hessians: f64,
    ) -> Self {
        NodeInfo {
            node_index,
            data_indices,
            depth,
            sum_gradients,
            sum_hessians,
            path_features: Vec::new(),
        }
    }
}

/// Serial tree learner implementing the core GBDT tree construction algorithm.
pub struct SerialTreeLearner {
    config: SerialTreeLearnerConfig,
    feature_sampler: FeatureSampler,
    histogram_builder: HistogramBuilder,
    split_finder: SplitFinder,
    split_evaluator: SplitEvaluator,
    constraint_manager: ConstraintManager,
    // Cache for histogram reuse
    node_histograms: HashMap<NodeIndex, Array1<f64>>,
}

impl SerialTreeLearner {
    /// Creates a new serial tree learner with the given configuration.
    pub fn new(config: SerialTreeLearnerConfig) -> anyhow::Result<Self> {
        // Create feature sampler
        let feature_sampler = FeatureSampler::with_strategy(
            config.feature_sampling.clone(),
            SamplingStrategy::Uniform,
        );

        // Create histogram pool and builder
        let histogram_pool_config = HistogramPoolConfig {
            max_bin: config.max_bin,
            num_features: 0, // Will be set when we know the actual number
            max_pool_size: 100,
            initial_pool_size: 20,
            use_double_precision: true,
        };
        let histogram_pool = HistogramPool::new(histogram_pool_config);
        let histogram_builder = HistogramBuilder::new(config.histogram_config.clone(), histogram_pool);

        // Create split finder
        let split_finder_config = SplitFinderConfig {
            min_data_in_leaf: config.min_data_in_leaf,
            min_sum_hessian_in_leaf: config.min_sum_hessian_in_leaf,
            lambda_l1: config.lambda_l1,
            lambda_l2: config.lambda_l2,
            min_split_gain: config.min_split_gain,
            max_bin: config.max_bin,
        };
        let split_finder = SplitFinder::new(split_finder_config);

        // Create split evaluator
        let split_evaluator_config = SplitEvaluatorConfig::default();
        let split_evaluator = SplitEvaluator::new(split_evaluator_config);

        // Create constraint manager
        let constraint_manager = ConstraintManager::new();

        Ok(SerialTreeLearner {
            config,
            feature_sampler,
            histogram_builder,
            split_finder,
            split_evaluator,
            constraint_manager,
            node_histograms: HashMap::new(),
        })
    }

    /// Trains a decision tree using the given dataset, gradients, and hessians.
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

        // Initialize tree with root node
        let mut tree = Tree::with_capacity(self.config.max_leaves, self.config.learning_rate);

        // Initialize root node statistics
        let total_gradients: f64 = gradients.iter().map(|&g| g as f64).sum();
        let total_hessians: f64 = hessians.iter().map(|&h| h as f64).sum();
        let all_data_indices: Vec<DataSize> = (0..dataset.num_data).collect();

        // Update root node with statistics
        if let Some(root_node) = tree.node_mut(0) {
            root_node.update_statistics(total_gradients, total_hessians, dataset.num_data);
            
            // Calculate and set root output
            let root_output = root_node.calculate_leaf_output(
                self.config.lambda_l1,
                self.config.lambda_l2,
            );
            root_node.set_leaf_output(root_output);
        }

        // Initialize queue with root node
        let mut node_queue = VecDeque::new();
        node_queue.push_back(NodeInfo::new(
            0,
            all_data_indices,
            0,
            total_gradients,
            total_hessians,
        ));

        // Clear histogram cache
        self.node_histograms.clear();

        // Build tree iteratively using breadth-first approach
        // This allows for histogram subtraction optimization between sibling nodes
        while let Some(node_info) = node_queue.pop_front() {
            if tree.num_leaves() >= self.config.max_leaves {
                break;
            }

            if node_info.depth >= self.config.max_depth {
                continue;
            }

            if let Some(split_result) = self.find_best_split(
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

                // Cache parent histogram for potential subtraction optimization
                if self.config.use_histogram_subtraction {
                    // We'll cache parent histograms when we construct them
                    // This is handled in the find_best_split method
                }

                // Add children to queue - order matters for subtraction optimization
                // Process both children to enable sibling subtraction
                node_queue.push_back(left_child_info);
                node_queue.push_back(right_child_info);
            }
        }

        // Set final leaf outputs for all leaf nodes
        self.finalize_tree_outputs(&mut tree)?;

        Ok(tree)
    }

    /// Finds the best split for a given node.
    fn find_best_split(
        &mut self,
        dataset: &Dataset,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
        node_info: &NodeInfo,
    ) -> anyhow::Result<Option<SplitInfo>> {
        // Check if node can be split
        if node_info.data_indices.len() < (2 * self.config.min_data_in_leaf) as usize {
            return Ok(None);
        }

        if node_info.sum_hessians < 2.0 * self.config.min_sum_hessian_in_leaf {
            return Ok(None);
        }

        // Sample features for this split
        let sampled_features = self.feature_sampler.sample_features(
            dataset.num_features,
            0, // iteration not used in this context
        )?;

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

        // Construct histograms for sampled features
        let histograms = self.construct_node_histograms(
            dataset,
            gradients,
            hessians,
            &node_info.data_indices,
            &allowed_features,
        )?;

        // Cache parent histogram for potential histogram subtraction optimization
        if self.config.use_histogram_subtraction {
            self.cache_histograms(node_info.node_index, &histograms, &allowed_features);
        }

        // Find best split among all features
        let mut best_split: Option<SplitInfo> = None;

        for (i, &feature_idx) in allowed_features.iter().enumerate() {
            if i >= histograms.len() {
                continue;
            }

            let bin_mapper = dataset.bin_mapper(feature_idx).unwrap();
            let split_candidate = self.split_finder.find_best_split_for_feature(
                feature_idx,
                &histograms[i].view(),
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
                    0.0, // parent output not needed for validation
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

    /// Constructs histograms for the specified features and data indices.
    fn construct_node_histograms(
        &self,
        dataset: &Dataset,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
        data_indices: &[DataSize],
        feature_indices: &[FeatureIndex],
    ) -> anyhow::Result<Vec<Array1<f64>>> {
        let mut histograms = Vec::with_capacity(feature_indices.len());

        for &feature_idx in feature_indices {
            let feature_column = dataset.features.column(feature_idx);
            let bin_mapper = dataset.bin_mapper(feature_idx).unwrap();

            let histogram = self.histogram_builder.construct_feature_histogram(
                &feature_column,
                gradients,
                hessians,
                data_indices,
                bin_mapper,
            )?;

            histograms.push(histogram);
        }

        Ok(histograms)
    }

    /// Constructs histograms using subtraction optimization (parent - sibling = child).
    /// This is a key optimization from the LightGBM C implementation.
    fn construct_histograms_with_subtraction(
        &mut self,
        dataset: &Dataset,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
        parent_info: &NodeInfo,
        sibling_info: &NodeInfo,
        current_info: &NodeInfo,
        feature_indices: &[FeatureIndex],
    ) -> anyhow::Result<Vec<Array1<f64>>> {
        // Check if we should use histogram subtraction optimization
        let should_use_subtraction = self.config.use_histogram_subtraction &&
            current_info.data_indices.len() < sibling_info.data_indices.len();

        if should_use_subtraction {
            // Try to get parent histogram from cache
            let parent_histograms_opt = self.get_cached_histograms(parent_info.node_index, feature_indices);
            
            if let Some(parent_histograms) = parent_histograms_opt {
                // Construct sibling histograms (larger child, build directly)
                let sibling_histograms = self.construct_node_histograms(
                    dataset,
                    gradients,
                    hessians,
                    &sibling_info.data_indices,
                    feature_indices,
                )?;

                // Cache sibling histograms for potential future use
                self.cache_histograms(sibling_info.node_index, &sibling_histograms, feature_indices);

                // Use histogram subtraction: current = parent - sibling
                let mut current_histograms = Vec::with_capacity(feature_indices.len());
                for i in 0..feature_indices.len() {
                    let current_histogram = self.histogram_builder.construct_histogram_by_subtraction(
                        &parent_histograms[i].view(),
                        &sibling_histograms[i].view(),
                    )?;
                    current_histograms.push(current_histogram);
                }

                return Ok(current_histograms);
            }
        }

        // Fallback to direct construction
        let current_histograms = self.construct_node_histograms(
            dataset,
            gradients,
            hessians,
            &current_info.data_indices,
            feature_indices,
        )?;

        // Cache histograms for potential future subtraction
        self.cache_histograms(current_info.node_index, &current_histograms, feature_indices);

        Ok(current_histograms)
    }

    /// Caches histograms for a node for potential histogram subtraction.
    fn cache_histograms(
        &mut self,
        node_index: NodeIndex,
        histograms: &[Array1<f64>],
        feature_indices: &[FeatureIndex],
    ) {
        // For simplicity, we cache a flattened version
        // In a full implementation, we might use more sophisticated caching
        if feature_indices.len() == 1 && !histograms.is_empty() {
            self.node_histograms.insert(node_index, histograms[0].clone());
        }
    }

    /// Retrieves cached histograms for a node.
    fn get_cached_histograms(
        &self,
        node_index: NodeIndex,
        feature_indices: &[FeatureIndex],
    ) -> Option<Vec<Array1<f64>>> {
        // Simple cache retrieval - in practice this would be more sophisticated
        if feature_indices.len() == 1 {
            self.node_histograms.get(&node_index).map(|h| vec![h.clone()])
        } else {
            None
        }
    }

    /// Applies a split to the tree and returns information about the child nodes.
    fn apply_split(
        &mut self,
        tree: &mut Tree,
        dataset: &Dataset,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
        parent_info: &NodeInfo,
        split: &SplitInfo,
    ) -> anyhow::Result<(NodeInfo, NodeInfo)> {
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

        let left_child_info = NodeInfo {
            node_index: left_child_idx,
            data_indices: left_indices,
            depth: parent_info.depth + 1,
            sum_gradients: left_sum_gradients,
            sum_hessians: left_sum_hessians,
            path_features: left_path_features,
        };

        let right_child_info = NodeInfo {
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
                // Handle missing values - default direction could be configurable
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
                        self.config.lambda_l1,
                        self.config.lambda_l2,
                    );
                    node.set_leaf_output(output);
                }
            }
        }

        Ok(())
    }

    /// Updates the learner configuration.
    pub fn update_config(&mut self, config: SerialTreeLearnerConfig) {
        self.config = config.clone();
        self.feature_sampler.update_config(config.feature_sampling);
        self.histogram_builder.update_config(config.histogram_config);
        
        // Update split finder configuration
        let split_finder_config = SplitFinderConfig {
            min_data_in_leaf: config.min_data_in_leaf,
            min_sum_hessian_in_leaf: config.min_sum_hessian_in_leaf,
            lambda_l1: config.lambda_l1,
            lambda_l2: config.lambda_l2,
            min_split_gain: config.min_split_gain,
            max_bin: config.max_bin,
        };
        self.split_finder.update_config(split_finder_config);
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &SerialTreeLearnerConfig {
        &self.config
    }

    /// Resets the learner state.
    pub fn reset(&mut self) {
        self.feature_sampler.reset();
        self.node_histograms.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_dataset_creation() {
        let features = Array2::from_shape_vec(
            (5, 3),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        ).unwrap();

        let dataset = Dataset::new(features, 10).unwrap();
        assert_eq!(dataset.num_data, 5);
        assert_eq!(dataset.num_features, 3);
        assert_eq!(dataset.bin_mappers.len(), 3);
    }

    #[test]
    fn test_serial_tree_learner_creation() {
        let config = SerialTreeLearnerConfig::default();
        let learner = SerialTreeLearner::new(config);
        assert!(learner.is_ok());
    }

    #[test]
    fn test_simple_tree_training() {
        let config = SerialTreeLearnerConfig {
            max_leaves: 3,
            max_depth: 2,
            min_data_in_leaf: 1,
            min_sum_hessian_in_leaf: 0.1,
            lambda_l1: 0.0,
            lambda_l2: 0.1,
            ..Default::default()
        };

        let mut learner = SerialTreeLearner::new(config).unwrap();

        // Create simple dataset
        let features = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 2.0],
        ).unwrap();
        let dataset = Dataset::new(features, 10).unwrap();

        // Create gradients and hessians
        let gradients = Array1::from(vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]);
        let hessians = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let result = learner.train(&dataset, &gradients.view(), &hessians.view(), 0);
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert!(tree.num_nodes() >= 1);
        assert!(tree.num_leaves() >= 1);
    }

    #[test]
    fn test_data_partitioning() {
        let config = SerialTreeLearnerConfig::default();
        let learner = SerialTreeLearner::new(config).unwrap();

        let features = Array2::from_shape_vec(
            (4, 1),
            vec![1.0, 2.0, 3.0, 4.0],
        ).unwrap();
        let dataset = Dataset::new(features, 10).unwrap();

        let data_indices = vec![0, 1, 2, 3];
        let (left, right) = learner.partition_data(&dataset, &data_indices, 0, 2.5).unwrap();

        assert_eq!(left, vec![0, 1]); // values 1.0, 2.0 <= 2.5
        assert_eq!(right, vec![2, 3]); // values 3.0, 4.0 > 2.5
    }

    #[test]
    fn test_node_info_creation() {
        let node_info = NodeInfo::new(0, vec![0, 1, 2], 1, 10.0, 5.0);
        assert_eq!(node_info.node_index, 0);
        assert_eq!(node_info.data_indices, vec![0, 1, 2]);
        assert_eq!(node_info.depth, 1);
        assert_eq!(node_info.sum_gradients, 10.0);
        assert_eq!(node_info.sum_hessians, 5.0);
        assert!(node_info.path_features.is_empty());
    }

    #[test]
    fn test_learner_config_update() {
        let config = SerialTreeLearnerConfig::default();
        let mut learner = SerialTreeLearner::new(config).unwrap();

        let new_config = SerialTreeLearnerConfig {
            max_leaves: 15,
            lambda_l1: 0.1,
            lambda_l2: 0.2,
            ..Default::default()
        };

        learner.update_config(new_config.clone());
        assert_eq!(learner.config().max_leaves, 15);
        assert_eq!(learner.config().lambda_l1, 0.1);
        assert_eq!(learner.config().lambda_l2, 0.2);
    }

    #[test]
    fn test_empty_dataset_handling() {
        let config = SerialTreeLearnerConfig::default();
        let mut learner = SerialTreeLearner::new(config).unwrap();

        let features = Array2::zeros((0, 1));
        let dataset = Dataset::new(features, 10).unwrap();
        let gradients = Array1::zeros(0);
        let hessians = Array1::zeros(0);

        let result = learner.train(&dataset, &gradients.view(), &hessians.view(), 0);
        assert!(result.is_err());
    }
}