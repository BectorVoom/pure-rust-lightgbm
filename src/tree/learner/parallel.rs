//! Parallel tree learner implementation for the Pure Rust LightGBM framework.
//!
//! This module provides a data-parallel tree learning algorithm that distributes
//! computation across multiple threads for improved performance on large datasets.

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
use crossbeam_channel::{Receiver, Sender, bounded};

/// Configuration for the parallel tree learner.
#[derive(Debug, Clone)]
pub struct ParallelTreeLearnerConfig {
    /// Base configuration from serial learner
    pub base_config: SerialTreeLearnerConfig,
    /// Number of threads to use for parallel processing
    pub num_threads: usize,
    /// Chunk size for data partitioning
    pub chunk_size: usize,
    /// Whether to use parallel histogram construction
    pub parallel_histograms: bool,
    /// Whether to use parallel split finding
    pub parallel_splits: bool,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

impl Default for ParallelTreeLearnerConfig {
    fn default() -> Self {
        ParallelTreeLearnerConfig {
            base_config: SerialTreeLearnerConfig::default(),
            num_threads: num_cpus::get(),
            chunk_size: 1024,
            parallel_histograms: true,
            parallel_splits: true,
            load_balancing: LoadBalancingStrategy::Dynamic,
        }
    }
}

/// Load balancing strategies for parallel processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    /// Static load balancing with equal chunks
    Static,
    /// Dynamic load balancing based on data distribution
    Dynamic,
    /// Work-stealing approach
    WorkStealing,
}

/// Thread-safe node information for parallel processing.
#[derive(Debug, Clone)]
struct ParallelNodeInfo {
    /// Node index in the tree
    node_index: NodeIndex,
    /// Data indices belonging to this node
    data_indices: Arc<Vec<DataSize>>,
    /// Node depth
    depth: usize,
    /// Sum of gradients in this node
    sum_gradients: f64,
    /// Sum of hessians in this node
    sum_hessians: f64,
    /// Features used in the path to this node
    path_features: Arc<Vec<FeatureIndex>>,
}

impl ParallelNodeInfo {
    fn new(
        node_index: NodeIndex,
        data_indices: Vec<DataSize>,
        depth: usize,
        sum_gradients: f64,
        sum_hessians: f64,
    ) -> Self {
        ParallelNodeInfo {
            node_index,
            data_indices: Arc::new(data_indices),
            depth,
            sum_gradients,
            sum_hessians,
            path_features: Arc::new(Vec::new()),
        }
    }

    fn with_path_features(mut self, path_features: Vec<FeatureIndex>) -> Self {
        self.path_features = Arc::new(path_features);
        self
    }
}

/// Work item for parallel processing.
#[derive(Debug)]
struct WorkItem {
    node_info: ParallelNodeInfo,
    priority: f64, // For load balancing
}

impl WorkItem {
    fn new(node_info: ParallelNodeInfo) -> Self {
        // Priority based on data size and depth (deeper nodes processed first)
        let priority = node_info.data_indices.len() as f64 * (1.0 + node_info.depth as f64 * 0.1);
        WorkItem { node_info, priority }
    }
}

impl PartialEq for WorkItem {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for WorkItem {}

impl PartialOrd for WorkItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.priority.partial_cmp(&self.priority) // Reverse order for max-heap
    }
}

impl Ord for WorkItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Parallel tree learner implementing data-parallel tree construction.
pub struct ParallelTreeLearner {
    config: ParallelTreeLearnerConfig,
    feature_sampler: Arc<Mutex<FeatureSampler>>,
    histogram_builder: Arc<HistogramBuilder>,
    split_finder: Arc<SplitFinder>,
    split_evaluator: Arc<SplitEvaluator>,
    constraint_manager: Arc<ConstraintManager>,
    thread_pool: rayon::ThreadPool,
}

impl ParallelTreeLearner {
    /// Creates a new parallel tree learner with the given configuration.
    pub fn new(config: ParallelTreeLearnerConfig) -> anyhow::Result<Self> {
        // Create thread pool
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_threads)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to create thread pool: {}", e))?;

        // Create feature sampler
        let feature_sampler = Arc::new(Mutex::new(FeatureSampler::with_strategy(
            config.base_config.feature_sampling.clone(),
            SamplingStrategy::Uniform,
        )));

        // Create histogram pool and builder
        let histogram_pool_config = HistogramPoolConfig {
            max_bin: config.base_config.max_bin,
            num_features: 0,
            max_pool_size: config.num_threads * 10,
            initial_pool_size: config.num_threads * 2,
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
        let split_finder = Arc::new(SplitFinder::new(split_finder_config));

        // Create split evaluator
        let split_evaluator_config = SplitEvaluatorConfig::default();
        let split_evaluator = Arc::new(SplitEvaluator::new(split_evaluator_config));

        // Create constraint manager
        let constraint_manager = Arc::new(ConstraintManager::new());

        Ok(ParallelTreeLearner {
            config,
            feature_sampler,
            histogram_builder,
            split_finder,
            split_evaluator,
            constraint_manager,
            thread_pool,
        })
    }

    /// Trains a decision tree using parallel processing.
    pub fn train(
        &self,
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
        let tree = Arc::new(RwLock::new(
            Tree::with_capacity(self.config.base_config.max_leaves, self.config.base_config.learning_rate)
        ));

        // Initialize root node statistics
        let total_gradients: f64 = gradients.iter().map(|&g| g as f64).sum();
        let total_hessians: f64 = hessians.iter().map(|&h| h as f64).sum();
        let all_data_indices: Vec<DataSize> = (0..dataset.num_data).collect();

        // Update root node
        {
            let mut tree_guard = tree.write().unwrap();
            if let Some(root_node) = tree_guard.node_mut(0) {
                root_node.update_statistics(total_gradients, total_hessians, dataset.num_data);
                let root_output = root_node.calculate_leaf_output(
                    self.config.base_config.lambda_l1,
                    self.config.base_config.lambda_l2,
                );
                root_node.set_leaf_output(root_output);
            }
        }

        // Create work queue for parallel processing
        let (work_sender, work_receiver) = bounded::<WorkItem>(self.config.num_threads * 2);
        
        // Initialize with root node
        let root_info = ParallelNodeInfo::new(
            0,
            all_data_indices,
            0,
            total_gradients,
            total_hessians,
        );
        work_sender.send(WorkItem::new(root_info))?;

        // Process nodes in parallel
        self.thread_pool.scope(|scope| {
            // Spawn worker threads
            for thread_id in 0..self.config.num_threads {
                let work_receiver = work_receiver.clone();
                let work_sender = work_sender.clone();
                let tree = Arc::clone(&tree);
                let dataset = dataset;
                let gradients = gradients;
                let hessians = hessians;

                scope.spawn(move |_| {
                    self.worker_thread(
                        thread_id,
                        &work_receiver,
                        &work_sender,
                        &tree,
                        dataset,
                        gradients,
                        hessians,
                    );
                });
            }

            // Drop sender to signal completion when no more work
            drop(work_sender);
        });

        // Finalize tree
        let mut final_tree = Arc::try_unwrap(tree)
            .map_err(|_| anyhow::anyhow!("Failed to unwrap tree"))?
            .into_inner()
            .unwrap();

        self.finalize_tree_outputs(&mut final_tree)?;

        Ok(final_tree)
    }

    /// Worker thread function for parallel tree construction.
    fn worker_thread(
        &self,
        _thread_id: usize,
        work_receiver: &Receiver<WorkItem>,
        work_sender: &Sender<WorkItem>,
        tree: &Arc<RwLock<Tree>>,
        dataset: &Dataset,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
    ) {
        while let Ok(work_item) = work_receiver.recv() {
            let node_info = work_item.node_info;

            // Check if we can continue splitting
            {
                let tree_guard = tree.read().unwrap();
                if tree_guard.num_leaves() >= self.config.base_config.max_leaves {
                    break;
                }
            }

            if node_info.depth >= self.config.base_config.max_depth {
                continue;
            }

            // Try to find and apply a split
            if let Ok(Some(split_result)) = self.find_best_split_parallel(
                dataset,
                gradients,
                hessians,
                &node_info,
            ) {
                if let Ok((left_child_info, right_child_info)) = self.apply_split_parallel(
                    tree,
                    dataset,
                    gradients,
                    hessians,
                    &node_info,
                    &split_result,
                ) {
                    // Add child nodes to work queue
                    let _ = work_sender.send(WorkItem::new(left_child_info));
                    let _ = work_sender.send(WorkItem::new(right_child_info));
                }
            }
        }
    }

    /// Finds the best split for a node in parallel.
    fn find_best_split_parallel(
        &self,
        dataset: &Dataset,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
        node_info: &ParallelNodeInfo,
    ) -> anyhow::Result<Option<SplitInfo>> {
        // Check if node can be split
        if node_info.data_indices.len() < (2 * self.config.base_config.min_data_in_leaf) as usize {
            return Ok(None);
        }

        if node_info.sum_hessians < 2.0 * self.config.base_config.min_sum_hessian_in_leaf {
            return Ok(None);
        }

        // Sample features
        let sampled_features = {
            let mut sampler = self.feature_sampler.lock().unwrap();
            sampler.sample_features(dataset.num_features, 0)?
        };

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

        // Construct histograms in parallel
        let histograms = if self.config.parallel_histograms {
            self.construct_histograms_parallel(
                dataset,
                gradients,
                hessians,
                &node_info.data_indices,
                &allowed_features,
            )?
        } else {
            self.construct_histograms_sequential(
                dataset,
                gradients,
                hessians,
                &node_info.data_indices,
                &allowed_features,
            )?
        };

        // Find best split in parallel
        if self.config.parallel_splits {
            self.find_best_split_from_histograms_parallel(
                &histograms,
                &allowed_features,
                dataset,
                node_info,
            )
        } else {
            self.find_best_split_from_histograms_sequential(
                &histograms,
                &allowed_features,
                dataset,
                node_info,
            )
        }
    }

    /// Constructs histograms for features in parallel.
    fn construct_histograms_parallel(
        &self,
        dataset: &Dataset,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
        data_indices: &[DataSize],
        feature_indices: &[FeatureIndex],
    ) -> anyhow::Result<Vec<Array1<f64>>> {
        let histograms: Result<Vec<_>, _> = feature_indices
            .par_iter()
            .map(|&feature_idx| {
                let feature_column = dataset.features.column(feature_idx);
                let bin_mapper = dataset.bin_mapper(feature_idx).unwrap();

                self.histogram_builder.construct_feature_histogram(
                    &feature_column,
                    gradients,
                    hessians,
                    data_indices,
                    bin_mapper,
                )
            })
            .collect();

        histograms.map_err(|e| anyhow::anyhow!("Parallel histogram construction failed: {}", e))
    }

    /// Constructs histograms for features sequentially.
    fn construct_histograms_sequential(
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

    /// Finds the best split from histograms in parallel.
    fn find_best_split_from_histograms_parallel(
        &self,
        histograms: &[Array1<f64>],
        feature_indices: &[FeatureIndex],
        dataset: &Dataset,
        node_info: &ParallelNodeInfo,
    ) -> anyhow::Result<Option<SplitInfo>> {
        let splits: Vec<Option<SplitInfo>> = histograms
            .par_iter()
            .zip(feature_indices.par_iter())
            .map(|(histogram, &feature_idx)| {
                let bin_mapper = dataset.bin_mapper(feature_idx).unwrap();
                let split_candidate = self.split_finder.find_best_split_for_feature(
                    feature_idx,
                    &histogram.view(),
                    node_info.sum_gradients,
                    node_info.sum_hessians,
                    node_info.data_indices.len() as DataSize,
                    bin_mapper.bin_upper_bounds(),
                );

                if let Some(split) = split_candidate {
                    let validation_result = self.constraint_manager.validate_split(
                        &split,
                        node_info.depth,
                        0.0,
                        &node_info.path_features,
                    );

                    if validation_result.is_valid() {
                        Some(split)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        // Find the best split among all candidates
        Ok(splits
            .into_iter()
            .flatten()
            .max_by(|a, b| a.gain.partial_cmp(&b.gain).unwrap_or(std::cmp::Ordering::Equal)))
    }

    /// Finds the best split from histograms sequentially.
    fn find_best_split_from_histograms_sequential(
        &self,
        histograms: &[Array1<f64>],
        feature_indices: &[FeatureIndex],
        dataset: &Dataset,
        node_info: &ParallelNodeInfo,
    ) -> anyhow::Result<Option<SplitInfo>> {
        let mut best_split: Option<SplitInfo> = None;

        for (i, &feature_idx) in feature_indices.iter().enumerate() {
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

    /// Applies a split to the tree in a thread-safe manner.
    fn apply_split_parallel(
        &self,
        tree: &Arc<RwLock<Tree>>,
        dataset: &Dataset,
        gradients: &ArrayView1<Score>,
        hessians: &ArrayView1<Score>,
        parent_info: &ParallelNodeInfo,
        split: &SplitInfo,
    ) -> anyhow::Result<(ParallelNodeInfo, ParallelNodeInfo)> {
        // Partition data based on split
        let (left_indices, right_indices) = self.partition_data_parallel(
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

        // Apply split to tree (requires write lock)
        let (left_child_idx, right_child_idx) = {
            let mut tree_guard = tree.write().unwrap();
            tree_guard.split_node(
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
            )?
        };

        // Create child node information
        let mut left_path_features = parent_info.path_features.as_ref().clone();
        left_path_features.push(split.feature);

        let mut right_path_features = parent_info.path_features.as_ref().clone();
        right_path_features.push(split.feature);

        let left_child_info = ParallelNodeInfo {
            node_index: left_child_idx,
            data_indices: Arc::new(left_indices),
            depth: parent_info.depth + 1,
            sum_gradients: left_sum_gradients,
            sum_hessians: left_sum_hessians,
            path_features: Arc::new(left_path_features),
        };

        let right_child_info = ParallelNodeInfo {
            node_index: right_child_idx,
            data_indices: Arc::new(right_indices),
            depth: parent_info.depth + 1,
            sum_gradients: right_sum_gradients,
            sum_hessians: right_sum_hessians,
            path_features: Arc::new(right_path_features),
        };

        Ok((left_child_info, right_child_info))
    }

    /// Partitions data indices based on a split condition in parallel.
    fn partition_data_parallel(
        &self,
        dataset: &Dataset,
        data_indices: &[DataSize],
        split_feature: FeatureIndex,
        threshold: f64,
    ) -> anyhow::Result<(Vec<DataSize>, Vec<DataSize>)> {
        let chunk_size = self.config.chunk_size.max(data_indices.len() / self.config.num_threads);
        
        let partitions: Vec<(Vec<DataSize>, Vec<DataSize>)> = data_indices
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut left = Vec::new();
                let mut right = Vec::new();

                for &data_idx in chunk {
                    let feature_value = dataset.features[[data_idx as usize, split_feature]];
                    
                    if feature_value.is_nan() {
                        left.push(data_idx);
                    } else if (feature_value as f64) <= threshold {
                        left.push(data_idx);
                    } else {
                        right.push(data_idx);
                    }
                }

                (left, right)
            })
            .collect();

        // Merge results
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for (left, right) in partitions {
            left_indices.extend(left);
            right_indices.extend(right);
        }

        Ok((left_indices, right_indices))
    }

    /// Finalizes leaf outputs for all leaf nodes in the tree.
    fn finalize_tree_outputs(&self, tree: &mut Tree) -> anyhow::Result<()> {
        let leaf_indices = tree.leaf_indices();

        leaf_indices.par_iter().try_for_each(|&leaf_idx| -> anyhow::Result<()> {
            // Note: This would require parallel mutable access to tree nodes,
            // which is not safe. In practice, we would need to collect the
            // outputs first and then apply them sequentially.
            Ok(())
        })?;

        // Sequential finalization for safety
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

    /// Updates the learner configuration.
    pub fn update_config(&mut self, config: ParallelTreeLearnerConfig) {
        self.config = config.clone();
        
        if let Ok(mut sampler) = self.feature_sampler.lock() {
            sampler.update_config(config.base_config.feature_sampling);
        }
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &ParallelTreeLearnerConfig {
        &self.config
    }

    /// Returns the number of threads used.
    pub fn num_threads(&self) -> usize {
        self.config.num_threads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_parallel_tree_learner_creation() {
        let config = ParallelTreeLearnerConfig::default();
        let learner = ParallelTreeLearner::new(config);
        assert!(learner.is_ok());
    }

    #[test]
    fn test_parallel_tree_training() {
        let config = ParallelTreeLearnerConfig {
            base_config: SerialTreeLearnerConfig {
                max_leaves: 3,
                max_depth: 2,
                min_data_in_leaf: 1,
                min_sum_hessian_in_leaf: 0.1,
                lambda_l1: 0.0,
                lambda_l2: 0.1,
                ..Default::default()
            },
            num_threads: 2,
            ..Default::default()
        };

        let learner = ParallelTreeLearner::new(config).unwrap();

        // Create simple dataset
        let features = Array2::from_shape_vec(
            (8, 2),
            vec![1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 2.0, 4.0, 2.0],
        ).unwrap();
        let dataset = Dataset::new(features, 10).unwrap();

        // Create gradients and hessians
        let gradients = Array1::from(vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -0.8, 0.8]);
        let hessians = Array1::from(vec![1.0; 8]);

        let result = learner.train(&dataset, &gradients.view(), &hessians.view(), 0);
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert!(tree.num_nodes() >= 1);
        assert!(tree.num_leaves() >= 1);
    }

    #[test]
    fn test_load_balancing_strategies() {
        for strategy in [
            LoadBalancingStrategy::Static,
            LoadBalancingStrategy::Dynamic,
            LoadBalancingStrategy::WorkStealing,
        ] {
            let config = ParallelTreeLearnerConfig {
                load_balancing: strategy,
                ..Default::default()
            };
            
            let learner = ParallelTreeLearner::new(config);
            assert!(learner.is_ok());
        }
    }

    #[test]
    fn test_parallel_data_partitioning() {
        let config = ParallelTreeLearnerConfig {
            num_threads: 2,
            chunk_size: 2,
            ..Default::default()
        };
        let learner = ParallelTreeLearner::new(config).unwrap();

        let features = Array2::from_shape_vec(
            (6, 1),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ).unwrap();
        let dataset = Dataset::new(features, 10).unwrap();

        let data_indices = vec![0, 1, 2, 3, 4, 5];
        let (left, right) = learner.partition_data_parallel(&dataset, &data_indices, 0, 3.5).unwrap();

        assert_eq!(left.len(), 3); // values 1.0, 2.0, 3.0 <= 3.5
        assert_eq!(right.len(), 3); // values 4.0, 5.0, 6.0 > 3.5
    }

    #[test]
    fn test_work_item_ordering() {
        let node_info1 = ParallelNodeInfo::new(0, vec![0, 1], 1, 10.0, 5.0);
        let node_info2 = ParallelNodeInfo::new(1, vec![2, 3, 4], 2, 15.0, 7.0);

        let work_item1 = WorkItem::new(node_info1);
        let work_item2 = WorkItem::new(node_info2);

        // Higher priority items should be processed first
        // Priority is based on data size and depth
        assert!(work_item2.priority > work_item1.priority);
    }

    #[test]
    fn test_config_update() {
        let config = ParallelTreeLearnerConfig::default();
        let mut learner = ParallelTreeLearner::new(config).unwrap();

        let new_config = ParallelTreeLearnerConfig {
            base_config: SerialTreeLearnerConfig {
                max_leaves: 15,
                lambda_l1: 0.1,
                ..Default::default()
            },
            num_threads: 4,
            ..Default::default()
        };

        learner.update_config(new_config.clone());
        assert_eq!(learner.config().base_config.max_leaves, 15);
        assert_eq!(learner.config().num_threads, 4);
    }
}