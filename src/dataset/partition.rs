//! Data partitioning utilities for Pure Rust LightGBM.
//!
//! This module provides functionality for partitioning datasets across
//! different nodes in tree learning algorithms, supporting both serial
//! and parallel tree construction.

use crate::core::error::{LightGBMError, Result};
use crate::core::types::*;
use ndarray::Array1;
use std::collections::HashMap;

/// Data partition structure for tree learning
#[derive(Debug, Clone)]
pub struct DataPartition {
    /// Indices of data points in each leaf
    leaf_indices: HashMap<NodeIndex, Vec<DataSize>>,
    /// Current leaf assignment for each data point
    data_to_leaf: Array1<NodeIndex>,
    /// Number of data points
    num_data: DataSize,
    /// Current number of leaves
    num_leaves: usize,
}

impl DataPartition {
    /// Create a new data partition
    pub fn new(num_data: DataSize) -> Self {
        let mut leaf_indices = HashMap::new();
        let data_indices: Vec<DataSize> = (0..num_data).collect();
        leaf_indices.insert(0, data_indices);

        DataPartition {
            leaf_indices,
            data_to_leaf: Array1::zeros(num_data as usize),
            num_data,
            num_leaves: 1,
        }
    }

    /// Initialize partition for a new tree
    pub fn init(&mut self, num_data: DataSize) -> Result<()> {
        self.num_data = num_data;
        self.num_leaves = 1;
        self.data_to_leaf.fill(0);

        self.leaf_indices.clear();
        let data_indices: Vec<DataSize> = (0..num_data).collect();
        self.leaf_indices.insert(0, data_indices);

        Ok(())
    }

    /// Split a leaf into two child leaves
    pub fn split_leaf(
        &mut self,
        leaf_id: NodeIndex,
        left_child_id: NodeIndex,
        right_child_id: NodeIndex,
        split_indices: &[bool],
    ) -> Result<()> {
        let parent_indices = self
            .leaf_indices
            .remove(&leaf_id)
            .ok_or_else(|| LightGBMError::internal(format!("Leaf {} not found", leaf_id)))?;

        if parent_indices.len() != split_indices.len() {
            return Err(LightGBMError::dimension_mismatch(
                format!("parent_indices: {}", parent_indices.len()),
                format!("split_indices: {}", split_indices.len()),
            ));
        }

        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for (i, &data_idx) in parent_indices.iter().enumerate() {
            if split_indices[i] {
                // Goes to left child
                left_indices.push(data_idx);
                self.data_to_leaf[data_idx as usize] = left_child_id;
            } else {
                // Goes to right child
                right_indices.push(data_idx);
                self.data_to_leaf[data_idx as usize] = right_child_id;
            }
        }

        self.leaf_indices.insert(left_child_id, left_indices);
        self.leaf_indices.insert(right_child_id, right_indices);
        self.num_leaves += 1;

        Ok(())
    }

    /// Get data indices for a specific leaf
    pub fn get_leaf_indices(&self, leaf_id: NodeIndex) -> Option<&[DataSize]> {
        self.leaf_indices.get(&leaf_id).map(|v| v.as_slice())
    }

    /// Get the leaf assignment for each data point
    pub fn get_data_to_leaf(&self) -> &Array1<NodeIndex> {
        &self.data_to_leaf
    }

    /// Get the number of data points in a leaf
    pub fn get_leaf_count(&self, leaf_id: NodeIndex) -> usize {
        self.leaf_indices.get(&leaf_id).map_or(0, |v| v.len())
    }

    /// Get all active leaf IDs
    pub fn get_active_leaves(&self) -> Vec<NodeIndex> {
        self.leaf_indices.keys().copied().collect()
    }

    /// Get the number of active leaves
    pub fn num_leaves(&self) -> usize {
        self.leaf_indices.len()
    }

    /// Reset the partition
    pub fn reset(&mut self) {
        self.leaf_indices.clear();
        self.data_to_leaf.fill(0);
        self.num_leaves = 0;
    }

    /// Validate the partition consistency
    pub fn validate(&self) -> Result<()> {
        // Check that all data points are assigned to exactly one leaf
        let mut total_data_points = 0;
        for indices in self.leaf_indices.values() {
            total_data_points += indices.len();
        }

        if total_data_points != self.num_data as usize {
            return Err(LightGBMError::internal(format!(
                "Partition inconsistency: expected {} data points, found {}",
                self.num_data, total_data_points
            )));
        }

        // Check that data_to_leaf assignments are consistent
        for (&leaf_id, indices) in &self.leaf_indices {
            for &data_idx in indices {
                if self.data_to_leaf[data_idx as usize] != leaf_id {
                    return Err(LightGBMError::internal(format!(
                        "Inconsistent leaf assignment for data point {}: partition says {}, array says {}",
                        data_idx, leaf_id, self.data_to_leaf[data_idx as usize]
                    )));
                }
            }
        }

        Ok(())
    }

    /// Get memory usage statistics
    pub fn memory_usage(&self) -> PartitionMemoryStats {
        let indices_memory = self
            .leaf_indices
            .values()
            .map(|v| v.capacity() * std::mem::size_of::<DataSize>())
            .sum::<usize>();

        let data_to_leaf_memory = self.data_to_leaf.len() * std::mem::size_of::<NodeIndex>();
        let hashmap_memory = self.leaf_indices.capacity()
            * (std::mem::size_of::<NodeIndex>() + std::mem::size_of::<Vec<DataSize>>());

        PartitionMemoryStats {
            total_bytes: indices_memory + data_to_leaf_memory + hashmap_memory,
            indices_bytes: indices_memory,
            data_to_leaf_bytes: data_to_leaf_memory,
            hashmap_bytes: hashmap_memory,
        }
    }
}

/// Memory usage statistics for data partition
#[derive(Debug, Clone)]
pub struct PartitionMemoryStats {
    /// Total memory usage in bytes
    pub total_bytes: usize,
    /// Memory used by leaf indices
    pub indices_bytes: usize,
    /// Memory used by data-to-leaf mapping
    pub data_to_leaf_bytes: usize,
    /// Memory used by HashMap overhead
    pub hashmap_bytes: usize,
}

/// Configuration for data partitioning
#[derive(Debug, Clone)]
pub struct PartitionConfig {
    /// Initial capacity for leaf indices
    pub initial_leaf_capacity: usize,
    /// Enable validation checks
    pub enable_validation: bool,
    /// Memory optimization mode
    pub memory_optimization: MemoryOptimization,
}

/// Memory optimization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOptimization {
    /// No optimization (fastest)
    None,
    /// Shrink vectors after splits (balanced)
    Shrink,
    /// Aggressive memory management (slowest but most memory efficient)
    Aggressive,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        PartitionConfig {
            initial_leaf_capacity: 1000,
            enable_validation: false,
            memory_optimization: MemoryOptimization::Shrink,
        }
    }
}

impl Default for MemoryOptimization {
    fn default() -> Self {
        MemoryOptimization::Shrink
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_creation() {
        let partition = DataPartition::new(100);
        assert_eq!(partition.num_data, 100);
        assert_eq!(partition.num_leaves(), 1);
        assert_eq!(partition.get_leaf_count(0), 100);
    }

    #[test]
    fn test_partition_split() {
        let mut partition = DataPartition::new(4);

        // Split root node (0) into children (1, 2)
        let split_indices = vec![true, false, true, false]; // data 0,2 go left, 1,3 go right
        partition.split_leaf(0, 1, 2, &split_indices).unwrap();

        assert_eq!(partition.num_leaves(), 2);
        assert_eq!(partition.get_leaf_count(1), 2); // left child
        assert_eq!(partition.get_leaf_count(2), 2); // right child

        let left_indices = partition.get_leaf_indices(1).unwrap();
        assert_eq!(left_indices, &[0, 2]);

        let right_indices = partition.get_leaf_indices(2).unwrap();
        assert_eq!(right_indices, &[1, 3]);
    }

    #[test]
    fn test_partition_validation() {
        let mut partition = DataPartition::new(10);
        assert!(partition.validate().is_ok());

        let split_indices = vec![true; 10];
        partition.split_leaf(0, 1, 2, &split_indices).unwrap();
        assert!(partition.validate().is_ok());
    }

    #[test]
    fn test_partition_reset() {
        let mut partition = DataPartition::new(10);
        let split_indices = vec![
            true, false, true, false, true, false, true, false, true, false,
        ];
        partition.split_leaf(0, 1, 2, &split_indices).unwrap();

        partition.reset();
        assert_eq!(partition.num_leaves(), 0);
        assert!(partition.get_active_leaves().is_empty());
    }

    #[test]
    fn test_memory_stats() {
        let partition = DataPartition::new(1000);
        let stats = partition.memory_usage();
        assert!(stats.total_bytes > 0);
        assert!(stats.indices_bytes > 0);
        assert!(stats.data_to_leaf_bytes > 0);
    }

    #[test]
    fn test_partition_config() {
        let config = PartitionConfig::default();
        assert_eq!(config.initial_leaf_capacity, 1000);
        assert!(!config.enable_validation);
        assert_eq!(config.memory_optimization, MemoryOptimization::Shrink);
    }
}
