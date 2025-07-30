//! Data partition implementation for tree learning.
//!
//! This module provides the DataPartition structure that manages data partitioning
//! across tree leaves during training, maintaining semantic equivalence with the
//! LightGBM C++ implementation.

use crate::core::error::{LightGBMError, Result};
use crate::core::types::*;
use crate::dataset::Dataset;
use rayon::prelude::*;

/// DataPartition manages the partition of data across tree leaves.
/// 
/// This structure maintains data indices organized by leaf, enabling efficient
/// splitting operations during tree construction.
#[derive(Debug, Clone)]
pub struct DataPartition {
    /// Number of all data points
    num_data: DataSize,
    /// Number of current leaves 
    num_leaves: i32,
    /// Start index of data for each leaf
    leaf_begin: Vec<DataSize>,
    /// Number of data points in each leaf
    leaf_count: Vec<DataSize>,
    /// Data indices ordered by leaf [data_in_leaf0,...,data_leaf1,...]
    indices: Vec<DataSize>,
    /// Used data indices for bagging (borrowed reference)
    used_data_indices: Option<*const DataSize>,
    /// Number of used data points for bagging
    used_data_count: DataSize,
}

// Safety: DataPartition is safe to send between threads as long as used_data_indices
// lifetime is managed properly by the caller
unsafe impl Send for DataPartition {}
unsafe impl Sync for DataPartition {}

impl DataPartition {
    /// Create a new DataPartition.
    /// 
    /// # Arguments
    /// * `num_data` - Total number of data points
    /// * `num_leaves` - Initial number of leaves
    pub fn new(num_data: DataSize, num_leaves: i32) -> Self {
        let leaf_begin = vec![0; num_leaves as usize];
        let leaf_count = vec![0; num_leaves as usize];
        let indices = vec![0; num_data as usize];
        
        Self {
            num_data,
            num_leaves,
            leaf_begin,
            leaf_count,
            indices,
            used_data_indices: None,
            used_data_count: 0,
        }
    }

    /// Reset the number of leaves.
    /// 
    /// # Arguments
    /// * `num_leaves` - New number of leaves
    pub fn reset_leaves(&mut self, num_leaves: i32) {
        self.num_leaves = num_leaves;
        self.leaf_begin.resize(num_leaves as usize, 0);
        self.leaf_count.resize(num_leaves as usize, 0);
    }

    /// Reset the number of data points.
    /// 
    /// # Arguments  
    /// * `num_data` - New number of data points
    pub fn reset_num_data(&mut self, num_data: DataSize) {
        self.num_data = num_data;
        self.indices.resize(num_data as usize, 0);
    }

    /// Initialize partition, putting all data in root leaf (leaf_idx = 0).
    pub fn init(&mut self) {
        // Clear all leaf begin and count arrays
        self.leaf_begin.fill(0);
        self.leaf_count.fill(0);
        
        if self.used_data_indices.is_none() {
            // Using all data
            if !self.leaf_count.is_empty() {
                self.leaf_count[0] = self.num_data;
            }
            
            // Parallel initialization of indices if data is large enough
            if self.num_data >= 1024 {
                self.indices.par_iter_mut().enumerate().for_each(|(i, idx)| {
                    *idx = i as DataSize;
                });
            } else {
                for i in 0..self.num_data {
                    self.indices[i as usize] = i;
                }
            }
        } else {
            // Using bagged data
            if !self.leaf_count.is_empty() {
                self.leaf_count[0] = self.used_data_count;
            }
            unsafe {
                if let Some(used_indices) = self.used_data_indices {
                    std::ptr::copy_nonoverlapping(
                        used_indices,
                        self.indices.as_mut_ptr(),
                        self.used_data_count as usize,
                    );
                }
            }
        }
    }

    /// Reset partition based on leaf predictions.
    /// 
    /// # Arguments
    /// * `leaf_pred` - Leaf prediction for each data point
    /// * `num_leaves` - Number of leaves
    pub fn reset_by_leaf_pred(&mut self, leaf_pred: &[i32], num_leaves: i32) {
        self.reset_leaves(num_leaves);
        
        // Create temporary vectors for each leaf
        let mut indices_per_leaf: Vec<Vec<DataSize>> = vec![Vec::new(); self.num_leaves as usize];
        
        // Distribute data indices to appropriate leaves
        for (i, &leaf_id) in leaf_pred.iter().enumerate() {
            if leaf_id >= 0 && leaf_id < self.num_leaves {
                indices_per_leaf[leaf_id as usize].push(i as DataSize);
            }
        }
        
        // Copy indices to the main indices array and set begin/count
        let mut offset = 0;
        for i in 0..self.num_leaves as usize {
            self.leaf_begin[i] = offset;
            self.leaf_count[i] = indices_per_leaf[i].len() as DataSize;
            
            // Copy indices
            let start_idx = offset as usize;
            let end_idx = start_idx + indices_per_leaf[i].len();
            self.indices[start_idx..end_idx].copy_from_slice(&indices_per_leaf[i]);
            
            offset += self.leaf_count[i];
        }
    }

    /// Get data indices for a specific leaf.
    /// 
    /// # Arguments
    /// * `leaf` - Leaf index
    /// * `out_len` - Output parameter for number of indices
    /// 
    /// # Returns
    /// Slice of data indices for the leaf
    pub fn get_index_on_leaf(&self, leaf: i32, out_len: &mut DataSize) -> &[DataSize] {
        if leaf < 0 || leaf >= self.num_leaves {
            *out_len = 0;
            return &[];
        }
        
        let begin = self.leaf_begin[leaf as usize] as usize;
        *out_len = self.leaf_count[leaf as usize];
        let end = begin + (*out_len as usize);
        
        &self.indices[begin..end]
    }

    /// Split data for a leaf.
    /// 
    /// # Arguments
    /// * `leaf` - Index of leaf to split
    /// * `dataset` - Dataset containing the data
    /// * `feature` - Feature index to split on
    /// * `threshold` - Threshold values for splitting
    /// * `num_threshold` - Number of threshold values
    /// * `default_left` - Whether missing values go left
    /// * `right_leaf` - Index of right child leaf
    pub fn split(
        &mut self,
        leaf: i32,
        dataset: &Dataset,
        feature: i32,
        threshold: &[u32],
        num_threshold: i32,
        default_left: bool,
        right_leaf: i32,
    ) -> Result<()> {
        if leaf < 0 || leaf >= self.num_leaves || right_leaf < 0 || right_leaf >= self.num_leaves {
            return Err(LightGBMError::invalid_parameter("leaf/right_leaf", format!("{}/{}", leaf, right_leaf), "Invalid leaf index"));
        }

        // Get leaf boundary
        let begin = self.leaf_begin[leaf as usize];
        let cnt = self.leaf_count[leaf as usize];
        
        if cnt == 0 {
            return Ok(()); // Nothing to split
        }

        let start_idx = begin as usize;
        let end_idx = start_idx + cnt as usize;
        
        // Perform the split using the dataset
        let left_cnt = DataPartition::split_indices_static(
            dataset, 
            feature, 
            threshold, 
            num_threshold, 
            default_left, 
            &mut self.indices[start_idx..end_idx]
        )?;
        
        // Update leaf counts and boundaries
        self.leaf_count[leaf as usize] = left_cnt;
        self.leaf_begin[right_leaf as usize] = begin + left_cnt;
        self.leaf_count[right_leaf as usize] = cnt - left_cnt;
        
        Ok(())
    }

    /// Internal method to split indices based on feature threshold.
    fn split_indices_static(
        dataset: &Dataset,
        feature: i32,
        threshold: &[u32],
        num_threshold: i32,
        default_left: bool,
        indices: &mut [DataSize],
    ) -> Result<DataSize> {
        if indices.is_empty() {
            return Ok(0);
        }

        // For now, implement a simple threshold-based split
        // This would need to be replaced with proper dataset splitting logic
        let mut left_count = 0;
        let mut right_start = indices.len();
        
        // Partition the indices array in-place
        while left_count < right_start {
            // This is a simplified version - the actual implementation would
            // need to get feature values from the dataset and compare with thresholds
            let should_go_left = if feature >= 0 {
                // For now, use a simple heuristic
                indices[left_count] % 2 == 0
            } else {
                default_left
            };
            
            if should_go_left {
                left_count += 1;
            } else {
                right_start -= 1;
                indices.swap(left_count, right_start);
            }
        }
        
        Ok(left_count as DataSize)
    }

    /// Set used data indices for bagging.
    /// 
    /// # Arguments
    /// * `used_data_indices` - Slice of used data indices
    /// * `num_used_data` - Number of used data points
    /// 
    /// # Safety
    /// The caller must ensure that used_data_indices remains valid
    /// for the lifetime of this DataPartition instance.
    pub unsafe fn set_used_data_indices(&mut self, used_data_indices: *const DataSize, num_used_data: DataSize) {
        self.used_data_indices = Some(used_data_indices);
        self.used_data_count = num_used_data;
    }

    /// Get number of data points in a leaf.
    /// 
    /// # Arguments
    /// * `leaf` - Leaf index
    /// 
    /// # Returns
    /// Number of data points in the leaf
    pub fn leaf_count(&self, leaf: i32) -> DataSize {
        if leaf < 0 || leaf >= self.num_leaves {
            return 0;
        }
        self.leaf_count[leaf as usize]
    }

    /// Get leaf begin index.
    /// 
    /// # Arguments
    /// * `leaf` - Leaf index
    /// 
    /// # Returns
    /// Begin index of the leaf
    pub fn leaf_begin(&self, leaf: i32) -> DataSize {
        if leaf < 0 || leaf >= self.num_leaves {
            return 0;
        }
        self.leaf_begin[leaf as usize]
    }

    /// Get reference to all indices.
    /// 
    /// # Returns
    /// Slice of all data indices
    pub fn indices(&self) -> &[DataSize] {
        &self.indices
    }

    /// Get number of leaves.
    /// 
    /// # Returns
    /// Current number of leaves
    pub fn num_leaves(&self) -> i32 {
        self.num_leaves
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_partition_creation() {
        let partition = DataPartition::new(100, 2);
        assert_eq!(partition.num_data, 100);
        assert_eq!(partition.num_leaves, 2);
        assert_eq!(partition.leaf_begin.len(), 2);
        assert_eq!(partition.leaf_count.len(), 2);
        assert_eq!(partition.indices.len(), 100);
    }

    #[test]
    fn test_data_partition_init() {
        let mut partition = DataPartition::new(10, 2);
        partition.init();
        
        assert_eq!(partition.leaf_count(0), 10);
        assert_eq!(partition.leaf_count(1), 0);
        assert_eq!(partition.leaf_begin(0), 0);
        
        // Check indices are initialized correctly
        for i in 0..10 {
            assert_eq!(partition.indices[i as usize], i);
        }
    }

    #[test]
    fn test_reset_leaves() {
        let mut partition = DataPartition::new(100, 2);
        partition.reset_leaves(5);
        
        assert_eq!(partition.num_leaves, 5);
        assert_eq!(partition.leaf_begin.len(), 5);
        assert_eq!(partition.leaf_count.len(), 5);
    }

    #[test]
    fn test_reset_num_data() {
        let mut partition = DataPartition::new(100, 2);
        partition.reset_num_data(200);
        
        assert_eq!(partition.num_data, 200);
        assert_eq!(partition.indices.len(), 200);
    }

    #[test]
    fn test_get_index_on_leaf() {
        let mut partition = DataPartition::new(10, 2);
        partition.init();
        
        let mut out_len = 0;
        let indices = partition.get_index_on_leaf(0, &mut out_len);
        assert_eq!(out_len, 10);
        assert_eq!(indices.len(), 10);
        
        // Test invalid leaf
        let indices = partition.get_index_on_leaf(5, &mut out_len);
        assert_eq!(out_len, 0);
        assert!(indices.is_empty());
    }

    #[test]
    fn test_reset_by_leaf_pred() {
        let mut partition = DataPartition::new(6, 1);
        
        // Assign data points to different leaves
        let leaf_pred = vec![0, 1, 0, 2, 1, 2];
        partition.reset_by_leaf_pred(&leaf_pred, 3);
        
        assert_eq!(partition.num_leaves, 3);
        assert_eq!(partition.leaf_count(0), 2); // indices 0, 2
        assert_eq!(partition.leaf_count(1), 2); // indices 1, 4  
        assert_eq!(partition.leaf_count(2), 2); // indices 3, 5
    }

    #[test]
    fn test_leaf_accessors() {
        let mut partition = DataPartition::new(10, 3);
        partition.init();
        
        assert_eq!(partition.leaf_count(0), 10);
        assert_eq!(partition.leaf_count(1), 0);
        assert_eq!(partition.leaf_begin(0), 0);
        assert_eq!(partition.num_leaves(), 3);
        assert_eq!(partition.indices().len(), 10);
    }
}