//! Leaf split candidate finding for Pure Rust LightGBM.
//!
//! This module provides the LeafSplits structure used to find split candidates
//! for a leaf during tree learning. It maintains gradient and hessian sums
//! and data indices for efficient split evaluation.

use crate::config::core::Config;
use crate::core::types::*;
use crate::dataset::partition::DataPartition;
use rayon::prelude::*;

/// Structure used to find split candidates for a leaf.
/// 
/// This is equivalent to the C++ LeafSplits class and provides multiple
/// initialization methods for different data configurations and scenarios.
#[derive(Debug, Clone)]
pub struct LeafSplits {
    /// Whether to force deterministic behavior
    deterministic: bool,
    /// Current leaf index
    leaf_index: i32,
    /// Number of data points in current leaf
    num_data_in_leaf: DataSize,
    /// Total number of data points
    num_data: DataSize,
    /// Sum of gradients in the current leaf
    sum_gradients: f64,
    /// Sum of hessians in the current leaf
    sum_hessians: f64,
    /// Sum of discretized gradients and hessians (for quantized mode)
    int_sum_gradients_and_hessians: i64,
    /// Indices of data points in the current leaf
    data_indices: Option<Vec<DataSize>>,
    /// Weight of the current leaf
    weight: f64,
}

impl LeafSplits {
    /// Create a new LeafSplits instance.
    /// 
    /// # Arguments
    /// * `num_data` - Total number of data points
    /// * `config` - Optional configuration (for deterministic flag)
    pub fn new(num_data: DataSize, config: Option<&Config>) -> Self {
        let deterministic = config.map_or(false, |c| c.deterministic);
        
        Self {
            deterministic,
            leaf_index: 0,
            num_data_in_leaf: num_data,
            num_data,
            sum_gradients: 0.0,
            sum_hessians: 0.0,
            int_sum_gradients_and_hessians: 0,
            data_indices: None,
            weight: 0.0,
        }
    }

    /// Reset the number of data points.
    pub fn reset_num_data(&mut self, num_data: DataSize) {
        self.num_data = num_data;
        self.num_data_in_leaf = num_data;
    }

    /// Initialize split on current leaf with partial data and computed sums.
    /// 
    /// # Arguments
    /// * `leaf` - Index of current leaf
    /// * `data_partition` - Current data partition
    /// * `sum_gradients` - Precomputed sum of gradients
    /// * `sum_hessians` - Precomputed sum of hessians
    /// * `weight` - Weight of the leaf
    pub fn init_with_sums(
        &mut self,
        leaf: i32,
        data_partition: &DataPartition,
        sum_gradients: f64,
        sum_hessians: f64,
        weight: f64,
    ) {
        self.leaf_index = leaf;
        if let Some(indices) = data_partition.get_leaf_indices(leaf as NodeIndex) {
            self.data_indices = Some(indices.to_vec());
            self.num_data_in_leaf = indices.len() as DataSize;
        } else {
            self.data_indices = None;
            self.num_data_in_leaf = 0;
        }
        self.sum_gradients = sum_gradients;
        self.sum_hessians = sum_hessians;
        self.weight = weight;
    }

    /// Initialize split on current leaf with partial data and discretized sums.
    /// 
    /// # Arguments
    /// * `leaf` - Index of current leaf
    /// * `data_partition` - Current data partition
    /// * `sum_gradients` - Precomputed sum of gradients
    /// * `sum_hessians` - Precomputed sum of hessians
    /// * `sum_gradients_and_hessians` - Discretized sum of gradients and hessians
    /// * `weight` - Weight of the leaf
    pub fn init_with_discretized_sums(
        &mut self,
        leaf: i32,
        data_partition: &DataPartition,
        sum_gradients: f64,
        sum_hessians: f64,
        sum_gradients_and_hessians: i64,
        weight: f64,
    ) {
        self.leaf_index = leaf;
        if let Some(indices) = data_partition.get_leaf_indices(leaf as NodeIndex) {
            self.data_indices = Some(indices.to_vec());
            self.num_data_in_leaf = indices.len() as DataSize;
        } else {
            self.data_indices = None;
            self.num_data_in_leaf = 0;
        }
        self.sum_gradients = sum_gradients;
        self.sum_hessians = sum_hessians;
        self.int_sum_gradients_and_hessians = sum_gradients_and_hessians;
        self.weight = weight;
    }

    /// Initialize split on current leaf with simple sums.
    /// 
    /// # Arguments
    /// * `leaf` - Index of current leaf
    /// * `sum_gradients` - Sum of gradients
    /// * `sum_hessians` - Sum of hessians
    pub fn init_simple(
        &mut self,
        leaf: i32,
        sum_gradients: f64,
        sum_hessians: f64,
    ) {
        self.leaf_index = leaf;
        self.sum_gradients = sum_gradients;
        self.sum_hessians = sum_hessians;
    }

    /// Initialize splits on the current leaf by traversing all data to sum up results.
    /// 
    /// # Arguments
    /// * `gradients` - Array of gradient values
    /// * `hessians` - Array of hessian values
    pub fn init_from_gradients(
        &mut self,
        gradients: &[Score],
        hessians: &[Score],
    ) {
        self.num_data_in_leaf = self.num_data;
        self.leaf_index = 0;
        self.data_indices = None;

        // Use parallel reduction for large datasets unless deterministic mode is required
        let use_parallel = self.num_data_in_leaf >= 1024 && !self.deterministic;

        let (tmp_sum_gradients, tmp_sum_hessians) = if use_parallel {
            gradients[..self.num_data_in_leaf as usize]
                .par_iter()
                .zip(hessians[..self.num_data_in_leaf as usize].par_iter())
                .map(|(&g, &h)| (g as f64, h as f64))
                .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
        } else {
            let mut sum_g = 0.0;
            let mut sum_h = 0.0;
            for i in 0..self.num_data_in_leaf as usize {
                sum_g += gradients[i] as f64;
                sum_h += hessians[i] as f64;
            }
            (sum_g, sum_h)
        };

        self.sum_gradients = tmp_sum_gradients;
        self.sum_hessians = tmp_sum_hessians;
    }

    /// Initialize splits from discretized gradients and hessians.
    /// 
    /// # Arguments
    /// * `int_gradients_and_hessians` - Discretized gradients and hessians as int8 array
    /// * `grad_scale` - Scaling factor to recover original gradients
    /// * `hess_scale` - Scaling factor to recover original hessians
    pub fn init_from_discretized_gradients(
        &mut self,
        int_gradients_and_hessians: &[i8],
        grad_scale: f64,
        hess_scale: f64,
    ) {
        self.num_data_in_leaf = self.num_data;
        self.leaf_index = 0;
        self.data_indices = None;

        // Use parallel reduction for large datasets
        let use_parallel = self.num_data_in_leaf >= 1024 && !self.deterministic;

        let (tmp_sum_gradients, tmp_sum_hessians, tmp_sum_gradients_and_hessians) = if use_parallel {
            (0..self.num_data_in_leaf as usize)
                .into_par_iter()
                .map(|i| {
                    let grad = int_gradients_and_hessians[2 * i + 1] as f64 * grad_scale;
                    let hess = int_gradients_and_hessians[2 * i] as f64 * hess_scale;
                    
                    // Pack gradients and hessians similar to C++ version
                    let packed_int_grad_and_hess = unsafe {
                        std::mem::transmute::<[i8; 2], i16>([
                            int_gradients_and_hessians[2 * i],
                            int_gradients_and_hessians[2 * i + 1],
                        ])
                    };
                    
                    let packed_long_int_grad_and_hess = 
                        ((packed_int_grad_and_hess as i8 as i64) << 32) |
                        ((packed_int_grad_and_hess & 0x00ff) as i64);
                    
                    (grad, hess, packed_long_int_grad_and_hess)
                })
                .reduce(
                    || (0.0, 0.0, 0i64),
                    |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
                )
        } else {
            let mut sum_g = 0.0;
            let mut sum_h = 0.0;
            let mut sum_packed = 0i64;
            
            for i in 0..self.num_data_in_leaf as usize {
                sum_g += int_gradients_and_hessians[2 * i + 1] as f64 * grad_scale;
                sum_h += int_gradients_and_hessians[2 * i] as f64 * hess_scale;
                
                let packed_int_grad_and_hess = unsafe {
                    std::mem::transmute::<[i8; 2], i16>([
                        int_gradients_and_hessians[2 * i],
                        int_gradients_and_hessians[2 * i + 1],
                    ])
                };
                
                let packed_long_int_grad_and_hess = 
                    ((packed_int_grad_and_hess as i8 as i64) << 32) |
                    ((packed_int_grad_and_hess & 0x00ff) as i64);
                
                sum_packed += packed_long_int_grad_and_hess;
            }
            (sum_g, sum_h, sum_packed)
        };

        self.sum_gradients = tmp_sum_gradients;
        self.sum_hessians = tmp_sum_hessians;
        self.int_sum_gradients_and_hessians = tmp_sum_gradients_and_hessians;
    }

    /// Initialize splits on current leaf of partial data from gradient/hessian arrays.
    /// 
    /// # Arguments
    /// * `leaf` - Index of current leaf
    /// * `data_partition` - Current data partition
    /// * `gradients` - Array of gradient values
    /// * `hessians` - Array of hessian values
    pub fn init_from_partial_gradients(
        &mut self,
        leaf: i32,
        data_partition: &DataPartition,
        gradients: &[Score],
        hessians: &[Score],
    ) {
        self.leaf_index = leaf;
        if let Some(indices) = data_partition.get_leaf_indices(leaf as NodeIndex) {
            self.data_indices = Some(indices.to_vec());
            self.num_data_in_leaf = indices.len() as DataSize;
            
            // Use parallel reduction for large datasets
            let use_parallel = self.num_data_in_leaf >= 1024 && !self.deterministic;
            
            let (tmp_sum_gradients, tmp_sum_hessians) = if use_parallel {
                indices
                    .par_iter()
                    .map(|&idx| (gradients[idx as usize] as f64, hessians[idx as usize] as f64))
                    .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
            } else {
                let mut sum_g = 0.0;
                let mut sum_h = 0.0;
                for &idx in indices {
                    sum_g += gradients[idx as usize] as f64;
                    sum_h += hessians[idx as usize] as f64;
                }
                (sum_g, sum_h)
            };
            
            self.sum_gradients = tmp_sum_gradients;
            self.sum_hessians = tmp_sum_hessians;
        } else {
            self.data_indices = None;
            self.num_data_in_leaf = 0;
            self.sum_gradients = 0.0;
            self.sum_hessians = 0.0;
        }
    }

    /// Initialize splits on current leaf of partial data from discretized gradients.
    /// 
    /// # Arguments
    /// * `leaf` - Index of current leaf
    /// * `data_partition` - Current data partition
    /// * `int_gradients_and_hessians` - Discretized gradients and hessians
    /// * `grad_scale` - Scaling factor for gradients
    /// * `hess_scale` - Scaling factor for hessians
    pub fn init_from_partial_discretized_gradients(
        &mut self,
        leaf: i32,
        data_partition: &DataPartition,
        int_gradients_and_hessians: &[i8],
        grad_scale: Score,
        hess_scale: Score,
    ) {
        self.leaf_index = leaf;
        if let Some(indices) = data_partition.get_leaf_indices(leaf as NodeIndex) {
            self.data_indices = Some(indices.to_vec());
            self.num_data_in_leaf = indices.len() as DataSize;
            
            // Use parallel reduction for large datasets
            let use_parallel = self.num_data_in_leaf >= 1024 && self.deterministic; // Note: using deterministic check as in C++
            
            let (tmp_sum_gradients, tmp_sum_hessians, tmp_sum_gradients_and_hessians) = if use_parallel {
                indices
                    .par_iter()
                    .enumerate()
                    .map(|(i, &idx)| {
                        let grad = int_gradients_and_hessians[2 * idx as usize + 1] as f64 * grad_scale as f64;
                        let hess = int_gradients_and_hessians[2 * idx as usize] as f64 * hess_scale as f64;
                        
                        let packed_int_grad_and_hess = unsafe {
                            std::mem::transmute::<[i8; 2], i16>([
                                int_gradients_and_hessians[2 * i],
                                int_gradients_and_hessians[2 * i + 1],
                            ])
                        };
                        
                        let packed_long_int_grad_and_hess = 
                            ((packed_int_grad_and_hess as i8 as i64) << 32) |
                            ((packed_int_grad_and_hess & 0x00ff) as i64);
                        
                        (grad, hess, packed_long_int_grad_and_hess)
                    })
                    .reduce(
                        || (0.0, 0.0, 0i64),
                        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
                    )
            } else {
                let mut sum_g = 0.0;
                let mut sum_h = 0.0;
                let mut sum_packed = 0i64;
                
                for (i, &idx) in indices.iter().enumerate() {
                    sum_g += int_gradients_and_hessians[2 * idx as usize + 1] as f64 * grad_scale as f64;
                    sum_h += int_gradients_and_hessians[2 * idx as usize] as f64 * hess_scale as f64;
                    
                    let packed_int_grad_and_hess = unsafe {
                        std::mem::transmute::<[i8; 2], i16>([
                            int_gradients_and_hessians[2 * i],
                            int_gradients_and_hessians[2 * i + 1],
                        ])
                    };
                    
                    let packed_long_int_grad_and_hess = 
                        ((packed_int_grad_and_hess as i8 as i64) << 32) |
                        ((packed_int_grad_and_hess & 0x00ff) as i64);
                    
                    sum_packed += packed_long_int_grad_and_hess;
                }
                (sum_g, sum_h, sum_packed)
            };
            
            self.sum_gradients = tmp_sum_gradients;
            self.sum_hessians = tmp_sum_hessians;
            self.int_sum_gradients_and_hessians = tmp_sum_gradients_and_hessians;
        } else {
            self.data_indices = None;
            self.num_data_in_leaf = 0;
            self.sum_gradients = 0.0;
            self.sum_hessians = 0.0;
            self.int_sum_gradients_and_hessians = 0;
        }
    }

    /// Initialize with just gradient and hessian sums.
    /// 
    /// # Arguments
    /// * `sum_gradients` - Sum of gradients
    /// * `sum_hessians` - Sum of hessians
    pub fn init_sums_only(
        &mut self,
        sum_gradients: f64,
        sum_hessians: f64,
    ) {
        self.leaf_index = 0;
        self.sum_gradients = sum_gradients;
        self.sum_hessians = sum_hessians;
    }

    /// Initialize with gradient and hessian sums plus discretized sum.
    /// 
    /// # Arguments
    /// * `sum_gradients` - Sum of gradients
    /// * `sum_hessians` - Sum of hessians
    /// * `int_sum_gradients_and_hessians` - Discretized sum
    pub fn init_sums_with_discretized(
        &mut self,
        sum_gradients: f64,
        sum_hessians: f64,
        int_sum_gradients_and_hessians: i64,
    ) {
        self.leaf_index = 0;
        self.sum_gradients = sum_gradients;
        self.sum_hessians = sum_hessians;
        self.int_sum_gradients_and_hessians = int_sum_gradients_and_hessians;
    }

    /// Initialize empty splits.
    pub fn init_empty(&mut self) {
        self.leaf_index = -1;
        self.data_indices = None;
        self.num_data_in_leaf = 0;
    }

    // Getter methods

    /// Get current leaf index.
    pub fn leaf_index(&self) -> i32 {
        self.leaf_index
    }

    /// Get number of data points in current leaf.
    pub fn num_data_in_leaf(&self) -> DataSize {
        self.num_data_in_leaf
    }

    /// Get sum of gradients of current leaf.
    pub fn sum_gradients(&self) -> f64 {
        self.sum_gradients
    }

    /// Get sum of hessians of current leaf.
    pub fn sum_hessians(&self) -> f64 {
        self.sum_hessians
    }

    /// Get sum of discretized gradients and hessians of current leaf.
    pub fn int_sum_gradients_and_hessians(&self) -> i64 {
        self.int_sum_gradients_and_hessians
    }

    /// Get indices of data in current leaf.
    pub fn data_indices(&self) -> Option<&[DataSize]> {
        self.data_indices.as_deref()
    }

    /// Get weight of current leaf.
    pub fn weight(&self) -> f64 {
        self.weight
    }
}

impl Default for LeafSplits {
    fn default() -> Self {
        Self::new(0, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::core::Config;

    #[test]
    fn test_leaf_splits_creation() {
        let config = Config::default();
        let leaf_splits = LeafSplits::new(100, Some(&config));
        
        assert_eq!(leaf_splits.num_data_in_leaf(), 100);
        assert_eq!(leaf_splits.leaf_index(), 0);
        assert_eq!(leaf_splits.sum_gradients(), 0.0);
        assert_eq!(leaf_splits.sum_hessians(), 0.0);
    }

    #[test]
    fn test_init_from_gradients() {
        let mut leaf_splits = LeafSplits::new(4, None);
        let gradients = vec![1.0, 2.0, 3.0, 4.0];
        let hessians = vec![0.5, 1.0, 1.5, 2.0];
        
        leaf_splits.init_from_gradients(&gradients, &hessians);
        
        assert_eq!(leaf_splits.sum_gradients(), 10.0);
        assert_eq!(leaf_splits.sum_hessians(), 5.0);
        assert_eq!(leaf_splits.num_data_in_leaf(), 4);
    }

    #[test]
    fn test_init_simple() {
        let mut leaf_splits = LeafSplits::new(100, None);
        leaf_splits.init_simple(5, 15.0, 25.0);
        
        assert_eq!(leaf_splits.leaf_index(), 5);
        assert_eq!(leaf_splits.sum_gradients(), 15.0);
        assert_eq!(leaf_splits.sum_hessians(), 25.0);
    }

    #[test]
    fn test_init_sums_only() {
        let mut leaf_splits = LeafSplits::new(100, None);
        leaf_splits.init_sums_only(42.0, 84.0);
        
        assert_eq!(leaf_splits.sum_gradients(), 42.0);
        assert_eq!(leaf_splits.sum_hessians(), 84.0);
        assert_eq!(leaf_splits.leaf_index(), 0);
    }

    #[test]
    fn test_init_empty() {
        let mut leaf_splits = LeafSplits::new(100, None);
        leaf_splits.init_empty();
        
        assert_eq!(leaf_splits.leaf_index(), -1);
        assert_eq!(leaf_splits.num_data_in_leaf(), 0);
        assert!(leaf_splits.data_indices().is_none());
    }

    #[test]
    fn test_reset_num_data() {
        let mut leaf_splits = LeafSplits::new(100, None);
        leaf_splits.reset_num_data(200);
        
        assert_eq!(leaf_splits.num_data_in_leaf(), 200);
    }

    #[test]
    fn test_deterministic_behavior() {
        let mut config = Config::default();
        config.deterministic = true;
        
        let leaf_splits = LeafSplits::new(100, Some(&config));
        assert!(leaf_splits.deterministic);
        
        let leaf_splits_non_det = LeafSplits::new(100, None);
        assert!(!leaf_splits_non_det.deterministic);
    }
}