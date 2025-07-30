//! # FeatureGroup Module
//!
//! This module implements the FeatureGroup struct, which is used to store data and 
//! provide operations on one feature group. It manages binned feature data for 
//! efficient gradient boosting tree construction.
//!
//! Key functionality includes:
//! - Multi-constructor support for different initialization patterns
//! - Binary serialization/deserialization
//! - Data pushing and memory management
//! - Support for both sparse and dense bin representations
//! - Multi-value bin support for categorical features

use crate::core::bin::{Bin, BinIterator, BinMapper, MultiValBin, SPARSE_THRESHOLD as kSparseThreshold};
use crate::core::error::Result;
use crate::core::meta::DataSizeT;
use crate::core::utils::binary_writer::BinaryWriter;

/// FeatureGroup manages binned feature data for one or more features
/// This is the Rust equivalent of the C++ FeatureGroup class
#[derive(Debug)]
pub struct FeatureGroup {
    /// Number of features in this group
    num_feature_: i32,
    
    /// Bin mappers for sub features - equivalent to std::vector<std::unique_ptr<BinMapper>>
    bin_mappers_: Vec<Box<BinMapper>>,
    
    /// Bin offsets for sub features
    bin_offsets_: Vec<u32>,
    
    /// Bin data of this feature (for single-value bins)
    bin_data_: Option<Box<dyn Bin>>,
    
    /// Multi-value bin data (for multi-value bins)
    multi_bin_data_: Vec<Box<dyn Bin>>,
    
    /// True if this feature uses multi-value bins
    is_multi_val_: bool,
    
    /// True if using dense multi-value bins
    is_dense_multi_val_: bool,
    
    /// True if this feature is sparse
    is_sparse_: bool,
    
    /// Total number of bins across all features
    num_total_bin_: i32,
}

impl FeatureGroup {
    /// Constructor for creating a FeatureGroup with multiple features
    /// 
    /// # Arguments
    /// * `num_feature` - Number of features in this group
    /// * `is_multi_val` - Whether to use multi-value bins (0 = false, >0 = true)
    /// * `bin_mappers` - Vector of bin mappers for features
    /// * `num_data` - Total number of data points
    /// * `group_id` - ID of this feature group
    pub fn new(
        num_feature: i32,
        is_multi_val: i8,
        bin_mappers: Vec<Box<BinMapper>>,
        num_data: DataSizeT,
        group_id: i32,
    ) -> Result<Self> {
        if bin_mappers.len() != num_feature as usize {
            return Err(crate::core::error::LightGBMError::Config {
                message: format!(
                    "Expected {} bin mappers, got {}",
                    num_feature,
                    bin_mappers.len()
                )
            });
        }

        let is_multi_val = is_multi_val > 0;
        
        // Calculate sum of sparse rates
        let mut sum_sparse_rate = 0.0;
        for mapper in &bin_mappers {
            sum_sparse_rate += mapper.sparse_rate();
        }
        sum_sparse_rate /= num_feature as f64;
        
        let mut offset = 1;
        let mut is_dense_multi_val = false;
        
        if sum_sparse_rate < <dyn MultiValBin>::MULTI_VAL_BIN_SPARSE_THRESHOLD && is_multi_val {
            // Use dense multi val bin
            offset = 0;
            is_dense_multi_val = true;
        }
        
        // Use bin at zero to store most_freq_bin only when not using dense multi val bin
        let mut num_total_bin = offset;
        
        // However, we should force to leave one bin, if dense multi val bin is the first bin
        // and its first feature has most freq bin > 0
        if group_id == 0 && num_feature > 0 && is_dense_multi_val 
            && bin_mappers[0].get_most_freq_bin() > 0 {
            num_total_bin = 1;
        }
        
        let mut bin_offsets = vec![num_total_bin as u32];
        
        for i in 0..num_feature as usize {
            let mut num_bin = bin_mappers[i].num_bin();
            if bin_mappers[i].get_most_freq_bin() == 0 {
                num_bin -= offset;
            }
            num_total_bin += num_bin;
            bin_offsets.push(num_total_bin as u32);
        }
        
        let mut feature_group = FeatureGroup {
            num_feature_: num_feature,
            bin_mappers_: bin_mappers,
            bin_offsets_: bin_offsets,
            bin_data_: None,
            multi_bin_data_: Vec::new(),
            is_multi_val_: is_multi_val,
            is_dense_multi_val_: is_dense_multi_val,
            is_sparse_: false,
            num_total_bin_: num_total_bin,
        };
        
        feature_group.create_bin_data(num_data, is_multi_val, true, false)?;
        
        Ok(feature_group)
    }
    
    /// Copy constructor for creating a FeatureGroup from another with different num_data
    pub fn from_other(other: &FeatureGroup, num_data: DataSizeT) -> Result<Self> {
        let mut bin_mappers = Vec::new();
        for mapper in &other.bin_mappers_ {
            bin_mappers.push(Box::new((**mapper).clone()));
        }
        
        let mut feature_group = FeatureGroup {
            num_feature_: other.num_feature_,
            is_multi_val_: other.is_multi_val_,
            is_dense_multi_val_: other.is_dense_multi_val_,
            is_sparse_: other.is_sparse_,
            num_total_bin_: other.num_total_bin_,
            bin_offsets_: other.bin_offsets_.clone(),
            bin_mappers_: bin_mappers,
            bin_data_: None,
            multi_bin_data_: Vec::new(),
        };
        
        feature_group.create_bin_data(num_data, other.is_multi_val_, !other.is_sparse_, other.is_sparse_)?;
        
        Ok(feature_group)
    }
    
    /// Constructor for single feature FeatureGroup
    pub fn new_single_feature(
        bin_mappers: Vec<Box<BinMapper>>,
        num_data: DataSizeT,
    ) -> Result<Self> {
        if bin_mappers.len() != 1 {
            return Err(crate::core::error::LightGBMError::Config {
                message: format!(
                    "Expected 1 bin mapper for single feature, got {}",
                    bin_mappers.len()
                )
            });
        }
        
        // Use bin at zero to store default_bin
        let mut num_total_bin = 1;
        let is_dense_multi_val = false;
        let mut bin_offsets = vec![num_total_bin as u32];
        
        let mut num_bin = bin_mappers[0].num_bin();
        if bin_mappers[0].get_most_freq_bin() == 0 {
            num_bin -= 1;
        }
        num_total_bin += num_bin as i32;
        bin_offsets.push(num_total_bin as u32);
        
        let mut feature_group = FeatureGroup {
            num_feature_: 1,
            bin_mappers_: bin_mappers,
            bin_offsets_: bin_offsets,
            bin_data_: None,
            multi_bin_data_: Vec::new(),
            is_multi_val_: false,
            is_dense_multi_val_: is_dense_multi_val,
            is_sparse_: false,
            num_total_bin_: num_total_bin,
        };
        
        feature_group.create_bin_data(num_data, false, false, false)?;
        
        Ok(feature_group)
    }
    
    /// Load definition from memory without data
    pub fn from_memory_definition(
        memory: &[u8],
        num_data: DataSizeT,
        group_id: i32,
    ) -> Result<Self> {
        let mut feature_group = FeatureGroup {
            num_feature_: 0,
            bin_mappers_: Vec::new(),
            bin_offsets_: Vec::new(),
            bin_data_: None,
            multi_bin_data_: Vec::new(),
            is_multi_val_: false,
            is_dense_multi_val_: false,
            is_sparse_: false,
            num_total_bin_: 0,
        };
        
        feature_group.load_definition_from_memory(memory, group_id)?;
        feature_group.allocate_bins(num_data)?;
        
        Ok(feature_group)
    }
    
    /// Load definition from memory with data
    pub fn from_memory_with_data(
        memory: &[u8], 
        num_all_data: DataSizeT,
        local_used_indices: &[DataSizeT],
        group_id: i32,
    ) -> Result<Self> {
        let mut feature_group = FeatureGroup {
            num_feature_: 0,
            bin_mappers_: Vec::new(),
            bin_offsets_: Vec::new(),
            bin_data_: None,
            multi_bin_data_: Vec::new(),
            is_multi_val_: false,
            is_dense_multi_val_: false,
            is_sparse_: false,
            num_total_bin_: 0,
        };
        
        // Load the definition schema first
        let memory_ptr = feature_group.load_definition_from_memory(memory, group_id)?;
        let remaining_memory = &memory[memory_ptr..];
        
        // Allocate memory for the data
        let num_data = if local_used_indices.is_empty() {
            num_all_data
        } else {
            local_used_indices.len() as DataSizeT
        };
        
        feature_group.allocate_bins(num_data)?;
        
        // Now load the actual data
        let mut data_offset = 0;
        if feature_group.is_multi_val_ {
            for i in 0..feature_group.num_feature_ as usize {
                feature_group.multi_bin_data_[i].load_from_memory(
                    &remaining_memory[data_offset..], 
                    local_used_indices
                )?;
                data_offset += feature_group.multi_bin_data_[i].sizes_in_byte();
            }
        } else if let Some(ref mut bin_data) = feature_group.bin_data_ {
            bin_data.load_from_memory(remaining_memory, local_used_indices)?;
        }
        
        Ok(feature_group)
    }
    
    /// Private helper to create bin data
    fn create_bin_data(
        &mut self,
        num_data: DataSizeT,
        is_multi_val: bool,
        force_dense: bool,
        force_sparse: bool,
    ) -> Result<()> {
        if is_multi_val {
            self.multi_bin_data_.clear();
            for i in 0..self.num_feature_ as usize {
                let addi = if self.bin_mappers_[i].get_most_freq_bin() == 0 { 0 } else { 1 };
                let num_bin = self.bin_mappers_[i].num_bin() + addi;
                
                if self.bin_mappers_[i].sparse_rate() >= kSparseThreshold {
                    self.multi_bin_data_.push(<dyn Bin>::create_sparse_bin(num_data, num_bin as i32));
                } else {
                    self.multi_bin_data_.push(<dyn Bin>::create_dense_bin(num_data, num_bin as i32));
                }
            }
            self.is_multi_val_ = true;
        } else {
            if force_sparse || 
               (!force_dense && self.num_feature_ == 1 && 
                self.bin_mappers_[0].sparse_rate() >= kSparseThreshold) {
                self.is_sparse_ = true;
                self.bin_data_ = Some(<dyn Bin>::create_sparse_bin(num_data, self.num_total_bin_));
            } else {
                self.is_sparse_ = false;
                self.bin_data_ = Some(<dyn Bin>::create_dense_bin(num_data, self.num_total_bin_));
            }
            self.is_multi_val_ = false;
        }
        Ok(())
    }
    
    /// Load the overall definition of the feature group from binary serialized data
    fn load_definition_from_memory(&mut self, memory: &[u8], group_id: i32) -> Result<usize> {
        let mut offset = 0;
        
        // Read is_multi_val_
        if memory.len() < offset + std::mem::size_of::<bool>() {
            return Err(crate::core::error::LightGBMError::Dataset {
                message: "Memory buffer too small".to_string()
            });
        }
        self.is_multi_val_ = memory[offset] != 0;
        offset += crate::core::utils::binary_writer::VecBinaryWriter::aligned_size(std::mem::size_of::<bool>(), 8);
        
        // Read is_dense_multi_val_
        if memory.len() < offset + std::mem::size_of::<bool>() {
            return Err(crate::core::error::LightGBMError::Dataset {
                message: "Memory buffer too small".to_string()
            });
        }
        self.is_dense_multi_val_ = memory[offset] != 0;
        offset += crate::core::utils::binary_writer::VecBinaryWriter::aligned_size(std::mem::size_of::<bool>(), 8);
        
        // Read is_sparse_
        if memory.len() < offset + std::mem::size_of::<bool>() {
            return Err(crate::core::error::LightGBMError::Dataset {
                message: "Memory buffer too small".to_string()
            });
        }
        self.is_sparse_ = memory[offset] != 0;
        offset += crate::core::utils::binary_writer::VecBinaryWriter::aligned_size(std::mem::size_of::<bool>(), 8);
        
        // Read num_feature_
        if memory.len() < offset + std::mem::size_of::<i32>() {
            return Err(crate::core::error::LightGBMError::Dataset {
                message: "Memory buffer too small".to_string()
            });
        }
        self.num_feature_ = i32::from_le_bytes([
            memory[offset], memory[offset + 1], memory[offset + 2], memory[offset + 3]
        ]);
        offset += crate::core::utils::binary_writer::VecBinaryWriter::aligned_size(std::mem::size_of::<i32>(), 8);
        
        // Get bin mappers
        self.bin_mappers_.clear();
        for _ in 0..self.num_feature_ {
            let mapper = BinMapper::from_memory(&memory[offset..])?;
            let size = mapper.sizes_in_byte();
            self.bin_mappers_.push(Box::new(mapper));
            offset += size;
        }
        
        // Calculate bin offsets
        self.bin_offsets_.clear();
        let offset_val = if self.is_dense_multi_val_ { 0 } else { 1 };
        
        // Use bin at zero to store most_freq_bin only when not using dense multi val bin
        self.num_total_bin_ = offset_val;
        
        // However, we should force to leave one bin, if dense multi val bin is the first bin
        // and its first feature has most freq bin > 0
        if group_id == 0 && self.num_feature_ > 0 && self.is_dense_multi_val_ 
            && self.bin_mappers_[0].get_most_freq_bin() > 0 {
            self.num_total_bin_ = 1;
        }
        
        self.bin_offsets_.push(self.num_total_bin_ as u32);
        
        for i in 0..self.num_feature_ as usize {
            let mut num_bin = self.bin_mappers_[i].num_bin() as i32;
            if self.bin_mappers_[i].get_most_freq_bin() == 0 {
                num_bin -= offset_val;
            }
            self.num_total_bin_ += num_bin;
            self.bin_offsets_.push(self.num_total_bin_ as u32);
        }
        
        Ok(offset)
    }
    
    /// Allocate the bins
    fn allocate_bins(&mut self, num_data: DataSizeT) -> Result<()> {
        if self.is_multi_val_ {
            self.multi_bin_data_.clear();
            for i in 0..self.num_feature_ as usize {
                let addi = if self.bin_mappers_[i].get_most_freq_bin() == 0 { 0 } else { 1 };
                let num_bin = self.bin_mappers_[i].num_bin() + addi;
                
                if self.bin_mappers_[i].sparse_rate() >= kSparseThreshold {
                    self.multi_bin_data_.push(<dyn Bin>::create_sparse_bin(num_data, num_bin as i32));
                } else {
                    self.multi_bin_data_.push(<dyn Bin>::create_dense_bin(num_data, num_bin as i32));
                }
            }
        } else {
            if self.is_sparse_ {
                self.bin_data_ = Some(<dyn Bin>::create_sparse_bin(num_data, self.num_total_bin_));
            } else {
                self.bin_data_ = Some(<dyn Bin>::create_dense_bin(num_data, self.num_total_bin_));
            }
        }
        Ok(())
    }
}

// Implement basic getters and utility methods
impl FeatureGroup {
    /// Get the number of features in this group
    pub fn num_feature(&self) -> i32 {
        self.num_feature_
    }
    
    /// Check if this feature group uses multi-value bins
    pub fn is_multi_val(&self) -> bool {
        self.is_multi_val_
    }
    
    /// Check if this feature group uses dense multi-value bins
    pub fn is_dense_multi_val(&self) -> bool {
        self.is_dense_multi_val_
    }
    
    /// Check if this feature group is sparse
    pub fn is_sparse(&self) -> bool {
        self.is_sparse_
    }
    
    /// Get the total number of bins
    pub fn num_total_bin(&self) -> i32 {
        self.num_total_bin_
    }
    
    /// Get bin offsets
    pub fn bin_offsets(&self) -> &[u32] {
        &self.bin_offsets_
    }
    
    /// Get bin mapper for a specific sub-features
    pub fn bin_mapper(&self, index: usize) -> Option<&BinMapper> {
        self.bin_mappers_.get(index).map(|b| b.as_ref())
    }

    /// Initialize for pushing in a streaming fashion
    pub fn init_streaming(&mut self, num_thread: i32, omp_max_threads: i32) {
        if self.is_multi_val_ {
            for i in 0..self.num_feature_ as usize {
                self.multi_bin_data_[i].init_streaming(num_thread as u32, omp_max_threads);
            }
        } else if let Some(ref mut bin_data) = self.bin_data_ {
            bin_data.init_streaming(num_thread as u32, omp_max_threads);
        }
    }

    /// Push one record, will auto convert to bin and push to bin data
    pub fn push_data(&mut self, tid: i32, sub_feature_idx: i32, line_idx: DataSizeT, value: f64) {
        let bin = self.bin_mappers_[sub_feature_idx as usize].value_to_bin(value);
        let most_freq_bin = self.bin_mappers_[sub_feature_idx as usize].get_most_freq_bin();
        
        if bin == most_freq_bin {
            return;
        }
        
        let mut adjusted_bin = bin;
        if most_freq_bin == 0 {
            adjusted_bin = bin.saturating_sub(1);
        }
        
        if self.is_multi_val_ {
            self.multi_bin_data_[sub_feature_idx as usize].push(tid, line_idx, adjusted_bin + 1);
        } else if let Some(ref mut bin_data) = self.bin_data_ {
            let final_bin = adjusted_bin + self.bin_offsets_[sub_feature_idx as usize];
            bin_data.push(tid, line_idx, final_bin);
        }
    }

    /// Resize the bin data
    pub fn resize(&mut self, num_data: DataSizeT) {
        if !self.is_multi_val_ {
            if let Some(ref mut bin_data) = self.bin_data_ {
                bin_data.resize(num_data);
            }
        } else {
            for i in 0..self.num_feature_ as usize {
                self.multi_bin_data_[i].resize(num_data);
            }
        }
    }

    /// Copy subrow from another FeatureGroup
    pub fn copy_subrow(&mut self, full_feature: &FeatureGroup, used_indices: &[DataSizeT], num_used_indices: DataSizeT) {
        if !self.is_multi_val_ {
            if let (Some(ref mut bin_data), Some(ref full_bin_data)) = (&mut self.bin_data_, &full_feature.bin_data_) {
                bin_data.copy_subrow(full_bin_data.as_ref(), used_indices, num_used_indices);
            }
        } else {
            for i in 0..self.num_feature_ as usize {
                self.multi_bin_data_[i].copy_subrow(
                    full_feature.multi_bin_data_[i].as_ref(), 
                    used_indices,
                    num_used_indices
                );
            }
        }
    }

    /// Copy subrow by column from another FeatureGroup
    pub fn copy_subrow_by_col(&mut self, full_feature: &FeatureGroup, used_indices: &[DataSizeT], num_used_indices: DataSizeT, fidx: i32) {
        if !self.is_multi_val_ {
            if let (Some(ref mut bin_data), Some(ref full_bin_data)) = (&mut self.bin_data_, &full_feature.bin_data_) {
                bin_data.copy_subrow(full_bin_data.as_ref(), used_indices, num_used_indices);
            }
        } else {
            self.multi_bin_data_[fidx as usize].copy_subrow(
                full_feature.multi_bin_data_[fidx as usize].as_ref(), 
                used_indices,
                num_used_indices
            );
        }
    }

    /// Add features from another FeatureGroup
    pub fn add_features_from(&mut self, other: &FeatureGroup, group_id: i32) -> Result<()> {
        if !self.is_multi_val_ || !other.is_multi_val_ {
            return Err(crate::core::error::LightGBMError::Config {
                message: "Both feature groups must use multi-value bins".to_string()
            });
        }

        // Every time when new features are added, we need to reconsider sparse or dense
        let mut sum_sparse_rate = 0.0;
        for i in 0..self.num_feature_ as usize {
            sum_sparse_rate += self.bin_mappers_[i].sparse_rate();
        }
        for i in 0..other.num_feature_ as usize {
            sum_sparse_rate += other.bin_mappers_[i].sparse_rate();
        }
        sum_sparse_rate /= (self.num_feature_ + other.num_feature_) as f64;
        
        let mut offset = 1;
        self.is_dense_multi_val_ = false;
        if sum_sparse_rate < <dyn MultiValBin>::MULTI_VAL_BIN_SPARSE_THRESHOLD && self.is_multi_val_ {
            // Use dense multi val bin
            offset = 0;
            self.is_dense_multi_val_ = true;
        }
        
        self.bin_offsets_.clear();
        self.num_total_bin_ = offset;
        
        // However, we should force to leave one bin, if dense multi val bin is the first bin
        // and its first feature has most freq bin > 0
        if group_id == 0 && self.num_feature_ > 0 && self.is_dense_multi_val_ 
            && self.bin_mappers_[0].get_most_freq_bin() > 0 {
            self.num_total_bin_ = 1;
        }
        
        self.bin_offsets_.push(self.num_total_bin_ as u32);
        
        for i in 0..self.num_feature_ as usize {
            let mut num_bin = self.bin_mappers_[i].num_bin();
            if self.bin_mappers_[i].get_most_freq_bin() == 0 {
                num_bin -= offset;
            }
            self.num_total_bin_ += num_bin;
            self.bin_offsets_.push(self.num_total_bin_ as u32);
        }
        
        for i in 0..other.num_feature_ as usize {
            let other_bin_mapper = &other.bin_mappers_[i];
            self.bin_mappers_.push(Box::new((**other_bin_mapper).clone()));
            let mut num_bin = other_bin_mapper.num_bin();
            if other_bin_mapper.get_most_freq_bin() == 0 {
                num_bin -= offset;
            }
            self.num_total_bin_ += num_bin;
            self.bin_offsets_.push(self.num_total_bin_ as u32);
            self.multi_bin_data_.push(other.multi_bin_data_[i].clone_bin());
        }
        
        self.num_feature_ += other.num_feature_;
        Ok(())
    }

    /// Create iterator for a sub-feature
    pub fn sub_feature_iterator(&self, sub_feature: i32) -> Option<Box<dyn BinIterator>> {
        let most_freq_bin = self.bin_mappers_[sub_feature as usize].get_most_freq_bin();
        
        if !self.is_multi_val_ {
            let min_bin = self.bin_offsets_[sub_feature as usize];
            let max_bin = self.bin_offsets_[sub_feature as usize + 1] - 1;
            if let Some(ref bin_data) = self.bin_data_ {
                return Some(bin_data.get_iterator(min_bin, max_bin, most_freq_bin));
            }
        } else {
            let addi = if self.bin_mappers_[sub_feature as usize].get_most_freq_bin() == 0 { 0 } else { 1 };
            let min_bin = 1;
            let max_bin = self.bin_mappers_[sub_feature as usize].num_bin() as u32 - 1 + addi;
            return Some(self.multi_bin_data_[sub_feature as usize].get_iterator(min_bin, max_bin, most_freq_bin));
        }
        
        None
    }

    /// Finish loading data
    pub fn finish_load(&mut self) {
        if self.is_multi_val_ {
            for i in 0..self.num_feature_ as usize {
                self.multi_bin_data_[i].finish_load();
            }
        } else if let Some(ref mut bin_data) = self.bin_data_ {
            bin_data.finish_load();
        }
    }

    /// Create iterator for the entire feature group
    pub fn feature_group_iterator(&self) -> Option<Box<dyn BinIterator>> {
        if self.is_multi_val_ {
            return None;
        }
        
        if let Some(ref bin_data) = self.bin_data_ {
            let min_bin = self.bin_offsets_[0];
            let max_bin = *self.bin_offsets_.last().unwrap() - 1;
            let most_freq_bin = 0;
            return Some(bin_data.get_iterator(min_bin, max_bin, most_freq_bin));
        }
        
        None
    }

    /// Get sizes in bytes for feature group data
    pub fn feature_group_sizes_in_byte(&self) -> usize {
        if let Some(ref bin_data) = self.bin_data_ {
            bin_data.sizes_in_byte()
        } else {
            0
        }
    }

    /// Get feature group data pointer
    pub fn feature_group_data(&self) -> Option<*const u8> {
        if self.is_multi_val_ {
            return None;
        }
        
        if let Some(ref bin_data) = self.bin_data_ {
            Some(bin_data.get_data())
        } else {
            None
        }
    }

    /// Convert bin to feature value
    pub fn bin_to_value(&self, sub_feature_idx: i32, bin: u32) -> f64 {
        self.bin_mappers_[sub_feature_idx as usize].bin_to_value(bin)
    }

    /// Get sizes in byte of this object
    pub fn sizes_in_byte(&self, include_data: bool) -> usize {
        let mut ret = crate::core::utils::binary_writer::VecBinaryWriter::aligned_size(std::mem::size_of::<bool>(), 8) +
                     crate::core::utils::binary_writer::VecBinaryWriter::aligned_size(std::mem::size_of::<bool>(), 8) +
                     crate::core::utils::binary_writer::VecBinaryWriter::aligned_size(std::mem::size_of::<bool>(), 8) +
                     crate::core::utils::binary_writer::VecBinaryWriter::aligned_size(std::mem::size_of::<i32>(), 8);
        
        for i in 0..self.num_feature_ as usize {
            ret += self.bin_mappers_[i].sizes_in_byte();
        }
        
        if include_data {
            if !self.is_multi_val_ {
                if let Some(ref bin_data) = self.bin_data_ {
                    ret += bin_data.sizes_in_byte();
                }
            } else {
                for i in 0..self.num_feature_ as usize {
                    ret += self.multi_bin_data_[i].sizes_in_byte();
                }
            }
        }
        
        ret
    }

    /// Get the maximum bin value for a feature
    pub fn feature_max_bin(&self, sub_feature_index: i32) -> u32 {
        if !self.is_multi_val_ {
            self.bin_offsets_[sub_feature_index as usize + 1] - 1
        } else {
            let addi = if self.bin_mappers_[sub_feature_index as usize].get_most_freq_bin() == 0 { 0 } else { 1 };
            self.bin_mappers_[sub_feature_index as usize].num_bin() as u32 - 1 + addi
        }
    }

    /// Get the minimum bin value for a feature
    pub fn feature_min_bin(&self, sub_feature_index: i32) -> u32 {
        if !self.is_multi_val_ {
            self.bin_offsets_[sub_feature_index as usize]
        } else {
            1
        }
    }
}