//! Training shared states for LightGBM implementation.
//!
//! This module provides a pure Rust implementation of the LightGBM training
//! shared states functionality, equivalent to train_share_states.cpp in the
//! original C++ implementation.

use crate::core::{constants::*, memory::AlignedBuffer, types::*};
use anyhow::Result;
use std::mem;
use std::ptr;
use std::sync::Arc;

/// Constants for histogram operations
const K_HIST_BUFFER_ENTRY_SIZE: usize = 2 * mem::size_of::<Hist>();
const K_INT16_HIST_BUFFER_ENTRY_SIZE: usize = 4; // Size for 16-bit histogram entries
const K_ZERO_THRESHOLD: f64 = 1e-35;
const MULTI_VAL_BIN_SPARSE_THRESHOLD: f64 = 0.25;

/// Feature group trait for compatibility
pub trait FeatureGroup: Send + Sync {
    /// Returns true if this feature group contains multi-value features
    fn is_multi_val(&self) -> bool;
    /// Returns true if this feature group uses dense multi-value storage
    fn is_dense_multi_val(&self) -> bool;
    /// Returns the number of features in this group
    fn num_feature(&self) -> usize;
    /// Returns the bin mappers for each feature in this group
    fn bin_mappers(&self) -> &[Arc<dyn BinMapper>];
    /// Returns the bin offsets for features in this group
    fn bin_offsets(&self) -> &[u32];
}

/// Bin mapper trait for compatibility
pub trait BinMapper: Send + Sync {
    /// Returns the number of bins for this feature
    fn num_bin(&self) -> i32;
    /// Returns the most frequently occurring bin
    fn get_most_freq_bin(&self) -> u32;
    /// Returns the sparsity rate of this feature (fraction of missing values)
    fn sparse_rate(&self) -> f64;
}

/// Multi-value bin trait for compatibility
pub trait MultiValBin: Send + Sync + std::fmt::Debug {
    /// Returns the number of bins in this multi-value bin structure
    fn num_bin(&self) -> i32;
    /// Returns the average number of elements per row
    fn num_element_per_row(&self) -> f64;
    /// Returns the offsets array for accessing bin data
    fn offsets(&self) -> &[u32];
    /// Returns true if this multi-value bin uses sparse storage
    fn is_sparse(&self) -> bool;

    /// Creates a new multi-value bin with the same structure but different parameters
    fn create_like(
        &self,
        num_data: DataSize,
        num_bin: i32,
        num_feature: usize,
        sum_dense_ratio: f64,
        offsets: Vec<u32>,
    ) -> Box<dyn MultiValBin>;

    /// Resizes this multi-value bin to accommodate new parameters
    fn re_size(
        &mut self,
        num_data: DataSize,
        num_bin: i32,
        num_feature: usize,
        sum_dense_ratio: f64,
        offsets: Vec<u32>,
    );

    /// Copies a subset of rows from another multi-value bin
    fn copy_subrow(&mut self, other: &dyn MultiValBin, indices: &[DataSize], count: DataSize);

    /// Copies a subset of columns from another multi-value bin
    fn copy_subcol(
        &mut self,
        other: &dyn MultiValBin,
        used_feature_index: &[i32],
        lower_bound: &[u32],
        upper_bound: &[u32],
        delta: &[u32],
    );

    /// Copies a subset of both rows and columns from another multi-value bin
    fn copy_subrow_and_subcol(
        &mut self,
        other: &dyn MultiValBin,
        indices: &[DataSize],
        count: DataSize,
        used_feature_index: &[i32],
        lower_bound: &[u32],
        upper_bound: &[u32],
        delta: &[u32],
    );

    /// Constructs histogram with ordered data indices for better cache locality
    fn construct_histogram_ordered(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        gradients: &[Score],
        hessians: &[Score],
        hist_data: *mut Hist,
    );

    /// Constructs histogram from gradient and hessian data
    fn construct_histogram(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        gradients: &[Score],
        hessians: &[Score],
        hist_data: *mut Hist,
    );

    /// Constructs histogram without using data indices (for contiguous data)
    fn construct_histogram_no_indices(
        &self,
        start: DataSize,
        end: DataSize,
        gradients: &[Score],
        hessians: &[Score],
        hist_data: *mut Hist,
    );
}

/// Multi-value bin wrapper for managing histogram operations
#[derive(Debug)]
pub struct MultiValBinWrapper {
    /// Feature groups contained in this wrapper
    feature_groups_contained: Vec<i32>,
    /// Number of threads for parallel operations
    num_threads: usize,
    /// Number of data points
    num_data: DataSize,
    /// Main multi-value bin
    multi_val_bin: Option<Box<dyn MultiValBin>>,
    /// Subset multi-value bin for optimization
    multi_val_bin_subset: Option<Box<dyn MultiValBin>>,
    /// Number of bins
    num_bin: i32,
    /// Number of bins aligned to memory boundaries
    num_bin_aligned: i32,
    /// Number of gradient quantization bins
    num_grad_quant_bins: i32,
    /// Whether using sub-column optimization
    is_use_subcol: bool,
    /// Whether using sub-row optimization
    is_use_subrow: bool,
    /// Whether sub-row has been copied
    is_subrow_copied: bool,
    /// Number of data blocks for parallel processing
    n_data_block: i32,
    /// Size of each data block
    data_block_size: DataSize,
    /// Minimum block size
    min_block_size: i32,
    /// Histogram move source indices
    hist_move_src: Vec<u32>,
    /// Histogram move destination indices
    hist_move_dest: Vec<u32>,
    /// Histogram move sizes
    hist_move_size: Vec<u32>,
    /// Pointer to original histogram data
    origin_hist_data: *mut Hist,
}

impl MultiValBinWrapper {
    /// Create a new multi-value bin wrapper
    pub fn new(
        bin: Option<Box<dyn MultiValBin>>,
        num_data: DataSize,
        feature_groups_contained: Vec<i32>,
        num_grad_quant_bins: i32,
    ) -> Self {
        let num_threads = rayon::current_num_threads();

        let (num_bin, num_bin_aligned) = if let Some(ref bin) = bin {
            let nb = bin.num_bin();
            let nba = (nb + ALIGNED_SIZE as i32 - 1) / ALIGNED_SIZE as i32 * ALIGNED_SIZE as i32;
            (nb, nba)
        } else {
            (0, 0)
        };

        Self {
            feature_groups_contained,
            num_threads,
            num_data,
            multi_val_bin: bin,
            multi_val_bin_subset: None,
            num_bin,
            num_bin_aligned,
            num_grad_quant_bins,
            is_use_subcol: false,
            is_use_subrow: false,
            is_subrow_copied: false,
            n_data_block: 1,
            data_block_size: 0,
            min_block_size: 32,
            hist_move_src: Vec::new(),
            hist_move_dest: Vec::new(),
            hist_move_size: Vec::new(),
            origin_hist_data: ptr::null_mut(),
        }
    }

    /// Check if the multi-value bin is sparse
    pub fn is_sparse(&self) -> bool {
        if let Some(ref bin) = self.multi_val_bin {
            bin.is_sparse()
        } else {
            false
        }
    }

    /// Initialize training with feature selection and bagging
    pub fn init_train(
        &mut self,
        group_feature_start: &[i32],
        feature_groups: &[Box<dyn FeatureGroup>],
        is_feature_used: &[i8],
        bagging_use_indices: Option<&[DataSize]>,
        bagging_indices_cnt: DataSize,
    ) -> Result<()> {
        self.is_use_subcol = false;

        if self.multi_val_bin.is_none() {
            return Ok(());
        }

        self.copy_multi_val_bin_subset(
            group_feature_start,
            feature_groups,
            is_feature_used,
            bagging_use_indices,
            bagging_indices_cnt,
        )?;

        let cur_multi_val_bin = if self.is_use_subcol || self.is_use_subrow {
            self.multi_val_bin_subset.as_ref()
        } else {
            self.multi_val_bin.as_ref()
        };

        if let Some(bin) = cur_multi_val_bin {
            self.num_bin = bin.num_bin();
            self.num_bin_aligned = (self.num_bin + ALIGNED_SIZE as i32 - 1) / ALIGNED_SIZE as i32
                * ALIGNED_SIZE as i32;
            let num_element_per_row = bin.num_element_per_row();
            self.min_block_size =
                ((0.3 * self.num_bin as f64 / (num_element_per_row + K_ZERO_THRESHOLD)) + 1.0)
                    .min(1024.0) as i32;
            self.min_block_size = self.min_block_size.max(32);
        }

        Ok(())
    }

    /// Move histogram data with different quantization settings
    pub fn hist_move<
        const USE_QUANT_GRAD: bool,
        const HIST_BITS: i32,
        const INNER_HIST_BITS: i32,
    >(
        &self,
        hist_buf: &AlignedBuffer<Hist>,
    ) {
        if !self.is_use_subcol && INNER_HIST_BITS != 8 {
            return;
        }

        if USE_QUANT_GRAD {
            match HIST_BITS {
                32 => self.hist_move_32bit(hist_buf),
                16 => {
                    if self.is_use_subcol {
                        self.hist_move_16bit_subcol(hist_buf);
                    } else {
                        assert_eq!(INNER_HIST_BITS, 8);
                        self.hist_move_16bit_no_subcol(hist_buf);
                    }
                }
                _ => {}
            }
        } else {
            self.hist_move_no_quant(hist_buf);
        }
    }

    /// Move histogram data for 32-bit quantized gradients
    fn hist_move_32bit(&self, hist_buf: &AlignedBuffer<Hist>) {
        let src_ptr = unsafe {
            let base = hist_buf.as_ptr() as *const i64;
            base.add(hist_buf.len() / 2 - self.num_bin_aligned as usize)
        };

        for i in 0..self.hist_move_src.len() {
            unsafe {
                let src = src_ptr.add(self.hist_move_src[i] as usize / 2);
                let dst =
                    (self.origin_hist_data as *mut i64).add(self.hist_move_dest[i] as usize / 2);
                ptr::copy_nonoverlapping(src, dst, self.hist_move_size[i] as usize / 2);
            }
        }
    }

    /// Move histogram data for 16-bit quantized gradients with sub-column optimization
    fn hist_move_16bit_subcol(&self, hist_buf: &AlignedBuffer<Hist>) {
        let src_ptr = unsafe {
            let base = hist_buf.as_ptr() as *const i32;
            base.add(hist_buf.len() / 2 - self.num_bin_aligned as usize)
        };

        for i in 0..self.hist_move_src.len() {
            unsafe {
                let src = src_ptr.add(self.hist_move_src[i] as usize / 2);
                let dst =
                    (self.origin_hist_data as *mut i32).add(self.hist_move_dest[i] as usize / 2);
                ptr::copy_nonoverlapping(src, dst, self.hist_move_size[i] as usize / 2);
            }
        }
    }

    /// Move histogram data for 16-bit quantized gradients without sub-column optimization
    fn hist_move_16bit_no_subcol(&self, hist_buf: &AlignedBuffer<Hist>) {
        let src_ptr = unsafe {
            let base = hist_buf.as_ptr() as *const i32;
            base.add(hist_buf.len() / 2)
        };
        let orig_ptr = self.origin_hist_data as *mut i32;

        for i in 0..self.num_bin as usize {
            unsafe {
                *orig_ptr.add(i) = *src_ptr.add(i);
            }
        }
    }

    /// Move histogram data without quantized gradients
    fn hist_move_no_quant(&self, hist_buf: &AlignedBuffer<Hist>) {
        let src_ptr = unsafe {
            hist_buf
                .as_ptr()
                .add(hist_buf.len() - 2 * self.num_bin_aligned as usize)
        };

        for i in 0..self.hist_move_src.len() {
            unsafe {
                let src = src_ptr.add(self.hist_move_src[i] as usize);
                let dst = self.origin_hist_data.add(self.hist_move_dest[i] as usize);
                ptr::copy_nonoverlapping(src, dst, self.hist_move_size[i] as usize);
            }
        }
    }

    /// Merge histogram data with different quantization settings
    pub fn hist_merge<
        const USE_QUANT_GRAD: bool,
        const HIST_BITS: i32,
        const INNER_HIST_BITS: i32,
    >(
        &self,
        hist_buf: &mut AlignedBuffer<Hist>,
    ) {
        let mut n_bin_block = 1;
        let mut bin_block_size = self.num_bin;
        self.calculate_block_info(self.num_bin, 512, &mut n_bin_block, &mut bin_block_size);

        if USE_QUANT_GRAD {
            match HIST_BITS {
                32 => self.hist_merge_32bit(hist_buf, n_bin_block, bin_block_size),
                16 => {
                    if INNER_HIST_BITS == 16 {
                        self.hist_merge_16bit_16bit(hist_buf, n_bin_block, bin_block_size);
                    } else if INNER_HIST_BITS == 8 {
                        self.hist_merge_16bit_8bit(hist_buf, n_bin_block, bin_block_size);
                    }
                }
                _ => {}
            }
        } else {
            self.hist_merge_no_quant(hist_buf, n_bin_block, bin_block_size);
        }
    }

    /// Calculate block information for parallel processing
    fn calculate_block_info(
        &self,
        total_size: i32,
        min_block_size: i32,
        n_blocks: &mut i32,
        block_size: &mut i32,
    ) {
        *n_blocks = (total_size + min_block_size - 1) / min_block_size;
        *n_blocks = (*n_blocks).min(self.num_threads as i32);
        *block_size = (total_size + *n_blocks - 1) / *n_blocks;
    }

    /// Merge histogram data for 32-bit quantized gradients
    fn hist_merge_32bit(
        &self,
        hist_buf: &mut AlignedBuffer<Hist>,
        n_bin_block: i32,
        bin_block_size: i32,
    ) {
        let dst_ptr = if self.is_use_subcol {
            unsafe {
                let base = hist_buf.as_mut_ptr() as *mut i64;
                base.add(hist_buf.len() / 2 - self.num_bin_aligned as usize)
            }
        } else {
            self.origin_hist_data as *mut i64
        };

        for t in 0..n_bin_block {
            let start = t * bin_block_size;
            let end = (start + bin_block_size).min(self.num_bin);

            for tid in 1..self.n_data_block {
                unsafe {
                    let src_ptr = (hist_buf.as_ptr() as *const i64)
                        .add(self.num_bin_aligned as usize * (tid - 1) as usize);

                    for i in start..end {
                        *dst_ptr.add(i as usize) += *src_ptr.add(i as usize);
                    }
                }
            }
        }
    }

    /// Merge histogram data for 16-bit quantized gradients (16-bit inner)
    fn hist_merge_16bit_16bit(
        &self,
        hist_buf: &mut AlignedBuffer<Hist>,
        n_bin_block: i32,
        bin_block_size: i32,
    ) {
        let dst_ptr = if self.is_use_subcol {
            unsafe {
                let base = hist_buf.as_mut_ptr() as *mut i32;
                base.add(hist_buf.len() / 2 - self.num_bin_aligned as usize)
            }
        } else {
            self.origin_hist_data as *mut i32
        };

        for t in 0..n_bin_block {
            let start = t * bin_block_size;
            let end = (start + bin_block_size).min(self.num_bin);

            for tid in 1..self.n_data_block {
                unsafe {
                    let src_ptr = (hist_buf.as_ptr() as *const i32)
                        .add(self.num_bin_aligned as usize * (tid - 1) as usize);

                    for i in start..end {
                        *dst_ptr.add(i as usize) += *src_ptr.add(i as usize);
                    }
                }
            }
        }
    }

    /// Merge histogram data for 16-bit quantized gradients (8-bit inner)
    fn hist_merge_16bit_8bit(
        &self,
        hist_buf: &mut AlignedBuffer<Hist>,
        n_bin_block: i32,
        bin_block_size: i32,
    ) {
        let dst_ptr = unsafe {
            let base = hist_buf.as_mut_ptr() as *mut i32;
            base.add(hist_buf.len() / 2)
        };

        // Zero out destination
        unsafe {
            ptr::write_bytes(
                dst_ptr,
                0,
                self.num_bin as usize * K_INT16_HIST_BUFFER_ENTRY_SIZE,
            );
        }

        for t in 0..n_bin_block {
            let start = t * bin_block_size;
            let end = (start + bin_block_size).min(self.num_bin);

            for tid in 0..self.n_data_block {
                unsafe {
                    let src_ptr = (hist_buf.as_ptr() as *const i16)
                        .add(self.num_bin_aligned as usize * tid as usize);

                    for i in start..end {
                        let packed_hist = *src_ptr.add(i as usize);
                        let packed_hist_int32 = (((packed_hist >> 8) as i8) as i32) << 16
                            | (packed_hist as i32 & 0x00ff);
                        *dst_ptr.add(i as usize) += packed_hist_int32;
                    }
                }
            }
        }
    }

    /// Merge histogram data without quantized gradients
    fn hist_merge_no_quant(
        &self,
        hist_buf: &mut AlignedBuffer<Hist>,
        n_bin_block: i32,
        bin_block_size: i32,
    ) {
        let dst_ptr = if self.is_use_subcol {
            unsafe {
                hist_buf
                    .as_mut_ptr()
                    .add(hist_buf.len() - 2 * self.num_bin_aligned as usize)
            }
        } else {
            self.origin_hist_data
        };

        for t in 0..n_bin_block {
            let start = t * bin_block_size;
            let end = (start + bin_block_size).min(self.num_bin);

            for tid in 1..self.n_data_block {
                unsafe {
                    let src_ptr = hist_buf
                        .as_ptr()
                        .add(self.num_bin_aligned as usize * 2 * (tid - 1) as usize);

                    for i in (start * 2)..(end * 2) {
                        *dst_ptr.add(i as usize) += *src_ptr.add(i as usize);
                    }
                }
            }
        }
    }

    /// Resize histogram buffer
    pub fn resize_hist_buf(
        &mut self,
        _hist_buf: &mut AlignedBuffer<Hist>,
        sub_multi_val_bin: &dyn MultiValBin,
        origin_hist_data: *mut Hist,
    ) -> Result<()> {
        self.num_bin = sub_multi_val_bin.num_bin();
        self.num_bin_aligned =
            (self.num_bin + ALIGNED_SIZE as i32 - 1) / ALIGNED_SIZE as i32 * ALIGNED_SIZE as i32;
        self.origin_hist_data = origin_hist_data;

        // For now, we assume the buffer is large enough
        // In a full implementation, we'd need to resize the AlignedBuffer if needed

        Ok(())
    }

    /// Copy multi-value bin subset for optimization
    fn copy_multi_val_bin_subset(
        &mut self,
        group_feature_start: &[i32],
        feature_groups: &[Box<dyn FeatureGroup>],
        is_feature_used: &[i8],
        bagging_use_indices: Option<&[DataSize]>,
        bagging_indices_cnt: DataSize,
    ) -> Result<()> {
        let mut sum_used_dense_ratio = 0.0;
        let mut sum_dense_ratio = 0.0;
        let mut num_used = 0;
        let mut total = 0;
        let mut used_feature_index = Vec::new();

        // Calculate dense ratios and collect used features
        for &i in &self.feature_groups_contained {
            let f_start = group_feature_start[i as usize] as usize;
            let feature_group = &feature_groups[i as usize];

            if feature_group.is_multi_val() {
                for j in 0..feature_group.num_feature() {
                    let dense_rate = 1.0 - feature_group.bin_mappers()[j].sparse_rate();
                    if is_feature_used[f_start + j] != 0 {
                        num_used += 1;
                        used_feature_index.push(total as i32);
                        sum_used_dense_ratio += dense_rate;
                    }
                    sum_dense_ratio += dense_rate;
                    total += 1;
                }
            } else {
                let mut is_group_used = false;
                let mut dense_rate = 0.0;
                for j in 0..feature_group.num_feature() {
                    if is_feature_used[f_start + j] != 0 {
                        is_group_used = true;
                    }
                    dense_rate += 1.0 - feature_group.bin_mappers()[j].sparse_rate();
                }
                if is_group_used {
                    num_used += 1;
                    used_feature_index.push(total as i32);
                    sum_used_dense_ratio += dense_rate;
                }
                sum_dense_ratio += dense_rate;
                total += 1;
            }
        }

        const K_SUBFEATURE_THRESHOLD: f64 = 0.6;

        if sum_used_dense_ratio >= sum_dense_ratio * K_SUBFEATURE_THRESHOLD {
            // Only need to copy subset
            if self.is_use_subrow && !self.is_subrow_copied {
                if let Some(ref bin) = self.multi_val_bin {
                    if self.multi_val_bin_subset.is_none() {
                        self.multi_val_bin_subset = Some(bin.create_like(
                            bagging_indices_cnt,
                            bin.num_bin(),
                            total,
                            bin.num_element_per_row(),
                            bin.offsets().to_vec(),
                        ));
                    } else if let Some(ref mut subset) = self.multi_val_bin_subset {
                        subset.re_size(
                            bagging_indices_cnt,
                            bin.num_bin(),
                            total,
                            bin.num_element_per_row(),
                            bin.offsets().to_vec(),
                        );
                    }

                    if let (Some(ref mut subset), Some(indices)) =
                        (&mut self.multi_val_bin_subset, bagging_use_indices)
                    {
                        subset.copy_subrow(bin.as_ref(), indices, bagging_indices_cnt);
                    }

                    self.is_subrow_copied = true;
                }
            }
        } else {
            self.is_use_subcol = true;
            self.setup_subcol_optimization(
                group_feature_start,
                feature_groups,
                is_feature_used,
                bagging_use_indices,
                bagging_indices_cnt,
                used_feature_index,
                num_used,
                sum_used_dense_ratio,
            )?;
        }

        Ok(())
    }

    /// Setup sub-column optimization
    fn setup_subcol_optimization(
        &mut self,
        group_feature_start: &[i32],
        feature_groups: &[Box<dyn FeatureGroup>],
        is_feature_used: &[i8],
        bagging_use_indices: Option<&[DataSize]>,
        bagging_indices_cnt: DataSize,
        used_feature_index: Vec<i32>,
        num_used: i32,
        sum_used_dense_ratio: f64,
    ) -> Result<()> {
        let mut upper_bound = Vec::new();
        let mut lower_bound = Vec::new();
        let mut delta = Vec::new();
        let mut offsets = Vec::new();

        self.hist_move_src.clear();
        self.hist_move_dest.clear();
        self.hist_move_size.clear();

        let offset = if let Some(ref bin) = self.multi_val_bin {
            if bin.is_sparse() {
                1
            } else {
                0
            }
        } else {
            0
        };

        let mut num_total_bin = offset;
        let mut new_num_total_bin = offset;
        offsets.push(new_num_total_bin as u32);

        // Process feature groups for sub-column setup
        for &i in &self.feature_groups_contained {
            let f_start = group_feature_start[i as usize] as usize;
            let feature_group = &feature_groups[i as usize];

            if feature_group.is_multi_val() {
                for j in 0..feature_group.num_feature() {
                    let bin_mapper = &feature_group.bin_mappers()[j];
                    if i == 0 && j == 0 && bin_mapper.get_most_freq_bin() > 0 {
                        num_total_bin = 1;
                    }
                    let mut cur_num_bin = bin_mapper.num_bin();
                    if bin_mapper.get_most_freq_bin() == 0 {
                        cur_num_bin -= offset;
                    }
                    num_total_bin += cur_num_bin;

                    if is_feature_used[f_start + j] != 0 {
                        new_num_total_bin += cur_num_bin;
                        offsets.push(new_num_total_bin as u32);
                        lower_bound.push((num_total_bin - cur_num_bin) as u32);
                        upper_bound.push(num_total_bin as u32);

                        self.hist_move_src
                            .push(((new_num_total_bin - cur_num_bin) * 2) as u32);
                        self.hist_move_dest
                            .push(((num_total_bin - cur_num_bin) * 2) as u32);
                        self.hist_move_size.push((cur_num_bin * 2) as u32);
                        delta.push((num_total_bin - new_num_total_bin) as u32);
                    }
                }
            } else {
                let mut is_group_used = false;
                for j in 0..feature_group.num_feature() {
                    if is_feature_used[f_start + j] != 0 {
                        is_group_used = true;
                        break;
                    }
                }
                let cur_num_bin = *feature_group.bin_offsets().last().unwrap() as i32 - offset;
                num_total_bin += cur_num_bin;

                if is_group_used {
                    new_num_total_bin += cur_num_bin;
                    offsets.push(new_num_total_bin as u32);
                    lower_bound.push((num_total_bin - cur_num_bin) as u32);
                    upper_bound.push(num_total_bin as u32);

                    self.hist_move_src
                        .push(((new_num_total_bin - cur_num_bin) * 2) as u32);
                    self.hist_move_dest
                        .push(((num_total_bin - cur_num_bin) * 2) as u32);
                    self.hist_move_size.push((cur_num_bin * 2) as u32);
                    delta.push((num_total_bin - new_num_total_bin) as u32);
                }
            }
        }

        // Avoid out of range
        lower_bound.push(num_total_bin as u32);
        upper_bound.push(num_total_bin as u32);

        let num_data = if self.is_use_subrow {
            bagging_indices_cnt
        } else {
            self.num_data
        };

        if let Some(ref bin) = self.multi_val_bin {
            if self.multi_val_bin_subset.is_none() {
                self.multi_val_bin_subset = Some(bin.create_like(
                    num_data,
                    new_num_total_bin,
                    num_used as usize,
                    sum_used_dense_ratio,
                    offsets,
                ));
            } else if let Some(ref mut subset) = self.multi_val_bin_subset {
                subset.re_size(
                    num_data,
                    new_num_total_bin,
                    num_used as usize,
                    sum_used_dense_ratio,
                    offsets,
                );
            }

            if let Some(ref mut subset) = self.multi_val_bin_subset {
                if self.is_use_subrow {
                    if let Some(indices) = bagging_use_indices {
                        subset.copy_subrow_and_subcol(
                            bin.as_ref(),
                            indices,
                            bagging_indices_cnt,
                            &used_feature_index,
                            &lower_bound,
                            &upper_bound,
                            &delta,
                        );
                    }
                    self.is_subrow_copied = false;
                } else {
                    subset.copy_subcol(
                        bin.as_ref(),
                        &used_feature_index,
                        &lower_bound,
                        &upper_bound,
                        &delta,
                    );
                }
            }
        }

        Ok(())
    }

    /// Set whether to use sub-row optimization
    pub fn set_use_subrow(&mut self, is_use_subrow: bool) {
        self.is_use_subrow = is_use_subrow;
    }

    /// Set whether sub-row has been copied
    pub fn set_subrow_copied(&mut self, is_subrow_copied: bool) {
        self.is_subrow_copied = is_subrow_copied;
    }
}

/// Training share states structure
#[derive(Debug)]
pub struct TrainingShareStates {
    /// Number of threads for parallel processing
    pub num_threads: usize,
    /// Whether to use column-wise processing
    pub is_col_wise: bool,
    /// Whether hessian is constant
    pub is_constant_hessian: bool,
    /// Indices for bagging
    pub bagging_use_indices: Option<Vec<DataSize>>,
    /// Number of bagging indices
    pub bagging_indices_cnt: DataSize,
    /// Multi-value bin wrapper
    multi_val_bin_wrapper: Option<MultiValBinWrapper>,
    /// Feature histogram offsets
    feature_hist_offsets: Vec<u32>,
    /// Total number of histogram bins
    num_hist_total_bin: i32,
    /// Histogram buffer
    hist_buf: AlignedBuffer<Hist>,
    /// Total number of bins
    num_total_bin: i32,
    /// Number of elements per row
    num_elements_per_row: f64,
}

impl TrainingShareStates {
    /// Create new training share states
    pub fn new() -> Result<Self> {
        Ok(Self {
            num_threads: rayon::current_num_threads(),
            is_col_wise: true,
            is_constant_hessian: true,
            bagging_use_indices: None,
            bagging_indices_cnt: 0,
            multi_val_bin_wrapper: None,
            feature_hist_offsets: Vec::new(),
            num_hist_total_bin: 0,
            hist_buf: AlignedBuffer::new(0)?,
            num_total_bin: 0,
            num_elements_per_row: 0.0,
        })
    }

    /// Get number of histogram total bins
    pub fn num_hist_total_bin(&self) -> i32 {
        self.num_hist_total_bin
    }

    /// Get feature histogram offsets
    pub fn feature_hist_offsets(&self) -> &[u32] {
        &self.feature_hist_offsets
    }

    /// Check if using sparse row-wise processing
    pub fn is_sparse_rowwise(&self) -> bool {
        if let Some(ref wrapper) = self.multi_val_bin_wrapper {
            wrapper.is_sparse()
        } else {
            false
        }
    }

    /// Set multi-value bin
    pub fn set_multi_val_bin(
        &mut self,
        bin: Option<Box<dyn MultiValBin>>,
        num_data: DataSize,
        feature_groups_contained: Vec<i32>,
        _dense_only: bool,
        _sparse_only: bool,
        num_grad_quant_bins: i32,
    ) -> Result<()> {
        if let Some(bin) = bin {
            self.num_total_bin += bin.num_bin();
            self.num_elements_per_row += bin.num_element_per_row();
            self.multi_val_bin_wrapper = Some(MultiValBinWrapper::new(
                Some(bin),
                num_data,
                feature_groups_contained,
                num_grad_quant_bins,
            ));
        }
        Ok(())
    }

    /// Calculate bin offsets for feature groups
    pub fn calc_bin_offsets(
        &mut self,
        feature_groups: &[Box<dyn FeatureGroup>],
        offsets: &mut Vec<u32>,
        is_col_wise: bool,
    ) -> Result<()> {
        offsets.clear();
        self.feature_hist_offsets.clear();

        if is_col_wise {
            self.calc_bin_offsets_col_wise(feature_groups, offsets)?;
        } else {
            self.calc_bin_offsets_row_wise(feature_groups, offsets)?;
        }

        self.num_hist_total_bin = *self.feature_hist_offsets.last().unwrap_or(&0) as i32;
        Ok(())
    }

    /// Calculate bin offsets for column-wise processing
    fn calc_bin_offsets_col_wise(
        &mut self,
        feature_groups: &[Box<dyn FeatureGroup>],
        offsets: &mut Vec<u32>,
    ) -> Result<()> {
        let mut cur_num_bin = 0u32;
        let mut hist_cur_num_bin = 0u32;

        for (group, feature_group) in feature_groups.iter().enumerate() {
            if feature_group.is_multi_val() {
                if feature_group.is_dense_multi_val() {
                    for i in 0..feature_group.num_feature() {
                        let bin_mapper = &feature_group.bin_mappers()[i];
                        if group == 0 && i == 0 && bin_mapper.get_most_freq_bin() > 0 {
                            cur_num_bin += 1;
                            hist_cur_num_bin += 1;
                        }
                        offsets.push(cur_num_bin);
                        self.feature_hist_offsets.push(hist_cur_num_bin);
                        let num_bin = bin_mapper.num_bin() as u32;
                        hist_cur_num_bin += num_bin;
                        if bin_mapper.get_most_freq_bin() == 0 {
                            *self.feature_hist_offsets.last_mut().unwrap() += 1;
                        }
                        cur_num_bin += num_bin;
                    }
                    offsets.push(cur_num_bin);
                    assert_eq!(cur_num_bin, *feature_group.bin_offsets().last().unwrap());
                } else {
                    cur_num_bin += 1;
                    hist_cur_num_bin += 1;
                    for i in 0..feature_group.num_feature() {
                        offsets.push(cur_num_bin);
                        self.feature_hist_offsets.push(hist_cur_num_bin);
                        let bin_mapper = &feature_group.bin_mappers()[i];
                        let mut num_bin = bin_mapper.num_bin() as u32;
                        if bin_mapper.get_most_freq_bin() == 0 {
                            num_bin -= 1;
                        }
                        hist_cur_num_bin += num_bin;
                        cur_num_bin += num_bin;
                    }
                    offsets.push(cur_num_bin);
                    assert_eq!(cur_num_bin, *feature_group.bin_offsets().last().unwrap());
                }
            } else {
                for i in 0..feature_group.num_feature() {
                    self.feature_hist_offsets
                        .push(hist_cur_num_bin + feature_group.bin_offsets()[i]);
                }
                hist_cur_num_bin += *feature_group.bin_offsets().last().unwrap();
            }
        }

        self.feature_hist_offsets.push(hist_cur_num_bin);
        Ok(())
    }

    /// Calculate bin offsets for row-wise processing
    fn calc_bin_offsets_row_wise(
        &mut self,
        feature_groups: &[Box<dyn FeatureGroup>],
        offsets: &mut Vec<u32>,
    ) -> Result<()> {
        let mut sum_dense_ratio = 0.0;
        let mut ncol = 0;

        // Calculate sparsity
        for feature_group in feature_groups {
            if feature_group.is_multi_val() {
                ncol += feature_group.num_feature();
            } else {
                ncol += 1;
            }
            for j in 0..feature_group.num_feature() {
                let bin_mapper = &feature_group.bin_mappers()[j];
                sum_dense_ratio += 1.0 - bin_mapper.sparse_rate();
            }
        }

        sum_dense_ratio /= ncol as f64;
        let is_sparse_row_wise = (1.0 - sum_dense_ratio) >= MULTI_VAL_BIN_SPARSE_THRESHOLD;

        if is_sparse_row_wise {
            self.calc_bin_offsets_sparse_row_wise(feature_groups, offsets)?;
        } else {
            self.calc_bin_offsets_dense_row_wise(feature_groups, offsets)?;
        }

        Ok(())
    }

    /// Calculate bin offsets for sparse row-wise processing
    fn calc_bin_offsets_sparse_row_wise(
        &mut self,
        feature_groups: &[Box<dyn FeatureGroup>],
        offsets: &mut Vec<u32>,
    ) -> Result<()> {
        let mut cur_num_bin = 1u32;
        let mut hist_cur_num_bin = 1u32;

        for feature_group in feature_groups {
            if feature_group.is_multi_val() {
                for i in 0..feature_group.num_feature() {
                    offsets.push(cur_num_bin);
                    self.feature_hist_offsets.push(hist_cur_num_bin);
                    let bin_mapper = &feature_group.bin_mappers()[i];
                    let mut num_bin = bin_mapper.num_bin() as u32;
                    if bin_mapper.get_most_freq_bin() == 0 {
                        num_bin -= 1;
                    }
                    cur_num_bin += num_bin;
                    hist_cur_num_bin += num_bin;
                }
            } else {
                offsets.push(cur_num_bin);
                cur_num_bin += *feature_group.bin_offsets().last().unwrap() - 1;
                for i in 0..feature_group.num_feature() {
                    self.feature_hist_offsets
                        .push(hist_cur_num_bin + feature_group.bin_offsets()[i] - 1);
                }
                hist_cur_num_bin += *feature_group.bin_offsets().last().unwrap() - 1;
            }
        }

        offsets.push(cur_num_bin);
        self.feature_hist_offsets.push(hist_cur_num_bin);
        Ok(())
    }

    /// Calculate bin offsets for dense row-wise processing
    fn calc_bin_offsets_dense_row_wise(
        &mut self,
        feature_groups: &[Box<dyn FeatureGroup>],
        offsets: &mut Vec<u32>,
    ) -> Result<()> {
        let mut cur_num_bin = 0u32;
        let mut hist_cur_num_bin = 0u32;

        for (group, feature_group) in feature_groups.iter().enumerate() {
            if feature_group.is_multi_val() {
                for i in 0..feature_group.num_feature() {
                    let bin_mapper = &feature_group.bin_mappers()[i];
                    if group == 0 && i == 0 && bin_mapper.get_most_freq_bin() > 0 {
                        cur_num_bin += 1;
                        hist_cur_num_bin += 1;
                    }
                    offsets.push(cur_num_bin);
                    self.feature_hist_offsets.push(hist_cur_num_bin);
                    let num_bin = bin_mapper.num_bin() as u32;
                    cur_num_bin += num_bin;
                    hist_cur_num_bin += num_bin;
                    if bin_mapper.get_most_freq_bin() == 0 {
                        *self.feature_hist_offsets.last_mut().unwrap() += 1;
                    }
                }
            } else {
                offsets.push(cur_num_bin);
                cur_num_bin += *feature_group.bin_offsets().last().unwrap();
                for i in 0..feature_group.num_feature() {
                    self.feature_hist_offsets
                        .push(hist_cur_num_bin + feature_group.bin_offsets()[i]);
                }
                hist_cur_num_bin += *feature_group.bin_offsets().last().unwrap();
            }
        }

        offsets.push(cur_num_bin);
        self.feature_hist_offsets.push(hist_cur_num_bin);
        Ok(())
    }

    /// Initialize training
    pub fn init_train(
        &mut self,
        group_feature_start: &[i32],
        feature_groups: &[Box<dyn FeatureGroup>],
        is_feature_used: &[i8],
    ) -> Result<()> {
        if let Some(ref mut wrapper) = self.multi_val_bin_wrapper {
            wrapper.init_train(
                group_feature_start,
                feature_groups,
                is_feature_used,
                self.bagging_use_indices.as_ref().map(|v| v.as_slice()),
                self.bagging_indices_cnt,
            )?;
        }
        Ok(())
    }

    /// Set whether to use sub-row optimization
    pub fn set_use_subrow(&mut self, is_use_subrow: bool) {
        if let Some(ref mut wrapper) = self.multi_val_bin_wrapper {
            wrapper.set_use_subrow(is_use_subrow);
        }
    }

    /// Set whether sub-row has been copied
    pub fn set_subrow_copied(&mut self, is_subrow_copied: bool) {
        if let Some(ref mut wrapper) = self.multi_val_bin_wrapper {
            wrapper.set_subrow_copied(is_subrow_copied);
        }
    }
}

impl Default for TrainingShareStates {
    fn default() -> Self {
        Self::new().expect("Failed to create default TrainingShareStates")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_val_bin_wrapper_creation() {
        let wrapper = MultiValBinWrapper::new(None, 100, vec![0, 1, 2], 16);
        assert_eq!(wrapper.num_data, 100);
        assert_eq!(wrapper.feature_groups_contained, vec![0, 1, 2]);
        assert_eq!(wrapper.num_grad_quant_bins, 16);
        assert!(!wrapper.is_sparse());
    }

    #[test]
    fn test_training_share_states_creation() {
        let states = TrainingShareStates::new().unwrap();
        assert!(states.is_col_wise);
        assert!(states.is_constant_hessian);
        assert_eq!(states.num_hist_total_bin(), 0);
        assert!(!states.is_sparse_rowwise());
    }

    #[test]
    fn test_training_share_states_default() {
        let states = TrainingShareStates::default();
        assert!(states.is_col_wise);
        assert_eq!(states.bagging_indices_cnt, 0);
        assert_eq!(states.num_total_bin, 0);
    }
}
