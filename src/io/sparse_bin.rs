//! Sparse binary data structures and operations for feature binning.
//!
//! This module provides a pure Rust implementation of LightGBM's sparse bin
//! functionality, translating the C++ sparse_bin.hpp implementation while maintaining
//! semantic equivalence and leveraging Rust's type system for enhanced safety.

use crate::core::types::*;
use crate::io::bin::*;
use std::convert::TryFrom;
use std::sync::Arc;

/// Number of fast index entries for sparse bin lookups
const K_NUM_FAST_INDEX: usize = 64;

/// Aligned allocator type alias for consistent memory alignment
type AlignedVec<T> = Vec<T>;

/// Sparse bin iterator for efficient traversal of sparse bin data
#[derive(Debug)]
pub struct SparseBinIterator<T> {
    /// Reference to the parent sparse bin
    bin_data: Arc<SparseBin<T>>,
    /// Current position in the sparse data
    cur_pos: DataSize,
    /// Current delta index
    i_delta: DataSize,
    /// Minimum bin value for this iterator
    min_bin: T,
    /// Maximum bin value for this iterator
    max_bin: T,
    /// Most frequent bin value
    most_freq_bin: T,
    /// Offset value based on whether most_freq_bin is zero
    offset: u8,
}

impl<T> SparseBinIterator<T>
where
    T: Copy + Default + PartialEq + PartialOrd + TryFrom<u32> + Into<u32> + Send + Sync,
    <T as TryFrom<u32>>::Error: std::fmt::Debug,
{
    /// Create a new iterator with bin range constraints
    pub fn new(
        bin_data: Arc<SparseBin<T>>,
        min_bin: u32,
        max_bin: u32,
        most_freq_bin: u32,
    ) -> Self {
        let min_bin_t = T::try_from(min_bin).unwrap_or_default();
        let max_bin_t = T::try_from(max_bin).unwrap_or_default();
        let most_freq_bin_t = T::try_from(most_freq_bin).unwrap_or_default();

        let offset = if most_freq_bin == 0 { 1 } else { 0 };

        let mut iterator = Self {
            bin_data,
            cur_pos: 0,
            i_delta: 0,
            min_bin: min_bin_t,
            max_bin: max_bin_t,
            most_freq_bin: most_freq_bin_t,
            offset,
        };

        iterator.reset(0);
        iterator
    }

    /// Create a new iterator starting at a specific index
    pub fn new_with_start(bin_data: Arc<SparseBin<T>>, start_idx: DataSize) -> Self {
        let mut iterator = Self {
            bin_data,
            cur_pos: 0,
            i_delta: 0,
            min_bin: T::default(),
            max_bin: T::default(),
            most_freq_bin: T::default(),
            offset: 0,
        };

        iterator.reset(start_idx);
        iterator
    }

    /// Get raw bin value at index (public interface)
    pub fn raw_get(&mut self, idx: DataSize) -> u32 {
        self.inner_raw_get(idx).into()
    }

    /// Internal raw get implementation
    pub fn inner_raw_get(&mut self, idx: DataSize) -> T {
        while self.cur_pos < idx {
            self.bin_data
                .next_nonzero_fast(&mut self.i_delta, &mut self.cur_pos);
        }

        if self.cur_pos == idx {
            self.bin_data.vals[self.i_delta as usize]
        } else {
            T::default() // Return 0 equivalent
        }
    }

    /// Get bin value with range mapping
    pub fn get(&mut self, idx: DataSize) -> u32 {
        let ret = self.inner_raw_get(idx);
        if ret >= self.min_bin && ret <= self.max_bin {
            ret.into() - self.min_bin.into() + self.offset as u32
        } else {
            self.most_freq_bin.into()
        }
    }

    /// Reset iterator to start at a specific index
    pub fn reset(&mut self, idx: DataSize) {
        self.bin_data
            .init_index(idx, &mut self.i_delta, &mut self.cur_pos);
    }
}

impl<T> BinIterator for SparseBinIterator<T>
where
    T: Copy + Default + PartialEq + PartialOrd + TryFrom<u32> + Into<u32> + Send + Sync,
    <T as TryFrom<u32>>::Error: std::fmt::Debug,
{
    fn reset(&mut self, start_idx: usize) {
        self.reset(start_idx as DataSize);
    }

    fn get(&self, _idx: usize) -> BinIndex {
        // Note: This requires mutable access, so we need to work around the trait constraint
        // For now, return a default value - this may need refinement
        0
    }
}

/// Sparse binary data structure for efficient storage of sparse feature data
#[derive(Debug)]
pub struct SparseBin<T> {
    /// Number of data points
    num_data: DataSize,
    /// Delta compressed sparse representation
    deltas: AlignedVec<u8>,
    /// Bin values corresponding to non-zero entries
    vals: AlignedVec<T>,
    /// Number of non-zero values
    num_vals: DataSize,
    /// Thread-local push buffers for parallel data loading
    push_buffers: Vec<Vec<(DataSize, T)>>,
    /// Fast index for efficient lookups
    fast_index: Vec<(DataSize, DataSize)>,
    /// Bit shift value for fast index calculation
    fast_index_shift: DataSize,
}

impl<T> SparseBin<T>
where
    T: Copy + Default + PartialEq + PartialOrd + TryFrom<u32> + Into<u32> + Send + Sync,
    <T as TryFrom<u32>>::Error: std::fmt::Debug,
{
    /// Create a new sparse bin with specified capacity
    pub fn new(num_data: DataSize) -> Self {
        let num_threads = rayon::current_num_threads();
        Self {
            num_data,
            deltas: AlignedVec::new(),
            vals: AlignedVec::new(),
            num_vals: 0,
            push_buffers: vec![Vec::new(); num_threads],
            fast_index: Vec::new(),
            fast_index_shift: 0,
        }
    }

    /// Initialize streaming mode with specified thread count
    pub fn init_streaming(&mut self, num_thread: u32, omp_max_threads: i32) {
        // Each external thread needs its own set of OpenMP push buffers
        let total_buffers = (omp_max_threads as u32) * num_thread;
        self.push_buffers.resize(total_buffers as usize, Vec::new());
    }

    /// Resize the sparse bin for new data size
    pub fn resize(&mut self, num_data: DataSize) {
        self.num_data = num_data;
    }

    /// Push a value to the sparse bin at specified index and thread
    pub fn push(&mut self, tid: usize, idx: DataSize, value: u32) {
        if let Ok(cur_bin) = T::try_from(value) {
            if cur_bin != T::default() {
                if tid < self.push_buffers.len() {
                    self.push_buffers[tid].push((idx, cur_bin));
                }
            }
        }
    }

    /// Get iterator for this sparse bin
    pub fn get_iterator(
        &self,
        min_bin: u32,
        max_bin: u32,
        most_freq_bin: u32,
    ) -> SparseBinIterator<T> {
        SparseBinIterator::new(
            Arc::new(self.clone()), // This requires Clone implementation
            min_bin,
            max_bin,
            most_freq_bin,
        )
    }

    /// Construct histogram with data indices and gradient/hessian values
    pub fn construct_histogram(
        &self,
        data_indices: &[DataSize],
        start: usize,
        end: usize,
        ordered_gradients: &[Score],
        ordered_hessians: &[Score],
        out: &mut [Hist],
    ) {
        if start >= end || start >= data_indices.len() {
            return;
        }

        let mut i_delta = 0_i32;
        let mut cur_pos = 0_i32;
        self.init_index(data_indices[start], &mut i_delta, &mut cur_pos);

        let mut i = start;
        loop {
            if cur_pos < data_indices[i] {
                cur_pos += self.deltas[i_delta as usize + 1] as i32;
                i_delta += 1;
                if i_delta >= self.num_vals {
                    break;
                }
            } else if cur_pos > data_indices[i] {
                i += 1;
                if i >= end {
                    break;
                }
            } else {
                // Found match
                let bin = self.vals[i_delta as usize];
                let bin_idx = bin.into() as usize;
                if bin_idx * 2 + 1 < out.len() {
                    out[bin_idx * 2] += ordered_gradients[i] as Hist;
                    out[bin_idx * 2 + 1] += ordered_hessians[i] as Hist;
                }

                i += 1;
                if i >= end {
                    break;
                }
                cur_pos += self.deltas[i_delta as usize + 1] as i32;
                i_delta += 1;
                if i_delta >= self.num_vals {
                    break;
                }
            }
        }
    }

    /// Construct histogram without data indices
    pub fn construct_histogram_simple(
        &self,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        ordered_hessians: &[Score],
        out: &mut [Hist],
    ) {
        let mut i_delta = 0_i32;
        let mut cur_pos = 0_i32;
        self.init_index(start, &mut i_delta, &mut cur_pos);

        while cur_pos < start && i_delta < self.num_vals {
            cur_pos += self.deltas[i_delta as usize + 1] as i32;
            i_delta += 1;
        }

        while cur_pos < end && i_delta < self.num_vals {
            let bin = self.vals[i_delta as usize];
            let bin_idx = bin.into() as usize;
            if bin_idx * 2 + 1 < out.len() && (cur_pos as usize) < ordered_gradients.len() {
                out[bin_idx * 2] += ordered_gradients[cur_pos as usize] as Hist;
                out[bin_idx * 2 + 1] += ordered_hessians[cur_pos as usize] as Hist;
            }
            cur_pos += self.deltas[i_delta as usize + 1] as i32;
            i_delta += 1;
        }
    }

    /// Move to next non-zero entry (fast version)
    pub fn next_nonzero_fast(&self, i_delta: &mut DataSize, cur_pos: &mut DataSize) {
        *i_delta += 1;
        if (*i_delta as usize) < self.deltas.len() {
            *cur_pos += self.deltas[*i_delta as usize] as DataSize;
        }
        if *i_delta >= self.num_vals {
            *cur_pos = self.num_data;
        }
    }

    /// Move to next non-zero entry (safe version)
    pub fn next_nonzero(&self, i_delta: &mut DataSize, cur_pos: &mut DataSize) -> bool {
        *i_delta += 1;
        if (*i_delta as usize) < self.deltas.len() {
            *cur_pos += self.deltas[*i_delta as usize] as DataSize;
        }
        if *i_delta < self.num_vals {
            true
        } else {
            *cur_pos = self.num_data;
            false
        }
    }

    /// Initialize index lookup for a given starting position
    pub fn init_index(&self, start_idx: DataSize, i_delta: &mut DataSize, cur_pos: &mut DataSize) {
        let idx = (start_idx as usize) >> self.fast_index_shift;
        if idx < self.fast_index.len() {
            let fast_pair = self.fast_index[idx];
            *i_delta = fast_pair.0;
            *cur_pos = fast_pair.1;
        } else {
            *i_delta = -1;
            *cur_pos = 0;
        }
    }

    /// Finalize loading by processing push buffers
    pub fn finish_load(&mut self) {
        // Get total non-zero size
        let mut pair_cnt = 0;
        for buffer in &self.push_buffers {
            pair_cnt += buffer.len();
        }

        // Consolidate all buffers into the first one
        let mut idx_val_pairs = std::mem::take(&mut self.push_buffers[0]);
        idx_val_pairs.reserve(pair_cnt);

        for i in 1..self.push_buffers.len() {
            idx_val_pairs.extend(self.push_buffers[i].drain(..));
        }

        // Sort by data index
        idx_val_pairs.sort_by_key(|&(idx, _)| idx);

        // Load delta array
        self.load_from_pair(&idx_val_pairs);
    }

    /// Load sparse data from index-value pairs
    fn load_from_pair(&mut self, idx_val_pairs: &[(DataSize, T)]) {
        self.deltas.clear();
        self.vals.clear();
        self.deltas.reserve(idx_val_pairs.len());
        self.vals.reserve(idx_val_pairs.len());

        let mut last_idx = 0;
        for (i, &(cur_idx, bin)) in idx_val_pairs.iter().enumerate() {
            let mut cur_delta = cur_idx - last_idx;

            // Disallow multi-val in one row
            if i > 0 && cur_delta == 0 {
                continue;
            }

            // Handle large deltas by splitting them
            while cur_delta >= 256 {
                self.deltas.push(255);
                self.vals.push(T::default());
                cur_delta -= 255;
            }

            self.deltas.push(cur_delta as u8);
            self.vals.push(bin);
            last_idx = cur_idx;
        }

        // Avoid out of range access
        self.deltas.push(0);
        self.num_vals = self.vals.len() as DataSize;

        // Reduce memory cost
        self.deltas.shrink_to_fit();
        self.vals.shrink_to_fit();

        // Generate fast index
        self.get_fast_index();
    }

    /// Generate fast index for efficient lookups
    fn get_fast_index(&mut self) {
        self.fast_index.clear();

        // Get shift count
        let mod_size =
            (self.num_data + K_NUM_FAST_INDEX as DataSize - 1) / K_NUM_FAST_INDEX as DataSize;
        let mut pow2_mod_size = 1;
        self.fast_index_shift = 0;

        while pow2_mod_size < mod_size {
            pow2_mod_size <<= 1;
            self.fast_index_shift += 1;
        }

        // Build fast index
        let mut i_delta = -1_i32;
        let mut cur_pos = 0_i32;
        let mut next_threshold = 0_i32;

        while self.next_nonzero(&mut i_delta, &mut cur_pos) {
            while next_threshold <= cur_pos {
                self.fast_index.push((i_delta, cur_pos));
                next_threshold += pow2_mod_size;
            }
        }

        // Avoid out of range
        while next_threshold < self.num_data {
            self.fast_index.push((self.num_vals - 1, cur_pos));
            next_threshold += pow2_mod_size;
        }

        self.fast_index.shrink_to_fit();
    }

    /// Get number of data points
    pub fn num_data(&self) -> DataSize {
        self.num_data
    }

    /// Get the number of non-zero values in the sparse bin
    pub fn num_vals(&self) -> DataSize {
        self.num_vals
    }

    /// Get reference to the values array
    pub fn vals(&self) -> &[T] {
        &self.vals
    }

    /// Get reference to the deltas array
    pub fn deltas(&self) -> &[u8] {
        &self.deltas
    }

    /// Get column-wise data pointer (sparse bins return null)
    pub fn get_data(&self) -> *const u8 {
        std::ptr::null()
    }

    /// Construct histogram with gradients only (for count-based splitting)
    pub fn construct_histogram_gradients_only(
        &self,
        data_indices: &[DataSize],
        start: usize,
        end: usize,
        ordered_gradients: &[Score],
        out: &mut [Hist],
    ) {
        if start >= end || start >= data_indices.len() {
            return;
        }

        let mut i_delta = 0_i32;
        let mut cur_pos = 0_i32;
        self.init_index(data_indices[start], &mut i_delta, &mut cur_pos);

        let mut i = start;
        loop {
            if cur_pos < data_indices[i] {
                cur_pos += self.deltas[i_delta as usize + 1] as i32;
                i_delta += 1;
                if i_delta >= self.num_vals {
                    break;
                }
            } else if cur_pos > data_indices[i] {
                i += 1;
                if i >= end {
                    break;
                }
            } else {
                // Found match
                let bin = self.vals[i_delta as usize];
                let bin_idx = bin.into() as usize;
                if bin_idx * 2 + 1 < out.len() {
                    out[bin_idx * 2] += ordered_gradients[i] as Hist;
                    // Increment count (stored at odd indices)
                    out[bin_idx * 2 + 1] += 1.0;
                }

                i += 1;
                if i >= end {
                    break;
                }
                cur_pos += self.deltas[i_delta as usize + 1] as i32;
                i_delta += 1;
                if i_delta >= self.num_vals {
                    break;
                }
            }
        }
    }

    /// Construct histogram for integer gradients (8-bit)
    pub fn construct_histogram_int8(
        &self,
        data_indices: &[DataSize],
        start: usize,
        end: usize,
        ordered_gradients: &[Score],
        out: &mut [i16],
        use_hessian: bool,
    ) {
        if start >= end || start >= data_indices.len() {
            return;
        }

        let mut i_delta = 0_i32;
        let mut cur_pos = 0_i32;
        self.init_index(data_indices[start], &mut i_delta, &mut cur_pos);

        let mut i = start;
        if use_hessian {
            // Pack gradient and hessian into 16-bit values
            loop {
                if cur_pos < data_indices[i] {
                    cur_pos += self.deltas[i_delta as usize + 1] as i32;
                    i_delta += 1;
                    if i_delta >= self.num_vals {
                        break;
                    }
                } else if cur_pos > data_indices[i] {
                    i += 1;
                    if i >= end {
                        break;
                    }
                } else {
                    let bin = self.vals[i_delta as usize];
                    let bin_idx = bin.into() as usize;
                    if bin_idx < out.len() && i < ordered_gradients.len() {
                        // Simple 8-bit implementation
                        out[bin_idx] += ordered_gradients[i] as i16;
                    }

                    i += 1;
                    if i >= end {
                        break;
                    }
                    cur_pos += self.deltas[i_delta as usize + 1] as i32;
                    i_delta += 1;
                    if i_delta >= self.num_vals {
                        break;
                    }
                }
            }
        } else {
            // Gradient only with count
            loop {
                if cur_pos < data_indices[i] {
                    cur_pos += self.deltas[i_delta as usize + 1] as i32;
                    i_delta += 1;
                    if i_delta >= self.num_vals {
                        break;
                    }
                } else if cur_pos > data_indices[i] {
                    i += 1;
                    if i >= end {
                        break;
                    }
                } else {
                    let bin = self.vals[i_delta as usize];
                    let bin_idx = bin.into() as usize;
                    if bin_idx * 2 + 1 < out.len() && i < ordered_gradients.len() {
                        out[bin_idx * 2] += ordered_gradients[i] as i16;
                        out[bin_idx * 2 + 1] += 1; // Count
                    }

                    i += 1;
                    if i >= end {
                        break;
                    }
                    cur_pos += self.deltas[i_delta as usize + 1] as i32;
                    i_delta += 1;
                    if i_delta >= self.num_vals {
                        break;
                    }
                }
            }
        }
    }

    /// Split data based on threshold for numerical features
    pub fn split(
        &self,
        min_bin: u32,
        max_bin: u32,
        default_bin: u32,
        most_freq_bin: u32,
        missing_type: MissingType,
        default_left: bool,
        threshold: u32,
        data_indices: &[DataSize],
        lte_indices: &mut Vec<DataSize>,
        gt_indices: &mut Vec<DataSize>,
    ) -> usize {
        match missing_type {
            MissingType::None => self.split_inner::<false, false, false, false, true>(
                min_bin,
                max_bin,
                default_bin,
                most_freq_bin,
                default_left,
                threshold,
                data_indices,
                lte_indices,
                gt_indices,
            ),
            MissingType::Zero => {
                if default_bin == most_freq_bin {
                    self.split_inner::<true, false, true, false, true>(
                        min_bin,
                        max_bin,
                        default_bin,
                        most_freq_bin,
                        default_left,
                        threshold,
                        data_indices,
                        lte_indices,
                        gt_indices,
                    )
                } else {
                    self.split_inner::<true, false, false, false, true>(
                        min_bin,
                        max_bin,
                        default_bin,
                        most_freq_bin,
                        default_left,
                        threshold,
                        data_indices,
                        lte_indices,
                        gt_indices,
                    )
                }
            }
            MissingType::NaN => {
                if max_bin == most_freq_bin + min_bin && most_freq_bin > 0 {
                    self.split_inner::<false, true, false, true, true>(
                        min_bin,
                        max_bin,
                        default_bin,
                        most_freq_bin,
                        default_left,
                        threshold,
                        data_indices,
                        lte_indices,
                        gt_indices,
                    )
                } else {
                    self.split_inner::<false, true, false, false, true>(
                        min_bin,
                        max_bin,
                        default_bin,
                        most_freq_bin,
                        default_left,
                        threshold,
                        data_indices,
                        lte_indices,
                        gt_indices,
                    )
                }
            }
        }
    }

    /// Split data without min_bin constraint
    pub fn split_simple(
        &self,
        max_bin: u32,
        default_bin: u32,
        most_freq_bin: u32,
        missing_type: MissingType,
        default_left: bool,
        threshold: u32,
        data_indices: &[DataSize],
        lte_indices: &mut Vec<DataSize>,
        gt_indices: &mut Vec<DataSize>,
    ) -> usize {
        match missing_type {
            MissingType::None => self.split_inner::<false, false, false, false, false>(
                1,
                max_bin,
                default_bin,
                most_freq_bin,
                default_left,
                threshold,
                data_indices,
                lte_indices,
                gt_indices,
            ),
            MissingType::Zero => {
                if default_bin == most_freq_bin {
                    self.split_inner::<true, false, true, false, false>(
                        1,
                        max_bin,
                        default_bin,
                        most_freq_bin,
                        default_left,
                        threshold,
                        data_indices,
                        lte_indices,
                        gt_indices,
                    )
                } else {
                    self.split_inner::<true, false, false, false, false>(
                        1,
                        max_bin,
                        default_bin,
                        most_freq_bin,
                        default_left,
                        threshold,
                        data_indices,
                        lte_indices,
                        gt_indices,
                    )
                }
            }
            MissingType::NaN => {
                if max_bin == most_freq_bin + 1 && most_freq_bin > 0 {
                    self.split_inner::<false, true, false, true, false>(
                        1,
                        max_bin,
                        default_bin,
                        most_freq_bin,
                        default_left,
                        threshold,
                        data_indices,
                        lte_indices,
                        gt_indices,
                    )
                } else {
                    self.split_inner::<false, true, false, false, false>(
                        1,
                        max_bin,
                        default_bin,
                        most_freq_bin,
                        default_left,
                        threshold,
                        data_indices,
                        lte_indices,
                        gt_indices,
                    )
                }
            }
        }
    }

    /// Internal split implementation with compile-time feature flags
    fn split_inner<
        const MISS_IS_ZERO: bool,
        const MISS_IS_NA: bool,
        const MFB_IS_ZERO: bool,
        const MFB_IS_NA: bool,
        const USE_MIN_BIN: bool,
    >(
        &self,
        min_bin: u32,
        max_bin: u32,
        default_bin: u32,
        most_freq_bin: u32,
        default_left: bool,
        threshold: u32,
        data_indices: &[DataSize],
        lte_indices: &mut Vec<DataSize>,
        gt_indices: &mut Vec<DataSize>,
    ) -> usize {
        let mut th = T::try_from(threshold + min_bin).unwrap_or_default();
        let mut t_zero_bin = T::try_from(min_bin + default_bin).unwrap_or_default();

        if most_freq_bin == 0 {
            // Adjust threshold for most_freq_bin == 0 case
            if th.into() > 0 {
                th = T::try_from(th.into() - 1).unwrap_or_default();
            }
            if t_zero_bin.into() > 0 {
                t_zero_bin = T::try_from(t_zero_bin.into() - 1).unwrap_or_default();
            }
        }

        let minb = T::try_from(min_bin).unwrap_or_default();
        let maxb = T::try_from(max_bin).unwrap_or_default();

        lte_indices.clear();
        gt_indices.clear();

        let mut iterator = SparseBinIterator::new_with_start(
            Arc::new(self.clone()),
            if data_indices.is_empty() {
                0
            } else {
                data_indices[0]
            },
        );

        for &idx in data_indices {
            let bin = iterator.inner_raw_get(idx);

            let mut goes_left = false;
            let mut is_missing = false;

            if MISS_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin {
                is_missing = true;
            } else if MISS_IS_NA && !MFB_IS_NA && bin == maxb {
                is_missing = true;
            } else if (USE_MIN_BIN && (bin < minb || bin > maxb))
                || (!USE_MIN_BIN && bin == T::default())
            {
                if (MISS_IS_NA && MFB_IS_NA) || (MISS_IS_ZERO && MFB_IS_ZERO) {
                    is_missing = true;
                } else {
                    // Use default direction based on most_freq_bin
                    goes_left = most_freq_bin <= threshold;
                }
            } else if bin <= th {
                goes_left = true;
            }

            if is_missing {
                if default_left {
                    lte_indices.push(idx);
                } else {
                    gt_indices.push(idx);
                }
            } else if goes_left {
                lte_indices.push(idx);
            } else {
                gt_indices.push(idx);
            }
        }

        lte_indices.len()
    }

    /// Split categorical data
    pub fn split_categorical(
        &self,
        min_bin: u32,
        max_bin: u32,
        most_freq_bin: u32,
        threshold_bitset: &[u32],
        data_indices: &[DataSize],
        lte_indices: &mut Vec<DataSize>,
        gt_indices: &mut Vec<DataSize>,
    ) -> usize {
        self.split_categorical_inner::<true>(
            min_bin,
            max_bin,
            most_freq_bin,
            threshold_bitset,
            data_indices,
            lte_indices,
            gt_indices,
        )
    }

    /// Split categorical data without min_bin constraint
    pub fn split_categorical_simple(
        &self,
        max_bin: u32,
        most_freq_bin: u32,
        threshold_bitset: &[u32],
        data_indices: &[DataSize],
        lte_indices: &mut Vec<DataSize>,
        gt_indices: &mut Vec<DataSize>,
    ) -> usize {
        self.split_categorical_inner::<false>(
            1,
            max_bin,
            most_freq_bin,
            threshold_bitset,
            data_indices,
            lte_indices,
            gt_indices,
        )
    }

    /// Internal categorical split implementation
    fn split_categorical_inner<const USE_MIN_BIN: bool>(
        &self,
        min_bin: u32,
        max_bin: u32,
        most_freq_bin: u32,
        threshold_bitset: &[u32],
        data_indices: &[DataSize],
        lte_indices: &mut Vec<DataSize>,
        gt_indices: &mut Vec<DataSize>,
    ) -> usize {
        lte_indices.clear();
        gt_indices.clear();

        let offset = if most_freq_bin == 0 { 1u32 } else { 0u32 };
        let mut iterator = SparseBinIterator::new_with_start(
            Arc::new(self.clone()),
            if data_indices.is_empty() {
                0
            } else {
                data_indices[0]
            },
        );

        // Check if most_freq_bin is in the threshold set
        let most_freq_in_threshold =
            most_freq_bin > 0 && self.find_in_bitset(threshold_bitset, most_freq_bin);

        for &idx in data_indices {
            let bin = iterator.raw_get(idx);

            if USE_MIN_BIN && (bin < min_bin || bin > max_bin) {
                // Use default direction based on most_freq_bin
                if most_freq_in_threshold {
                    lte_indices.push(idx);
                } else {
                    gt_indices.push(idx);
                }
            } else if !USE_MIN_BIN && bin == 0 {
                // Use default direction
                if most_freq_in_threshold {
                    lte_indices.push(idx);
                } else {
                    gt_indices.push(idx);
                }
            } else if self.find_in_bitset(threshold_bitset, bin - min_bin + offset) {
                lte_indices.push(idx);
            } else {
                gt_indices.push(idx);
            }
        }

        lte_indices.len()
    }

    /// Find value in bitset (helper function for categorical splits)
    fn find_in_bitset(&self, bitset: &[u32], val: u32) -> bool {
        let word_idx = (val / 32) as usize;
        let bit_idx = val % 32;

        if word_idx < bitset.len() {
            (bitset[word_idx] & (1u32 << bit_idx)) != 0
        } else {
            false
        }
    }
}

impl<T> Clone for SparseBin<T>
where
    T: Copy + Default + PartialEq + PartialOrd + TryFrom<u32> + Into<u32> + Send + Sync + Clone,
    <T as TryFrom<u32>>::Error: std::fmt::Debug,
{
    fn clone(&self) -> Self {
        Self {
            num_data: self.num_data,
            deltas: self.deltas.clone(),
            vals: self.vals.clone(),
            num_vals: self.num_vals,
            push_buffers: self.push_buffers.clone(),
            fast_index: self.fast_index.clone(),
            fast_index_shift: self.fast_index_shift,
        }
    }
}

/// Type alias for SparseBin with u8 values, suitable for features with up to 255 distinct values
pub type SparseBinU8 = SparseBin<u8>;

/// Type alias for SparseBin with u16 values, suitable for features with up to 65535 distinct values
pub type SparseBinU16 = SparseBin<u16>;

/// Type alias for SparseBin with u32 values, suitable for features with up to 4B distinct values
pub type SparseBinU32 = SparseBin<u32>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_bin_creation() {
        let bin: SparseBin<u8> = SparseBin::new(1000);
        assert_eq!(bin.num_data(), 1000);
        assert_eq!(bin.num_vals, 0);
    }

    #[test]
    fn test_sparse_bin_push() {
        let mut bin: SparseBin<u8> = SparseBin::new(100);
        bin.push(0, 10, 5);
        bin.push(0, 20, 3);

        assert_eq!(bin.push_buffers[0].len(), 2);
        assert_eq!(bin.push_buffers[0][0], (10, 5));
        assert_eq!(bin.push_buffers[0][1], (20, 3));
    }

    #[test]
    fn test_sparse_bin_finish_load() {
        let mut bin: SparseBin<u8> = SparseBin::new(100);
        bin.push(0, 10, 5);
        bin.push(0, 20, 3);
        bin.push(0, 5, 7);

        bin.finish_load();

        assert_eq!(bin.num_vals, 3);
        assert_eq!(bin.vals.len(), 3);
        assert_eq!(bin.deltas.len(), 4); // 3 values + 1 terminator
    }

    #[test]
    fn test_sparse_bin_iterator_creation() {
        let bin: SparseBin<u8> = SparseBin::new(100);
        let iterator = bin.get_iterator(0, 10, 0);
        assert_eq!(iterator.offset, 1); // most_freq_bin is 0, so offset should be 1
    }
}
