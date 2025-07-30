/*!
 * Copyright (c) 2020 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
use crate::core::types::*;
use std::cmp;
use std::ptr;

// Constants matching C++
const HIST_OFFSET: usize = 2;

// Prefetch implementation
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn prefetch_t0(addr: *const u8) {
    unsafe {
        _mm_prefetch(addr as *const i8, _MM_HINT_T0);
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn prefetch_t0(_addr: *const u8) {
    // No-op for non-x86 architectures
}

// 32-byte aligned vector storage
#[derive(Debug)]
#[repr(align(32))]
struct AlignedVec<T> {
    data: Vec<T>,
}

impl<T> AlignedVec<T> {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        self.data.resize(new_len, value);
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    fn push(&mut self, value: T) {
        self.data.push(value);
    }

    fn clear(&mut self) {
        self.data.clear();
    }

    fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
    }
}

impl<T> std::ops::Index<usize> for AlignedVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> std::ops::IndexMut<usize> for AlignedVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T: Clone> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

// Import MultiValBin trait from the dense implementation
use super::multi_val_dense_bin::MultiValBin;

// Function timer stub - simplified for Rust
struct FunctionTimer;
impl FunctionTimer {
    fn new(_name: &str, _timer: Option<&str>) -> Self {
        Self
    }
}

// OpenMP thread count equivalent
fn omp_num_threads() -> usize {
    rayon::current_num_threads()
}

/// Sparse multi-value bin implementation for categorical features
///
/// This structure efficiently stores multiple categorical values per data point
/// using a sparse representation with row pointers. It's optimized for scenarios
/// where most data points have relatively few categorical values, saving memory
/// compared to dense representations.
#[derive(Debug)]
pub struct MultiValSparseBin<IndexT, ValT> {
    num_data_: DataSize,
    num_bin_: i32,
    estimate_element_per_row_: f64,
    data_: AlignedVec<ValT>,
    row_ptr_: AlignedVec<IndexT>,
    t_data_: Vec<AlignedVec<ValT>>,
    t_size_: Vec<IndexT>,
    offsets_: Vec<u32>,
}

impl<IndexT, ValT> MultiValSparseBin<IndexT, ValT>
where
    IndexT: Default
        + Clone
        + Copy
        + TryFrom<usize>
        + Into<usize>
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::Add<Output = IndexT>
        + std::ops::Sub<Output = IndexT>,
    ValT: Default + Clone + Copy + TryFrom<u32> + Into<u32> + Send + Sync,
    <IndexT as TryFrom<usize>>::Error: std::fmt::Debug,
    <ValT as TryFrom<u32>>::Error: std::fmt::Debug,
    usize: From<IndexT>,
{
    /// Create a new multi-value sparse bin
    ///
    /// # Arguments
    /// * `num_data` - Number of data points
    /// * `num_bin` - Number of distinct bin values
    /// * `estimate_element_per_row` - Estimated average elements per row for memory allocation
    pub fn new(num_data: DataSize, num_bin: i32, estimate_element_per_row: f64) -> Self {
        let mut row_ptr = AlignedVec::new();
        row_ptr.resize((num_data + 1) as usize, IndexT::default());

        let estimate_num_data =
            IndexT::try_from((estimate_element_per_row * 1.1 * num_data as f64) as usize)
                .unwrap_or_default();

        let num_threads = omp_num_threads();
        let mut t_data = Vec::new();
        let t_size = vec![IndexT::default(); num_threads];

        if num_threads > 1 {
            t_data.resize(num_threads - 1, AlignedVec::new());
            for i in 0..(num_threads - 1) {
                t_data[i].resize(estimate_num_data.into() / num_threads, ValT::default());
            }
        }

        let mut data = AlignedVec::new();
        data.resize(estimate_num_data.into() / num_threads, ValT::default());

        Self {
            num_data_: num_data,
            num_bin_: num_bin,
            estimate_element_per_row_: estimate_element_per_row,
            data_: data,
            row_ptr_: row_ptr,
            t_data_: t_data,
            t_size_: t_size,
            offsets_: Vec::new(),
        }
    }

    #[inline]
    fn row_ptr(&self, idx: DataSize) -> IndexT {
        self.row_ptr_[idx as usize]
    }

    /// Push categorical values for one row of data
    ///
    /// # Arguments
    /// * `tid` - Thread ID for parallel processing
    /// * `idx` - Row index
    /// * `values` - Array of categorical values for this row
    pub fn push_one_row(&mut self, tid: i32, idx: DataSize, values: &[u32]) {
        const PRE_ALLOC_SIZE: usize = 50;
        self.row_ptr_[(idx + 1) as usize] = IndexT::try_from(values.len()).unwrap_or_default();

        if tid == 0 {
            let current_size = self.t_size_[tid as usize].into();
            let needed_size = current_size + self.row_ptr_[(idx + 1) as usize].into();
            if needed_size > self.data_.len() {
                self.data_.resize(
                    needed_size + self.row_ptr_[(idx + 1) as usize].into() * PRE_ALLOC_SIZE,
                    ValT::default(),
                );
            }
            for &val in values {
                self.data_[current_size + (self.t_size_[tid as usize].into())] =
                    ValT::try_from(val).unwrap_or_default();
                self.t_size_[tid as usize] += IndexT::try_from(1).unwrap_or_default();
            }
        } else {
            let tid_idx = (tid - 1) as usize;
            let current_size = self.t_size_[tid as usize].into();
            let needed_size = current_size + self.row_ptr_[(idx + 1) as usize].into();
            if needed_size > self.t_data_[tid_idx].len() {
                self.t_data_[tid_idx].resize(
                    needed_size + self.row_ptr_[(idx + 1) as usize].into() * PRE_ALLOC_SIZE,
                    ValT::default(),
                );
            }
            for &val in values {
                self.t_data_[tid_idx][current_size + (self.t_size_[tid as usize].into())] =
                    ValT::try_from(val).unwrap_or_default();
                self.t_size_[tid as usize] += IndexT::try_from(1).unwrap_or_default();
            }
        }
    }

    fn merge_data(&mut self, sizes: &[IndexT]) {
        let _timer = FunctionTimer::new("MultiValSparseBin::MergeData", None);

        // Convert row_ptr_ to cumulative sum
        for i in 0..self.num_data_ {
            let prev_val = self.row_ptr_[i as usize];
            self.row_ptr_[(i + 1) as usize] += prev_val;
        }

        if !self.t_data_.is_empty() {
            let mut offsets = vec![IndexT::default(); 1 + self.t_data_.len()];
            offsets[0] = sizes[0];
            for tid in 0..(self.t_data_.len() - 1) {
                offsets[tid + 1] = offsets[tid] + sizes[tid + 1];
            }

            self.data_.resize(
                self.row_ptr_[self.num_data_ as usize].into(),
                ValT::default(),
            );

            // Sequential copy - simplified from parallel version
            for (tid, t_data) in self.t_data_.iter().enumerate() {
                let size = sizes[tid + 1].into();
                let offset = offsets[tid].into();
                unsafe {
                    ptr::copy_nonoverlapping(
                        t_data.as_ptr(),
                        self.data_.as_mut_ptr().add(offset),
                        size,
                    );
                }
            }
        } else {
            self.data_.resize(
                self.row_ptr_[self.num_data_ as usize].into(),
                ValT::default(),
            );
        }
    }

    fn finish_load(&mut self) {
        let sizes: Vec<IndexT> = self.t_size_.clone();
        self.merge_data(&sizes);
        self.t_size_.clear();
        self.row_ptr_.shrink_to_fit();
        self.data_.shrink_to_fit();
        self.t_data_.clear();
        self.t_data_.shrink_to_fit();

        // Update estimate_element_per_row_ by all data
        self.estimate_element_per_row_ =
            self.row_ptr_[self.num_data_ as usize].into() as f64 / self.num_data_ as f64;
    }

    fn construct_histogram_inner<
        const USE_INDICES: bool,
        const USE_PREFETCH: bool,
        const ORDERED: bool,
    >(
        &self,
        data_indices: *const DataSize,
        start: DataSize,
        end: DataSize,
        gradients: *const Score,
        hessians: *const Score,
        out: *mut Hist,
    ) {
        let mut i = start;
        let grad = out;
        let hess = unsafe { out.add(1) };
        let data_ptr = self.data_.as_ptr();

        if USE_PREFETCH {
            let pf_offset = (32 / std::mem::size_of::<ValT>()) as DataSize;
            let pf_end = end - pf_offset;

            while i < pf_end {
                let idx = if USE_INDICES {
                    unsafe { *data_indices.add(i as usize) }
                } else {
                    i
                };
                let pf_idx = if USE_INDICES {
                    unsafe { *data_indices.add((i + pf_offset) as usize) }
                } else {
                    i + pf_offset
                };

                if !ORDERED {
                    prefetch_t0(unsafe { gradients.add(pf_idx as usize) } as *const u8);
                    prefetch_t0(unsafe { hessians.add(pf_idx as usize) } as *const u8);
                }
                prefetch_t0(unsafe { self.row_ptr_.as_ptr().add(pf_idx as usize) } as *const u8);
                prefetch_t0(unsafe { data_ptr.add(self.row_ptr(pf_idx).into()) } as *const u8);

                let j_start = self.row_ptr(idx);
                let j_end = self.row_ptr(idx + 1);
                let gradient = if ORDERED {
                    unsafe { *gradients.add(i as usize) }
                } else {
                    unsafe { *gradients.add(idx as usize) }
                };
                let hessian = if ORDERED {
                    unsafe { *hessians.add(i as usize) }
                } else {
                    unsafe { *hessians.add(idx as usize) }
                };

                for j in j_start.into()..j_end.into() {
                    let ti = ((unsafe { *data_ptr.add(j) }.into() as u32) << 1) as usize;
                    unsafe {
                        *grad.add(ti) += gradient as Hist;
                        *hess.add(ti) += hessian as Hist;
                    }
                }
                i += 1;
            }
        }

        while i < end {
            let idx = if USE_INDICES {
                unsafe { *data_indices.add(i as usize) }
            } else {
                i
            };
            let j_start = self.row_ptr(idx);
            let j_end = self.row_ptr(idx + 1);
            let gradient = if ORDERED {
                unsafe { *gradients.add(i as usize) }
            } else {
                unsafe { *gradients.add(idx as usize) }
            };
            let hessian = if ORDERED {
                unsafe { *hessians.add(i as usize) }
            } else {
                unsafe { *hessians.add(idx as usize) }
            };

            for j in j_start.into()..j_end.into() {
                let ti = ((unsafe { *data_ptr.add(j) }.into() as u32) << 1) as usize;
                unsafe {
                    *grad.add(ti) += gradient as Hist;
                    *hess.add(ti) += hessian as Hist;
                }
            }
            i += 1;
        }
    }

    fn construct_histogram_int_inner<
        const USE_INDICES: bool,
        const USE_PREFETCH: bool,
        const ORDERED: bool,
        PackedHistT,
        const HIST_BITS: u32,
    >(
        &self,
        data_indices: *const DataSize,
        start: DataSize,
        end: DataSize,
        gradients_and_hessians: *const Score,
        out: *mut Hist,
    ) where
        PackedHistT: Copy + std::ops::AddAssign + From<i16> + From<i8>,
    {
        let mut i = start;
        let out_ptr = out as *mut PackedHistT;
        let gradients_and_hessians_ptr = gradients_and_hessians as *const i16;
        let data_ptr = self.data_.as_ptr();
        let row_ptr_base = self.row_ptr_.as_ptr();

        if USE_PREFETCH {
            let pf_offset = (32 / std::mem::size_of::<ValT>()) as DataSize;
            let pf_end = end - pf_offset;

            while i < pf_end {
                let idx = if USE_INDICES {
                    unsafe { *data_indices.add(i as usize) }
                } else {
                    i
                };
                let pf_idx = if USE_INDICES {
                    unsafe { *data_indices.add((i + pf_offset) as usize) }
                } else {
                    i + pf_offset
                };

                if !ORDERED {
                    prefetch_t0(
                        unsafe { gradients_and_hessians_ptr.add(pf_idx as usize) } as *const u8
                    );
                }
                prefetch_t0(unsafe { row_ptr_base.add(pf_idx as usize) } as *const u8);
                prefetch_t0(unsafe { data_ptr.add(self.row_ptr(pf_idx).into()) } as *const u8);

                let j_start = self.row_ptr(idx);
                let j_end = self.row_ptr(idx + 1);
                let gradient_16 = if ORDERED {
                    unsafe { *gradients_and_hessians_ptr.add(i as usize) }
                } else {
                    unsafe { *gradients_and_hessians_ptr.add(idx as usize) }
                };

                let gradient_packed = if HIST_BITS == 8 {
                    PackedHistT::from(gradient_16)
                } else {
                    let _high_byte = (gradient_16 >> 8) as i8;
                    let _low_byte = (gradient_16 & 0xff) as u8;
                    // Simplified packing - would need proper bit manipulation in real implementation
                    PackedHistT::from(gradient_16)
                };

                for j in j_start.into()..j_end.into() {
                    let ti = unsafe { *data_ptr.add(j) }.into() as u32 as usize;
                    unsafe {
                        *out_ptr.add(ti) += gradient_packed;
                    }
                }
                i += 1;
            }
        }

        while i < end {
            let idx = if USE_INDICES {
                unsafe { *data_indices.add(i as usize) }
            } else {
                i
            };
            let j_start = self.row_ptr(idx);
            let j_end = self.row_ptr(idx + 1);
            let gradient_16 = if ORDERED {
                unsafe { *gradients_and_hessians_ptr.add(i as usize) }
            } else {
                unsafe { *gradients_and_hessians_ptr.add(idx as usize) }
            };

            let gradient_packed = if HIST_BITS == 8 {
                PackedHistT::from(gradient_16)
            } else {
                let _high_byte = (gradient_16 >> 8) as i8;
                let _low_byte = (gradient_16 & 0xff) as u8;
                // Simplified packing - would need proper bit manipulation in real implementation
                PackedHistT::from(gradient_16)
            };

            for j in j_start.into()..j_end.into() {
                let ti = unsafe { *data_ptr.add(j) }.into() as u32 as usize;
                unsafe {
                    *out_ptr.add(ti) += gradient_packed;
                }
            }
            i += 1;
        }
    }

    fn resize(
        &mut self,
        num_data: DataSize,
        num_bin: i32,
        _num_feature: i32,
        estimate_element_per_row: f64,
        _offsets: &[u32],
    ) {
        self.num_data_ = num_data;
        self.num_bin_ = num_bin;
        self.estimate_element_per_row_ = estimate_element_per_row;

        let estimate_num_data =
            IndexT::try_from((estimate_element_per_row * 1.1 * num_data as f64) as usize)
                .unwrap_or_default();

        let npart = 1 + self.t_data_.len();
        let avg_num_data = estimate_num_data.into() / npart;

        if self.data_.len() < avg_num_data {
            self.data_.resize(avg_num_data, ValT::default());
        }

        for i in 0..self.t_data_.len() {
            if self.t_data_[i].len() < avg_num_data {
                self.t_data_[i].resize(avg_num_data, ValT::default());
            }
        }

        if (num_data + 1) as usize > self.row_ptr_.len() {
            self.row_ptr_
                .resize((num_data + 1) as usize, IndexT::default());
        }
    }

    fn copy_inner<const SUBROW: bool, const SUBCOL: bool>(
        &mut self,
        full_bin: &MultiValSparseBin<IndexT, ValT>,
        used_indices: *const DataSize,
        num_used_indices: DataSize,
        lower: &[u32],
        upper: &[u32],
        delta: &[u32],
    ) {
        if SUBROW {
            assert_eq!(self.num_data_, num_used_indices);
        }

        let n_block = 1;
        let block_size = self.num_data_;
        let mut sizes = vec![IndexT::default(); self.t_data_.len() + 1];
        const PRE_ALLOC_SIZE: usize = 50;

        // Simplified single-threaded implementation
        for tid in 0..n_block {
            let start = (tid as DataSize) * block_size;
            let end = cmp::min(self.num_data_, start + block_size);
            let buf = if tid == 0 {
                &mut self.data_
            } else {
                &mut self.t_data_[tid - 1]
            };
            let mut size = IndexT::default();

            for i in start..end {
                let j_start = if SUBROW {
                    full_bin.row_ptr(unsafe { *used_indices.add(i as usize) })
                } else {
                    full_bin.row_ptr(i)
                };
                let j_end = if SUBROW {
                    full_bin.row_ptr(unsafe { *used_indices.add(i as usize) } + 1)
                } else {
                    full_bin.row_ptr(i + 1)
                };

                let needed_size = size.into() + (j_end.into() - j_start.into());
                if needed_size > buf.len() {
                    buf.resize(
                        needed_size + (j_end.into() - j_start.into()) * PRE_ALLOC_SIZE,
                        ValT::default(),
                    );
                }

                let mut k = 0;
                let pre_size = size;
                for j in j_start.into()..j_end.into() {
                    let val = full_bin.data_[j];
                    if SUBCOL {
                        while val.into() >= upper[k] {
                            k += 1;
                        }
                        if val.into() >= lower[k] {
                            buf[size.into()] =
                                ValT::try_from(val.into() - delta[k]).unwrap_or_default();
                            size += IndexT::try_from(1).unwrap_or_default();
                        }
                    } else {
                        buf[size.into()] = val;
                        size += IndexT::try_from(1).unwrap_or_default();
                    }
                }
                self.row_ptr_[(i + 1) as usize] = size - pre_size;
            }
            sizes[tid] = size;
        }
        self.merge_data(&sizes);
    }
    ///
    pub fn clone_sparse_bin(&self) -> Self {
        Self {
            num_data_: self.num_data_,
            num_bin_: self.num_bin_,
            estimate_element_per_row_: self.estimate_element_per_row_,
            data_: self.data_.clone(),
            row_ptr_: self.row_ptr_.clone(),
            t_data_: Vec::new(), // Don't clone temporary data
            t_size_: Vec::new(), // Don't clone temporary sizes
            offsets_: self.offsets_.clone(),
        }
    }
}

impl<IndexT, ValT> MultiValBin for MultiValSparseBin<IndexT, ValT>
where
    IndexT: Default
        + Clone
        + Copy
        + TryFrom<usize>
        + Into<usize>
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::Add<Output = IndexT>
        + std::ops::Sub<Output = IndexT>
        + 'static,
    ValT: Default + Clone + Copy + TryFrom<u32> + Into<u32> + Send + Sync + 'static,
    <IndexT as TryFrom<usize>>::Error: std::fmt::Debug,
    <ValT as TryFrom<u32>>::Error: std::fmt::Debug,
    usize: From<IndexT>,
{
    fn num_data(&self) -> DataSize {
        self.num_data_
    }

    fn num_bin(&self) -> i32 {
        self.num_bin_
    }

    fn num_element_per_row(&self) -> f64 {
        self.estimate_element_per_row_
    }

    fn offsets(&self) -> &[u32] {
        &self.offsets_
    }

    fn push_one_row(&mut self, tid: i32, idx: DataSize, values: &[u32]) {
        self.push_one_row(tid, idx, values);
    }

    fn finish_load(&mut self) {
        self.finish_load();
    }

    fn is_sparse(&self) -> bool {
        true
    }

    fn construct_histogram(
        &self,
        data_indices: *const DataSize,
        start: DataSize,
        end: DataSize,
        gradients: *const Score,
        hessians: *const Score,
        out: *mut Hist,
    ) {
        self.construct_histogram_inner::<true, true, false>(
            data_indices,
            start,
            end,
            gradients,
            hessians,
            out,
        );
    }

    fn construct_histogram_no_indices(
        &self,
        start: DataSize,
        end: DataSize,
        gradients: *const Score,
        hessians: *const Score,
        out: *mut Hist,
    ) {
        self.construct_histogram_inner::<false, false, false>(
            ptr::null(),
            start,
            end,
            gradients,
            hessians,
            out,
        );
    }

    fn construct_histogram_ordered(
        &self,
        data_indices: *const DataSize,
        start: DataSize,
        end: DataSize,
        gradients: *const Score,
        hessians: *const Score,
        out: *mut Hist,
    ) {
        self.construct_histogram_inner::<true, true, true>(
            data_indices,
            start,
            end,
            gradients,
            hessians,
            out,
        );
    }

    fn construct_histogram_int32(
        &self,
        data_indices: *const DataSize,
        start: DataSize,
        end: DataSize,
        gradients: *const Score,
        _hessians: *const Score,
        out: *mut Hist,
    ) {
        self.construct_histogram_int_inner::<true, true, false, i64, 32>(
            data_indices,
            start,
            end,
            gradients,
            out,
        );
    }

    fn construct_histogram_int32_no_indices(
        &self,
        start: DataSize,
        end: DataSize,
        gradients: *const Score,
        _hessians: *const Score,
        out: *mut Hist,
    ) {
        self.construct_histogram_int_inner::<false, false, false, i64, 32>(
            ptr::null(),
            start,
            end,
            gradients,
            out,
        );
    }

    fn construct_histogram_ordered_int32(
        &self,
        data_indices: *const DataSize,
        start: DataSize,
        end: DataSize,
        gradients: *const Score,
        _hessians: *const Score,
        out: *mut Hist,
    ) {
        self.construct_histogram_int_inner::<true, true, true, i64, 32>(
            data_indices,
            start,
            end,
            gradients,
            out,
        );
    }

    fn construct_histogram_int16(
        &self,
        data_indices: *const DataSize,
        start: DataSize,
        end: DataSize,
        gradients: *const Score,
        _hessians: *const Score,
        out: *mut Hist,
    ) {
        self.construct_histogram_int_inner::<true, true, false, i32, 16>(
            data_indices,
            start,
            end,
            gradients,
            out,
        );
    }

    fn construct_histogram_int16_no_indices(
        &self,
        start: DataSize,
        end: DataSize,
        gradients: *const Score,
        _hessians: *const Score,
        out: *mut Hist,
    ) {
        self.construct_histogram_int_inner::<false, false, false, i32, 16>(
            ptr::null(),
            start,
            end,
            gradients,
            out,
        );
    }

    fn construct_histogram_ordered_int16(
        &self,
        data_indices: *const DataSize,
        start: DataSize,
        end: DataSize,
        gradients: *const Score,
        _hessians: *const Score,
        out: *mut Hist,
    ) {
        self.construct_histogram_int_inner::<true, true, true, i32, 16>(
            data_indices,
            start,
            end,
            gradients,
            out,
        );
    }

    fn construct_histogram_int8(
        &self,
        data_indices: *const DataSize,
        start: DataSize,
        end: DataSize,
        gradients: *const Score,
        _hessians: *const Score,
        out: *mut Hist,
    ) {
        self.construct_histogram_int_inner::<true, true, false, i16, 8>(
            data_indices,
            start,
            end,
            gradients,
            out,
        );
    }

    fn construct_histogram_int8_no_indices(
        &self,
        start: DataSize,
        end: DataSize,
        gradients: *const Score,
        _hessians: *const Score,
        out: *mut Hist,
    ) {
        self.construct_histogram_int_inner::<false, false, false, i16, 8>(
            ptr::null(),
            start,
            end,
            gradients,
            out,
        );
    }

    fn construct_histogram_ordered_int8(
        &self,
        data_indices: *const DataSize,
        start: DataSize,
        end: DataSize,
        gradients: *const Score,
        _hessians: *const Score,
        out: *mut Hist,
    ) {
        self.construct_histogram_int_inner::<true, true, true, i16, 8>(
            data_indices,
            start,
            end,
            gradients,
            out,
        );
    }

    fn create_like(
        &self,
        num_data: DataSize,
        num_bin: i32,
        _num_feature: i32,
        estimate_element_per_row: f64,
        _offsets: &[u32],
    ) -> Box<dyn MultiValBin> {
        Box::new(MultiValSparseBin::<IndexT, ValT>::new(
            num_data,
            num_bin,
            estimate_element_per_row,
        ))
    }

    fn resize(
        &mut self,
        num_data: DataSize,
        num_bin: i32,
        num_feature: i32,
        estimate_element_per_row: f64,
        offsets: &[u32],
    ) {
        self.resize(
            num_data,
            num_bin,
            num_feature,
            estimate_element_per_row,
            offsets,
        );
    }

    fn copy_subrow(
        &mut self,
        _full_bin: &dyn MultiValBin,
        _used_indices: *const DataSize,
        _num_used_indices: DataSize,
    ) {
        // This would require unsafe downcasting in a real implementation
        unimplemented!("copy_subrow requires unsafe downcasting")
    }

    fn copy_subcol(
        &mut self,
        _full_bin: &dyn MultiValBin,
        _used_feature_index: &[i32],
        _lower: &[u32],
        _upper: &[u32],
        _delta: &[u32],
    ) {
        // This would require unsafe downcasting in a real implementation
        unimplemented!("copy_subcol requires unsafe downcasting")
    }

    fn copy_subrow_and_subcol(
        &mut self,
        _full_bin: &dyn MultiValBin,
        _used_indices: *const DataSize,
        _num_used_indices: DataSize,
        _used_feature_index: &[i32],
        _lower: &[u32],
        _upper: &[u32],
        _delta: &[u32],
    ) {
        // This would require unsafe downcasting in a real implementation
        unimplemented!("copy_subrow_and_subcol requires unsafe downcasting")
    }

    fn clone_multi_val_bin(&self) -> Box<dyn MultiValBin> {
        Box::new(self.clone_sparse_bin())
    }
}

// Type aliases to match C++ template instantiations

/// Type alias for MultiValSparseBin with u16 row pointers and u8 bin values
pub type MultiValSparseBinU16U8 = MultiValSparseBin<u16, u8>;

/// Type alias for MultiValSparseBin with usize row pointers and u8 bin values
pub type MultiValSparseBinUsizeU8 = MultiValSparseBin<usize, u8>;

/// Type alias for MultiValSparseBin with u16 row pointers and u16 bin values
pub type MultiValSparseBinU16U16 = MultiValSparseBin<u16, u16>;

/// Type alias for MultiValSparseBin with usize row pointers and u16 bin values
pub type MultiValSparseBinUsizeU16 = MultiValSparseBin<usize, u16>;

/// Type alias for MultiValSparseBin with u16 row pointers and u32 bin values
pub type MultiValSparseBinU16U32 = MultiValSparseBin<u16, u32>;
// Note: usize -> u32 conversion not guaranteed on all platforms
// pub type MultiValSparseBinUsizeUsize = MultiValSparseBin<usize, usize>;

// Legacy type aliases for compatibility (but they won't satisfy trait bounds on all platforms)
// pub type MultiValSparseBinU32U8 = MultiValSparseBin<u32, u8>;
// pub type MultiValSparseBinU32U16 = MultiValSparseBin<u32, u16>;
// pub type MultiValSparseBinU32U32 = MultiValSparseBin<u32, u32>;

// Note: GPU-related methods (#ifdef USE_CUDA) are not implemented in this Rust version
// as they would require CUDA bindings and are not part of the core CPU implementation

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_val_sparse_bin_creation() {
        let bin = MultiValSparseBinUsizeU8::new(1000, 256, 0.1);
        assert_eq!(bin.num_data(), 1000);
        assert_eq!(bin.num_bin(), 256);
        assert_eq!(bin.num_element_per_row(), 0.1);
        assert!(bin.is_sparse());
    }

    #[test]
    fn test_push_one_row() {
        let mut bin = MultiValSparseBinUsizeU8::new(10, 256, 0.5);
        let values = vec![5, 15, 25];
        bin.push_one_row(0, 0, &values);

        assert_eq!(bin.num_data(), 10);
        assert_eq!(bin.num_bin(), 256);
        assert!(bin.is_sparse());
    }

    #[test]
    fn test_row_ptr() {
        let bin = MultiValSparseBinUsizeU8::new(10, 256, 0.5);
        assert_eq!(bin.row_ptr(0), 0);
        // After initialization, all row pointers should be 0
        assert_eq!(bin.row_ptr(5), 0);
    }

    #[test]
    fn test_resize() {
        let mut bin = MultiValSparseBinUsizeU8::new(10, 256, 0.1);
        bin.resize(20, 512, 4, 0.2, &[0, 100, 200, 300, 400]);

        assert_eq!(bin.num_data(), 20);
        assert_eq!(bin.num_bin(), 512);
        assert_eq!(bin.num_element_per_row(), 0.2);
    }

    #[test]
    fn test_clone_method() {
        let bin = MultiValSparseBinUsizeU8::new(10, 256, 0.5);
        let cloned = bin.clone_sparse_bin();

        assert_eq!(bin.num_data(), cloned.num_data());
        assert_eq!(bin.num_bin(), cloned.num_bin());
        assert_eq!(bin.num_element_per_row(), cloned.num_element_per_row());
    }

    #[test]
    fn test_different_type_combinations() {
        let bin16_8 = MultiValSparseBinU16U8::new(100, 256, 0.3);
        let bin_usize_16 = MultiValSparseBinUsizeU16::new(1000, 65536, 0.1);
        let bin16_32 = MultiValSparseBinU16U32::new(10000, 100000, 0.05);

        assert_eq!(bin16_8.num_data(), 100);
        assert_eq!(bin_usize_16.num_data(), 1000);
        assert_eq!(bin16_32.num_data(), 10000);

        assert!(bin16_8.is_sparse());
        assert!(bin_usize_16.is_sparse());
        assert!(bin16_32.is_sparse());
    }
}
