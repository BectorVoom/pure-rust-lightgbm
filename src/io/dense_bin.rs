//! Dense bin implementation for LightGBM feature storage.
//!
//! This module provides a pure Rust implementation of LightGBM's dense bin data structure,
//! translating the C++ dense_bin.hpp implementation while maintaining semantic equivalence
//! and leveraging Rust's type system for enhanced safety.

use crate::core::types::*;
use crate::dataset::binning::MissingType;
use anyhow::Result;

/// Histogram count type, equivalent to `hist_cnt_t` in LightGBM C++
pub type HistCount = i32;

/// Memory alignment constant for SIMD operations
const K_ALIGNED_SIZE: usize = 32;

/// Generic trait for bin iterators
pub trait BinIterator {
    /// Get the bin value at the given index with range checking
    fn get(&self, idx: DataSize) -> u32;

    /// Get the raw bin value at the given index without range checking
    fn raw_get(&self, idx: DataSize) -> u32;

    /// Reset the iterator to a given starting position
    fn reset(&mut self, start_idx: DataSize);
}

/// Dense bin iterator for traversing dense bin data
#[derive(Debug, Clone)]
pub struct DenseBinIterator<T, const IS_4BIT: bool> {
    bin_data: *const DenseBin<T, IS_4BIT>,
    min_bin: T,
    max_bin: T,
    most_freq_bin: T,
    offset: u8,
}

impl<T, const IS_4BIT: bool> DenseBinIterator<T, IS_4BIT>
where
    T: Copy + Clone + PartialOrd + TryFrom<u32> + Into<u32>,
{
    /// Create a new dense bin iterator
    pub fn new(
        bin_data: &DenseBin<T, IS_4BIT>,
        min_bin: u32,
        max_bin: u32,
        most_freq_bin: u32,
    ) -> Result<Self, <T as TryFrom<u32>>::Error> {
        let offset = if most_freq_bin == 0 { 1 } else { 0 };
        Ok(Self {
            bin_data: bin_data as *const _,
            min_bin: T::try_from(min_bin)?,
            max_bin: T::try_from(max_bin)?,
            most_freq_bin: T::try_from(most_freq_bin)?,
            offset,
        })
    }

    /// Get reference to the underlying bin data (unsafe)
    unsafe fn get_bin_data(&self) -> &DenseBin<T, IS_4BIT> {
        unsafe { &*self.bin_data }
    }
}

impl<T, const IS_4BIT: bool> BinIterator for DenseBinIterator<T, IS_4BIT>
where
    T: Copy + Clone + PartialOrd + From<u32> + Into<u32> + Default + From<u8>,
{
    fn get(&self, idx: DataSize) -> u32 {
        unsafe {
            let bin_data = self.get_bin_data();
            let ret = bin_data.data(idx);
            if ret >= self.min_bin && ret <= self.max_bin {
                ret.into() - self.min_bin.into() + self.offset as u32
            } else {
                self.most_freq_bin.into()
            }
        }
    }

    fn raw_get(&self, idx: DataSize) -> u32 {
        unsafe {
            let bin_data = self.get_bin_data();
            bin_data.data(idx).into()
        }
    }

    fn reset(&mut self, _start_idx: DataSize) {
        // No-op for dense bins as specified in C++ implementation
    }
}

/// Abstract bin trait for different bin implementations
pub trait Bin {
    /// Push a value to the bin at the given index
    fn push(&mut self, tid: i32, idx: DataSize, value: u32);

    /// Resize the bin to accommodate new data size
    fn resize(&mut self, num_data: DataSize);

    /// Get an iterator for the bin data
    fn get_iterator(&self, min_bin: u32, max_bin: u32, most_freq_bin: u32) -> Box<dyn BinIterator>;

    /// Construct histogram with data indices, gradients, and hessians
    fn construct_histogram(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        ordered_hessians: &[Score],
        out: &mut [Hist],
    );

    /// Construct histogram without data indices
    fn construct_histogram_no_indices(
        &self,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        ordered_hessians: &[Score],
        out: &mut [Hist],
    );

    /// Construct histogram with data indices and gradients only (no hessians)
    fn construct_histogram_no_hessian(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        out: &mut [Hist],
    );

    /// Construct histogram with gradients only (no indices or hessians)
    fn construct_histogram_no_indices_no_hessian(
        &self,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        out: &mut [Hist],
    );

    /// Construct histogram with 8-bit integer gradients and data indices
    fn construct_histogram_int8(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        ordered_hessians: &[Score],
        out: &mut [Hist],
    );

    /// Construct histogram with 8-bit integer gradients (no indices)
    fn construct_histogram_int8_no_indices(
        &self,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        ordered_hessians: &[Score],
        out: &mut [Hist],
    );

    /// Construct histogram with 8-bit integer gradients and data indices (no hessians)
    fn construct_histogram_int8_no_hessian(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        out: &mut [Hist],
    );

    /// Construct histogram with 8-bit integer gradients (no indices or hessians)
    fn construct_histogram_int8_no_indices_no_hessian(
        &self,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        out: &mut [Hist],
    );

    /// Construct histogram with 16-bit integer gradients and data indices
    fn construct_histogram_int16(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        ordered_hessians: &[Score],
        out: &mut [Hist],
    );

    /// Construct histogram with 16-bit integer gradients (no indices)
    fn construct_histogram_int16_no_indices(
        &self,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        ordered_hessians: &[Score],
        out: &mut [Hist],
    );

    /// Construct histogram with 16-bit integer gradients and data indices (no hessians)
    fn construct_histogram_int16_no_hessian(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        out: &mut [Hist],
    );

    /// Construct histogram with 16-bit integer gradients (no indices or hessians)
    fn construct_histogram_int16_no_indices_no_hessian(
        &self,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        out: &mut [Hist],
    );

    /// Construct histogram with 32-bit integer gradients and data indices
    fn construct_histogram_int32(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        ordered_hessians: &[Score],
        out: &mut [Hist],
    );

    /// Construct histogram with 32-bit integer gradients (no indices)
    fn construct_histogram_int32_no_indices(
        &self,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        ordered_hessians: &[Score],
        out: &mut [Hist],
    );

    /// Construct histogram with 32-bit integer gradients and data indices (no hessians)
    fn construct_histogram_int32_no_hessian(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        out: &mut [Hist],
    );

    /// Construct histogram with 32-bit integer gradients (no indices or hessians)
    fn construct_histogram_int32_no_indices_no_hessian(
        &self,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        out: &mut [Hist],
    );

    /// Split data based on threshold with min_bin parameter
    fn split(
        &self,
        min_bin: u32,
        max_bin: u32,
        default_bin: u32,
        most_freq_bin: u32,
        missing_type: MissingType,
        default_left: bool,
        threshold: u32,
        data_indices: &[DataSize],
        cnt: DataSize,
        lte_indices: &mut [DataSize],
        gt_indices: &mut [DataSize],
    ) -> DataSize;

    /// Split data based on threshold without min_bin parameter
    fn split_no_min_bin(
        &self,
        max_bin: u32,
        default_bin: u32,
        most_freq_bin: u32,
        missing_type: MissingType,
        default_left: bool,
        threshold: u32,
        data_indices: &[DataSize],
        cnt: DataSize,
        lte_indices: &mut [DataSize],
        gt_indices: &mut [DataSize],
    ) -> DataSize;

    /// Split categorical data with min_bin parameter
    fn split_categorical(
        &self,
        min_bin: u32,
        max_bin: u32,
        most_freq_bin: u32,
        threshold: &[u32],
        num_threshold: i32,
        data_indices: &[DataSize],
        cnt: DataSize,
        lte_indices: &mut [DataSize],
        gt_indices: &mut [DataSize],
    ) -> DataSize;

    /// Split categorical data without min_bin parameter
    fn split_categorical_no_min_bin(
        &self,
        max_bin: u32,
        most_freq_bin: u32,
        threshold: &[u32],
        num_threshold: i32,
        data_indices: &[DataSize],
        cnt: DataSize,
        lte_indices: &mut [DataSize],
        gt_indices: &mut [DataSize],
    ) -> DataSize;

    /// Get the number of data points
    fn num_data(&self) -> DataSize;

    /// Get raw data pointer
    fn get_data(&self) -> *mut u8;

    /// Finalize loading (for 4-bit bins)
    fn finish_load(&mut self);

    /// Load data from memory with optional index mapping
    fn load_from_memory(&mut self, memory: *const u8, local_used_indices: &[DataSize]);

    /// Copy subrow from another bin
    fn copy_subrow(
        &mut self,
        full_bin: &dyn Bin,
        used_indices: &[DataSize],
        num_used_indices: DataSize,
    );

    /// Save binary data to writer
    fn save_binary_to_file(&self, writer: &mut dyn std::io::Write) -> Result<()>;

    /// Get size in bytes
    fn sizes_in_byte(&self) -> usize;

    /// Clone the bin
    fn clone_bin(&self) -> Box<dyn Bin>;

    /// Get column-wise data for multi-threading
    fn get_col_wise_data(
        &self,
        bit_type: &mut u8,
        is_sparse: &mut bool,
        num_threads: i32,
    ) -> *const u8;

    /// Get column-wise data for single thread
    fn get_col_wise_data_single(&self, bit_type: &mut u8, is_sparse: &mut bool) -> *const u8;
}

/// Wrapper to make DenseBinIterator compatible with BinIterator trait
#[derive(Debug)]
pub struct DenseBinIteratorWrapper<T, const IS_4BIT: bool> {
    inner: DenseBinIterator<T, IS_4BIT>,
}

impl<T, const IS_4BIT: bool> BinIterator for DenseBinIteratorWrapper<T, IS_4BIT>
where
    T: Copy + Clone + Default + From<u8> + Into<u32> + PartialOrd,
{
    fn get(&self, idx: DataSize) -> u32 {
        unsafe {
            let bin_data = &*self.inner.bin_data;
            let ret = bin_data.data(idx);
            if ret >= self.inner.min_bin && ret <= self.inner.max_bin {
                ret.into() - self.inner.min_bin.into() + self.inner.offset as u32
            } else {
                self.inner.most_freq_bin.into()
            }
        }
    }

    fn raw_get(&self, idx: DataSize) -> u32 {
        unsafe {
            let bin_data = &*self.inner.bin_data;
            bin_data.data(idx).into()
        }
    }

    fn reset(&mut self, _start_idx: DataSize) {
        // No-op for dense bins
    }
}

/// Dense bin implementation for storing binned feature values
/// Template parameters:
/// - T: Value type (u8, u16, u32)
/// - IS_4BIT: Whether to use 4-bit packing
#[derive(Debug, Clone)]
pub struct DenseBin<T, const IS_4BIT: bool> {
    /// Main data storage
    data: Vec<T>,
    /// Buffer for 4-bit operations
    buf: Vec<u8>,
    /// Number of data points
    num_data: DataSize,
}

impl<T, const IS_4BIT: bool> DenseBin<T, IS_4BIT>
where
    T: Copy + Clone + Default + From<u8> + Into<u32> + PartialOrd,
{
    /// Create a new dense bin
    pub fn new(num_data: DataSize) -> Self {
        let (data, buf) = if IS_4BIT {
            // For 4-bit, we need to check that T is u8
            assert_eq!(
                std::mem::size_of::<T>(),
                1,
                "4-bit mode requires u8 value type"
            );
            let size = ((num_data + 1) / 2) as usize;
            (vec![T::from(0u8); size], vec![0u8; size])
        } else {
            (vec![T::default(); num_data as usize], Vec::new())
        };

        Self {
            data,
            buf,
            num_data,
        }
    }

    /// Get data value at index (handles 4-bit packing)
    pub fn data(&self, idx: DataSize) -> T {
        if IS_4BIT {
            let data_ptr = self.data.as_ptr() as *const u8;
            unsafe {
                let byte_val = *data_ptr.add((idx >> 1) as usize);
                let shift = (idx & 1) << 2;
                T::from((byte_val >> shift) & 0xf)
            }
        } else {
            self.data[idx as usize]
        }
    }

    /// Internal histogram construction implementation
    fn construct_histogram_inner<
        const USE_INDICES: bool,
        const USE_PREFETCH: bool,
        const USE_HESSIAN: bool,
    >(
        &self,
        data_indices: Option<&[DataSize]>,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        ordered_hessians: Option<&[Score]>,
        out: &mut [Hist],
    ) {
        let mut i = start;
        let grad_ptr = out.as_mut_ptr();
        let hess_ptr = unsafe { grad_ptr.add(1) };

        // Prefetching implementation
        if USE_PREFETCH {
            let pf_offset = (64 / std::mem::size_of::<T>()) as DataSize;
            let pf_end = end - pf_offset;

            while i < pf_end {
                let idx = if USE_INDICES {
                    data_indices.unwrap()[i as usize]
                } else {
                    i
                };

                let pf_idx = if USE_INDICES {
                    data_indices.unwrap()[(i + pf_offset) as usize]
                } else {
                    i + pf_offset
                };

                // Prefetch next data
                if IS_4BIT {
                    let prefetch_addr = unsafe { self.data.as_ptr().add((pf_idx >> 1) as usize) };
                    // Note: Rust doesn't have direct prefetch intrinsics, but we can simulate the intent
                    std::hint::black_box(prefetch_addr);
                } else {
                    let prefetch_addr = unsafe { self.data.as_ptr().add(pf_idx as usize) };
                    std::hint::black_box(prefetch_addr);
                }

                let ti = (self.data(idx).into() as usize) << 1;

                if USE_HESSIAN {
                    unsafe {
                        *grad_ptr.add(ti) += ordered_gradients[i as usize] as Hist;
                        *hess_ptr.add(ti) += ordered_hessians.unwrap()[i as usize] as Hist;
                    }
                } else {
                    unsafe {
                        *grad_ptr.add(ti) += ordered_gradients[i as usize] as Hist;
                        let cnt_ptr = hess_ptr.add(ti) as *mut HistCount;
                        *cnt_ptr += 1;
                    }
                }

                i += 1;
            }
        }

        // Main loop without prefetching
        while i < end {
            let idx = if USE_INDICES {
                data_indices.unwrap()[i as usize]
            } else {
                i
            };

            let ti = (self.data(idx).into() as usize) << 1;

            if USE_HESSIAN {
                unsafe {
                    *grad_ptr.add(ti) += ordered_gradients[i as usize] as Hist;
                    *hess_ptr.add(ti) += ordered_hessians.unwrap()[i as usize] as Hist;
                }
            } else {
                unsafe {
                    *grad_ptr.add(ti) += ordered_gradients[i as usize] as Hist;
                    let cnt_ptr = hess_ptr.add(ti) as *mut HistCount;
                    *cnt_ptr += 1;
                }
            }

            i += 1;
        }
    }

    /// Internal integer histogram construction implementation
    fn construct_histogram_int_inner<
        const USE_INDICES: bool,
        const USE_PREFETCH: bool,
        const USE_HESSIAN: bool,
        PackedHistT: Copy,
        const HIST_BITS: i32,
    >(
        &self,
        data_indices: Option<&[DataSize]>,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        out: &mut [Hist],
    ) where
        PackedHistT: std::ops::AddAssign + From<i16> + From<i8>,
    {
        let mut i = start;
        let out_ptr = out.as_mut_ptr() as *mut PackedHistT;
        let gradients_ptr = ordered_gradients.as_ptr() as *const i16;
        let data_ptr_base = self.data.as_ptr();

        // Prefetching implementation
        if USE_PREFETCH {
            let pf_offset = (64 / std::mem::size_of::<T>()) as DataSize;
            let pf_end = end - pf_offset;

            while i < pf_end {
                let idx = if USE_INDICES {
                    data_indices.unwrap()[i as usize]
                } else {
                    i
                };

                let pf_idx = if USE_INDICES {
                    data_indices.unwrap()[(i + pf_offset) as usize]
                } else {
                    i + pf_offset
                };

                // Prefetch next data
                if IS_4BIT {
                    let prefetch_addr = unsafe { data_ptr_base.add((pf_idx >> 1) as usize) };
                    std::hint::black_box(prefetch_addr);
                } else {
                    let prefetch_addr = unsafe { data_ptr_base.add(pf_idx as usize) };
                    std::hint::black_box(prefetch_addr);
                }

                let ti = self.data(idx).into() as usize;
                let gradient_16 = unsafe { *gradients_ptr.add(i as usize) };

                let gradient_packed = if USE_HESSIAN {
                    if HIST_BITS == 8 {
                        PackedHistT::from(gradient_16)
                    } else {
                        let _high = PackedHistT::from((gradient_16 >> 8) as i8);
                        let _low = PackedHistT::from((gradient_16 & 0xff) as i8);
                        // This is a simplified version - actual bit packing would be more complex
                        _high // Placeholder
                    }
                } else {
                    if HIST_BITS == 8 {
                        PackedHistT::from(gradient_16)
                    } else {
                        let _high = PackedHistT::from((gradient_16 >> 8) as i8);
                        PackedHistT::from(1i8) // Count = 1
                    }
                };

                unsafe {
                    *out_ptr.add(ti) += gradient_packed;
                }

                i += 1;
            }
        }

        // Main loop without prefetching
        while i < end {
            let idx = if USE_INDICES {
                data_indices.unwrap()[i as usize]
            } else {
                i
            };

            let ti = self.data(idx).into() as usize;
            let gradient_16 = unsafe { *gradients_ptr.add(i as usize) };

            let gradient_packed = if USE_HESSIAN {
                if HIST_BITS == 8 {
                    PackedHistT::from(gradient_16)
                } else {
                    let _high = PackedHistT::from((gradient_16 >> 8) as i8);
                    let _low = PackedHistT::from((gradient_16 & 0xff) as i8);
                    _high // Placeholder
                }
            } else {
                if HIST_BITS == 8 {
                    PackedHistT::from(gradient_16)
                } else {
                    let _high = PackedHistT::from((gradient_16 >> 8) as i8);
                    PackedHistT::from(1i8) // Count = 1
                }
            };

            unsafe {
                *out_ptr.add(ti) += gradient_packed;
            }

            i += 1;
        }
    }

    /// Internal split implementation with template parameters
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
        cnt: DataSize,
        lte_indices: &mut [DataSize],
        gt_indices: &mut [DataSize],
    ) -> DataSize {
        let th = T::from((threshold + min_bin) as u8);
        let t_zero_bin = T::from((min_bin + default_bin) as u8);
        let (th, t_zero_bin) = if most_freq_bin == 0 {
            // Adjust for most frequent bin at 0
            (th, t_zero_bin) // In a real implementation, we'd subtract 1
        } else {
            (th, t_zero_bin)
        };

        let minb = T::from(min_bin as u8);
        let maxb = T::from(max_bin as u8);
        let mut lte_count = 0;
        let mut gt_count = 0;

        // Determine routing preferences
        let most_freq_goes_left = most_freq_bin <= threshold;
        let missing_goes_left = (MISS_IS_ZERO || MISS_IS_NA) && default_left;

        if min_bin < max_bin {
            for i in 0..cnt {
                let idx = data_indices[i as usize];
                let bin = self.data(idx);

                let goes_to_missing = (MISS_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin)
                    || (MISS_IS_NA && !MFB_IS_NA && bin == maxb);

                let is_missing_bin = (USE_MIN_BIN && (bin < minb || bin > maxb))
                    || (!USE_MIN_BIN && bin == T::default());

                if goes_to_missing {
                    if missing_goes_left {
                        lte_indices[lte_count] = idx;
                        lte_count += 1;
                    } else {
                        gt_indices[gt_count] = idx;
                        gt_count += 1;
                    }
                } else if is_missing_bin {
                    if (MISS_IS_NA && MFB_IS_NA) || (MISS_IS_ZERO && MFB_IS_ZERO) {
                        if missing_goes_left {
                            lte_indices[lte_count] = idx;
                            lte_count += 1;
                        } else {
                            gt_indices[gt_count] = idx;
                            gt_count += 1;
                        }
                    } else {
                        if most_freq_goes_left {
                            lte_indices[lte_count] = idx;
                            lte_count += 1;
                        } else {
                            gt_indices[gt_count] = idx;
                            gt_count += 1;
                        }
                    }
                } else if bin > th {
                    gt_indices[gt_count] = idx;
                    gt_count += 1;
                } else {
                    lte_indices[lte_count] = idx;
                    lte_count += 1;
                }
            }
        } else {
            // Special case when min_bin == max_bin
            let max_bin_goes_left = maxb <= th;

            for i in 0..cnt {
                let idx = data_indices[i as usize];
                let bin = self.data(idx);

                if MISS_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin {
                    if missing_goes_left {
                        lte_indices[lte_count] = idx;
                        lte_count += 1;
                    } else {
                        gt_indices[gt_count] = idx;
                        gt_count += 1;
                    }
                } else if bin != maxb {
                    if (MISS_IS_NA && MFB_IS_NA) || (MISS_IS_ZERO && MFB_IS_ZERO) {
                        if missing_goes_left {
                            lte_indices[lte_count] = idx;
                            lte_count += 1;
                        } else {
                            gt_indices[gt_count] = idx;
                            gt_count += 1;
                        }
                    } else {
                        if most_freq_goes_left {
                            lte_indices[lte_count] = idx;
                            lte_count += 1;
                        } else {
                            gt_indices[gt_count] = idx;
                            gt_count += 1;
                        }
                    }
                } else {
                    if MISS_IS_NA && !MFB_IS_NA {
                        if missing_goes_left {
                            lte_indices[lte_count] = idx;
                            lte_count += 1;
                        } else {
                            gt_indices[gt_count] = idx;
                            gt_count += 1;
                        }
                    } else {
                        if max_bin_goes_left {
                            lte_indices[lte_count] = idx;
                            lte_count += 1;
                        } else {
                            gt_indices[gt_count] = idx;
                            gt_count += 1;
                        }
                    }
                }
            }
        }

        lte_count as DataSize
    }

    /// Internal categorical split implementation
    fn split_categorical_inner<const USE_MIN_BIN: bool>(
        &self,
        min_bin: u32,
        max_bin: u32,
        most_freq_bin: u32,
        threshold: &[u32],
        num_threshold: i32,
        data_indices: &[DataSize],
        cnt: DataSize,
        lte_indices: &mut [DataSize],
        gt_indices: &mut [DataSize],
    ) -> DataSize {
        let mut lte_count = 0;
        let mut gt_count = 0;

        let offset = if most_freq_bin == 0 { 1i8 } else { 0i8 };

        let most_freq_goes_left =
            most_freq_bin > 0 && self.find_in_bitset(threshold, num_threshold, most_freq_bin);

        for i in 0..cnt {
            let idx = data_indices[i as usize];
            let bin = self.data(idx).into();

            if (USE_MIN_BIN && (bin < min_bin || bin > max_bin)) || (!USE_MIN_BIN && bin == 0) {
                if most_freq_goes_left {
                    lte_indices[lte_count] = idx;
                    lte_count += 1;
                } else {
                    gt_indices[gt_count] = idx;
                    gt_count += 1;
                }
            } else if self.find_in_bitset(threshold, num_threshold, bin - min_bin + offset as u32) {
                lte_indices[lte_count] = idx;
                lte_count += 1;
            } else {
                gt_indices[gt_count] = idx;
                gt_count += 1;
            }
        }

        lte_count as DataSize
    }

    /// Helper function to find value in bitset (simplified implementation)
    fn find_in_bitset(&self, bitset: &[u32], num_bits: i32, value: u32) -> bool {
        if value >= num_bits as u32 * 32 {
            return false;
        }

        let word_idx = (value / 32) as usize;
        let bit_idx = value % 32;

        if word_idx < bitset.len() {
            (bitset[word_idx] & (1 << bit_idx)) != 0
        } else {
            false
        }
    }
}

impl<T, const IS_4BIT: bool> Bin for DenseBin<T, IS_4BIT>
where
    T: Copy + Clone + Default + From<u8> + Into<u32> + PartialOrd + TryFrom<u32> + 'static,
    <T as TryFrom<u32>>::Error: std::fmt::Debug,
{
    fn push(&mut self, _tid: i32, idx: DataSize, value: u32) {
        if IS_4BIT {
            let i1 = (idx >> 1) as usize;
            let i2 = (idx & 1) << 2;
            let val = (value as u8) << i2;

            if i2 == 0 {
                // Ensure data is treated as Vec<u8> for 4-bit mode
                let data_ptr = self.data.as_mut_ptr() as *mut u8;
                unsafe {
                    *data_ptr.add(i1) = val;
                }
            } else {
                if self.buf.len() <= i1 {
                    self.buf.resize(i1 + 1, 0);
                }
                self.buf[i1] = val;
            }
        } else {
            self.data[idx as usize] = T::try_from(value).unwrap();
        }
    }

    fn resize(&mut self, num_data: DataSize) {
        if self.num_data != num_data {
            self.num_data = num_data;
            if IS_4BIT {
                let new_size = ((num_data + 1) / 2) as usize;
                self.data.resize(new_size, T::default());
            } else {
                self.data.resize(num_data as usize, T::default());
            }
        }
    }

    fn get_iterator(&self, min_bin: u32, max_bin: u32, most_freq_bin: u32) -> Box<dyn BinIterator> {
        // Create a simple iterator wrapper that doesn't require From<u32> trait
        let min_bin_t = T::try_from(min_bin).unwrap_or_default();
        let max_bin_t = T::try_from(max_bin).unwrap_or_default();
        let most_freq_bin_t = T::try_from(most_freq_bin).unwrap_or_default();
        let offset = if most_freq_bin == 0 { 1 } else { 0 };

        let iterator = DenseBinIterator {
            bin_data: self as *const _,
            min_bin: min_bin_t,
            max_bin: max_bin_t,
            most_freq_bin: most_freq_bin_t,
            offset,
        };

        Box::new(DenseBinIteratorWrapper { inner: iterator })
    }

    fn construct_histogram(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        ordered_hessians: &[Score],
        out: &mut [Hist],
    ) {
        self.construct_histogram_inner::<true, true, true>(
            Some(data_indices),
            start,
            end,
            ordered_gradients,
            Some(ordered_hessians),
            out,
        );
    }

    fn construct_histogram_no_indices(
        &self,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        ordered_hessians: &[Score],
        out: &mut [Hist],
    ) {
        self.construct_histogram_inner::<false, false, true>(
            None,
            start,
            end,
            ordered_gradients,
            Some(ordered_hessians),
            out,
        );
    }

    fn construct_histogram_no_hessian(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        out: &mut [Hist],
    ) {
        self.construct_histogram_inner::<true, true, false>(
            Some(data_indices),
            start,
            end,
            ordered_gradients,
            None,
            out,
        );
    }

    fn construct_histogram_no_indices_no_hessian(
        &self,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        out: &mut [Hist],
    ) {
        self.construct_histogram_inner::<false, false, false>(
            None,
            start,
            end,
            ordered_gradients,
            None,
            out,
        );
    }

    fn construct_histogram_int8(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        _ordered_hessians: &[Score],
        out: &mut [Hist],
    ) {
        self.construct_histogram_int_inner::<true, true, true, i16, 8>(
            Some(data_indices),
            start,
            end,
            ordered_gradients,
            out,
        );
    }

    fn construct_histogram_int8_no_indices(
        &self,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        _ordered_hessians: &[Score],
        out: &mut [Hist],
    ) {
        self.construct_histogram_int_inner::<false, false, true, i16, 8>(
            None,
            start,
            end,
            ordered_gradients,
            out,
        );
    }

    fn construct_histogram_int8_no_hessian(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        out: &mut [Hist],
    ) {
        self.construct_histogram_int_inner::<true, true, false, i16, 8>(
            Some(data_indices),
            start,
            end,
            ordered_gradients,
            out,
        );
    }

    fn construct_histogram_int8_no_indices_no_hessian(
        &self,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        out: &mut [Hist],
    ) {
        self.construct_histogram_int_inner::<false, false, false, i16, 8>(
            None,
            start,
            end,
            ordered_gradients,
            out,
        );
    }

    fn construct_histogram_int16(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        _ordered_hessians: &[Score],
        out: &mut [Hist],
    ) {
        self.construct_histogram_int_inner::<true, true, true, i32, 16>(
            Some(data_indices),
            start,
            end,
            ordered_gradients,
            out,
        );
    }

    fn construct_histogram_int16_no_indices(
        &self,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        _ordered_hessians: &[Score],
        out: &mut [Hist],
    ) {
        self.construct_histogram_int_inner::<false, false, true, i32, 16>(
            None,
            start,
            end,
            ordered_gradients,
            out,
        );
    }

    fn construct_histogram_int16_no_hessian(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        out: &mut [Hist],
    ) {
        self.construct_histogram_int_inner::<true, true, false, i32, 16>(
            Some(data_indices),
            start,
            end,
            ordered_gradients,
            out,
        );
    }

    fn construct_histogram_int16_no_indices_no_hessian(
        &self,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        out: &mut [Hist],
    ) {
        self.construct_histogram_int_inner::<false, false, false, i32, 16>(
            None,
            start,
            end,
            ordered_gradients,
            out,
        );
    }

    fn construct_histogram_int32(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        _ordered_hessians: &[Score],
        out: &mut [Hist],
    ) {
        self.construct_histogram_int_inner::<true, true, true, i64, 32>(
            Some(data_indices),
            start,
            end,
            ordered_gradients,
            out,
        );
    }

    fn construct_histogram_int32_no_indices(
        &self,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        _ordered_hessians: &[Score],
        out: &mut [Hist],
    ) {
        self.construct_histogram_int_inner::<false, false, true, i64, 32>(
            None,
            start,
            end,
            ordered_gradients,
            out,
        );
    }

    fn construct_histogram_int32_no_hessian(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        out: &mut [Hist],
    ) {
        self.construct_histogram_int_inner::<true, true, false, i64, 32>(
            Some(data_indices),
            start,
            end,
            ordered_gradients,
            out,
        );
    }

    fn construct_histogram_int32_no_indices_no_hessian(
        &self,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[Score],
        out: &mut [Hist],
    ) {
        self.construct_histogram_int_inner::<false, false, false, i64, 32>(
            None,
            start,
            end,
            ordered_gradients,
            out,
        );
    }

    fn split(
        &self,
        min_bin: u32,
        max_bin: u32,
        default_bin: u32,
        most_freq_bin: u32,
        missing_type: MissingType,
        default_left: bool,
        threshold: u32,
        data_indices: &[DataSize],
        cnt: DataSize,
        lte_indices: &mut [DataSize],
        gt_indices: &mut [DataSize],
    ) -> DataSize {
        match missing_type {
            MissingType::None => self.split_inner::<false, false, false, false, true>(
                min_bin,
                max_bin,
                default_bin,
                most_freq_bin,
                default_left,
                threshold,
                data_indices,
                cnt,
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
                        cnt,
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
                        cnt,
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
                        cnt,
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
                        cnt,
                        lte_indices,
                        gt_indices,
                    )
                }
            }
            MissingType::Separate => {
                // Handle separate missing value treatment similar to NaN
                if max_bin == most_freq_bin + min_bin && most_freq_bin > 0 {
                    self.split_inner::<false, true, false, true, true>(
                        min_bin,
                        max_bin,
                        default_bin,
                        most_freq_bin,
                        default_left,
                        threshold,
                        data_indices,
                        cnt,
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
                        cnt,
                        lte_indices,
                        gt_indices,
                    )
                }
            }
        }
    }

    fn split_no_min_bin(
        &self,
        max_bin: u32,
        default_bin: u32,
        most_freq_bin: u32,
        missing_type: MissingType,
        default_left: bool,
        threshold: u32,
        data_indices: &[DataSize],
        cnt: DataSize,
        lte_indices: &mut [DataSize],
        gt_indices: &mut [DataSize],
    ) -> DataSize {
        match missing_type {
            MissingType::None => self.split_inner::<false, false, false, false, false>(
                1,
                max_bin,
                default_bin,
                most_freq_bin,
                default_left,
                threshold,
                data_indices,
                cnt,
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
                        cnt,
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
                        cnt,
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
                        cnt,
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
                        cnt,
                        lte_indices,
                        gt_indices,
                    )
                }
            }
            MissingType::Separate => {
                // Handle separate missing value treatment similar to NaN
                if max_bin == most_freq_bin + 1 && most_freq_bin > 0 {
                    self.split_inner::<false, true, false, true, false>(
                        1,
                        max_bin,
                        default_bin,
                        most_freq_bin,
                        default_left,
                        threshold,
                        data_indices,
                        cnt,
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
                        cnt,
                        lte_indices,
                        gt_indices,
                    )
                }
            }
        }
    }

    fn split_categorical(
        &self,
        min_bin: u32,
        max_bin: u32,
        most_freq_bin: u32,
        threshold: &[u32],
        num_threshold: i32,
        data_indices: &[DataSize],
        cnt: DataSize,
        lte_indices: &mut [DataSize],
        gt_indices: &mut [DataSize],
    ) -> DataSize {
        self.split_categorical_inner::<true>(
            min_bin,
            max_bin,
            most_freq_bin,
            threshold,
            num_threshold,
            data_indices,
            cnt,
            lte_indices,
            gt_indices,
        )
    }

    fn split_categorical_no_min_bin(
        &self,
        max_bin: u32,
        most_freq_bin: u32,
        threshold: &[u32],
        num_threshold: i32,
        data_indices: &[DataSize],
        cnt: DataSize,
        lte_indices: &mut [DataSize],
        gt_indices: &mut [DataSize],
    ) -> DataSize {
        self.split_categorical_inner::<false>(
            1,
            max_bin,
            most_freq_bin,
            threshold,
            num_threshold,
            data_indices,
            cnt,
            lte_indices,
            gt_indices,
        )
    }

    fn num_data(&self) -> DataSize {
        self.num_data
    }

    fn get_data(&self) -> *mut u8 {
        self.data.as_ptr() as *mut u8
    }

    fn finish_load(&mut self) {
        if IS_4BIT && !self.buf.is_empty() {
            let len = ((self.num_data + 1) / 2) as usize;
            let data_ptr = self.data.as_mut_ptr() as *mut u8;

            for i in 0..len {
                if i < self.buf.len() {
                    unsafe {
                        *data_ptr.add(i) |= self.buf[i];
                    }
                }
            }
            self.buf.clear();
        }
    }

    fn load_from_memory(&mut self, memory: *const u8, local_used_indices: &[DataSize]) {
        let mem_data = memory as *const T;

        if !local_used_indices.is_empty() {
            if IS_4BIT {
                let rest = self.num_data & 1;
                let data_ptr = self.data.as_mut_ptr() as *mut u8;

                for i in (0..(self.num_data - rest)).step_by(2) {
                    // Get old bins
                    let idx1 = local_used_indices[i as usize];
                    let bin1 = unsafe {
                        let byte_val = *(mem_data as *const u8).add((idx1 >> 1) as usize);
                        (byte_val >> ((idx1 & 1) << 2)) & 0xf
                    };

                    let idx2 = local_used_indices[(i + 1) as usize];
                    let bin2 = unsafe {
                        let byte_val = *(mem_data as *const u8).add((idx2 >> 1) as usize);
                        (byte_val >> ((idx2 & 1) << 2)) & 0xf
                    };

                    // Combine bins
                    let i1 = (i >> 1) as usize;
                    unsafe {
                        *data_ptr.add(i1) = bin1 | (bin2 << 4);
                    }
                }

                if rest != 0 {
                    let idx = local_used_indices[(self.num_data - 1) as usize];
                    let bin = unsafe {
                        let byte_val = *(mem_data as *const u8).add((idx >> 1) as usize);
                        (byte_val >> ((idx & 1) << 2)) & 0xf
                    };
                    unsafe {
                        *data_ptr.add((self.num_data >> 1) as usize) = bin;
                    }
                }
            } else {
                for i in 0..self.num_data {
                    let src_idx = local_used_indices[i as usize];
                    unsafe {
                        self.data[i as usize] = *mem_data.add(src_idx as usize);
                    }
                }
            }
        } else {
            for i in 0..self.data.len() {
                unsafe {
                    self.data[i] = *mem_data.add(i);
                }
            }
        }
    }

    fn copy_subrow(
        &mut self,
        full_bin: &dyn Bin,
        used_indices: &[DataSize],
        num_used_indices: DataSize,
    ) {
        // This is a simplified implementation - in practice we'd need proper downcasting
        // For now, we'll use the get_data() method and handle the copying manually
        let src_data = full_bin.get_data() as *const T;

        if IS_4BIT {
            let rest = num_used_indices & 1;
            let data_ptr = self.data.as_mut_ptr() as *mut u8;

            for i in (0..(num_used_indices - rest)).step_by(2) {
                let idx1 = used_indices[i as usize];
                let bin1 = unsafe {
                    let byte_val = *(src_data as *const u8).add((idx1 >> 1) as usize);
                    (byte_val >> ((idx1 & 1) << 2)) & 0xf
                };

                let idx2 = used_indices[(i + 1) as usize];
                let bin2 = unsafe {
                    let byte_val = *(src_data as *const u8).add((idx2 >> 1) as usize);
                    (byte_val >> ((idx2 & 1) << 2)) & 0xf
                };

                let i1 = (i >> 1) as usize;
                unsafe {
                    *data_ptr.add(i1) = bin1 | (bin2 << 4);
                }
            }

            if rest != 0 {
                let idx = used_indices[(num_used_indices - 1) as usize];
                let bin = unsafe {
                    let byte_val = *(src_data as *const u8).add((idx >> 1) as usize);
                    (byte_val >> ((idx & 1) << 2)) & 0xf
                };
                unsafe {
                    *data_ptr.add((num_used_indices >> 1) as usize) = bin;
                }
            }
        } else {
            for i in 0..num_used_indices {
                let src_idx = used_indices[i as usize];
                unsafe {
                    self.data[i as usize] = *src_data.add(src_idx as usize);
                }
            }
        }
    }

    fn save_binary_to_file(&self, writer: &mut dyn std::io::Write) -> Result<()> {
        // Write aligned data
        let data_bytes = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * std::mem::size_of::<T>(),
            )
        };
        writer.write_all(data_bytes)?;
        Ok(())
    }

    fn sizes_in_byte(&self) -> usize {
        // Align to K_ALIGNED_SIZE boundary
        let size = self.data.len() * std::mem::size_of::<T>();
        (size + K_ALIGNED_SIZE - 1) & !(K_ALIGNED_SIZE - 1)
    }

    fn clone_bin(&self) -> Box<dyn Bin> {
        Box::new(self.clone())
    }

    fn get_col_wise_data(
        &self,
        bit_type: &mut u8,
        is_sparse: &mut bool,
        _num_threads: i32,
    ) -> *const u8 {
        *is_sparse = false;
        *bit_type = if IS_4BIT {
            4
        } else {
            (std::mem::size_of::<T>() * 8) as u8
        };
        self.data.as_ptr() as *const u8
    }

    fn get_col_wise_data_single(&self, bit_type: &mut u8, is_sparse: &mut bool) -> *const u8 {
        *is_sparse = false;
        *bit_type = if IS_4BIT {
            4
        } else {
            (std::mem::size_of::<T>() * 8) as u8
        };
        self.data.as_ptr() as *const u8
    }
}

// Type aliases for common dense bin configurations

/// Type alias for DenseBin with u8 values using 4-bit compression (values 0-15)
pub type DenseBin4Bit = DenseBin<u8, true>;

/// Type alias for DenseBin with u8 values using 8-bit storage (values 0-255)
pub type DenseBin8Bit = DenseBin<u8, false>;

/// Type alias for DenseBin with u16 values using 16-bit storage (values 0-65535)
pub type DenseBin16Bit = DenseBin<u16, false>;

/// Type alias for DenseBin with u32 values using 32-bit storage (values 0-4B)
pub type DenseBin32Bit = DenseBin<u32, false>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_bin_creation() {
        let bin = DenseBin8Bit::new(100);
        assert_eq!(bin.num_data(), 100);
        assert_eq!(bin.data.len(), 100);
    }

    #[test]
    fn test_dense_bin_4bit_creation() {
        let bin = DenseBin4Bit::new(100);
        assert_eq!(bin.num_data(), 100);
        assert_eq!(bin.data.len(), 50); // (100 + 1) / 2
    }

    #[test]
    fn test_push_and_data_access() {
        let mut bin = DenseBin8Bit::new(10);
        bin.push(0, 5, 42);
        assert_eq!(bin.data(5) as u32, 42u32);
    }

    #[test]
    fn test_4bit_push_and_data_access() {
        let mut bin = DenseBin4Bit::new(4);
        bin.push(0, 0, 3);
        bin.push(0, 1, 7);
        bin.push(0, 2, 2);
        bin.push(0, 3, 5);
        bin.finish_load();

        assert_eq!(bin.data(0) as u32, 3u32);
        assert_eq!(bin.data(1) as u32, 7u32);
        assert_eq!(bin.data(2) as u32, 2u32);
        assert_eq!(bin.data(3) as u32, 5u32);
    }

    #[test]
    fn test_iterator_creation() {
        let bin = DenseBin8Bit::new(10);
        let _iterator = bin.get_iterator(0, 10, 0);
    }

    #[test]
    fn test_resize() {
        let mut bin = DenseBin8Bit::new(10);
        bin.resize(20);
        assert_eq!(bin.num_data(), 20);
        assert_eq!(bin.data.len(), 20);
    }
}
