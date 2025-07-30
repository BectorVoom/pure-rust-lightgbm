//! Copyright (c) 2016 Microsoft Corporation. All rights reserved.
//! Licensed under the MIT License. See LICENSE file in the project root for license information.
//!
//! # Bin Module
//!
//! This module provides the core binning functionality for LightGBM, including:
//! - Feature value to bin mapping (BinMapper)
//! - Bin data storage and access (Bin trait)
//! - Multi-value bin support (MultiValBin trait)
//! - Histogram construction for efficient gradient boosting
//!
//! The binning system discretizes continuous features into bins to enable
//! efficient histogram-based tree learning algorithms.

use crate::core::error::Result;
use crate::core::meta::{CommSizeT, ScoreT};
use crate::core::types::{DataSize, MissingType};
use crate::core::utils::binary_writer::BinaryWriter;
use crate::core::utils::common::Common;
use std::collections::HashMap;
use std::fmt;

// =====================================
// Type Definitions and Constants
// =====================================

/// Type for histogram values - 64-bit float for numerical stability
pub type HistT = f64;

/// Type for integer histogram values - 32-bit signed integer
pub type IntHistT = i32;

/// Type for histogram count values - 64-bit unsigned integer for large counts
pub type HistCntT = u64;

// Compile-time assertion equivalent
// In C++: static_assert(sizeof(hist_t) == sizeof(hist_cnt_t), "Histogram entry size is not correct");
const _: () = {
    assert!(std::mem::size_of::<HistT>() == std::mem::size_of::<HistCntT>());
};

/// Size of a histogram entry in bytes (gradient + hessian)
pub const HIST_ENTRY_SIZE: usize = 2 * std::mem::size_of::<HistT>();

/// Size of a 32-bit integer histogram entry in bytes
pub const INT32_HIST_ENTRY_SIZE: usize = 2 * std::mem::size_of::<IntHistT>();

/// Size of a 16-bit integer histogram entry in bytes
pub const INT16_HIST_ENTRY_SIZE: usize = 2 * std::mem::size_of::<i16>();

/// Histogram offset for accessing gradient/hessian pairs
pub const HIST_OFFSET: usize = 2;

/// Sparsity threshold for determining sparse vs dense bin representation
pub const SPARSE_THRESHOLD: f64 = 0.7;

// =====================================
// Enums
// =====================================

/// Type of bin for features
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinType {
    /// Numerical/continuous feature bin
    Numerical,
    /// Categorical feature bin
    Categorical,
}

impl Default for BinType {
    fn default() -> Self {
        BinType::Numerical
    }
}

impl fmt::Display for BinType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinType::Numerical => write!(f, "numerical"),
            BinType::Categorical => write!(f, "categorical"),
        }
    }
}

// =====================================
// Histogram Access Macros
// =====================================

/// Get gradient value from histogram at bin index i
/// Equivalent to C++ macro: #define GET_GRAD(hist, i) hist[(i) << 1]
#[inline]
pub fn get_grad(hist: &[HistT], i: usize) -> HistT {
    hist[i << 1]
}

/// Get hessian value from histogram at bin index i
/// Equivalent to C++ macro: #define GET_HESS(hist, i) hist[((i) << 1) + 1]
#[inline]
pub fn get_hess(hist: &[HistT], i: usize) -> HistT {
    hist[(i << 1) + 1]
}

/// Set gradient value in histogram at bin index i
#[inline]
pub fn set_grad(hist: &mut [HistT], i: usize, value: HistT) {
    hist[i << 1] = value;
}

/// Set hessian value in histogram at bin index i
#[inline]
pub fn set_hess(hist: &mut [HistT], i: usize, value: HistT) {
    hist[(i << 1) + 1] = value;
}

/// Add gradient and hessian values to histogram at bin index i
#[inline]
pub fn add_grad_hess(hist: &mut [HistT], i: usize, grad: HistT, hess: HistT) {
    hist[i << 1] += grad;
    hist[(i << 1) + 1] += hess;
}

// =====================================
// Histogram Reduction Functions
// =====================================

/// Reducer function for histogram summation in distributed training
/// Equivalent to C++ HistogramSumReducer function
pub fn histogram_sum_reducer(src: &[u8], dst: &mut [u8], type_size: usize, len: CommSizeT) {
    let mut used_size = 0;
    let mut src_offset = 0;
    let mut dst_offset = 0;

    while used_size < len as usize {
        // Convert bytes to HistT values
        let src_bytes = &src[src_offset..src_offset + type_size];
        let dst_bytes = &mut dst[dst_offset..dst_offset + type_size];

        // Interpret as HistT and add
        let src_val =
            HistT::from_ne_bytes(src_bytes.try_into().expect("Invalid byte slice for HistT"));
        let mut dst_val =
            HistT::from_ne_bytes(dst_bytes.try_into().expect("Invalid byte slice for HistT"));

        dst_val += src_val;
        dst_bytes.copy_from_slice(&dst_val.to_ne_bytes());

        src_offset += type_size;
        dst_offset += type_size;
        used_size += type_size;
    }
}

/// Reducer function for 32-bit integer histograms with parallel processing
/// Equivalent to C++ Int32HistogramSumReducer function
pub fn int32_histogram_sum_reducer(src: &[u8], dst: &mut [u8], type_size: usize, len: CommSizeT) {
    // Convert byte slices to i64 slices for SIMD processing
    let src_ptr = src.as_ptr() as *const i64;
    let dst_ptr = dst.as_mut_ptr() as *mut i64;
    let steps = (len as usize + (type_size * 2) - 1) / (type_size * 2);

    unsafe {
        let src_slice = std::slice::from_raw_parts(src_ptr, steps);
        let dst_slice = std::slice::from_raw_parts_mut(dst_ptr, steps);

        // Simple sequential processing for now (parallel processing would require rayon)
        for (dst_val, src_val) in dst_slice.iter_mut().zip(src_slice.iter()) {
            *dst_val += *src_val;
        }
    }
}

/// Reducer function for 16-bit integer histograms with parallel processing
/// Equivalent to C++ Int16HistogramSumReducer function
pub fn int16_histogram_sum_reducer(src: &[u8], dst: &mut [u8], type_size: usize, len: CommSizeT) {
    // Convert byte slices to i32 slices for SIMD processing
    let src_ptr = src.as_ptr() as *const i32;
    let dst_ptr = dst.as_mut_ptr() as *mut i32;
    let steps = (len as usize + (type_size * 2) - 1) / (type_size * 2);

    unsafe {
        let src_slice = std::slice::from_raw_parts(src_ptr, steps);
        let dst_slice = std::slice::from_raw_parts_mut(dst_ptr, steps);

        // Simple sequential processing for now (parallel processing would require rayon)
        for (dst_val, src_val) in dst_slice.iter_mut().zip(src_slice.iter()) {
            *dst_val += *src_val;
        }
    }
}

// =====================================
// BinMapper Implementation
// =====================================

/// This struct is used to convert feature values into bin,
/// and store some meta information for bins
#[derive(Debug, Clone)]
pub struct BinMapper {
    /// Number of bins
    num_bin: i32,
    /// Missing value type
    missing_type: MissingType,
    /// Store upper bound for each bin
    bin_upper_bound: Vec<f64>,
    /// True if this feature is trivial (contains only one bin)
    is_trivial: bool,
    /// Sparse rate of this bins (num_bin0/num_data)
    sparse_rate: f64,
    /// Type of this bin
    bin_type: BinType,
    /// Mapper from categorical value to bin index
    categorical_2_bin: HashMap<i32, u32>,
    /// Mapper from bin index to categorical value
    bin_2_categorical: Vec<i32>,
    /// Minimal feature value
    min_val: f64,
    /// Maximum feature value
    max_val: f64,
    /// Bin value of feature value 0
    default_bin: u32,
    /// Most frequent bin
    most_freq_bin: u32,
}

impl BinMapper {
    /// Create a new empty BinMapper
    pub fn new() -> Self {
        Self {
            num_bin: 0,
            missing_type: MissingType::default(),
            bin_upper_bound: Vec::new(),
            is_trivial: false,
            sparse_rate: 0.0,
            bin_type: BinType::default(),
            categorical_2_bin: HashMap::new(),
            bin_2_categorical: Vec::new(),
            min_val: 0.0,
            max_val: 0.0,
            default_bin: 0,
            most_freq_bin: 0,
        }
    }

    /// Copy constructor equivalent
    pub fn from_other(other: &BinMapper) -> Self {
        other.clone()
    }

    /// Constructor from memory buffer
    pub fn from_memory(memory: &[u8]) -> Result<Self> {
        let mut mapper = Self::new();
        mapper.copy_from(memory)?;
        Ok(mapper)
    }

    /// Check if this BinMapper aligns with another BinMapper
    pub fn check_align(&self, other: &BinMapper) -> bool {
        if self.num_bin != other.num_bin {
            return false;
        }
        if self.missing_type != other.missing_type {
            return false;
        }
        if self.bin_type == BinType::Numerical {
            for i in 0..self.num_bin as usize {
                if i >= self.bin_upper_bound.len() || i >= other.bin_upper_bound.len() {
                    return false;
                }
                if self.bin_upper_bound[i] != other.bin_upper_bound[i] {
                    return false;
                }
            }
        } else {
            for i in 0..self.num_bin as usize {
                if i >= self.bin_2_categorical.len() || i >= other.bin_2_categorical.len() {
                    return false;
                }
                if self.bin_2_categorical[i] != other.bin_2_categorical[i] {
                    return false;
                }
            }
        }
        true
    }

    /// Get number of bins
    #[inline]
    pub fn num_bin(&self) -> i32 {
        self.num_bin
    }

    /// Get missing type
    #[inline]
    pub fn missing_type(&self) -> MissingType {
        self.missing_type
    }

    /// Check if bin is trivial (contains only one bin)
    #[inline]
    pub fn is_trivial(&self) -> bool {
        self.is_trivial
    }

    /// Get sparsity of this bin (num_zero_bins / num_data)
    #[inline]
    pub fn sparse_rate(&self) -> f64 {
        self.sparse_rate
    }

    /// Get bin type
    #[inline]
    pub fn bin_type(&self) -> BinType {
        self.bin_type
    }

    /// Save binary data to file using the provided binary writer
    pub fn save_binary_to_file(&self, writer: &mut dyn BinaryWriter) -> Result<()> {
        // Implementation would write binary data
        // This is a placeholder for the actual implementation
        Ok(())
    }

    /// Map bin index to feature value
    #[inline]
    pub fn bin_to_value(&self, bin: u32) -> f64 {
        if self.bin_type == BinType::Numerical {
            if let Some(value) = self.bin_upper_bound.get(bin as usize) {
                *value
            } else {
                0.0
            }
        } else {
            if let Some(value) = self.bin_2_categorical.get(bin as usize) {
                *value as f64
            } else {
                0.0
            }
        }
    }

    /// Get maximum categorical value
    pub fn max_cat_value(&self) -> i32 {
        if self.bin_2_categorical.is_empty() {
            return 0;
        }
        let mut max_cat_value = self.bin_2_categorical[0];
        for &value in &self.bin_2_categorical[1..] {
            if value > max_cat_value {
                max_cat_value = value;
            }
        }
        max_cat_value
    }

    /// Get sizes in bytes of this object
    pub fn sizes_in_byte(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.bin_upper_bound.len() * std::mem::size_of::<f64>()
            + self.categorical_2_bin.len()
                * (std::mem::size_of::<i32>() + std::mem::size_of::<u32>())
            + self.bin_2_categorical.len() * std::mem::size_of::<i32>()
    }

    /// Map feature value to bin index
    #[inline]
    pub fn value_to_bin(&self, value: f64) -> u32 {
        let mut value = value;

        if value.is_nan() {
            if self.bin_type == BinType::Categorical {
                return 0;
            } else if self.missing_type == MissingType::NaN {
                return (self.num_bin - 1) as u32;
            } else {
                value = 0.0;
            }
        }

        if self.bin_type == BinType::Numerical {
            // Binary search to find bin
            let mut l = 0;
            let mut r = self.num_bin - 1;
            if self.missing_type == MissingType::NaN {
                r -= 1;
            }

            while l < r {
                let m = (r + l - 1) / 2;
                if let Some(&upper_bound) = self.bin_upper_bound.get(m as usize) {
                    if value <= upper_bound {
                        r = m;
                    } else {
                        l = m + 1;
                    }
                } else {
                    break;
                }
            }
            l as u32
        } else {
            let int_value = value as i32;
            // Convert negative value to NaN bin
            if int_value < 0 {
                return 0;
            }

            if let Some(&bin) = self.categorical_2_bin.get(&int_value) {
                bin
            } else {
                0
            }
        }
    }

    /// Get the default bin when value is 0
    #[inline]
    pub fn get_default_bin(&self) -> u32 {
        self.default_bin
    }

    /// Get the most frequent bin
    #[inline]
    pub fn get_most_freq_bin(&self) -> u32 {
        self.most_freq_bin
    }

    /// Find bins for feature values based on the provided configuration
    /// 
    /// # Arguments
    /// * `values` - Mutable array of feature values to analyze
    /// * `num_values` - Number of values to process
    /// * `total_sample_cnt` - Total number of samples in the dataset
    /// * `max_bin` - Maximum number of bins to create
    /// * `min_data_in_bin` - Minimum data points required per bin
    /// * `min_split_data` - Minimum data points required for a split
    /// * `pre_filter` - Whether to pre-filter the data
    /// * `bin_type` - Type of binning (numerical or categorical)
    /// * `use_missing` - Whether to handle missing values
    /// * `zero_as_missing` - Whether to treat zero as missing value
    /// * `forced_upper_bounds` - Predefined upper bounds for bins
    pub fn find_bin(
        &mut self,
        values: &mut [f64],
        num_values: usize,
        total_sample_cnt: usize,
        max_bin: i32,
        min_data_in_bin: i32,
        min_split_data: i32,
        pre_filter: bool,
        bin_type: BinType,
        use_missing: bool,
        zero_as_missing: bool,
        forced_upper_bounds: &[f64],
    ) {
        // This is a complex implementation that would analyze the feature values
        // and create appropriate bins. For now, this is a placeholder.
        self.bin_type = bin_type;
        self.num_bin = max_bin.min(num_values as i32);

        if self.num_bin <= 1 {
            self.is_trivial = true;
        }

        // Set default values for now
        if bin_type == BinType::Numerical {
            if !values.is_empty() {
                self.min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
                self.max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            }
        }
    }

    /// Serialize this object to buffer
    pub fn copy_to(&self, buffer: &mut [u8]) -> Result<()> {
        // Implementation would serialize the object to bytes
        // This is a placeholder for the actual implementation
        Ok(())
    }

    /// Deserialize this object from buffer
    pub fn copy_from(&mut self, buffer: &[u8]) -> Result<()> {
        // Implementation would deserialize the object from bytes
        // This is a placeholder for the actual implementation
        Ok(())
    }

    /// Get bin info as string
    pub fn bin_info_string(&self) -> String {
        if self.bin_type == BinType::Categorical {
            Common::join(&self.bin_2_categorical, ":")
        } else {
            format!("[{}:{}]", self.min_val, self.max_val)
        }
    }
}

impl Default for BinMapper {
    fn default() -> Self {
        Self::new()
    }
}

// =====================================
// BinIterator Trait
// =====================================

/// Iterator for one bin column
pub trait BinIterator {
    /// Get bin data on specific row index
    fn get(&self, idx: DataSize) -> u32;

    /// Get raw bin data on specific row index
    fn raw_get(&self, idx: DataSize) -> u32;

    /// Reset iterator to specific index
    fn reset(&mut self, idx: DataSize);
}

// =====================================
// Bin Trait
// =====================================

/// Interface for bin data. This trait stores bin data for one feature.
/// Unlike OrderedBin, this stores data by original order.
/// Note that it may cause cache misses when construct histogram,
/// but it doesn't need to re-order operation, so it will be faster than OrderedBin for dense feature
pub trait Bin: Send + Sync + std::fmt::Debug {
    /// Initialize for pushing. By default, no action needed.
    /// num_thread: The number of external threads that will be calling the push APIs
    /// omp_max_threads: The maximum number of OpenMP threads to allocate for
    fn init_streaming(&mut self, _num_thread: u32, _omp_max_threads: i32) {
        // Default implementation does nothing
    }

    /// Push one record
    /// tid: Thread id
    /// idx: Index of record
    /// value: bin value of record
    fn push(&mut self, tid: i32, idx: DataSize, value: u32);

    /// Copy subset of rows
    fn copy_subrow(
        &mut self,
        full_bin: &dyn Bin,
        used_indices: &[DataSize],
        num_used_indices: DataSize,
    );

    /// Get bin iterator of this bin for specific feature
    /// min_bin: min_bin of current used feature
    /// max_bin: max_bin of current used feature
    /// most_freq_bin: most frequent bin
    fn get_iterator(&self, min_bin: u32, max_bin: u32, most_freq_bin: u32) -> Box<dyn BinIterator>;

    /// Save binary data to file
    fn save_binary_to_file(&self, writer: &mut dyn BinaryWriter) -> Result<()>;

    /// Load from memory
    fn load_from_memory(&mut self, memory: &[u8], local_used_indices: &[DataSize]) -> Result<()>;

    /// Get sizes in byte of this object
    fn sizes_in_byte(&self) -> usize;

    /// Number of all data
    fn num_data(&self) -> DataSize;

    /// Get data pointer
    fn get_data(&self) -> *const u8;

    /// Resize the bin
    fn resize(&mut self, num_data: DataSize);

    /// Construct histogram of this feature
    /// Note: We use ordered_gradients and ordered_hessians to improve cache hit chance
    /// The naive solution is using gradients[data_indices[i]] for data_indices[i] to get gradients,
    /// which is not cache friendly, since the access of memory is not continuous.
    /// ordered_gradients and ordered_hessians are preprocessed, and they are re-ordered by data_indices.
    /// Ordered_gradients[i] is aligned with data_indices[i]'s gradients (same for ordered_hessians).
    fn construct_histogram(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[ScoreT],
        ordered_hessians: &[ScoreT],
        out: &mut [HistT],
    );

    /// Construct histogram without data indices
    fn construct_histogram_no_indices(
        &self,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[ScoreT],
        ordered_hessians: &[ScoreT],
        out: &mut [HistT],
    );

    /// Construct histogram with int8 optimization
    fn construct_histogram_int8(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[ScoreT],
        ordered_hessians: &[ScoreT],
        out: &mut [HistT],
    );

    /// Construct histogram with int8 optimization, no indices
    fn construct_histogram_int8_no_indices(
        &self,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[ScoreT],
        ordered_hessians: &[ScoreT],
        out: &mut [HistT],
    );

    /// Construct histogram with int16 optimization
    fn construct_histogram_int16(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[ScoreT],
        ordered_hessians: &[ScoreT],
        out: &mut [HistT],
    );

    /// Construct histogram with int16 optimization, no indices
    fn construct_histogram_int16_no_indices(
        &self,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[ScoreT],
        ordered_hessians: &[ScoreT],
        out: &mut [HistT],
    );

    /// Construct histogram with int32 optimization
    fn construct_histogram_int32(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[ScoreT],
        ordered_hessians: &[ScoreT],
        out: &mut [HistT],
    );

    /// Construct histogram with int32 optimization, no indices
    fn construct_histogram_int32_no_indices(
        &self,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[ScoreT],
        ordered_hessians: &[ScoreT],
        out: &mut [HistT],
    );

    /// Construct histogram for gradients only (no hessians)
    fn construct_histogram_grad_only(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[ScoreT],
        out: &mut [HistT],
    );

    /// Construct histogram for gradients only, no indices
    fn construct_histogram_grad_only_no_indices(
        &self,
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[ScoreT],
        out: &mut [HistT],
    );

    /// Split data based on threshold
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

    /// Split categorical data
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

    /// Split data (simplified version)
    fn split_simple(
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

    /// Split categorical data (simplified version)
    fn split_categorical_simple(
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

    /// After pushed all feature data, call this could have better refactor for bin data
    fn finish_load(&mut self);

    /// Deep copy the bin
    fn clone_bin(&self) -> Box<dyn Bin>;

    /// Get column-wise data for GPU usage
    fn get_col_wise_data(
        &self,
        bit_type: &mut u8,
        is_sparse: &mut bool,
        bin_iterator: &mut [Box<dyn BinIterator>],
        num_threads: i32,
    ) -> *const u8;

    /// Get column-wise data (simplified version)
    fn get_col_wise_data_simple(
        &self,
        bit_type: &mut u8,
        is_sparse: &mut bool,
        bin_iterator: &mut Box<dyn BinIterator>,
    ) -> *const u8;
}

/// Factory methods for creating bins
impl dyn Bin {
    /// Create object for bin data of one feature, used for dense feature
    pub fn create_dense_bin(num_data: DataSize, num_bin: i32) -> Box<dyn Bin> {
        // This would return an actual implementation
        // For now, this is a placeholder
        unimplemented!("DenseBin implementation needed")
    }

    /// Create object for bin data of one feature, used for sparse feature
    pub fn create_sparse_bin(num_data: DataSize, num_bin: i32) -> Box<dyn Bin> {
        // This would return an actual implementation
        // For now, this is a placeholder
        unimplemented!("SparseBin implementation needed")
    }
}

// =====================================
// MultiValBin Trait
// =====================================

/// Interface for multi-value bin data, used for features that can have multiple values per sample
pub trait MultiValBin: Send + Sync {
    /// Number of data points
    fn num_data(&self) -> DataSize;

    /// Number of bins
    fn num_bin(&self) -> i32;

    /// Average number of elements per row
    fn num_element_per_row(&self) -> f64;

    /// Get offsets array
    fn offsets(&self) -> &[u32];

    /// Push one row of data
    fn push_one_row(&mut self, tid: i32, idx: DataSize, values: &[u32]);

    /// Copy subset of rows
    fn copy_subrow(
        &mut self,
        full_bin: &dyn MultiValBin,
        used_indices: &[DataSize],
        num_used_indices: DataSize,
    );

    /// Create a similar multi-value bin
    fn create_like(
        &self,
        num_data: DataSize,
        num_bin: i32,
        num_feature: i32,
        estimate_element_per_row: f64,
        offsets: &[u32],
    ) -> Box<dyn MultiValBin>;

    /// Copy subset of columns
    fn copy_subcol(
        &mut self,
        full_bin: &dyn MultiValBin,
        used_feature_index: &[i32],
        lower: &[u32],
        upper: &[u32],
        delta: &[u32],
    );

    /// Resize the multi-value bin
    fn resize(
        &mut self,
        num_data: DataSize,
        num_bin: i32,
        num_feature: i32,
        estimate_element_per_row: f64,
        offsets: &[u32],
    );

    /// Copy subset of rows and columns
    fn copy_subrow_and_subcol(
        &mut self,
        full_bin: &dyn MultiValBin,
        used_indices: &[DataSize],
        num_used_indices: DataSize,
        used_feature_index: &[i32],
        lower: &[u32],
        upper: &[u32],
        delta: &[u32],
    );

    /// Construct histogram with data indices
    fn construct_histogram(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        gradients: &[ScoreT],
        hessians: &[ScoreT],
        out: &mut [HistT],
    );

    /// Construct histogram without data indices
    fn construct_histogram_no_indices(
        &self,
        start: DataSize,
        end: DataSize,
        gradients: &[ScoreT],
        hessians: &[ScoreT],
        out: &mut [HistT],
    );

    /// Construct histogram with ordered gradients and hessians
    fn construct_histogram_ordered(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[ScoreT],
        ordered_hessians: &[ScoreT],
        out: &mut [HistT],
    );

    /// Construct histogram with int32 optimization
    fn construct_histogram_int32(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        gradients: &[ScoreT],
        hessians: &[ScoreT],
        out: &mut [HistT],
    );

    /// Construct histogram with int32 optimization, no indices
    fn construct_histogram_int32_no_indices(
        &self,
        start: DataSize,
        end: DataSize,
        gradients: &[ScoreT],
        hessians: &[ScoreT],
        out: &mut [HistT],
    );

    /// Construct histogram with int32 optimization and ordered data
    fn construct_histogram_ordered_int32(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[ScoreT],
        ordered_hessians: &[ScoreT],
        out: &mut [HistT],
    );

    /// Construct histogram with int16 optimization
    fn construct_histogram_int16(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        gradients: &[ScoreT],
        hessians: &[ScoreT],
        out: &mut [HistT],
    );

    /// Construct histogram with int16 optimization, no indices
    fn construct_histogram_int16_no_indices(
        &self,
        start: DataSize,
        end: DataSize,
        gradients: &[ScoreT],
        hessians: &[ScoreT],
        out: &mut [HistT],
    );

    /// Construct histogram with int16 optimization and ordered data
    fn construct_histogram_ordered_int16(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[ScoreT],
        ordered_hessians: &[ScoreT],
        out: &mut [HistT],
    );

    /// Construct histogram with int8 optimization
    fn construct_histogram_int8(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        gradients: &[ScoreT],
        hessians: &[ScoreT],
        out: &mut [HistT],
    );

    /// Construct histogram with int8 optimization, no indices
    fn construct_histogram_int8_no_indices(
        &self,
        start: DataSize,
        end: DataSize,
        gradients: &[ScoreT],
        hessians: &[ScoreT],
        out: &mut [HistT],
    );

    /// Construct histogram with int8 optimization and ordered data
    fn construct_histogram_ordered_int8(
        &self,
        data_indices: &[DataSize],
        start: DataSize,
        end: DataSize,
        ordered_gradients: &[ScoreT],
        ordered_hessians: &[ScoreT],
        out: &mut [HistT],
    );

    /// Finish loading data
    fn finish_load(&mut self);

    /// Check if this is a sparse representation
    fn is_sparse(&self) -> bool;

    /// Deep copy the multi-value bin
    fn clone_multi_val_bin(&self) -> Box<dyn MultiValBin>;

    /// Get row-wise data for GPU usage (CUDA-specific)
    #[cfg(feature = "gpu")]
    fn get_row_wise_data(
        &self,
        bit_type: &mut u8,
        total_size: &mut usize,
        is_sparse: &mut bool,
        out_data_ptr: &mut *const u8,
        data_ptr_bit_type: &mut u8,
    ) -> *const u8;
}

/// Factory methods for creating multi-value bins
impl dyn MultiValBin {
    /// Sparsity threshold for multi-value bins
    pub const MULTI_VAL_BIN_SPARSE_THRESHOLD: f64 = 0.25;

    /// Create a multi-value bin (automatically chooses dense or sparse based on sparsity)
    pub fn create_multi_val_bin(
        num_data: DataSize,
        num_bin: i32,
        num_feature: i32,
        sparse_rate: f64,
        offsets: &[u32],
    ) -> Box<dyn MultiValBin> {
        if sparse_rate > Self::MULTI_VAL_BIN_SPARSE_THRESHOLD {
            Self::create_multi_val_sparse_bin(num_data, num_bin, sparse_rate)
        } else {
            Self::create_multi_val_dense_bin(num_data, num_bin, num_feature, offsets)
        }
    }

    /// Create a dense multi-value bin
    pub fn create_multi_val_dense_bin(
        num_data: DataSize,
        num_bin: i32,
        num_feature: i32,
        offsets: &[u32],
    ) -> Box<dyn MultiValBin> {
        // This would return an actual implementation
        // For now, this is a placeholder
        unimplemented!("MultiValDenseBin implementation needed")
    }

    /// Create a sparse multi-value bin
    pub fn create_multi_val_sparse_bin(
        num_data: DataSize,
        num_bin: i32,
        estimate_element_per_row: f64,
    ) -> Box<dyn MultiValBin> {
        // This would return an actual implementation
        // For now, this is a placeholder
        unimplemented!("MultiValSparseBin implementation needed")
    }
}

// =====================================
// Tests
// =====================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hist_access_functions() {
        let mut hist = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        // Test gradient access
        assert_eq!(get_grad(&hist, 0), 1.0);
        assert_eq!(get_grad(&hist, 1), 3.0);
        assert_eq!(get_grad(&hist, 2), 5.0);

        // Test hessian access
        assert_eq!(get_hess(&hist, 0), 2.0);
        assert_eq!(get_hess(&hist, 1), 4.0);
        assert_eq!(get_hess(&hist, 2), 6.0);

        // Test setting values
        set_grad(&mut hist, 1, 10.0);
        set_hess(&mut hist, 1, 20.0);
        assert_eq!(hist[2], 10.0);
        assert_eq!(hist[3], 20.0);

        // Test adding values
        add_grad_hess(&mut hist, 0, 1.0, 2.0);
        assert_eq!(hist[0], 2.0);
        assert_eq!(hist[1], 4.0);
    }

    #[test]
    fn test_bin_type() {
        let numerical = BinType::Numerical;
        let categorical = BinType::Categorical;

        assert_eq!(numerical.to_string(), "numerical");
        assert_eq!(categorical.to_string(), "categorical");
        assert_eq!(BinType::default(), BinType::Numerical);
    }

    #[test]
    fn test_bin_mapper_creation() {
        let mapper = BinMapper::new();
        assert_eq!(mapper.num_bin(), 0);
        assert_eq!(mapper.bin_type(), BinType::Numerical);
        assert_eq!(mapper.missing_type(), MissingType::default());
        assert!(mapper.is_trivial() == false);
    }

    #[test]
    fn test_bin_mapper_value_to_bin() {
        let mapper = BinMapper::new();
        // With empty mapper, should return 0
        assert_eq!(mapper.value_to_bin(5.0), 0);
        assert_eq!(mapper.value_to_bin(f64::NAN), 0);
    }

    #[test]
    fn test_constants() {
        assert_eq!(HIST_ENTRY_SIZE, 16); // 2 * 8 bytes for f64
        assert_eq!(INT32_HIST_ENTRY_SIZE, 8); // 2 * 4 bytes for i32
        assert_eq!(INT16_HIST_ENTRY_SIZE, 4); // 2 * 2 bytes for i16
        assert_eq!(HIST_OFFSET, 2);
        assert_eq!(SPARSE_THRESHOLD, 0.7);
    }
}
