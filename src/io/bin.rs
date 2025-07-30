//! Binary data structures and operations for feature binning.
//!
//! This module provides a pure Rust implementation of LightGBM's bin mapping
//! functionality, translating the C++ bin.cpp implementation while maintaining
//! semantic equivalence and leveraging Rust's type system for enhanced safety.

use crate::core::{constants::*, error::*, types::*};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Zero threshold constant for feature binning
const K_ZERO_THRESHOLD: f64 = 1e-35;

/// Sparse threshold for determining bin storage strategy
const K_SPARSE_THRESHOLD: f64 = 0.8;

/// Threshold for multi-value sparse bins
const MULTI_VAL_BIN_SPARSE_THRESHOLD: f64 = 0.25;

/// NaN representation as f64
const NAN: f64 = f64::NAN;

/// Bin type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinType {
    /// Numerical feature bin
    NumericalBin,
    /// Categorical feature bin
    CategoricalBin,
}

impl Default for BinType {
    fn default() -> Self {
        BinType::NumericalBin
    }
}

/// Iterator for sparse bin data
pub trait BinIterator {
    ///
    fn reset(&mut self, start_idx: usize);
    ///
    fn get(&self, idx: usize) -> BinIndex;
}

/// Binary mapper for feature discretization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinMapper {
    /// Number of bins for this feature
    num_bin_: i32,
    /// Missing value handling type
    missing_type_: MissingType,
    /// Whether this feature is trivial (single value)
    is_trivial_: bool,
    /// Sparsity rate of the most frequent bin
    sparse_rate_: f64,
    /// Type of binning (numerical or categorical)
    bin_type_: BinType,
    /// Upper bounds for numerical bins
    bin_upper_bound_: Vec<f64>,
    /// Mapping from bin index to categorical value
    bin_2_categorical_: Vec<i32>,
    /// Mapping from categorical value to bin index
    categorical_2_bin_: HashMap<i32, u32>,
    /// Minimum feature value
    min_val_: f64,
    /// Maximum feature value
    max_val_: f64,
    /// Default bin index (usually for zero values)
    default_bin_: u32,
    /// Most frequent bin index
    most_freq_bin_: u32,
}

impl BinMapper {
    /// Create a new default BinMapper
    pub fn new() -> Self {
        Self {
            num_bin_: 1,
            is_trivial_: true,
            bin_type_: BinType::NumericalBin,
            bin_upper_bound_: vec![f64::INFINITY],
            bin_2_categorical_: Vec::new(),
            categorical_2_bin_: HashMap::new(),
            missing_type_: MissingType::None,
            sparse_rate_: 0.0,
            min_val_: 0.0,
            max_val_: 0.0,
            default_bin_: 0,
            most_freq_bin_: 0,
        }
    }

    /// Create a copy of another BinMapper
    pub fn copy_from(other: &BinMapper) -> Self {
        Self {
            num_bin_: other.num_bin_,
            missing_type_: other.missing_type_,
            is_trivial_: other.is_trivial_,
            sparse_rate_: other.sparse_rate_,
            bin_type_: other.bin_type_,
            bin_upper_bound_: if other.bin_type_ == BinType::NumericalBin {
                other.bin_upper_bound_.clone()
            } else {
                Vec::new()
            },
            bin_2_categorical_: if other.bin_type_ == BinType::CategoricalBin {
                other.bin_2_categorical_.clone()
            } else {
                Vec::new()
            },
            categorical_2_bin_: if other.bin_type_ == BinType::CategoricalBin {
                other.categorical_2_bin_.clone()
            } else {
                HashMap::new()
            },
            min_val_: other.min_val_,
            max_val_: other.max_val_,
            default_bin_: other.default_bin_,
            most_freq_bin_: other.most_freq_bin_,
        }
    }

    /// Convert a feature value to its corresponding bin index
    pub fn value_to_bin(&self, value: f64) -> u32 {
        if value.is_nan() {
            if self.missing_type_ == MissingType::NaN {
                return (self.num_bin_ - 1) as u32;
            } else {
                return self.default_bin_;
            }
        }

        if self.bin_type_ == BinType::NumericalBin {
            let mut bin_idx = 0;
            for (i, &upper_bound) in self.bin_upper_bound_.iter().enumerate() {
                if value <= upper_bound {
                    bin_idx = i;
                    break;
                }
            }
            bin_idx as u32
        } else {
            // Categorical bin
            let int_val = value as i32;
            if int_val < 0 {
                0 // NaN bin for negative values
            } else {
                *self.categorical_2_bin_.get(&int_val).unwrap_or(&0)
            }
        }
    }

    /// Find bins for the given feature values
    pub fn find_bin(
        &mut self,
        values: &mut [f64],
        num_sample_values: usize,
        total_sample_cnt: usize,
        max_bin: i32,
        min_data_in_bin: i32,
        min_split_data: i32,
        pre_filter: bool,
        bin_type: BinType,
        use_missing: bool,
        zero_as_missing: bool,
        forced_upper_bounds: &[f64],
    ) -> Result<()> {
        let mut na_cnt = 0;
        let mut non_na_cnt = 0;

        // Filter out NaN values
        for i in 0..num_sample_values {
            if !values[i].is_nan() {
                if non_na_cnt != i {
                    values[non_na_cnt] = values[i];
                }
                non_na_cnt += 1;
            }
        }

        // Determine missing type
        if !use_missing {
            self.missing_type_ = MissingType::None;
        } else if zero_as_missing {
            self.missing_type_ = MissingType::Zero;
        } else if non_na_cnt == num_sample_values {
            self.missing_type_ = MissingType::None;
        } else {
            self.missing_type_ = MissingType::NaN;
            na_cnt = num_sample_values - non_na_cnt;
        }

        self.bin_type_ = bin_type;
        self.default_bin_ = 0;
        let zero_cnt = (total_sample_cnt - num_sample_values - na_cnt) as i32;

        // Find distinct values and their counts
        let mut distinct_values = Vec::new();
        let mut counts = Vec::new();

        values[..non_na_cnt].sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Push zero in the front if needed
        if non_na_cnt == 0 || (values[0] > 0.0 && zero_cnt > 0) {
            distinct_values.push(0.0);
            counts.push(zero_cnt);
        }

        if non_na_cnt > 0 {
            distinct_values.push(values[0]);
            counts.push(1);
        }

        for i in 1..non_na_cnt {
            if !self.check_double_equal_ordered(values[i - 1], values[i]) {
                if values[i - 1] < 0.0 && values[i] > 0.0 {
                    distinct_values.push(0.0);
                    counts.push(zero_cnt);
                }
                distinct_values.push(values[i]);
                counts.push(1);
            } else {
                // Use the larger value
                *distinct_values.last_mut().unwrap() = values[i];
                *counts.last_mut().unwrap() += 1;
            }
        }

        // Push zero in the back if needed
        if non_na_cnt > 0 && values[non_na_cnt - 1] < 0.0 && zero_cnt > 0 {
            distinct_values.push(0.0);
            counts.push(zero_cnt);
        }

        self.min_val_ = *distinct_values.first().unwrap_or(&0.0);
        self.max_val_ = *distinct_values.last().unwrap_or(&0.0);

        let num_distinct_values = distinct_values.len();
        let mut cnt_in_bin: Vec<i32>;

        if self.bin_type_ == BinType::NumericalBin {
            // Handle numerical binning
            if self.missing_type_ == MissingType::Zero {
                self.bin_upper_bound_ = find_bin_with_zero_as_one_bin(
                    &distinct_values,
                    &counts,
                    num_distinct_values,
                    max_bin,
                    total_sample_cnt,
                    min_data_in_bin,
                    forced_upper_bounds,
                )?;
                if self.bin_upper_bound_.len() == 2 {
                    self.missing_type_ = MissingType::None;
                }
            } else if self.missing_type_ == MissingType::None {
                self.bin_upper_bound_ = find_bin_with_zero_as_one_bin(
                    &distinct_values,
                    &counts,
                    num_distinct_values,
                    max_bin,
                    total_sample_cnt,
                    min_data_in_bin,
                    forced_upper_bounds,
                )?;
            } else {
                self.bin_upper_bound_ = find_bin_with_zero_as_one_bin(
                    &distinct_values,
                    &counts,
                    num_distinct_values,
                    max_bin - 1,
                    total_sample_cnt - na_cnt,
                    min_data_in_bin,
                    forced_upper_bounds,
                )?;
                self.bin_upper_bound_.push(NAN);
            }

            self.num_bin_ = self.bin_upper_bound_.len() as i32;

            // Calculate counts in each bin
            cnt_in_bin = vec![0; self.num_bin_ as usize];
            let mut i_bin = 0;
            for (i, &distinct_val) in distinct_values.iter().enumerate() {
                while distinct_val > self.bin_upper_bound_[i_bin]
                    && i_bin < (self.num_bin_ - 1) as usize
                {
                    i_bin += 1;
                }
                cnt_in_bin[i_bin] += counts[i];
            }

            if self.missing_type_ == MissingType::NaN {
                cnt_in_bin[(self.num_bin_ - 1) as usize] = na_cnt as i32;
            }
        } else {
            // Handle categorical binning
            return self.handle_categorical_binning(
                &distinct_values,
                &counts,
                total_sample_cnt,
                na_cnt,
                max_bin,
                min_data_in_bin,
            );
        }

        // Check if feature is trivial
        if self.num_bin_ <= 1 {
            self.is_trivial_ = true;
        } else {
            self.is_trivial_ = false;
        }

        // Check if we need to filter
        if !self.is_trivial_
            && pre_filter
            && need_filter(
                &cnt_in_bin,
                total_sample_cnt as i32,
                min_split_data,
                self.bin_type_,
            )
        {
            self.is_trivial_ = true;
        }

        if !self.is_trivial_ {
            self.default_bin_ = self.value_to_bin(0.0);
            self.most_freq_bin_ = arg_max(&cnt_in_bin) as u32;
            let max_sparse_rate =
                cnt_in_bin[self.most_freq_bin_ as usize] as f64 / total_sample_cnt as f64;

            // Use default_bin when not so sparse
            if self.most_freq_bin_ != self.default_bin_ && max_sparse_rate < K_SPARSE_THRESHOLD {
                self.most_freq_bin_ = self.default_bin_;
            }
            self.sparse_rate_ =
                cnt_in_bin[self.most_freq_bin_ as usize] as f64 / total_sample_cnt as f64;
        } else {
            self.sparse_rate_ = 1.0;
        }

        Ok(())
    }

    /// Handle categorical feature binning
    fn handle_categorical_binning(
        &mut self,
        distinct_values: &[f64],
        counts: &[i32],
        total_sample_cnt: usize,
        na_cnt: usize,
        max_bin: i32,
        min_data_in_bin: i32,
    ) -> Result<()> {
        // Convert to integer values
        let mut distinct_values_int = Vec::new();
        let mut counts_int = Vec::new();
        let mut additional_na_cnt = 0;

        for (i, &val) in distinct_values.iter().enumerate() {
            let int_val = val as i32;
            if int_val < 0 {
                additional_na_cnt += counts[i];
                log::warn!("Met negative value in categorical features, will convert it to NaN");
            } else if distinct_values_int.is_empty()
                || int_val != *distinct_values_int.last().unwrap()
            {
                distinct_values_int.push(int_val);
                counts_int.push(counts[i]);
            } else {
                *counts_int.last_mut().unwrap() += counts[i];
            }
        }

        let total_na_cnt = na_cnt + additional_na_cnt as usize;
        let rest_cnt = total_sample_cnt - total_na_cnt;

        if rest_cnt > 0 {
            const SPARSE_RATIO: i32 = 100;
            if !distinct_values_int.is_empty()
                && distinct_values_int.last().unwrap() / SPARSE_RATIO
                    > distinct_values_int.len() as i32
            {
                log::warn!("Met categorical feature which contains sparse values. Consider renumbering to consecutive integers started from zero");
            }

            // Sort by counts in descending order
            let mut pairs: Vec<(i32, i32)> = counts_int
                .into_iter()
                .zip(distinct_values_int.into_iter())
                .collect();
            pairs.sort_by(|a, b| b.0.cmp(&a.0)); // Sort by count descending
            counts_int = pairs.iter().map(|(count, _)| *count).collect();
            distinct_values_int = pairs.iter().map(|(_, val)| *val).collect();

            // Filter categorical values with small counts
            let cut_cnt = ((total_sample_cnt - total_na_cnt) as f64 * 0.99) as i32;
            let mut cur_cat_idx = 0;
            self.categorical_2_bin_.clear();
            self.bin_2_categorical_.clear();
            let mut used_cnt = 0;
            let mut distinct_cnt = distinct_values_int.len();
            if total_na_cnt > 0 {
                distinct_cnt += 1;
            }
            let max_bin = std::cmp::min(distinct_cnt as i32, max_bin);
            let mut cnt_in_bin = Vec::new();

            // Push dummy bin for NaN
            self.bin_2_categorical_.push(-1);
            self.categorical_2_bin_.insert(-1, 0);
            cnt_in_bin.push(0);
            self.num_bin_ = 1;

            while cur_cat_idx < distinct_values_int.len()
                && (used_cnt < cut_cnt || self.num_bin_ < max_bin)
            {
                if counts_int[cur_cat_idx] < min_data_in_bin && cur_cat_idx > 1 {
                    break;
                }
                self.bin_2_categorical_
                    .push(distinct_values_int[cur_cat_idx]);
                self.categorical_2_bin_
                    .insert(distinct_values_int[cur_cat_idx], self.num_bin_ as u32);
                used_cnt += counts_int[cur_cat_idx];
                cnt_in_bin.push(counts_int[cur_cat_idx]);
                self.num_bin_ += 1;
                cur_cat_idx += 1;
            }

            // Determine final missing type
            if cur_cat_idx == distinct_values_int.len() && total_na_cnt == 0 {
                self.missing_type_ = MissingType::None;
            } else {
                self.missing_type_ = MissingType::NaN;
            }

            // Fix count of NaN bin
            cnt_in_bin[0] = (total_sample_cnt - used_cnt as usize) as i32;
        }

        Ok(())
    }

    /// Check if two double values are equal with tolerance
    fn check_double_equal_ordered(&self, a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    /// Get the number of bins
    pub fn num_bin(&self) -> i32 {
        self.num_bin_
    }

    /// Check if this feature is trivial
    pub fn is_trivial(&self) -> bool {
        self.is_trivial_
    }

    /// Get the bin type
    pub fn bin_type(&self) -> BinType {
        self.bin_type_
    }

    /// Get the sparse rate
    pub fn sparse_rate(&self) -> f64 {
        self.sparse_rate_
    }

    /// Get the missing type
    pub fn missing_type(&self) -> MissingType {
        self.missing_type_
    }

    /// Get the default bin
    pub fn default_bin(&self) -> u32 {
        self.default_bin_
    }

    /// Get the most frequent bin
    pub fn most_freq_bin(&self) -> u32 {
        self.most_freq_bin_
    }
}

impl Default for BinMapper {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if filtering is needed based on bin counts
pub fn need_filter(cnt_in_bin: &[i32], total_cnt: i32, filter_cnt: i32, bin_type: BinType) -> bool {
    if bin_type == BinType::NumericalBin {
        let mut sum_left = 0;
        for i in 0..(cnt_in_bin.len() - 1) {
            sum_left += cnt_in_bin[i];
            if sum_left >= filter_cnt && total_cnt - sum_left >= filter_cnt {
                return false;
            }
        }
    } else if cnt_in_bin.len() <= 2 {
        for i in 0..(cnt_in_bin.len() - 1) {
            let sum_left = cnt_in_bin[i];
            if sum_left >= filter_cnt && total_cnt - sum_left >= filter_cnt {
                return false;
            }
        }
    } else {
        return false;
    }
    true
}

/// Find bins using greedy algorithm
pub fn greedy_find_bin(
    distinct_values: &[f64],
    counts: &[i32],
    num_distinct_values: usize,
    max_bin: i32,
    total_cnt: usize,
    min_data_in_bin: i32,
) -> Result<Vec<f64>> {
    let mut bin_upper_bound = Vec::new();

    if max_bin <= 0 {
        return Err(LightGBMError::invalid_parameter(
            "max_bin",
            max_bin.to_string(),
            "must be greater than 0",
        )
        .into());
    }

    if num_distinct_values <= max_bin as usize {
        let mut cur_cnt_inbin = 0;
        for i in 0..(num_distinct_values - 1) {
            cur_cnt_inbin += counts[i];
            if cur_cnt_inbin >= min_data_in_bin {
                let val =
                    get_double_upper_bound((distinct_values[i] + distinct_values[i + 1]) / 2.0);
                if bin_upper_bound.is_empty()
                    || !check_double_equal_ordered(bin_upper_bound.last().unwrap(), &val)
                {
                    bin_upper_bound.push(val);
                    cur_cnt_inbin = 0;
                }
            }
        }
        bin_upper_bound.push(f64::INFINITY);
    } else {
        let mut max_bin = max_bin;
        if min_data_in_bin > 0 {
            max_bin = std::cmp::min(max_bin, (total_cnt / min_data_in_bin as usize) as i32);
            max_bin = std::cmp::max(max_bin, 1);
        }
        let mean_bin_size = total_cnt as f64 / max_bin as f64;

        let mut rest_bin_cnt = max_bin;
        let mut rest_sample_cnt = total_cnt as i32;
        let mut is_big_count_value = vec![false; num_distinct_values];

        for i in 0..num_distinct_values {
            if counts[i] as f64 >= mean_bin_size {
                is_big_count_value[i] = true;
                rest_bin_cnt -= 1;
                rest_sample_cnt -= counts[i];
            }
        }
        let mut mean_bin_size = rest_sample_cnt as f64 / rest_bin_cnt as f64;
        let mut upper_bounds = vec![f64::INFINITY; max_bin as usize];
        let mut lower_bounds = vec![f64::INFINITY; max_bin as usize];

        let mut bin_cnt = 0;
        lower_bounds[bin_cnt] = distinct_values[0];
        let mut cur_cnt_inbin = 0;
        let mut rest_sample_cnt = rest_sample_cnt;

        for i in 0..(num_distinct_values - 1) {
            if !is_big_count_value[i] {
                rest_sample_cnt -= counts[i];
            }
            cur_cnt_inbin += counts[i];

            // Need a new bin
            if is_big_count_value[i]
                || cur_cnt_inbin as f64 >= mean_bin_size
                || (is_big_count_value[i + 1] && cur_cnt_inbin as f64 >= mean_bin_size * 0.5)
            {
                upper_bounds[bin_cnt] = distinct_values[i];
                bin_cnt += 1;
                if bin_cnt < max_bin as usize {
                    lower_bounds[bin_cnt] = distinct_values[i + 1];
                }
                if bin_cnt >= (max_bin - 1) as usize {
                    break;
                }
                cur_cnt_inbin = 0;
                if !is_big_count_value[i] {
                    rest_bin_cnt -= 1;
                    mean_bin_size = rest_sample_cnt as f64 / rest_bin_cnt as f64;
                }
            }
        }
        bin_cnt += 1;

        // Update bin upper bounds
        for i in 0..(bin_cnt - 1) {
            let val = get_double_upper_bound((upper_bounds[i] + lower_bounds[i + 1]) / 2.0);
            if bin_upper_bound.is_empty()
                || !check_double_equal_ordered(bin_upper_bound.last().unwrap(), &val)
            {
                bin_upper_bound.push(val);
            }
        }
        bin_upper_bound.push(f64::INFINITY);
    }

    Ok(bin_upper_bound)
}

/// Find bins with zero as one separate bin
pub fn find_bin_with_zero_as_one_bin(
    distinct_values: &[f64],
    counts: &[i32],
    num_distinct_values: usize,
    max_bin: i32,
    total_sample_cnt: usize,
    min_data_in_bin: i32,
    forced_upper_bounds: &[f64],
) -> Result<Vec<f64>> {
    if forced_upper_bounds.is_empty() {
        find_bin_with_zero_as_one_bin_simple(
            distinct_values,
            counts,
            num_distinct_values,
            max_bin,
            total_sample_cnt,
            min_data_in_bin,
        )
    } else {
        find_bin_with_predefined_bin(
            distinct_values,
            counts,
            num_distinct_values,
            max_bin,
            total_sample_cnt,
            min_data_in_bin,
            forced_upper_bounds,
        )
    }
}

/// Simple version of find_bin_with_zero_as_one_bin without forced bounds
pub fn find_bin_with_zero_as_one_bin_simple(
    distinct_values: &[f64],
    counts: &[i32],
    num_distinct_values: usize,
    max_bin: i32,
    total_sample_cnt: usize,
    min_data_in_bin: i32,
) -> Result<Vec<f64>> {
    let mut bin_upper_bound = Vec::new();
    let mut left_cnt_data = 0;
    let mut cnt_zero = 0;
    let mut right_cnt_data = 0;

    for i in 0..num_distinct_values {
        if distinct_values[i] <= -K_ZERO_THRESHOLD {
            left_cnt_data += counts[i];
        } else if distinct_values[i] > K_ZERO_THRESHOLD {
            right_cnt_data += counts[i];
        } else {
            cnt_zero += counts[i];
        }
    }

    let mut left_cnt = None;
    for i in 0..num_distinct_values {
        if distinct_values[i] > -K_ZERO_THRESHOLD {
            left_cnt = Some(i);
            break;
        }
    }
    let left_cnt = left_cnt.unwrap_or(num_distinct_values);

    if left_cnt > 0 && max_bin > 1 {
        let left_max_bin = ((left_cnt_data as f64) / (total_sample_cnt - cnt_zero as usize) as f64
            * (max_bin - 1) as f64) as i32;
        let left_max_bin = std::cmp::max(1, left_max_bin);
        bin_upper_bound = greedy_find_bin(
            &distinct_values[..left_cnt],
            &counts[..left_cnt],
            left_cnt,
            left_max_bin,
            left_cnt_data as usize,
            min_data_in_bin,
        )?;
        if !bin_upper_bound.is_empty() {
            *bin_upper_bound.last_mut().unwrap() = -K_ZERO_THRESHOLD;
        }
    }

    let mut right_start = None;
    for i in left_cnt..num_distinct_values {
        if distinct_values[i] > K_ZERO_THRESHOLD {
            right_start = Some(i);
            break;
        }
    }

    let right_max_bin = max_bin - 1 - bin_upper_bound.len() as i32;
    if let Some(right_start) = right_start {
        if right_max_bin > 0 {
            let right_bounds = greedy_find_bin(
                &distinct_values[right_start..],
                &counts[right_start..],
                num_distinct_values - right_start,
                right_max_bin,
                right_cnt_data as usize,
                min_data_in_bin,
            )?;
            bin_upper_bound.push(K_ZERO_THRESHOLD);
            bin_upper_bound.extend(right_bounds);
        } else {
            bin_upper_bound.push(f64::INFINITY);
        }
    } else {
        bin_upper_bound.push(f64::INFINITY);
    }

    Ok(bin_upper_bound)
}

/// Find bins with predefined forced bounds
pub fn find_bin_with_predefined_bin(
    distinct_values: &[f64],
    counts: &[i32],
    num_distinct_values: usize,
    max_bin: i32,
    total_sample_cnt: usize,
    min_data_in_bin: i32,
    forced_upper_bounds: &[f64],
) -> Result<Vec<f64>> {
    let mut bin_upper_bound = Vec::new();

    // Get number of positive and negative distinct values
    let mut left_cnt = None;
    for i in 0..num_distinct_values {
        if distinct_values[i] > -K_ZERO_THRESHOLD {
            left_cnt = Some(i);
            break;
        }
    }
    let left_cnt = left_cnt.unwrap_or(num_distinct_values);

    let mut right_start = None;
    for i in left_cnt..num_distinct_values {
        if distinct_values[i] > K_ZERO_THRESHOLD {
            right_start = Some(i);
            break;
        }
    }

    // Include zero bounds and infinity bound
    if max_bin == 2 {
        if left_cnt == 0 {
            bin_upper_bound.push(K_ZERO_THRESHOLD);
        } else {
            bin_upper_bound.push(-K_ZERO_THRESHOLD);
        }
    } else if max_bin >= 3 {
        if left_cnt > 0 {
            bin_upper_bound.push(-K_ZERO_THRESHOLD);
        }
        if right_start.is_some() {
            bin_upper_bound.push(K_ZERO_THRESHOLD);
        }
    }
    bin_upper_bound.push(f64::INFINITY);

    // Add forced bounds, excluding zeros
    let max_to_insert = max_bin - bin_upper_bound.len() as i32;
    let mut num_inserted = 0;
    for &forced_bound in forced_upper_bounds.iter() {
        if num_inserted >= max_to_insert {
            break;
        }
        if forced_bound.abs() > K_ZERO_THRESHOLD {
            bin_upper_bound.push(forced_bound);
            num_inserted += 1;
        }
    }
    bin_upper_bound.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Find remaining bounds
    let free_bins = max_bin - bin_upper_bound.len() as i32;
    let mut bounds_to_add: Vec<f64> = Vec::new();
    let mut value_ind = 0;

    for i in 0..bin_upper_bound.len() {
        let mut cnt_in_bin = 0;
        let mut distinct_cnt_in_bin = 0;
        let bin_start = value_ind;

        while value_ind < num_distinct_values && distinct_values[value_ind] < bin_upper_bound[i] {
            cnt_in_bin += counts[value_ind];
            distinct_cnt_in_bin += 1;
            value_ind += 1;
        }

        let bins_remaining = max_bin - bin_upper_bound.len() as i32 - bounds_to_add.len() as i32;
        let mut num_sub_bins =
            ((cnt_in_bin as f64 * free_bins as f64) / total_sample_cnt as f64).round() as i32;
        num_sub_bins = std::cmp::min(num_sub_bins, bins_remaining) + 1;

        if i == bin_upper_bound.len() - 1 {
            num_sub_bins = bins_remaining + 1;
        }

        let new_upper_bounds = greedy_find_bin(
            &distinct_values[bin_start..bin_start + distinct_cnt_in_bin],
            &counts[bin_start..bin_start + distinct_cnt_in_bin],
            distinct_cnt_in_bin,
            num_sub_bins,
            cnt_in_bin as usize,
            min_data_in_bin,
        )?;

        // Add all bounds except the last one (infinity)
        bounds_to_add.extend(&new_upper_bounds[..new_upper_bounds.len() - 1]);
    }

    bin_upper_bound.extend(bounds_to_add);
    bin_upper_bound.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if bin_upper_bound.len() > max_bin as usize {
        return Err(LightGBMError::invalid_parameter(
            "bin_upper_bound.len()",
            bin_upper_bound.len().to_string(),
            "exceeds max_bin",
        )
        .into());
    }

    Ok(bin_upper_bound)
}

/// Get double upper bound for a value (compatibility with C++)
fn get_double_upper_bound(val: f64) -> f64 {
    // Simple implementation - in C++ this would handle precision issues
    val
}

/// Check if two double values are equal with tolerance
fn check_double_equal_ordered(a: &f64, b: &f64) -> bool {
    (a - b).abs() < EPSILON
}

/// Find argument of maximum value in array
fn arg_max(arr: &[i32]) -> usize {
    let mut max_idx = 0;
    let mut max_val = arr[0];
    for (i, &val) in arr.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    max_idx
}

/// Abstract Bin trait
pub trait Bin {
    ///
    fn get_col_wise_data(&self, bit_type: &mut u8, is_sparse: &mut bool) -> *const u8;
}

/// Bin factory functions
#[derive(Debug)]
pub struct BinFactory;

impl BinFactory {
    ///
    pub fn create_dense_bin(num_data: DataSize, num_bin: i32) -> Box<dyn Bin> {
        if num_bin <= 16 {
            Box::new(DenseBin::<u8, true>::new(num_data))
        } else if num_bin <= 256 {
            Box::new(DenseBin::<u8, false>::new(num_data))
        } else if num_bin <= 65536 {
            Box::new(DenseBin::<u16, false>::new(num_data))
        } else {
            Box::new(DenseBin::<u32, false>::new(num_data))
        }
    }

    /// Create a sparse bin with appropriate value type based on number of bins
    /// 
    /// Automatically selects u8, u16, or u32 storage based on the number of bins:
    /// - u8 for up to 256 bins
    /// - u16 for up to 65536 bins  
    /// - u32 for more bins
    pub fn create_sparse_bin(num_data: DataSize, num_bin: i32) -> Box<dyn Bin> {
        if num_bin <= 256 {
            Box::new(SparseBin::<u8>::new(num_data))
        } else if num_bin <= 65536 {
            Box::new(SparseBin::<u16>::new(num_data))
        } else {
            Box::new(SparseBin::<u32>::new(num_data))
        }
    }
}

/// Dense bin implementation for different bit types
#[derive(Debug)]
pub struct DenseBin<T, const USE_4BIT: bool> {
    data_: Vec<T>,
    num_data_: DataSize,
}

impl<T, const USE_4BIT: bool> DenseBin<T, USE_4BIT>
where
    T: Clone + Default + Copy,
{
    ///
    pub fn new(num_data: DataSize) -> Self {
        Self {
            data_: vec![T::default(); num_data as usize],
            num_data_: num_data,
        }
    }
    ///
    pub fn get_col_wise_data(&self, bit_type: &mut u8, is_sparse: &mut bool) -> *const T {
        *is_sparse = false;
        *bit_type = if USE_4BIT {
            4
        } else {
            std::mem::size_of::<T>() as u8 * 8
        };
        self.data_.as_ptr()
    }
}

impl<T, const USE_4BIT: bool> Bin for DenseBin<T, USE_4BIT>
where
    T: Clone + Default + Copy,
{
    fn get_col_wise_data(&self, bit_type: &mut u8, is_sparse: &mut bool) -> *const u8 {
        *is_sparse = false;
        *bit_type = if USE_4BIT {
            4
        } else {
            std::mem::size_of::<T>() as u8 * 8
        };
        self.data_.as_ptr() as *const u8
    }
}

/// Sparse bin implementation
#[derive(Debug)]
pub struct SparseBin<T> {
    indices_: Vec<DataSize>,
    vals_: Vec<T>,
    num_data_: DataSize,
}

impl<T> SparseBin<T>
where
    T: Clone + Default + Copy,
{
    ///
    pub fn new(num_data: DataSize) -> Self {
        Self {
            indices_: Vec::new(),
            vals_: Vec::new(),
            num_data_: num_data,
        }
    }
    ///
    pub fn get_col_wise_data(&self, bit_type: &mut u8, is_sparse: &mut bool) -> Option<&[T]> {
        *is_sparse = true;
        *bit_type = std::mem::size_of::<T>() as u8 * 8;
        None // Sparse bins don't return direct data pointer
    }
}

impl<T> Bin for SparseBin<T>
where
    T: Clone + Default + Copy,
{
    fn get_col_wise_data(&self, bit_type: &mut u8, is_sparse: &mut bool) -> *const u8 {
        *is_sparse = true;
        *bit_type = std::mem::size_of::<T>() as u8 * 8;
        std::ptr::null() // Sparse bins don't return direct data pointer
    }
}
/// Multi-value dense bin
#[derive(Debug)]
pub struct MultiValDenseBin<T> {
    data_: Vec<T>,
    num_data_: DataSize,
    num_feature_: i32,
    offsets_: Vec<u32>,
}

impl<T> MultiValDenseBin<T>
where
    T: Clone + Default + Copy,
{
    /// Create a new multi-value dense bin
    /// 
    /// # Arguments
    /// * `num_data` - Number of data points
    /// * `_num_bin` - Number of bins (unused in current implementation)
    /// * `num_feature` - Number of features
    /// * `offsets` - Feature offset array
    pub fn new(num_data: DataSize, _num_bin: i32, num_feature: i32, offsets: Vec<u32>) -> Self {
        let data_size = (num_data as usize) * (num_feature as usize);
        Self {
            data_: vec![T::default(); data_size],
            num_data_: num_data,
            num_feature_: num_feature,
            offsets_: offsets,
        }
    }
}

/// Multi-value sparse bin
#[derive(Debug)]
pub struct MultiValSparseBin<RowPtrType, BinType> {
    row_ptr_: Vec<RowPtrType>,
    data_: Vec<BinType>,
    num_data_: DataSize,
    num_bin_: i32,
}

impl<RowPtrType, BinType> MultiValSparseBin<RowPtrType, BinType>
where
    RowPtrType: Clone + Default + Copy,
    BinType: Clone + Default + Copy,
{
    /// Create a new multi-value sparse bin
    /// 
    /// # Arguments
    /// * `num_data` - Number of data points
    /// * `num_bin` - Number of bins
    /// * `_estimate_element_per_row` - Estimated elements per row (unused)
    pub fn new(num_data: DataSize, num_bin: i32, _estimate_element_per_row: f64) -> Self {
        Self {
            row_ptr_: vec![RowPtrType::default(); (num_data + 1) as usize],
            data_: Vec::new(),
            num_data_: num_data,
            num_bin_: num_bin,
        }
    }
}

/// Multi-value bin trait
pub trait MultiValBin {
    /// Get pointer to row-wise data representation
    fn get_row_wise_data(&self) -> *const u8;
}

/// Multi-value bin factory
#[derive(Debug)]
pub struct MultiValBinFactory;
///MultiValBinFactory
impl MultiValBinFactory {
    ///
    pub fn create_multi_val_bin(
        num_data: DataSize,
        num_bin: i32,
        num_feature: i32,
        sparse_rate: f64,
        offsets: &[u32],
    ) -> Box<dyn MultiValBin> {
        if sparse_rate >= MULTI_VAL_BIN_SPARSE_THRESHOLD {
            let average_element_per_row = (1.0 - sparse_rate) * num_feature as f64;
            Self::create_multi_val_sparse_bin(num_data, num_bin, average_element_per_row)
        } else {
            Self::create_multi_val_dense_bin(num_data, num_bin, num_feature, offsets)
        }
    }
    ///
    pub fn create_multi_val_dense_bin(
        num_data: DataSize,
        num_bin: i32,
        num_feature: i32,
        offsets: &[u32],
    ) -> Box<dyn MultiValBin> {
        // Calculate max bin of all features to select the int type
        let mut max_bin = 0;
        for i in 0..offsets.len() - 1 {
            let feature_bin = offsets[i + 1] - offsets[i];
            if feature_bin > max_bin {
                max_bin = feature_bin;
            }
        }

        if max_bin <= 256 {
            Box::new(MultiValDenseBin::<u8>::new(
                num_data,
                num_bin,
                num_feature,
                offsets.to_vec(),
            ))
        } else if max_bin <= 65536 {
            Box::new(MultiValDenseBin::<u16>::new(
                num_data,
                num_bin,
                num_feature,
                offsets.to_vec(),
            ))
        } else {
            Box::new(MultiValDenseBin::<u32>::new(
                num_data,
                num_bin,
                num_feature,
                offsets.to_vec(),
            ))
        }
    }
    ///
    pub fn create_multi_val_sparse_bin(
        num_data: DataSize,
        num_bin: i32,
        estimate_element_per_row: f64,
    ) -> Box<dyn MultiValBin> {
        let estimate_total_entries = (estimate_element_per_row * 1.1 * num_data as f64) as usize;

        if estimate_total_entries <= u16::MAX as usize {
            if num_bin <= 256 {
                Box::new(MultiValSparseBin::<u16, u8>::new(
                    num_data,
                    num_bin,
                    estimate_element_per_row,
                ))
            } else if num_bin <= 65536 {
                Box::new(MultiValSparseBin::<u16, u16>::new(
                    num_data,
                    num_bin,
                    estimate_element_per_row,
                ))
            } else {
                Box::new(MultiValSparseBin::<u16, u32>::new(
                    num_data,
                    num_bin,
                    estimate_element_per_row,
                ))
            }
        } else if estimate_total_entries <= u32::MAX as usize {
            if num_bin <= 256 {
                Box::new(MultiValSparseBin::<u32, u8>::new(
                    num_data,
                    num_bin,
                    estimate_element_per_row,
                ))
            } else if num_bin <= 65536 {
                Box::new(MultiValSparseBin::<u32, u16>::new(
                    num_data,
                    num_bin,
                    estimate_element_per_row,
                ))
            } else {
                Box::new(MultiValSparseBin::<u32, u32>::new(
                    num_data,
                    num_bin,
                    estimate_element_per_row,
                ))
            }
        } else {
            // Use usize for very large datasets
            if num_bin <= 256 {
                Box::new(MultiValSparseBin::<usize, u8>::new(
                    num_data,
                    num_bin,
                    estimate_element_per_row,
                ))
            } else if num_bin <= 65536 {
                Box::new(MultiValSparseBin::<usize, u16>::new(
                    num_data,
                    num_bin,
                    estimate_element_per_row,
                ))
            } else {
                Box::new(MultiValSparseBin::<usize, u32>::new(
                    num_data,
                    num_bin,
                    estimate_element_per_row,
                ))
            }
        }
    }
}

impl<T> MultiValBin for MultiValDenseBin<T>
where
    T: Clone + Default + Copy,
{
    fn get_row_wise_data(&self) -> *const u8 {
        self.data_.as_ptr() as *const u8
    }
}

impl<RowPtrType, BinType> MultiValBin for MultiValSparseBin<RowPtrType, BinType>
where
    RowPtrType: Clone + Default + Copy,
    BinType: Clone + Default + Copy,
{
    fn get_row_wise_data(&self) -> *const u8 {
        self.data_.as_ptr() as *const u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bin_mapper_creation() {
        let mapper = BinMapper::new();
        assert_eq!(mapper.num_bin(), 1);
        assert!(mapper.is_trivial());
        assert_eq!(mapper.bin_type(), BinType::NumericalBin);
    }

    #[test]
    fn test_bin_mapper_copy() {
        let mapper1 = BinMapper::new();
        let mapper2 = BinMapper::copy_from(&mapper1);
        assert_eq!(mapper1.num_bin(), mapper2.num_bin());
        assert_eq!(mapper1.is_trivial(), mapper2.is_trivial());
        assert_eq!(mapper1.bin_type(), mapper2.bin_type());
    }

    #[test]
    fn test_value_to_bin_numerical() {
        let mut mapper = BinMapper::new();
        mapper.bin_upper_bound_ = vec![0.5, 1.5, f64::INFINITY];
        mapper.num_bin_ = 3;
        mapper.bin_type_ = BinType::NumericalBin;

        assert_eq!(mapper.value_to_bin(0.3), 0);
        assert_eq!(mapper.value_to_bin(1.0), 1);
        assert_eq!(mapper.value_to_bin(2.0), 2);
    }

    #[test]
    fn test_value_to_bin_categorical() {
        let mut mapper = BinMapper::new();
        mapper.bin_type_ = BinType::CategoricalBin;
        mapper.categorical_2_bin_.insert(1, 0);
        mapper.categorical_2_bin_.insert(2, 1);
        mapper.categorical_2_bin_.insert(3, 2);

        assert_eq!(mapper.value_to_bin(1.0), 0);
        assert_eq!(mapper.value_to_bin(2.0), 1);
        assert_eq!(mapper.value_to_bin(3.0), 2);
        assert_eq!(mapper.value_to_bin(999.0), 0); // Unknown category maps to bin 0
    }

    #[test]
    fn test_need_filter() {
        let cnt_in_bin = vec![5, 10, 15, 20];
        assert!(!need_filter(&cnt_in_bin, 50, 5, BinType::NumericalBin));
        assert!(need_filter(&cnt_in_bin, 50, 30, BinType::NumericalBin));
    }

    #[test]
    fn test_greedy_find_bin() {
        let distinct_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let counts = vec![10, 20, 30, 20, 10];
        let result = greedy_find_bin(&distinct_values, &counts, 5, 3, 90, 5).unwrap();
        assert!(!result.is_empty());
        assert_eq!(*result.last().unwrap(), f64::INFINITY);
    }

    #[test]
    fn test_bin_type_serialization() {
        let bin_type = BinType::CategoricalBin;
        let serialized = serde_json::to_string(&bin_type).unwrap();
        let deserialized: BinType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(bin_type, deserialized);
    }

    #[test]
    fn test_dense_bin_creation() {
        let bin = DenseBin::<u8, false>::new(100);
        assert_eq!(bin.data_.len(), 100);
        assert_eq!(bin.num_data_, 100);
    }

    #[test]
    fn test_sparse_bin_creation() {
        let bin = SparseBin::<u16>::new(100);
        assert_eq!(bin.num_data_, 100);
        assert!(bin.indices_.is_empty());
        assert!(bin.vals_.is_empty());
    }
}
