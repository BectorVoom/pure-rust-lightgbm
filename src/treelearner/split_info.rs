//! Split information structures for tree learning.
//!
//! This module provides data structures for storing information about
//! split points found during tree construction, including split gains,
//! thresholds, gradients, and hessians.

use crate::core::types::DataSize;
use std::cmp::Ordering;

/// Minimum score value, equivalent to kMinScore in C++ LightGBM.
/// Used to initialize gain values to negative infinity.
pub const K_MIN_SCORE: f64 = f64::NEG_INFINITY;

/// Split information structure equivalent to C++ SplitInfo.
/// Used to store information for gain split points during tree learning.
#[derive(Debug, Clone)]
pub struct SplitInfo {
    /// Feature index (-1 if no valid split)
    pub feature: i32,
    /// Split threshold
    pub threshold: u32,
    /// Left number of data after split
    pub left_count: DataSize,
    /// Right number of data after split
    pub right_count: DataSize,
    /// Number of categorical thresholds
    pub num_cat_threshold: i32,
    /// Left output after split
    pub left_output: f64,
    /// Right output after split
    pub right_output: f64,
    /// Split gain
    pub gain: f64,
    /// Left sum gradient after split
    pub left_sum_gradient: f64,
    /// Left sum hessian after split
    pub left_sum_hessian: f64,
    /// Left sum discretized gradient and hessian after split
    pub left_sum_gradient_and_hessian: i64,
    /// Right sum gradient after split
    pub right_sum_gradient: f64,
    /// Right sum hessian after split
    pub right_sum_hessian: f64,
    /// Right sum discretized gradient and hessian after split
    pub right_sum_gradient_and_hessian: i64,
    /// Categorical threshold values
    pub cat_threshold: Vec<u32>,
    /// True if default split is left
    pub default_left: bool,
    /// Monotone constraint type
    pub monotone_type: i8,
}

impl SplitInfo {
    /// Create a new SplitInfo with default values
    pub fn new() -> Self {
        Self {
            feature: -1,
            threshold: 0,
            left_count: 0,
            right_count: 0,
            num_cat_threshold: 0,
            left_output: 0.0,
            right_output: 0.0,
            gain: K_MIN_SCORE,
            left_sum_gradient: 0.0,
            left_sum_hessian: 0.0,
            left_sum_gradient_and_hessian: 0,
            right_sum_gradient: 0.0,
            right_sum_hessian: 0.0,
            right_sum_gradient_and_hessian: 0,
            cat_threshold: Vec::new(),
            default_left: true,
            monotone_type: 0,
        }
    }

    /// Calculate the size needed for serialization buffer
    pub fn size(max_cat_threshold: i32) -> usize {
        2 * std::mem::size_of::<i32>()
            + std::mem::size_of::<u32>()
            + std::mem::size_of::<bool>()
            + 7 * std::mem::size_of::<f64>()
            + 2 * std::mem::size_of::<DataSize>()
            + (max_cat_threshold as usize) * std::mem::size_of::<u32>()
            + std::mem::size_of::<i8>()
            + 2 * std::mem::size_of::<i64>()
    }

    /// Copy data to a buffer (equivalent to C++ CopyTo)
    pub fn copy_to(&self, buffer: &mut [u8]) {
        let mut offset = 0;

        // Copy feature
        let feature_bytes = self.feature.to_le_bytes();
        buffer[offset..offset + feature_bytes.len()].copy_from_slice(&feature_bytes);
        offset += feature_bytes.len();

        // Copy left_count
        let left_count_bytes = self.left_count.to_le_bytes();
        buffer[offset..offset + left_count_bytes.len()].copy_from_slice(&left_count_bytes);
        offset += left_count_bytes.len();

        // Copy right_count
        let right_count_bytes = self.right_count.to_le_bytes();
        buffer[offset..offset + right_count_bytes.len()].copy_from_slice(&right_count_bytes);
        offset += right_count_bytes.len();

        // Copy gain
        let gain_bytes = self.gain.to_le_bytes();
        buffer[offset..offset + gain_bytes.len()].copy_from_slice(&gain_bytes);
        offset += gain_bytes.len();

        // Copy threshold
        let threshold_bytes = self.threshold.to_le_bytes();
        buffer[offset..offset + threshold_bytes.len()].copy_from_slice(&threshold_bytes);
        offset += threshold_bytes.len();

        // Copy left_output
        let left_output_bytes = self.left_output.to_le_bytes();
        buffer[offset..offset + left_output_bytes.len()].copy_from_slice(&left_output_bytes);
        offset += left_output_bytes.len();

        // Copy right_output
        let right_output_bytes = self.right_output.to_le_bytes();
        buffer[offset..offset + right_output_bytes.len()].copy_from_slice(&right_output_bytes);
        offset += right_output_bytes.len();

        // Copy left_sum_gradient
        let left_sum_gradient_bytes = self.left_sum_gradient.to_le_bytes();
        buffer[offset..offset + left_sum_gradient_bytes.len()].copy_from_slice(&left_sum_gradient_bytes);
        offset += left_sum_gradient_bytes.len();

        // Copy left_sum_hessian
        let left_sum_hessian_bytes = self.left_sum_hessian.to_le_bytes();
        buffer[offset..offset + left_sum_hessian_bytes.len()].copy_from_slice(&left_sum_hessian_bytes);
        offset += left_sum_hessian_bytes.len();

        // Copy left_sum_gradient_and_hessian
        let left_sum_gradient_and_hessian_bytes = self.left_sum_gradient_and_hessian.to_le_bytes();
        buffer[offset..offset + left_sum_gradient_and_hessian_bytes.len()].copy_from_slice(&left_sum_gradient_and_hessian_bytes);
        offset += left_sum_gradient_and_hessian_bytes.len();

        // Copy right_sum_gradient
        let right_sum_gradient_bytes = self.right_sum_gradient.to_le_bytes();
        buffer[offset..offset + right_sum_gradient_bytes.len()].copy_from_slice(&right_sum_gradient_bytes);
        offset += right_sum_gradient_bytes.len();

        // Copy right_sum_hessian
        let right_sum_hessian_bytes = self.right_sum_hessian.to_le_bytes();
        buffer[offset..offset + right_sum_hessian_bytes.len()].copy_from_slice(&right_sum_hessian_bytes);
        offset += right_sum_hessian_bytes.len();

        // Copy right_sum_gradient_and_hessian
        let right_sum_gradient_and_hessian_bytes = self.right_sum_gradient_and_hessian.to_le_bytes();
        buffer[offset..offset + right_sum_gradient_and_hessian_bytes.len()].copy_from_slice(&right_sum_gradient_and_hessian_bytes);
        offset += right_sum_gradient_and_hessian_bytes.len();

        // Copy default_left
        let default_left_bytes = [if self.default_left { 1u8 } else { 0u8 }];
        buffer[offset..offset + default_left_bytes.len()].copy_from_slice(&default_left_bytes);
        offset += default_left_bytes.len();

        // Copy monotone_type
        let monotone_type_bytes = self.monotone_type.to_le_bytes();
        buffer[offset..offset + monotone_type_bytes.len()].copy_from_slice(&monotone_type_bytes);
        offset += monotone_type_bytes.len();

        // Copy num_cat_threshold
        let num_cat_threshold_bytes = self.num_cat_threshold.to_le_bytes();
        buffer[offset..offset + num_cat_threshold_bytes.len()].copy_from_slice(&num_cat_threshold_bytes);
        offset += num_cat_threshold_bytes.len();

        // Copy cat_threshold data
        for &threshold in &self.cat_threshold {
            let threshold_bytes = threshold.to_le_bytes();
            buffer[offset..offset + threshold_bytes.len()].copy_from_slice(&threshold_bytes);
            offset += threshold_bytes.len();
        }
    }

    /// Copy data from a buffer (equivalent to C++ CopyFrom)
    pub fn copy_from(&mut self, buffer: &[u8]) {
        let mut offset = 0;

        // Copy feature
        let mut feature_bytes = [0u8; 4];
        feature_bytes.copy_from_slice(&buffer[offset..offset + 4]);
        self.feature = i32::from_le_bytes(feature_bytes);
        offset += 4;

        // Copy left_count
        let mut left_count_bytes = [0u8; 4];
        left_count_bytes.copy_from_slice(&buffer[offset..offset + 4]);
        self.left_count = DataSize::from_le_bytes(left_count_bytes);
        offset += 4;

        // Copy right_count
        let mut right_count_bytes = [0u8; 4];
        right_count_bytes.copy_from_slice(&buffer[offset..offset + 4]);
        self.right_count = DataSize::from_le_bytes(right_count_bytes);
        offset += 4;

        // Copy gain
        let mut gain_bytes = [0u8; 8];
        gain_bytes.copy_from_slice(&buffer[offset..offset + 8]);
        self.gain = f64::from_le_bytes(gain_bytes);
        offset += 8;

        // Copy threshold
        let mut threshold_bytes = [0u8; 4];
        threshold_bytes.copy_from_slice(&buffer[offset..offset + 4]);
        self.threshold = u32::from_le_bytes(threshold_bytes);
        offset += 4;

        // Copy left_output
        let mut left_output_bytes = [0u8; 8];
        left_output_bytes.copy_from_slice(&buffer[offset..offset + 8]);
        self.left_output = f64::from_le_bytes(left_output_bytes);
        offset += 8;

        // Copy right_output
        let mut right_output_bytes = [0u8; 8];
        right_output_bytes.copy_from_slice(&buffer[offset..offset + 8]);
        self.right_output = f64::from_le_bytes(right_output_bytes);
        offset += 8;

        // Copy left_sum_gradient
        let mut left_sum_gradient_bytes = [0u8; 8];
        left_sum_gradient_bytes.copy_from_slice(&buffer[offset..offset + 8]);
        self.left_sum_gradient = f64::from_le_bytes(left_sum_gradient_bytes);
        offset += 8;

        // Copy left_sum_hessian
        let mut left_sum_hessian_bytes = [0u8; 8];
        left_sum_hessian_bytes.copy_from_slice(&buffer[offset..offset + 8]);
        self.left_sum_hessian = f64::from_le_bytes(left_sum_hessian_bytes);
        offset += 8;

        // Copy left_sum_gradient_and_hessian
        let mut left_sum_gradient_and_hessian_bytes = [0u8; 8];
        left_sum_gradient_and_hessian_bytes.copy_from_slice(&buffer[offset..offset + 8]);
        self.left_sum_gradient_and_hessian = i64::from_le_bytes(left_sum_gradient_and_hessian_bytes);
        offset += 8;

        // Copy right_sum_gradient
        let mut right_sum_gradient_bytes = [0u8; 8];
        right_sum_gradient_bytes.copy_from_slice(&buffer[offset..offset + 8]);
        self.right_sum_gradient = f64::from_le_bytes(right_sum_gradient_bytes);
        offset += 8;

        // Copy right_sum_hessian
        let mut right_sum_hessian_bytes = [0u8; 8];
        right_sum_hessian_bytes.copy_from_slice(&buffer[offset..offset + 8]);
        self.right_sum_hessian = f64::from_le_bytes(right_sum_hessian_bytes);
        offset += 8;

        // Copy right_sum_gradient_and_hessian
        let mut right_sum_gradient_and_hessian_bytes = [0u8; 8];
        right_sum_gradient_and_hessian_bytes.copy_from_slice(&buffer[offset..offset + 8]);
        self.right_sum_gradient_and_hessian = i64::from_le_bytes(right_sum_gradient_and_hessian_bytes);
        offset += 8;

        // Copy default_left
        self.default_left = buffer[offset] != 0;
        offset += 1;

        // Copy monotone_type
        self.monotone_type = buffer[offset] as i8;
        offset += 1;

        // Copy num_cat_threshold
        let mut num_cat_threshold_bytes = [0u8; 4];
        num_cat_threshold_bytes.copy_from_slice(&buffer[offset..offset + 4]);
        self.num_cat_threshold = i32::from_le_bytes(num_cat_threshold_bytes);
        offset += 4;

        // Copy cat_threshold data
        self.cat_threshold.clear();
        self.cat_threshold.reserve(self.num_cat_threshold as usize);
        for _ in 0..self.num_cat_threshold {
            let mut threshold_bytes = [0u8; 4];
            threshold_bytes.copy_from_slice(&buffer[offset..offset + 4]);
            self.cat_threshold.push(u32::from_le_bytes(threshold_bytes));
            offset += 4;
        }
    }

    /// Reset split info to default values (equivalent to C++ Reset)
    pub fn reset(&mut self) {
        self.feature = -1;
        self.gain = K_MIN_SCORE;
    }

    /// Normalize gain value (replace NaN with K_MIN_SCORE)
    fn normalize_gain(gain: f64) -> f64 {
        if gain.is_nan() {
            K_MIN_SCORE
        } else {
            gain
        }
    }

    /// Normalize feature value (replace -1 with i32::MAX)
    fn normalize_feature(feature: i32) -> i32 {
        if feature == -1 {
            i32::MAX
        } else {
            feature
        }
    }
}

impl Default for SplitInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for SplitInfo {
    /// Test if a candidate SplitInfo is equivalent to this one (equivalent to C++ operator==)
    fn eq(&self, other: &Self) -> bool {
        let local_gain = Self::normalize_gain(self.gain);
        let other_gain = Self::normalize_gain(other.gain);

        if local_gain != other_gain {
            return false;
        }

        // If same gain, splits are only equal if they also use the same feature
        let local_feature = Self::normalize_feature(self.feature);
        let other_feature = Self::normalize_feature(other.feature);

        local_feature == other_feature
    }
}

impl PartialOrd for SplitInfo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SplitInfo {
    /// Compare SplitInfo instances (equivalent to C++ operator>)
    fn cmp(&self, other: &Self) -> Ordering {
        let local_gain = Self::normalize_gain(self.gain);
        let other_gain = Self::normalize_gain(other.gain);

        if local_gain != other_gain {
            return local_gain.partial_cmp(&other_gain).unwrap_or(Ordering::Equal);
        }

        // If gains are identical, choose the feature with the smaller index
        let local_feature = Self::normalize_feature(self.feature);
        let other_feature = Self::normalize_feature(other.feature);

        other_feature.cmp(&local_feature) // Note: reversed for smaller feature index preference
    }
}

impl Eq for SplitInfo {}

/// Lightweight split information structure equivalent to C++ LightSplitInfo.
/// Contains only essential split information for performance-critical operations.
#[derive(Debug, Clone)]
pub struct LightSplitInfo {
    /// Feature index (-1 if no valid split)
    pub feature: i32,
    /// Split gain
    pub gain: f64,
    /// Left number of data after split
    pub left_count: DataSize,
    /// Right number of data after split
    pub right_count: DataSize,
}

impl LightSplitInfo {
    /// Create a new LightSplitInfo with default values
    pub fn new() -> Self {
        Self {
            feature: -1,
            gain: K_MIN_SCORE,
            left_count: 0,
            right_count: 0,
        }
    }

    /// Reset split info to default values (equivalent to C++ Reset)
    pub fn reset(&mut self) {
        self.feature = -1;
        self.gain = K_MIN_SCORE;
    }

    /// Copy data from a SplitInfo (equivalent to C++ CopyFrom)
    pub fn copy_from_split_info(&mut self, other: &SplitInfo) {
        self.feature = other.feature;
        self.gain = other.gain;
        self.left_count = other.left_count;
        self.right_count = other.right_count;
    }

    /// Copy data from a buffer (equivalent to C++ CopyFrom)
    pub fn copy_from(&mut self, buffer: &[u8]) {
        let mut offset = 0;

        // Copy feature
        let mut feature_bytes = [0u8; 4];
        feature_bytes.copy_from_slice(&buffer[offset..offset + 4]);
        self.feature = i32::from_le_bytes(feature_bytes);
        offset += 4;

        // Copy left_count
        let mut left_count_bytes = [0u8; 4];
        left_count_bytes.copy_from_slice(&buffer[offset..offset + 4]);
        self.left_count = DataSize::from_le_bytes(left_count_bytes);
        offset += 4;

        // Copy right_count
        let mut right_count_bytes = [0u8; 4];
        right_count_bytes.copy_from_slice(&buffer[offset..offset + 4]);
        self.right_count = DataSize::from_le_bytes(right_count_bytes);
        offset += 4;

        // Copy gain
        let mut gain_bytes = [0u8; 8];
        gain_bytes.copy_from_slice(&buffer[offset..offset + 8]);
        self.gain = f64::from_le_bytes(gain_bytes);
    }

    /// Normalize gain value (replace NaN with K_MIN_SCORE)
    fn normalize_gain(gain: f64) -> f64 {
        if gain.is_nan() {
            K_MIN_SCORE
        } else {
            gain
        }
    }

    /// Normalize feature value (replace -1 with i32::MAX)
    fn normalize_feature(feature: i32) -> i32 {
        if feature == -1 {
            i32::MAX
        } else {
            feature
        }
    }
}

impl Default for LightSplitInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for LightSplitInfo {
    /// Test if a candidate LightSplitInfo is equivalent to this one (equivalent to C++ operator==)
    fn eq(&self, other: &Self) -> bool {
        let local_gain = Self::normalize_gain(self.gain);
        let other_gain = Self::normalize_gain(other.gain);

        if local_gain != other_gain {
            return false;
        }

        // If same gain, splits are only equal if they also use the same feature
        let local_feature = Self::normalize_feature(self.feature);
        let other_feature = Self::normalize_feature(other.feature);

        local_feature == other_feature
    }
}

impl PartialOrd for LightSplitInfo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LightSplitInfo {
    /// Compare LightSplitInfo instances (equivalent to C++ operator>)
    fn cmp(&self, other: &Self) -> Ordering {
        let local_gain = Self::normalize_gain(self.gain);
        let other_gain = Self::normalize_gain(other.gain);

        if local_gain != other_gain {
            return local_gain.partial_cmp(&other_gain).unwrap_or(Ordering::Equal);
        }

        // If gains are identical, choose the feature with the smaller index
        let local_feature = Self::normalize_feature(self.feature);
        let other_feature = Self::normalize_feature(other.feature);

        other_feature.cmp(&local_feature) // Note: reversed for smaller feature index preference
    }
}

impl Eq for LightSplitInfo {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_info_new() {
        let split_info = SplitInfo::new();
        assert_eq!(split_info.feature, -1);
        assert_eq!(split_info.gain, K_MIN_SCORE);
        assert_eq!(split_info.threshold, 0);
        assert_eq!(split_info.left_count, 0);
        assert_eq!(split_info.right_count, 0);
        assert_eq!(split_info.default_left, true);
    }

    #[test]
    fn test_split_info_reset() {
        let mut split_info = SplitInfo::new();
        split_info.feature = 5;
        split_info.gain = 1.5;
        split_info.reset();
        assert_eq!(split_info.feature, -1);
        assert_eq!(split_info.gain, K_MIN_SCORE);
    }

    #[test]
    fn test_light_split_info_new() {
        let light_split_info = LightSplitInfo::new();
        assert_eq!(light_split_info.feature, -1);
        assert_eq!(light_split_info.gain, K_MIN_SCORE);
        assert_eq!(light_split_info.left_count, 0);
        assert_eq!(light_split_info.right_count, 0);
    }

    #[test]
    fn test_light_split_info_copy_from_split_info() {
        let mut split_info = SplitInfo::new();
        split_info.feature = 3;
        split_info.gain = 2.5;
        split_info.left_count = 100;
        split_info.right_count = 200;

        let mut light_split_info = LightSplitInfo::new();
        light_split_info.copy_from_split_info(&split_info);

        assert_eq!(light_split_info.feature, 3);
        assert_eq!(light_split_info.gain, 2.5);
        assert_eq!(light_split_info.left_count, 100);
        assert_eq!(light_split_info.right_count, 200);
    }

    #[test]
    fn test_split_info_comparison() {
        let mut split1 = SplitInfo::new();
        split1.gain = 1.0;
        split1.feature = 2;

        let mut split2 = SplitInfo::new();
        split2.gain = 2.0;
        split2.feature = 1;

        assert!(split2 > split1); // Higher gain wins

        // Test equal gains
        split1.gain = 2.0;
        assert!(split1 > split2); // Lower feature index wins when gains are equal
    }

    #[test]
    fn test_light_split_info_comparison() {
        let mut split1 = LightSplitInfo::new();
        split1.gain = 1.0;
        split1.feature = 2;

        let mut split2 = LightSplitInfo::new();
        split2.gain = 2.0;
        split2.feature = 1;

        assert!(split2 > split1); // Higher gain wins

        // Test equal gains
        split1.gain = 2.0;
        assert!(split1 > split2); // Lower feature index wins when gains are equal
    }

    #[test]
    fn test_nan_handling() {
        let mut split1 = SplitInfo::new();
        split1.gain = f64::NAN;
        split1.feature = 1;

        let mut split2 = SplitInfo::new();
        split2.gain = 1.0;
        split2.feature = 2;

        assert!(split2 > split1); // Non-NaN gain beats NaN
        assert_eq!(split1, split1); // NaN should equal itself after normalization
    }

    #[test]
    fn test_serialization_size() {
        let size = SplitInfo::size(10);
        assert!(size > 0);
        // Should include space for all fields plus 10 categorical thresholds
        let expected_min_size = std::mem::size_of::<i32>() * 2 // feature, num_cat_threshold
            + std::mem::size_of::<u32>() // threshold
            + std::mem::size_of::<bool>() // default_left
            + std::mem::size_of::<f64>() * 7 // 7 double fields
            + std::mem::size_of::<DataSize>() * 2 // left_count, right_count
            + std::mem::size_of::<u32>() * 10 // cat_threshold
            + std::mem::size_of::<i8>() // monotone_type
            + std::mem::size_of::<i64>() * 2; // 2 i64 fields
        assert!(size >= expected_min_size);
    }

    #[test]
    fn test_copy_to_from_buffer() {
        let mut original = SplitInfo::new();
        original.feature = 5;
        original.gain = 3.14;
        original.threshold = 42;
        original.left_count = 100;
        original.right_count = 200;
        original.left_output = 1.5;
        original.right_output = 2.5;
        original.default_left = false;
        original.monotone_type = 1;
        original.num_cat_threshold = 2;
        original.cat_threshold = vec![10, 20];

        let buffer_size = SplitInfo::size(2);
        let mut buffer = vec![0u8; buffer_size];
        original.copy_to(&mut buffer);

        let mut copy = SplitInfo::new();
        copy.copy_from(&buffer);

        assert_eq!(copy.feature, original.feature);
        assert_eq!(copy.gain, original.gain);
        assert_eq!(copy.threshold, original.threshold);
        assert_eq!(copy.left_count, original.left_count);
        assert_eq!(copy.right_count, original.right_count);
        assert_eq!(copy.left_output, original.left_output);
        assert_eq!(copy.right_output, original.right_output);
        assert_eq!(copy.default_left, original.default_left);
        assert_eq!(copy.monotone_type, original.monotone_type);
        assert_eq!(copy.num_cat_threshold, original.num_cat_threshold);
        assert_eq!(copy.cat_threshold, original.cat_threshold);
    }
}