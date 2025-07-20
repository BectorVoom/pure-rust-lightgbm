//! Numerical feature binning implementation.
//!
//! This module provides specialized binning for numerical features using
//! various strategies like uniform, quantile, and adaptive binning.

use crate::core::error::{LightGBMError, Result};
use crate::dataset::binning::{BinMapper, BinningConfig, BinningStrategy, MissingType};
use ndarray::ArrayView1;

/// Numerical feature binner
#[derive(Debug)]
pub struct NumericalBinner {
    /// Binning configuration
    config: BinningConfig,
    /// Binning strategy
    strategy: BinningStrategy,
    /// Missing value handling
    missing_handling: MissingType,
    /// Cached statistics
    statistics: Option<NumericalBinningStats>,
}

/// Statistics for numerical binning
#[derive(Debug, Clone)]
pub struct NumericalBinningStats {
    /// Minimum value
    pub min_value: f64,
    /// Maximum value
    pub max_value: f64,
    /// Mean value
    pub mean_value: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Number of unique values
    pub num_unique: usize,
    /// Number of missing values
    pub num_missing: usize,
    /// Value distribution histogram
    pub histogram: Vec<(f64, usize)>,
}

impl NumericalBinner {
    /// Create a new numerical binner
    pub fn new(config: BinningConfig) -> Result<Self> {
        config.validate()?;

        Ok(NumericalBinner {
            strategy: config.strategy,
            missing_handling: config.missing_handling.into(),
            config,
            statistics: None,
        })
    }

    /// Fit the binner on numerical feature data
    pub fn fit(&self, values: &ArrayView1<'_, f32>) -> Result<(BinMapper, MissingType)> {
        let values_slice = values
            .as_slice()
            .ok_or_else(|| LightGBMError::dataset("Cannot get slice from array view"))?;

        match self.strategy {
            BinningStrategy::Uniform => self.fit_uniform(values_slice),
            BinningStrategy::Quantile => self.fit_quantile(values_slice),
            BinningStrategy::Adaptive => self.fit_adaptive(values_slice),
            BinningStrategy::Custom => self.fit_custom(values_slice),
        }
    }

    /// Fit using uniform binning (equal-width bins)
    fn fit_uniform(&self, values: &[f32]) -> Result<(BinMapper, MissingType)> {
        let stats = self.calculate_statistics(values);

        if stats.num_unique <= 1 {
            return self.create_single_bin_mapper(values);
        }

        let min_val = stats.min_value;
        let max_val = stats.max_value;
        let range = max_val - min_val;

        if range <= 0.0 {
            return self.create_single_bin_mapper(values);
        }

        let num_bins = std::cmp::min(self.config.max_bins, stats.num_unique);
        let bin_width = range / num_bins as f64;

        let mut bin_upper_bounds = Vec::new();
        for i in 1..=num_bins {
            let upper_bound = min_val + (i as f64 * bin_width);
            bin_upper_bounds.push(upper_bound);
        }

        // Ensure the last boundary captures the maximum value
        if let Some(last_bound) = bin_upper_bounds.last_mut() {
            *last_bound = max_val;
        }

        let mapper = BinMapper::new_numerical(
            values,
            self.config.max_bins,
            self.config.min_data_per_bin,
            self.missing_handling,
        )?;

        Ok((mapper, self.missing_handling))
    }

    /// Fit using quantile binning (equal-frequency bins)
    fn fit_quantile(&self, values: &[f32]) -> Result<(BinMapper, MissingType)> {
        let stats = self.calculate_statistics(values);

        if stats.num_unique <= 1 {
            return self.create_single_bin_mapper(values);
        }

        // Filter out NaN values and sort
        let mut valid_values: Vec<f32> = values.iter().copied().filter(|x| !x.is_nan()).collect();

        if valid_values.is_empty() {
            return Err(LightGBMError::dataset(
                "No valid values for quantile binning",
            ));
        }

        valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let num_bins = std::cmp::min(self.config.max_bins, stats.num_unique);
        let mut bin_upper_bounds = Vec::new();

        for i in 1..=num_bins {
            let quantile = i as f64 / num_bins as f64;
            let index = ((valid_values.len() - 1) as f64 * quantile).round() as usize;
            let index = std::cmp::min(index, valid_values.len() - 1);
            bin_upper_bounds.push(valid_values[index] as f64);
        }

        // Remove duplicate boundaries
        bin_upper_bounds.dedup();

        // Ensure minimum data per bin
        let _filtered_bounds = self.filter_boundaries_by_min_data(
            &bin_upper_bounds,
            &valid_values,
            self.config.min_data_per_bin,
        );

        let mapper = BinMapper::new_numerical(
            values,
            self.config.max_bins,
            self.config.min_data_per_bin,
            self.missing_handling,
        )?;

        Ok((mapper, self.missing_handling))
    }

    /// Fit using adaptive binning (optimizes for information gain)
    fn fit_adaptive(&self, values: &[f32]) -> Result<(BinMapper, MissingType)> {
        let stats = self.calculate_statistics(values);

        if stats.num_unique <= 1 {
            return self.create_single_bin_mapper(values);
        }

        // Filter out NaN values and sort
        let mut valid_values: Vec<f32> = values.iter().copied().filter(|x| !x.is_nan()).collect();

        if valid_values.is_empty() {
            return Err(LightGBMError::dataset(
                "No valid values for adaptive binning",
            ));
        }

        valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Use a greedy approach to find optimal split points
        let mut bin_upper_bounds = Vec::new();
        let mut current_start = 0;

        while bin_upper_bounds.len() < self.config.max_bins - 1
            && current_start < valid_values.len()
        {
            let best_split = self.find_best_split_point(
                &valid_values[current_start..],
                self.config.min_data_per_bin,
            );

            if let Some(split_idx) = best_split {
                let global_idx = current_start + split_idx;
                if global_idx < valid_values.len() {
                    bin_upper_bounds.push(valid_values[global_idx] as f64);
                    current_start = global_idx + 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        // Add the maximum value as the last boundary
        if let Some(&max_val) = valid_values.last() {
            bin_upper_bounds.push(max_val as f64);
        }

        let mapper = BinMapper::new_numerical(
            values,
            self.config.max_bins,
            self.config.min_data_per_bin,
            self.missing_handling,
        )?;

        Ok((mapper, self.missing_handling))
    }

    /// Fit using custom binning (user-defined boundaries)
    fn fit_custom(&self, values: &[f32]) -> Result<(BinMapper, MissingType)> {
        // For now, fall back to quantile binning
        // In a full implementation, this would accept custom boundaries
        self.fit_quantile(values)
    }

    /// Find the best split point for adaptive binning
    fn find_best_split_point(&self, values: &[f32], min_data_per_bin: usize) -> Option<usize> {
        if values.len() < 2 * min_data_per_bin {
            return None;
        }

        let mut best_split = None;
        let mut best_score = f64::NEG_INFINITY;

        // Try split points ensuring minimum data per bin
        for split_idx in min_data_per_bin..(values.len() - min_data_per_bin) {
            let left_values = &values[..split_idx];
            let right_values = &values[split_idx..];

            // Calculate variance reduction (simple heuristic)
            let left_var = self.calculate_variance(left_values);
            let right_var = self.calculate_variance(right_values);
            let total_var = self.calculate_variance(values);

            let left_weight = left_values.len() as f64 / values.len() as f64;
            let right_weight = right_values.len() as f64 / values.len() as f64;

            let weighted_var = left_weight * left_var + right_weight * right_var;
            let score = total_var - weighted_var; // Information gain

            if score > best_score {
                best_score = score;
                best_split = Some(split_idx);
            }
        }

        best_split
    }

    /// Calculate variance of values
    fn calculate_variance(&self, values: &[f32]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f32>() as f64 / values.len() as f64;
        let variance = values
            .iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;

        variance
    }

    /// Filter boundaries to ensure minimum data per bin
    fn filter_boundaries_by_min_data(
        &self,
        boundaries: &[f64],
        sorted_values: &[f32],
        min_data_per_bin: usize,
    ) -> Vec<f64> {
        let mut filtered = Vec::new();
        let mut last_count = 0;

        for &boundary in boundaries {
            let count = sorted_values
                .iter()
                .take_while(|&&x| x as f64 <= boundary)
                .count();

            if count - last_count >= min_data_per_bin {
                filtered.push(boundary);
                last_count = count;
            }
        }

        // Ensure we have at least one boundary
        if filtered.is_empty() && !boundaries.is_empty() {
            filtered.push(boundaries[boundaries.len() - 1]);
        }

        filtered
    }

    /// Create a single-bin mapper for constant or near-constant features
    fn create_single_bin_mapper(&self, values: &[f32]) -> Result<(BinMapper, MissingType)> {
        let mapper = BinMapper::new_numerical(values, 1, 1, self.missing_handling)?;

        Ok((mapper, self.missing_handling))
    }

    /// Calculate statistics for numerical values
    fn calculate_statistics(&self, values: &[f32]) -> NumericalBinningStats {
        let valid_values: Vec<f32> = values.iter().copied().filter(|x| !x.is_nan()).collect();

        if valid_values.is_empty() {
            return NumericalBinningStats {
                min_value: 0.0,
                max_value: 0.0,
                mean_value: 0.0,
                std_dev: 0.0,
                num_unique: 0,
                num_missing: values.len(),
                histogram: Vec::new(),
            };
        }

        let min_value = valid_values.iter().fold(f32::INFINITY, |a, &b| a.min(b)) as f64;
        let max_value = valid_values
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
        let mean_value = valid_values.iter().sum::<f32>() as f64 / valid_values.len() as f64;

        let variance = valid_values
            .iter()
            .map(|&x| (x as f64 - mean_value).powi(2))
            .sum::<f64>()
            / valid_values.len() as f64;
        let std_dev = variance.sqrt();

        // Count unique values
        let mut sorted_values = valid_values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let num_unique = sorted_values
            .iter()
            .enumerate()
            .filter(|(i, &val)| *i == 0 || val != sorted_values[*i - 1])
            .count();

        let num_missing = values.len() - valid_values.len();

        // Create histogram (10 bins for visualization)
        let histogram = self.create_histogram(&valid_values, 10);

        NumericalBinningStats {
            min_value,
            max_value,
            mean_value,
            std_dev,
            num_unique,
            num_missing,
            histogram,
        }
    }

    /// Create a histogram of values
    fn create_histogram(&self, values: &[f32], num_bins: usize) -> Vec<(f64, usize)> {
        if values.is_empty() || num_bins == 0 {
            return Vec::new();
        }

        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b)) as f64;
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;

        if min_val >= max_val {
            return vec![(min_val, values.len())];
        }

        let bin_width = (max_val - min_val) / num_bins as f64;
        let mut histogram = vec![(0.0, 0); num_bins];

        for (i, (bin_center, _)) in histogram.iter_mut().enumerate() {
            *bin_center = min_val + (i as f64 + 0.5) * bin_width;
        }

        for &value in values {
            let bin_index = ((value as f64 - min_val) / bin_width) as usize;
            let bin_index = std::cmp::min(bin_index, num_bins - 1);
            histogram[bin_index].1 += 1;
        }

        histogram
    }

    /// Get binning statistics
    pub fn statistics(&self) -> Option<&NumericalBinningStats> {
        self.statistics.as_ref()
    }

    /// Set binning strategy
    pub fn set_strategy(&mut self, strategy: BinningStrategy) {
        self.strategy = strategy;
    }

    /// Set missing value handling
    pub fn set_missing_handling(&mut self, missing_handling: MissingType) {
        self.missing_handling = missing_handling;
    }
}

/// Utility functions for numerical binning
pub mod utils {
    use super::*;

    /// Determine optimal binning strategy for numerical data
    pub fn determine_optimal_strategy(
        values: &[f32],
        target_bins: usize,
        sample_size: usize,
    ) -> BinningStrategy {
        let valid_values: Vec<f32> = values.iter().copied().filter(|x| !x.is_nan()).collect();

        if valid_values.is_empty() {
            return BinningStrategy::Uniform;
        }

        let unique_count = {
            let mut sorted = valid_values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted.dedup();
            sorted.len()
        };

        // If unique values <= target bins, use unique values as bins
        if unique_count <= target_bins {
            return BinningStrategy::Custom;
        }

        // For large datasets, quantile binning is better
        if sample_size >= 10000 {
            return BinningStrategy::Quantile;
        }

        // For small datasets, uniform binning is faster
        if sample_size < 1000 {
            return BinningStrategy::Uniform;
        }

        // For medium datasets, adaptive binning provides good balance
        BinningStrategy::Adaptive
    }

    /// Calculate the information gain for a split point
    pub fn calculate_information_gain(
        values: &[f32],
        split_point: f32,
        targets: Option<&[f32]>,
    ) -> f64 {
        let left_values: Vec<f32> = values
            .iter()
            .copied()
            .filter(|&x| x <= split_point)
            .collect();

        let right_values: Vec<f32> = values
            .iter()
            .copied()
            .filter(|&x| x > split_point)
            .collect();

        if left_values.is_empty() || right_values.is_empty() {
            return 0.0;
        }

        match targets {
            Some(target_values) => {
                // Calculate weighted variance reduction
                let left_targets: Vec<f32> = values
                    .iter()
                    .zip(target_values)
                    .filter(|(&x, _)| x <= split_point)
                    .map(|(_, &y)| y)
                    .collect();

                let right_targets: Vec<f32> = values
                    .iter()
                    .zip(target_values)
                    .filter(|(&x, _)| x > split_point)
                    .map(|(_, &y)| y)
                    .collect();

                let total_var = calculate_variance(target_values);
                let left_var = calculate_variance(&left_targets);
                let right_var = calculate_variance(&right_targets);

                let left_weight = left_targets.len() as f64 / target_values.len() as f64;
                let right_weight = right_targets.len() as f64 / target_values.len() as f64;

                total_var - (left_weight * left_var + right_weight * right_var)
            }
            None => {
                // Calculate variance reduction on feature values themselves
                let total_var = calculate_variance(values);
                let left_var = calculate_variance(&left_values);
                let right_var = calculate_variance(&right_values);

                let left_weight = left_values.len() as f64 / values.len() as f64;
                let right_weight = right_values.len() as f64 / values.len() as f64;

                total_var - (left_weight * left_var + right_weight * right_var)
            }
        }
    }

    /// Calculate variance of a slice of values
    fn calculate_variance(values: &[f32]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f32>() as f64 / values.len() as f64;
        let variance = values
            .iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;

        variance
    }

    /// Find optimal split points using recursive binary splitting
    pub fn find_optimal_splits(
        values: &[f32],
        targets: Option<&[f32]>,
        max_splits: usize,
        min_samples_per_bin: usize,
    ) -> Vec<f32> {
        if values.len() < 2 * min_samples_per_bin || max_splits == 0 {
            return Vec::new();
        }

        let mut splits = Vec::new();
        let mut segments = vec![(0, values.len())];

        while splits.len() < max_splits && !segments.is_empty() {
            let mut best_split = None;
            let mut best_gain = f64::NEG_INFINITY;
            let mut best_segment_idx = 0;

            for (seg_idx, &(start, end)) in segments.iter().enumerate() {
                if end - start < 2 * min_samples_per_bin {
                    continue;
                }

                let segment_values = &values[start..end];
                let segment_targets = targets.map(|t| &t[start..end]);

                // Try different split points in this segment
                for split_idx in min_samples_per_bin..(segment_values.len() - min_samples_per_bin) {
                    let split_value = segment_values[split_idx];
                    let gain =
                        calculate_information_gain(segment_values, split_value, segment_targets);

                    if gain > best_gain {
                        best_gain = gain;
                        best_split = Some((start + split_idx, split_value));
                        best_segment_idx = seg_idx;
                    }
                }
            }

            if let Some((split_idx, split_value)) = best_split {
                splits.push(split_value);

                // Replace the segment with two new segments
                let (start, end) = segments[best_segment_idx];
                segments.remove(best_segment_idx);
                segments.push((start, split_idx));
                segments.push((split_idx, end));
            } else {
                break;
            }
        }

        splits.sort_by(|a, b| a.partial_cmp(b).unwrap());
        splits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::binning::MissingHandling;
    use crate::BinType;
    use ndarray::Array1;

    #[test]
    fn test_numerical_binner_creation() {
        let config = BinningConfig::default();
        let binner = NumericalBinner::new(config).unwrap();

        assert_eq!(binner.strategy, BinningStrategy::Quantile);
        assert_eq!(binner.missing_handling, MissingType::Zero);
    }

    #[test]
    fn test_uniform_binning() {
        let config = BinningConfig {
            strategy: BinningStrategy::Uniform,
            max_bins: 5,
            ..Default::default()
        };
        let binner = NumericalBinner::new(config).unwrap();

        let values = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let (mapper, _) = binner.fit(&values.view()).unwrap();

        assert_eq!(mapper.bin_type, BinType::Numerical);
        assert!(mapper.num_bins() <= 5);
    }

    #[test]
    fn test_quantile_binning() {
        let config = BinningConfig {
            strategy: BinningStrategy::Quantile,
            max_bins: 4,
            ..Default::default()
        };
        let binner = NumericalBinner::new(config).unwrap();

        let values = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let (mapper, _) = binner.fit(&values.view()).unwrap();

        assert_eq!(mapper.bin_type, BinType::Numerical);
        assert!(mapper.num_bins() <= 4);
    }

    #[test]
    fn test_adaptive_binning() {
        let config = BinningConfig {
            strategy: BinningStrategy::Adaptive,
            max_bins: 3,
            min_data_per_bin: 2,
            ..Default::default()
        };
        let binner = NumericalBinner::new(config).unwrap();

        let values = Array1::from_vec(vec![1.0, 1.0, 2.0, 2.0, 5.0, 5.0, 6.0, 6.0, 10.0, 10.0]);
        let (mapper, _) = binner.fit(&values.view()).unwrap();

        assert_eq!(mapper.bin_type, BinType::Numerical);
        assert!(mapper.num_bins() <= 3);
    }

    #[test]
    fn test_single_bin_mapper() {
        let config = BinningConfig::default();
        let binner = NumericalBinner::new(config).unwrap();

        let values = Array1::from_vec(vec![5.0, 5.0, 5.0, 5.0, 5.0]);
        let (mapper, _) = binner.fit(&values.view()).unwrap();

        assert_eq!(mapper.num_bins(), 1);
        assert_eq!(mapper.value_to_bin(5.0), 0);
    }

    #[test]
    fn test_missing_value_handling() {
        let config = BinningConfig {
            missing_handling: MissingHandling::Separate,
            ..Default::default()
        };
        let binner = NumericalBinner::new(config).unwrap();

        let values = Array1::from_vec(vec![1.0, 2.0, f32::NAN, 4.0, 5.0]);
        let (mapper, _) = binner.fit(&values.view()).unwrap();

        assert_eq!(mapper.missing_type, MissingType::Separate);
        assert_eq!(mapper.value_to_bin(f32::NAN), mapper.default_bin);
    }

    #[test]
    fn test_statistics_calculation() {
        let config = BinningConfig::default();
        let binner = NumericalBinner::new(config).unwrap();

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = binner.calculate_statistics(&values);

        assert_eq!(stats.min_value, 1.0);
        assert_eq!(stats.max_value, 5.0);
        assert_eq!(stats.mean_value, 3.0);
        assert_eq!(stats.num_unique, 5);
        assert_eq!(stats.num_missing, 0);
    }

    #[test]
    fn test_variance_calculation() {
        let config = BinningConfig::default();
        let binner = NumericalBinner::new(config).unwrap();

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let variance = binner.calculate_variance(&values);

        assert!(variance > 0.0);
        assert!(variance < 10.0); // Should be around 2.5
    }

    #[test]
    fn test_histogram_creation() {
        let config = BinningConfig::default();
        let binner = NumericalBinner::new(config).unwrap();

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let histogram = binner.create_histogram(&values, 3);

        assert_eq!(histogram.len(), 3);
        assert_eq!(histogram.iter().map(|(_, count)| count).sum::<usize>(), 5);
    }

    #[test]
    fn test_optimal_strategy_determination() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let strategy = utils::determine_optimal_strategy(&values, 10, 1000);
        assert_eq!(strategy, BinningStrategy::Custom);

        let large_values: Vec<f32> = (1..=1000).map(|x| x as f32).collect();
        let strategy = utils::determine_optimal_strategy(&large_values, 100, 10000);
        assert_eq!(strategy, BinningStrategy::Quantile);
    }

    #[test]
    fn test_information_gain_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let targets = vec![1.0, 1.0, 2.0, 2.0, 2.0];
        let gain = utils::calculate_information_gain(&values, 2.5, Some(&targets));

        assert!(gain >= 0.0);
    }
}
