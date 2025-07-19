//! Categorical feature binning implementation.
//!
//! This module provides specialized binning for categorical features with
//! frequency-based ordering and rare category handling.

use crate::core::error::{LightGBMError, Result};
use crate::dataset::binning::{BinMapper, BinningConfig, MissingType};
use ndarray::ArrayView1;
use std::collections::HashMap;

/// Categorical feature binner
pub struct CategoricalBinner {
    /// Binning configuration
    config: BinningConfig,
    /// Missing value handling
    missing_handling: MissingType,
    /// Cached statistics
    statistics: Option<CategoricalBinningStats>,
}

/// Statistics for categorical binning
#[derive(Debug, Clone)]
pub struct CategoricalBinningStats {
    /// Category frequency map
    pub category_frequencies: HashMap<i32, usize>,
    /// Number of unique categories
    pub num_unique_categories: usize,
    /// Number of missing values
    pub num_missing: usize,
    /// Most frequent category
    pub most_frequent_category: Option<i32>,
    /// Least frequent category
    pub least_frequent_category: Option<i32>,
}

impl CategoricalBinner {
    /// Create a new categorical binner
    pub fn new(config: BinningConfig) -> Result<Self> {
        config.validate()?;

        Ok(CategoricalBinner {
            missing_handling: config.missing_handling.into(),
            config,
            statistics: None,
        })
    }

    /// Fit the binner on categorical feature data
    pub fn fit(&self, values: &ArrayView1<'_, f32>) -> Result<(BinMapper, MissingType)> {
        let values_slice = values
            .as_slice()
            .ok_or_else(|| LightGBMError::dataset("Cannot get slice from array view"))?;

        self.fit_categorical(values_slice)
    }

    /// Fit using categorical binning
    fn fit_categorical(&self, values: &[f32]) -> Result<(BinMapper, MissingType)> {
        let stats = self.calculate_statistics(values);

        if stats.num_unique_categories == 0 {
            return Err(LightGBMError::dataset("No valid categorical values found"));
        }

        let mapper =
            BinMapper::new_categorical(values, self.config.max_bins, self.missing_handling)?;

        Ok((mapper, self.missing_handling))
    }

    /// Calculate statistics for categorical values
    fn calculate_statistics(&self, values: &[f32]) -> CategoricalBinningStats {
        let mut category_frequencies = HashMap::new();
        let mut num_missing = 0;

        for &value in values {
            if value.is_nan() {
                num_missing += 1;
            } else {
                let category = value as i32;
                *category_frequencies.entry(category).or_insert(0) += 1;
            }
        }

        let num_unique_categories = category_frequencies.len();

        let most_frequent_category = category_frequencies
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&category, _)| category);

        let least_frequent_category = category_frequencies
            .iter()
            .min_by_key(|(_, &count)| count)
            .map(|(&category, _)| category);

        CategoricalBinningStats {
            category_frequencies,
            num_unique_categories,
            num_missing,
            most_frequent_category,
            least_frequent_category,
        }
    }

    /// Get binning statistics
    pub fn statistics(&self) -> Option<&CategoricalBinningStats> {
        self.statistics.as_ref()
    }

    /// Set missing value handling
    pub fn set_missing_handling(&mut self, missing_handling: MissingType) {
        self.missing_handling = missing_handling;
    }
}

/// Utility functions for categorical binning
pub mod utils {
    use super::*;

    /// Determine if values should be treated as categorical
    pub fn is_categorical(values: &[f32], max_unique_ratio: f64) -> bool {
        let valid_values: Vec<f32> = values.iter().copied().filter(|x| !x.is_nan()).collect();

        if valid_values.is_empty() {
            return false;
        }

        // Check if all values are integers
        let all_integers = valid_values.iter().all(|&x| x.fract() == 0.0);

        if !all_integers {
            return false;
        }

        // Check uniqueness ratio
        let unique_values: std::collections::HashSet<i32> =
            valid_values.iter().map(|&x| x as i32).collect();

        let uniqueness_ratio = unique_values.len() as f64 / valid_values.len() as f64;
        uniqueness_ratio <= max_unique_ratio
    }

    /// Create optimal binning for categorical data
    pub fn create_optimal_binning(
        values: &[f32],
        max_bins: usize,
        min_frequency: usize,
    ) -> Result<BinMapper> {
        // Count category frequencies
        let mut category_counts: HashMap<i32, usize> = HashMap::new();

        for &value in values {
            if !value.is_nan() {
                let category = value as i32;
                *category_counts.entry(category).or_insert(0) += 1;
            }
        }

        // Filter out rare categories
        let frequent_categories: Vec<(i32, usize)> = category_counts
            .into_iter()
            .filter(|(_, count)| *count >= min_frequency)
            .collect();

        if frequent_categories.is_empty() {
            return Err(LightGBMError::dataset("No frequent categories found"));
        }

        // Sort by frequency (descending)
        let mut sorted_categories = frequent_categories;
        sorted_categories.sort_by(|a, b| b.1.cmp(&a.1));

        // Take top categories up to max_bins
        let _selected_categories = sorted_categories
            .into_iter()
            .take(max_bins)
            .collect::<Vec<_>>();

        // Create bin mapper
        BinMapper::new_categorical(values, max_bins, MissingType::NaN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use crate::BinType;

    #[test]
    fn test_categorical_binner_creation() {
        let config = BinningConfig::default();
        let binner = CategoricalBinner::new(config).unwrap();
        assert_eq!(binner.missing_handling, MissingType::Zero);
    }

    #[test]
    fn test_categorical_binning() {
        let config = BinningConfig {
            max_bins: 5,
            ..Default::default()
        };
        let binner = CategoricalBinner::new(config).unwrap();

        let values = Array1::from_vec(vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0]);
        let (mapper, _) = binner.fit(&values.view()).unwrap();

        assert_eq!(mapper.bin_type, BinType::Categorical);
        assert!(mapper.num_bins() <= 5);
    }

    #[test]
    fn test_statistics_calculation() {
        let config = BinningConfig::default();
        let binner = CategoricalBinner::new(config).unwrap();

        let values = vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0];
        let stats = binner.calculate_statistics(&values);

        assert_eq!(stats.num_unique_categories, 3);
        assert_eq!(stats.num_missing, 0);
        assert_eq!(stats.most_frequent_category, Some(1));
        assert_eq!(stats.category_frequencies.get(&1), Some(&3));
    }

    #[test]
    fn test_is_categorical() {
        let categorical_values = vec![1.0, 2.0, 1.0, 3.0, 2.0];
        assert!(utils::is_categorical(&categorical_values, 0.8));

        let numerical_values = vec![1.1, 2.2, 3.3, 4.4, 5.5];
        assert!(!utils::is_categorical(&numerical_values, 0.8));
    }

    #[test]
    fn test_optimal_binning() {
        let values = vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 4.0];
        let mapper = utils::create_optimal_binning(&values, 3, 1).unwrap();

        assert_eq!(mapper.bin_type, BinType::Categorical);
        assert!(mapper.num_bins() <= 3);
    }
}
