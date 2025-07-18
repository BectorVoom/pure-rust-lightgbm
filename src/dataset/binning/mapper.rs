//! Bin mapper implementation for feature discretization.
//!
//! This module provides the core BinMapper structure that handles the mapping
//! of continuous feature values to discrete bins for histogram construction.

use crate::core::types::*;
use crate::core::error::{Result, LightGBMError};
use crate::core::constants::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Bin mapper for feature discretization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinMapper {
    /// Bin boundaries for numerical features
    pub bin_upper_bounds: Vec<f64>,
    /// Category to bin mapping for categorical features
    pub category_to_bin: HashMap<i32, BinIndex>,
    /// Bin to category mapping for categorical features
    pub bin_to_category: HashMap<BinIndex, i32>,
    /// Maximum number of bins
    pub max_bins: usize,
    /// Actual number of bins used
    pub num_bins: usize,
    /// Feature type (numerical or categorical)
    pub bin_type: BinType,
    /// Missing value handling strategy
    pub missing_type: MissingType,
    /// Default bin for missing values
    pub default_bin: BinIndex,
    /// Minimum value in the feature
    pub min_value: f64,
    /// Maximum value in the feature
    pub max_value: f64,
    /// Number of unique values
    pub num_unique_values: usize,
    /// Memory usage in bytes
    pub memory_usage: usize,
}

/// Bin type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinType {
    /// Numerical feature binning
    Numerical,
    /// Categorical feature binning
    Categorical,
}

/// Missing value handling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissingType {
    /// No missing values
    None,
    /// Missing values represented as zero
    Zero,
    /// Missing values represented as NaN
    NaN,
    /// Missing values get a separate bin
    Separate,
}

impl Default for MissingType {
    fn default() -> Self {
        MissingType::NaN
    }
}

/// Configuration for bin mapping
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinConfig {
    /// Maximum number of bins
    pub max_bins: usize,
    /// Minimum data points per bin
    pub min_data_per_bin: usize,
    /// Missing value handling
    pub missing_handling: MissingType,
    /// Force integer bins for low-cardinality features
    pub force_integer_bins: bool,
    /// Use sparse representation for categorical features
    pub use_sparse: bool,
}

impl Default for BinConfig {
    fn default() -> Self {
        BinConfig {
            max_bins: DEFAULT_MAX_BIN,
            min_data_per_bin: 3,
            missing_handling: MissingType::NaN,
            force_integer_bins: false,
            use_sparse: false,
        }
    }
}

impl BinMapper {
    /// Create a new numerical bin mapper
    pub fn new_numerical(
        values: &[f32],
        max_bins: usize,
        min_data_per_bin: usize,
        missing_type: MissingType,
    ) -> Result<Self> {
        // Filter out NaN values
        let mut valid_values: Vec<f32> = values.iter()
            .copied()
            .filter(|x| !x.is_nan())
            .collect();
        
        if valid_values.is_empty() {
            return Err(LightGBMError::dataset("No valid values found for binning"));
        }
        
        // Sort values for quantile calculation
        valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let min_value = valid_values[0] as f64;
        let max_value = valid_values[valid_values.len() - 1] as f64;
        let num_unique_values = Self::count_unique_values(&valid_values);
        
        // Determine actual number of bins
        let num_bins = std::cmp::min(max_bins, num_unique_values);
        let effective_bins = if num_bins <= 1 { 1 } else { num_bins };
        
        // Create bin boundaries
        let bin_upper_bounds = if effective_bins == 1 {
            vec![max_value]
        } else if num_unique_values <= max_bins {
            // Use unique values as boundaries
            Self::create_unique_value_boundaries(&valid_values)
        } else {
            // Use quantile-based boundaries
            Self::create_quantile_boundaries(&valid_values, effective_bins, min_data_per_bin)
        };
        
        // Determine default bin for missing values
        let default_bin = match missing_type {
            MissingType::None => 0,
            MissingType::Zero => 0,
            MissingType::NaN => 0,
            MissingType::Separate => bin_upper_bounds.len() as BinIndex,
        };
        
        // Calculate memory usage
        let memory_usage = Self::calculate_memory_usage_numerical(&bin_upper_bounds);
        let num_bins = bin_upper_bounds.len();
        
        Ok(BinMapper {
            bin_upper_bounds,
            category_to_bin: HashMap::new(),
            bin_to_category: HashMap::new(),
            max_bins,
            num_bins,
            bin_type: BinType::Numerical,
            missing_type,
            default_bin,
            min_value,
            max_value,
            num_unique_values,
            memory_usage,
        })
    }
    
    /// Create a new categorical bin mapper
    pub fn new_categorical(
        values: &[f32],
        max_bins: usize,
        missing_type: MissingType,
    ) -> Result<Self> {
        // Convert to integers and count occurrences
        let mut category_counts: HashMap<i32, usize> = HashMap::new();
        let mut has_missing = false;
        
        for &value in values {
            if value.is_nan() {
                has_missing = true;
            } else {
                let category = value as i32;
                *category_counts.entry(category).or_insert(0) += 1;
            }
        }
        
        if category_counts.is_empty() && !has_missing {
            return Err(LightGBMError::dataset("No valid categorical values found"));
        }
        
        // Sort categories by frequency (most frequent first)
        let mut categories: Vec<(i32, usize)> = category_counts.into_iter().collect();
        categories.sort_by(|a, b| b.1.cmp(&a.1));
        
        // Create mappings
        let mut category_to_bin = HashMap::new();
        let mut bin_to_category = HashMap::new();
        
        let effective_bins = std::cmp::min(max_bins, categories.len());
        
        for (bin_idx, (category, _count)) in categories.iter().take(effective_bins).enumerate() {
            let bin = bin_idx as BinIndex;
            category_to_bin.insert(*category, bin);
            bin_to_category.insert(bin, *category);
        }
        
        // Handle rare categories (map to last bin)
        if categories.len() > effective_bins {
            let last_bin = (effective_bins - 1) as BinIndex;
            for (category, _count) in categories.iter().skip(effective_bins) {
                category_to_bin.insert(*category, last_bin);
            }
        }
        
        // Determine default bin for missing values
        let default_bin = match missing_type {
            MissingType::None => 0,
            MissingType::Zero => 0,
            MissingType::NaN => 0,
            MissingType::Separate => effective_bins as BinIndex,
        };
        
        // Calculate min/max values
        let min_value = categories.iter().map(|(cat, _)| *cat as f64).fold(f64::INFINITY, f64::min);
        let max_value = categories.iter().map(|(cat, _)| *cat as f64).fold(f64::NEG_INFINITY, f64::max);
        
        // Calculate memory usage
        let memory_usage = Self::calculate_memory_usage_categorical(&category_to_bin, &bin_to_category);
        
        Ok(BinMapper {
            bin_upper_bounds: Vec::new(),
            category_to_bin,
            bin_to_category,
            max_bins,
            num_bins: effective_bins,
            bin_type: BinType::Categorical,
            missing_type,
            default_bin,
            min_value,
            max_value,
            num_unique_values: categories.len(),
            memory_usage,
        })
    }
    
    /// Map a feature value to its bin index
    pub fn value_to_bin(&self, value: f32) -> BinIndex {
        if value.is_nan() {
            return self.default_bin;
        }
        
        match self.bin_type {
            BinType::Numerical => {
                let value_f64 = value as f64;
                
                // Handle edge cases
                if value_f64 <= self.min_value {
                    return 0;
                }
                if value_f64 >= self.max_value {
                    return (self.num_bins - 1) as BinIndex;
                }
                
                // Binary search for the appropriate bin
                for (bin_idx, &boundary) in self.bin_upper_bounds.iter().enumerate() {
                    if value_f64 <= boundary {
                        return bin_idx as BinIndex;
                    }
                }
                
                // Fallback to last bin
                (self.num_bins - 1) as BinIndex
            }
            BinType::Categorical => {
                let category = value as i32;
                self.category_to_bin.get(&category)
                    .copied()
                    .unwrap_or(self.default_bin)
            }
        }
    }
    
    /// Map a bin index back to a representative value
    pub fn bin_to_value(&self, bin: BinIndex) -> f32 {
        match self.bin_type {
            BinType::Numerical => {
                if bin == 0 {
                    // For first bin, return midpoint between min_value and first boundary
                    if self.bin_upper_bounds.is_empty() {
                        self.min_value as f32
                    } else {
                        ((self.min_value + self.bin_upper_bounds[0]) / 2.0) as f32
                    }
                } else if (bin as usize) < self.bin_upper_bounds.len() {
                    // For other bins, return midpoint between previous boundary and current boundary
                    let current_upper = self.bin_upper_bounds[bin as usize];
                    let previous_upper = if bin == 1 {
                        self.min_value
                    } else {
                        self.bin_upper_bounds[(bin - 1) as usize]
                    };
                    ((previous_upper + current_upper) / 2.0) as f32
                } else {
                    // For bins beyond the defined bounds, return max_value
                    self.max_value as f32
                }
            }
            BinType::Categorical => {
                self.bin_to_category.get(&bin)
                    .copied()
                    .unwrap_or(0) as f32
            }
        }
    }
    
    /// Get the number of bins
    pub fn num_bins(&self) -> usize {
        self.num_bins
    }
    
    /// Get the number of unique values
    pub fn num_unique_values(&self) -> usize {
        self.num_unique_values
    }
    
    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.memory_usage
    }
    
    /// Check if this is a numerical bin mapper
    pub fn is_numerical(&self) -> bool {
        self.bin_type == BinType::Numerical
    }
    
    /// Check if this is a categorical bin mapper
    pub fn is_categorical(&self) -> bool {
        self.bin_type == BinType::Categorical
    }
    
    /// Get bin boundaries (for numerical features)
    pub fn bin_boundaries(&self) -> &[f64] {
        &self.bin_upper_bounds
    }
    
    /// Get category mappings (for categorical features)
    pub fn category_mappings(&self) -> &HashMap<i32, BinIndex> {
        &self.category_to_bin
    }
    
    /// Validate the bin mapper
    pub fn validate(&self) -> Result<()> {
        match self.bin_type {
            BinType::Numerical => {
                if self.bin_upper_bounds.is_empty() {
                    return Err(LightGBMError::dataset("Numerical bin mapper has no boundaries"));
                }
                
                // Check boundaries are sorted
                for i in 1..self.bin_upper_bounds.len() {
                    if self.bin_upper_bounds[i] <= self.bin_upper_bounds[i - 1] {
                        return Err(LightGBMError::dataset("Bin boundaries are not sorted"));
                    }
                }
            }
            BinType::Categorical => {
                if self.category_to_bin.is_empty() {
                    return Err(LightGBMError::dataset("Categorical bin mapper has no mappings"));
                }
                
                // Check consistency between mappings
                for (&category, &bin) in &self.category_to_bin {
                    if let Some(&mapped_category) = self.bin_to_category.get(&bin) {
                        if category != mapped_category && bin != (self.num_bins - 1) as BinIndex {
                            // Allow multiple categories to map to the last bin (rare categories)
                            return Err(LightGBMError::dataset("Inconsistent category mappings"));
                        }
                    }
                }
            }
        }
        
        // Check number of bins
        if self.num_bins == 0 {
            return Err(LightGBMError::dataset("Bin mapper has zero bins"));
        }
        
        if self.num_bins > self.max_bins {
            return Err(LightGBMError::dataset("Number of bins exceeds maximum"));
        }
        
        Ok(())
    }
    
    /// Convert to a different missing type
    pub fn convert_missing_type(&mut self, new_missing_type: MissingType) {
        self.missing_type = new_missing_type;
        self.default_bin = match new_missing_type {
            MissingType::None => 0,
            MissingType::Zero => 0,
            MissingType::NaN => 0,
            MissingType::Separate => self.num_bins as BinIndex,
        };
    }
    
    /// Create unique value boundaries
    fn create_unique_value_boundaries(values: &[f32]) -> Vec<f64> {
        let mut unique_values: Vec<f32> = values.iter().copied().collect();
        unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_values.dedup();
        unique_values.into_iter().map(|x| x as f64).collect()
    }
    
    /// Create quantile-based boundaries
    fn create_quantile_boundaries(
        values: &[f32],
        num_bins: usize,
        min_data_per_bin: usize,
    ) -> Vec<f64> {
        let n = values.len();
        let mut boundaries = Vec::new();
        
        for i in 1..=num_bins {
            let quantile = i as f64 / num_bins as f64;
            let index = ((n - 1) as f64 * quantile).round() as usize;
            let index = std::cmp::min(index, n - 1);
            boundaries.push(values[index] as f64);
        }
        
        // Remove duplicate boundaries
        boundaries.dedup();
        
        // Ensure minimum data per bin
        let min_bin_size = std::cmp::max(1, min_data_per_bin);
        let mut final_boundaries = Vec::new();
        let mut last_index = 0;
        
        for &boundary in &boundaries {
            let current_index = values.binary_search_by(|x| x.partial_cmp(&(boundary as f32)).unwrap())
                .unwrap_or_else(|x| x);
            
            if current_index - last_index >= min_bin_size {
                final_boundaries.push(boundary);
                last_index = current_index;
            }
        }
        
        // Ensure we have at least one boundary
        if final_boundaries.is_empty() {
            final_boundaries.push(values[values.len() - 1] as f64);
        }
        
        final_boundaries
    }
    
    /// Count unique values in a sorted array
    fn count_unique_values(values: &[f32]) -> usize {
        if values.is_empty() {
            return 0;
        }
        
        let mut count = 1;
        let mut last_value = values[0];
        
        for &value in values.iter().skip(1) {
            if value != last_value {
                count += 1;
                last_value = value;
            }
        }
        
        count
    }
    
    /// Calculate memory usage for numerical bin mapper
    fn calculate_memory_usage_numerical(boundaries: &[f64]) -> usize {
        boundaries.len() * std::mem::size_of::<f64>() + std::mem::size_of::<BinMapper>()
    }
    
    /// Calculate memory usage for categorical bin mapper
    fn calculate_memory_usage_categorical(
        category_to_bin: &HashMap<i32, BinIndex>,
        bin_to_category: &HashMap<BinIndex, i32>,
    ) -> usize {
        let map1_size = category_to_bin.len() * (std::mem::size_of::<i32>() + std::mem::size_of::<BinIndex>());
        let map2_size = bin_to_category.len() * (std::mem::size_of::<BinIndex>() + std::mem::size_of::<i32>());
        map1_size + map2_size + std::mem::size_of::<BinMapper>()
    }
}

/// Utility functions for bin mapping
pub mod utils {
    use super::*;
    
    /// Create optimal bin mapper for given data
    pub fn create_optimal_mapper(
        values: &[f32],
        feature_type: FeatureType,
        config: &BinConfig,
    ) -> Result<BinMapper> {
        match feature_type {
            FeatureType::Numerical => {
                BinMapper::new_numerical(
                    values,
                    config.max_bins,
                    config.min_data_per_bin,
                    config.missing_handling,
                )
            }
            FeatureType::Categorical => {
                BinMapper::new_categorical(
                    values,
                    config.max_bins,
                    config.missing_handling,
                )
            }
        }
    }
    
    /// Estimate optimal number of bins
    pub fn estimate_optimal_bins(
        values: &[f32],
        feature_type: FeatureType,
        max_bins: usize,
    ) -> usize {
        let valid_values: Vec<f32> = values.iter()
            .copied()
            .filter(|x| !x.is_nan())
            .collect();
        
        if valid_values.is_empty() {
            return 1;
        }
        
        match feature_type {
            FeatureType::Numerical => {
                let unique_count = BinMapper::count_unique_values(&valid_values);
                std::cmp::min(max_bins, unique_count)
            }
            FeatureType::Categorical => {
                let unique_categories: std::collections::HashSet<i32> = valid_values.iter()
                    .map(|&x| x as i32)
                    .collect();
                std::cmp::min(max_bins, unique_categories.len())
            }
        }
    }
    
    /// Check if feature should be treated as categorical
    pub fn is_categorical_feature(values: &[f32], max_categories: usize) -> bool {
        let valid_values: Vec<f32> = values.iter()
            .copied()
            .filter(|x| !x.is_nan())
            .collect();
        
        if valid_values.is_empty() {
            return false;
        }
        
        // Check if all values are integers
        let all_integers = valid_values.iter().all(|&x| x.fract() == 0.0);
        
        if !all_integers {
            return false;
        }
        
        // Check number of unique values
        let unique_values: std::collections::HashSet<i32> = valid_values.iter()
            .map(|&x| x as i32)
            .collect();
        
        unique_values.len() <= max_categories
    }
    
    /// Merge bin mappers for ensemble models
    pub fn merge_mappers(mappers: &[BinMapper]) -> Result<BinMapper> {
        if mappers.is_empty() {
            return Err(LightGBMError::dataset("No mappers to merge"));
        }
        
        let first_mapper = &mappers[0];
        
        // Check compatibility
        for mapper in mappers.iter().skip(1) {
            if mapper.bin_type != first_mapper.bin_type {
                return Err(LightGBMError::dataset("Cannot merge mappers of different types"));
            }
            
            if mapper.missing_type != first_mapper.missing_type {
                return Err(LightGBMError::dataset("Cannot merge mappers with different missing types"));
            }
        }
        
        // For numerical mappers, take the union of all boundaries
        match first_mapper.bin_type {
            BinType::Numerical => {
                let mut all_boundaries = Vec::new();
                for mapper in mappers {
                    all_boundaries.extend(&mapper.bin_upper_bounds);
                }
                
                all_boundaries.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());
                all_boundaries.dedup();
                
                let mut merged = first_mapper.clone();
                merged.bin_upper_bounds = all_boundaries;
                merged.num_bins = merged.bin_upper_bounds.len();
                
                Ok(merged)
            }
            BinType::Categorical => {
                let mut merged_category_to_bin = HashMap::new();
                let mut merged_bin_to_category = HashMap::new();
                let mut next_bin = 0;
                
                for mapper in mappers {
                    for (&category, &_bin) in &mapper.category_to_bin {
                        if !merged_category_to_bin.contains_key(&category) {
                            merged_category_to_bin.insert(category, next_bin);
                            merged_bin_to_category.insert(next_bin, category);
                            next_bin += 1;
                        }
                    }
                }
                
                let mut merged = first_mapper.clone();
                merged.category_to_bin = merged_category_to_bin;
                merged.bin_to_category = merged_bin_to_category;
                merged.num_bins = next_bin as usize;
                
                Ok(merged)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_numerical_bin_mapper() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mapper = BinMapper::new_numerical(&values, 3, 1, MissingType::NaN).unwrap();
        
        assert_eq!(mapper.bin_type, BinType::Numerical);
        assert_eq!(mapper.num_bins(), 3);
        assert_eq!(mapper.value_to_bin(1.0), 0);
        assert_eq!(mapper.value_to_bin(3.0), 1);
        assert_eq!(mapper.value_to_bin(5.0), 2);
    }
    
    #[test]
    fn test_categorical_bin_mapper() {
        let values = vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0];
        let mapper = BinMapper::new_categorical(&values, 10, MissingType::NaN).unwrap();
        
        assert_eq!(mapper.bin_type, BinType::Categorical);
        assert_eq!(mapper.num_bins(), 3);
        assert_eq!(mapper.value_to_bin(1.0), 0);  // Most frequent
        assert_eq!(mapper.value_to_bin(2.0), 1);  // Second most frequent
        assert_eq!(mapper.value_to_bin(3.0), 2);  // Least frequent
    }
    
    #[test]
    fn test_missing_value_handling() {
        let values = vec![1.0, 2.0, f32::NAN, 4.0, 5.0];
        let mapper = BinMapper::new_numerical(&values, 3, 1, MissingType::Separate).unwrap();
        
        assert_eq!(mapper.value_to_bin(f32::NAN), mapper.default_bin);
        assert_ne!(mapper.value_to_bin(1.0), mapper.default_bin);
    }
    
    #[test]
    fn test_bin_mapper_validation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mapper = BinMapper::new_numerical(&values, 3, 1, MissingType::NaN).unwrap();
        
        assert!(mapper.validate().is_ok());
    }
    
    #[test]
    fn test_bin_to_value_conversion() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mapper = BinMapper::new_numerical(&values, 3, 1, MissingType::NaN).unwrap();
        
        let bin = mapper.value_to_bin(2.5);
        let representative_value = mapper.bin_to_value(bin);
        assert!(representative_value >= 2.0 && representative_value <= 3.0);
    }
    
    #[test]
    fn test_categorical_feature_detection() {
        let categorical_values = vec![1.0, 2.0, 1.0, 3.0, 2.0];
        assert!(utils::is_categorical_feature(&categorical_values, 10));
        
        let numerical_values = vec![1.1, 2.2, 3.3, 4.4, 5.5];
        assert!(!utils::is_categorical_feature(&numerical_values, 10));
    }
    
    #[test]
    fn test_optimal_bins_estimation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let num_bins = utils::estimate_optimal_bins(&values, FeatureType::Numerical, 5);
        assert_eq!(num_bins, 5);
        
        let categorical_values = vec![1.0, 2.0, 1.0, 3.0, 2.0];
        let num_bins = utils::estimate_optimal_bins(&categorical_values, FeatureType::Categorical, 10);
        assert_eq!(num_bins, 3);
    }
    
    #[test]
    fn test_unique_value_boundaries() {
        let values = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0];
        let boundaries = BinMapper::create_unique_value_boundaries(&values);
        assert_eq!(boundaries, vec![1.0, 2.0, 3.0, 4.0]);
    }
    
    #[test]
    fn test_memory_usage_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mapper = BinMapper::new_numerical(&values, 3, 1, MissingType::NaN).unwrap();
        
        assert!(mapper.memory_usage() > 0);
    }
}