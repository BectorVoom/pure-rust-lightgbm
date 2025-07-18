//! Core dataset structure for Pure Rust LightGBM.
//!
//! This module provides the main Dataset structure that holds training data,
//! labels, and metadata in an optimized format for gradient boosting training.

use crate::core::error::{LightGBMError, Result};
use crate::core::types::*;

use crate::dataset::binning::BinMapper;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Main dataset structure for LightGBM training and prediction
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Feature matrix (num_data × num_features)
    features: Array2<f32>,
    /// Target labels (num_data,)
    labels: Array1<f32>,
    /// Sample weights (optional)
    weights: Option<Array1<f32>>,
    /// Group information for ranking (optional)
    groups: Option<Array1<DataSize>>,
    /// Number of data points
    num_data: DataSize,
    /// Number of features
    num_features: usize,
    /// Feature names for interpretability
    feature_names: Option<Vec<String>>,
    /// Feature types (numerical or categorical)
    feature_types: Vec<FeatureType>,
    /// Feature binning information
    bin_mappers: Vec<BinMapper>,
    /// Missing value indicators
    missing_values: Option<Array2<bool>>,
    /// Dataset metadata
    metadata: DatasetMetadata,
    /// Cached statistics
    statistics: Option<crate::dataset::DatasetStatistics>,
    /// Memory usage tracking
    memory_usage: usize,
}

/// Dataset metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Dataset name
    pub name: String,
    /// Dataset description
    pub description: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Source file path
    pub source_path: Option<String>,
    /// Data format
    pub format: String,
    /// Version information
    pub version: String,
    /// Custom properties
    pub properties: HashMap<String, String>,
}

impl Default for DatasetMetadata {
    fn default() -> Self {
        DatasetMetadata {
            name: "untitled".to_string(),
            description: "LightGBM dataset".to_string(),
            created_at: chrono::Utc::now(),
            source_path: None,
            format: "memory".to_string(),
            version: "1.0".to_string(),
            properties: HashMap::new(),
        }
    }
}

/// Dataset information structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetInfo {
    /// Number of samples
    pub num_samples: usize,
    /// Number of features
    pub num_features: usize,
    /// Number of classes (for classification)
    pub num_classes: Option<usize>,
    /// Feature types
    pub feature_types: Vec<FeatureType>,
    /// Has missing values
    pub has_missing_values: bool,
    /// Memory size estimate in bytes
    pub memory_size_bytes: usize,
    /// Sparsity ratio
    pub sparsity: f64,
    /// Feature names
    pub feature_names: Option<Vec<String>>,
}

impl Dataset {
    /// Create a new dataset from arrays
    pub fn new(
        features: Array2<f32>,
        labels: Array1<f32>,
        weights: Option<Array1<f32>>,
        groups: Option<Array1<DataSize>>,
        feature_names: Option<Vec<String>>,
        feature_types: Option<Vec<FeatureType>>,
    ) -> Result<Self> {
        let num_data = features.nrows() as DataSize;
        let num_features = features.ncols();

        // Validate dimensions
        if labels.len() != num_data as usize {
            return Err(LightGBMError::dimension_mismatch(
                format!("features rows: {}", num_data),
                format!("labels length: {}", labels.len()),
            ));
        }

        if let Some(ref weights) = weights {
            if weights.len() != num_data as usize {
                return Err(LightGBMError::dimension_mismatch(
                    format!("features rows: {}", num_data),
                    format!("weights length: {}", weights.len()),
                ));
            }
        }

        if let Some(ref groups) = groups {
            if groups.len() != num_data as usize {
                return Err(LightGBMError::dimension_mismatch(
                    format!("features rows: {}", num_data),
                    format!("groups length: {}", groups.len()),
                ));
            }
        }

        // Validate feature names
        if let Some(ref names) = feature_names {
            if names.len() != num_features {
                return Err(LightGBMError::dimension_mismatch(
                    format!("features columns: {}", num_features),
                    format!("feature names length: {}", names.len()),
                ));
            }
        }

        // Determine feature types
        let feature_types = match feature_types {
            Some(types) => {
                if types.len() != num_features {
                    return Err(LightGBMError::dimension_mismatch(
                        format!("features columns: {}", num_features),
                        format!("feature types length: {}", types.len()),
                    ));
                }
                types
            }
            None => {
                // Auto-detect feature types
                crate::dataset::utils::detect_feature_types(&features)
            }
        };

        // Calculate memory usage
        let memory_usage = Self::calculate_memory_usage(&features, &labels, &weights, &groups);

        Ok(Dataset {
            features,
            labels,
            weights,
            groups,
            num_data,
            num_features,
            feature_names,
            feature_types,
            bin_mappers: Vec::new(),
            missing_values: None,
            metadata: DatasetMetadata::default(),
            statistics: None,
            memory_usage,
        })
    }

    /// Create a dataset builder
    pub fn builder() -> DatasetBuilder {
        DatasetBuilder::new()
    }

    /// Get number of data points
    pub fn num_data(&self) -> usize {
        self.num_data as usize
    }

    /// Get number of features
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Get feature matrix view
    pub fn features(&self) -> ArrayView2<f32> {
        self.features.view()
    }

    /// Get raw feature matrix reference (for internal use)
    pub fn features_raw(&self) -> &Array2<f32> {
        &self.features
    }

    /// Get labels view
    pub fn labels(&self) -> ArrayView1<f32> {
        self.labels.view()
    }

    /// Get a single label at the specified index
    pub fn label(&self, index: usize) -> Result<f32> {
        self.labels.get(index)
            .copied()
            .ok_or_else(|| LightGBMError::dataset(format!("Label index {} out of bounds", index)))
    }

    /// Get weights view
    pub fn weights(&self) -> Option<ArrayView1<f32>> {
        self.weights.as_ref().map(|w| w.view())
    }

    /// Check if dataset has weights
    pub fn has_weights(&self) -> bool {
        self.weights.is_some()
    }

    /// Check if dataset has bin mappers
    pub fn has_bin_mappers(&self) -> bool {
        !self.bin_mappers.is_empty()
    }

    /// Get number of bin mappers
    pub fn num_bin_mappers(&self) -> usize {
        self.bin_mappers.len()
    }

    /// Get preprocessing statistics (placeholder)
    pub fn preprocessing_stats(&self) -> crate::dataset::DatasetStatistics {
        // Return default statistics for now
        crate::dataset::DatasetStatistics::default()
    }

    /// Get groups view
    pub fn groups(&self) -> Option<ArrayView1<DataSize>> {
        self.groups.as_ref().map(|g| g.view())
    }

    /// Get feature names
    pub fn feature_names(&self) -> Option<&[String]> {
        self.feature_names.as_deref()
    }

    /// Get feature name by index
    pub fn feature_name(&self, index: usize) -> Option<&str> {
        self.feature_names.as_ref()?.get(index).map(|s| s.as_str())
    }

    /// Get feature types
    pub fn feature_types(&self) -> &[FeatureType] {
        &self.feature_types
    }

    /// Get feature type by index
    pub fn feature_type(&self, index: usize) -> Option<FeatureType> {
        self.feature_types.get(index).copied()
    }

    /// Get feature data by index
    pub fn feature_data(&self, index: usize) -> Vec<f32> {
        self.features.column(index).to_vec()
    }

    /// Get bin mappers
    pub fn bin_mappers(&self) -> &[BinMapper] {
        &self.bin_mappers
    }

    /// Get bin mapper by feature index
    pub fn bin_mapper(&self, index: usize) -> Option<&BinMapper> {
        self.bin_mappers.get(index)
    }

    /// Get dataset metadata
    pub fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    /// Get mutable dataset metadata
    pub fn metadata_mut(&mut self) -> &mut DatasetMetadata {
        &mut self.metadata
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.memory_usage
    }

    /// Check if dataset has missing values
    pub fn has_missing_values(&self) -> bool {
        self.missing_values.is_some()
    }

    /// Get missing values matrix
    pub fn missing_values(&self) -> Option<ArrayView2<bool>> {
        self.missing_values.as_ref().map(|mv| mv.view())
    }

    /// Set bin mappers
    pub fn set_bin_mappers(&mut self, bin_mappers: Vec<BinMapper>) -> Result<()> {
        if bin_mappers.len() != self.num_features {
            return Err(LightGBMError::dimension_mismatch(
                format!("features: {}", self.num_features),
                format!("bin mappers: {}", bin_mappers.len()),
            ));
        }
        self.bin_mappers = bin_mappers;
        Ok(())
    }

    /// Set missing values matrix
    pub fn set_missing_values(&mut self, missing_values: Array2<bool>) -> Result<()> {
        if missing_values.nrows() != self.num_data as usize
            || missing_values.ncols() != self.num_features
        {
            return Err(LightGBMError::dimension_mismatch(
                format!("dataset: {}x{}", self.num_data, self.num_features),
                format!(
                    "missing values: {}x{}",
                    missing_values.nrows(),
                    missing_values.ncols()
                ),
            ));
        }
        self.missing_values = Some(missing_values);
        Ok(())
    }

    /// Get dataset statistics
    pub fn statistics(&self) -> Result<&crate::dataset::DatasetStatistics> {
        if let Some(ref stats) = self.statistics {
            Ok(stats)
        } else {
            Err(LightGBMError::dataset(
                "Statistics not computed. Call compute_statistics() first.",
            ))
        }
    }

    /// Compute dataset statistics
    pub fn compute_statistics(&mut self) {
        self.statistics = Some(crate::dataset::utils::calculate_statistics(self));
    }

    /// Get dataset information
    pub fn info(&self) -> DatasetInfo {
        DatasetInfo {
            num_samples: self.num_data as usize,
            num_features: self.num_features,
            num_classes: self.detect_num_classes(),
            feature_types: self.feature_types.clone(),
            has_missing_values: self.has_missing_values(),
            memory_size_bytes: self.memory_usage,
            sparsity: self.calculate_sparsity(),
            feature_names: self.feature_names.clone(),
        }
    }

    /// Validate dataset
    pub fn validate(&self) -> crate::dataset::DatasetValidationResult {
        crate::dataset::utils::validate_dataset(self)
    }

    /// Create a subset of the dataset
    pub fn subset(&self, indices: &[usize]) -> Result<Self> {
        if indices.iter().any(|&i| i >= self.num_data as usize) {
            return Err(LightGBMError::index_out_of_bounds(
                indices.iter().max().unwrap_or(&0).to_owned(),
                self.num_data as usize,
            ));
        }

        let new_features = Array2::from_shape_fn((indices.len(), self.num_features), |(i, j)| {
            self.features[[indices[i], j]]
        });

        let new_labels = Array1::from_iter(indices.iter().map(|&i| self.labels[i]));

        let new_weights = self
            .weights
            .as_ref()
            .map(|w| Array1::from_iter(indices.iter().map(|&i| w[i])));

        let new_groups = self
            .groups
            .as_ref()
            .map(|g| Array1::from_iter(indices.iter().map(|&i| g[i])));

        Self::new(
            new_features,
            new_labels,
            new_weights,
            new_groups,
            self.feature_names.clone(),
            Some(self.feature_types.clone()),
        )
    }

    /// Split dataset into training and validation sets
    pub fn split(
        &self,
        train_ratio: f64,
        shuffle: bool,
        seed: Option<u64>,
    ) -> Result<(Self, Self)> {
        if train_ratio <= 0.0 || train_ratio >= 1.0 {
            return Err(LightGBMError::invalid_parameter(
                "train_ratio",
                train_ratio.to_string(),
                "must be in range (0.0, 1.0)",
            ));
        }

        let mut indices: Vec<usize> = (0..self.num_data as usize).collect();

        if shuffle {
            use rand::{Rng, SeedableRng};
            let mut rng = match seed {
                Some(s) => rand::rngs::StdRng::seed_from_u64(s),
                None => rand::rngs::StdRng::from_entropy(),
            };

            for i in (1..indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }
        }

        let split_point = (indices.len() as f64 * train_ratio) as usize;
        let train_indices = &indices[..split_point];
        let val_indices = &indices[split_point..];

        let train_dataset = self.subset(train_indices)?;
        let val_dataset = self.subset(val_indices)?;

        Ok((train_dataset, val_dataset))
    }

    /// Convert features to bins
    pub fn to_bins(&self, feature_index: usize) -> Result<Vec<BinIndex>> {
        if feature_index >= self.num_features {
            return Err(LightGBMError::index_out_of_bounds(
                feature_index,
                self.num_features,
            ));
        }

        let bin_mapper = self.bin_mappers.get(feature_index).ok_or_else(|| {
            LightGBMError::dataset("Bin mapper not found. Call set_bin_mappers() first.")
        })?;

        let feature_data = self.features.column(feature_index);
        let bins = feature_data
            .iter()
            .map(|&value| bin_mapper.value_to_bin(value))
            .collect();

        Ok(bins)
    }

    /// Get unique classes from labels
    pub fn unique_classes(&self) -> Vec<f32> {
        let mut classes: Vec<f32> = self.labels.iter().copied().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        classes.dedup();
        classes
    }

    /// Detect number of classes
    fn detect_num_classes(&self) -> Option<usize> {
        let unique_classes = self.unique_classes();
        if unique_classes.len() <= 1000 {
            // Reasonable limit for classification
            Some(unique_classes.len())
        } else {
            None // Likely regression
        }
    }

    /// Calculate sparsity ratio
    fn calculate_sparsity(&self) -> f64 {
        if let Some(ref missing_values) = self.missing_values {
            let total_elements = missing_values.len();
            let missing_elements = missing_values.iter().filter(|&&x| x).count();
            missing_elements as f64 / total_elements as f64
        } else {
            // Count NaN values
            let mut missing_count = 0;
            for &value in self.features.iter() {
                if value.is_nan() {
                    missing_count += 1;
                }
            }
            missing_count as f64 / self.features.len() as f64
        }
    }

    /// Calculate memory usage
    fn calculate_memory_usage(
        features: &Array2<f32>,
        labels: &Array1<f32>,
        weights: &Option<Array1<f32>>,
        groups: &Option<Array1<DataSize>>,
    ) -> usize {
        let mut usage = 0;

        // Features matrix
        usage += features.len() * std::mem::size_of::<f32>();

        // Labels array
        usage += labels.len() * std::mem::size_of::<f32>();

        // Weights array
        if let Some(ref w) = weights {
            usage += w.len() * std::mem::size_of::<f32>();
        }

        // Groups array
        if let Some(ref g) = groups {
            usage += g.len() * std::mem::size_of::<DataSize>();
        }

        usage
    }
}

/// Dataset builder for constructing datasets with validation
pub struct DatasetBuilder {
    features: Option<Array2<f32>>,
    labels: Option<Array1<f32>>,
    weights: Option<Array1<f32>>,
    groups: Option<Array1<DataSize>>,
    feature_names: Option<Vec<String>>,
    feature_types: Option<Vec<FeatureType>>,
    metadata: DatasetMetadata,
}

impl DatasetBuilder {
    /// Create a new dataset builder
    pub fn new() -> Self {
        DatasetBuilder {
            features: None,
            labels: None,
            weights: None,
            groups: None,
            feature_names: None,
            feature_types: None,
            metadata: DatasetMetadata::default(),
        }
    }

    /// Set features
    pub fn features(mut self, features: Array2<f32>) -> Self {
        self.features = Some(features);
        self
    }

    /// Set labels
    pub fn labels(mut self, labels: Array1<f32>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Set weights
    pub fn weights(mut self, weights: Array1<f32>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Set groups
    pub fn groups(mut self, groups: Array1<DataSize>) -> Self {
        self.groups = Some(groups);
        self
    }

    /// Set feature names
    pub fn feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Set feature types
    pub fn feature_types(mut self, types: Vec<FeatureType>) -> Self {
        self.feature_types = Some(types);
        self
    }

    /// Set metadata
    pub fn metadata(mut self, metadata: DatasetMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Build the dataset
    pub fn build(self) -> Result<Dataset> {
        let features = self
            .features
            .ok_or_else(|| LightGBMError::dataset("Features are required"))?;

        let labels = self
            .labels
            .ok_or_else(|| LightGBMError::dataset("Labels are required"))?;

        let mut dataset = Dataset::new(
            features,
            labels,
            self.weights,
            self.groups,
            self.feature_names,
            self.feature_types,
        )?;

        dataset.metadata = self.metadata;

        Ok(dataset)
    }
}

impl Default for DatasetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Thread-safe dataset wrapper
pub struct ThreadSafeDataset {
    dataset: Arc<Dataset>,
}

impl ThreadSafeDataset {
    pub fn new(dataset: Dataset) -> Self {
        ThreadSafeDataset {
            dataset: Arc::new(dataset),
        }
    }

    pub fn dataset(&self) -> &Dataset {
        &self.dataset
    }
}

impl Clone for ThreadSafeDataset {
    fn clone(&self) -> Self {
        ThreadSafeDataset {
            dataset: Arc::clone(&self.dataset),
        }
    }
}

unsafe impl Send for ThreadSafeDataset {}
unsafe impl Sync for ThreadSafeDataset {}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_dataset_creation() {
        let features =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        let labels = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);

        let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();

        assert_eq!(dataset.num_data(), 4);
        assert_eq!(dataset.num_features(), 2);
        assert_eq!(dataset.labels().len(), 4);
        assert_eq!(dataset.features().nrows(), 4);
        assert_eq!(dataset.features().ncols(), 2);
    }

    #[test]
    fn test_dataset_builder() {
        let features = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let labels = Array1::from_vec(vec![0.0, 1.0]);
        let feature_names = vec!["feature1".to_string(), "feature2".to_string()];

        let dataset = Dataset::builder()
            .features(features)
            .labels(labels)
            .feature_names(feature_names)
            .build()
            .unwrap();

        assert_eq!(dataset.num_data(), 2);
        assert_eq!(dataset.num_features(), 2);
        assert_eq!(dataset.feature_names().unwrap().len(), 2);
        assert_eq!(dataset.feature_name(0), Some("feature1"));
    }

    #[test]
    fn test_dataset_validation() {
        // Create dataset with 10 samples to meet validation requirements
        let features = Array2::from_shape_vec(
            (10, 2), 
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0
            ]
        ).unwrap();
        let labels = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

        let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();
        let validation_result = dataset.validate();

        assert!(validation_result.is_valid);
        assert!(validation_result.errors.is_empty());
    }

    #[test]
    fn test_dataset_split() {
        let features = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0,
            ],
        )
        .unwrap();
        let labels = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

        let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();
        let (train, val) = dataset.split(0.8, false, Some(42)).unwrap();

        assert_eq!(train.num_data(), 8);
        assert_eq!(val.num_data(), 2);
        assert_eq!(train.num_features(), 2);
        assert_eq!(val.num_features(), 2);
    }

    #[test]
    fn test_dataset_subset() {
        let features =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let labels = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);

        let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();
        let subset = dataset.subset(&[0, 2]).unwrap();

        assert_eq!(subset.num_data(), 2);
        assert_eq!(subset.num_features(), 2);
        assert_eq!(subset.labels()[0], 0.0);
        assert_eq!(subset.labels()[1], 0.0);
    }

    #[test]
    fn test_dataset_info() {
        let features =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let labels = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);

        let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();
        let info = dataset.info();

        assert_eq!(info.num_samples, 4);
        assert_eq!(info.num_features, 2);
        assert_eq!(info.num_classes, Some(2));
        assert!(!info.has_missing_values);
    }

    #[test]
    fn test_thread_safe_dataset() {
        let features = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let labels = Array1::from_vec(vec![0.0, 1.0]);

        let dataset = Dataset::new(features, labels, None, None, None, None).unwrap();
        let thread_safe = ThreadSafeDataset::new(dataset);

        assert_eq!(thread_safe.dataset().num_data(), 2);
        assert_eq!(thread_safe.dataset().num_features(), 2);
    }
}
