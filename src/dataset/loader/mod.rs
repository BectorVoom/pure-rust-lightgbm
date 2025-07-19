//! Polars DataFrames

pub mod loader;
use crate::core::error::{LightGBMError, Result};
use crate::dataset::{Dataset, DatasetConfig};
pub use loader::PolarsLoader;
use ndarray::{Array1, Array2};
use polars::datatypes::PlSmallStr;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Data loader configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoaderConfig {
    /// Dataset configuration
    pub dataset_config: DatasetConfig,
    /// Skip header row
    pub skip_header: bool,
    /// Delimiter character
    pub delimiter: char,
    /// Quote character
    pub quote_char: char,
    /// Escape character
    pub escape_char: Option<char>,
    /// Encoding
    pub encoding: String,
    /// Maximum rows to read
    pub max_rows: Option<usize>,
    /// Chunk size for reading
    pub chunk_size: usize,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        LoaderConfig {
            dataset_config: DatasetConfig::default(),
            skip_header: true,
            delimiter: ',',
            quote_char: '"',
            escape_char: None,
            encoding: "utf-8".to_string(),
            max_rows: None,
            chunk_size: 10000,
        }
    }
}

/// Data loader trait
pub trait DataLoader {
    /// Load data from source
    fn load<P: AsRef<Path>>(&self, path: P) -> Result<Dataset>;

    /// Get loader configuration
    fn config(&self) -> &LoaderConfig;
}

/// Loader error type
#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("CSV parsing error: {0}")]
    Csv(String),
    #[error("Format error: {0}")]
    Format(String),
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
}

/// Array loader for in-memory data
pub struct ArrayLoader {
    config: LoaderConfig,
}

impl ArrayLoader {
    pub fn new(dataset_config: DatasetConfig) -> Result<Self> {
        Ok(ArrayLoader {
            config: LoaderConfig {
                dataset_config,
                ..Default::default()
            },
        })
    }

    /// Load dataset from in-memory arrays
    pub fn load_arrays(
        &self,
        features: Array2<f32>,
        labels: Array1<f32>,
        weights: Option<Array1<f32>>,
    ) -> Result<Dataset> {
        log::info!(
            "Loading dataset from arrays: {}x{}",
            features.nrows(),
            features.ncols()
        );

        // Validate dimensions
        if features.nrows() != labels.len() {
            return Err(LightGBMError::dimension_mismatch(
                format!("features rows: {}", features.nrows()),
                format!("labels length: {}", labels.len()),
            ));
        }

        if let Some(ref weights) = weights {
            if weights.len() != labels.len() {
                return Err(LightGBMError::dimension_mismatch(
                    format!("labels length: {}", labels.len()),
                    format!("weights length: {}", weights.len()),
                ));
            }
        }

        // Generate feature names if not provided
        let feature_names = if let Some(ref names) = self.config.dataset_config.feature_names {
            if names.len() != features.ncols() {
                return Err(LightGBMError::dimension_mismatch(
                    format!("features columns: {}", features.ncols()),
                    format!("feature names length: {}", names.len()),
                ));
            }
            Some(names.clone())
        } else {
            Some(
                (0..features.ncols())
                    .map(|i| format!("feature_{}", i))
                    .collect(),
            )
        };

        // Detect missing values
        let mut missing_mask = Array2::<bool>::default((features.nrows(), features.ncols()));
        let mut has_missing = false;

        for ((i, j), &value) in features.indexed_iter() {
            if value.is_nan() {
                missing_mask[[i, j]] = true;
                has_missing = true;
            }
        }

        // Create dataset
        let mut dataset = Dataset::new(
            features,
            labels,
            weights,
            None, // groups
            feature_names,
            None, // feature_types - will be auto-detected
        )?;

        // Set missing values mask if any missing values detected
        if has_missing {
            dataset.set_missing_values(missing_mask)?;
        }

        // Create bin mappers if binning is configured
        if self.config.dataset_config.max_bin > 0 {
            let mut bin_mappers = Vec::new();
            let categorical_features = self
                .config
                .dataset_config
                .categorical_features
                .clone()
                .unwrap_or_default();

            for feature_idx in 0..dataset.num_features() {
                let is_categorical = categorical_features.contains(&feature_idx);

                // Create a dummy bin mapper for now
                // In a complete implementation, this would analyze the feature values
                // and create appropriate bin boundaries
                use std::collections::HashMap;

                let bin_mapper = if is_categorical {
                    crate::dataset::binning::BinMapper {
                        bin_upper_bounds: vec![],
                        category_to_bin: HashMap::new(),
                        bin_to_category: HashMap::new(),
                        max_bins: 32,
                        num_bins: 3,
                        bin_type: crate::dataset::binning::BinType::Categorical,
                        missing_type: crate::dataset::binning::MissingType::None,
                        default_bin: 0,
                        min_value: 0.0,
                        max_value: 10.0,
                        num_unique_values: 3,
                        memory_usage: 100,
                    }
                } else {
                    crate::dataset::binning::BinMapper {
                        bin_upper_bounds: vec![0.0, 0.5, 1.0], // Dummy boundaries
                        category_to_bin: HashMap::new(),
                        bin_to_category: HashMap::new(),
                        max_bins: 32,
                        num_bins: 3,
                        bin_type: crate::dataset::binning::BinType::Numerical,
                        missing_type: crate::dataset::binning::MissingType::None,
                        default_bin: 0,
                        min_value: 0.0,
                        max_value: 1.0,
                        num_unique_values: 100,
                        memory_usage: 100,
                    }
                };

                bin_mappers.push(bin_mapper);
            }

            dataset.set_bin_mappers(bin_mappers)?;
        }

        // Set metadata
        let metadata = dataset.metadata_mut();
        metadata.source_path = Some("In-memory arrays".to_string());
        metadata.format = "arrays".to_string();
        metadata
            .properties
            .insert("source".to_string(), "memory".to_string());

        Ok(dataset)
    }

    /// Load dataset from separate feature and label vectors
    pub fn load_from_vecs(
        &self,
        features_vec: Vec<Vec<f32>>,
        labels_vec: Vec<f32>,
        weights_vec: Option<Vec<f32>>,
        feature_names: Option<Vec<PlSmallStr>>,
    ) -> Result<Dataset> {
        if features_vec.is_empty() {
            return Err(LightGBMError::data_loading("Features vector is empty"));
        }

        let num_rows = features_vec.len();
        let num_cols = features_vec[0].len();

        // Validate all rows have same number of features
        for (i, row) in features_vec.iter().enumerate() {
            if row.len() != num_cols {
                return Err(LightGBMError::dimension_mismatch(
                    format!("expected columns: {}", num_cols),
                    format!("row {} columns: {}", i, row.len()),
                ));
            }
        }

        // Convert to ndarray
        let mut features_flat = Vec::with_capacity(num_rows * num_cols);
        for row in features_vec {
            features_flat.extend(row);
        }

        let features =
            Array2::from_shape_vec((num_rows, num_cols), features_flat).map_err(|e| {
                LightGBMError::data_loading(format!("Failed to create features array: {}", e))
            })?;

        let labels = Array1::from_vec(labels_vec);

        let weights = weights_vec.map(Array1::from_vec);

        // Set feature names in config for load_arrays to use
        let mut modified_config = self.config.dataset_config.clone();
        modified_config.feature_names = feature_names;

        let loader = ArrayLoader {
            config: LoaderConfig {
                dataset_config: modified_config,
                ..self.config.clone()
            },
        };

        loader.load_arrays(features, labels, weights)
    }

    /// Load dataset from flattened data
    pub fn load_from_flat(
        &self,
        features_flat: Vec<f32>,
        num_rows: usize,
        num_cols: usize,
        labels: Vec<f32>,
        weights: Option<Vec<f32>>,
    ) -> Result<Dataset> {
        if features_flat.len() != num_rows * num_cols {
            return Err(LightGBMError::dimension_mismatch(
                format!("expected elements: {}", num_rows * num_cols),
                format!("features vector length: {}", features_flat.len()),
            ));
        }

        let features =
            Array2::from_shape_vec((num_rows, num_cols), features_flat).map_err(|e| {
                LightGBMError::data_loading(format!("Failed to create features array: {}", e))
            })?;

        let labels = Array1::from_vec(labels);
        let weights = weights.map(Array1::from_vec);

        self.load_arrays(features, labels, weights)
    }

    /// Create a simple dataset for testing
    pub fn create_test_dataset(
        &self,
        num_samples: usize,
        num_features: usize,
        seed: Option<u64>,
    ) -> Result<Dataset> {
        use rand::{Rng, SeedableRng};

        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        // Generate random features
        let mut features_flat = Vec::with_capacity(num_samples * num_features);
        for _ in 0..(num_samples * num_features) {
            features_flat.push(rng.gen_range(-5.0..5.0));
        }

        // Generate labels (simple linear combination for testing)
        let mut labels = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            let start_idx = i * num_features;
            let end_idx = start_idx + num_features;
            let feature_sum: f32 = features_flat[start_idx..end_idx].iter().sum();
            labels.push(if feature_sum > 0.0 { 1.0 } else { 0.0 });
        }

        self.load_from_flat(features_flat, num_samples, num_features, labels, None)
    }
}

impl DataLoader for ArrayLoader {
    fn load<P: AsRef<Path>>(&self, _path: P) -> Result<Dataset> {
        Err(LightGBMError::not_implemented(
            "Array loader does not support file loading - use load_arrays() instead",
        ))
    }

    fn config(&self) -> &LoaderConfig {
        &self.config
    }
}

/// Binary loader for LightGBM binary format
pub struct BinaryLoader {
    config: LoaderConfig,
}

impl BinaryLoader {
    pub fn new(dataset_config: DatasetConfig) -> Result<Self> {
        Ok(BinaryLoader {
            config: LoaderConfig {
                dataset_config,
                ..Default::default()
            },
        })
    }
}

impl DataLoader for BinaryLoader {
    fn load<P: AsRef<Path>>(&self, _path: P) -> Result<Dataset> {
        // TODO: Implement binary loading according to design document (LightGBMError::NotImplemented remains)
        Err(LightGBMError::not_implemented("Binary loading"))
    }

    fn config(&self) -> &LoaderConfig {
        &self.config
    }
}

/// Memory-mapped loader for large datasets
pub struct MemoryMappedLoader {
    config: LoaderConfig,
}

impl MemoryMappedLoader {
    pub fn new(dataset_config: DatasetConfig) -> Result<Self> {
        Ok(MemoryMappedLoader {
            config: LoaderConfig {
                dataset_config,
                ..Default::default()
            },
        })
    }
}

impl DataLoader for MemoryMappedLoader {
    fn load<P: AsRef<Path>>(&self, _path: P) -> Result<Dataset> {
        // TODO: Implement memory-mapped loading according to design document (LightGBMError::NotImplemented remains)
        Err(LightGBMError::not_implemented("Memory-mapped loading"))
    }

    fn config(&self) -> &LoaderConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loader_config_default() {
        let config = LoaderConfig::default();
        assert_eq!(config.delimiter, ',');
        assert_eq!(config.quote_char, '"');
        assert!(config.skip_header);
        assert_eq!(config.encoding, "utf-8");
    }

    #[test]
    fn test_array_loader_creation() {
        let dataset_config = DatasetConfig::default();
        let loader = ArrayLoader::new(dataset_config).unwrap();
        assert_eq!(loader.config().chunk_size, 10000);
    }
}
