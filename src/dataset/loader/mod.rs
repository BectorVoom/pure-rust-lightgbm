//! Data loading utilities for various formats.
//!
//! This module provides loaders for different data formats including CSV,
//! Polars DataFrames, Arrow tables, and Parquet files.

use crate::core::types::*;
use crate::core::error::{Result, LightGBMError};
use crate::dataset::{Dataset, DatasetConfig};
use ndarray::{Array1, Array2};
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
    Csv(#[from] csv::Error),
    #[error("Format error: {0}")]
    Format(String),
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
}

/// CSV loader
pub struct CsvLoader {
    config: LoaderConfig,
}

impl CsvLoader {
    pub fn new(dataset_config: DatasetConfig) -> Result<Self> {
        Ok(CsvLoader {
            config: LoaderConfig {
                dataset_config,
                ..Default::default()
            },
        })
    }
    
    /// Load CSV with custom configuration
    pub fn with_config(mut self, delimiter: char, has_header: bool, quote_char: char) -> Self {
        self.config.delimiter = delimiter;
        self.config.skip_header = has_header;
        self.config.quote_char = quote_char;
        self
    }
    
    /// Load CSV data from file path
    pub fn load_csv<P: AsRef<Path>>(&self, path: P) -> Result<Dataset> {
        use csv::ReaderBuilder;
        use std::fs::File;
        
        let path = path.as_ref();
        log::info!("Loading CSV file: {}", path.display());
        
        // Open and validate file
        let file = File::open(path)
            .map_err(|e| LightGBMError::data_loading(format!("Failed to open file {}: {}", path.display(), e)))?;
        
        // Configure CSV reader
        let mut reader = ReaderBuilder::new()
            .delimiter(self.config.delimiter as u8)
            .quote(self.config.quote_char as u8)
            .has_headers(self.config.skip_header)
            .flexible(true)
            .from_reader(file);
        
        // Get headers if available
        let headers = if self.config.skip_header {
            Some(reader.headers()?.clone())
        } else {
            None
        };
        
        // Parse all records
        let mut records = Vec::new();
        let mut num_columns = 0;
        
        for (line_num, result) in reader.records().enumerate() {
            let record = result.map_err(|e| {
                LightGBMError::data_loading(format!("CSV parsing error at line {}: {}", line_num + 1, e))
            })?;
            
            if num_columns == 0 {
                num_columns = record.len();
            } else if record.len() != num_columns {
                return Err(LightGBMError::data_loading(format!(
                    "Inconsistent column count at line {}: expected {}, got {}",
                    line_num + 1, num_columns, record.len()
                )));
            }
            
            // Apply max_rows limit
            if let Some(max_rows) = self.config.max_rows {
                if records.len() >= max_rows {
                    log::info!("Reached max_rows limit of {}, stopping", max_rows);
                    break;
                }
            }
            
            records.push(record);
        }
        
        if records.is_empty() {
            return Err(LightGBMError::data_loading("CSV file is empty or contains no valid data"));
        }
        
        log::info!("Loaded {} rows with {} columns", records.len(), num_columns);
        
        // Convert to Dataset
        self.convert_to_dataset(records, headers)
    }
    
    /// Convert CSV records to Dataset
    fn convert_to_dataset(&self, records: Vec<csv::StringRecord>, headers: Option<csv::StringRecord>) -> Result<Dataset> {
        let num_rows = records.len();
        let num_cols = records[0].len();
        
        // Determine target column
        let target_col_idx = if let Some(ref target_col) = self.config.dataset_config.target_column {
            if let Some(ref headers) = headers {
                headers.iter().position(|h| h == target_col)
                    .ok_or_else(|| LightGBMError::data_loading(format!("Target column '{}' not found", target_col)))?
            } else {
                target_col.parse::<usize>()
                    .map_err(|_| LightGBMError::data_loading("Target column must be column name (with headers) or index"))?
            }
        } else {
            num_cols - 1  // Default: last column is target
        };
        
        // Separate features and labels
        let feature_cols: Vec<usize> = (0..num_cols).filter(|&i| i != target_col_idx).collect();
        let num_features = feature_cols.len();
        
        // Initialize arrays
        let mut features = Array2::<f32>::zeros((num_rows, num_features));
        let mut labels = Array1::<f32>::zeros(num_rows);
        let mut missing_mask = Array2::<bool>::from_elem((num_rows, num_features), false);
        
        // Parse data
        for (row_idx, record) in records.iter().enumerate() {
            // Parse label
            let label_str = &record[target_col_idx];
            labels[row_idx] = self.parse_numeric_value(label_str)
                .ok_or_else(|| LightGBMError::data_loading(format!(
                    "Invalid target value '{}' at row {}", label_str, row_idx + 1
                )))?;
            
            // Parse features
            for (feat_idx, &col_idx) in feature_cols.iter().enumerate() {
                let value_str = &record[col_idx];
                if let Some(value) = self.parse_numeric_value(value_str) {
                    features[[row_idx, feat_idx]] = value;
                    missing_mask[[row_idx, feat_idx]] = false;
                } else {
                    // Handle missing value
                    features[[row_idx, feat_idx]] = f32::NAN;
                    missing_mask[[row_idx, feat_idx]] = true;
                }
            }
        }
        
        // Generate feature names
        let feature_names = if let Some(ref headers) = headers {
            feature_cols.iter().map(|&i| headers[i].to_string()).collect()
        } else {
            feature_cols.iter().map(|&i| format!("feature_{}", i)).collect()
        };
        
        // Create Dataset
        Dataset::new(
            features,
            labels,
            None,  // weights
            None,  // groups
            Some(feature_names),
            None,  // feature_types - will be inferred
        )
    }
    
    /// Parse numeric value with missing value detection
    fn parse_numeric_value(&self, value: &str) -> Option<f32> {
        let trimmed = value.trim();
        
        // Check for common missing value representations
        if trimmed.is_empty() 
            || trimmed.eq_ignore_ascii_case("na") 
            || trimmed.eq_ignore_ascii_case("nan") 
            || trimmed.eq_ignore_ascii_case("null") 
            || trimmed == "?" 
            || trimmed == "-" {
            return None;
        }
        
        // Try to parse as float
        trimmed.parse::<f32>().ok()
    }
    
    /// Infer data types from a sample of the data
    pub fn infer_types<P: AsRef<Path>>(&self, path: P, sample_rows: usize) -> Result<Vec<FeatureType>> {
        use csv::ReaderBuilder;
        use std::fs::File;
        
        let file = File::open(path)?;
        let mut reader = ReaderBuilder::new()
            .delimiter(self.config.delimiter as u8)
            .has_headers(self.config.skip_header)
            .from_reader(file);
        
        let mut sample_records = Vec::new();
        for (i, result) in reader.records().enumerate() {
            if i >= sample_rows {
                break;
            }
            sample_records.push(result?);
        }
        
        if sample_records.is_empty() {
            return Err(LightGBMError::data_loading("No data available for type inference"));
        }
        
        let num_cols = sample_records[0].len();
        let mut column_types = vec![FeatureType::Numerical; num_cols];
        
        // Analyze each column
        for col_idx in 0..num_cols {
            let mut numeric_count = 0;
            let mut total_count = 0;
            let mut unique_values = std::collections::HashSet::new();
            
            for record in &sample_records {
                let value = record[col_idx].trim();
                if !value.is_empty() {
                    total_count += 1;
                    unique_values.insert(value.to_string());
                    
                    if self.parse_numeric_value(value).is_some() {
                        numeric_count += 1;
                    }
                }
            }
            
            // Determine type based on analysis
            let numeric_ratio = if total_count > 0 {
                numeric_count as f64 / total_count as f64
            } else {
                0.0
            };
            
            // If less than 90% numeric or very few unique values, consider categorical
            if numeric_ratio < 0.9 || unique_values.len() <= 20 {
                column_types[col_idx] = FeatureType::Categorical;
            }
        }
        
        Ok(column_types)
    }
}

impl DataLoader for CsvLoader {
    fn load<P: AsRef<Path>>(&self, path: P) -> Result<Dataset> {
        self.load_csv(path)
    }
    
    fn config(&self) -> &LoaderConfig {
        &self.config
    }
}

/// Polars loader
#[cfg(feature = "polars")]
pub struct PolarsLoader {
    config: LoaderConfig,
}

#[cfg(feature = "polars")]
impl PolarsLoader {
    pub fn new(dataset_config: DatasetConfig) -> Result<Self> {
        Ok(PolarsLoader {
            config: LoaderConfig {
                dataset_config,
                ..Default::default()
            },
        })
    }
    
    /// Load from Polars DataFrame directly
    pub fn load_dataframe(&self, df: &polars::prelude::DataFrame) -> Result<Dataset> {
        use polars::prelude::*;
        
        log::info!("Loading from Polars DataFrame with shape: {:?}", df.shape());
        
        // Validate DataFrame is not empty
        if df.height() == 0 {
            return Err(LightGBMError::data_loading("DataFrame is empty"));
        }
        
        // Determine target column
        let target_col_name = if let Some(ref target_col) = self.config.dataset_config.target_column {
            target_col.clone()
        } else {
            // Default to last column
            df.get_column_names().last()
                .ok_or_else(|| LightGBMError::data_loading("No columns in DataFrame"))?
                .to_string()
        };
        
        // Get feature columns (all except target)
        let feature_columns: Vec<String> = df.get_column_names()
            .iter()
            .filter(|&&col| col != target_col_name.as_str())
            .map(|&col| col.to_string())
            .collect();
        
        if feature_columns.is_empty() {
            return Err(LightGBMError::data_loading("No feature columns found"));
        }
        
        // Extract labels
        let labels = self.extract_labels_from_dataframe(df, &target_col_name)?;
        
        // Extract features
        let features = self.extract_features_from_dataframe(df, &feature_columns)?;
        
        // Extract weights if specified
        let weights = if let Some(ref weight_col) = self.config.dataset_config.weight_column {
            Some(self.extract_weights_from_dataframe(df, weight_col)?)
        } else {
            None
        };
        
        // Create Dataset
        Dataset::new(
            features,
            labels,
            weights,
            None, // groups
            Some(feature_columns),
            None, // feature_types - will be inferred
        )
    }
    
    /// Load CSV file using Polars (much faster than standard CSV loader)
    pub fn load_csv<P: AsRef<Path>>(&self, path: P) -> Result<Dataset> {
        // TODO: Fix polars API compatibility
        Err(LightGBMError::not_implemented("Polars CSV loading temporarily disabled"))
    }
    
    /// Load Parquet file using Polars
    pub fn load_parquet<P: AsRef<Path>>(&self, path: P) -> Result<Dataset> {
        // TODO: Fix polars API compatibility
        Err(LightGBMError::not_implemented("Polars Parquet loading temporarily disabled"))
    }
    
    /// Extract labels from DataFrame
    fn extract_labels_from_dataframe(&self, df: &polars::prelude::DataFrame, target_col: &str) -> Result<Array1<f32>> {
        use polars::prelude::*;
        
        let series = df.column(target_col)
            .map_err(|e| LightGBMError::data_loading(format!("Target column '{}' not found: {}", target_col, e)))?;
        
        // Convert to f32 array
        let labels = match series.dtype() {
            DataType::Float32 => {
                let ca = series.f32()
                    .map_err(|e| LightGBMError::data_loading(format!("Failed to extract f32 labels: {}", e)))?;
                
                let mut labels_vec = Vec::with_capacity(ca.len());
                for opt_val in ca.iter() {
                    labels_vec.push(opt_val.unwrap_or(f32::NAN));
                }
                Array1::from_vec(labels_vec)
            },
            DataType::Float64 => {
                let ca = series.f64()
                    .map_err(|e| LightGBMError::data_loading(format!("Failed to extract f64 labels: {}", e)))?;
                
                let mut labels_vec = Vec::with_capacity(ca.len());
                for opt_val in ca.iter() {
                    labels_vec.push(opt_val.unwrap_or(f64::NAN) as f32);
                }
                Array1::from_vec(labels_vec)
            },
            DataType::Int32 => {
                let ca = series.i32()
                    .map_err(|e| LightGBMError::data_loading(format!("Failed to extract i32 labels: {}", e)))?;
                
                let mut labels_vec = Vec::with_capacity(ca.len());
                for opt_val in ca.iter() {
                    labels_vec.push(opt_val.unwrap_or(0) as f32);
                }
                Array1::from_vec(labels_vec)
            },
            DataType::Int64 => {
                let ca = series.i64()
                    .map_err(|e| LightGBMError::data_loading(format!("Failed to extract i64 labels: {}", e)))?;
                
                let mut labels_vec = Vec::with_capacity(ca.len());
                for opt_val in ca.iter() {
                    labels_vec.push(opt_val.unwrap_or(0) as f32);
                }
                Array1::from_vec(labels_vec)
            },
            _ => {
                return Err(LightGBMError::data_loading(format!(
                    "Unsupported target column type: {:?}", series.dtype()
                )));
            }
        };
        
        Ok(labels)
    }
    
    /// Extract features from DataFrame
    fn extract_features_from_dataframe(&self, df: &polars::prelude::DataFrame, feature_cols: &[String]) -> Result<Array2<f32>> {
        use polars::prelude::*;
        
        let num_rows = df.height();
        let num_features = feature_cols.len();
        let mut features = Array2::<f32>::zeros((num_rows, num_features));
        
        for (feat_idx, col_name) in feature_cols.iter().enumerate() {
            let series = df.column(col_name)
                .map_err(|e| LightGBMError::data_loading(format!("Feature column '{}' not found: {}", col_name, e)))?;
            
            // Convert series to f32 values
            let values = match series.dtype() {
                DataType::Float32 => {
                    let ca = series.f32()
                        .map_err(|e| LightGBMError::data_loading(format!("Failed to extract f32 from {}: {}", col_name, e)))?;
                    
                    ca.iter().map(|opt_val| opt_val.unwrap_or(f32::NAN)).collect::<Vec<f32>>()
                },
                DataType::Float64 => {
                    let ca = series.f64()
                        .map_err(|e| LightGBMError::data_loading(format!("Failed to extract f64 from {}: {}", col_name, e)))?;
                    
                    ca.iter().map(|opt_val| opt_val.unwrap_or(f64::NAN) as f32).collect::<Vec<f32>>()
                },
                DataType::Int32 => {
                    let ca = series.i32()
                        .map_err(|e| LightGBMError::data_loading(format!("Failed to extract i32 from {}: {}", col_name, e)))?;
                    
                    ca.iter().map(|opt_val| opt_val.unwrap_or(0) as f32).collect::<Vec<f32>>()
                },
                DataType::Int64 => {
                    let ca = series.i64()
                        .map_err(|e| LightGBMError::data_loading(format!("Failed to extract i64 from {}: {}", col_name, e)))?;
                    
                    ca.iter().map(|opt_val| opt_val.unwrap_or(0) as f32).collect::<Vec<f32>>()
                },
                DataType::Boolean => {
                    let ca = series.bool()
                        .map_err(|e| LightGBMError::data_loading(format!("Failed to extract bool from {}: {}", col_name, e)))?;
                    
                    ca.iter().map(|opt_val| if opt_val.unwrap_or(false) { 1.0 } else { 0.0 }).collect::<Vec<f32>>()
                },
                DataType::String => {
                    // Handle categorical string data - for now, just error
                    return Err(LightGBMError::data_loading(format!(
                        "String column '{}' requires encoding. Use preprocessing module first.", col_name
                    )));
                },
                _ => {
                    return Err(LightGBMError::data_loading(format!(
                        "Unsupported feature column type for '{}': {:?}", col_name, series.dtype()
                    )));
                }
            };
            
            // Copy values to features array
            for (row_idx, &value) in values.iter().enumerate() {
                features[[row_idx, feat_idx]] = value;
            }
        }
        
        Ok(features)
    }
    
    /// Extract weights from DataFrame
    fn extract_weights_from_dataframe(&self, df: &polars::prelude::DataFrame, weight_col: &str) -> Result<Array1<f32>> {
        use polars::prelude::*;
        
        let series = df.column(weight_col)
            .map_err(|e| LightGBMError::data_loading(format!("Weight column '{}' not found: {}", weight_col, e)))?;
        
        // Convert to f32 array (similar to labels)
        let weights = match series.dtype() {
            DataType::Float32 => {
                let ca = series.f32()
                    .map_err(|e| LightGBMError::data_loading(format!("Failed to extract f32 weights: {}", e)))?;
                
                let mut weights_vec = Vec::with_capacity(ca.len());
                for opt_val in ca.iter() {
                    weights_vec.push(opt_val.unwrap_or(1.0));
                }
                Array1::from_vec(weights_vec)
            },
            DataType::Float64 => {
                let ca = series.f64()
                    .map_err(|e| LightGBMError::data_loading(format!("Failed to extract f64 weights: {}", e)))?;
                
                let mut weights_vec = Vec::with_capacity(ca.len());
                for opt_val in ca.iter() {
                    weights_vec.push(opt_val.unwrap_or(1.0) as f32);
                }
                Array1::from_vec(weights_vec)
            },
            _ => {
                return Err(LightGBMError::data_loading(format!(
                    "Unsupported weight column type: {:?}", series.dtype()
                )));
            }
        };
        
        Ok(weights)
    }
}

#[cfg(feature = "polars")]
impl DataLoader for PolarsLoader {
    fn load<P: AsRef<Path>>(&self, path: P) -> Result<Dataset> {
        let path_str = path.as_ref().to_string_lossy();
        
        // Determine file type and use appropriate loader
        if path_str.ends_with(".csv") || path_str.ends_with(".tsv") {
            self.load_csv(path)
        } else if path_str.ends_with(".parquet") {
            self.load_parquet(path)
        } else {
            Err(LightGBMError::data_loading(format!(
                "Unsupported file format for Polars loader: {}", path_str
            )))
        }
    }
    
    fn config(&self) -> &LoaderConfig {
        &self.config
    }
}

/// Arrow loader
#[cfg(feature = "arrow")]
pub struct ArrowLoader {
    config: LoaderConfig,
}

#[cfg(feature = "arrow")]
impl ArrowLoader {
    pub fn new(dataset_config: DatasetConfig) -> Result<Self> {
        Ok(ArrowLoader {
            config: LoaderConfig {
                dataset_config,
                ..Default::default()
            },
        })
    }
    
    pub fn load_table(&self, _table: &arrow::array::RecordBatch) -> Result<Dataset> {
        // TODO: Implement Arrow loading
        Err(LightGBMError::not_implemented("Arrow loading"))
    }
}

#[cfg(feature = "arrow")]
impl DataLoader for ArrowLoader {
    fn load<P: AsRef<Path>>(&self, _path: P) -> Result<Dataset> {
        // TODO: Implement Arrow file loading
        Err(LightGBMError::not_implemented("Arrow file loading"))
    }
    
    fn config(&self) -> &LoaderConfig {
        &self.config
    }
}

/// Parquet loader
#[cfg(feature = "parquet")]
pub struct ParquetLoader {
    config: LoaderConfig,
}

#[cfg(feature = "parquet")]
impl ParquetLoader {
    pub fn new(dataset_config: DatasetConfig) -> Result<Self> {
        Ok(ParquetLoader {
            config: LoaderConfig {
                dataset_config,
                ..Default::default()
            },
        })
    }
}

#[cfg(feature = "parquet")]
impl DataLoader for ParquetLoader {
    fn load<P: AsRef<Path>>(&self, _path: P) -> Result<Dataset> {
        // TODO: Implement Parquet loading
        Err(LightGBMError::not_implemented("Parquet loading"))
    }
    
    fn config(&self) -> &LoaderConfig {
        &self.config
    }
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
        log::info!("Loading dataset from arrays: {}x{}", features.nrows(), features.ncols());
        
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
            Some((0..features.ncols())
                .map(|i| format!("feature_{}", i))
                .collect())
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
        
        // Set metadata
        let metadata = dataset.metadata_mut();
        metadata.source_path = Some("In-memory arrays".to_string());
        metadata.format = "arrays".to_string();
        metadata.properties.insert("source".to_string(), "memory".to_string());
        
        Ok(dataset)
    }
    
    /// Load dataset from separate feature and label vectors
    pub fn load_from_vecs(
        &self,
        features_vec: Vec<Vec<f32>>,
        labels_vec: Vec<f32>,
        weights_vec: Option<Vec<f32>>,
        feature_names: Option<Vec<String>>,
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
        
        let features = Array2::from_shape_vec((num_rows, num_cols), features_flat)
            .map_err(|e| LightGBMError::data_loading(format!("Failed to create features array: {}", e)))?;
        
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
        
        let features = Array2::from_shape_vec((num_rows, num_cols), features_flat)
            .map_err(|e| LightGBMError::data_loading(format!("Failed to create features array: {}", e)))?;
        
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
        
        self.load_from_flat(
            features_flat,
            num_samples,
            num_features,
            labels,
            None,
        )
    }
}

impl DataLoader for ArrayLoader {
    fn load<P: AsRef<Path>>(&self, _path: P) -> Result<Dataset> {
        Err(LightGBMError::not_implemented("Array loader does not support file loading - use load_arrays() instead"))
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
        // TODO: Implement binary loading
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
        // TODO: Implement memory-mapped loading
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
    fn test_csv_loader_creation() {
        let dataset_config = DatasetConfig::default();
        let loader = CsvLoader::new(dataset_config).unwrap();
        assert_eq!(loader.config().delimiter, ',');
    }
    
    #[test]
    fn test_array_loader_creation() {
        let dataset_config = DatasetConfig::default();
        let loader = ArrayLoader::new(dataset_config).unwrap();
        assert_eq!(loader.config().chunk_size, 10000);
    }
}