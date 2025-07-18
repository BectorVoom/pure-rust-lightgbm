//! Polars DataFrame loader for Pure Rust LightGBM.
//!
//! This module provides efficient data loading from Polars DataFrames with support
//! for various data types, lazy evaluation, and optimized memory usage.

use crate::core::types::*;
use crate::core::error::{Result, LightGBMError};
use crate::dataset::{Dataset, DatasetConfig};
use super::{DataLoader, LoaderConfig};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[cfg(feature = "polars")]
use polars::prelude::*;

/// Polars-specific configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolarsConfig {
    /// Target column name
    pub target_column: String,
    /// Feature columns to include (None = all except target)
    pub feature_columns: Option<Vec<String>>,
    /// Weight column name
    pub weight_column: Option<String>,
    /// Group column for ranking
    pub group_column: Option<String>,
    /// Enable lazy evaluation
    pub lazy_evaluation: bool,
    /// Chunk size for processing large DataFrames
    pub chunk_size: Option<usize>,
    /// Maximum memory usage (MB)
    pub max_memory_mb: Option<usize>,
    /// Enable parallel processing
    pub parallel: bool,
    /// Projection pushdown optimization
    pub projection_pushdown: bool,
    /// Predicate pushdown optimization
    pub predicate_pushdown: bool,
}

impl Default for PolarsConfig {
    fn default() -> Self {
        PolarsConfig {
            target_column: "target".to_string(),
            feature_columns: None,
            weight_column: None,
            group_column: None,
            lazy_evaluation: true,
            chunk_size: Some(10000),
            max_memory_mb: None,
            parallel: true,
            projection_pushdown: true,
            predicate_pushdown: true,
        }
    }
}

/// Polars data loader
#[cfg(feature = "polars")]
pub struct PolarsLoader {
    /// Loader configuration
    config: LoaderConfig,
    /// Polars-specific configuration
    polars_config: PolarsConfig,
    /// Dataset configuration
    dataset_config: DatasetConfig,
}

#[cfg(feature = "polars")]
impl PolarsLoader {
    /// Create a new Polars loader
    pub fn new(dataset_config: DatasetConfig) -> Result<Self> {
        let polars_config = PolarsConfig {
            target_column: dataset_config.target_column.clone()
                .unwrap_or_else(|| "target".to_string()),
            feature_columns: dataset_config.feature_columns.clone(),
            weight_column: dataset_config.weight_column.clone(),
            group_column: dataset_config.group_column.clone(),
            ..Default::default()
        };

        Ok(PolarsLoader {
            config: LoaderConfig {
                dataset_config: dataset_config.clone(),
                ..Default::default()
            },
            polars_config,
            dataset_config,
        })
    }

    /// Create Polars loader with custom configuration
    pub fn with_polars_config(mut self, polars_config: PolarsConfig) -> Self {
        self.polars_config = polars_config;
        self
    }

    /// Set target column
    pub fn with_target_column<S: Into<String>>(mut self, target: S) -> Self {
        self.polars_config.target_column = target.into();
        self
    }

    /// Set feature columns
    pub fn with_feature_columns(mut self, features: Vec<String>) -> Self {
        self.polars_config.feature_columns = Some(features);
        self
    }

    /// Set weight column
    pub fn with_weight_column<S: Into<String>>(mut self, weight: S) -> Self {
        self.polars_config.weight_column = Some(weight.into());
        self
    }

    /// Enable lazy evaluation
    pub fn with_lazy_evaluation(mut self, lazy: bool) -> Self {
        self.polars_config.lazy_evaluation = lazy;
        self
    }

    /// Set chunk size for processing
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.polars_config.chunk_size = Some(chunk_size);
        self
    }

    /// Enable parallel processing
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.polars_config.parallel = parallel;
        self
    }

    /// Load from Polars DataFrame directly
    pub fn load_dataframe(&self, df: &DataFrame) -> Result<Dataset> {
        log::info!("Loading from Polars DataFrame with shape: {:?}", df.shape());

        // Validate DataFrame
        self.validate_dataframe(df)?;

        // Get column information
        let target_col_name = &self.polars_config.target_column;
        let feature_columns = self.determine_feature_columns(df)?;

        // Extract components
        let labels = self.extract_labels_from_dataframe(df, target_col_name)?;
        let features = self.extract_features_from_dataframe(df, &feature_columns)?;
        let weights = self.extract_weights_from_dataframe(df)?;
        let groups = self.extract_groups_from_dataframe(df)?;

        // Detect missing values
        let missing_values = self.detect_missing_values_from_dataframe(df, &feature_columns)?;

        // Create Dataset
        let mut dataset = Dataset::new(
            features,
            labels,
            weights,
            groups,
            Some(feature_columns),
            None, // feature_types - will be auto-detected
        )?;

        // Set missing values if any detected
        if let Some(missing_mask) = missing_values {
            dataset.set_missing_values(missing_mask)?;
        }

        // Set metadata
        let metadata = dataset.metadata_mut();
        metadata.source_path = Some("Polars DataFrame".to_string());
        metadata.format = "polars".to_string();
        metadata.properties.insert("num_rows".to_string(), df.height().to_string());
        metadata.properties.insert("num_cols".to_string(), df.width().to_string());

        Ok(dataset)
    }

    /// Load CSV file using Polars (faster than standard CSV loader)
    pub fn load_csv<P: AsRef<Path>>(&self, path: P) -> Result<Dataset> {
        let path = path.as_ref();
        log::info!("Loading CSV with Polars: {}", path.display());

        // Configure CSV read options
        let mut csv_read_options = CsvReadOptions::default()
            .with_has_header(true)
            .with_separator(b',')
            .with_quote_char(Some(b'"'))
            .with_null_values(Some(NullValues::AllColumnsSingle("".to_string())));

        // Apply configuration
        if let Some(max_rows) = self.config.max_rows {
            csv_read_options = csv_read_options.with_n_rows(Some(max_rows));
        }

        // Create LazyFrame for optimized reading
        let lazy_frame = if self.polars_config.lazy_evaluation {
            LazyFrame::scan_csv(path, csv_read_options)
                .map_err(|e| LightGBMError::data_loading(format!("Polars CSV scan error: {}", e)))?
        } else {
            // Direct read for smaller files
            let df = CsvReader::from_path(path)
                .map_err(|e| LightGBMError::data_loading(format!("Polars CSV reader error: {}", e)))?
                .has_header(true)
                .with_separator(b',')
                .finish()
                .map_err(|e| LightGBMError::data_loading(format!("Polars CSV read error: {}", e)))?;
            
            return self.load_dataframe(&df);
        };

        // Apply optimizations
        let optimized_lazy_frame = self.apply_optimizations(lazy_frame)?;

        // Collect DataFrame
        let df = optimized_lazy_frame
            .collect()
            .map_err(|e| LightGBMError::data_loading(format!("Polars DataFrame collection error: {}", e)))?;

        // Convert to Dataset
        self.load_dataframe(&df)
    }

    /// Load Parquet file using Polars
    pub fn load_parquet<P: AsRef<Path>>(&self, path: P) -> Result<Dataset> {
        let path = path.as_ref();
        log::info!("Loading Parquet with Polars: {}", path.display());

        let mut scan_args = ScanArgsParquet::default();
        
        // Apply row limit if specified
        if let Some(max_rows) = self.config.max_rows {
            scan_args = scan_args.with_n_rows(Some(max_rows));
        }

        // Create LazyFrame for optimized reading
        let lazy_frame = if self.polars_config.lazy_evaluation {
            LazyFrame::scan_parquet(path, scan_args)
                .map_err(|e| LightGBMError::data_loading(format!("Polars Parquet scan error: {}", e)))?
        } else {
            // Direct read
            let df = LazyFrame::scan_parquet(path, scan_args)
                .map_err(|e| LightGBMError::data_loading(format!("Polars Parquet scan error: {}", e)))?
                .collect()
                .map_err(|e| LightGBMError::data_loading(format!("Polars Parquet read error: {}", e)))?;
            
            return self.load_dataframe(&df);
        };

        // Apply optimizations
        let optimized_lazy_frame = self.apply_optimizations(lazy_frame)?;

        // Collect DataFrame
        let df = optimized_lazy_frame
            .collect()
            .map_err(|e| LightGBMError::data_loading(format!("Polars DataFrame collection error: {}", e)))?;

        // Convert to Dataset
        self.load_dataframe(&df)
    }

    /// Apply Polars optimizations to LazyFrame
    fn apply_optimizations(&self, lazy_frame: LazyFrame) -> Result<LazyFrame> {
        let mut optimized = lazy_frame;

        // Apply column selection if specified
        if let Some(ref feature_cols) = self.polars_config.feature_columns {
            let mut columns_to_select = feature_cols.clone();
            columns_to_select.push(self.polars_config.target_column.clone());
            
            if let Some(ref weight_col) = self.polars_config.weight_column {
                columns_to_select.push(weight_col.clone());
            }
            
            if let Some(ref group_col) = self.polars_config.group_column {
                columns_to_select.push(group_col.clone());
            }

            optimized = optimized.select([cols(columns_to_select)]);
        }

        // Apply memory limit if specified
        if let Some(memory_limit) = self.polars_config.max_memory_mb {
            // Polars doesn't have a direct memory limit, but we can add streaming
            // for large datasets to reduce memory usage
            if memory_limit < 1024 { // Less than 1GB, use streaming
                optimized = optimized.with_streaming(true);
            }
        }

        Ok(optimized)
    }

    /// Validate DataFrame before processing
    fn validate_dataframe(&self, df: &DataFrame) -> Result<()> {
        if df.height() == 0 {
            return Err(LightGBMError::data_loading("DataFrame is empty"));
        }

        if df.width() == 0 {
            return Err(LightGBMError::data_loading("DataFrame has no columns"));
        }

        // Check if target column exists
        if !df.get_column_names().contains(&self.polars_config.target_column.as_str()) {
            return Err(LightGBMError::data_loading(format!(
                "Target column '{}' not found in DataFrame", 
                self.polars_config.target_column
            )));
        }

        // Check if specified feature columns exist
        if let Some(ref feature_cols) = self.polars_config.feature_columns {
            let df_columns: std::collections::HashSet<&str> = df.get_column_names().into_iter().collect();
            for feature_col in feature_cols {
                if !df_columns.contains(feature_col.as_str()) {
                    return Err(LightGBMError::data_loading(format!(
                        "Feature column '{}' not found in DataFrame", 
                        feature_col
                    )));
                }
            }
        }

        // Check if weight column exists
        if let Some(ref weight_col) = self.polars_config.weight_column {
            if !df.get_column_names().contains(&weight_col.as_str()) {
                return Err(LightGBMError::data_loading(format!(
                    "Weight column '{}' not found in DataFrame", 
                    weight_col
                )));
            }
        }

        // Check if group column exists
        if let Some(ref group_col) = self.polars_config.group_column {
            if !df.get_column_names().contains(&group_col.as_str()) {
                return Err(LightGBMError::data_loading(format!(
                    "Group column '{}' not found in DataFrame", 
                    group_col
                )));
            }
        }

        Ok(())
    }

    /// Determine feature columns
    fn determine_feature_columns(&self, df: &DataFrame) -> Result<Vec<String>> {
        if let Some(ref feature_cols) = self.polars_config.feature_columns {
            Ok(feature_cols.clone())
        } else {
            // Use all columns except target, weight, and group columns
            let mut excluded_columns = std::collections::HashSet::new();
            excluded_columns.insert(self.polars_config.target_column.as_str());
            
            if let Some(ref weight_col) = self.polars_config.weight_column {
                excluded_columns.insert(weight_col.as_str());
            }
            
            if let Some(ref group_col) = self.polars_config.group_column {
                excluded_columns.insert(group_col.as_str());
            }

            let feature_columns: Vec<String> = df.get_column_names()
                .into_iter()
                .filter(|&col| !excluded_columns.contains(col))
                .map(|col| col.to_string())
                .collect();

            if feature_columns.is_empty() {
                return Err(LightGBMError::data_loading("No feature columns found"));
            }

            Ok(feature_columns)
        }
    }

    /// Extract labels from DataFrame
    fn extract_labels_from_dataframe(&self, df: &DataFrame, target_col: &str) -> Result<Array1<f32>> {
        let series = df.column(target_col)
            .map_err(|e| LightGBMError::data_loading(format!("Target column '{}' not found: {}", target_col, e)))?;

        self.series_to_f32_array(series, target_col)
    }

    /// Extract features from DataFrame
    fn extract_features_from_dataframe(&self, df: &DataFrame, feature_cols: &[String]) -> Result<Array2<f32>> {
        let num_rows = df.height();
        let num_features = feature_cols.len();
        let mut features = Array2::<f32>::zeros((num_rows, num_features));

        for (feat_idx, col_name) in feature_cols.iter().enumerate() {
            let series = df.column(col_name)
                .map_err(|e| LightGBMError::data_loading(format!("Feature column '{}' not found: {}", col_name, e)))?;

            let values = self.series_to_f32_array(series, col_name)?;
            
            for (row_idx, &value) in values.iter().enumerate() {
                features[[row_idx, feat_idx]] = value;
            }
        }

        Ok(features)
    }

    /// Extract weights from DataFrame
    fn extract_weights_from_dataframe(&self, df: &DataFrame) -> Result<Option<Array1<f32>>> {
        if let Some(ref weight_col) = self.polars_config.weight_column {
            let series = df.column(weight_col)
                .map_err(|e| LightGBMError::data_loading(format!("Weight column '{}' not found: {}", weight_col, e)))?;

            let weights = self.series_to_f32_array(series, weight_col)?;
            Ok(Some(weights))
        } else {
            Ok(None)
        }
    }

    /// Extract groups from DataFrame
    fn extract_groups_from_dataframe(&self, df: &DataFrame) -> Result<Option<Array1<DataSize>>> {
        if let Some(ref group_col) = self.polars_config.group_column {
            let series = df.column(group_col)
                .map_err(|e| LightGBMError::data_loading(format!("Group column '{}' not found: {}", group_col, e)))?;

            let groups = self.series_to_datasize_array(series, group_col)?;
            Ok(Some(groups))
        } else {
            Ok(None)
        }
    }

    /// Detect missing values in DataFrame
    fn detect_missing_values_from_dataframe(&self, df: &DataFrame, feature_cols: &[String]) -> Result<Option<Array2<bool>>> {
        let num_rows = df.height();
        let num_features = feature_cols.len();
        let mut missing_mask = Array2::<bool>::zeros((num_rows, num_features));
        let mut has_missing = false;

        for (feat_idx, col_name) in feature_cols.iter().enumerate() {
            let series = df.column(col_name)
                .map_err(|e| LightGBMError::data_loading(format!("Feature column '{}' not found: {}", col_name, e)))?;

            // Check for null values in the series
            if series.null_count() > 0 {
                has_missing = true;
                
                // Mark missing values
                for (row_idx, is_null) in series.is_null().bool()
                    .map_err(|e| LightGBMError::data_loading(format!("Failed to check nulls in {}: {}", col_name, e)))?
                    .into_iter()
                    .enumerate() {
                    missing_mask[[row_idx, feat_idx]] = is_null.unwrap_or(false);
                }
            }
        }

        if has_missing {
            Ok(Some(missing_mask))
        } else {
            Ok(None)
        }
    }

    /// Convert Polars Series to f32 Array1
    fn series_to_f32_array(&self, series: &Series, col_name: &str) -> Result<Array1<f32>> {
        let values = match series.dtype() {
            DataType::Float32 => {
                let ca = series.f32()
                    .map_err(|e| LightGBMError::data_loading(format!("Failed to extract f32 from {}: {}", col_name, e)))?;
                
                ca.into_iter().map(|opt_val| opt_val.unwrap_or(f32::NAN)).collect::<Vec<f32>>()
            },
            DataType::Float64 => {
                let ca = series.f64()
                    .map_err(|e| LightGBMError::data_loading(format!("Failed to extract f64 from {}: {}", col_name, e)))?;
                
                ca.into_iter().map(|opt_val| opt_val.unwrap_or(f64::NAN) as f32).collect::<Vec<f32>>()
            },
            DataType::Int8 => {
                let ca = series.i8()
                    .map_err(|e| LightGBMError::data_loading(format!("Failed to extract i8 from {}: {}", col_name, e)))?;
                
                ca.into_iter().map(|opt_val| opt_val.unwrap_or(0) as f32).collect::<Vec<f32>>()
            },
            DataType::Int16 => {
                let ca = series.i16()
                    .map_err(|e| LightGBMError::data_loading(format!("Failed to extract i16 from {}: {}", col_name, e)))?;
                
                ca.into_iter().map(|opt_val| opt_val.unwrap_or(0) as f32).collect::<Vec<f32>>()
            },
            DataType::Int32 => {
                let ca = series.i32()
                    .map_err(|e| LightGBMError::data_loading(format!("Failed to extract i32 from {}: {}", col_name, e)))?;
                
                ca.into_iter().map(|opt_val| opt_val.unwrap_or(0) as f32).collect::<Vec<f32>>()
            },
            DataType::Int64 => {
                let ca = series.i64()
                    .map_err(|e| LightGBMError::data_loading(format!("Failed to extract i64 from {}: {}", col_name, e)))?;
                
                ca.into_iter().map(|opt_val| opt_val.unwrap_or(0) as f32).collect::<Vec<f32>>()
            },
            DataType::UInt8 => {
                let ca = series.u8()
                    .map_err(|e| LightGBMError::data_loading(format!("Failed to extract u8 from {}: {}", col_name, e)))?;
                
                ca.into_iter().map(|opt_val| opt_val.unwrap_or(0) as f32).collect::<Vec<f32>>()
            },
            DataType::UInt16 => {
                let ca = series.u16()
                    .map_err(|e| LightGBMError::data_loading(format!("Failed to extract u16 from {}: {}", col_name, e)))?;
                
                ca.into_iter().map(|opt_val| opt_val.unwrap_or(0) as f32).collect::<Vec<f32>>()
            },
            DataType::UInt32 => {
                let ca = series.u32()
                    .map_err(|e| LightGBMError::data_loading(format!("Failed to extract u32 from {}: {}", col_name, e)))?;
                
                ca.into_iter().map(|opt_val| opt_val.unwrap_or(0) as f32).collect::<Vec<f32>>()
            },
            DataType::UInt64 => {
                let ca = series.u64()
                    .map_err(|e| LightGBMError::data_loading(format!("Failed to extract u64 from {}: {}", col_name, e)))?;
                
                ca.into_iter().map(|opt_val| opt_val.unwrap_or(0) as f32).collect::<Vec<f32>>()
            },
            DataType::Boolean => {
                let ca = series.bool()
                    .map_err(|e| LightGBMError::data_loading(format!("Failed to extract bool from {}: {}", col_name, e)))?;
                
                ca.into_iter().map(|opt_val| if opt_val.unwrap_or(false) { 1.0 } else { 0.0 }).collect::<Vec<f32>>()
            },
            DataType::String => {
                return Err(LightGBMError::data_loading(format!(
                    "String column '{}' requires encoding. Use preprocessing module first.", col_name
                )));
            },
            _ => {
                return Err(LightGBMError::data_loading(format!(
                    "Unsupported data type for column '{}': {:?}", col_name, series.dtype()
                )));
            }
        };

        Ok(Array1::from_vec(values))
    }

    /// Convert Polars Series to DataSize Array1
    fn series_to_datasize_array(&self, series: &Series, col_name: &str) -> Result<Array1<DataSize>> {
        let values = match series.dtype() {
            DataType::Int32 => {
                let ca = series.i32()
                    .map_err(|e| LightGBMError::data_loading(format!("Failed to extract i32 from {}: {}", col_name, e)))?;
                
                ca.into_iter().map(|opt_val| opt_val.unwrap_or(0)).collect::<Vec<DataSize>>()
            },
            DataType::Int64 => {
                let ca = series.i64()
                    .map_err(|e| LightGBMError::data_loading(format!("Failed to extract i64 from {}: {}", col_name, e)))?;
                
                ca.into_iter().map(|opt_val| opt_val.unwrap_or(0) as DataSize).collect::<Vec<DataSize>>()
            },
            DataType::UInt32 => {
                let ca = series.u32()
                    .map_err(|e| LightGBMError::data_loading(format!("Failed to extract u32 from {}: {}", col_name, e)))?;
                
                ca.into_iter().map(|opt_val| opt_val.unwrap_or(0) as DataSize).collect::<Vec<DataSize>>()
            },
            DataType::UInt64 => {
                let ca = series.u64()
                    .map_err(|e| LightGBMError::data_loading(format!("Failed to extract u64 from {}: {}", col_name, e)))?;
                
                ca.into_iter().map(|opt_val| opt_val.unwrap_or(0) as DataSize).collect::<Vec<DataSize>>()
            },
            _ => {
                return Err(LightGBMError::data_loading(format!(
                    "Unsupported data type for group column '{}': {:?}", col_name, series.dtype()
                )));
            }
        };

        Ok(Array1::from_vec(values))
    }

    /// Get configuration
    pub fn polars_config(&self) -> &PolarsConfig {
        &self.polars_config
    }

    /// Estimate memory usage for DataFrame
    pub fn estimate_memory_usage(&self, df: &DataFrame) -> PolarsMemoryEstimate {
        let num_rows = df.height();
        let num_cols = df.width();
        
        // Estimate based on column types
        let mut estimated_bytes = 0;
        for column in df.get_columns() {
            let column_bytes = match column.dtype() {
                DataType::Boolean => num_rows,
                DataType::UInt8 | DataType::Int8 => num_rows,
                DataType::UInt16 | DataType::Int16 => num_rows * 2,
                DataType::UInt32 | DataType::Int32 | DataType::Float32 => num_rows * 4,
                DataType::UInt64 | DataType::Int64 | DataType::Float64 => num_rows * 8,
                DataType::String => num_rows * 20, // Rough estimate for strings
                _ => num_rows * 8, // Default estimate
            };
            estimated_bytes += column_bytes;
        }

        // Add overhead for Dataset conversion
        let dataset_overhead = num_rows * num_cols * std::mem::size_of::<f32>();
        let total_estimated = estimated_bytes + dataset_overhead;

        PolarsMemoryEstimate {
            dataframe_bytes: estimated_bytes,
            dataset_bytes: dataset_overhead,
            total_estimated,
            num_rows,
            num_cols,
        }
    }
}

/// Memory usage estimate for Polars DataFrame loading
#[derive(Debug, Clone)]
pub struct PolarsMemoryEstimate {
    /// Estimated DataFrame memory usage
    pub dataframe_bytes: usize,
    /// Estimated Dataset memory usage after conversion
    pub dataset_bytes: usize,
    /// Total estimated memory usage
    pub total_estimated: usize,
    /// Number of rows
    pub num_rows: usize,
    /// Number of columns
    pub num_cols: usize,
}

#[cfg(feature = "polars")]
impl DataLoader for PolarsLoader {
    fn load<P: AsRef<Path>>(&self, path: P) -> Result<Dataset> {
        let path_str = path.as_ref().to_string_lossy();
        
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

// Stub implementation when polars feature is not enabled
#[cfg(not(feature = "polars"))]
pub struct PolarsLoader;

#[cfg(not(feature = "polars"))]
impl PolarsLoader {
    pub fn new(_dataset_config: DatasetConfig) -> Result<Self> {
        Err(LightGBMError::not_implemented(
            "Polars support requires 'polars' feature to be enabled"
        ))
    }
}

#[cfg(not(feature = "polars"))]
impl DataLoader for PolarsLoader {
    fn load<P: AsRef<Path>>(&self, _path: P) -> Result<Dataset> {
        Err(LightGBMError::not_implemented(
            "Polars support requires 'polars' feature to be enabled"
        ))
    }

    fn config(&self) -> &LoaderConfig {
        unreachable!("PolarsLoader cannot be created without polars feature")
    }
}

#[cfg(test)]
#[cfg(feature = "polars")]
mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_polars_config_default() {
        let config = PolarsConfig::default();
        assert_eq!(config.target_column, "target");
        assert!(config.lazy_evaluation);
        assert!(config.parallel);
        assert!(config.projection_pushdown);
        assert!(config.predicate_pushdown);
    }

    #[test]
    fn test_polars_loader_creation() -> Result<()> {
        let dataset_config = DatasetConfig::new()
            .with_target_column("target");
        let loader = PolarsLoader::new(dataset_config)?;
        assert_eq!(loader.polars_config().target_column, "target");
        Ok(())
    }

    #[test]
    fn test_polars_loader_configuration() -> Result<()> {
        let dataset_config = DatasetConfig::default();
        let loader = PolarsLoader::new(dataset_config)?
            .with_target_column("label")
            .with_feature_columns(vec!["feat1".to_string(), "feat2".to_string()])
            .with_weight_column("weight")
            .with_lazy_evaluation(false)
            .with_chunk_size(5000)
            .with_parallel(false);

        assert_eq!(loader.polars_config().target_column, "label");
        assert_eq!(loader.polars_config().feature_columns, Some(vec!["feat1".to_string(), "feat2".to_string()]));
        assert_eq!(loader.polars_config().weight_column, Some("weight".to_string()));
        assert!(!loader.polars_config().lazy_evaluation);
        assert_eq!(loader.polars_config().chunk_size, Some(5000));
        assert!(!loader.polars_config().parallel);
        Ok(())
    }

    #[test]
    fn test_polars_dataframe_loading() -> Result<()> {
        // Create a test DataFrame
        let df = df! [
            "feature1" => [1.0f64, 2.0, 3.0, 4.0],
            "feature2" => [10.0f64, 20.0, 30.0, 40.0],
            "target" => [0i32, 1, 0, 1],
        ].map_err(|e| LightGBMError::data_loading(format!("Failed to create test DataFrame: {}", e)))?;

        // Create loader and load DataFrame
        let dataset_config = DatasetConfig::new()
            .with_target_column("target");
        let loader = PolarsLoader::new(dataset_config)?;
        let dataset = loader.load_dataframe(&df)?;

        // Verify dataset properties
        assert_eq!(dataset.num_data(), 4);
        assert_eq!(dataset.num_features(), 2);
        assert_eq!(dataset.feature_names().unwrap().len(), 2);
        assert_eq!(dataset.feature_names().unwrap()[0], "feature1");
        assert_eq!(dataset.feature_names().unwrap()[1], "feature2");

        // Verify data
        let features = dataset.features();
        assert_eq!(features[[0, 0]], 1.0);
        assert_eq!(features[[0, 1]], 10.0);
        assert_eq!(features[[1, 0]], 2.0);
        assert_eq!(features[[1, 1]], 20.0);

        let labels = dataset.labels();
        assert_eq!(labels[0], 0.0);
        assert_eq!(labels[1], 1.0);
        assert_eq!(labels[2], 0.0);
        assert_eq!(labels[3], 1.0);

        Ok(())
    }

    #[test]
    fn test_polars_with_missing_values() -> Result<()> {
        // Create a test DataFrame with missing values
        let mut df = df! [
            "feature1" => [Some(1.0f64), None, Some(3.0), Some(4.0)],
            "feature2" => [Some(10.0f64), Some(20.0), None, Some(40.0)],
            "target" => [0i32, 1, 0, 1],
        ].map_err(|e| LightGBMError::data_loading(format!("Failed to create test DataFrame: {}", e)))?;

        // Create loader and load DataFrame
        let dataset_config = DatasetConfig::new()
            .with_target_column("target");
        let loader = PolarsLoader::new(dataset_config)?;
        let dataset = loader.load_dataframe(&df)?;

        // Verify missing values are detected
        assert_eq!(dataset.num_data(), 4);
        assert_eq!(dataset.num_features(), 2);
        assert!(dataset.has_missing_values());

        let features = dataset.features();
        assert_eq!(features[[0, 0]], 1.0);
        assert!(features[[1, 0]].is_nan());
        assert_eq!(features[[2, 0]], 3.0);
        assert_eq!(features[[3, 0]], 4.0);

        assert_eq!(features[[0, 1]], 10.0);
        assert_eq!(features[[1, 1]], 20.0);
        assert!(features[[2, 1]].is_nan());
        assert_eq!(features[[3, 1]], 40.0);

        Ok(())
    }

    #[test]
    fn test_memory_estimation() -> Result<()> {
        let df = df! [
            "feature1" => [1.0f64, 2.0, 3.0],
            "feature2" => [10.0f64, 20.0, 30.0],
            "target" => [0i32, 1, 0],
        ].map_err(|e| LightGBMError::data_loading(format!("Failed to create test DataFrame: {}", e)))?;

        let dataset_config = DatasetConfig::default();
        let loader = PolarsLoader::new(dataset_config)?;
        
        let estimate = loader.estimate_memory_usage(&df);
        assert!(estimate.dataframe_bytes > 0);
        assert!(estimate.dataset_bytes > 0);
        assert!(estimate.total_estimated > 0);
        assert_eq!(estimate.num_rows, 3);
        assert_eq!(estimate.num_cols, 3);
        Ok(())
    }
}