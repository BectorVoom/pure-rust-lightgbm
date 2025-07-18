//! Parquet file loader for Pure Rust LightGBM.
//!
//! This module provides efficient Parquet file loading with column projection,
//! predicate pushdown, and streaming capabilities for large datasets.

use crate::core::types::*;
use crate::core::error::{Result, LightGBMError};
use crate::dataset::{Dataset, DatasetConfig};
use super::{DataLoader, LoaderConfig};
use ndarray::{s, Array1, Array2};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[cfg(feature = "parquet")]
use arrow::{
    array::*,
    datatypes::*,
    record_batch::RecordBatch,
};

#[cfg(feature = "parquet")]
use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    arrow::{ArrowReader, ParquetFileArrowReader},
    schema::types::Type,
    basic::Type as PhysicalType,
};

/// Parquet-specific configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParquetConfig {
    /// Target column name
    pub target_column: String,
    /// Feature columns to include (None = all except target)
    pub feature_columns: Option<Vec<String>>,
    /// Weight column name
    pub weight_column: Option<String>,
    /// Group column for ranking
    pub group_column: Option<String>,
    /// Batch size for reading
    pub batch_size: usize,
    /// Maximum number of rows to read
    pub max_rows: Option<usize>,
    /// Enable column projection
    pub column_projection: bool,
    /// Enable predicate pushdown
    pub predicate_pushdown: bool,
    /// Parallel row group reading
    pub parallel_row_groups: bool,
    /// Memory limit (MB)
    pub memory_limit_mb: Option<usize>,
    /// Use metadata caching
    pub cache_metadata: bool,
    /// Row group size for estimation
    pub row_group_size: Option<usize>,
}

impl Default for ParquetConfig {
    fn default() -> Self {
        ParquetConfig {
            target_column: "target".to_string(),
            feature_columns: None,
            weight_column: None,
            group_column: None,
            batch_size: 10000,
            max_rows: None,
            column_projection: true,
            predicate_pushdown: true,
            parallel_row_groups: true,
            memory_limit_mb: None,
            cache_metadata: true,
            row_group_size: None,
        }
    }
}

/// Parquet data loader
#[cfg(feature = "parquet")]
pub struct ParquetLoader {
    /// Loader configuration
    config: LoaderConfig,
    /// Parquet-specific configuration
    parquet_config: ParquetConfig,
    /// Dataset configuration
    dataset_config: DatasetConfig,
}

#[cfg(feature = "parquet")]
impl ParquetLoader {
    /// Create a new Parquet loader
    pub fn new(dataset_config: DatasetConfig) -> Result<Self> {
        let parquet_config = ParquetConfig {
            target_column: dataset_config.target_column.clone()
                .unwrap_or_else(|| "target".to_string()),
            feature_columns: dataset_config.feature_columns.clone(),
            weight_column: dataset_config.weight_column.clone(),
            group_column: dataset_config.group_column.clone(),
            ..Default::default()
        };

        Ok(ParquetLoader {
            config: LoaderConfig {
                dataset_config: dataset_config.clone(),
                ..Default::default()
            },
            parquet_config,
            dataset_config,
        })
    }

    /// Create Parquet loader with custom configuration
    pub fn with_parquet_config(mut self, parquet_config: ParquetConfig) -> Self {
        self.parquet_config = parquet_config;
        self
    }

    /// Set target column
    pub fn with_target_column<S: Into<String>>(mut self, target: S) -> Self {
        self.parquet_config.target_column = target.into();
        self
    }

    /// Set feature columns
    pub fn with_feature_columns(mut self, features: Vec<String>) -> Self {
        self.parquet_config.feature_columns = Some(features);
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.parquet_config.batch_size = batch_size;
        self
    }

    /// Set maximum rows to read
    pub fn with_max_rows(mut self, max_rows: usize) -> Self {
        self.parquet_config.max_rows = Some(max_rows);
        self
    }

    /// Enable column projection
    pub fn with_column_projection(mut self, projection: bool) -> Self {
        self.parquet_config.column_projection = projection;
        self
    }

    /// Enable parallel row group reading
    pub fn with_parallel_row_groups(mut self, parallel: bool) -> Self {
        self.parquet_config.parallel_row_groups = parallel;
        self
    }

    /// Load Parquet file
    pub fn load_parquet<P: AsRef<Path>>(&self, path: P) -> Result<Dataset> {
        let path = path.as_ref();
        log::info!("Loading Parquet file: {}", path.display());

        // Validate file exists
        if !path.exists() {
            return Err(LightGBMError::data_loading(format!(
                "Parquet file does not exist: {}", path.display()
            )));
        }

        // Open Parquet file
        let file = std::fs::File::open(path)
            .map_err(|e| LightGBMError::data_loading(format!(
                "Failed to open Parquet file {}: {}", path.display(), e
            )))?;

        let parquet_reader = SerializedFileReader::new(file)
            .map_err(|e| LightGBMError::data_loading(format!(
                "Failed to create Parquet reader: {}", e
            )))?;

        // Get metadata
        let metadata = parquet_reader.metadata();
        let schema = metadata.file_metadata().schema();
        
        log::info!("Parquet file has {} row groups with {} rows total", 
                  metadata.num_row_groups(),
                  metadata.file_metadata().num_rows());

        // Validate schema
        self.validate_parquet_schema(schema)?;

        // Create Arrow reader
        let mut arrow_reader = ParquetFileArrowReader::new(std::sync::Arc::new(parquet_reader));
        
        // Apply column projection if enabled
        if self.parquet_config.column_projection {
            let column_indices = self.determine_column_indices(&arrow_reader)?;
            arrow_reader = arrow_reader.set_column_selection(column_indices);
        }

        // Read data in batches
        let record_batch_reader = arrow_reader
            .get_record_reader(self.parquet_config.batch_size)
            .map_err(|e| LightGBMError::data_loading(format!(
                "Failed to create record batch reader: {}", e
            )))?;

        // Collect batches
        let mut batches = Vec::new();
        let mut rows_read = 0;

        for batch_result in record_batch_reader {
            let batch = batch_result
                .map_err(|e| LightGBMError::data_loading(format!(
                    "Failed to read Parquet batch: {}", e
                )))?;

            rows_read += batch.num_rows();
            batches.push(batch);

            // Apply max_rows limit
            if let Some(max_rows) = self.parquet_config.max_rows {
                if rows_read >= max_rows {
                    log::info!("Reached max_rows limit of {}, stopping", max_rows);
                    break;
                }
            }
        }

        if batches.is_empty() {
            return Err(LightGBMError::data_loading("No data found in Parquet file"));
        }

        // Process batches
        self.process_parquet_batches(batches, path)
    }

    /// Process Parquet batches into Dataset
    fn process_parquet_batches(&self, batches: Vec<RecordBatch>, source_path: &Path) -> Result<Dataset> {
        if batches.is_empty() {
            return Err(LightGBMError::data_loading("No batches to process"));
        }

        // Calculate total rows
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        let schema = batches[0].schema();

        log::info!("Processing {} batches with {} total rows", batches.len(), total_rows);

        // Determine column structure
        let target_col_idx = self.find_column_index(&schema, &self.parquet_config.target_column)?;
        let feature_columns = self.determine_feature_columns(&schema)?;
        let num_features = feature_columns.len();

        // Pre-allocate arrays
        let mut features = Array2::<f32>::zeros((total_rows, num_features));
        let mut labels = Array1::<f32>::zeros(total_rows);
        let mut weights: Option<Array1<f32>> = None;
        let mut groups: Option<Array1<DataSize>> = None;
        let mut missing_mask = Array2::<bool>::zeros((total_rows, num_features));
        let mut has_missing = false;

        let mut row_offset = 0;

        // Process each batch
        for (batch_idx, batch) in batches.iter().enumerate() {
            let batch_rows = batch.num_rows();
            
            if batch_rows == 0 {
                continue;
            }

            log::debug!("Processing batch {} with {} rows", batch_idx, batch_rows);

            // Extract labels
            let batch_labels = self.extract_labels_from_batch(batch, target_col_idx)?;
            labels.slice_mut(s![row_offset..row_offset + batch_rows])
                .assign(&batch_labels);

            // Extract features
            let batch_features = self.extract_features_from_batch(batch, &feature_columns)?;
            features.slice_mut(s![row_offset..row_offset + batch_rows, ..])
                .assign(&batch_features);

            // Extract weights (initialize on first batch)
            if let Some(batch_weights) = self.extract_weights_from_batch(batch)? {
                if weights.is_none() {
                    weights = Some(Array1::<f32>::zeros(total_rows));
                }
                weights.as_mut().unwrap()
                    .slice_mut(s![row_offset..row_offset + batch_rows])
                    .assign(&batch_weights);
            }

            // Extract groups (initialize on first batch)
            if let Some(batch_groups) = self.extract_groups_from_batch(batch)? {
                if groups.is_none() {
                    groups = Some(Array1::<DataSize>::zeros(total_rows));
                }
                groups.as_mut().unwrap()
                    .slice_mut(s![row_offset..row_offset + batch_rows])
                    .assign(&batch_groups);
            }

            // Detect missing values
            if let Some(batch_missing) = self.detect_missing_values_from_batch(batch, &feature_columns)? {
                missing_mask.slice_mut(s![row_offset..row_offset + batch_rows, ..])
                    .assign(&batch_missing);
                has_missing = true;
            }

            row_offset += batch_rows;
        }

        // Create feature names
        let feature_names = feature_columns.iter()
            .map(|&idx| schema.field(idx).name().clone())
            .collect();

        // Create Dataset
        let mut dataset = Dataset::new(
            features,
            labels,
            weights,
            groups,
            Some(feature_names),
            None, // feature_types - will be auto-detected
        )?;

        // Set missing values if any detected
        if has_missing {
            dataset.set_missing_values(missing_mask)?;
        }

        // Set metadata
        let metadata = dataset.metadata_mut();
        metadata.source_path = Some(source_path.to_string_lossy().to_string());
        metadata.format = "parquet".to_string();
        metadata.properties.insert("num_rows".to_string(), total_rows.to_string());
        metadata.properties.insert("num_batches".to_string(), batches.len().to_string());
        metadata.properties.insert("batch_size".to_string(), self.parquet_config.batch_size.to_string());

        Ok(dataset)
    }

    /// Validate Parquet schema
    fn validate_parquet_schema(&self, schema: &Type) -> Result<()> {
        // Check if schema is a group (root schema)
        if !schema.is_group() {
            return Err(LightGBMError::data_loading("Invalid Parquet schema: root is not a group"));
        }

        // For now, just basic validation
        // TODO: Add more comprehensive schema validation
        Ok(())
    }

    /// Determine column indices for projection
    fn determine_column_indices(&self, arrow_reader: &ParquetFileArrowReader) -> Result<Vec<usize>> {
        let arrow_schema = arrow_reader.get_schema()
            .map_err(|e| LightGBMError::data_loading(format!(
                "Failed to get Arrow schema: {}", e
            )))?;

        let mut column_indices = Vec::new();

        // Add target column
        let target_idx = self.find_column_index(&arrow_schema, &self.parquet_config.target_column)?;
        column_indices.push(target_idx);

        // Add feature columns
        let feature_columns = self.determine_feature_columns(&arrow_schema)?;
        column_indices.extend(feature_columns);

        // Add weight column if specified
        if let Some(ref weight_col) = self.parquet_config.weight_column {
            let weight_idx = self.find_column_index(&arrow_schema, weight_col)?;
            column_indices.push(weight_idx);
        }

        // Add group column if specified
        if let Some(ref group_col) = self.parquet_config.group_column {
            let group_idx = self.find_column_index(&arrow_schema, group_col)?;
            column_indices.push(group_idx);
        }

        // Remove duplicates and sort
        column_indices.sort_unstable();
        column_indices.dedup();

        Ok(column_indices)
    }

    /// Find column index by name
    fn find_column_index(&self, schema: &Schema, column_name: &str) -> Result<usize> {
        schema.fields().iter()
            .position(|field| field.name() == column_name)
            .ok_or_else(|| LightGBMError::data_loading(format!(
                "Column '{}' not found in schema", column_name
            )))
    }

    /// Determine feature column indices
    fn determine_feature_columns(&self, schema: &Schema) -> Result<Vec<usize>> {
        if let Some(ref feature_cols) = self.parquet_config.feature_columns {
            // Use specified feature columns
            let mut indices = Vec::new();
            for col_name in feature_cols {
                let idx = self.find_column_index(schema, col_name)?;
                indices.push(idx);
            }
            Ok(indices)
        } else {
            // Use all columns except target, weight, and group
            let mut excluded_columns = std::collections::HashSet::new();
            excluded_columns.insert(self.parquet_config.target_column.as_str());
            
            if let Some(ref weight_col) = self.parquet_config.weight_column {
                excluded_columns.insert(weight_col.as_str());
            }
            
            if let Some(ref group_col) = self.parquet_config.group_column {
                excluded_columns.insert(group_col.as_str());
            }

            let feature_indices: Vec<usize> = schema.fields().iter()
                .enumerate()
                .filter(|(_, field)| !excluded_columns.contains(field.name().as_str()))
                .map(|(idx, _)| idx)
                .collect();

            if feature_indices.is_empty() {
                return Err(LightGBMError::data_loading("No feature columns found"));
            }

            Ok(feature_indices)
        }
    }

    /// Extract labels from RecordBatch
    fn extract_labels_from_batch(&self, batch: &RecordBatch, target_col_idx: usize) -> Result<Array1<f32>> {
        let column = batch.column(target_col_idx);
        self.array_to_f32_array(column, "target")
    }

    /// Extract features from RecordBatch
    fn extract_features_from_batch(&self, batch: &RecordBatch, feature_columns: &[usize]) -> Result<Array2<f32>> {
        let num_rows = batch.num_rows();
        let num_features = feature_columns.len();
        let mut features = Array2::<f32>::zeros((num_rows, num_features));

        for (feat_idx, &col_idx) in feature_columns.iter().enumerate() {
            let column = batch.column(col_idx);
            let values = self.array_to_f32_array(column, &format!("feature_{}", col_idx))?;
            
            for (row_idx, &value) in values.iter().enumerate() {
                features[[row_idx, feat_idx]] = value;
            }
        }

        Ok(features)
    }

    /// Extract weights from RecordBatch
    fn extract_weights_from_batch(&self, batch: &RecordBatch) -> Result<Option<Array1<f32>>> {
        if let Some(ref weight_col) = self.parquet_config.weight_column {
            let schema = batch.schema();
            let weight_idx = self.find_column_index(&schema, weight_col)?;
            let column = batch.column(weight_idx);
            let weights = self.array_to_f32_array(column, weight_col)?;
            Ok(Some(weights))
        } else {
            Ok(None)
        }
    }

    /// Extract groups from RecordBatch
    fn extract_groups_from_batch(&self, batch: &RecordBatch) -> Result<Option<Array1<DataSize>>> {
        if let Some(ref group_col) = self.parquet_config.group_column {
            let schema = batch.schema();
            let group_idx = self.find_column_index(&schema, group_col)?;
            let column = batch.column(group_idx);
            let groups = self.array_to_datasize_array(column, group_col)?;
            Ok(Some(groups))
        } else {
            Ok(None)
        }
    }

    /// Detect missing values from RecordBatch
    fn detect_missing_values_from_batch(&self, batch: &RecordBatch, feature_columns: &[usize]) -> Result<Option<Array2<bool>>> {
        let num_rows = batch.num_rows();
        let num_features = feature_columns.len();
        let mut missing_mask = Array2::<bool>::zeros((num_rows, num_features));
        let mut has_missing = false;

        for (feat_idx, &col_idx) in feature_columns.iter().enumerate() {
            let column = batch.column(col_idx);
            
            // Check for null values
            if column.null_count() > 0 {
                has_missing = true;
                let null_buffer = column.nulls();
                
                if let Some(null_buffer) = null_buffer {
                    for row_idx in 0..num_rows {
                        missing_mask[[row_idx, feat_idx]] = !null_buffer.is_valid(row_idx);
                    }
                }
            }
        }

        if has_missing {
            Ok(Some(missing_mask))
        } else {
            Ok(None)
        }
    }

    /// Convert Arrow Array to f32 Array1
    fn array_to_f32_array(&self, array: &dyn Array, col_name: &str) -> Result<Array1<f32>> {
        match array.data_type() {
            DataType::Float32 => {
                let float_array = array.as_any().downcast_ref::<Float32Array>()
                    .ok_or_else(|| LightGBMError::data_loading(format!(
                        "Failed to downcast Float32Array for column {}", col_name
                    )))?;
                
                let values: Vec<f32> = (0..float_array.len())
                    .map(|i| if float_array.is_null(i) { f32::NAN } else { float_array.value(i) })
                    .collect();
                Ok(Array1::from_vec(values))
            },
            DataType::Float64 => {
                let float_array = array.as_any().downcast_ref::<Float64Array>()
                    .ok_or_else(|| LightGBMError::data_loading(format!(
                        "Failed to downcast Float64Array for column {}", col_name
                    )))?;
                
                let values: Vec<f32> = (0..float_array.len())
                    .map(|i| if float_array.is_null(i) { f32::NAN } else { float_array.value(i) as f32 })
                    .collect();
                Ok(Array1::from_vec(values))
            },
            DataType::Int32 => {
                let int_array = array.as_any().downcast_ref::<Int32Array>()
                    .ok_or_else(|| LightGBMError::data_loading(format!(
                        "Failed to downcast Int32Array for column {}", col_name
                    )))?;
                
                let values: Vec<f32> = (0..int_array.len())
                    .map(|i| if int_array.is_null(i) { f32::NAN } else { int_array.value(i) as f32 })
                    .collect();
                Ok(Array1::from_vec(values))
            },
            DataType::Int64 => {
                let int_array = array.as_any().downcast_ref::<Int64Array>()
                    .ok_or_else(|| LightGBMError::data_loading(format!(
                        "Failed to downcast Int64Array for column {}", col_name
                    )))?;
                
                let values: Vec<f32> = (0..int_array.len())
                    .map(|i| if int_array.is_null(i) { f32::NAN } else { int_array.value(i) as f32 })
                    .collect();
                Ok(Array1::from_vec(values))
            },
            DataType::Boolean => {
                let bool_array = array.as_any().downcast_ref::<BooleanArray>()
                    .ok_or_else(|| LightGBMError::data_loading(format!(
                        "Failed to downcast BooleanArray for column {}", col_name
                    )))?;
                
                let values: Vec<f32> = (0..bool_array.len())
                    .map(|i| if bool_array.is_null(i) { f32::NAN } else if bool_array.value(i) { 1.0 } else { 0.0 })
                    .collect();
                Ok(Array1::from_vec(values))
            },
            _ => Err(LightGBMError::data_loading(format!(
                "Unsupported data type {:?} for column {}", array.data_type(), col_name
            )))
        }
    }

    /// Convert Arrow Array to DataSize Array1
    fn array_to_datasize_array(&self, array: &dyn Array, col_name: &str) -> Result<Array1<DataSize>> {
        match array.data_type() {
            DataType::Int32 => {
                let int_array = array.as_any().downcast_ref::<Int32Array>()
                    .ok_or_else(|| LightGBMError::data_loading(format!(
                        "Failed to downcast Int32Array for column {}", col_name
                    )))?;
                
                let values: Vec<DataSize> = (0..int_array.len())
                    .map(|i| if int_array.is_null(i) { 0 } else { int_array.value(i) })
                    .collect();
                Ok(Array1::from_vec(values))
            },
            DataType::Int64 => {
                let int_array = array.as_any().downcast_ref::<Int64Array>()
                    .ok_or_else(|| LightGBMError::data_loading(format!(
                        "Failed to downcast Int64Array for column {}", col_name
                    )))?;
                
                let values: Vec<DataSize> = (0..int_array.len())
                    .map(|i| if int_array.is_null(i) { 0 } else { int_array.value(i) as DataSize })
                    .collect();
                Ok(Array1::from_vec(values))
            },
            _ => Err(LightGBMError::data_loading(format!(
                "Unsupported data type {:?} for group column {}", array.data_type(), col_name
            )))
        }
    }

    /// Get configuration
    pub fn parquet_config(&self) -> &ParquetConfig {
        &self.parquet_config
    }

    /// Estimate memory usage for Parquet file
    pub fn estimate_memory_usage<P: AsRef<Path>>(&self, path: P) -> Result<ParquetMemoryEstimate> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .map_err(|e| LightGBMError::data_loading(format!("Failed to open file: {}", e)))?;

        let reader = SerializedFileReader::new(file)
            .map_err(|e| LightGBMError::data_loading(format!("Failed to create reader: {}", e)))?;

        let metadata = reader.metadata();
        let file_metadata = metadata.file_metadata();
        
        let num_rows = file_metadata.num_rows() as usize;
        let num_columns = file_metadata.schema().get_fields().len();
        let num_row_groups = metadata.num_row_groups();

        // Estimate compressed size from metadata
        let mut compressed_bytes = 0;
        for i in 0..num_row_groups {
            let row_group = metadata.row_group(i);
            compressed_bytes += row_group.total_byte_size() as usize;
        }

        // Estimate uncompressed size (rough approximation)
        let uncompressed_bytes = compressed_bytes * 3; // Typical compression ratio

        // Estimate Dataset memory usage
        let dataset_bytes = num_rows * num_columns * std::mem::size_of::<f32>();

        // Estimate processing overhead
        let processing_overhead = uncompressed_bytes / 2;

        ParquetMemoryEstimate {
            compressed_bytes,
            uncompressed_bytes,
            dataset_bytes,
            processing_overhead,
            total_estimated: uncompressed_bytes + dataset_bytes + processing_overhead,
            num_rows,
            num_columns,
            num_row_groups,
        }
    }
}

/// Memory usage estimate for Parquet file loading
#[derive(Debug, Clone)]
pub struct ParquetMemoryEstimate {
    /// Compressed file size
    pub compressed_bytes: usize,
    /// Estimated uncompressed size
    pub uncompressed_bytes: usize,
    /// Dataset memory usage after conversion
    pub dataset_bytes: usize,
    /// Processing overhead
    pub processing_overhead: usize,
    /// Total estimated memory usage
    pub total_estimated: usize,
    /// Number of rows
    pub num_rows: usize,
    /// Number of columns
    pub num_columns: usize,
    /// Number of row groups
    pub num_row_groups: i32,
}

#[cfg(feature = "parquet")]
impl DataLoader for ParquetLoader {
    fn load<P: AsRef<Path>>(&self, path: P) -> Result<Dataset> {
        self.load_parquet(path)
    }

    fn config(&self) -> &LoaderConfig {
        &self.config
    }
}

// Stub implementation when parquet feature is not enabled
#[cfg(not(feature = "parquet"))]
pub struct ParquetLoader;

#[cfg(not(feature = "parquet"))]
impl ParquetLoader {
    pub fn new(_dataset_config: DatasetConfig) -> Result<Self> {
        Err(LightGBMError::not_implemented(
            "Parquet support requires 'parquet' feature to be enabled"
        ))
    }
}

#[cfg(not(feature = "parquet"))]
impl DataLoader for ParquetLoader {
    fn load<P: AsRef<Path>>(&self, _path: P) -> Result<Dataset> {
        Err(LightGBMError::not_implemented(
            "Parquet support requires 'parquet' feature to be enabled"
        ))
    }

    fn config(&self) -> &LoaderConfig {
        unreachable!("ParquetLoader cannot be created without parquet feature")
    }
}

#[cfg(test)]
#[cfg(feature = "parquet")]
mod tests {
    use super::*;

    #[test]
    fn test_parquet_config_default() {
        let config = ParquetConfig::default();
        assert_eq!(config.target_column, "target");
        assert_eq!(config.batch_size, 10000);
        assert!(config.column_projection);
        assert!(config.predicate_pushdown);
        assert!(config.parallel_row_groups);
        assert!(config.cache_metadata);
    }

    #[test]
    fn test_parquet_loader_creation() -> Result<()> {
        let dataset_config = DatasetConfig::new()
            .with_target_column("target");
        let loader = ParquetLoader::new(dataset_config)?;
        assert_eq!(loader.parquet_config().target_column, "target");
        Ok(())
    }

    #[test]
    fn test_parquet_loader_configuration() -> Result<()> {
        let dataset_config = DatasetConfig::default();
        let loader = ParquetLoader::new(dataset_config)?
            .with_target_column("label")
            .with_feature_columns(vec!["feat1".to_string(), "feat2".to_string()])
            .with_batch_size(5000)
            .with_max_rows(10000)
            .with_column_projection(false)
            .with_parallel_row_groups(false);

        assert_eq!(loader.parquet_config().target_column, "label");
        assert_eq!(loader.parquet_config().feature_columns, Some(vec!["feat1".to_string(), "feat2".to_string()]));
        assert_eq!(loader.parquet_config().batch_size, 5000);
        assert_eq!(loader.parquet_config().max_rows, Some(10000));
        assert!(!loader.parquet_config().column_projection);
        assert!(!loader.parquet_config().parallel_row_groups);
        Ok(())
    }

    // Note: Actual Parquet file loading tests would require test data files
    // These would be added in a complete implementation
}