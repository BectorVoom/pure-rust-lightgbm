//! Apache Arrow loader for Pure Rust LightGBM.
//!
//! This module provides efficient data loading from Apache Arrow formats including
//! RecordBatch, Table, and IPC streams with zero-copy optimization where possible.

use crate::core::types::*;
use crate::core::error::{Result, LightGBMError};
use crate::dataset::{Dataset, DatasetConfig};
use super::{DataLoader, LoaderConfig};
use ndarray::{s, Array1, Array2};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "arrow")]
use arrow::{
    array::*,
    datatypes::*,
    record_batch::RecordBatch,
    ipc::{reader::FileReader, writer::FileWriter},
    json::ReaderBuilder as JsonReaderBuilder,
};

/// Arrow-specific configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArrowConfig {
    /// Target column name
    pub target_column: String,
    /// Feature columns to include (None = all except target)
    pub feature_columns: Option<Vec<String>>,
    /// Weight column name
    pub weight_column: Option<String>,
    /// Group column for ranking
    pub group_column: Option<String>,
    /// Batch size for processing large datasets
    pub batch_size: usize,
    /// Maximum number of batches to read
    pub max_batches: Option<usize>,
    /// Enable zero-copy optimization where possible
    pub zero_copy: bool,
    /// Parallel processing for multiple batches
    pub parallel: bool,
    /// Memory limit (MB)
    pub memory_limit_mb: Option<usize>,
}

impl Default for ArrowConfig {
    fn default() -> Self {
        ArrowConfig {
            target_column: "target".to_string(),
            feature_columns: None,
            weight_column: None,
            group_column: None,
            batch_size: 10000,
            max_batches: None,
            zero_copy: true,
            parallel: true,
            memory_limit_mb: None,
        }
    }
}

/// Apache Arrow data loader
#[cfg(feature = "arrow")]
pub struct ArrowLoader {
    /// Loader configuration
    config: LoaderConfig,
    /// Arrow-specific configuration
    arrow_config: ArrowConfig,
    /// Dataset configuration
    dataset_config: DatasetConfig,
}

#[cfg(feature = "arrow")]
impl ArrowLoader {
    /// Create a new Arrow loader
    pub fn new(dataset_config: DatasetConfig) -> Result<Self> {
        let arrow_config = ArrowConfig {
            target_column: dataset_config.target_column.clone()
                .unwrap_or_else(|| "target".to_string()),
            feature_columns: dataset_config.feature_columns.clone(),
            weight_column: dataset_config.weight_column.clone(),
            group_column: dataset_config.group_column.clone(),
            ..Default::default()
        };

        Ok(ArrowLoader {
            config: LoaderConfig {
                dataset_config: dataset_config.clone(),
                ..Default::default()
            },
            arrow_config,
            dataset_config,
        })
    }

    /// Create Arrow loader with custom configuration
    pub fn with_arrow_config(mut self, arrow_config: ArrowConfig) -> Self {
        self.arrow_config = arrow_config;
        self
    }

    /// Set target column
    pub fn with_target_column<S: Into<String>>(mut self, target: S) -> Self {
        self.arrow_config.target_column = target.into();
        self
    }

    /// Set feature columns
    pub fn with_feature_columns(mut self, features: Vec<String>) -> Self {
        self.arrow_config.feature_columns = Some(features);
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.arrow_config.batch_size = batch_size;
        self
    }

    /// Enable zero-copy optimization
    pub fn with_zero_copy(mut self, zero_copy: bool) -> Self {
        self.arrow_config.zero_copy = zero_copy;
        self
    }

    /// Enable parallel processing
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.arrow_config.parallel = parallel;
        self
    }

    /// Load from Arrow RecordBatch
    pub fn load_record_batch(&self, batch: &RecordBatch) -> Result<Dataset> {
        log::info!("Loading from Arrow RecordBatch with {} rows and {} columns", 
                  batch.num_rows(), batch.num_columns());

        // Validate RecordBatch
        self.validate_record_batch(batch)?;

        // Process the batch
        self.process_single_batch(batch)
    }

    /// Load from Arrow Table (multiple RecordBatches)
    pub fn load_table(&self, batches: &[RecordBatch]) -> Result<Dataset> {
        log::info!("Loading from Arrow Table with {} batches", batches.len());

        if batches.is_empty() {
            return Err(LightGBMError::data_loading("Table contains no RecordBatches"));
        }

        // Validate all batches have the same schema
        let first_schema = batches[0].schema();
        for (i, batch) in batches.iter().enumerate().skip(1) {
            if batch.schema() != first_schema {
                return Err(LightGBMError::data_loading(format!(
                    "Schema mismatch in batch {}: expected {:?}, got {:?}",
                    i, first_schema, batch.schema()
                )));
            }
        }

        // Process batches based on configuration
        if self.arrow_config.parallel && batches.len() > 1 {
            self.process_batches_parallel(batches)
        } else {
            self.process_batches_sequential(batches)
        }
    }

    /// Load from Arrow IPC file
    pub fn load_ipc_file<P: AsRef<Path>>(&self, path: P) -> Result<Dataset> {
        let path = path.as_ref();
        log::info!("Loading Arrow IPC file: {}", path.display());

        let file = std::fs::File::open(path)
            .map_err(|e| LightGBMError::data_loading(format!(
                "Failed to open Arrow IPC file {}: {}", path.display(), e
            )))?;

        let reader = FileReader::try_new(file, None)
            .map_err(|e| LightGBMError::data_loading(format!(
                "Failed to create Arrow IPC reader: {}", e
            )))?;

        // Collect all batches
        let mut batches = Vec::new();
        let mut batches_read = 0;

        for batch_result in reader {
            let batch = batch_result
                .map_err(|e| LightGBMError::data_loading(format!(
                    "Failed to read Arrow batch: {}", e
                )))?;

            batches.push(batch);
            batches_read += 1;

            // Apply max_batches limit
            if let Some(max_batches) = self.arrow_config.max_batches {
                if batches_read >= max_batches {
                    log::info!("Reached max_batches limit of {}, stopping", max_batches);
                    break;
                }
            }
        }

        // Process collected batches
        self.load_table(&batches)
    }

    /// Load from JSON file using Arrow JSON reader
    pub fn load_json<P: AsRef<Path>>(&self, path: P) -> Result<Dataset> {
        let path = path.as_ref();
        log::info!("Loading JSON file with Arrow: {}", path.display());

        let file = std::fs::File::open(path)
            .map_err(|e| LightGBMError::data_loading(format!(
                "Failed to open JSON file {}: {}", path.display(), e
            )))?;

        let mut json_reader = JsonReaderBuilder::new(Arc::new(Schema::empty()))
            .with_batch_size(self.arrow_config.batch_size)
            .build(file)
            .map_err(|e| LightGBMError::data_loading(format!(
                "Failed to create JSON reader: {}", e
            )))?;

        // Read all batches
        let mut batches = Vec::new();
        while let Some(batch_result) = json_reader.next() {
            let batch = batch_result
                .map_err(|e| LightGBMError::data_loading(format!(
                    "Failed to read JSON batch: {}", e
                )))?;
            batches.push(batch);
        }

        self.load_table(&batches)
    }

    /// Process single RecordBatch
    fn process_single_batch(&self, batch: &RecordBatch) -> Result<Dataset> {
        let schema = batch.schema();
        let num_rows = batch.num_rows();

        // Determine column indices
        let target_col_idx = self.find_column_index(&schema, &self.arrow_config.target_column)?;
        let feature_columns = self.determine_feature_columns(&schema)?;
        
        // Extract data
        let labels = self.extract_labels_from_batch(batch, target_col_idx)?;
        let features = self.extract_features_from_batch(batch, &feature_columns)?;
        let weights = self.extract_weights_from_batch(batch)?;
        let groups = self.extract_groups_from_batch(batch)?;

        // Detect missing values
        let missing_values = self.detect_missing_values_from_batch(batch, &feature_columns)?;

        // Create Dataset
        let feature_names = feature_columns.iter()
            .map(|&idx| schema.field(idx).name().clone())
            .collect();

        let mut dataset = Dataset::new(
            features,
            labels,
            weights,
            groups,
            Some(feature_names),
            None, // feature_types - will be auto-detected
        )?;

        // Set missing values if any detected
        if let Some(missing_mask) = missing_values {
            dataset.set_missing_values(missing_mask)?;
        }

        // Set metadata
        let metadata = dataset.metadata_mut();
        metadata.source_path = Some("Arrow RecordBatch".to_string());
        metadata.format = "arrow".to_string();
        metadata.properties.insert("num_rows".to_string(), num_rows.to_string());
        metadata.properties.insert("num_cols".to_string(), schema.fields().len().to_string());

        Ok(dataset)
    }

    /// Process multiple batches sequentially
    fn process_batches_sequential(&self, batches: &[RecordBatch]) -> Result<Dataset> {
        // Calculate total rows
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        let schema = batches[0].schema();

        // Determine column structure
        let target_col_idx = self.find_column_index(&schema, &self.arrow_config.target_column)?;
        let feature_columns = self.determine_feature_columns(&schema)?;
        let num_features = feature_columns.len();

        // Pre-allocate arrays
        let mut features = Array2::<f32>::zeros((total_rows, num_features));
        let mut labels = Array1::<f32>::zeros(total_rows);
        let mut weights: Option<Array1<f32>> = None;
        let mut groups: Option<Array1<DataSize>> = None;
        let mut missing_mask = Array2::<bool>::zeros((total_rows, num_features));

        let mut row_offset = 0;

        // Process each batch
        for batch in batches {
            let batch_rows = batch.num_rows();

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
        let has_missing = missing_mask.iter().any(|&x| x);
        if has_missing {
            dataset.set_missing_values(missing_mask)?;
        }

        // Set metadata
        let metadata = dataset.metadata_mut();
        metadata.source_path = Some(format!("Arrow Table ({} batches)", batches.len()));
        metadata.format = "arrow".to_string();
        metadata.properties.insert("num_rows".to_string(), total_rows.to_string());
        metadata.properties.insert("num_batches".to_string(), batches.len().to_string());

        Ok(dataset)
    }

    /// Process multiple batches in parallel (placeholder for future implementation)
    fn process_batches_parallel(&self, batches: &[RecordBatch]) -> Result<Dataset> {
        // For now, fall back to sequential processing
        // TODO: Implement parallel processing using rayon
        log::warn!("Parallel batch processing not yet implemented, falling back to sequential");
        self.process_batches_sequential(batches)
    }

    /// Validate RecordBatch before processing
    fn validate_record_batch(&self, batch: &RecordBatch) -> Result<()> {
        if batch.num_rows() == 0 {
            return Err(LightGBMError::data_loading("RecordBatch is empty"));
        }

        if batch.num_columns() == 0 {
            return Err(LightGBMError::data_loading("RecordBatch has no columns"));
        }

        let schema = batch.schema();

        // Check if target column exists
        if self.find_column_index(&schema, &self.arrow_config.target_column).is_err() {
            return Err(LightGBMError::data_loading(format!(
                "Target column '{}' not found in RecordBatch", 
                self.arrow_config.target_column
            )));
        }

        Ok(())
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
        if let Some(ref feature_cols) = self.arrow_config.feature_columns {
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
            excluded_columns.insert(self.arrow_config.target_column.as_str());
            
            if let Some(ref weight_col) = self.arrow_config.weight_column {
                excluded_columns.insert(weight_col.as_str());
            }
            
            if let Some(ref group_col) = self.arrow_config.group_column {
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
        if let Some(ref weight_col) = self.arrow_config.weight_column {
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
        if let Some(ref group_col) = self.arrow_config.group_column {
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
                    .map(|i| float_array.value(i))
                    .collect();
                Ok(Array1::from_vec(values))
            },
            DataType::Float64 => {
                let float_array = array.as_any().downcast_ref::<Float64Array>()
                    .ok_or_else(|| LightGBMError::data_loading(format!(
                        "Failed to downcast Float64Array for column {}", col_name
                    )))?;
                
                let values: Vec<f32> = (0..float_array.len())
                    .map(|i| float_array.value(i) as f32)
                    .collect();
                Ok(Array1::from_vec(values))
            },
            DataType::Int32 => {
                let int_array = array.as_any().downcast_ref::<Int32Array>()
                    .ok_or_else(|| LightGBMError::data_loading(format!(
                        "Failed to downcast Int32Array for column {}", col_name
                    )))?;
                
                let values: Vec<f32> = (0..int_array.len())
                    .map(|i| int_array.value(i) as f32)
                    .collect();
                Ok(Array1::from_vec(values))
            },
            DataType::Int64 => {
                let int_array = array.as_any().downcast_ref::<Int64Array>()
                    .ok_or_else(|| LightGBMError::data_loading(format!(
                        "Failed to downcast Int64Array for column {}", col_name
                    )))?;
                
                let values: Vec<f32> = (0..int_array.len())
                    .map(|i| int_array.value(i) as f32)
                    .collect();
                Ok(Array1::from_vec(values))
            },
            DataType::Boolean => {
                let bool_array = array.as_any().downcast_ref::<BooleanArray>()
                    .ok_or_else(|| LightGBMError::data_loading(format!(
                        "Failed to downcast BooleanArray for column {}", col_name
                    )))?;
                
                let values: Vec<f32> = (0..bool_array.len())
                    .map(|i| if bool_array.value(i) { 1.0 } else { 0.0 })
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
                    .map(|i| int_array.value(i))
                    .collect();
                Ok(Array1::from_vec(values))
            },
            DataType::Int64 => {
                let int_array = array.as_any().downcast_ref::<Int64Array>()
                    .ok_or_else(|| LightGBMError::data_loading(format!(
                        "Failed to downcast Int64Array for column {}", col_name
                    )))?;
                
                let values: Vec<DataSize> = (0..int_array.len())
                    .map(|i| int_array.value(i) as DataSize)
                    .collect();
                Ok(Array1::from_vec(values))
            },
            _ => Err(LightGBMError::data_loading(format!(
                "Unsupported data type {:?} for group column {}", array.data_type(), col_name
            )))
        }
    }

    /// Get configuration
    pub fn arrow_config(&self) -> &ArrowConfig {
        &self.arrow_config
    }

    /// Estimate memory usage for Arrow data
    pub fn estimate_memory_usage(&self, batch: &RecordBatch) -> ArrowMemoryEstimate {
        let num_rows = batch.num_rows();
        let num_cols = batch.num_columns();
        
        // Estimate based on Arrow memory usage
        let arrow_bytes = batch.get_array_memory_size();
        
        // Estimate Dataset conversion overhead
        let dataset_bytes = num_rows * num_cols * std::mem::size_of::<f32>();
        let total_estimated = arrow_bytes + dataset_bytes;

        ArrowMemoryEstimate {
            arrow_bytes,
            dataset_bytes,
            total_estimated,
            num_rows,
            num_cols,
        }
    }
}

/// Memory usage estimate for Arrow data loading
#[derive(Debug, Clone)]
pub struct ArrowMemoryEstimate {
    /// Estimated Arrow memory usage
    pub arrow_bytes: usize,
    /// Estimated Dataset memory usage after conversion
    pub dataset_bytes: usize,
    /// Total estimated memory usage
    pub total_estimated: usize,
    /// Number of rows
    pub num_rows: usize,
    /// Number of columns
    pub num_cols: usize,
}

#[cfg(feature = "arrow")]
impl DataLoader for ArrowLoader {
    fn load<P: AsRef<Path>>(&self, path: P) -> Result<Dataset> {
        let path_str = path.as_ref().to_string_lossy();
        
        if path_str.ends_with(".arrow") || path_str.ends_with(".ipc") {
            self.load_ipc_file(path)
        } else if path_str.ends_with(".json") {
            self.load_json(path)
        } else {
            Err(LightGBMError::data_loading(format!(
                "Unsupported file format for Arrow loader: {}", path_str
            )))
        }
    }

    fn config(&self) -> &LoaderConfig {
        &self.config
    }
}

// Stub implementation when arrow feature is not enabled
#[cfg(not(feature = "arrow"))]
pub struct ArrowLoader;

#[cfg(not(feature = "arrow"))]
impl ArrowLoader {
    pub fn new(_dataset_config: DatasetConfig) -> Result<Self> {
        Err(LightGBMError::not_implemented(
            "Arrow support requires 'arrow' feature to be enabled"
        ))
    }
}

#[cfg(not(feature = "arrow"))]
impl DataLoader for ArrowLoader {
    fn load<P: AsRef<Path>>(&self, _path: P) -> Result<Dataset> {
        Err(LightGBMError::not_implemented(
            "Arrow support requires 'arrow' feature to be enabled"
        ))
    }

    fn config(&self) -> &LoaderConfig {
        unreachable!("ArrowLoader cannot be created without arrow feature")
    }
}

#[cfg(test)]
#[cfg(feature = "arrow")]
mod tests {
    use super::*;
    use arrow::{
        array::{Float32Array, Int32Array},
        datatypes::{Field, Schema},
        record_batch::RecordBatch,
    };
    use std::sync::Arc;

    #[test]
    fn test_arrow_config_default() {
        let config = ArrowConfig::default();
        assert_eq!(config.target_column, "target");
        assert_eq!(config.batch_size, 10000);
        assert!(config.zero_copy);
        assert!(config.parallel);
    }

    #[test]
    fn test_arrow_loader_creation() -> Result<()> {
        let dataset_config = DatasetConfig::new()
            .with_target_column("target");
        let loader = ArrowLoader::new(dataset_config)?;
        assert_eq!(loader.arrow_config().target_column, "target");
        Ok(())
    }

    #[test]
    fn test_arrow_loader_configuration() -> Result<()> {
        let dataset_config = DatasetConfig::default();
        let loader = ArrowLoader::new(dataset_config)?
            .with_target_column("label")
            .with_feature_columns(vec!["feat1".to_string(), "feat2".to_string()])
            .with_batch_size(5000)
            .with_zero_copy(false)
            .with_parallel(false);

        assert_eq!(loader.arrow_config().target_column, "label");
        assert_eq!(loader.arrow_config().feature_columns, Some(vec!["feat1".to_string(), "feat2".to_string()]));
        assert_eq!(loader.arrow_config().batch_size, 5000);
        assert!(!loader.arrow_config().zero_copy);
        assert!(!loader.arrow_config().parallel);
        Ok(())
    }

    #[test]
    fn test_arrow_record_batch_loading() -> Result<()> {
        // Create test schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("feature1", DataType::Float32, false),
            Field::new("feature2", DataType::Float32, false),
            Field::new("target", DataType::Int32, false),
        ]));

        // Create test arrays
        let feature1 = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        let feature2 = Float32Array::from(vec![10.0, 20.0, 30.0, 40.0]);
        let target = Int32Array::from(vec![0, 1, 0, 1]);

        // Create RecordBatch
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(feature1),
                Arc::new(feature2),
                Arc::new(target),
            ],
        ).map_err(|e| LightGBMError::data_loading(format!("Failed to create test RecordBatch: {}", e)))?;

        // Create loader and load RecordBatch
        let dataset_config = DatasetConfig::new()
            .with_target_column("target");
        let loader = ArrowLoader::new(dataset_config)?;
        let dataset = loader.load_record_batch(&batch)?;

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
    fn test_memory_estimation() -> Result<()> {
        // Create a simple test RecordBatch
        let schema = Arc::new(Schema::new(vec![
            Field::new("feature1", DataType::Float32, false),
            Field::new("target", DataType::Int32, false),
        ]));

        let feature1 = Float32Array::from(vec![1.0, 2.0, 3.0]);
        let target = Int32Array::from(vec![0, 1, 0]);

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(feature1), Arc::new(target)],
        ).map_err(|e| LightGBMError::data_loading(format!("Failed to create test RecordBatch: {}", e)))?;

        let dataset_config = DatasetConfig::default();
        let loader = ArrowLoader::new(dataset_config)?;
        
        let estimate = loader.estimate_memory_usage(&batch);
        assert!(estimate.arrow_bytes > 0);
        assert!(estimate.dataset_bytes > 0);
        assert!(estimate.total_estimated > 0);
        assert_eq!(estimate.num_rows, 3);
        assert_eq!(estimate.num_cols, 2);
        
        Ok(())
    }
}