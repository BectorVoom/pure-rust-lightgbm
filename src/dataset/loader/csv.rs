//! CSV file loader for Pure Rust LightGBM.
//!
//! This module provides efficient CSV file loading with comprehensive error handling,
//! type detection, and missing value support.

use crate::core::types::*;
use crate::core::error::{Result, LightGBMError};
use crate::dataset::{Dataset, DatasetConfig};
use super::{DataLoader, LoaderConfig};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::fs::File;
use csv::{ReaderBuilder, StringRecord};

/// CSV-specific configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CsvConfig {
    /// Has header row
    pub has_header: bool,
    /// Field delimiter
    pub delimiter: char,
    /// Quote character
    pub quote_char: char,
    /// Escape character
    pub escape_char: Option<char>,
    /// Comment character
    pub comment_char: Option<char>,
    /// Maximum number of rows to read
    pub max_rows: Option<usize>,
    /// Skip rows from the beginning
    pub skip_rows: usize,
    /// Flexible parsing (allow variable column count)
    pub flexible: bool,
    /// Trim whitespace from fields
    pub trim: bool,
    /// Encoding
    pub encoding: String,
    /// Buffer size for reading
    pub buffer_size: usize,
}

impl Default for CsvConfig {
    fn default() -> Self {
        CsvConfig {
            has_header: true,
            delimiter: ',',
            quote_char: '"',
            escape_char: None,
            comment_char: None,
            max_rows: None,
            skip_rows: 0,
            flexible: false,
            trim: true,
            encoding: "utf-8".to_string(),
            buffer_size: 8192,
        }
    }
}

/// CSV data loader
pub struct CsvLoader {
    /// Loader configuration
    config: LoaderConfig,
    /// CSV-specific configuration
    csv_config: CsvConfig,
    /// Dataset configuration
    dataset_config: DatasetConfig,
}

impl CsvLoader {
    /// Create a new CSV loader
    pub fn new(dataset_config: DatasetConfig) -> Result<Self> {
        println!("DEBUG: CsvLoader::new called");
        Ok(CsvLoader {
            config: LoaderConfig {
                dataset_config: dataset_config.clone(),
                ..Default::default()
            },
            csv_config: CsvConfig::default(),
            dataset_config,
        })
    }

    /// Create CSV loader with custom configuration
    pub fn with_csv_config(mut self, csv_config: CsvConfig) -> Self {
        self.csv_config = csv_config;
        self
    }

    /// Set delimiter character
    pub fn with_delimiter(mut self, delimiter: char) -> Self {
        self.csv_config.delimiter = delimiter;
        self
    }

    /// Set whether file has header
    pub fn with_header(mut self, has_header: bool) -> Self {
        self.csv_config.has_header = has_header;
        self
    }

    /// Set quote character
    pub fn with_quote_char(mut self, quote_char: char) -> Self {
        self.csv_config.quote_char = quote_char;
        self
    }

    /// Set maximum rows to read
    pub fn with_max_rows(mut self, max_rows: usize) -> Self {
        self.csv_config.max_rows = Some(max_rows);
        self
    }

    /// Enable flexible parsing
    pub fn with_flexible(mut self, flexible: bool) -> Self {
        self.csv_config.flexible = flexible;
        self
    }

    /// Load CSV file
    pub fn load_csv<P: AsRef<Path>>(&self, path: P) -> Result<Dataset> {
        let path = path.as_ref();
        println!("DEBUG: CsvLoader::load_csv called for path: {}", path.display());
        log::info!("Loading CSV file: {}", path.display());

        // Validate file exists and is readable
        if !path.exists() {
            return Err(LightGBMError::data_loading(format!(
                "File does not exist: {}", path.display()
            )));
        }

        if !path.is_file() {
            return Err(LightGBMError::data_loading(format!(
                "Path is not a file: {}", path.display()
            )));
        }

        // Open file
        let file = File::open(path)
            .map_err(|e| LightGBMError::data_loading(format!(
                "Failed to open file {}: {}", path.display(), e
            )))?;

        // Configure CSV reader
        let mut reader = ReaderBuilder::new()
            .delimiter(self.csv_config.delimiter as u8)
            .quote(self.csv_config.quote_char as u8)
            .has_headers(self.csv_config.has_header)
            .flexible(self.csv_config.flexible)
            .trim(csv::Trim::All)
            .buffer_capacity(self.csv_config.buffer_size)
            .from_reader(file);

        // Handle escape character
        if let Some(escape_char) = self.csv_config.escape_char {
            reader = ReaderBuilder::new()
                .delimiter(self.csv_config.delimiter as u8)
                .quote(self.csv_config.quote_char as u8)
                .escape(Some(escape_char as u8))
                .has_headers(self.csv_config.has_header)
                .flexible(self.csv_config.flexible)
                .trim(csv::Trim::All)
                .buffer_capacity(self.csv_config.buffer_size)
                .from_reader(File::open(path).map_err(|e| {
                    LightGBMError::data_loading(format!("Failed to reopen file: {}", e))
                })?);
        }

        // Handle comment character
        if let Some(comment_char) = self.csv_config.comment_char {
            reader = ReaderBuilder::new()
                .delimiter(self.csv_config.delimiter as u8)
                .quote(self.csv_config.quote_char as u8)
                .comment(Some(comment_char as u8))
                .has_headers(self.csv_config.has_header)
                .flexible(self.csv_config.flexible)
                .trim(csv::Trim::All)
                .buffer_capacity(self.csv_config.buffer_size)
                .from_reader(File::open(path).map_err(|e| {
                    LightGBMError::data_loading(format!("Failed to reopen file: {}", e))
                })?);
        }

        // Read headers if available
        let headers = if self.csv_config.has_header {
            Some(reader.headers()
                .map_err(|e| LightGBMError::data_loading(format!("Failed to read headers: {}", e)))?
                .clone())
        } else {
            None
        };

        // Skip initial rows if configured
        let mut records_iter = reader.records();
        for _ in 0..self.csv_config.skip_rows {
            if let Some(result) = records_iter.next() {
                result.map_err(|e| LightGBMError::data_loading(format!("Failed to skip row: {}", e)))?;
            }
        }

        // Read all records
        let mut records = Vec::new();
        let mut num_columns = 0;
        let mut rows_read = 0;

        for (line_num, result) in records_iter.enumerate() {
            let record = result.map_err(|e| {
                LightGBMError::data_loading(format!(
                    "CSV parsing error at line {}: {}", 
                    line_num + self.csv_config.skip_rows + 1, 
                    e
                ))
            })?;

            // Validate column count consistency
            if num_columns == 0 {
                num_columns = record.len();
                if num_columns == 0 {
                    return Err(LightGBMError::data_loading("CSV file has no columns"));
                }
            } else if !self.csv_config.flexible && record.len() != num_columns {
                return Err(LightGBMError::data_loading(format!(
                    "Inconsistent column count at line {}: expected {}, got {}",
                    line_num + self.csv_config.skip_rows + 1, 
                    num_columns, 
                    record.len()
                )));
            }

            records.push(record);
            rows_read += 1;

            // Apply max_rows limit
            if let Some(max_rows) = self.csv_config.max_rows {
                if rows_read >= max_rows {
                    log::info!("Reached max_rows limit of {}, stopping", max_rows);
                    break;
                }
            }
        }

        if records.is_empty() {
            return Err(LightGBMError::data_loading("CSV file contains no data rows"));
        }

        log::info!("Loaded {} rows with {} columns", records.len(), num_columns);

        // Convert to Dataset
        self.convert_to_dataset(records, headers)
    }

    /// Convert CSV records to Dataset
    fn convert_to_dataset(
        &self, 
        records: Vec<StringRecord>, 
        headers: Option<StringRecord>
    ) -> Result<Dataset> {
        let num_rows = records.len();
        let num_cols = records[0].len();

        // Determine target column index
        let target_col_idx = self.determine_target_column(&headers, num_cols)?;
        
        // Determine weight column index if specified
        let weight_col_idx = self.determine_weight_column(&headers, num_cols)?;

        log::debug!("CSV parsing: num_cols={}, target_col_idx={}, weight_col_idx={:?}", 
                   num_cols, target_col_idx, weight_col_idx);

        // Determine any additional columns to exclude (like weight columns that are auto-detected)
        let mut excluded_columns = vec![target_col_idx];
        if let Some(weight_idx) = weight_col_idx {
            excluded_columns.push(weight_idx);
        }
        
        log::debug!("Starting auto-exclusion logic, excluded_columns so far: {:?}", excluded_columns);
        
        // Additionally exclude any columns that look like weight columns by name (if headers available)
        if let Some(ref headers) = headers {
            log::debug!("Available headers: {:?}", headers.iter().collect::<Vec<_>>());
            let common_weight_names = ["weight", "weights", "sample_weight", "instance_weight"];
            for (col_idx, header) in headers.iter().enumerate() {
                log::debug!("Checking header '{}' at index {} for weight patterns", header, col_idx);
                if common_weight_names.iter().any(|&name| header.eq_ignore_ascii_case(name)) {
                    if !excluded_columns.contains(&col_idx) {
                        excluded_columns.push(col_idx);
                        log::debug!("Auto-excluding weight-like column '{}' at index {}", header, col_idx);
                    }
                }
            }
        } else {
            log::debug!("No headers available for auto-exclusion");
        }

        // Separate feature columns (exclude target, weight, and auto-detected weight columns)
        let feature_cols: Vec<usize> = (0..num_cols)
            .filter(|&i| !excluded_columns.contains(&i))
            .collect();
        let num_features = feature_cols.len();

        log::debug!("CSV parsing: feature_cols={:?}, num_features={}", feature_cols, num_features);

        if num_features == 0 {
            return Err(LightGBMError::data_loading("No feature columns available after removing target and weight columns"));
        }

        // Initialize arrays
        let mut features = Array2::<f32>::zeros((num_rows, num_features));
        let mut labels = Array1::<f32>::zeros(num_rows);
        let mut missing_mask = Array2::<bool>::from_elem((num_rows, num_features), false);
        let mut weights = if weight_col_idx.is_some() {
            Some(Array1::<f32>::zeros(num_rows))
        } else {
            None
        };

        // Parse data with detailed error reporting
        for (row_idx, record) in records.iter().enumerate() {
            // Parse target/label
            let label_str = &record[target_col_idx];
            match self.try_parse_numeric_value(label_str) {
                Some(value) => {
                    labels[row_idx] = value;
                }
                None => {
                    // Handle missing target value
                    if self.dataset_config.use_missing_as_zero {
                        labels[row_idx] = 0.0;
                    } else {
                        labels[row_idx] = f32::NAN;
                    }
                    log::debug!("Missing target value at row {}, using NaN", row_idx + 1);
                }
            }

            // Parse weight if present
            if let (Some(ref mut weights_array), Some(weight_idx)) = (weights.as_mut(), weight_col_idx) {
                let weight_str = &record[weight_idx];
                match self.try_parse_numeric_value(weight_str) {
                    Some(value) => {
                        weights_array[row_idx] = value;
                    }
                    None => {
                        // Handle missing weight value - default to 1.0 for missing weights
                        weights_array[row_idx] = 1.0;
                        log::debug!("Missing weight value at row {}, using default weight 1.0", row_idx + 1);
                    }
                }
            }

            // Parse features
            for (feat_idx, &col_idx) in feature_cols.iter().enumerate() {
                let value_str = &record[col_idx];
                match self.try_parse_numeric_value(value_str) {
                    Some(value) => {
                        features[[row_idx, feat_idx]] = value;
                        missing_mask[[row_idx, feat_idx]] = false;
                    }
                    None => {
                        println!("DEBUG: Missing value detected at row {}, feature {}, value: '{}'", row_idx, feat_idx, value_str);
                        // Handle missing value based on configuration
                        if self.dataset_config.use_missing_as_zero {
                            features[[row_idx, feat_idx]] = 0.0;
                        } else {
                            features[[row_idx, feat_idx]] = f32::NAN;
                        }
                        missing_mask[[row_idx, feat_idx]] = true;
                    }
                }
            }
        }

        // Generate feature names
        let feature_names = self.generate_feature_names(&headers, &feature_cols);

        // Create dataset with metadata
        let mut dataset = Dataset::new(
            features,
            labels,
            weights, // weights parsed from CSV
            None, // groups - can be added later
            Some(feature_names),
            None, // feature_types - will be auto-detected
        )?;

        // Set missing values mask if any missing values detected
        let has_missing = missing_mask.iter().any(|&x| x);
        let missing_count = missing_mask.iter().filter(|&&x| x).count();
        
        println!("DEBUG: Missing values detected: {}, count: {}", has_missing, missing_count);
        println!("DEBUG: Missing mask shape: {:?}", missing_mask.dim());
        
        if has_missing {
            dataset.set_missing_values(missing_mask)?;
        }

        // Set metadata
        let metadata = dataset.metadata_mut();
        metadata.source_path = Some(format!("CSV file loaded"));
        metadata.format = "csv".to_string();
        metadata.properties.insert("num_rows".to_string(), num_rows.to_string());
        metadata.properties.insert("num_cols".to_string(), num_cols.to_string());
        metadata.properties.insert("has_header".to_string(), self.csv_config.has_header.to_string());
        metadata.properties.insert("delimiter".to_string(), self.csv_config.delimiter.to_string());

        Ok(dataset)
    }

    /// Determine target column index
    fn determine_target_column(&self, headers: &Option<StringRecord>, num_cols: usize) -> Result<usize> {
        if let Some(ref target_col) = self.dataset_config.target_column {
            if let Some(ref headers) = headers {
                // Search by column name
                headers.iter()
                    .position(|h| h == target_col)
                    .ok_or_else(|| LightGBMError::data_loading(format!(
                        "Target column '{}' not found in headers", target_col
                    )))
            } else {
                // Parse as column index
                target_col.parse::<usize>()
                    .map_err(|_| LightGBMError::data_loading(
                        "Target column must be column name (with headers) or numeric index"
                    ))
                    .and_then(|idx| {
                        if idx >= num_cols {
                            Err(LightGBMError::data_loading(format!(
                                "Target column index {} out of bounds (num_cols: {})", idx, num_cols
                            )))
                        } else {
                            Ok(idx)
                        }
                    })
            }
        } else {
            // Default: last column is target
            Ok(num_cols - 1)
        }
    }

    /// Determine weight column index (returns None if no weight column specified)
    fn determine_weight_column(&self, headers: &Option<StringRecord>, num_cols: usize) -> Result<Option<usize>> {
        if let Some(ref weight_col) = self.dataset_config.weight_column {
            if let Some(ref headers) = headers {
                // Search by column name
                headers.iter()
                    .position(|h| h == weight_col)
                    .map(Some)
                    .ok_or_else(|| LightGBMError::data_loading(format!(
                        "Weight column '{}' not found in headers", weight_col
                    )))
            } else {
                // Parse as column index
                weight_col.parse::<usize>()
                    .map_err(|_| LightGBMError::data_loading(
                        "Weight column must be column name (with headers) or numeric index"
                    ))
                    .and_then(|idx| {
                        if idx >= num_cols {
                            Err(LightGBMError::data_loading(format!(
                                "Weight column index {} out of bounds (num_cols: {})", idx, num_cols
                            )))
                        } else {
                            Ok(Some(idx))
                        }
                    })
            }
        } else {
            // No weight column specified
            Ok(None)
        }
    }

    /// Generate feature names
    fn generate_feature_names(&self, headers: &Option<StringRecord>, feature_cols: &[usize]) -> Vec<String> {
        if let Some(ref headers) = headers {
            feature_cols.iter()
                .map(|&i| headers[i].to_string())
                .collect()
        } else {
            feature_cols.iter()
                .map(|&i| format!("feature_{}", i))
                .collect()
        }
    }

    /// Parse numeric value with error context
    fn parse_numeric_value(&self, value: &str, row: usize, col: usize, is_target: bool) -> Result<f32> {
        self.try_parse_numeric_value(value)
            .ok_or_else(|| {
                let col_type = if is_target { "target" } else { "feature" };
                LightGBMError::data_loading(format!(
                    "Invalid {} value '{}' at row {}, column {}", 
                    col_type, value, row + 1, col
                ))
            })
    }

    /// Try to parse numeric value, returning None for missing values
    fn try_parse_numeric_value(&self, value: &str) -> Option<f32> {
        let trimmed = value.trim();

        // Check for common missing value representations
        if trimmed.is_empty()
            || trimmed.eq_ignore_ascii_case("na")
            || trimmed.eq_ignore_ascii_case("nan")
            || trimmed.eq_ignore_ascii_case("null")
            || trimmed.eq_ignore_ascii_case("none")
            || trimmed.eq_ignore_ascii_case("#n/a")
            || trimmed == "?"
            || trimmed == "-"
            || trimmed == "."
        {
            return None;
        }

        // Try to parse as float
        trimmed.parse::<f32>().ok().and_then(|f| {
            if f.is_finite() {
                Some(f)
            } else {
                None
            }
        })
    }

    /// Infer data types from a sample of the data
    pub fn infer_types<P: AsRef<Path>>(&self, path: P, sample_rows: usize) -> Result<Vec<FeatureType>> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|e| LightGBMError::data_loading(format!("Failed to open file: {}", e)))?;

        let mut reader = ReaderBuilder::new()
            .delimiter(self.csv_config.delimiter as u8)
            .has_headers(self.csv_config.has_header)
            .flexible(self.csv_config.flexible)
            .from_reader(file);

        let mut sample_records = Vec::new();
        for (i, result) in reader.records().enumerate() {
            if i >= sample_rows {
                break;
            }
            let record = result.map_err(|e| LightGBMError::data_loading(format!("Failed to read sample: {}", e)))?;
            sample_records.push(record);
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
            let mut all_integers = true;

            for record in &sample_records {
                if col_idx < record.len() {
                    let value = record[col_idx].trim();
                    if !value.is_empty() {
                        total_count += 1;
                        unique_values.insert(value.to_string());

                        if let Some(parsed_value) = self.try_parse_numeric_value(value) {
                            numeric_count += 1;
                            if parsed_value.fract() != 0.0 {
                                all_integers = false;
                            }
                        }
                    }
                }
            }

            // Determine type based on analysis
            let numeric_ratio = if total_count > 0 {
                numeric_count as f64 / total_count as f64
            } else {
                0.0
            };

            // Heuristics for categorical detection:
            // 1. Less than 90% numeric values
            // 2. Few unique values (≤ 20)
            // 3. All integers with few unique values
            if numeric_ratio < 0.9 
                || unique_values.len() <= 20 
                || (all_integers && unique_values.len() <= 50) {
                column_types[col_idx] = FeatureType::Categorical;
            }
        }

        Ok(column_types)
    }

    /// Get configuration
    pub fn csv_config(&self) -> &CsvConfig {
        &self.csv_config
    }

    /// Estimate memory usage for loading
    pub fn estimate_memory_usage<P: AsRef<Path>>(&self, path: P) -> Result<CsvMemoryEstimate> {
        let path = path.as_ref();
        let metadata = std::fs::metadata(path)
            .map_err(|e| LightGBMError::data_loading(format!("Failed to read file metadata: {}", e)))?;

        let file_size = metadata.len() as usize;
        
        // Rough estimates based on file size
        // CSV parsing typically uses 2-3x file size during processing
        let parsing_overhead = file_size * 2;
        
        // Final dataset size depends on data types (assume mostly f32)
        let estimated_rows = file_size / 50; // Rough estimate: 50 bytes per row average
        let estimated_cols = 10; // Default estimate
        let dataset_size = estimated_rows * estimated_cols * std::mem::size_of::<f32>();

        Ok(CsvMemoryEstimate {
            file_size,
            parsing_overhead,
            dataset_size,
            total_estimated: file_size + parsing_overhead + dataset_size,
        })
    }
}

/// Memory usage estimate for CSV loading
#[derive(Debug, Clone)]
pub struct CsvMemoryEstimate {
    /// Original file size in bytes
    pub file_size: usize,
    /// Estimated overhead during parsing
    pub parsing_overhead: usize,
    /// Estimated final dataset size
    pub dataset_size: usize,
    /// Total estimated memory usage
    pub total_estimated: usize,
}

impl DataLoader for CsvLoader {
    fn load<P: AsRef<Path>>(&self, path: P) -> Result<Dataset> {
        self.load_csv(path)
    }

    fn config(&self) -> &LoaderConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_csv_config_default() {
        let config = CsvConfig::default();
        assert!(config.has_header);
        assert_eq!(config.delimiter, ',');
        assert_eq!(config.quote_char, '"');
        assert!(config.trim);
        assert_eq!(config.encoding, "utf-8");
    }

    #[test]
    fn test_csv_loader_creation() {
        let dataset_config = DatasetConfig::default();
        let loader = CsvLoader::new(dataset_config).unwrap();
        assert_eq!(loader.csv_config().delimiter, ',');
        assert!(loader.csv_config().has_header);
    }

    #[test]
    fn test_csv_loader_configuration() {
        let dataset_config = DatasetConfig::default();
        let loader = CsvLoader::new(dataset_config)
            .unwrap()
            .with_delimiter(';')
            .with_header(false)
            .with_max_rows(1000)
            .with_flexible(true);

        assert_eq!(loader.csv_config().delimiter, ';');
        assert!(!loader.csv_config().has_header);
        assert_eq!(loader.csv_config().max_rows, Some(1000));
        assert!(loader.csv_config().flexible);
    }

    #[test]
    fn test_parse_numeric_values() {
        let dataset_config = DatasetConfig::default();
        let loader = CsvLoader::new(dataset_config).unwrap();

        // Valid numeric values
        assert_eq!(loader.try_parse_numeric_value("123"), Some(123.0));
        assert_eq!(loader.try_parse_numeric_value("123.45"), Some(123.45));
        assert_eq!(loader.try_parse_numeric_value("-123.45"), Some(-123.45));
        assert_eq!(loader.try_parse_numeric_value("  123.45  "), Some(123.45));

        // Missing value representations
        assert_eq!(loader.try_parse_numeric_value(""), None);
        assert_eq!(loader.try_parse_numeric_value("NA"), None);
        assert_eq!(loader.try_parse_numeric_value("nan"), None);
        assert_eq!(loader.try_parse_numeric_value("NULL"), None);
        assert_eq!(loader.try_parse_numeric_value("?"), None);
        assert_eq!(loader.try_parse_numeric_value("-"), None);

        // Invalid values
        assert_eq!(loader.try_parse_numeric_value("abc"), None);
        assert_eq!(loader.try_parse_numeric_value("12.34.56"), None);
    }

    #[test]
    fn test_csv_loading() -> Result<()> {
        // Create a temporary CSV file
        let mut temp_file = NamedTempFile::new()
            .map_err(|e| LightGBMError::data_loading(format!("Failed to create temp file: {}", e)))?;
        
        writeln!(temp_file, "feature1,feature2,target")
            .map_err(|e| LightGBMError::data_loading(format!("Failed to write to temp file: {}", e)))?;
        writeln!(temp_file, "1.0,2.0,0")
            .map_err(|e| LightGBMError::data_loading(format!("Failed to write to temp file: {}", e)))?;
        writeln!(temp_file, "3.0,4.0,1")
            .map_err(|e| LightGBMError::data_loading(format!("Failed to write to temp file: {}", e)))?;
        writeln!(temp_file, "5.0,6.0,0")
            .map_err(|e| LightGBMError::data_loading(format!("Failed to write to temp file: {}", e)))?;

        // Configure and load
        let dataset_config = DatasetConfig::new()
            .with_target_column("target");
        let loader = CsvLoader::new(dataset_config)?;
        let dataset = loader.load(temp_file.path())?;

        // Verify dataset properties
        assert_eq!(dataset.num_data(), 3);
        assert_eq!(dataset.num_features(), 2);
        assert_eq!(dataset.feature_names().unwrap().len(), 2);
        assert_eq!(dataset.feature_names().unwrap()[0], "feature1");
        assert_eq!(dataset.feature_names().unwrap()[1], "feature2");

        // Verify data
        let features = dataset.features();
        assert_eq!(features[[0, 0]], 1.0);
        assert_eq!(features[[0, 1]], 2.0);
        assert_eq!(features[[1, 0]], 3.0);
        assert_eq!(features[[1, 1]], 4.0);
        assert_eq!(features[[2, 0]], 5.0);
        assert_eq!(features[[2, 1]], 6.0);

        let labels = dataset.labels();
        assert_eq!(labels[0], 0.0);
        assert_eq!(labels[1], 1.0);
        assert_eq!(labels[2], 0.0);

        Ok(())
    }

    #[test]
    fn test_csv_with_missing_values() -> Result<()> {
        // Create a temporary CSV file with missing values
        let mut temp_file = NamedTempFile::new()
            .map_err(|e| LightGBMError::data_loading(format!("Failed to create temp file: {}", e)))?;
        
        writeln!(temp_file, "feature1,feature2,target")
            .map_err(|e| LightGBMError::data_loading(format!("Failed to write to temp file: {}", e)))?;
        writeln!(temp_file, "1.0,NA,0")
            .map_err(|e| LightGBMError::data_loading(format!("Failed to write to temp file: {}", e)))?;
        writeln!(temp_file, ",4.0,1")
            .map_err(|e| LightGBMError::data_loading(format!("Failed to write to temp file: {}", e)))?;
        writeln!(temp_file, "5.0,6.0,0")
            .map_err(|e| LightGBMError::data_loading(format!("Failed to write to temp file: {}", e)))?;

        // Configure and load
        let dataset_config = DatasetConfig::new()
            .with_target_column("target");
        let loader = CsvLoader::new(dataset_config)?;
        let dataset = loader.load(temp_file.path())?;

        // Verify missing values are handled
        assert_eq!(dataset.num_data(), 3);
        assert_eq!(dataset.num_features(), 2);
        assert!(dataset.has_missing_values());

        let features = dataset.features();
        assert_eq!(features[[0, 0]], 1.0);
        assert!(features[[0, 1]].is_nan());
        assert!(features[[1, 0]].is_nan());
        assert_eq!(features[[1, 1]], 4.0);
        assert_eq!(features[[2, 0]], 5.0);
        assert_eq!(features[[2, 1]], 6.0);

        Ok(())
    }

    #[test]
    fn test_memory_estimation() {
        let dataset_config = DatasetConfig::default();
        let loader = CsvLoader::new(dataset_config).unwrap();
        
        // Create a small temporary file for testing
        let temp_file = NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), "a,b,c\n1,2,3\n").unwrap();
        
        let estimate = loader.estimate_memory_usage(temp_file.path()).unwrap();
        assert!(estimate.file_size > 0);
        assert!(estimate.total_estimated > estimate.file_size);
    }
}