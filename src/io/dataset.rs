//! Dataset I/O operations for LightGBM.
//!
//! This module provides functionality for loading, saving, and serializing
//! LightGBM datasets, corresponding to the C++ dataset.cpp implementation.

use crate::core::error::{LightGBMError, Result};
use crate::core::types::*;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use serde::{Deserialize, Serialize};

/// Binary file token for LightGBM files
const BINARY_FILE_TOKEN: &str = "______LightGBM_Binary_File_Token______\n";

/// Binary serialized reference token
const BINARY_SERIALIZED_REFERENCE_TOKEN: &str = "______LightGBM_Binary_Serialized_Token______\n";

/// Serialized reference version
const SERIALIZED_REFERENCE_VERSION: &str = "v1";

/// Dataset I/O operations handler
#[derive(Debug)]
pub struct DatasetIO {
    /// Data filename
    data_filename: String,
    /// Number of data points
    num_data: DataSize,
    /// Whether loading is finished
    is_finish_load: bool,
    /// Whether waiting for manual finish
    wait_for_manual_finish: bool,
    /// Whether has raw data
    has_raw: bool,
    /// Number of features
    num_features: usize,
    /// Number of groups
    num_groups: i32,
    /// Group bin boundaries
    group_bin_boundaries: Vec<u32>,
    /// Feature groups
    feature_groups: Vec<Box<dyn FeatureGroup>>,
    /// Metadata
    metadata: DatasetMetadata,
    /// Numeric feature map
    numeric_feature_map: Vec<i32>,
    /// Device type
    device_type: String,
    /// GPU device ID
    gpu_device_id: i32,
}

/// Feature group trait for polymorphic behavior
pub trait FeatureGroup: std::fmt::Debug {
    /// Finishes loading the feature group
    fn finish_load(&mut self) -> Result<()>;
    /// Returns the size in bytes for serialization
    fn sizes_in_byte(&self) -> usize;
    /// Serializes the feature group to binary format
    fn serialize_to_binary(&self, writer: &mut dyn Write) -> Result<()>;
}

/// Dataset metadata for I/O operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Number of data points
    pub num_data: DataSize,
    /// Number of features
    pub num_features: usize,
    /// Labels
    pub labels: Vec<f32>,
    /// Weights
    pub weights: Option<Vec<f32>>,
    /// Groups (for ranking)
    pub groups: Option<Vec<DataSize>>,
    /// Query boundaries
    pub query_boundaries: Option<Vec<DataSize>>,
    /// Custom metadata properties
    pub properties: HashMap<String, String>,
}

impl Default for DatasetIO {
    fn default() -> Self {
        Self::new()
    }
}

impl DatasetIO {
    /// Create a new DatasetIO instance (equivalent to C++ Dataset::Dataset())
    pub fn new() -> Self {
        DatasetIO {
            data_filename: "noname".to_string(),
            num_data: 0,
            is_finish_load: false,
            wait_for_manual_finish: false,
            has_raw: false,
            num_features: 0,
            num_groups: 0,
            group_bin_boundaries: vec![0],
            feature_groups: Vec::new(),
            metadata: DatasetMetadata::default(),
            numeric_feature_map: Vec::new(),
            device_type: "cpu".to_string(),
            gpu_device_id: 0,
        }
    }

    /// Create a new DatasetIO instance with specified number of data points
    /// (equivalent to C++ Dataset::Dataset(data_size_t num_data))
    pub fn with_num_data(num_data: DataSize) -> Result<Self> {
        if num_data <= 0 {
            return Err(LightGBMError::invalid_parameter(
                "num_data",
                num_data.to_string(),
                "must be greater than 0",
            ));
        }

        let mut dataset = Self::new();
        dataset.num_data = num_data;
        dataset.metadata.init(num_data, None, None)?;
        Ok(dataset)
    }

    /// Finish loading the dataset (equivalent to C++ Dataset::FinishLoad())
    pub fn finish_load(&mut self) -> Result<()> {
        if self.is_finish_load {
            return Ok(());
        }

        // Finish loading feature groups
        if self.num_groups > 0 {
            for group in &mut self.feature_groups {
                group.finish_load()?;
            }
        }

        // Finish loading metadata
        self.metadata.finish_load()?;

        // TODO: Add CUDA support when available
        #[cfg(feature = "gpu")]
        if self.device_type == "cuda" {
            self.create_cuda_column_data()?;
            self.metadata.create_cuda_metadata(self.gpu_device_id)?;
        }

        self.is_finish_load = true;
        Ok(())
    }

    /// Save dataset to binary file (equivalent to C++ Dataset::SaveBinaryFile())
    pub fn save_binary_file(&self, bin_filename: Option<&str>) -> Result<()> {
        let bin_filename = match bin_filename {
            Some(filename) if !filename.is_empty() => {
                if filename == self.data_filename {
                    eprintln!("Warning: Binary file {} already exists", filename);
                    return Ok(());
                }
                filename.to_string()
            }
            _ => format!("{}.bin", self.data_filename),
        };

        if Path::new(&bin_filename).exists() {
            return Err(LightGBMError::io_error(format!(
                "File {} exists, cannot save binary to it",
                bin_filename
            )));
        }

        let file = File::create(&bin_filename).map_err(|e| {
            LightGBMError::io_error(format!("Cannot create binary file {}: {}", bin_filename, e))
        })?;

        let mut writer = BufWriter::new(file);

        println!("Saving data to binary file {}", bin_filename);

        // Write binary file token
        writer.write_all(BINARY_FILE_TOKEN.as_bytes())?;

        // Write header information
        self.serialize_header(&mut writer)?;

        // Write metadata size and data
        let metadata_size = self.metadata.sizes_in_byte();
        writer.write_all(&metadata_size.to_le_bytes())?;
        self.metadata.save_binary_to_file(&mut writer)?;

        // Write feature groups
        for group in &self.feature_groups {
            let feature_size = group.sizes_in_byte();
            writer.write_all(&feature_size.to_le_bytes())?;
            group.serialize_to_binary(&mut writer)?;
        }

        // Write raw data if available
        if self.has_raw {
            self.write_raw_data(&mut writer)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Serialize dataset header (equivalent to C++ Dataset::SerializeHeader())
    pub fn serialize_header(&self, writer: &mut dyn Write) -> Result<()> {
        // Write version information
        writer.write_all(&2u32.to_le_bytes())?; // version length
        writer.write_all(SERIALIZED_REFERENCE_VERSION.as_bytes())?;
        
        // Write basic dataset information
        writer.write_all(&self.num_data.to_le_bytes())?;
        writer.write_all(&(self.num_features as u32).to_le_bytes())?;
        writer.write_all(&self.num_groups.to_le_bytes())?;
        
        // Write group boundaries
        writer.write_all(&(self.group_bin_boundaries.len() as u32).to_le_bytes())?;
        for &boundary in &self.group_bin_boundaries {
            writer.write_all(&boundary.to_le_bytes())?;
        }

        Ok(())
    }

    /// Serialize reference data (equivalent to C++ Dataset::SerializeReference())
    pub fn serialize_reference(&self, buffer: &mut Vec<u8>) -> Result<()> {
        // Write serialized reference token
        buffer.extend_from_slice(BINARY_SERIALIZED_REFERENCE_TOKEN.as_bytes());
        
        // Write header
        let mut temp_buffer = Vec::new();
        self.serialize_header(&mut temp_buffer)?;
        buffer.extend_from_slice(&temp_buffer);

        // Write metadata
        let metadata_size = self.metadata.sizes_in_byte();
        buffer.extend_from_slice(&metadata_size.to_le_bytes());
        self.metadata.serialize_to_buffer(buffer)?;

        Ok(())
    }

    /// Copy feature mapper from another dataset (equivalent to C++ Dataset::CopyFeatureMapperFrom())
    pub fn copy_feature_mapper_from(&mut self, other: &DatasetIO) -> Result<()> {
        if other.num_groups <= 0 {
            return Err(LightGBMError::dataset("Source dataset has no feature groups"));
        }

        self.num_features = other.num_features;
        self.num_groups = other.num_groups;
        self.group_bin_boundaries = other.group_bin_boundaries.clone();
        
        // Deep copy feature groups would require more complex trait design
        // For now, we'll copy the basic structure
        self.numeric_feature_map = other.numeric_feature_map.clone();

        Ok(())
    }

    /// Create validation dataset (equivalent to C++ Dataset::CreateValid())
    pub fn create_valid(&mut self, reference: &DatasetIO) -> Result<()> {
        self.copy_feature_mapper_from(reference)?;
        self.metadata.init(self.num_data, None, None)?;
        Ok(())
    }

    /// Resize dataset (equivalent to C++ Dataset::ReSize())
    pub fn resize(&mut self, num_data: DataSize) -> Result<()> {
        if self.num_data == num_data {
            return Ok(());
        }

        self.num_data = num_data;
        self.metadata.resize(num_data)?;

        // Resize feature groups
        for _group in &mut self.feature_groups {
            // Feature groups would need resize capability
            // This is a placeholder for the actual implementation
        }

        Ok(())
    }

    /// Set float field from raw data (equivalent to C++ Dataset::SetFloatField())
    pub fn set_float_field(&mut self, field_name: &str, field_data: &[f32]) -> Result<bool> {
        match field_name {
            "label" => {
                if field_data.len() != self.num_data as usize {
                    return Err(LightGBMError::dimension_mismatch(
                        format!("num_data: {}", self.num_data),
                        format!("label data length: {}", field_data.len()),
                    ));
                }
                self.metadata.labels = field_data.to_vec();
                Ok(true)
            }
            "weight" => {
                if field_data.len() != self.num_data as usize {
                    return Err(LightGBMError::dimension_mismatch(
                        format!("num_data: {}", self.num_data),
                        format!("weight data length: {}", field_data.len()),
                    ));
                }
                self.metadata.weights = Some(field_data.to_vec());
                Ok(true)
            }
            _ => Ok(false), // Unknown field
        }
    }

    /// Set integer field from raw data (equivalent to C++ Dataset::SetIntField())
    pub fn set_int_field(&mut self, field_name: &str, field_data: &[i32]) -> Result<bool> {
        match field_name {
            "group" => {
                if field_data.len() != self.num_data as usize {
                    return Err(LightGBMError::dimension_mismatch(
                        format!("num_data: {}", self.num_data),
                        format!("group data length: {}", field_data.len()),
                    ));
                }
                self.metadata.groups = Some(field_data.iter().map(|&x| x as DataSize).collect());
                Ok(true)
            }
            _ => Ok(false), // Unknown field
        }
    }

    /// Get float field data (equivalent to C++ Dataset::GetFloatField())
    pub fn get_float_field(&self, field_name: &str) -> Result<Option<&[f32]>> {
        match field_name {
            "label" => Ok(Some(&self.metadata.labels)),
            "weight" => Ok(self.metadata.weights.as_deref()),
            _ => Ok(None), // Unknown field
        }
    }

    /// Get integer field data (equivalent to C++ Dataset::GetIntField())
    pub fn get_int_field(&self, field_name: &str) -> Result<Option<Vec<i32>>> {
        match field_name {
            "group" => Ok(self.metadata.groups.as_ref().map(|g| g.iter().map(|&x| x as i32).collect())),
            _ => Ok(None), // Unknown field
        }
    }

    /// Write raw data to binary format
    fn write_raw_data(&self, _writer: &mut dyn Write) -> Result<()> {
        // This would require access to the raw feature data
        // Implementation depends on how raw data is stored
        // For now, this is a placeholder
        Ok(())
    }

    /// Get data filename
    pub fn data_filename(&self) -> &str {
        &self.data_filename
    }

    /// Set data filename  
    pub fn set_data_filename(&mut self, filename: String) {
        self.data_filename = filename;
    }

    /// Get number of data points
    pub fn num_data(&self) -> DataSize {
        self.num_data
    }

    /// Get number of features
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Check if loading is finished
    pub fn is_finish_load(&self) -> bool {
        self.is_finish_load
    }

    /// Get metadata reference
    pub fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    /// Get mutable metadata reference
    pub fn metadata_mut(&mut self) -> &mut DatasetMetadata {
        &mut self.metadata
    }

    #[cfg(feature = "gpu")]
    fn create_cuda_column_data(&mut self) -> Result<()> {
        // TODO: Implement CUDA column data creation
        Ok(())
    }
}

impl DatasetMetadata {
    /// Create default metadata
    pub fn default() -> Self {
        DatasetMetadata {
            num_data: 0,
            num_features: 0,
            labels: Vec::new(),
            weights: None,
            groups: None,
            query_boundaries: None,
            properties: HashMap::new(),
        }
    }

    /// Initialize metadata (equivalent to C++ Metadata::Init())
    pub fn init(&mut self, num_data: DataSize, _label_idx: Option<i32>, weight_idx: Option<i32>) -> Result<()> {
        self.num_data = num_data;
        self.labels.resize(num_data as usize, 0.0);
        
        if weight_idx.is_some() {
            self.weights = Some(vec![1.0; num_data as usize]);
        }

        Ok(())
    }

    /// Finish loading metadata (equivalent to C++ Metadata::FinishLoad())
    pub fn finish_load(&mut self) -> Result<()> {
        // Process query boundaries if groups are present
        if let Some(ref groups) = self.groups {
            self.query_boundaries = Some(self.calculate_query_boundaries(groups)?);
        }
        Ok(())
    }

    /// Resize metadata (equivalent to C++ Metadata::Resize())
    pub fn resize(&mut self, num_data: DataSize) -> Result<()> {
        self.num_data = num_data;
        self.labels.resize(num_data as usize, 0.0);
        
        if let Some(ref mut weights) = self.weights {
            weights.resize(num_data as usize, 1.0);
        }

        if let Some(ref mut groups) = self.groups {
            groups.resize(num_data as usize, 0);
        }

        Ok(())
    }

    /// Get size in bytes for serialization
    pub fn sizes_in_byte(&self) -> usize {
        let mut size = 0;
        
        // Basic fields
        size += std::mem::size_of::<DataSize>(); // num_data
        size += std::mem::size_of::<usize>(); // num_features
        
        // Labels
        size += std::mem::size_of::<usize>(); // labels length
        size += self.labels.len() * std::mem::size_of::<f32>();
        
        // Weights (optional)
        size += std::mem::size_of::<bool>(); // has_weights flag
        if let Some(ref weights) = self.weights {
            size += weights.len() * std::mem::size_of::<f32>();
        }
        
        // Groups (optional)
        size += std::mem::size_of::<bool>(); // has_groups flag
        if let Some(ref groups) = self.groups {
            size += groups.len() * std::mem::size_of::<DataSize>();
        }

        size
    }

    /// Save binary data to file
    pub fn save_binary_to_file(&self, writer: &mut dyn Write) -> Result<()> {
        // Write basic information
        writer.write_all(&self.num_data.to_le_bytes())?;
        writer.write_all(&self.num_features.to_le_bytes())?;
        
        // Write labels
        writer.write_all(&self.labels.len().to_le_bytes())?;
        for &label in &self.labels {
            writer.write_all(&label.to_le_bytes())?;
        }
        
        // Write weights
        let has_weights = self.weights.is_some();
        writer.write_all(&[has_weights as u8])?;
        if let Some(ref weights) = self.weights {
            for &weight in weights {
                writer.write_all(&weight.to_le_bytes())?;
            }
        }
        
        // Write groups
        let has_groups = self.groups.is_some();
        writer.write_all(&[has_groups as u8])?;
        if let Some(ref groups) = self.groups {
            for &group in groups {
                writer.write_all(&group.to_le_bytes())?;
            }
        }

        Ok(())
    }

    /// Serialize to buffer
    pub fn serialize_to_buffer(&self, buffer: &mut Vec<u8>) -> Result<()> {
        self.save_binary_to_file(buffer)?;
        Ok(())
    }

    /// Calculate query boundaries from group data
    fn calculate_query_boundaries(&self, groups: &[DataSize]) -> Result<Vec<DataSize>> {
        let mut boundaries = vec![0];
        let mut current_group = groups.get(0).copied().unwrap_or(0);
        let mut current_count = 0;

        for &group in groups {
            if group != current_group {
                boundaries.push(current_count);
                current_group = group;
            }
            current_count += 1;
        }
        
        boundaries.push(current_count);
        Ok(boundaries)
    }

    #[cfg(feature = "gpu")]
    pub fn create_cuda_metadata(&mut self, device_id: i32) -> Result<()> {
        // TODO: Implement CUDA metadata creation
        Ok(())
    }
}

/// Helper functions (equivalent to C++ utility functions)

/// Create one feature per group mapping (equivalent to C++ OneFeaturePerGroup())
pub fn one_feature_per_group(used_features: &[i32]) -> Vec<Vec<i32>> {
    used_features.iter().map(|&feature| vec![feature]).collect()
}

/// Get conflict count for feature grouping (equivalent to C++ GetConflictCount())
pub fn get_conflict_count(mark: &[bool], indices: &[i32], max_cnt: DataSize) -> Option<i32> {
    let mut count = 0;
    for &index in indices {
        if index >= 0 && (index as usize) < mark.len() && mark[index as usize] {
            count += 1;
            if count > max_cnt {
                return None; // Conflict count exceeds limit
            }
        }
    }
    Some(count)
}

/// Mark indices as used (equivalent to C++ MarkUsed())
pub fn mark_used(mark: &mut [bool], indices: &[i32]) {
    for &index in indices {
        if index >= 0 && (index as usize) < mark.len() {
            mark[index as usize] = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_io_creation() {
        let dataset = DatasetIO::new();
        assert_eq!(dataset.data_filename(), "noname");
        assert_eq!(dataset.num_data(), 0);
        assert!(!dataset.is_finish_load());
    }

    #[test]
    fn test_dataset_io_with_num_data() {
        let dataset = DatasetIO::with_num_data(100).unwrap();
        assert_eq!(dataset.num_data(), 100);
        assert_eq!(dataset.metadata.num_data, 100);
    }

    #[test]
    fn test_invalid_num_data() {
        let result = DatasetIO::with_num_data(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_set_float_field() {
        let mut dataset = DatasetIO::with_num_data(3).unwrap();
        let labels = vec![1.0, 0.0, 1.0];
        
        let result = dataset.set_float_field("label", &labels).unwrap();
        assert!(result);
        assert_eq!(dataset.metadata.labels, labels);
    }

    #[test]
    fn test_one_feature_per_group() {
        let features = vec![0, 1, 2];
        let groups = one_feature_per_group(&features);
        
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0], vec![0]);
        assert_eq!(groups[1], vec![1]);
        assert_eq!(groups[2], vec![2]);
    }

    #[test]
    fn test_get_conflict_count() {
        let mark = vec![true, false, true, false];
        let indices = vec![0, 2];
        
        let count = get_conflict_count(&mark, &indices, 10);
        assert_eq!(count, Some(2));
    }

    #[test]
    fn test_mark_used() {
        let mut mark = vec![false; 4];
        let indices = vec![0, 2];
        
        mark_used(&mut mark, &indices);
        assert!(mark[0]);
        assert!(!mark[1]);
        assert!(mark[2]);
        assert!(!mark[3]);
    }
}