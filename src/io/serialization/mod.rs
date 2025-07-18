//! Model serialization module for Pure Rust LightGBM.
//!
//! This module provides serialization and deserialization capabilities for
//! LightGBM models in various formats including native Rust bincode, JSON,
//! and compatibility with original LightGBM model files.

pub mod bincode;
pub mod json;
pub mod lightgbm;

use crate::core::error::{Result, LightGBMError};
use crate::io::SerializableModel;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

/// Supported serialization formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SerializationFormat {
    /// Native Rust bincode format (fast, compact)
    Bincode,
    /// JSON format (human-readable, portable)
    Json,
    /// LightGBM text format (compatible with original LightGBM)
    LightGbm,
}

impl Default for SerializationFormat {
    fn default() -> Self {
        SerializationFormat::Bincode
    }
}

impl std::fmt::Display for SerializationFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SerializationFormat::Bincode => write!(f, "bincode"),
            SerializationFormat::Json => write!(f, "json"),
            SerializationFormat::LightGbm => write!(f, "lightgbm"),
        }
    }
}

impl std::str::FromStr for SerializationFormat {
    type Err = LightGBMError;
    
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "bincode" | "bin" => Ok(SerializationFormat::Bincode),
            "json" => Ok(SerializationFormat::Json),
            "lightgbm" | "lgb" | "txt" => Ok(SerializationFormat::LightGbm),
            _ => Err(LightGBMError::serialization(format!("Unknown format: {}", s))),
        }
    }
}

/// Configuration for serialization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SerializationConfig {
    /// Serialization format
    pub format: SerializationFormat,
    /// Pretty print JSON (only for JSON format)
    pub pretty_json: bool,
    /// Include metadata in serialization
    pub include_metadata: bool,
    /// Include feature importance
    pub include_feature_importance: bool,
    /// Include training history
    pub include_training_history: bool,
    /// Compression level (0-9)
    pub compression_level: u8,
    /// Validate before serialization
    pub validate_before_serialize: bool,
    /// Skip validation during deserialization
    pub skip_validation: bool,
}

impl Default for SerializationConfig {
    fn default() -> Self {
        SerializationConfig {
            format: SerializationFormat::Bincode,
            pretty_json: false,
            include_metadata: true,
            include_feature_importance: true,
            include_training_history: false,
            compression_level: 3,
            validate_before_serialize: true,
            skip_validation: false,
        }
    }
}

/// Serialization error types
#[derive(Debug, thiserror::Error)]
pub enum SerializationError {
    #[error("Serialization failed: {0}")]
    SerializationFailed(String),
    #[error("Deserialization failed: {0}")]
    DeserializationFailed(String),
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("Invalid model data: {0}")]
    InvalidModelData(String),
    #[error("Validation failed: {0}")]
    ValidationFailed(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Bincode error: {0}")]
    BincodeError(#[from] bincode::Error),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("Format detection error: {0}")]
    FormatDetectionError(String),
    #[error("Compatibility error: {0}")]
    CompatibilityError(String),
}

impl From<SerializationError> for LightGBMError {
    fn from(err: SerializationError) -> Self {
        LightGBMError::serialization(err.to_string())
    }
}

/// Trait for model serializers
pub trait ModelSerializer: Send + Sync {
    /// Serialize a model to bytes
    fn serialize(&self, model: &dyn SerializableModel) -> Result<Vec<u8>>;
    
    /// Serialize a model to writer
    fn serialize_to_writer(
        &self,
        model: &dyn SerializableModel,
        writer: &mut dyn Write,
    ) -> Result<()>;
    
    /// Get the format this serializer handles
    fn format(&self) -> SerializationFormat;
    
    /// Get configuration
    fn config(&self) -> &SerializationConfig;
    
    /// Set configuration
    fn set_config(&mut self, config: SerializationConfig);
    
    /// Estimate serialized size
    fn estimate_size(&self, model: &dyn SerializableModel) -> usize;
    
    /// Validate model before serialization
    fn validate(&self, model: &dyn SerializableModel) -> Result<()> {
        if self.config().validate_before_serialize {
            model.validate()?;
        }
        Ok(())
    }
}

/// Trait for model deserializers
pub trait ModelDeserializer: Send + Sync {
    /// Deserialize a model from bytes
    fn deserialize(&self, data: &[u8]) -> Result<Box<dyn SerializableModel>>;
    
    /// Deserialize a model from reader
    fn deserialize_from_reader(
        &self,
        reader: &mut dyn Read,
    ) -> Result<Box<dyn SerializableModel>>;
    
    /// Get the format this deserializer handles
    fn format(&self) -> SerializationFormat;
    
    /// Get configuration
    fn config(&self) -> &SerializationConfig;
    
    /// Set configuration
    fn set_config(&mut self, config: SerializationConfig);
    
    /// Validate model after deserialization
    fn validate(&self, model: &dyn SerializableModel) -> Result<()> {
        if !self.config().skip_validation {
            model.validate()?;
        }
        Ok(())
    }
}

/// Serialization utilities
pub mod utils {
    use super::*;
    use std::path::Path;
    
    /// Detect serialization format from file extension
    pub fn detect_format_from_extension(path: &Path) -> Option<SerializationFormat> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .and_then(|ext| match ext.to_lowercase().as_str() {
                "lgb" | "bin" => Some(SerializationFormat::Bincode),
                "json" => Some(SerializationFormat::Json),
                "txt" | "model" => Some(SerializationFormat::LightGbm),
                _ => None,
            })
    }
    
    /// Detect serialization format from file content
    pub fn detect_format_from_content(data: &[u8]) -> Option<SerializationFormat> {
        // Check for JSON format (starts with '{' or '[')
        if data.starts_with(b"{") || data.starts_with(b"[") {
            return Some(SerializationFormat::Json);
        }
        
        // Check for LightGBM text format (starts with "tree" or "feature_names")
        if data.starts_with(b"tree") || data.starts_with(b"feature_names") {
            return Some(SerializationFormat::LightGbm);
        }
        
        // Check for bincode format (binary data)
        if data.len() >= 8 && is_likely_bincode(data) {
            return Some(SerializationFormat::Bincode);
        }
        
        None
    }
    
    /// Check if data is likely bincode format
    fn is_likely_bincode(data: &[u8]) -> bool {
        // Simple heuristic: bincode usually starts with length fields
        if data.len() < 8 {
            return false;
        }
        
        // Check for reasonable length values in the first 8 bytes
        let first_u64 = u64::from_le_bytes([
            data[0], data[1], data[2], data[3],
            data[4], data[5], data[6], data[7],
        ]);
        
        // Reasonable range for model data size
        first_u64 > 0 && first_u64 < (1024 * 1024 * 1024) // 1GB limit
    }
    
    /// Get recommended file extension for format
    pub fn get_file_extension(format: SerializationFormat) -> &'static str {
        match format {
            SerializationFormat::Bincode => ".lgb",
            SerializationFormat::Json => ".json",
            SerializationFormat::LightGbm => ".txt",
        }
    }
    
    /// Create serializer for format
    pub fn create_serializer(format: SerializationFormat) -> Result<Box<dyn ModelSerializer>> {
        match format {
            SerializationFormat::Bincode => Ok(Box::new(bincode::BincodeSerializer::new()?)),
            SerializationFormat::Json => Ok(Box::new(json::JsonSerializer::new()?)),
            SerializationFormat::LightGbm => Ok(Box::new(lightgbm::LightGbmSerializer::new()?)),
        }
    }
    
    /// Create deserializer for format
    pub fn create_deserializer(format: SerializationFormat) -> Result<Box<dyn ModelDeserializer>> {
        match format {
            SerializationFormat::Bincode => Ok(Box::new(bincode::BincodeDeserializer::new()?)),
            SerializationFormat::Json => Ok(Box::new(json::JsonDeserializer::new()?)),
            SerializationFormat::LightGbm => Ok(Box::new(lightgbm::LightGbmDeserializer::new()?)),
        }
    }
    
    /// Convert between formats
    pub fn convert_model(
        input_data: &[u8],
        from_format: SerializationFormat,
        to_format: SerializationFormat,
    ) -> Result<Vec<u8>> {
        if from_format == to_format {
            return Ok(input_data.to_vec());
        }
        
        // Deserialize from source format
        let deserializer = create_deserializer(from_format)?;
        let model = deserializer.deserialize(input_data)?;
        
        // Serialize to target format
        let serializer = create_serializer(to_format)?;
        serializer.serialize(model.as_ref())
    }
    
    /// Validate serialized data
    pub fn validate_serialized_data(data: &[u8], format: SerializationFormat) -> Result<()> {
        // Try to deserialize to check validity
        let deserializer = create_deserializer(format)?;
        let _model = deserializer.deserialize(data)?;
        Ok(())
    }
    
    /// Get format information
    pub fn format_info(format: SerializationFormat) -> FormatInfo {
        match format {
            SerializationFormat::Bincode => FormatInfo {
                name: "Bincode",
                description: "Native Rust binary format (fast, compact)",
                is_binary: true,
                is_human_readable: false,
                supports_compression: true,
                typical_size_ratio: 1.0,
            },
            SerializationFormat::Json => FormatInfo {
                name: "JSON",
                description: "JavaScript Object Notation (human-readable, portable)",
                is_binary: false,
                is_human_readable: true,
                supports_compression: true,
                typical_size_ratio: 3.0,
            },
            SerializationFormat::LightGbm => FormatInfo {
                name: "LightGBM",
                description: "Original LightGBM text format (compatible)",
                is_binary: false,
                is_human_readable: true,
                supports_compression: false,
                typical_size_ratio: 2.5,
            },
        }
    }
}

/// Information about a serialization format
#[derive(Debug, Clone)]
pub struct FormatInfo {
    /// Format name
    pub name: &'static str,
    /// Format description
    pub description: &'static str,
    /// Whether the format is binary
    pub is_binary: bool,
    /// Whether the format is human-readable
    pub is_human_readable: bool,
    /// Whether the format supports compression
    pub supports_compression: bool,
    /// Typical size ratio compared to bincode
    pub typical_size_ratio: f64,
}

/// Serialization statistics
#[derive(Debug, Clone)]
pub struct SerializationStats {
    /// Original size in bytes
    pub original_size: usize,
    /// Serialized size in bytes
    pub serialized_size: usize,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Serialization time in milliseconds
    pub serialization_time_ms: u64,
    /// Deserialization time in milliseconds
    pub deserialization_time_ms: u64,
}

impl SerializationStats {
    /// Create new statistics
    pub fn new(original_size: usize, serialized_size: usize) -> Self {
        let compression_ratio = if original_size > 0 {
            serialized_size as f64 / original_size as f64
        } else {
            0.0
        };
        
        SerializationStats {
            original_size,
            serialized_size,
            compression_ratio,
            serialization_time_ms: 0,
            deserialization_time_ms: 0,
        }
    }
    
    /// Set timing information
    pub fn with_timing(mut self, serialization_ms: u64, deserialization_ms: u64) -> Self {
        self.serialization_time_ms = serialization_ms;
        self.deserialization_time_ms = deserialization_ms;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_serialization_format_display() {
        assert_eq!(SerializationFormat::Bincode.to_string(), "bincode");
        assert_eq!(SerializationFormat::Json.to_string(), "json");
        assert_eq!(SerializationFormat::LightGbm.to_string(), "lightgbm");
    }
    
    #[test]
    fn test_serialization_format_from_str() {
        assert_eq!("bincode".parse::<SerializationFormat>().unwrap(), SerializationFormat::Bincode);
        assert_eq!("json".parse::<SerializationFormat>().unwrap(), SerializationFormat::Json);
        assert_eq!("lightgbm".parse::<SerializationFormat>().unwrap(), SerializationFormat::LightGbm);
        assert_eq!("txt".parse::<SerializationFormat>().unwrap(), SerializationFormat::LightGbm);
        
        assert!("unknown".parse::<SerializationFormat>().is_err());
    }
    
    #[test]
    fn test_serialization_config_default() {
        let config = SerializationConfig::default();
        assert_eq!(config.format, SerializationFormat::Bincode);
        assert!(!config.pretty_json);
        assert!(config.include_metadata);
        assert!(config.validate_before_serialize);
    }
    
    #[test]
    fn test_utils_detect_format_from_extension() {
        use std::path::Path;
        
        assert_eq!(
            utils::detect_format_from_extension(Path::new("model.lgb")),
            Some(SerializationFormat::Bincode)
        );
        assert_eq!(
            utils::detect_format_from_extension(Path::new("model.json")),
            Some(SerializationFormat::Json)
        );
        assert_eq!(
            utils::detect_format_from_extension(Path::new("model.txt")),
            Some(SerializationFormat::LightGbm)
        );
        assert_eq!(
            utils::detect_format_from_extension(Path::new("model.unknown")),
            None
        );
    }
    
    #[test]
    fn test_utils_detect_format_from_content() {
        assert_eq!(
            utils::detect_format_from_content(b"{\"model\": \"test\"}"),
            Some(SerializationFormat::Json)
        );
        assert_eq!(
            utils::detect_format_from_content(b"tree"),
            Some(SerializationFormat::LightGbm)
        );
        assert_eq!(
            utils::detect_format_from_content(b"feature_names"),
            Some(SerializationFormat::LightGbm)
        );
    }
    
    #[test]
    fn test_utils_file_extensions() {
        assert_eq!(utils::get_file_extension(SerializationFormat::Bincode), ".lgb");
        assert_eq!(utils::get_file_extension(SerializationFormat::Json), ".json");
        assert_eq!(utils::get_file_extension(SerializationFormat::LightGbm), ".txt");
    }
    
    #[test]
    fn test_format_info() {
        let info = utils::format_info(SerializationFormat::Bincode);
        assert_eq!(info.name, "Bincode");
        assert!(info.is_binary);
        assert!(!info.is_human_readable);
        assert!(info.supports_compression);
        
        let info = utils::format_info(SerializationFormat::Json);
        assert_eq!(info.name, "JSON");
        assert!(!info.is_binary);
        assert!(info.is_human_readable);
        
        let info = utils::format_info(SerializationFormat::LightGbm);
        assert_eq!(info.name, "LightGBM");
        assert!(!info.is_binary);
        assert!(info.is_human_readable);
        assert!(!info.supports_compression);
    }
    
    #[test]
    fn test_serialization_stats() {
        let stats = SerializationStats::new(1000, 500);
        assert_eq!(stats.original_size, 1000);
        assert_eq!(stats.serialized_size, 500);
        assert_eq!(stats.compression_ratio, 0.5);
        
        let stats = stats.with_timing(100, 50);
        assert_eq!(stats.serialization_time_ms, 100);
        assert_eq!(stats.deserialization_time_ms, 50);
    }
}