//! Model persistence layer for Pure Rust LightGBM.
//!
//! This module provides comprehensive model serialization and deserialization
//! capabilities with support for multiple formats including native Rust bincode,
//! JSON, and compatibility with original LightGBM model files.

pub mod serialization;
pub mod model_file;
pub mod format;

// Re-export commonly used types
pub use serialization::{
    SerializationFormat, SerializationConfig, SerializationError,
    ModelSerializer, ModelDeserializer,
};
pub use model_file::{ModelFile, ModelFileMetadata, ModelFileError};
pub use format::{FormatDetector, ModelFormat, FormatDetectionError};

use crate::core::types::*;
use crate::core::error::{Result, LightGBMError};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::collections::HashMap;

/// Model persistence configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PersistenceConfig {
    /// Default serialization format
    pub default_format: SerializationFormat,
    /// Compression settings
    pub compression: CompressionConfig,
    /// Metadata settings
    pub metadata: MetadataConfig,
    /// Validation settings
    pub validation: ValidationConfig,
    /// Backward compatibility settings
    pub compatibility: CompatibilityConfig,
}

/// Compression configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable compression
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level (0-9)
    pub level: u8,
}

/// Compression algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// Gzip compression
    Gzip,
    /// Zstd compression
    Zstd,
    /// LZ4 compression
    Lz4,
}

/// Metadata configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetadataConfig {
    /// Include training metadata
    pub include_training_info: bool,
    /// Include feature metadata
    pub include_feature_info: bool,
    /// Include performance metrics
    pub include_metrics: bool,
    /// Include custom properties
    pub include_custom_properties: bool,
}

/// Validation configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Validate model integrity on save
    pub validate_on_save: bool,
    /// Validate model integrity on load
    pub validate_on_load: bool,
    /// Verify model compatibility
    pub verify_compatibility: bool,
    /// Check for data corruption
    pub check_corruption: bool,
}

/// Compatibility configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompatibilityConfig {
    /// Target LightGBM version
    pub target_version: Option<String>,
    /// Enable legacy format support
    pub enable_legacy_support: bool,
    /// Enable experimental features
    pub enable_experimental: bool,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        PersistenceConfig {
            default_format: SerializationFormat::Bincode,
            compression: CompressionConfig::default(),
            metadata: MetadataConfig::default(),
            validation: ValidationConfig::default(),
            compatibility: CompatibilityConfig::default(),
        }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        CompressionConfig {
            enabled: true,
            algorithm: CompressionAlgorithm::Zstd,
            level: 3,
        }
    }
}

impl Default for MetadataConfig {
    fn default() -> Self {
        MetadataConfig {
            include_training_info: true,
            include_feature_info: true,
            include_metrics: true,
            include_custom_properties: true,
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        ValidationConfig {
            validate_on_save: true,
            validate_on_load: true,
            verify_compatibility: true,
            check_corruption: true,
        }
    }
}

impl Default for CompatibilityConfig {
    fn default() -> Self {
        CompatibilityConfig {
            target_version: None,
            enable_legacy_support: false,
            enable_experimental: false,
        }
    }
}

/// Model persistence manager
pub struct ModelPersistence {
    /// Configuration
    config: PersistenceConfig,
    /// Serializers for different formats
    serializers: HashMap<SerializationFormat, Box<dyn ModelSerializer>>,
    /// Deserializers for different formats
    deserializers: HashMap<SerializationFormat, Box<dyn ModelDeserializer>>,
    /// Format detector
    format_detector: FormatDetector,
}

impl ModelPersistence {
    /// Create a new model persistence manager
    pub fn new(config: PersistenceConfig) -> Result<Self> {
        let mut serializers: HashMap<SerializationFormat, Box<dyn ModelSerializer>> = HashMap::new();
        let mut deserializers: HashMap<SerializationFormat, Box<dyn ModelDeserializer>> = HashMap::new();
        
        // Initialize serializers
        serializers.insert(
            SerializationFormat::Bincode,
            Box::new(serialization::bincode::BincodeSerializer::new()?),
        );
        serializers.insert(
            SerializationFormat::Json,
            Box::new(serialization::json::JsonSerializer::new()?),
        );
        serializers.insert(
            SerializationFormat::LightGbm,
            Box::new(serialization::lightgbm::LightGbmSerializer::new()?),
        );
        
        // Initialize deserializers
        deserializers.insert(
            SerializationFormat::Bincode,
            Box::new(serialization::bincode::BincodeDeserializer::new()?),
        );
        deserializers.insert(
            SerializationFormat::Json,
            Box::new(serialization::json::JsonDeserializer::new()?),
        );
        deserializers.insert(
            SerializationFormat::LightGbm,
            Box::new(serialization::lightgbm::LightGbmDeserializer::new()?),
        );
        
        let format_detector = FormatDetector::new();
        
        Ok(ModelPersistence {
            config,
            serializers,
            deserializers,
            format_detector,
        })
    }
    
    /// Save model to file
    pub fn save_model<P: AsRef<Path>>(
        &self,
        model: &dyn SerializableModel,
        path: P,
        format: Option<SerializationFormat>,
    ) -> Result<()> {
        let path = path.as_ref();
        log::info!("Saving model to: {}", path.display());
        
        // Determine format
        let format = format.unwrap_or_else(|| {
            self.format_detector.detect_format_from_path(path)
                .unwrap_or(self.config.default_format)
        });
        
        // Get serializer
        let serializer = self.serializers.get(&format)
            .ok_or_else(|| LightGBMError::config(format!("No serializer for format: {:?}", format)))?;
        
        // Validate model if configured
        if self.config.validation.validate_on_save {
            self.validate_model(model)?;
        }
        
        // Create model file
        let model_file = ModelFile::new(path, format, &self.config)?;
        
        // Serialize model
        let serialized_data = serializer.serialize(model)?;
        
        // Apply compression if enabled
        let final_data = if self.config.compression.enabled {
            self.compress_data(&serialized_data)?
        } else {
            serialized_data
        };
        
        // Write to file
        model_file.write_data(&final_data)?;
        
        log::info!("Model saved successfully with format: {:?}", format);
        Ok(())
    }
    
    /// Load model from file
    pub fn load_model<P: AsRef<Path>>(
        &self,
        path: P,
        format: Option<SerializationFormat>,
    ) -> Result<Box<dyn SerializableModel>> {
        let path = path.as_ref();
        log::info!("Loading model from: {}", path.display());
        
        // Determine format
        let format = format.unwrap_or_else(|| {
            self.format_detector.detect_format_from_file(path)
                .unwrap_or(self.config.default_format)
        });
        
        // Get deserializer
        let deserializer = self.deserializers.get(&format)
            .ok_or_else(|| LightGBMError::config(format!("No deserializer for format: {:?}", format)))?;
        
        // Open model file
        let model_file = ModelFile::open(path)?;
        
        // Read data
        let compressed_data = model_file.read_data()?;
        
        // Decompress if needed
        let serialized_data = if self.config.compression.enabled {
            self.decompress_data(&compressed_data)?
        } else {
            compressed_data
        };
        
        // Deserialize model
        let model = deserializer.deserialize(&serialized_data)?;
        
        // Validate model if configured
        if self.config.validation.validate_on_load {
            self.validate_model(model.as_ref())?;
        }
        
        log::info!("Model loaded successfully with format: {:?}", format);
        Ok(model)
    }
    
    /// Validate model integrity
    fn validate_model(&self, model: &dyn SerializableModel) -> Result<()> {
        // Check if model has required components
        if model.feature_names().is_empty() {
            return Err(LightGBMError::model("Model has no feature names"));
        }
        
        if model.num_features() == 0 {
            return Err(LightGBMError::model("Model has no features"));
        }
        
        // Additional validation checks can be added here
        Ok(())
    }
    
    /// Compress data
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self.config.compression.algorithm {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            CompressionAlgorithm::Gzip => {
                use std::io::Write;
                let mut encoder = flate2::write::GzEncoder::new(
                    Vec::new(),
                    flate2::Compression::new(self.config.compression.level as u32),
                );
                encoder.write_all(data)?;
                encoder.finish().map_err(|e| LightGBMError::serialization(format!("Gzip compression failed: {}", e)))
            }
            CompressionAlgorithm::Zstd => {
                zstd::encode_all(data, self.config.compression.level as i32)
                    .map_err(|e| LightGBMError::serialization(format!("Zstd compression failed: {}", e)))
            }
            CompressionAlgorithm::Lz4 => {
                lz4_flex::compress_prepend_size(data)
                    .map_err(|e| LightGBMError::serialization(format!("LZ4 compression failed: {:?}", e)))
            }
        }
    }
    
    /// Decompress data
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self.config.compression.algorithm {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            CompressionAlgorithm::Gzip => {
                use std::io::Read;
                let mut decoder = flate2::read::GzDecoder::new(data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)?;
                Ok(decompressed)
            }
            CompressionAlgorithm::Zstd => {
                zstd::decode_all(data)
                    .map_err(|e| LightGBMError::serialization(format!("Zstd decompression failed: {}", e)))
            }
            CompressionAlgorithm::Lz4 => {
                lz4_flex::decompress_size_prepended(data)
                    .map_err(|e| LightGBMError::serialization(format!("LZ4 decompression failed: {:?}", e)))
            }
        }
    }
    
    /// Get supported formats
    pub fn supported_formats(&self) -> Vec<SerializationFormat> {
        self.serializers.keys().copied().collect()
    }
    
    /// Get configuration
    pub fn config(&self) -> &PersistenceConfig {
        &self.config
    }
    
    /// Set configuration
    pub fn set_config(&mut self, config: PersistenceConfig) {
        self.config = config;
    }
}

/// Trait for serializable models
pub trait SerializableModel: Send + Sync {
    /// Get model type
    fn model_type(&self) -> &'static str;
    
    /// Get feature names
    fn feature_names(&self) -> &[String];
    
    /// Get number of features
    fn num_features(&self) -> usize;
    
    /// Get model metadata
    fn metadata(&self) -> HashMap<String, String>;
    
    /// Set model metadata
    fn set_metadata(&mut self, metadata: HashMap<String, String>);
    
    /// Get model version
    fn version(&self) -> String;
    
    /// Get model creation timestamp
    fn created_at(&self) -> chrono::DateTime<chrono::Utc>;
    
    /// Validate model consistency
    fn validate(&self) -> Result<()>;
    
    /// Clone the model
    fn clone_model(&self) -> Box<dyn SerializableModel>;
}

/// Model metadata structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model type
    pub model_type: String,
    /// Model version
    pub version: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Number of features
    pub num_features: usize,
    /// Training parameters
    pub training_params: HashMap<String, String>,
    /// Model metrics
    pub metrics: HashMap<String, f64>,
    /// Custom properties
    pub custom_properties: HashMap<String, String>,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        ModelMetadata {
            model_type: "lightgbm".to_string(),
            version: "1.0.0".to_string(),
            created_at: chrono::Utc::now(),
            feature_names: Vec::new(),
            num_features: 0,
            training_params: HashMap::new(),
            metrics: HashMap::new(),
            custom_properties: HashMap::new(),
        }
    }
}

/// Utility functions for model persistence
pub mod utils {
    use super::*;
    
    /// Create a default persistence configuration
    pub fn default_config() -> PersistenceConfig {
        PersistenceConfig::default()
    }
    
    /// Create a minimal persistence configuration (no compression, no validation)
    pub fn minimal_config() -> PersistenceConfig {
        PersistenceConfig {
            default_format: SerializationFormat::Bincode,
            compression: CompressionConfig {
                enabled: false,
                algorithm: CompressionAlgorithm::None,
                level: 0,
            },
            metadata: MetadataConfig {
                include_training_info: false,
                include_feature_info: true,
                include_metrics: false,
                include_custom_properties: false,
            },
            validation: ValidationConfig {
                validate_on_save: false,
                validate_on_load: false,
                verify_compatibility: false,
                check_corruption: false,
            },
            compatibility: CompatibilityConfig::default(),
        }
    }
    
    /// Create a production-ready persistence configuration
    pub fn production_config() -> PersistenceConfig {
        PersistenceConfig {
            default_format: SerializationFormat::Bincode,
            compression: CompressionConfig {
                enabled: true,
                algorithm: CompressionAlgorithm::Zstd,
                level: 3,
            },
            metadata: MetadataConfig {
                include_training_info: true,
                include_feature_info: true,
                include_metrics: true,
                include_custom_properties: true,
            },
            validation: ValidationConfig {
                validate_on_save: true,
                validate_on_load: true,
                verify_compatibility: true,
                check_corruption: true,
            },
            compatibility: CompatibilityConfig {
                target_version: Some("3.0.0".to_string()),
                enable_legacy_support: false,
                enable_experimental: false,
            },
        }
    }
    
    /// Get file extension for serialization format
    pub fn format_extension(format: SerializationFormat) -> &'static str {
        match format {
            SerializationFormat::Bincode => ".lgb",
            SerializationFormat::Json => ".json",
            SerializationFormat::LightGbm => ".txt",
        }
    }
    
    /// Estimate serialized size
    pub fn estimate_size(model: &dyn SerializableModel) -> usize {
        // Simple estimation based on feature count
        // In practice, this would be more sophisticated
        let base_size = 1024; // Base overhead
        let feature_size = model.num_features() * 64; // ~64 bytes per feature
        let metadata_size = model.metadata().len() * 32; // ~32 bytes per metadata entry
        
        base_size + feature_size + metadata_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_persistence_config_default() {
        let config = PersistenceConfig::default();
        assert_eq!(config.default_format, SerializationFormat::Bincode);
        assert!(config.compression.enabled);
        assert!(config.validation.validate_on_save);
    }
    
    #[test]
    fn test_compression_config() {
        let config = CompressionConfig {
            enabled: true,
            algorithm: CompressionAlgorithm::Gzip,
            level: 6,
        };
        assert!(config.enabled);
        assert_eq!(config.algorithm, CompressionAlgorithm::Gzip);
        assert_eq!(config.level, 6);
    }
    
    #[test]
    fn test_model_metadata_default() {
        let metadata = ModelMetadata::default();
        assert_eq!(metadata.model_type, "lightgbm");
        assert_eq!(metadata.version, "1.0.0");
        assert_eq!(metadata.num_features, 0);
    }
    
    #[test]
    fn test_utils_functions() {
        let config = utils::default_config();
        assert_eq!(config.default_format, SerializationFormat::Bincode);
        
        let minimal = utils::minimal_config();
        assert!(!minimal.compression.enabled);
        assert!(!minimal.validation.validate_on_save);
        
        let production = utils::production_config();
        assert!(production.compression.enabled);
        assert!(production.validation.validate_on_save);
    }
    
    #[test]
    fn test_format_extensions() {
        assert_eq!(utils::format_extension(SerializationFormat::Bincode), ".lgb");
        assert_eq!(utils::format_extension(SerializationFormat::Json), ".json");
        assert_eq!(utils::format_extension(SerializationFormat::LightGbm), ".txt");
    }
}