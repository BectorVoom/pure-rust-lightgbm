//! Bincode serialization for Pure Rust LightGBM models.
//!
//! This module provides fast, compact binary serialization using the bincode format.
//! Bincode is the default serialization format for optimal performance and storage efficiency.

use crate::core::error::{Result, LightGBMError};
use crate::io::serialization::{
    ModelDeserializer, ModelSerializer, SerializationConfig, SerializationError,
    SerializationFormat, SerializationStats,
};
use crate::io::SerializableModel;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::time::Instant;

/// Bincode serialization configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BincodeConfig {
    /// Base serialization config
    pub base: SerializationConfig,
    /// Byte order (little endian by default)
    pub little_endian: bool,
    /// Use fixed-size integer encoding
    pub fixed_int_encoding: bool,
    /// Maximum serialized size limit
    pub size_limit: Option<u64>,
    /// Enable strict mode (reject unknown fields)
    pub strict_mode: bool,
}

impl Default for BincodeConfig {
    fn default() -> Self {
        BincodeConfig {
            base: SerializationConfig {
                format: SerializationFormat::Bincode,
                ..Default::default()
            },
            little_endian: true,
            fixed_int_encoding: false,
            size_limit: Some(1024 * 1024 * 1024), // 1GB limit
            strict_mode: false,
        }
    }
}

/// Bincode serializer
pub struct BincodeSerializer {
    config: BincodeConfig,
    stats: Option<SerializationStats>,
}

impl BincodeSerializer {
    /// Create a new bincode serializer
    pub fn new() -> Result<Self> {
        Ok(BincodeSerializer {
            config: BincodeConfig::default(),
            stats: None,
        })
    }

    /// Create a new bincode serializer with custom configuration
    pub fn with_config(config: BincodeConfig) -> Result<Self> {
        Ok(BincodeSerializer {
            config,
            stats: None,
        })
    }

    /// Get bincode configuration
    fn get_bincode_config(&self) -> bincode::Config {
        let mut config = bincode::config();
        
        if self.config.little_endian {
            config = config.little_endian();
        } else {
            config = config.big_endian();
        }
        
        if self.config.fixed_int_encoding {
            config = config.fixed_int_encoding();
        } else {
            config = config.varint_encoding();
        }
        
        if let Some(limit) = self.config.size_limit {
            config = config.limit(limit);
        }
        
        config
    }

    /// Serialize model to bincode format
    fn serialize_internal(&self, model: &dyn SerializableModel) -> Result<Vec<u8>> {
        let start_time = Instant::now();
        
        // Validate model if configured
        if self.config.base.validate_before_serialize {
            model.validate()?;
        }

        // Create serializable representation
        let serializable_model = SerializableModelWrapper::from_model(model, &self.config.base)?;
        
        // Serialize using bincode
        let bincode_config = self.get_bincode_config();
        let serialized = bincode_config.serialize(&serializable_model)
            .map_err(|e| SerializationError::BincodeError(e.to_string()))?;

        let serialization_time = start_time.elapsed().as_millis() as u64;
        
        // Update statistics
        if let Some(ref mut stats) = self.stats.as_mut() {
            stats.serialization_time_ms = serialization_time;
            stats.serialized_size = serialized.len();
        }

        Ok(serialized)
    }

    /// Get serialization statistics
    pub fn stats(&self) -> Option<&SerializationStats> {
        self.stats.as_ref()
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = None;
    }
}

impl Default for BincodeSerializer {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

impl ModelSerializer for BincodeSerializer {
    fn serialize(&self, model: &dyn SerializableModel) -> Result<Vec<u8>> {
        self.serialize_internal(model)
    }

    fn serialize_to_writer(
        &self,
        model: &dyn SerializableModel,
        writer: &mut dyn Write,
    ) -> Result<()> {
        let data = self.serialize_internal(model)?;
        writer.write_all(&data)?;
        Ok(())
    }

    fn format(&self) -> SerializationFormat {
        SerializationFormat::Bincode
    }

    fn config(&self) -> &SerializationConfig {
        &self.config.base
    }

    fn set_config(&mut self, config: SerializationConfig) {
        self.config.base = config;
    }

    fn estimate_size(&self, model: &dyn SerializableModel) -> usize {
        // Estimation based on model complexity
        let base_size = 1024; // Base overhead
        let feature_size = model.num_features() * 64; // ~64 bytes per feature
        let metadata_size = model.metadata().len() * 32; // ~32 bytes per metadata entry
        
        // Bincode is typically 20-30% more compact than JSON
        let estimated_size = base_size + feature_size + metadata_size;
        (estimated_size as f64 * 0.7) as usize
    }
}

/// Bincode deserializer
pub struct BincodeDeserializer {
    config: BincodeConfig,
    stats: Option<SerializationStats>,
}

impl BincodeDeserializer {
    /// Create a new bincode deserializer
    pub fn new() -> Result<Self> {
        Ok(BincodeDeserializer {
            config: BincodeConfig::default(),
            stats: None,
        })
    }

    /// Create a new bincode deserializer with custom configuration
    pub fn with_config(config: BincodeConfig) -> Result<Self> {
        Ok(BincodeDeserializer {
            config,
            stats: None,
        })
    }

    /// Get bincode configuration
    fn get_bincode_config(&self) -> bincode::Config {
        let mut config = bincode::config();
        
        if self.config.little_endian {
            config = config.little_endian();
        } else {
            config = config.big_endian();
        }
        
        if self.config.fixed_int_encoding {
            config = config.fixed_int_encoding();
        } else {
            config = config.varint_encoding();
        }
        
        if let Some(limit) = self.config.size_limit {
            config = config.limit(limit);
        }
        
        config
    }

    /// Deserialize model from bincode format
    fn deserialize_internal(&self, data: &[u8]) -> Result<Box<dyn SerializableModel>> {
        let start_time = Instant::now();
        
        // Deserialize using bincode
        let bincode_config = self.get_bincode_config();
        let wrapper: SerializableModelWrapper = bincode_config.deserialize(data)
            .map_err(|e| SerializationError::BincodeError(e.to_string()))?;

        // Convert to model
        let model = wrapper.to_model()?;
        
        // Validate model if configured
        if !self.config.base.skip_validation {
            model.validate()?;
        }

        let deserialization_time = start_time.elapsed().as_millis() as u64;
        
        // Update statistics
        if let Some(ref mut stats) = self.stats.as_mut() {
            stats.deserialization_time_ms = deserialization_time;
            stats.original_size = data.len();
        }

        Ok(model)
    }

    /// Get deserialization statistics
    pub fn stats(&self) -> Option<&SerializationStats> {
        self.stats.as_ref()
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = None;
    }
}

impl Default for BincodeDeserializer {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

impl ModelDeserializer for BincodeDeserializer {
    fn deserialize(&self, data: &[u8]) -> Result<Box<dyn SerializableModel>> {
        self.deserialize_internal(data)
    }

    fn deserialize_from_reader(
        &self,
        reader: &mut dyn Read,
    ) -> Result<Box<dyn SerializableModel>> {
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;
        self.deserialize_internal(&data)
    }

    fn format(&self) -> SerializationFormat {
        SerializationFormat::Bincode
    }

    fn config(&self) -> &SerializationConfig {
        &self.config.base
    }

    fn set_config(&mut self, config: SerializationConfig) {
        self.config.base = config;
    }
}

/// Wrapper for serializable model data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableModelWrapper {
    /// Model type identifier
    pub model_type: String,
    /// Model version
    pub version: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Number of features
    pub num_features: usize,
    /// Model metadata
    pub metadata: std::collections::HashMap<String, String>,
    /// Model data (serialized trees, parameters, etc.)
    pub model_data: ModelData,
    /// Serialization configuration used
    pub serialization_config: SerializationConfig,
}

impl SerializableModelWrapper {
    /// Create wrapper from model
    pub fn from_model(
        model: &dyn SerializableModel,
        config: &SerializationConfig,
    ) -> Result<Self> {
        let model_data = ModelData::from_model(model, config)?;
        
        Ok(SerializableModelWrapper {
            model_type: model.model_type().to_string(),
            version: model.version(),
            created_at: model.created_at(),
            feature_names: model.feature_names().to_vec(),
            num_features: model.num_features(),
            metadata: model.metadata(),
            model_data,
            serialization_config: config.clone(),
        })
    }

    /// Convert wrapper to model
    pub fn to_model(&self) -> Result<Box<dyn SerializableModel>> {
        self.model_data.to_model(&self.serialization_config)
    }
}

/// Model data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelData {
    /// Serialized trees
    pub trees: Vec<TreeData>,
    /// Training parameters
    pub params: TrainingParams,
    /// Objective function type
    pub objective: ObjectiveType,
    /// Feature importance scores
    pub feature_importance: Option<Vec<f64>>,
    /// Training history
    pub training_history: Option<TrainingHistory>,
    /// Additional model-specific data
    pub extra_data: std::collections::HashMap<String, Vec<u8>>,
}

impl ModelData {
    /// Create model data from serializable model
    pub fn from_model(
        model: &dyn SerializableModel,
        config: &SerializationConfig,
    ) -> Result<Self> {
        // This is a simplified implementation
        // In a real implementation, you'd extract actual model data
        Ok(ModelData {
            trees: Vec::new(), // Would be populated from actual model
            params: TrainingParams::default(),
            objective: ObjectiveType::Regression,
            feature_importance: if config.include_feature_importance {
                Some(vec![0.0; model.num_features()])
            } else {
                None
            },
            training_history: if config.include_training_history {
                Some(TrainingHistory::default())
            } else {
                None
            },
            extra_data: std::collections::HashMap::new(),
        })
    }

    /// Convert model data to serializable model
    pub fn to_model(
        &self,
        _config: &SerializationConfig,
    ) -> Result<Box<dyn SerializableModel>> {
        // This is a simplified implementation
        // In a real implementation, you'd reconstruct the actual model
        Ok(Box::new(BincodeModel::new(self.clone())))
    }
}

/// Tree data for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeData {
    /// Tree structure
    pub nodes: Vec<NodeData>,
    /// Tree metadata
    pub metadata: TreeMetadata,
}

/// Node data for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeData {
    /// Node type (leaf or internal)
    pub node_type: NodeType,
    /// Feature index (for internal nodes)
    pub feature: Option<usize>,
    /// Split threshold (for internal nodes)
    pub threshold: Option<f64>,
    /// Left child index
    pub left_child: Option<usize>,
    /// Right child index
    pub right_child: Option<usize>,
    /// Leaf value (for leaf nodes)
    pub leaf_value: Option<f64>,
    /// Node weight
    pub weight: f64,
    /// Node count
    pub count: usize,
}

/// Node type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    Internal,
    Leaf,
}

/// Tree metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeMetadata {
    /// Number of nodes
    pub num_nodes: usize,
    /// Number of leaves
    pub num_leaves: usize,
    /// Tree depth
    pub depth: usize,
    /// Tree weight
    pub weight: f64,
}

/// Training parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParams {
    /// Learning rate
    pub learning_rate: f64,
    /// Number of leaves
    pub num_leaves: usize,
    /// Maximum depth
    pub max_depth: i32,
    /// Regularization parameters
    pub lambda_l1: f64,
    pub lambda_l2: f64,
    /// Sampling parameters
    pub feature_fraction: f64,
    pub bagging_fraction: f64,
    /// Additional parameters
    pub extra_params: std::collections::HashMap<String, String>,
}

impl Default for TrainingParams {
    fn default() -> Self {
        TrainingParams {
            learning_rate: 0.1,
            num_leaves: 31,
            max_depth: -1,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            feature_fraction: 1.0,
            bagging_fraction: 1.0,
            extra_params: std::collections::HashMap::new(),
        }
    }
}

/// Objective function type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObjectiveType {
    Regression,
    Binary,
    Multiclass,
    Ranking,
}

/// Training history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHistory {
    /// Training loss values
    pub train_loss: Vec<f64>,
    /// Validation loss values
    pub valid_loss: Vec<f64>,
    /// Training metrics
    pub train_metrics: std::collections::HashMap<String, Vec<f64>>,
    /// Validation metrics
    pub valid_metrics: std::collections::HashMap<String, Vec<f64>>,
    /// Early stopping information
    pub early_stopping: Option<EarlyStoppingInfo>,
}

impl Default for TrainingHistory {
    fn default() -> Self {
        TrainingHistory {
            train_loss: Vec::new(),
            valid_loss: Vec::new(),
            train_metrics: std::collections::HashMap::new(),
            valid_metrics: std::collections::HashMap::new(),
            early_stopping: None,
        }
    }
}

/// Early stopping information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingInfo {
    /// Best iteration
    pub best_iteration: usize,
    /// Best score
    pub best_score: f64,
    /// Stopping reason
    pub stopping_reason: String,
}

/// Simple bincode model implementation
pub struct BincodeModel {
    data: ModelData,
    metadata: std::collections::HashMap<String, String>,
}

impl BincodeModel {
    /// Create new bincode model
    pub fn new(data: ModelData) -> Self {
        BincodeModel {
            data,
            metadata: std::collections::HashMap::new(),
        }
    }
}

impl SerializableModel for BincodeModel {
    fn model_type(&self) -> &'static str {
        "bincode_model"
    }

    fn feature_names(&self) -> &[String] {
        // Would return actual feature names from model
        &[]
    }

    fn num_features(&self) -> usize {
        self.data.params.num_leaves // Placeholder
    }

    fn metadata(&self) -> std::collections::HashMap<String, String> {
        self.metadata.clone()
    }

    fn set_metadata(&mut self, metadata: std::collections::HashMap<String, String>) {
        self.metadata = metadata;
    }

    fn version(&self) -> String {
        "1.0.0".to_string()
    }

    fn created_at(&self) -> chrono::DateTime<chrono::Utc> {
        chrono::Utc::now()
    }

    fn validate(&self) -> Result<()> {
        // Basic validation
        if self.data.trees.is_empty() {
            return Err(LightGBMError::model("Model has no trees"));
        }
        Ok(())
    }

    fn clone_model(&self) -> Box<dyn SerializableModel> {
        Box::new(BincodeModel {
            data: self.data.clone(),
            metadata: self.metadata.clone(),
        })
    }
}

/// Bincode serialization utilities
pub mod utils {
    use super::*;
    
    /// Create optimized serializer for performance
    pub fn performance_serializer() -> Result<BincodeSerializer> {
        let config = BincodeConfig {
            base: SerializationConfig {
                format: SerializationFormat::Bincode,
                pretty_json: false,
                include_metadata: false,
                include_feature_importance: false,
                include_training_history: false,
                compression_level: 0,
                validate_before_serialize: false,
                skip_validation: true,
            },
            little_endian: true,
            fixed_int_encoding: true,
            size_limit: None,
            strict_mode: false,
        };
        
        BincodeSerializer::with_config(config)
    }
    
    /// Create optimized deserializer for performance
    pub fn performance_deserializer() -> Result<BincodeDeserializer> {
        let config = BincodeConfig {
            base: SerializationConfig {
                format: SerializationFormat::Bincode,
                pretty_json: false,
                include_metadata: false,
                include_feature_importance: false,
                include_training_history: false,
                compression_level: 0,
                validate_before_serialize: false,
                skip_validation: true,
            },
            little_endian: true,
            fixed_int_encoding: true,
            size_limit: None,
            strict_mode: false,
        };
        
        BincodeDeserializer::with_config(config)
    }
    
    /// Create compact serializer for minimal size
    pub fn compact_serializer() -> Result<BincodeSerializer> {
        let config = BincodeConfig {
            base: SerializationConfig {
                format: SerializationFormat::Bincode,
                pretty_json: false,
                include_metadata: false,
                include_feature_importance: false,
                include_training_history: false,
                compression_level: 9,
                validate_before_serialize: true,
                skip_validation: false,
            },
            little_endian: true,
            fixed_int_encoding: false, // Variable encoding for smaller size
            size_limit: Some(1024 * 1024 * 1024),
            strict_mode: true,
        };
        
        BincodeSerializer::with_config(config)
    }
    
    /// Get bincode version info
    pub fn version_info() -> BincodeVersionInfo {
        BincodeVersionInfo {
            bincode_version: env!("CARGO_PKG_VERSION").to_string(),
            format_version: "1.0.0".to_string(),
            compatibility_version: "1.0.0".to_string(),
        }
    }
    
    /// Estimate compression ratio
    pub fn estimate_compression_ratio(data_size: usize) -> f64 {
        // Bincode is already compact, so compression ratio is minimal
        match data_size {
            0..=1024 => 1.0,
            1025..=10240 => 0.9,
            10241..=102400 => 0.8,
            _ => 0.7,
        }
    }
}

/// Bincode version information
#[derive(Debug, Clone)]
pub struct BincodeVersionInfo {
    /// Bincode crate version
    pub bincode_version: String,
    /// Format version
    pub format_version: String,
    /// Compatibility version
    pub compatibility_version: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bincode_serializer_creation() {
        let serializer = BincodeSerializer::new();
        assert!(serializer.is_ok());
        
        let serializer = serializer.unwrap();
        assert_eq!(serializer.format(), SerializationFormat::Bincode);
    }
    
    #[test]
    fn test_bincode_deserializer_creation() {
        let deserializer = BincodeDeserializer::new();
        assert!(deserializer.is_ok());
        
        let deserializer = deserializer.unwrap();
        assert_eq!(deserializer.format(), SerializationFormat::Bincode);
    }
    
    #[test]
    fn test_bincode_config_default() {
        let config = BincodeConfig::default();
        assert_eq!(config.base.format, SerializationFormat::Bincode);
        assert!(config.little_endian);
        assert!(!config.fixed_int_encoding);
        assert_eq!(config.size_limit, Some(1024 * 1024 * 1024));
    }
    
    #[test]
    fn test_training_params_default() {
        let params = TrainingParams::default();
        assert_eq!(params.learning_rate, 0.1);
        assert_eq!(params.num_leaves, 31);
        assert_eq!(params.max_depth, -1);
        assert_eq!(params.lambda_l1, 0.0);
        assert_eq!(params.lambda_l2, 0.0);
    }
    
    #[test]
    fn test_training_history_default() {
        let history = TrainingHistory::default();
        assert!(history.train_loss.is_empty());
        assert!(history.valid_loss.is_empty());
        assert!(history.train_metrics.is_empty());
        assert!(history.valid_metrics.is_empty());
        assert!(history.early_stopping.is_none());
    }
    
    #[test]
    fn test_node_types() {
        assert_eq!(NodeType::Internal, NodeType::Internal);
        assert_eq!(NodeType::Leaf, NodeType::Leaf);
        assert_ne!(NodeType::Internal, NodeType::Leaf);
    }
    
    #[test]
    fn test_objective_types() {
        assert_eq!(ObjectiveType::Regression, ObjectiveType::Regression);
        assert_eq!(ObjectiveType::Binary, ObjectiveType::Binary);
        assert_eq!(ObjectiveType::Multiclass, ObjectiveType::Multiclass);
        assert_eq!(ObjectiveType::Ranking, ObjectiveType::Ranking);
    }
    
    #[test]
    fn test_utils_functions() {
        assert!(utils::performance_serializer().is_ok());
        assert!(utils::performance_deserializer().is_ok());
        assert!(utils::compact_serializer().is_ok());
        
        let version_info = utils::version_info();
        assert!(!version_info.bincode_version.is_empty());
        assert!(!version_info.format_version.is_empty());
        
        assert_eq!(utils::estimate_compression_ratio(500), 1.0);
        assert_eq!(utils::estimate_compression_ratio(5000), 0.9);
        assert_eq!(utils::estimate_compression_ratio(50000), 0.8);
        assert_eq!(utils::estimate_compression_ratio(500000), 0.7);
    }
}