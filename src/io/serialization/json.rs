//! JSON serialization for Pure Rust LightGBM models.
//!
//! This module provides human-readable JSON serialization for model portability,
//! debugging, and integration with other systems. While less compact than bincode,
//! JSON format offers excellent readability and cross-platform compatibility.

use crate::core::error::{Result, LightGBMError};
use crate::io::serialization::{
    ModelDeserializer, ModelSerializer, SerializationConfig, SerializationError,
    SerializationFormat, SerializationStats,
};
use crate::io::SerializableModel;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::io::{Read, Write};
use std::time::Instant;

/// JSON serialization configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonConfig {
    /// Base serialization config
    pub base: SerializationConfig,
    /// Pretty print JSON output
    pub pretty: bool,
    /// Include null values in output
    pub include_null: bool,
    /// Use compact representation for arrays
    pub compact_arrays: bool,
    /// Include type information
    pub include_type_info: bool,
    /// Include schema version
    pub include_schema_version: bool,
    /// Maximum nesting depth
    pub max_depth: usize,
    /// Escape unicode characters
    pub escape_unicode: bool,
}

impl Default for JsonConfig {
    fn default() -> Self {
        JsonConfig {
            base: SerializationConfig {
                format: SerializationFormat::Json,
                pretty_json: true,
                ..Default::default()
            },
            pretty: true,
            include_null: false,
            compact_arrays: true,
            include_type_info: true,
            include_schema_version: true,
            max_depth: 100,
            escape_unicode: false,
        }
    }
}

/// JSON serializer for LightGBM models
pub struct JsonSerializer {
    config: JsonConfig,
    stats: Option<SerializationStats>,
}

impl JsonSerializer {
    /// Create a new JSON serializer
    pub fn new() -> Result<Self> {
        Ok(JsonSerializer {
            config: JsonConfig::default(),
            stats: None,
        })
    }

    /// Create a new JSON serializer with custom configuration
    pub fn with_config(config: JsonConfig) -> Result<Self> {
        Ok(JsonSerializer {
            config,
            stats: None,
        })
    }

    /// Serialize model to JSON format
    fn serialize_internal(&self, model: &dyn SerializableModel) -> Result<Vec<u8>> {
        let start_time = Instant::now();
        
        // Validate model if configured
        if self.config.base.validate_before_serialize {
            model.validate()?;
        }

        // Create JSON representation
        let json_model = JsonModelWrapper::from_model(model, &self.config)?;
        
        // Serialize to JSON
        let json_string = if self.config.pretty {
            serde_json::to_string_pretty(&json_model)
        } else {
            serde_json::to_string(&json_model)
        };
        
        let json_bytes = json_string
            .map_err(|e| SerializationError::SerializationFailed(e.to_string()))?
            .into_bytes();

        let serialization_time = start_time.elapsed().as_millis() as u64;
        
        // Update statistics
        if let Some(ref mut stats) = self.stats.as_mut() {
            stats.serialization_time_ms = serialization_time;
            stats.serialized_size = json_bytes.len();
        }

        Ok(json_bytes)
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

impl Default for JsonSerializer {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

impl ModelSerializer for JsonSerializer {
    fn serialize(&self, model: &dyn SerializableModel) -> Result<Vec<u8>> {
        self.serialize_internal(model)
    }

    fn serialize_to_writer(
        &self,
        model: &dyn SerializableModel,
        writer: &mut dyn Write,
    ) -> Result<()> {
        // Validate model if configured
        if self.config.base.validate_before_serialize {
            model.validate()?;
        }

        // Create JSON representation
        let json_model = JsonModelWrapper::from_model(model, &self.config)?;
        
        // Serialize directly to writer
        if self.config.pretty {
            serde_json::to_writer_pretty(writer, &json_model)
        } else {
            serde_json::to_writer(writer, &json_model)
        }
        .map_err(|e| SerializationError::SerializationFailed(e.to_string()))?;
        
        Ok(())
    }

    fn format(&self) -> SerializationFormat {
        SerializationFormat::Json
    }

    fn config(&self) -> &SerializationConfig {
        &self.config.base
    }

    fn set_config(&mut self, config: SerializationConfig) {
        self.config.base = config;
    }

    fn estimate_size(&self, model: &dyn SerializableModel) -> usize {
        // JSON is typically 2-4x larger than bincode
        let base_size = 2048; // Base JSON overhead
        let feature_size = model.num_features() * 128; // ~128 bytes per feature in JSON
        let metadata_size = model.metadata().len() * 64; // ~64 bytes per metadata entry
        
        let estimated_size = base_size + feature_size + metadata_size;
        if self.config.pretty {
            (estimated_size as f64 * 1.3) as usize // Pretty printing adds ~30% overhead
        } else {
            estimated_size
        }
    }
}

/// JSON deserializer for LightGBM models
pub struct JsonDeserializer {
    config: JsonConfig,
    stats: Option<SerializationStats>,
}

impl JsonDeserializer {
    /// Create a new JSON deserializer
    pub fn new() -> Result<Self> {
        Ok(JsonDeserializer {
            config: JsonConfig::default(),
            stats: None,
        })
    }

    /// Create a new JSON deserializer with custom configuration
    pub fn with_config(config: JsonConfig) -> Result<Self> {
        Ok(JsonDeserializer {
            config,
            stats: None,
        })
    }

    /// Deserialize model from JSON format
    fn deserialize_internal(&self, data: &[u8]) -> Result<Box<dyn SerializableModel>> {
        let start_time = Instant::now();
        
        // Parse JSON
        let json_str = std::str::from_utf8(data)
            .map_err(|e| SerializationError::DeserializationFailed(
                format!("Invalid UTF-8: {}", e)
            ))?;
        
        let json_model: JsonModelWrapper = serde_json::from_str(json_str)
            .map_err(|e| SerializationError::DeserializationFailed(e.to_string()))?;
        
        // Validate schema version if included
        if self.config.include_schema_version {
            self.validate_schema_version(&json_model)?;
        }
        
        // Convert to model
        let model = json_model.to_model()?;
        
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

    /// Validate schema version
    fn validate_schema_version(&self, model: &JsonModelWrapper) -> Result<()> {
        if let Some(ref schema_version) = model.schema_version {
            if schema_version != "1.0.0" {
                return Err(SerializationError::ValidationFailed(
                    format!("Unsupported schema version: {}", schema_version)
                ).into());
            }
        }
        Ok(())
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

impl Default for JsonDeserializer {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

impl ModelDeserializer for JsonDeserializer {
    fn deserialize(&self, data: &[u8]) -> Result<Box<dyn SerializableModel>> {
        self.deserialize_internal(data)
    }

    fn deserialize_from_reader(
        &self,
        reader: &mut dyn Read,
    ) -> Result<Box<dyn SerializableModel>> {
        let start_time = Instant::now();
        
        // Parse JSON from reader
        let json_model: JsonModelWrapper = serde_json::from_reader(reader)
            .map_err(|e| SerializationError::DeserializationFailed(e.to_string()))?;
        
        // Validate schema version if included
        if self.config.include_schema_version {
            self.validate_schema_version(&json_model)?;
        }
        
        // Convert to model
        let model = json_model.to_model()?;
        
        // Validate model if configured
        if !self.config.base.skip_validation {
            model.validate()?;
        }

        let deserialization_time = start_time.elapsed().as_millis() as u64;
        
        // Update statistics
        if let Some(ref mut stats) = self.stats.as_mut() {
            stats.deserialization_time_ms = deserialization_time;
        }

        Ok(model)
    }

    fn format(&self) -> SerializationFormat {
        SerializationFormat::Json
    }

    fn config(&self) -> &SerializationConfig {
        &self.config.base
    }

    fn set_config(&mut self, config: SerializationConfig) {
        self.config.base = config;
    }
}

/// JSON model wrapper for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonModelWrapper {
    /// Schema version
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema_version: Option<String>,
    /// Model type
    pub model_type: String,
    /// Model version
    pub version: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Feature information
    pub features: FeatureInfo,
    /// Model metadata
    pub metadata: Map<String, Value>,
    /// Model configuration
    pub config: ModelConfig,
    /// Model data
    pub model: ModelJsonData,
    /// Training information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training: Option<TrainingInfo>,
    /// Serialization information
    pub serialization: SerializationInfo,
}

impl JsonModelWrapper {
    /// Create wrapper from model
    pub fn from_model(
        model: &dyn SerializableModel,
        config: &JsonConfig,
    ) -> Result<Self> {
        let metadata = model.metadata();
        let json_metadata = metadata.into_iter()
            .map(|(k, v)| (k, Value::String(v)))
            .collect();
        
        let model_data = ModelJsonData::from_model(model, config)?;
        
        Ok(JsonModelWrapper {
            schema_version: if config.include_schema_version {
                Some("1.0.0".to_string())
            } else {
                None
            },
            model_type: model.model_type().to_string(),
            version: model.version(),
            created_at: model.created_at(),
            features: FeatureInfo {
                names: model.feature_names().to_vec(),
                count: model.num_features(),
                types: vec!["numerical".to_string(); model.num_features()], // Simplified
            },
            metadata: json_metadata,
            config: ModelConfig::default(),
            model: model_data,
            training: if config.base.include_training_history {
                Some(TrainingInfo::default())
            } else {
                None
            },
            serialization: SerializationInfo {
                format: "json".to_string(),
                version: "1.0.0".to_string(),
                timestamp: chrono::Utc::now(),
                pretty: config.pretty,
            },
        })
    }

    /// Convert wrapper to model
    pub fn to_model(&self) -> Result<Box<dyn SerializableModel>> {
        let mut model = JsonModel::new(self.model.clone());
        
        // Set metadata
        let metadata = self.metadata.iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
            .collect();
        model.set_metadata(metadata);
        
        Ok(Box::new(model))
    }
}

/// Feature information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureInfo {
    /// Feature names
    pub names: Vec<String>,
    /// Feature count
    pub count: usize,
    /// Feature types
    pub types: Vec<String>,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Objective function
    pub objective: String,
    /// Number of classes (for classification)
    pub num_class: Option<usize>,
    /// Training parameters
    pub params: Map<String, Value>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            objective: "regression".to_string(),
            num_class: None,
            params: Map::new(),
        }
    }
}

/// Model data in JSON format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelJsonData {
    /// Trees
    pub trees: Vec<JsonTree>,
    /// Tree info
    pub tree_info: Vec<TreeInfo>,
    /// Leaf output
    pub leaf_output: Option<Vec<f64>>,
    /// Leaf weight
    pub leaf_weight: Option<Vec<f64>>,
    /// Leaf count
    pub leaf_count: Option<Vec<usize>>,
    /// Feature importance
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_importance: Option<Vec<f64>>,
}

impl ModelJsonData {
    /// Create from model
    pub fn from_model(
        model: &dyn SerializableModel,
        config: &JsonConfig,
    ) -> Result<Self> {
        Ok(ModelJsonData {
            trees: Vec::new(), // Would be populated from actual model
            tree_info: Vec::new(),
            leaf_output: None,
            leaf_weight: None,
            leaf_count: None,
            feature_importance: if config.base.include_feature_importance {
                Some(vec![0.0; model.num_features()])
            } else {
                None
            },
        })
    }
}

/// JSON tree representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonTree {
    /// Tree structure
    pub tree_structure: JsonTreeStructure,
    /// Tree index
    pub tree_index: usize,
    /// Tree weight
    pub tree_weight: Option<f64>,
}

/// JSON tree structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonTreeStructure {
    /// Tree nodes
    pub tree: Vec<JsonNode>,
    /// Number of leaves
    pub num_leaves: usize,
    /// Number of nodes
    pub num_nodes: usize,
}

/// JSON node representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonNode {
    /// Split feature index
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split_feature: Option<usize>,
    /// Split gain
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split_gain: Option<f64>,
    /// Threshold
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    /// Decision type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decision_type: Option<String>,
    /// Default left
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_left: Option<bool>,
    /// Missing type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub missing_type: Option<String>,
    /// Left child
    #[serde(skip_serializing_if = "Option::is_none")]
    pub left_child: Option<usize>,
    /// Right child
    #[serde(skip_serializing_if = "Option::is_none")]
    pub right_child: Option<usize>,
    /// Leaf index
    #[serde(skip_serializing_if = "Option::is_none")]
    pub leaf_index: Option<usize>,
    /// Leaf value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub leaf_value: Option<f64>,
    /// Leaf weight
    #[serde(skip_serializing_if = "Option::is_none")]
    pub leaf_weight: Option<f64>,
    /// Leaf count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub leaf_count: Option<usize>,
    /// Internal value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub internal_value: Option<f64>,
    /// Internal weight
    #[serde(skip_serializing_if = "Option::is_none")]
    pub internal_weight: Option<f64>,
    /// Internal count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub internal_count: Option<usize>,
}

/// Tree information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeInfo {
    /// Tree index
    pub tree_index: usize,
    /// Number of leaves
    pub num_leaves: usize,
    /// Number of cat
    pub num_cat: usize,
    /// Shrinkage
    pub shrinkage: f64,
}

/// Training information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingInfo {
    /// Training parameters
    pub params: Map<String, Value>,
    /// Training history
    pub history: TrainingHistory,
    /// Validation results
    pub validation: Option<ValidationResults>,
}

impl Default for TrainingInfo {
    fn default() -> Self {
        TrainingInfo {
            params: Map::new(),
            history: TrainingHistory::default(),
            validation: None,
        }
    }
}

/// Training history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHistory {
    /// Training iterations
    pub iterations: Vec<u32>,
    /// Training loss
    pub train_loss: Vec<f64>,
    /// Validation loss
    pub valid_loss: Vec<f64>,
    /// Additional metrics
    pub metrics: Map<String, Vec<f64>>,
}

impl Default for TrainingHistory {
    fn default() -> Self {
        TrainingHistory {
            iterations: Vec::new(),
            train_loss: Vec::new(),
            valid_loss: Vec::new(),
            metrics: Map::new(),
        }
    }
}

/// Validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    /// Best iteration
    pub best_iteration: usize,
    /// Best score
    pub best_score: f64,
    /// Early stopping
    pub early_stopping: Option<EarlyStoppingInfo>,
}

/// Early stopping information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingInfo {
    /// Stopped at iteration
    pub stopped_at: usize,
    /// Reason
    pub reason: String,
    /// Tolerance
    pub tolerance: f64,
}

/// Serialization information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializationInfo {
    /// Format used
    pub format: String,
    /// Format version
    pub version: String,
    /// Serialization timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Pretty printed
    pub pretty: bool,
}

/// Simple JSON model implementation
pub struct JsonModel {
    data: ModelJsonData,
    metadata: std::collections::HashMap<String, String>,
    feature_names: Vec<String>,
}

impl JsonModel {
    /// Create new JSON model
    pub fn new(data: ModelJsonData) -> Self {
        JsonModel {
            data,
            metadata: std::collections::HashMap::new(),
            feature_names: Vec::new(),
        }
    }

    /// Set feature names
    pub fn set_feature_names(&mut self, names: Vec<String>) {
        self.feature_names = names;
    }
}

impl SerializableModel for JsonModel {
    fn model_type(&self) -> &'static str {
        "json_model"
    }

    fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    fn num_features(&self) -> usize {
        self.feature_names.len()
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
        
        if self.feature_names.is_empty() {
            return Err(LightGBMError::model("Model has no feature names"));
        }
        
        Ok(())
    }

    fn clone_model(&self) -> Box<dyn SerializableModel> {
        Box::new(JsonModel {
            data: self.data.clone(),
            metadata: self.metadata.clone(),
            feature_names: self.feature_names.clone(),
        })
    }
}

/// JSON serialization utilities
pub mod utils {
    use super::*;
    
    /// Create pretty JSON serializer
    pub fn pretty_serializer() -> Result<JsonSerializer> {
        let config = JsonConfig {
            pretty: true,
            include_type_info: true,
            include_schema_version: true,
            ..Default::default()
        };
        
        JsonSerializer::with_config(config)
    }
    
    /// Create compact JSON serializer
    pub fn compact_serializer() -> Result<JsonSerializer> {
        let config = JsonConfig {
            pretty: false,
            include_null: false,
            compact_arrays: true,
            include_type_info: false,
            include_schema_version: false,
            ..Default::default()
        };
        
        JsonSerializer::with_config(config)
    }
    
    /// Create debug JSON serializer (includes everything)
    pub fn debug_serializer() -> Result<JsonSerializer> {
        let config = JsonConfig {
            base: SerializationConfig {
                format: SerializationFormat::Json,
                pretty_json: true,
                include_metadata: true,
                include_feature_importance: true,
                include_training_history: true,
                ..Default::default()
            },
            pretty: true,
            include_null: true,
            compact_arrays: false,
            include_type_info: true,
            include_schema_version: true,
            ..Default::default()
        };
        
        JsonSerializer::with_config(config)
    }
    
    /// Validate JSON structure
    pub fn validate_json_structure(json_str: &str) -> Result<()> {
        let _: Value = serde_json::from_str(json_str)
            .map_err(|e| SerializationError::ValidationFailed(e.to_string()))?;
        Ok(())
    }
    
    /// Pretty print JSON string
    pub fn pretty_print(json_str: &str) -> Result<String> {
        let value: Value = serde_json::from_str(json_str)
            .map_err(|e| SerializationError::DeserializationFailed(e.to_string()))?;
        
        serde_json::to_string_pretty(&value)
            .map_err(|e| SerializationError::SerializationFailed(e.to_string()))
    }
    
    /// Minify JSON string
    pub fn minify(json_str: &str) -> Result<String> {
        let value: Value = serde_json::from_str(json_str)
            .map_err(|e| SerializationError::DeserializationFailed(e.to_string()))?;
        
        serde_json::to_string(&value)
            .map_err(|e| SerializationError::SerializationFailed(e.to_string()))
    }
    
    /// Get JSON schema version
    pub fn get_schema_version() -> &'static str {
        "1.0.0"
    }
    
    /// Check if JSON is model format
    pub fn is_model_json(json_str: &str) -> bool {
        if let Ok(value) = serde_json::from_str::<Value>(json_str) {
            if let Some(obj) = value.as_object() {
                return obj.contains_key("model_type") && 
                       obj.contains_key("model") &&
                       obj.contains_key("features");
            }
        }
        false
    }
    
    /// Estimate JSON size
    pub fn estimate_json_size(
        num_features: usize,
        num_trees: usize,
        num_leaves_per_tree: usize,
        pretty: bool,
    ) -> usize {
        let base_size = 1024;
        let feature_size = num_features * 50; // ~50 bytes per feature
        let tree_size = num_trees * num_leaves_per_tree * 200; // ~200 bytes per node
        let metadata_size = 2048; // Base metadata
        
        let total_size = base_size + feature_size + tree_size + metadata_size;
        
        if pretty {
            (total_size as f64 * 1.5) as usize // Pretty printing adds ~50% overhead
        } else {
            total_size
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_json_serializer_creation() {
        let serializer = JsonSerializer::new();
        assert!(serializer.is_ok());
        
        let serializer = serializer.unwrap();
        assert_eq!(serializer.format(), SerializationFormat::Json);
    }
    
    #[test]
    fn test_json_deserializer_creation() {
        let deserializer = JsonDeserializer::new();
        assert!(deserializer.is_ok());
        
        let deserializer = deserializer.unwrap();
        assert_eq!(deserializer.format(), SerializationFormat::Json);
    }
    
    #[test]
    fn test_json_config_default() {
        let config = JsonConfig::default();
        assert_eq!(config.base.format, SerializationFormat::Json);
        assert!(config.pretty);
        assert!(!config.include_null);
        assert!(config.compact_arrays);
        assert!(config.include_type_info);
        assert!(config.include_schema_version);
    }
    
    #[test]
    fn test_feature_info() {
        let feature_info = FeatureInfo {
            names: vec!["feature1".to_string(), "feature2".to_string()],
            count: 2,
            types: vec!["numerical".to_string(), "categorical".to_string()],
        };
        
        assert_eq!(feature_info.names.len(), 2);
        assert_eq!(feature_info.count, 2);
        assert_eq!(feature_info.types.len(), 2);
    }
    
    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.objective, "regression");
        assert!(config.num_class.is_none());
        assert!(config.params.is_empty());
    }
    
    #[test]
    fn test_training_info_default() {
        let training_info = TrainingInfo::default();
        assert!(training_info.params.is_empty());
        assert!(training_info.history.iterations.is_empty());
        assert!(training_info.validation.is_none());
    }
    
    #[test]
    fn test_utils_functions() {
        assert!(utils::pretty_serializer().is_ok());
        assert!(utils::compact_serializer().is_ok());
        assert!(utils::debug_serializer().is_ok());
        
        let json_str = r#"{"test": "value"}"#;
        assert!(utils::validate_json_structure(json_str).is_ok());
        
        let pretty = utils::pretty_print(json_str).unwrap();
        assert!(pretty.contains("{\n"));
        
        let minified = utils::minify(&pretty).unwrap();
        assert!(!minified.contains("{\n"));
        
        assert_eq!(utils::get_schema_version(), "1.0.0");
        
        let model_json = r#"{"model_type": "test", "model": {}, "features": {"names": []}}"#;
        assert!(utils::is_model_json(model_json));
        
        let non_model_json = r#"{"test": "value"}"#;
        assert!(!utils::is_model_json(non_model_json));
    }
    
    #[test]
    fn test_json_size_estimation() {
        let size = utils::estimate_json_size(10, 5, 31, false);
        assert!(size > 0);
        
        let pretty_size = utils::estimate_json_size(10, 5, 31, true);
        assert!(pretty_size > size);
    }
    
    #[test]
    fn test_serialization_info() {
        let info = SerializationInfo {
            format: "json".to_string(),
            version: "1.0.0".to_string(),
            timestamp: chrono::Utc::now(),
            pretty: true,
        };
        
        assert_eq!(info.format, "json");
        assert_eq!(info.version, "1.0.0");
        assert!(info.pretty);
    }
}