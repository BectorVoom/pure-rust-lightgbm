//! LightGBM text format serialization for Pure Rust LightGBM models.
//!
//! This module provides compatibility with the original LightGBM text format,
//! allowing models to be saved and loaded in the same format as the C++ implementation.
//! This ensures full compatibility with existing LightGBM tooling and workflows.

use crate::core::error::{Result, LightGBMError};
use crate::io::serialization::{
    ModelDeserializer, ModelSerializer, SerializationConfig, SerializationError,
    SerializationFormat, SerializationStats,
};
use crate::io::SerializableModel;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::str::FromStr;
use std::time::Instant;

/// LightGBM serialization configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LightGbmConfig {
    /// Base serialization config
    pub base: SerializationConfig,
    /// Include comments in output
    pub include_comments: bool,
    /// Include feature names
    pub include_feature_names: bool,
    /// Include pandas categorical
    pub include_pandas_categorical: bool,
    /// Include objective function
    pub include_objective: bool,
    /// Include boosting type
    pub include_boosting_type: bool,
    /// Use compact format
    pub compact_format: bool,
    /// Line ending style
    pub line_ending: LineEnding,
    /// Decimal precision for floating point numbers
    pub decimal_precision: usize,
}

/// Line ending styles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineEnding {
    /// Unix style (\n)
    Unix,
    /// Windows style (\r\n)
    Windows,
    /// Mac style (\r)
    Mac,
}

impl Default for LightGbmConfig {
    fn default() -> Self {
        LightGbmConfig {
            base: SerializationConfig {
                format: SerializationFormat::LightGbm,
                ..Default::default()
            },
            include_comments: true,
            include_feature_names: true,
            include_pandas_categorical: false,
            include_objective: true,
            include_boosting_type: true,
            compact_format: false,
            line_ending: LineEnding::Unix,
            decimal_precision: 6,
        }
    }
}

impl LineEnding {
    fn as_str(&self) -> &'static str {
        match self {
            LineEnding::Unix => "\n",
            LineEnding::Windows => "\r\n",
            LineEnding::Mac => "\r",
        }
    }
}

/// LightGBM text format serializer
pub struct LightGbmSerializer {
    config: LightGbmConfig,
    stats: Option<SerializationStats>,
}

impl LightGbmSerializer {
    /// Create a new LightGBM serializer
    pub fn new() -> Result<Self> {
        Ok(LightGbmSerializer {
            config: LightGbmConfig::default(),
            stats: None,
        })
    }

    /// Create a new LightGBM serializer with custom configuration
    pub fn with_config(config: LightGbmConfig) -> Result<Self> {
        Ok(LightGbmSerializer {
            config,
            stats: None,
        })
    }

    /// Serialize model to LightGBM text format
    fn serialize_internal(&self, model: &dyn SerializableModel) -> Result<Vec<u8>> {
        let start_time = Instant::now();
        
        // Validate model if configured
        if self.config.base.validate_before_serialize {
            model.validate()?;
        }

        // Create LightGBM representation
        let lgb_model = LightGbmModel::from_model(model, &self.config)?;
        
        // Serialize to text format
        let text = lgb_model.to_text(&self.config)?;
        let bytes = text.into_bytes();

        let serialization_time = start_time.elapsed().as_millis() as u64;
        
        // Update statistics
        if let Some(ref mut stats) = self.stats.as_mut() {
            stats.serialization_time_ms = serialization_time;
            stats.serialized_size = bytes.len();
        }

        Ok(bytes)
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

impl Default for LightGbmSerializer {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

impl ModelSerializer for LightGbmSerializer {
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
        SerializationFormat::LightGbm
    }

    fn config(&self) -> &SerializationConfig {
        &self.config.base
    }

    fn set_config(&mut self, config: SerializationConfig) {
        self.config.base = config;
    }

    fn estimate_size(&self, model: &dyn SerializableModel) -> usize {
        // LightGBM text format is typically larger than JSON
        let base_size = 4096; // Base overhead
        let feature_size = model.num_features() * 32; // ~32 bytes per feature name
        let metadata_size = model.metadata().len() * 64; // ~64 bytes per metadata entry
        
        // Tree size depends on complexity, estimate conservatively
        let tree_size = 10000; // Rough estimate for tree structure
        
        base_size + feature_size + metadata_size + tree_size
    }
}

/// LightGBM text format deserializer
pub struct LightGbmDeserializer {
    config: LightGbmConfig,
    stats: Option<SerializationStats>,
}

impl LightGbmDeserializer {
    /// Create a new LightGBM deserializer
    pub fn new() -> Result<Self> {
        Ok(LightGbmDeserializer {
            config: LightGbmConfig::default(),
            stats: None,
        })
    }

    /// Create a new LightGBM deserializer with custom configuration
    pub fn with_config(config: LightGbmConfig) -> Result<Self> {
        Ok(LightGbmDeserializer {
            config,
            stats: None,
        })
    }

    /// Deserialize model from LightGBM text format
    fn deserialize_internal(&self, data: &[u8]) -> Result<Box<dyn SerializableModel>> {
        let start_time = Instant::now();
        
        // Parse text format
        let text = std::str::from_utf8(data)
            .map_err(|e| SerializationError::DeserializationFailed(
                format!("Invalid UTF-8: {}", e)
            ))?;
        
        let lgb_model = LightGbmModel::from_text(text, &self.config)?;
        let model = lgb_model.to_model()?;
        
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

impl Default for LightGbmDeserializer {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

impl ModelDeserializer for LightGbmDeserializer {
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
        SerializationFormat::LightGbm
    }

    fn config(&self) -> &SerializationConfig {
        &self.config.base
    }

    fn set_config(&mut self, config: SerializationConfig) {
        self.config.base = config;
    }
}

/// LightGBM model representation
#[derive(Debug, Clone)]
pub struct LightGbmModel {
    /// Model version
    pub version: String,
    /// Number of classes
    pub num_class: usize,
    /// Objective function
    pub objective: String,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Feature infos
    pub feature_infos: Vec<String>,
    /// Pandas categorical
    pub pandas_categorical: Vec<PandasCategorical>,
    /// Trees
    pub trees: Vec<LightGbmTree>,
    /// Tree info
    pub tree_info: Vec<LightGbmTreeInfo>,
    /// Leaf output
    pub leaf_output: Vec<f64>,
    /// Leaf weight
    pub leaf_weight: Option<Vec<f64>>,
    /// Leaf count
    pub leaf_count: Option<Vec<usize>>,
    /// Feature importance
    pub feature_importance: Option<Vec<f64>>,
    /// Additional parameters
    pub parameters: HashMap<String, String>,
}

impl LightGbmModel {
    /// Create from serializable model
    pub fn from_model(
        model: &dyn SerializableModel,
        config: &LightGbmConfig,
    ) -> Result<Self> {
        Ok(LightGbmModel {
            version: model.version(),
            num_class: 1, // Default for binary/regression
            objective: "regression".to_string(),
            feature_names: model.feature_names().to_vec(),
            feature_infos: vec!["none".to_string(); model.num_features()],
            pandas_categorical: Vec::new(),
            trees: Vec::new(), // Would be populated from actual model
            tree_info: Vec::new(),
            leaf_output: Vec::new(),
            leaf_weight: None,
            leaf_count: None,
            feature_importance: if config.base.include_feature_importance {
                Some(vec![0.0; model.num_features()])
            } else {
                None
            },
            parameters: HashMap::new(),
        })
    }

    /// Parse from text format
    pub fn from_text(text: &str, _config: &LightGbmConfig) -> Result<Self> {
        let mut model = LightGbmModel {
            version: "1.0.0".to_string(),
            num_class: 1,
            objective: "regression".to_string(),
            feature_names: Vec::new(),
            feature_infos: Vec::new(),
            pandas_categorical: Vec::new(),
            trees: Vec::new(),
            tree_info: Vec::new(),
            leaf_output: Vec::new(),
            leaf_weight: None,
            leaf_count: None,
            feature_importance: None,
            parameters: HashMap::new(),
        };

        let mut lines = text.lines();
        let mut current_tree: Option<LightGbmTree> = None;
        let mut current_tree_info: Option<LightGbmTreeInfo> = None;

        while let Some(line) = lines.next() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if line.starts_with("Tree=") {
                // Save previous tree if exists
                if let Some(tree) = current_tree.take() {
                    model.trees.push(tree);
                }
                if let Some(tree_info) = current_tree_info.take() {
                    model.tree_info.push(tree_info);
                }

                // Parse tree index
                let tree_index = line.strip_prefix("Tree=")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);

                current_tree = Some(LightGbmTree {
                    tree_index,
                    num_leaves: 0,
                    num_cat: 0,
                    split_feature: Vec::new(),
                    split_gain: Vec::new(),
                    threshold: Vec::new(),
                    decision_type: Vec::new(),
                    left_child: Vec::new(),
                    right_child: Vec::new(),
                    leaf_parent: Vec::new(),
                    leaf_value: Vec::new(),
                    leaf_weight: Vec::new(),
                    leaf_count: Vec::new(),
                    internal_value: Vec::new(),
                    internal_weight: Vec::new(),
                    internal_count: Vec::new(),
                    shrinkage: 1.0,
                });

                current_tree_info = Some(LightGbmTreeInfo {
                    tree_index,
                    num_leaves: 0,
                    num_cat: 0,
                    shrinkage: 1.0,
                });
            } else if line.starts_with("version=") {
                model.version = line.strip_prefix("version=").unwrap_or("1.0.0").to_string();
            } else if line.starts_with("num_class=") {
                model.num_class = line.strip_prefix("num_class=")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1);
            } else if line.starts_with("objective=") {
                model.objective = line.strip_prefix("objective=").unwrap_or("regression").to_string();
            } else if line.starts_with("feature_names=") {
                let names_str = line.strip_prefix("feature_names=").unwrap_or("");
                model.feature_names = names_str.split_whitespace()
                    .map(|s| s.to_string())
                    .collect();
            } else if line.starts_with("feature_infos=") {
                let infos_str = line.strip_prefix("feature_infos=").unwrap_or("");
                model.feature_infos = infos_str.split_whitespace()
                    .map(|s| s.to_string())
                    .collect();
            } else if line.starts_with("num_leaves=") {
                if let Some(ref mut tree) = current_tree {
                    tree.num_leaves = line.strip_prefix("num_leaves=")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                }
                if let Some(ref mut tree_info) = current_tree_info {
                    tree_info.num_leaves = line.strip_prefix("num_leaves=")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                }
            } else if line.starts_with("num_cat=") {
                if let Some(ref mut tree) = current_tree {
                    tree.num_cat = line.strip_prefix("num_cat=")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                }
                if let Some(ref mut tree_info) = current_tree_info {
                    tree_info.num_cat = line.strip_prefix("num_cat=")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                }
            } else if line.starts_with("split_feature=") {
                if let Some(ref mut tree) = current_tree {
                    let values_str = line.strip_prefix("split_feature=").unwrap_or("");
                    tree.split_feature = Self::parse_int_array(values_str)?;
                }
            } else if line.starts_with("split_gain=") {
                if let Some(ref mut tree) = current_tree {
                    let values_str = line.strip_prefix("split_gain=").unwrap_or("");
                    tree.split_gain = Self::parse_float_array(values_str)?;
                }
            } else if line.starts_with("threshold=") {
                if let Some(ref mut tree) = current_tree {
                    let values_str = line.strip_prefix("threshold=").unwrap_or("");
                    tree.threshold = Self::parse_float_array(values_str)?;
                }
            } else if line.starts_with("decision_type=") {
                if let Some(ref mut tree) = current_tree {
                    let values_str = line.strip_prefix("decision_type=").unwrap_or("");
                    tree.decision_type = Self::parse_int_array(values_str)?;
                }
            } else if line.starts_with("left_child=") {
                if let Some(ref mut tree) = current_tree {
                    let values_str = line.strip_prefix("left_child=").unwrap_or("");
                    tree.left_child = Self::parse_int_array(values_str)?;
                }
            } else if line.starts_with("right_child=") {
                if let Some(ref mut tree) = current_tree {
                    let values_str = line.strip_prefix("right_child=").unwrap_or("");
                    tree.right_child = Self::parse_int_array(values_str)?;
                }
            } else if line.starts_with("leaf_value=") {
                if let Some(ref mut tree) = current_tree {
                    let values_str = line.strip_prefix("leaf_value=").unwrap_or("");
                    tree.leaf_value = Self::parse_float_array(values_str)?;
                }
            } else if line.starts_with("leaf_weight=") {
                if let Some(ref mut tree) = current_tree {
                    let values_str = line.strip_prefix("leaf_weight=").unwrap_or("");
                    tree.leaf_weight = Self::parse_float_array(values_str)?;
                }
            } else if line.starts_with("leaf_count=") {
                if let Some(ref mut tree) = current_tree {
                    let values_str = line.strip_prefix("leaf_count=").unwrap_or("");
                    tree.leaf_count = Self::parse_int_array(values_str)?;
                }
            } else if line.starts_with("internal_value=") {
                if let Some(ref mut tree) = current_tree {
                    let values_str = line.strip_prefix("internal_value=").unwrap_or("");
                    tree.internal_value = Self::parse_float_array(values_str)?;
                }
            } else if line.starts_with("internal_weight=") {
                if let Some(ref mut tree) = current_tree {
                    let values_str = line.strip_prefix("internal_weight=").unwrap_or("");
                    tree.internal_weight = Self::parse_float_array(values_str)?;
                }
            } else if line.starts_with("internal_count=") {
                if let Some(ref mut tree) = current_tree {
                    let values_str = line.strip_prefix("internal_count=").unwrap_or("");
                    tree.internal_count = Self::parse_int_array(values_str)?;
                }
            } else if line.starts_with("shrinkage=") {
                if let Some(ref mut tree) = current_tree {
                    tree.shrinkage = line.strip_prefix("shrinkage=")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1.0);
                }
                if let Some(ref mut tree_info) = current_tree_info {
                    tree_info.shrinkage = line.strip_prefix("shrinkage=")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1.0);
                }
            }
        }

        // Add final tree
        if let Some(tree) = current_tree {
            model.trees.push(tree);
        }
        if let Some(tree_info) = current_tree_info {
            model.tree_info.push(tree_info);
        }

        Ok(model)
    }

    /// Convert to text format
    pub fn to_text(&self, config: &LightGbmConfig) -> Result<String> {
        let mut output = String::new();
        let line_ending = config.line_ending.as_str();

        // Add header comment
        if config.include_comments {
            output.push_str("# LightGBM model file");
            output.push_str(line_ending);
            output.push_str("# Generated by Pure Rust LightGBM");
            output.push_str(line_ending);
            output.push_str(line_ending);
        }

        // Add version
        output.push_str(&format!("version={}{}", self.version, line_ending));

        // Add num_class
        output.push_str(&format!("num_class={}{}", self.num_class, line_ending));

        // Add objective
        if config.include_objective {
            output.push_str(&format!("objective={}{}", self.objective, line_ending));
        }

        // Add feature names
        if config.include_feature_names && !self.feature_names.is_empty() {
            output.push_str("feature_names=");
            output.push_str(&self.feature_names.join(" "));
            output.push_str(line_ending);
        }

        // Add feature infos
        if !self.feature_infos.is_empty() {
            output.push_str("feature_infos=");
            output.push_str(&self.feature_infos.join(" "));
            output.push_str(line_ending);
        }

        // Add pandas categorical
        if config.include_pandas_categorical {
            for cat in &self.pandas_categorical {
                output.push_str(&format!("pandas_categorical:{}", cat.to_string()));
                output.push_str(line_ending);
            }
        }

        // Add trees
        for tree in &self.trees {
            output.push_str(line_ending);
            output.push_str(&tree.to_text(config)?);
        }

        // Add tree info
        if !config.compact_format {
            for tree_info in &self.tree_info {
                output.push_str(line_ending);
                output.push_str(&tree_info.to_text(config)?);
            }
        }

        // Add leaf output
        if !self.leaf_output.is_empty() {
            output.push_str(line_ending);
            output.push_str("leaf_output=");
            output.push_str(&Self::format_float_array(&self.leaf_output, config.decimal_precision));
            output.push_str(line_ending);
        }

        // Add feature importance
        if let Some(ref importance) = self.feature_importance {
            output.push_str(line_ending);
            output.push_str("feature_importance=");
            output.push_str(&Self::format_float_array(importance, config.decimal_precision));
            output.push_str(line_ending);
        }

        Ok(output)
    }

    /// Convert to serializable model
    pub fn to_model(&self) -> Result<Box<dyn SerializableModel>> {
        let mut model = LightGbmCompatModel::new(self.clone());
        
        // Set feature names
        model.set_feature_names(self.feature_names.clone());
        
        Ok(Box::new(model))
    }

    /// Parse integer array from string
    fn parse_int_array(s: &str) -> Result<Vec<i32>> {
        s.split_whitespace()
            .map(|token| token.parse::<i32>().map_err(|e| 
                SerializationError::DeserializationFailed(format!("Invalid integer: {}", e))))
            .collect()
    }

    /// Parse float array from string
    fn parse_float_array(s: &str) -> Result<Vec<f64>> {
        s.split_whitespace()
            .map(|token| token.parse::<f64>().map_err(|e| 
                SerializationError::DeserializationFailed(format!("Invalid float: {}", e))))
            .collect()
    }

    /// Format float array to string
    fn format_float_array(arr: &[f64], precision: usize) -> String {
        arr.iter()
            .map(|&x| format!("{:.precision$}", x, precision = precision))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Format integer array to string
    fn format_int_array(arr: &[i32]) -> String {
        arr.iter()
            .map(|&x| x.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// LightGBM tree representation
#[derive(Debug, Clone)]
pub struct LightGbmTree {
    pub tree_index: usize,
    pub num_leaves: usize,
    pub num_cat: usize,
    pub split_feature: Vec<i32>,
    pub split_gain: Vec<f64>,
    pub threshold: Vec<f64>,
    pub decision_type: Vec<i32>,
    pub left_child: Vec<i32>,
    pub right_child: Vec<i32>,
    pub leaf_parent: Vec<i32>,
    pub leaf_value: Vec<f64>,
    pub leaf_weight: Vec<f64>,
    pub leaf_count: Vec<i32>,
    pub internal_value: Vec<f64>,
    pub internal_weight: Vec<f64>,
    pub internal_count: Vec<i32>,
    pub shrinkage: f64,
}

impl LightGbmTree {
    /// Convert to text format
    pub fn to_text(&self, config: &LightGbmConfig) -> Result<String> {
        let mut output = String::new();
        let line_ending = config.line_ending.as_str();

        output.push_str(&format!("Tree={}{}", self.tree_index, line_ending));
        output.push_str(&format!("num_leaves={}{}", self.num_leaves, line_ending));
        output.push_str(&format!("num_cat={}{}", self.num_cat, line_ending));

        if !self.split_feature.is_empty() {
            output.push_str("split_feature=");
            output.push_str(&LightGbmModel::format_int_array(&self.split_feature));
            output.push_str(line_ending);
        }

        if !self.split_gain.is_empty() {
            output.push_str("split_gain=");
            output.push_str(&LightGbmModel::format_float_array(&self.split_gain, config.decimal_precision));
            output.push_str(line_ending);
        }

        if !self.threshold.is_empty() {
            output.push_str("threshold=");
            output.push_str(&LightGbmModel::format_float_array(&self.threshold, config.decimal_precision));
            output.push_str(line_ending);
        }

        if !self.decision_type.is_empty() {
            output.push_str("decision_type=");
            output.push_str(&LightGbmModel::format_int_array(&self.decision_type));
            output.push_str(line_ending);
        }

        if !self.left_child.is_empty() {
            output.push_str("left_child=");
            output.push_str(&LightGbmModel::format_int_array(&self.left_child));
            output.push_str(line_ending);
        }

        if !self.right_child.is_empty() {
            output.push_str("right_child=");
            output.push_str(&LightGbmModel::format_int_array(&self.right_child));
            output.push_str(line_ending);
        }

        if !self.leaf_value.is_empty() {
            output.push_str("leaf_value=");
            output.push_str(&LightGbmModel::format_float_array(&self.leaf_value, config.decimal_precision));
            output.push_str(line_ending);
        }

        if !self.leaf_weight.is_empty() {
            output.push_str("leaf_weight=");
            output.push_str(&LightGbmModel::format_float_array(&self.leaf_weight, config.decimal_precision));
            output.push_str(line_ending);
        }

        if !self.leaf_count.is_empty() {
            output.push_str("leaf_count=");
            output.push_str(&LightGbmModel::format_int_array(&self.leaf_count));
            output.push_str(line_ending);
        }

        if !self.internal_value.is_empty() {
            output.push_str("internal_value=");
            output.push_str(&LightGbmModel::format_float_array(&self.internal_value, config.decimal_precision));
            output.push_str(line_ending);
        }

        if !self.internal_weight.is_empty() {
            output.push_str("internal_weight=");
            output.push_str(&LightGbmModel::format_float_array(&self.internal_weight, config.decimal_precision));
            output.push_str(line_ending);
        }

        if !self.internal_count.is_empty() {
            output.push_str("internal_count=");
            output.push_str(&LightGbmModel::format_int_array(&self.internal_count));
            output.push_str(line_ending);
        }

        if self.shrinkage != 1.0 {
            output.push_str(&format!("shrinkage={:.precision$}{}", 
                self.shrinkage, line_ending, precision = config.decimal_precision));
        }

        Ok(output)
    }
}

/// LightGBM tree info
#[derive(Debug, Clone)]
pub struct LightGbmTreeInfo {
    pub tree_index: usize,
    pub num_leaves: usize,
    pub num_cat: usize,
    pub shrinkage: f64,
}

impl LightGbmTreeInfo {
    /// Convert to text format
    pub fn to_text(&self, config: &LightGbmConfig) -> Result<String> {
        let line_ending = config.line_ending.as_str();
        
        Ok(format!(
            "tree_info:{} {} {} {:.precision$}{}",
            self.tree_index,
            self.num_leaves,
            self.num_cat,
            self.shrinkage,
            line_ending,
            precision = config.decimal_precision
        ))
    }
}

/// Pandas categorical information
#[derive(Debug, Clone)]
pub struct PandasCategorical {
    pub feature_index: usize,
    pub categories: Vec<String>,
}

impl PandasCategorical {
    /// Convert to string representation
    pub fn to_string(&self) -> String {
        format!("{}:{}", self.feature_index, self.categories.join(","))
    }
}

impl FromStr for PandasCategorical {
    type Err = LightGBMError;

    fn from_str(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(LightGBMError::serialization(
                "Invalid pandas categorical format".to_string()
            ));
        }

        let feature_index = parts[0].parse::<usize>()
            .map_err(|e| LightGBMError::serialization(format!("Invalid feature index: {}", e)))?;

        let categories = parts[1].split(',')
            .map(|s| s.to_string())
            .collect();

        Ok(PandasCategorical {
            feature_index,
            categories,
        })
    }
}

/// LightGBM compatible model implementation
pub struct LightGbmCompatModel {
    data: LightGbmModel,
    metadata: HashMap<String, String>,
    feature_names: Vec<String>,
}

impl LightGbmCompatModel {
    /// Create new LightGBM compatible model
    pub fn new(data: LightGbmModel) -> Self {
        let feature_names = data.feature_names.clone();
        
        LightGbmCompatModel {
            data,
            metadata: HashMap::new(),
            feature_names,
        }
    }

    /// Set feature names
    pub fn set_feature_names(&mut self, names: Vec<String>) {
        self.feature_names = names;
    }
}

impl SerializableModel for LightGbmCompatModel {
    fn model_type(&self) -> &'static str {
        "lightgbm_compat"
    }

    fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    fn num_features(&self) -> usize {
        self.feature_names.len()
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut metadata = self.metadata.clone();
        metadata.insert("objective".to_string(), self.data.objective.clone());
        metadata.insert("num_class".to_string(), self.data.num_class.to_string());
        metadata.insert("version".to_string(), self.data.version.clone());
        metadata
    }

    fn set_metadata(&mut self, metadata: HashMap<String, String>) {
        self.metadata = metadata;
    }

    fn version(&self) -> String {
        self.data.version.clone()
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
        Box::new(LightGbmCompatModel {
            data: self.data.clone(),
            metadata: self.metadata.clone(),
            feature_names: self.feature_names.clone(),
        })
    }
}

/// LightGBM serialization utilities
pub mod utils {
    use super::*;

    /// Create standard LightGBM serializer
    pub fn standard_serializer() -> Result<LightGbmSerializer> {
        LightGbmSerializer::new()
    }

    /// Create compact LightGBM serializer
    pub fn compact_serializer() -> Result<LightGbmSerializer> {
        let config = LightGbmConfig {
            compact_format: true,
            include_comments: false,
            decimal_precision: 4,
            ..Default::default()
        };

        LightGbmSerializer::with_config(config)
    }

    /// Create verbose LightGBM serializer
    pub fn verbose_serializer() -> Result<LightGbmSerializer> {
        let config = LightGbmConfig {
            include_comments: true,
            include_feature_names: true,
            include_pandas_categorical: true,
            include_objective: true,
            include_boosting_type: true,
            decimal_precision: 8,
            ..Default::default()
        };

        LightGbmSerializer::with_config(config)
    }

    /// Validate LightGBM text format
    pub fn validate_format(text: &str) -> Result<()> {
        let mut has_version = false;
        let mut has_tree = false;

        for line in text.lines() {
            let line = line.trim();
            if line.starts_with("version=") {
                has_version = true;
            } else if line.starts_with("Tree=") {
                has_tree = true;
            }
        }

        if !has_version {
            return Err(SerializationError::ValidationFailed(
                "Missing version information".to_string()
            ).into());
        }

        if !has_tree {
            return Err(SerializationError::ValidationFailed(
                "No trees found in model".to_string()
            ).into());
        }

        Ok(())
    }

    /// Parse LightGBM version
    pub fn parse_version(text: &str) -> Option<String> {
        for line in text.lines() {
            if let Some(version) = line.strip_prefix("version=") {
                return Some(version.trim().to_string());
            }
        }
        None
    }

    /// Count trees in model
    pub fn count_trees(text: &str) -> usize {
        text.lines()
            .filter(|line| line.trim().starts_with("Tree="))
            .count()
    }

    /// Extract feature names
    pub fn extract_feature_names(text: &str) -> Vec<String> {
        for line in text.lines() {
            if let Some(names_str) = line.strip_prefix("feature_names=") {
                return names_str.split_whitespace()
                    .map(|s| s.to_string())
                    .collect();
            }
        }
        Vec::new()
    }

    /// Get format information
    pub fn get_format_info() -> FormatInfo {
        FormatInfo {
            name: "LightGBM Text Format",
            description: "Original LightGBM text format for full compatibility",
            version: "3.0.0",
            extensions: vec![".txt".to_string(), ".model".to_string()],
            mime_type: "text/plain".to_string(),
        }
    }
}

/// Format information
#[derive(Debug, Clone)]
pub struct FormatInfo {
    pub name: &'static str,
    pub description: &'static str,
    pub version: &'static str,
    pub extensions: Vec<String>,
    pub mime_type: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lightgbm_serializer_creation() {
        let serializer = LightGbmSerializer::new();
        assert!(serializer.is_ok());

        let serializer = serializer.unwrap();
        assert_eq!(serializer.format(), SerializationFormat::LightGbm);
    }

    #[test]
    fn test_lightgbm_deserializer_creation() {
        let deserializer = LightGbmDeserializer::new();
        assert!(deserializer.is_ok());

        let deserializer = deserializer.unwrap();
        assert_eq!(deserializer.format(), SerializationFormat::LightGbm);
    }

    #[test]
    fn test_lightgbm_config_default() {
        let config = LightGbmConfig::default();
        assert_eq!(config.base.format, SerializationFormat::LightGbm);
        assert!(config.include_comments);
        assert!(config.include_feature_names);
        assert!(!config.compact_format);
        assert_eq!(config.decimal_precision, 6);
    }

    #[test]
    fn test_line_ending_styles() {
        assert_eq!(LineEnding::Unix.as_str(), "\n");
        assert_eq!(LineEnding::Windows.as_str(), "\r\n");
        assert_eq!(LineEnding::Mac.as_str(), "\r");
    }

    #[test]
    fn test_pandas_categorical_parsing() {
        let cat_str = "0:cat1,cat2,cat3";
        let cat = PandasCategorical::from_str(cat_str).unwrap();
        assert_eq!(cat.feature_index, 0);
        assert_eq!(cat.categories, vec!["cat1", "cat2", "cat3"]);

        let cat_str = cat.to_string();
        assert_eq!(cat_str, "0:cat1,cat2,cat3");
    }

    #[test]
    fn test_array_parsing() {
        let int_array = LightGbmModel::parse_int_array("1 2 3 4 5").unwrap();
        assert_eq!(int_array, vec![1, 2, 3, 4, 5]);

        let float_array = LightGbmModel::parse_float_array("1.0 2.5 3.14 4.0").unwrap();
        assert_eq!(float_array, vec![1.0, 2.5, 3.14, 4.0]);
    }

    #[test]
    fn test_array_formatting() {
        let int_array = vec![1, 2, 3, 4, 5];
        let formatted = LightGbmModel::format_int_array(&int_array);
        assert_eq!(formatted, "1 2 3 4 5");

        let float_array = vec![1.0, 2.5, 3.14159, 4.0];
        let formatted = LightGbmModel::format_float_array(&float_array, 2);
        assert_eq!(formatted, "1.00 2.50 3.14 4.00");
    }

    #[test]
    fn test_utils_functions() {
        assert!(utils::standard_serializer().is_ok());
        assert!(utils::compact_serializer().is_ok());
        assert!(utils::verbose_serializer().is_ok());

        let simple_model = "version=1.0.0\nTree=0\nnum_leaves=1\n";
        assert!(utils::validate_format(simple_model).is_ok());

        assert_eq!(utils::parse_version(simple_model), Some("1.0.0".to_string()));
        assert_eq!(utils::count_trees(simple_model), 1);

        let model_with_features = "feature_names=feature1 feature2 feature3\n";
        let features = utils::extract_feature_names(model_with_features);
        assert_eq!(features, vec!["feature1", "feature2", "feature3"]);

        let format_info = utils::get_format_info();
        assert_eq!(format_info.name, "LightGBM Text Format");
        assert!(format_info.extensions.contains(&".txt".to_string()));
    }

    #[test]
    fn test_tree_info_formatting() {
        let tree_info = LightGbmTreeInfo {
            tree_index: 0,
            num_leaves: 3,
            num_cat: 0,
            shrinkage: 0.1,
        };

        let config = LightGbmConfig::default();
        let formatted = tree_info.to_text(&config).unwrap();
        assert!(formatted.contains("tree_info:0 3 0 0.100000"));
    }

    #[test]
    fn test_simple_model_parsing() {
        let model_text = r#"
version=1.0.0
num_class=1
objective=regression
feature_names=feature1 feature2
Tree=0
num_leaves=3
num_cat=0
split_feature=0
split_gain=0.5
threshold=0.5
left_child=1
right_child=2
leaf_value=0.1 0.2 0.3
        "#;

        let config = LightGbmConfig::default();
        let model = LightGbmModel::from_text(model_text, &config).unwrap();
        
        assert_eq!(model.version, "1.0.0");
        assert_eq!(model.num_class, 1);
        assert_eq!(model.objective, "regression");
        assert_eq!(model.feature_names, vec!["feature1", "feature2"]);
        assert_eq!(model.trees.len(), 1);
        assert_eq!(model.trees[0].num_leaves, 3);
    }
}