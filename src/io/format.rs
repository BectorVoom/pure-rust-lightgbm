//! Format detection utilities for Pure Rust LightGBM.
//!
//! This module provides automatic detection of serialization formats for model files
//! based on file extensions, content analysis, and metadata inspection.

use crate::core::error::{Result, LightGBMError};
use crate::io::serialization::SerializationFormat;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// Model format detection errors
#[derive(Debug, thiserror::Error)]
pub enum FormatDetectionError {
    #[error("Unable to detect format from path: {0}")]
    PathDetectionFailed(String),
    #[error("Unable to detect format from content: {0}")]
    ContentDetectionFailed(String),
    #[error("Unsupported file format: {0}")]
    UnsupportedFormat(String),
    #[error("File read error: {0}")]
    FileReadError(#[from] std::io::Error),
    #[error("Invalid file content: {0}")]
    InvalidContent(String),
    #[error("Format detection timeout")]
    Timeout,
}

impl From<FormatDetectionError> for LightGBMError {
    fn from(err: FormatDetectionError) -> Self {
        LightGBMError::serialization(err.to_string())
    }
}

/// Supported model formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFormat {
    /// Native Rust bincode format
    Bincode,
    /// JSON format
    Json,
    /// Original LightGBM text format
    LightGbm,
    /// Unknown format
    Unknown,
}

impl From<ModelFormat> for SerializationFormat {
    fn from(format: ModelFormat) -> Self {
        match format {
            ModelFormat::Bincode => SerializationFormat::Bincode,
            ModelFormat::Json => SerializationFormat::Json,
            ModelFormat::LightGbm => SerializationFormat::LightGbm,
            ModelFormat::Unknown => SerializationFormat::Bincode, // Default fallback
        }
    }
}

impl From<SerializationFormat> for ModelFormat {
    fn from(format: SerializationFormat) -> Self {
        match format {
            SerializationFormat::Bincode => ModelFormat::Bincode,
            SerializationFormat::Json => ModelFormat::Json,
            SerializationFormat::LightGbm => ModelFormat::LightGbm,
        }
    }
}

/// Format detection configuration
#[derive(Debug, Clone)]
pub struct FormatDetectionConfig {
    /// Enable content-based detection
    pub enable_content_detection: bool,
    /// Enable extension-based detection
    pub enable_extension_detection: bool,
    /// Enable magic number detection
    pub enable_magic_detection: bool,
    /// Maximum bytes to read for content detection
    pub max_content_bytes: usize,
    /// Enable strict mode (fail on ambiguous detection)
    pub strict_mode: bool,
    /// Timeout for detection in milliseconds
    pub timeout_ms: u64,
}

impl Default for FormatDetectionConfig {
    fn default() -> Self {
        FormatDetectionConfig {
            enable_content_detection: true,
            enable_extension_detection: true,
            enable_magic_detection: true,
            max_content_bytes: 8192, // 8KB
            strict_mode: false,
            timeout_ms: 1000, // 1 second
        }
    }
}

/// Format detector for model files
pub struct FormatDetector {
    config: FormatDetectionConfig,
}

impl FormatDetector {
    /// Create a new format detector with default configuration
    pub fn new() -> Self {
        FormatDetector {
            config: FormatDetectionConfig::default(),
        }
    }

    /// Create a new format detector with custom configuration
    pub fn with_config(config: FormatDetectionConfig) -> Self {
        FormatDetector { config }
    }

    /// Detect format from file path
    pub fn detect_format_from_path<P: AsRef<Path>>(&self, path: P) -> Result<SerializationFormat> {
        let path = path.as_ref();
        
        if !self.config.enable_extension_detection {
            return Err(FormatDetectionError::PathDetectionFailed(
                "Extension detection disabled".to_string()
            ).into());
        }

        // Try extension-based detection first
        if let Some(format) = self.detect_from_extension(path) {
            return Ok(format.into());
        }

        // If extension detection fails, try content detection
        if self.config.enable_content_detection {
            self.detect_format_from_file(path)
        } else {
            Err(FormatDetectionError::PathDetectionFailed(
                format!("Cannot detect format from path: {}", path.display())
            ).into())
        }
    }

    /// Detect format from file content
    pub fn detect_format_from_file<P: AsRef<Path>>(&self, path: P) -> Result<SerializationFormat> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Err(FormatDetectionError::FileReadError(
                std::io::Error::new(std::io::ErrorKind::NotFound, "File not found")
            ).into());
        }

        let mut file = File::open(path)?;
        let mut buffer = vec![0u8; self.config.max_content_bytes];
        let bytes_read = file.read(&mut buffer)?;
        buffer.truncate(bytes_read);

        self.detect_format_from_content(&buffer)
    }

    /// Detect format from raw content
    pub fn detect_format_from_content(&self, content: &[u8]) -> Result<SerializationFormat> {
        if content.is_empty() {
            return Err(FormatDetectionError::InvalidContent("Empty content".to_string()).into());
        }

        let mut detected_formats = Vec::new();

        // Magic number detection
        if self.config.enable_magic_detection {
            if let Some(format) = self.detect_from_magic_numbers(content) {
                detected_formats.push(format);
            }
        }

        // Content pattern detection
        if self.config.enable_content_detection {
            if let Some(format) = self.detect_from_content_patterns(content) {
                detected_formats.push(format);
            }
        }

        // Resolve conflicts
        self.resolve_format_conflicts(detected_formats)
    }

    /// Detect format from file extension
    fn detect_from_extension(&self, path: &Path) -> Option<ModelFormat> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .and_then(|ext| {
                match ext.to_lowercase().as_str() {
                    "lgb" | "bin" | "bincode" => Some(ModelFormat::Bincode),
                    "json" => Some(ModelFormat::Json),
                    "txt" | "model" | "lightgbm" => Some(ModelFormat::LightGbm),
                    _ => None,
                }
            })
    }

    /// Detect format from magic numbers
    fn detect_from_magic_numbers(&self, content: &[u8]) -> Option<ModelFormat> {
        if content.len() < 4 {
            return None;
        }

        // Check for JSON magic numbers
        if content.starts_with(b"{") || content.starts_with(b"[") {
            return Some(ModelFormat::Json);
        }

        // Check for LightGBM text format patterns
        if content.starts_with(b"tree") || 
           content.starts_with(b"feature_names") ||
           content.starts_with(b"objective") ||
           content.starts_with(b"Tree=") {
            return Some(ModelFormat::LightGbm);
        }

        // Check for bincode format (binary data with reasonable structure)
        if self.is_likely_bincode(content) {
            return Some(ModelFormat::Bincode);
        }

        None
    }

    /// Detect format from content patterns
    fn detect_from_content_patterns(&self, content: &[u8]) -> Option<ModelFormat> {
        // Try to parse as UTF-8 text first
        if let Ok(text) = std::str::from_utf8(content) {
            return self.detect_from_text_patterns(text);
        }

        // If not valid UTF-8, likely binary (bincode)
        if self.is_likely_bincode(content) {
            return Some(ModelFormat::Bincode);
        }

        None
    }

    /// Detect format from text patterns
    fn detect_from_text_patterns(&self, text: &str) -> Option<ModelFormat> {
        let text = text.trim();
        
        // JSON detection
        if (text.starts_with('{') && text.ends_with('}')) ||
           (text.starts_with('[') && text.ends_with(']')) {
            // Additional JSON validation
            if self.is_likely_json(text) {
                return Some(ModelFormat::Json);
            }
        }

        // LightGBM text format detection
        if self.is_likely_lightgbm_text(text) {
            return Some(ModelFormat::LightGbm);
        }

        None
    }

    /// Check if content is likely bincode format
    fn is_likely_bincode(&self, content: &[u8]) -> bool {
        if content.len() < 8 {
            return false;
        }

        // Check for reasonable length fields at the start
        let first_u64 = u64::from_le_bytes([
            content[0], content[1], content[2], content[3],
            content[4], content[5], content[6], content[7],
        ]);

        // Reasonable range for serialized data size
        if first_u64 == 0 || first_u64 > (1024 * 1024 * 1024) { // 1GB limit
            return false;
        }

        // Check for binary patterns typical of bincode
        let mut zero_count = 0;
        let mut non_printable_count = 0;
        
        for &byte in content.iter().take(256) {
            if byte == 0 {
                zero_count += 1;
            } else if byte < 32 || byte > 126 {
                non_printable_count += 1;
            }
        }

        // Binary data typically has more non-printable characters
        (non_printable_count as f64 / 256.0) > 0.3 || zero_count > 10
    }

    /// Check if text is likely JSON format
    fn is_likely_json(&self, text: &str) -> bool {
        // Simple JSON structure validation
        let json_indicators = [
            "\"model\"", "\"trees\"", "\"features\"", "\"objective\"",
            "\"version\"", "\"metadata\"", "\"config\"", "\"params\"",
        ];

        let mut indicator_count = 0;
        for indicator in &json_indicators {
            if text.contains(indicator) {
                indicator_count += 1;
            }
        }

        // If we find multiple JSON indicators, likely JSON
        indicator_count >= 2
    }

    /// Check if text is likely LightGBM text format
    fn is_likely_lightgbm_text(&self, text: &str) -> bool {
        let lightgbm_indicators = [
            "tree", "feature_names", "objective", "num_class",
            "Tree=", "split_feature", "split_gain", "threshold",
            "left_child", "right_child", "leaf_value", "leaf_weight",
            "internal_value", "internal_weight", "internal_count",
            "leaf_count", "shrinkage", "pandas_categorical",
        ];

        let mut indicator_count = 0;
        let lines: Vec<&str> = text.lines().take(50).collect(); // Check first 50 lines
        
        for line in &lines {
            for indicator in &lightgbm_indicators {
                if line.contains(indicator) {
                    indicator_count += 1;
                    break;
                }
            }
        }

        // If we find multiple LightGBM indicators, likely LightGBM format
        indicator_count >= 3
    }

    /// Resolve conflicts between detected formats
    fn resolve_format_conflicts(&self, formats: Vec<ModelFormat>) -> Result<SerializationFormat> {
        if formats.is_empty() {
            return Err(FormatDetectionError::ContentDetectionFailed(
                "No format detected".to_string()
            ).into());
        }

        if formats.len() == 1 {
            return Ok(formats[0].into());
        }

        // Multiple formats detected
        if self.config.strict_mode {
            return Err(FormatDetectionError::ContentDetectionFailed(
                format!("Multiple formats detected: {:?}", formats)
            ).into());
        }

        // Priority-based resolution
        let format_priority = [
            ModelFormat::LightGbm,  // Highest priority (most specific)
            ModelFormat::Json,      // Medium priority
            ModelFormat::Bincode,   // Lowest priority (fallback)
        ];

        for priority_format in &format_priority {
            if formats.contains(priority_format) {
                return Ok((*priority_format).into());
            }
        }

        // Fallback to first detected format
        Ok(formats[0].into())
    }

    /// Get detection configuration
    pub fn config(&self) -> &FormatDetectionConfig {
        &self.config
    }

    /// Set detection configuration
    pub fn set_config(&mut self, config: FormatDetectionConfig) {
        self.config = config;
    }

    /// Validate detected format against file content
    pub fn validate_format(&self, content: &[u8], expected_format: SerializationFormat) -> Result<bool> {
        let detected_format = self.detect_format_from_content(content)?;
        Ok(detected_format == expected_format)
    }

    /// Get format confidence score (0.0 to 1.0)
    pub fn get_confidence_score(&self, content: &[u8], format: SerializationFormat) -> Result<f64> {
        let model_format: ModelFormat = format.into();
        
        let mut score = 0.0;
        let mut max_score = 0.0;

        // Magic number score
        max_score += 0.4;
        if let Some(detected) = self.detect_from_magic_numbers(content) {
            if detected == model_format {
                score += 0.4;
            }
        }

        // Content pattern score
        max_score += 0.6;
        if let Some(detected) = self.detect_from_content_patterns(content) {
            if detected == model_format {
                score += 0.6;
            }
        }

        if max_score > 0.0 {
            Ok(score / max_score)
        } else {
            Ok(0.0)
        }
    }

    /// Get detailed format information
    pub fn get_format_info(&self, content: &[u8]) -> Result<FormatInfo> {
        let format = self.detect_format_from_content(content)?;
        let confidence = self.get_confidence_score(content, format)?;
        
        Ok(FormatInfo {
            format,
            confidence,
            content_size: content.len(),
            is_binary: matches!(format, SerializationFormat::Bincode),
            is_compressed: self.detect_compression(content),
            estimated_model_size: self.estimate_model_size(content),
        })
    }

    /// Detect if content is compressed
    fn detect_compression(&self, content: &[u8]) -> bool {
        if content.len() < 4 {
            return false;
        }

        // Check for common compression magic numbers
        // Gzip
        if content.starts_with(&[0x1f, 0x8b]) {
            return true;
        }
        
        // Zstd
        if content.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
            return true;
        }
        
        // LZ4
        if content.starts_with(&[0x04, 0x22, 0x4d, 0x18]) {
            return true;
        }

        false
    }

    /// Estimate model size from content
    fn estimate_model_size(&self, content: &[u8]) -> usize {
        // Simple estimation - in practice this would be more sophisticated
        if self.detect_compression(content) {
            content.len() * 3 // Assume 3:1 compression ratio
        } else {
            content.len()
        }
    }
}

impl Default for FormatDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Format detection information
#[derive(Debug, Clone)]
pub struct FormatInfo {
    /// Detected format
    pub format: SerializationFormat,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Content size in bytes
    pub content_size: usize,
    /// Whether the format is binary
    pub is_binary: bool,
    /// Whether the content is compressed
    pub is_compressed: bool,
    /// Estimated model size in bytes
    pub estimated_model_size: usize,
}

/// Utility functions for format detection
pub mod utils {
    use super::*;

    /// Quick format detection from file path
    pub fn quick_detect_from_path<P: AsRef<Path>>(path: P) -> Option<SerializationFormat> {
        let detector = FormatDetector::new();
        detector.detect_format_from_path(path).ok()
    }

    /// Quick format detection from content
    pub fn quick_detect_from_content(content: &[u8]) -> Option<SerializationFormat> {
        let detector = FormatDetector::new();
        detector.detect_format_from_content(content).ok()
    }

    /// Check if format is supported
    pub fn is_format_supported(format: SerializationFormat) -> bool {
        matches!(format, 
            SerializationFormat::Bincode | 
            SerializationFormat::Json | 
            SerializationFormat::LightGbm
        )
    }

    /// Get format name
    pub fn format_name(format: SerializationFormat) -> &'static str {
        match format {
            SerializationFormat::Bincode => "Bincode",
            SerializationFormat::Json => "JSON",
            SerializationFormat::LightGbm => "LightGBM",
        }
    }

    /// Get format description
    pub fn format_description(format: SerializationFormat) -> &'static str {
        match format {
            SerializationFormat::Bincode => "Native Rust binary format (fast, compact)",
            SerializationFormat::Json => "JavaScript Object Notation (human-readable, portable)",
            SerializationFormat::LightGbm => "Original LightGBM text format (compatible)",
        }
    }

    /// Create a strict format detector
    pub fn strict_detector() -> FormatDetector {
        FormatDetector::with_config(FormatDetectionConfig {
            strict_mode: true,
            ..Default::default()
        })
    }

    /// Create a fast format detector (extension-only)
    pub fn fast_detector() -> FormatDetector {
        FormatDetector::with_config(FormatDetectionConfig {
            enable_content_detection: false,
            enable_magic_detection: false,
            enable_extension_detection: true,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_format_detection_from_extension() {
        let detector = FormatDetector::new();
        
        assert_eq!(
            detector.detect_from_extension(&PathBuf::from("model.lgb")),
            Some(ModelFormat::Bincode)
        );
        assert_eq!(
            detector.detect_from_extension(&PathBuf::from("model.json")),
            Some(ModelFormat::Json)
        );
        assert_eq!(
            detector.detect_from_extension(&PathBuf::from("model.txt")),
            Some(ModelFormat::LightGbm)
        );
        assert_eq!(
            detector.detect_from_extension(&PathBuf::from("model.unknown")),
            None
        );
    }

    #[test]
    fn test_format_detection_from_magic_numbers() {
        let detector = FormatDetector::new();
        
        assert_eq!(
            detector.detect_from_magic_numbers(b"{\"model\": \"test\"}"),
            Some(ModelFormat::Json)
        );
        assert_eq!(
            detector.detect_from_magic_numbers(b"tree"),
            Some(ModelFormat::LightGbm)
        );
        assert_eq!(
            detector.detect_from_magic_numbers(b"feature_names"),
            Some(ModelFormat::LightGbm)
        );
    }

    #[test]
    fn test_json_detection() {
        let detector = FormatDetector::new();
        let json_content = r#"{"model": "lightgbm", "trees": [], "features": []}"#;
        
        assert!(detector.is_likely_json(json_content));
        assert_eq!(
            detector.detect_from_text_patterns(json_content),
            Some(ModelFormat::Json)
        );
    }

    #[test]
    fn test_lightgbm_detection() {
        let detector = FormatDetector::new();
        let lightgbm_content = r#"
tree
feature_names=feature1 feature2 feature3
objective=regression
Tree=0
num_leaves=3
split_feature=0 1
split_gain=0.1 0.2
threshold=0.5 1.0
        "#;
        
        assert!(detector.is_likely_lightgbm_text(lightgbm_content));
        assert_eq!(
            detector.detect_from_text_patterns(lightgbm_content),
            Some(ModelFormat::LightGbm)
        );
    }

    #[test]
    fn test_bincode_detection() {
        let detector = FormatDetector::new();
        let bincode_content = [
            0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // Length: 16
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,  // Binary data
            0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,  // More binary data
        ];
        
        assert!(detector.is_likely_bincode(&bincode_content));
        assert_eq!(
            detector.detect_from_magic_numbers(&bincode_content),
            Some(ModelFormat::Bincode)
        );
    }

    #[test]
    fn test_format_conversion() {
        assert_eq!(
            SerializationFormat::from(ModelFormat::Bincode),
            SerializationFormat::Bincode
        );
        assert_eq!(
            SerializationFormat::from(ModelFormat::Json),
            SerializationFormat::Json
        );
        assert_eq!(
            SerializationFormat::from(ModelFormat::LightGbm),
            SerializationFormat::LightGbm
        );
    }

    #[test]
    fn test_conflict_resolution() {
        let detector = FormatDetector::new();
        
        // Test with no conflicts
        let formats = vec![ModelFormat::Json];
        assert_eq!(
            detector.resolve_format_conflicts(formats).unwrap(),
            SerializationFormat::Json
        );
        
        // Test with multiple formats (should prioritize LightGBM)
        let formats = vec![ModelFormat::Bincode, ModelFormat::LightGbm, ModelFormat::Json];
        assert_eq!(
            detector.resolve_format_conflicts(formats).unwrap(),
            SerializationFormat::LightGbm
        );
    }

    #[test]
    fn test_utils_functions() {
        assert!(utils::is_format_supported(SerializationFormat::Bincode));
        assert!(utils::is_format_supported(SerializationFormat::Json));
        assert!(utils::is_format_supported(SerializationFormat::LightGbm));
        
        assert_eq!(utils::format_name(SerializationFormat::Bincode), "Bincode");
        assert_eq!(utils::format_name(SerializationFormat::Json), "JSON");
        assert_eq!(utils::format_name(SerializationFormat::LightGbm), "LightGBM");
    }

    #[test]
    fn test_detector_configurations() {
        let strict_detector = utils::strict_detector();
        assert!(strict_detector.config().strict_mode);
        
        let fast_detector = utils::fast_detector();
        assert!(!fast_detector.config().enable_content_detection);
        assert!(!fast_detector.config().enable_magic_detection);
        assert!(fast_detector.config().enable_extension_detection);
    }
}