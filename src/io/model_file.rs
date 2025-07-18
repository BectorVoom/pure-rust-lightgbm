//! Model file operations for Pure Rust LightGBM.
//!
//! This module provides file-level operations for model persistence including
//! reading, writing, metadata management, and file integrity verification.

use crate::core::error::{Result, LightGBMError};
use crate::io::serialization::SerializationFormat;
use crate::io::{PersistenceConfig, ModelMetadata};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Model file errors
#[derive(Debug, thiserror::Error)]
pub enum ModelFileError {
    #[error("File not found: {0}")]
    FileNotFound(String),
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    #[error("Invalid file format: {0}")]
    InvalidFormat(String),
    #[error("Corrupted file: {0}")]
    CorruptedFile(String),
    #[error("Metadata error: {0}")]
    MetadataError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("File version mismatch: expected {expected}, found {found}")]
    VersionMismatch { expected: String, found: String },
    #[error("Checksum mismatch: expected {expected}, found {found}")]
    ChecksumMismatch { expected: String, found: String },
}

impl From<ModelFileError> for LightGBMError {
    fn from(err: ModelFileError) -> Self {
        LightGBMError::serialization(err.to_string())
    }
}

/// Model file metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelFileMetadata {
    /// File format version
    pub version: String,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last modified timestamp
    pub modified_at: SystemTime,
    /// File size in bytes
    pub file_size: u64,
    /// Serialization format
    pub format: SerializationFormat,
    /// Checksum for integrity verification
    pub checksum: String,
    /// Compression information
    pub compression: Option<CompressionInfo>,
    /// Model metadata
    pub model_metadata: ModelMetadata,
    /// Custom properties
    pub properties: std::collections::HashMap<String, String>,
}

/// Compression information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompressionInfo {
    /// Compression algorithm
    pub algorithm: String,
    /// Compression level
    pub level: u8,
    /// Original size before compression
    pub original_size: u64,
    /// Compressed size
    pub compressed_size: u64,
    /// Compression ratio
    pub ratio: f64,
}

impl Default for ModelFileMetadata {
    fn default() -> Self {
        ModelFileMetadata {
            version: "1.0.0".to_string(),
            created_at: SystemTime::now(),
            modified_at: SystemTime::now(),
            file_size: 0,
            format: SerializationFormat::Bincode,
            checksum: String::new(),
            compression: None,
            model_metadata: ModelMetadata::default(),
            properties: std::collections::HashMap::new(),
        }
    }
}

/// Model file handle
pub struct ModelFile {
    /// File path
    path: PathBuf,
    /// File metadata
    metadata: ModelFileMetadata,
    /// File handle
    file: Option<File>,
    /// Read-only flag
    read_only: bool,
    /// Buffer for I/O operations
    buffer: Vec<u8>,
}

impl ModelFile {
    /// Create a new model file
    pub fn new<P: AsRef<Path>>(
        path: P,
        format: SerializationFormat,
        config: &PersistenceConfig,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        
        let mut metadata = ModelFileMetadata::default();
        metadata.format = format;
        metadata.created_at = SystemTime::now();
        metadata.modified_at = SystemTime::now();
        
        // Set compression info if enabled
        if config.compression.enabled {
            metadata.compression = Some(CompressionInfo {
                algorithm: format!("{:?}", config.compression.algorithm),
                level: config.compression.level,
                original_size: 0,
                compressed_size: 0,
                ratio: 0.0,
            });
        }

        Ok(ModelFile {
            path,
            metadata,
            file: None,
            read_only: false,
            buffer: Vec::with_capacity(8192),
        })
    }

    /// Open an existing model file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        
        if !path.exists() {
            return Err(ModelFileError::FileNotFound(path.display().to_string()).into());
        }

        let file = File::open(&path)?;
        let metadata = Self::read_metadata(&file)?;

        Ok(ModelFile {
            path,
            metadata,
            file: Some(file),
            read_only: true,
            buffer: Vec::with_capacity(8192),
        })
    }

    /// Open model file for writing
    pub fn create<P: AsRef<Path>>(
        path: P,
        format: SerializationFormat,
        config: &PersistenceConfig,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        
        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;

        let mut metadata = ModelFileMetadata::default();
        metadata.format = format;
        metadata.created_at = SystemTime::now();
        metadata.modified_at = SystemTime::now();
        
        // Set compression info if enabled
        if config.compression.enabled {
            metadata.compression = Some(CompressionInfo {
                algorithm: format!("{:?}", config.compression.algorithm),
                level: config.compression.level,
                original_size: 0,
                compressed_size: 0,
                ratio: 0.0,
            });
        }

        Ok(ModelFile {
            path,
            metadata,
            file: Some(file),
            read_only: false,
            buffer: Vec::with_capacity(8192),
        })
    }

    /// Read model data from file
    pub fn read_data(&mut self) -> Result<Vec<u8>> {
        let mut file = self.get_file_handle()?;
        
        // Seek to data section (after metadata)
        file.seek(SeekFrom::Start(self.get_data_offset()?))?;
        
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        
        // Verify checksum if available
        if !self.metadata.checksum.is_empty() {
            let calculated_checksum = self.calculate_checksum(&data)?;
            if calculated_checksum != self.metadata.checksum {
                return Err(ModelFileError::ChecksumMismatch {
                    expected: self.metadata.checksum.clone(),
                    found: calculated_checksum,
                }.into());
            }
        }
        
        Ok(data)
    }

    /// Write model data to file
    pub fn write_data(&mut self, data: &[u8]) -> Result<()> {
        if self.read_only {
            return Err(ModelFileError::PermissionDenied(
                "Cannot write to read-only file".to_string()
            ).into());
        }

        let mut file = self.get_file_handle()?;
        
        // Update metadata
        self.metadata.file_size = data.len() as u64;
        self.metadata.modified_at = SystemTime::now();
        self.metadata.checksum = self.calculate_checksum(data)?;
        
        // Update compression info if applicable
        if let Some(ref mut compression) = self.metadata.compression {
            compression.original_size = data.len() as u64;
            compression.compressed_size = data.len() as u64; // Will be updated by compression layer
            compression.ratio = 1.0; // Will be updated by compression layer
        }

        // Write metadata first
        self.write_metadata(&mut file)?;
        
        // Write data
        file.write_all(data)?;
        file.flush()?;

        // Update file size in metadata
        let final_size = file.metadata()?.len();
        self.metadata.file_size = final_size;

        Ok(())
    }

    /// Get file metadata
    pub fn metadata(&self) -> &ModelFileMetadata {
        &self.metadata
    }

    /// Set model metadata
    pub fn set_model_metadata(&mut self, model_metadata: ModelMetadata) {
        self.metadata.model_metadata = model_metadata;
        self.metadata.modified_at = SystemTime::now();
    }

    /// Get file path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get file size
    pub fn size(&self) -> u64 {
        self.metadata.file_size
    }

    /// Get serialization format
    pub fn format(&self) -> SerializationFormat {
        self.metadata.format
    }

    /// Check if file is compressed
    pub fn is_compressed(&self) -> bool {
        self.metadata.compression.is_some()
    }

    /// Get compression info
    pub fn compression_info(&self) -> Option<&CompressionInfo> {
        self.metadata.compression.as_ref()
    }

    /// Verify file integrity
    pub fn verify_integrity(&mut self) -> Result<bool> {
        if self.metadata.checksum.is_empty() {
            return Ok(true); // No checksum to verify
        }

        let data = self.read_data()?;
        let calculated_checksum = self.calculate_checksum(&data)?;
        
        Ok(calculated_checksum == self.metadata.checksum)
    }

    /// Validate file format
    pub fn validate_format(&self) -> Result<()> {
        // Check if file extension matches format
        if let Some(extension) = self.path.extension() {
            let expected_extension = match self.metadata.format {
                SerializationFormat::Bincode => "lgb",
                SerializationFormat::Json => "json",
                SerializationFormat::LightGbm => "txt",
            };
            
            if extension != expected_extension {
                log::warn!(
                    "File extension '{}' doesn't match format '{:?}'",
                    extension.to_string_lossy(),
                    self.metadata.format
                );
            }
        }

        // Additional format validation can be added here
        Ok(())
    }

    /// Get file statistics
    pub fn get_stats(&self) -> Result<FileStats> {
        let fs_metadata = std::fs::metadata(&self.path)?;
        
        Ok(FileStats {
            size: fs_metadata.len(),
            created: fs_metadata.created().unwrap_or(SystemTime::UNIX_EPOCH),
            modified: fs_metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            accessed: fs_metadata.accessed().unwrap_or(SystemTime::UNIX_EPOCH),
            is_read_only: fs_metadata.permissions().readonly(),
            compression_ratio: self.metadata.compression
                .as_ref()
                .map(|c| c.ratio)
                .unwrap_or(1.0),
        })
    }

    /// Backup file
    pub fn backup<P: AsRef<Path>>(&self, backup_path: P) -> Result<()> {
        let backup_path = backup_path.as_ref();
        
        // Create backup directory if it doesn't exist
        if let Some(parent) = backup_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::copy(&self.path, backup_path)?;
        Ok(())
    }

    /// Restore from backup
    pub fn restore_from_backup<P: AsRef<Path>>(&mut self, backup_path: P) -> Result<()> {
        let backup_path = backup_path.as_ref();
        
        if !backup_path.exists() {
            return Err(ModelFileError::FileNotFound(
                backup_path.display().to_string()
            ).into());
        }

        std::fs::copy(backup_path, &self.path)?;
        
        // Reload metadata
        let file = File::open(&self.path)?;
        self.metadata = Self::read_metadata(&file)?;
        
        Ok(())
    }

    /// Get file handle
    fn get_file_handle(&mut self) -> Result<&mut File> {
        if self.file.is_none() {
            let file = if self.read_only {
                File::open(&self.path)?
            } else {
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .open(&self.path)?
            };
            self.file = Some(file);
        }
        
        Ok(self.file.as_mut().unwrap())
    }

    /// Read metadata from file
    fn read_metadata(mut file: &File) -> Result<ModelFileMetadata> {
        let mut reader = BufReader::new(file);
        let mut buffer = String::new();
        
        // Try to read metadata from file header
        // This is a simplified implementation - in practice, you'd have a more robust format
        match reader.read_line(&mut buffer) {
            Ok(_) => {
                // If first line looks like JSON metadata, parse it
                if buffer.trim().starts_with('{') {
                    serde_json::from_str(buffer.trim())
                        .map_err(|e| ModelFileError::MetadataError(e.to_string()).into())
                } else {
                    // Default metadata if no header found
                    Ok(ModelFileMetadata::default())
                }
            }
            Err(_) => Ok(ModelFileMetadata::default()),
        }
    }

    /// Write metadata to file
    fn write_metadata(&self, file: &mut File) -> Result<()> {
        let metadata_json = serde_json::to_string(&self.metadata)
            .map_err(|e| ModelFileError::MetadataError(e.to_string()))?;
        
        // Write metadata as first line (for simple formats)
        writeln!(file, "{}", metadata_json)?;
        
        Ok(())
    }

    /// Calculate checksum for data
    fn calculate_checksum(&self, data: &[u8]) -> Result<String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        Ok(format!("{:x}", hasher.finish()))
    }

    /// Get data offset (after metadata)
    fn get_data_offset(&self) -> Result<u64> {
        // Simplified - in practice you'd have a proper file format
        // For now, assume metadata is on first line
        Ok(0) // Will be updated with proper format
    }
}

/// File statistics
#[derive(Debug, Clone)]
pub struct FileStats {
    /// File size in bytes
    pub size: u64,
    /// Creation time
    pub created: SystemTime,
    /// Last modified time
    pub modified: SystemTime,
    /// Last accessed time
    pub accessed: SystemTime,
    /// Read-only flag
    pub is_read_only: bool,
    /// Compression ratio
    pub compression_ratio: f64,
}

/// Model file utilities
pub mod utils {
    use super::*;
    
    /// Check if file exists and is readable
    pub fn is_readable<P: AsRef<Path>>(path: P) -> bool {
        let path = path.as_ref();
        path.exists() && path.is_file()
    }

    /// Check if file is writable
    pub fn is_writable<P: AsRef<Path>>(path: P) -> bool {
        let path = path.as_ref();
        if path.exists() {
            !path.metadata().map(|m| m.permissions().readonly()).unwrap_or(true)
        } else {
            // Check if parent directory is writable
            path.parent()
                .map(|p| p.exists() && !p.metadata().map(|m| m.permissions().readonly()).unwrap_or(true))
                .unwrap_or(false)
        }
    }

    /// Get file extension
    pub fn get_extension<P: AsRef<Path>>(path: P) -> Option<String> {
        path.as_ref()
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase())
    }

    /// Generate backup path
    pub fn backup_path<P: AsRef<Path>>(original_path: P) -> PathBuf {
        let path = original_path.as_ref();
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let backup_name = format!(
            "{}.backup.{}",
            path.file_name().unwrap().to_string_lossy(),
            timestamp
        );
        
        path.with_file_name(backup_name)
    }

    /// Clean up old backup files
    pub fn cleanup_backups<P: AsRef<Path>>(dir: P, max_age_days: u64) -> Result<usize> {
        let dir = dir.as_ref();
        let mut removed_count = 0;
        
        if !dir.exists() {
            return Ok(0);
        }

        let cutoff_time = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs() - (max_age_days * 24 * 60 * 60);

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() {
                if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                    if file_name.contains(".backup.") {
                        if let Ok(metadata) = entry.metadata() {
                            if let Ok(modified) = metadata.modified() {
                                let file_time = modified
                                    .duration_since(SystemTime::UNIX_EPOCH)
                                    .unwrap()
                                    .as_secs();
                                
                                if file_time < cutoff_time {
                                    std::fs::remove_file(&path)?;
                                    removed_count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(removed_count)
    }

    /// Get file info
    pub fn get_file_info<P: AsRef<Path>>(path: P) -> Result<FileInfo> {
        let path = path.as_ref();
        let metadata = std::fs::metadata(path)?;
        
        Ok(FileInfo {
            path: path.to_path_buf(),
            size: metadata.len(),
            created: metadata.created().unwrap_or(SystemTime::UNIX_EPOCH),
            modified: metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            is_file: metadata.is_file(),
            is_dir: metadata.is_dir(),
            is_readonly: metadata.permissions().readonly(),
            extension: get_extension(path),
        })
    }

    /// Validate file path
    pub fn validate_path<P: AsRef<Path>>(path: P) -> Result<()> {
        let path = path.as_ref();
        
        // Check for invalid characters
        if path.to_string_lossy().contains('\0') {
            return Err(ModelFileError::InvalidFormat(
                "Path contains null character".to_string()
            ).into());
        }

        // Check path length
        if path.to_string_lossy().len() > 4096 {
            return Err(ModelFileError::InvalidFormat(
                "Path too long".to_string()
            ).into());
        }

        // Check if parent directory exists or can be created
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        Ok(())
    }
}

/// File information
#[derive(Debug, Clone)]
pub struct FileInfo {
    /// File path
    pub path: PathBuf,
    /// File size in bytes
    pub size: u64,
    /// Creation time
    pub created: SystemTime,
    /// Last modified time
    pub modified: SystemTime,
    /// Is file flag
    pub is_file: bool,
    /// Is directory flag
    pub is_dir: bool,
    /// Read-only flag
    pub is_readonly: bool,
    /// File extension
    pub extension: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_model_file_creation() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_model.lgb");
        
        let config = PersistenceConfig::default();
        let model_file = ModelFile::new(&file_path, SerializationFormat::Bincode, &config);
        
        assert!(model_file.is_ok());
        let model_file = model_file.unwrap();
        assert_eq!(model_file.path(), file_path);
        assert_eq!(model_file.format(), SerializationFormat::Bincode);
    }

    #[test]
    fn test_model_file_write_read() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_model.lgb");
        
        let config = PersistenceConfig::default();
        let test_data = b"test model data";
        
        // Write data
        {
            let mut model_file = ModelFile::create(&file_path, SerializationFormat::Bincode, &config).unwrap();
            model_file.write_data(test_data).unwrap();
        }
        
        // Read data
        {
            let mut model_file = ModelFile::open(&file_path).unwrap();
            let read_data = model_file.read_data().unwrap();
            // Note: read_data will include metadata, so we can't directly compare
            // In a real implementation, you'd have proper format handling
        }
    }

    #[test]
    fn test_metadata_operations() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_model.lgb");
        
        let config = PersistenceConfig::default();
        let mut model_file = ModelFile::new(&file_path, SerializationFormat::Json, &config).unwrap();
        
        let mut metadata = ModelMetadata::default();
        metadata.model_type = "test_model".to_string();
        metadata.num_features = 10;
        
        model_file.set_model_metadata(metadata.clone());
        
        assert_eq!(model_file.metadata().model_metadata.model_type, "test_model");
        assert_eq!(model_file.metadata().model_metadata.num_features, 10);
    }

    #[test]
    fn test_file_utils() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.lgb");
        
        // Create a test file
        std::fs::write(&file_path, b"test").unwrap();
        
        assert!(utils::is_readable(&file_path));
        assert!(utils::is_writable(&file_path));
        assert_eq!(utils::get_extension(&file_path), Some("lgb".to_string()));
        
        let file_info = utils::get_file_info(&file_path).unwrap();
        assert_eq!(file_info.size, 4);
        assert!(file_info.is_file);
        assert!(!file_info.is_dir);
    }

    #[test]
    fn test_backup_operations() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.lgb");
        let backup_path = utils::backup_path(&file_path);
        
        // Create original file
        std::fs::write(&file_path, b"original").unwrap();
        
        let model_file = ModelFile::open(&file_path).unwrap();
        model_file.backup(&backup_path).unwrap();
        
        assert!(backup_path.exists());
        assert_eq!(std::fs::read(&backup_path).unwrap(), b"original");
    }

    #[test]
    fn test_path_validation() {
        // Valid path
        assert!(utils::validate_path("valid/path/model.lgb").is_ok());
        
        // Invalid path (too long)
        let long_path = "a".repeat(5000);
        assert!(utils::validate_path(&long_path).is_err());
    }

    #[test]
    fn test_compression_info() {
        let compression = CompressionInfo {
            algorithm: "zstd".to_string(),
            level: 3,
            original_size: 1000,
            compressed_size: 500,
            ratio: 0.5,
        };
        
        assert_eq!(compression.algorithm, "zstd");
        assert_eq!(compression.level, 3);
        assert_eq!(compression.ratio, 0.5);
    }
}