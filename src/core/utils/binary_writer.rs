/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

use std::io;

/// An interface for serializing binary data to a buffer
pub trait BinaryWriter {
    /// Append data to this binary target
    ///
    /// # Arguments
    /// * `data` - Buffer to write from
    ///
    /// # Returns
    /// Number of bytes written, or an error if the write fails
    fn write(&mut self, data: &[u8]) -> io::Result<usize>;

    /// Append data to this binary target aligned on a given byte size boundary
    ///
    /// # Arguments
    /// * `data` - Buffer to write from
    /// * `alignment` - The size of bytes to align to in whole increments (default: 8)
    ///
    /// # Returns
    /// Number of bytes written, or an error if the write fails
    fn aligned_write(&mut self, data: &[u8], alignment: usize) -> io::Result<usize> where Self: Sized {
        let bytes_written = self.write(data)?;
        let data_len = data.len();

        if data_len % alignment != 0 {
            let padding = Self::aligned_size(data_len, alignment) - data_len;
            let padding_bytes = vec![0u8; padding];
            let padding_written = self.write(&padding_bytes)?;
            Ok(bytes_written + padding_written)
        } else {
            Ok(bytes_written)
        }
    }

    /// The aligned size of a buffer length
    ///
    /// # Arguments
    /// * `bytes` - The number of bytes in a buffer
    /// * `alignment` - The size of bytes to align to in whole increments (default: 8)
    ///
    /// # Returns
    /// Number of aligned bytes
    fn aligned_size(bytes: usize, alignment: usize) -> usize where Self: Sized {
        if bytes % alignment == 0 {
            bytes
        } else {
            bytes / alignment * alignment + alignment
        }
    }
}

/// A simple implementation of BinaryWriter that writes to a Vec<u8>
#[derive(Debug, Clone)]
pub struct VecBinaryWriter {
    buffer: Vec<u8>,
}

impl VecBinaryWriter {
    /// Create a new VecBinaryWriter with an empty buffer
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    /// Create a new VecBinaryWriter with a pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
        }
    }

    /// Get a reference to the internal buffer
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    /// Get the length of the buffer
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Take ownership of the internal buffer
    pub fn into_buffer(self) -> Vec<u8> {
        self.buffer
    }
}

impl Default for VecBinaryWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BinaryWriter for VecBinaryWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.buffer.extend_from_slice(data);
        Ok(data.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_binary_writer_basic() {
        let mut writer = VecBinaryWriter::new();
        let data = b"hello world";

        let bytes_written = writer.write(data).unwrap();
        assert_eq!(bytes_written, data.len());
        assert_eq!(writer.buffer(), data);
        assert_eq!(writer.len(), data.len());
        assert!(!writer.is_empty());
    }

    #[test]
    fn test_aligned_size() {
        assert_eq!(VecBinaryWriter::aligned_size(0, 8), 0);
        assert_eq!(VecBinaryWriter::aligned_size(1, 8), 8);
        assert_eq!(VecBinaryWriter::aligned_size(7, 8), 8);
        assert_eq!(VecBinaryWriter::aligned_size(8, 8), 8);
        assert_eq!(VecBinaryWriter::aligned_size(9, 8), 16);
        assert_eq!(VecBinaryWriter::aligned_size(15, 8), 16);
        assert_eq!(VecBinaryWriter::aligned_size(16, 8), 16);

        // Test different alignment values
        assert_eq!(VecBinaryWriter::aligned_size(5, 4), 8);
        assert_eq!(VecBinaryWriter::aligned_size(10, 16), 16);
    }

    #[test]
    fn test_aligned_write() {
        let mut writer = VecBinaryWriter::new();
        let data = b"hello"; // 5 bytes

        let bytes_written = writer.aligned_write(data, 8).unwrap();
        assert_eq!(bytes_written, 8); // 5 bytes data + 3 bytes padding
        assert_eq!(writer.len(), 8);

        let buffer = writer.buffer();
        assert_eq!(&buffer[0..5], b"hello");
        assert_eq!(&buffer[5..8], &[0, 0, 0]); // padding
    }

    #[test]
    fn test_aligned_write_already_aligned() {
        let mut writer = VecBinaryWriter::new();
        let data = b"hellowor"; // 8 bytes, already aligned to 8

        let bytes_written = writer.aligned_write(data, 8).unwrap();
        assert_eq!(bytes_written, 8); // No padding needed
        assert_eq!(writer.len(), 8);
        assert_eq!(writer.buffer(), data);
    }

    #[test]
    fn test_multiple_writes() {
        let mut writer = VecBinaryWriter::new();

        writer.write(b"hello").unwrap();
        writer.write(b" ").unwrap();
        writer.write(b"world").unwrap();

        assert_eq!(writer.buffer(), b"hello world");
        assert_eq!(writer.len(), 11);
    }

    #[test]
    fn test_clear() {
        let mut writer = VecBinaryWriter::new();
        writer.write(b"hello").unwrap();
        assert!(!writer.is_empty());

        writer.clear();
        assert!(writer.is_empty());
        assert_eq!(writer.len(), 0);
    }

    #[test]
    fn test_with_capacity() {
        let writer = VecBinaryWriter::with_capacity(100);
        assert!(writer.is_empty());
        assert_eq!(writer.len(), 0);
    }

    #[test]
    fn test_into_buffer() {
        let mut writer = VecBinaryWriter::new();
        writer.write(b"test").unwrap();

        let buffer = writer.into_buffer();
        assert_eq!(buffer, b"test");
    }
}
