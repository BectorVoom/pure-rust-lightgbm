/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

use crate::core::utils::binary_writer::BinaryWriter;
use std::io;

/// An implementation for serializing binary data to an auto-expanding memory buffer
#[derive(Debug, Clone)]
pub struct ByteBuffer {
    buffer: Vec<u8>,
}

impl ByteBuffer {
    /// Create a new ByteBuffer with an empty buffer
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
        }
    }

    /// Create a new ByteBuffer with a pre-allocated capacity
    pub fn with_initial_size(initial_size: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(initial_size),
        }
    }

    /// Reserve capacity for the buffer
    pub fn reserve(&mut self, capacity: usize) {
        self.buffer.reserve(capacity);
    }

    /// Get the current size of the buffer
    pub fn get_size(&self) -> usize {
        self.buffer.len()
    }

    /// Get the byte at the specified index
    /// 
    /// # Panics
    /// Panics if index is out of bounds
    pub fn get_at(&self, index: usize) -> u8 {
        self.buffer[index]
    }

    /// Get a raw pointer to the buffer data
    /// 
    /// # Safety
    /// The caller must ensure that the pointer is not used after the ByteBuffer is dropped
    /// or modified in a way that invalidates the pointer.
    pub fn data(&self) -> *const u8 {
        self.buffer.as_ptr()
    }

    /// Get a mutable raw pointer to the buffer data
    /// 
    /// # Safety
    /// The caller must ensure that the pointer is not used after the ByteBuffer is dropped
    /// or modified in a way that invalidates the pointer.
    pub fn data_mut(&mut self) -> *mut u8 {
        self.buffer.as_mut_ptr()
    }

    /// Get a reference to the internal buffer
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }
}

impl Default for ByteBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl BinaryWriter for ByteBuffer {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        // Mimic the C++ behavior of writing byte by byte
        for &byte in data {
            self.buffer.push(byte);
        }
        Ok(data.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_buffer_new() {
        let buffer = ByteBuffer::new();
        assert_eq!(buffer.get_size(), 0);
    }

    #[test]
    fn test_byte_buffer_with_initial_size() {
        let buffer = ByteBuffer::with_initial_size(100);
        assert_eq!(buffer.get_size(), 0);
        // Note: we can't easily test capacity in Rust without unsafe code
    }

    #[test]
    fn test_write() {
        let mut buffer = ByteBuffer::new();
        let data = b"hello world";
        
        let bytes_written = buffer.write(data).unwrap();
        assert_eq!(bytes_written, data.len());
        assert_eq!(buffer.get_size(), data.len());
    }

    #[test]
    fn test_get_at() {
        let mut buffer = ByteBuffer::new();
        let data = b"test";
        buffer.write(data).unwrap();
        
        assert_eq!(buffer.get_at(0), b't');
        assert_eq!(buffer.get_at(1), b'e');
        assert_eq!(buffer.get_at(2), b's');
        assert_eq!(buffer.get_at(3), b't');
    }

    #[test]
    #[should_panic]
    fn test_get_at_out_of_bounds() {
        let buffer = ByteBuffer::new();
        buffer.get_at(0); // Should panic on empty buffer
    }

    #[test]
    fn test_reserve() {
        let mut buffer = ByteBuffer::new();
        buffer.reserve(100);
        // Reserve doesn't change size, only capacity
        assert_eq!(buffer.get_size(), 0);
    }

    #[test]
    fn test_data_pointer() {
        let mut buffer = ByteBuffer::new();
        let data = b"test data";
        buffer.write(data).unwrap();
        
        let ptr = buffer.data();
        assert!(!ptr.is_null());
        
        // Verify we can read the data through the pointer
        unsafe {
            let slice = std::slice::from_raw_parts(ptr, buffer.get_size());
            assert_eq!(slice, data);
        }
    }

    #[test]
    fn test_data_mut_pointer() {
        let mut buffer = ByteBuffer::new();
        let data = b"test data";
        buffer.write(data).unwrap();
        
        let ptr = buffer.data_mut();
        assert!(!ptr.is_null());
        
        // Verify we can modify data through the pointer
        unsafe {
            *ptr = b'T'; // Change first byte from 't' to 'T'
        }
        
        assert_eq!(buffer.get_at(0), b'T');
    }

    #[test]
    fn test_buffer_reference() {
        let mut buffer = ByteBuffer::new();
        let data = b"hello";
        buffer.write(data).unwrap();
        
        assert_eq!(buffer.buffer(), data);
    }

    #[test]
    fn test_multiple_writes() {
        let mut buffer = ByteBuffer::new();
        
        buffer.write(b"hello").unwrap();
        buffer.write(b" ").unwrap();
        buffer.write(b"world").unwrap();
        
        assert_eq!(buffer.buffer(), b"hello world");
        assert_eq!(buffer.get_size(), 11);
    }

    #[test]
    fn test_byte_by_byte_write() {
        let mut buffer = ByteBuffer::new();
        let data = b"test";
        
        // Write each byte individually to test the byte-by-byte behavior
        for &byte in data {
            buffer.write(&[byte]).unwrap();
        }
        
        assert_eq!(buffer.buffer(), data);
        assert_eq!(buffer.get_size(), 4);
    }

    #[test]
    fn test_default() {
        let buffer = ByteBuffer::default();
        assert_eq!(buffer.get_size(), 0);
    }
}
