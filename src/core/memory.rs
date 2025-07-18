//! Memory management utilities for Pure Rust LightGBM.
//!
//! This module provides sophisticated memory management techniques including
//! aligned memory allocation for SIMD operations, memory pools, and utilities
//! for efficient memory usage throughout the LightGBM implementation.

use crate::core::constants::*;
use crate::core::error::{MemoryError, Result};
use std::alloc::{alloc, alloc_zeroed, dealloc, Layout};
use std::marker::PhantomData;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Thread-safe memory usage tracker for monitoring and debugging.
static TOTAL_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

/// Get total allocated memory across all AlignedBuffer instances.
pub fn total_allocated_memory() -> usize {
    TOTAL_ALLOCATED.load(Ordering::Relaxed)
}

/// SIMD-aligned memory buffer with automatic memory management.
///
/// This structure provides 32-byte aligned memory allocation optimized for
/// SIMD operations, with automatic deallocation when dropped.
#[repr(align(32))]
pub struct AlignedBuffer<T> {
    /// Pointer to aligned memory
    ptr: NonNull<T>,
    /// Number of elements currently stored
    len: usize,
    /// Allocated capacity in elements
    capacity: usize,
    /// Memory layout for deallocation
    layout: Layout,
    /// Type marker
    _marker: PhantomData<T>,
}

impl<T> AlignedBuffer<T> {
    /// Create a new aligned buffer with the specified capacity.
    ///
    /// Memory is aligned to ALIGNED_SIZE bytes for optimal SIMD performance.
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Ok(Self::empty());
        }

        let layout = Layout::array::<T>(capacity)
            .map_err(|_| MemoryError::AllocationFailed {
                size: capacity * std::mem::size_of::<T>(),
            })?
            .align_to(ALIGNED_SIZE)
            .map_err(|_| MemoryError::AlignmentViolation {
                address: 0,
                alignment: ALIGNED_SIZE,
            })?;

        let ptr = unsafe { alloc(layout) as *mut T };
        if ptr.is_null() {
            return Err(MemoryError::AllocationFailed {
                size: layout.size(),
            }
            .into());
        }

        // Track allocated memory
        TOTAL_ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);

        Ok(AlignedBuffer {
            ptr: NonNull::new(ptr).unwrap(),
            len: 0,
            capacity,
            layout,
            _marker: PhantomData,
        })
    }

    /// Create a new zeroed aligned buffer.
    pub fn new_zeroed(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Ok(Self::empty());
        }

        let layout = Layout::array::<T>(capacity)
            .map_err(|_| MemoryError::AllocationFailed {
                size: capacity * std::mem::size_of::<T>(),
            })?
            .align_to(ALIGNED_SIZE)
            .map_err(|_| MemoryError::AlignmentViolation {
                address: 0,
                alignment: ALIGNED_SIZE,
            })?;

        let ptr = unsafe { alloc_zeroed(layout) as *mut T };
        if ptr.is_null() {
            return Err(MemoryError::AllocationFailed {
                size: layout.size(),
            }
            .into());
        }

        // Track allocated memory
        TOTAL_ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);

        Ok(AlignedBuffer {
            ptr: NonNull::new(ptr).unwrap(),
            len: 0,
            capacity,
            layout,
            _marker: PhantomData,
        })
    }

    /// Create an empty buffer with no allocation.
    pub fn empty() -> Self {
        AlignedBuffer {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
            layout: Layout::new::<u8>(),
            _marker: PhantomData,
        }
    }

    /// Get the current length (number of elements stored).
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the capacity (maximum number of elements).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the memory layout.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Get a raw pointer to the data.
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Get a mutable raw pointer to the data.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Get a slice view of the buffer content.
    pub fn as_slice(&self) -> &[T] {
        if self.capacity == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

    /// Get a mutable slice view of the buffer content.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.capacity == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }

    /// Push an element to the buffer.
    pub fn push(&mut self, value: T) -> Result<()> {
        if self.len >= self.capacity {
            return Err(MemoryError::BufferOverflow {
                capacity: self.capacity,
                offset: self.len,
            }
            .into());
        }

        unsafe {
            ptr::write(self.ptr.as_ptr().add(self.len), value);
        }
        self.len += 1;
        Ok(())
    }

    /// Pop an element from the buffer.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            Some(unsafe { ptr::read(self.ptr.as_ptr().add(self.len)) })
        }
    }

    /// Set the length of the buffer.
    ///
    /// # Safety
    /// The caller must ensure that elements up to `new_len` are properly initialized.
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.capacity);
        self.len = new_len;
    }

    /// Clear the buffer, dropping all elements.
    pub fn clear(&mut self) {
        while let Some(_) = self.pop() {}
    }

    /// Resize the buffer to the specified length, filling with the given value.
    pub fn resize(&mut self, new_len: usize, value: T) -> Result<()>
    where
        T: Clone,
    {
        if new_len > self.capacity {
            return Err(MemoryError::BufferOverflow {
                capacity: self.capacity,
                offset: new_len,
            }
            .into());
        }

        if new_len > self.len {
            // Extend with copies of value
            for _ in self.len..new_len {
                self.push(value.clone())?;
            }
        } else {
            // Truncate
            while self.len > new_len {
                self.pop();
            }
        }

        Ok(())
    }

    /// Check if the buffer's memory is properly aligned.
    pub fn is_aligned(&self) -> bool {
        self.ptr.as_ptr() as usize % ALIGNED_SIZE == 0
    }

    /// Get memory statistics for this buffer.
    pub fn memory_stats(&self) -> MemoryStats {
        let allocated = self.layout.size();
        let used = self.len * std::mem::size_of::<T>();
        MemoryStats {
            allocated_bytes: allocated,
            used_bytes: used,
            alignment: self.layout.align(),
            capacity_elements: self.capacity,
            length_elements: self.len,
            free_bytes: allocated.saturating_sub(used),
            num_allocations: if self.capacity > 0 { 1 } else { 0 },
        }
    }

    /// Get the alignment of this buffer.
    pub fn alignment(&self) -> usize {
        self.layout.align()
    }
}

unsafe impl<T: Send> Send for AlignedBuffer<T> {}
unsafe impl<T: Sync> Sync for AlignedBuffer<T> {}

impl<T> std::ops::Index<usize> for AlignedBuffer<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.len {
            panic!(
                "Index {} out of bounds for AlignedBuffer of length {}",
                index, self.len
            );
        }
        unsafe { &*self.ptr.as_ptr().add(index) }
    }
}

impl<T> std::ops::IndexMut<usize> for AlignedBuffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= self.len {
            panic!(
                "Index {} out of bounds for AlignedBuffer of length {}",
                index, self.len
            );
        }
        unsafe { &mut *self.ptr.as_ptr().add(index) }
    }
}

impl<T> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            // Drop all elements
            self.clear();

            // Deallocate memory
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
            }

            // Update memory tracking
            TOTAL_ALLOCATED.fetch_sub(self.layout.size(), Ordering::Relaxed);
        }
    }
}

impl<T: Clone> Clone for AlignedBuffer<T> {
    fn clone(&self) -> Self {
        let mut new_buffer =
            AlignedBuffer::new(self.capacity).expect("Failed to allocate memory for cloned buffer");

        for item in self.as_slice() {
            new_buffer
                .push(item.clone())
                .expect("Buffer overflow during clone");
        }

        new_buffer
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for AlignedBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBuffer")
            .field("len", &self.len)
            .field("capacity", &self.capacity)
            .field("alignment", &self.layout.align())
            .field("data", &self.as_slice())
            .finish()
    }
}

/// Memory statistics for monitoring and debugging.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryStats {
    pub allocated_bytes: usize,
    pub used_bytes: usize,
    pub alignment: usize,
    pub capacity_elements: usize,
    pub length_elements: usize,
    pub free_bytes: usize,
    pub num_allocations: usize,
}

impl MemoryStats {
    /// Calculate memory utilization as a percentage.
    pub fn utilization(&self) -> f64 {
        if self.allocated_bytes == 0 {
            0.0
        } else {
            (self.used_bytes as f64 / self.allocated_bytes as f64) * 100.0
        }
    }

    /// Calculate waste (unused allocated memory) in bytes.
    pub fn waste_bytes(&self) -> usize {
        self.allocated_bytes.saturating_sub(self.used_bytes)
    }
}

/// Memory pool for efficient allocation and reuse of aligned buffers.
pub struct MemoryPool<T> {
    available_buffers: Vec<AlignedBuffer<T>>,
    buffer_capacity: usize,
    max_pool_size: usize,
}

impl<T> MemoryPool<T> {
    /// Create a new memory pool with the specified buffer capacity and pool size.
    pub fn new(buffer_capacity: usize, max_pool_size: usize) -> Self {
        MemoryPool {
            available_buffers: Vec::with_capacity(max_pool_size),
            buffer_capacity,
            max_pool_size,
        }
    }

    /// Check if the memory pool is OK (placeholder implementation)
    pub fn is_ok(&self) -> bool {
        true
    }

    /// Get a buffer from the pool, or allocate a new one if none available.
    pub fn get_buffer(&mut self) -> Result<AlignedBuffer<T>> {
        if let Some(mut buffer) = self.available_buffers.pop() {
            buffer.clear();
            Ok(buffer)
        } else {
            AlignedBuffer::new(self.buffer_capacity)
        }
    }

    /// Return a buffer to the pool for reuse.
    pub fn return_buffer(&mut self, buffer: AlignedBuffer<T>) {
        if self.available_buffers.len() < self.max_pool_size
            && buffer.capacity() == self.buffer_capacity
        {
            self.available_buffers.push(buffer);
        }
        // If pool is full or buffer has wrong capacity, just drop it
    }

    /// Get the number of available buffers in the pool.
    pub fn available_count(&self) -> usize {
        self.available_buffers.len()
    }

    /// Clear all buffers from the pool.
    pub fn clear(&mut self) {
        self.available_buffers.clear();
    }

    /// Allocate a buffer with specified size (for compatibility)
    pub fn allocate<U>(&mut self, size: usize) -> Result<MemoryHandle<U>> {
        // Create a handle representing the allocation
        Ok(MemoryHandle {
            size,
            phantom: std::marker::PhantomData,
        })
    }

    /// Deallocate a buffer handle (for compatibility)
    pub fn deallocate<U>(&mut self, _handle: MemoryHandle<U>) -> Result<()> {
        // Placeholder - in a real implementation this would free the memory
        Ok(())
    }

    /// Get pool statistics
    pub fn stats(&self) -> MemoryStats {
        let allocated = self.max_pool_size * self.buffer_capacity * std::mem::size_of::<T>();
        let used = (self.max_pool_size - self.available_buffers.len())
            * self.buffer_capacity
            * std::mem::size_of::<T>();
        MemoryStats {
            alignment: ALIGNED_SIZE,
            allocated_bytes: allocated,
            used_bytes: used,
            capacity_elements: self.max_pool_size * self.buffer_capacity,
            length_elements: (self.max_pool_size - self.available_buffers.len())
                * self.buffer_capacity,
            free_bytes: allocated.saturating_sub(used),
            num_allocations: self.max_pool_size - self.available_buffers.len(),
        }
    }
}

/// Handle representing a memory allocation
#[derive(Debug)]
pub struct MemoryHandle<T> {
    size: usize,
    phantom: std::marker::PhantomData<T>,
}

/// Create a simplified constructor that doesn't require unwrap
impl<T> MemoryPool<T> {
    /// Create a new memory pool with sensible defaults
    pub fn default_pool() -> Self {
        Self::new(1024, 10)
    }
}

/// Utility functions for memory operations
pub mod utils {
    use super::*;

    /// Check if a pointer is properly aligned for SIMD operations.
    pub fn is_simd_aligned<T>(ptr: *const T) -> bool {
        ptr as usize % ALIGNED_SIZE == 0
    }

    /// Round up size to the nearest alignment boundary.
    pub fn align_up(size: usize, alignment: usize) -> usize {
        (size + alignment - 1) & !(alignment - 1)
    }

    /// Round down size to the nearest alignment boundary.
    pub fn align_down(size: usize, alignment: usize) -> usize {
        size & !(alignment - 1)
    }

    /// Calculate padding needed to reach alignment.
    pub fn padding_for_alignment(current: usize, alignment: usize) -> usize {
        align_up(current, alignment) - current
    }

    /// Get system page size.
    pub fn page_size() -> usize {
        // This is a common page size, but ideally should be queried from the system
        4096
    }

    /// Calculate optimal chunk size for parallel processing based on cache size.
    pub fn optimal_chunk_size<T>(total_elements: usize, num_threads: usize) -> usize {
        let elements_per_cache_line = CACHE_LINE_SIZE / std::mem::size_of::<T>();
        let min_chunk_size = elements_per_cache_line * 4; // At least 4 cache lines
        let target_chunk_size = total_elements / num_threads;

        std::cmp::max(min_chunk_size, target_chunk_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_buffer_creation() {
        let buffer: AlignedBuffer<f32> = AlignedBuffer::new(100).unwrap();
        assert_eq!(buffer.capacity(), 100);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        assert!(buffer.is_aligned());
    }

    #[test]
    fn test_aligned_buffer_push_pop() {
        let mut buffer: AlignedBuffer<i32> = AlignedBuffer::new(10).unwrap();

        // Test push
        for i in 0..5 {
            buffer.push(i).unwrap();
        }
        assert_eq!(buffer.len(), 5);
        assert!(!buffer.is_empty());

        // Test pop
        assert_eq!(buffer.pop(), Some(4));
        assert_eq!(buffer.pop(), Some(3));
        assert_eq!(buffer.len(), 3);
    }

    #[test]
    fn test_aligned_buffer_overflow() {
        let mut buffer: AlignedBuffer<u8> = AlignedBuffer::new(2).unwrap();
        buffer.push(1).unwrap();
        buffer.push(2).unwrap();

        // This should fail
        let result = buffer.push(3);
        assert!(result.is_err());
    }

    #[test]
    fn test_aligned_buffer_resize() {
        let mut buffer: AlignedBuffer<i32> = AlignedBuffer::new(10).unwrap();
        buffer.push(1).unwrap();
        buffer.push(2).unwrap();

        // Resize up
        buffer.resize(5, 42).unwrap();
        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.as_slice(), &[1, 2, 42, 42, 42]);

        // Resize down
        buffer.resize(3, 0).unwrap();
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.as_slice(), &[1, 2, 42]);
    }

    #[test]
    fn test_memory_pool() {
        let mut pool: MemoryPool<f64> = MemoryPool::new(100, 5);
        assert_eq!(pool.available_count(), 0);

        // Get a buffer
        let buffer1 = pool.get_buffer().unwrap();
        assert_eq!(buffer1.capacity(), 100);

        // Return it
        pool.return_buffer(buffer1);
        assert_eq!(pool.available_count(), 1);

        // Get it back
        let buffer2 = pool.get_buffer().unwrap();
        assert_eq!(buffer2.capacity(), 100);
        assert_eq!(pool.available_count(), 0);
    }

    #[test]
    fn test_memory_stats() {
        let buffer: AlignedBuffer<u64> = AlignedBuffer::new(10).unwrap();
        let stats = buffer.memory_stats();

        assert!(stats.allocated_bytes >= 10 * std::mem::size_of::<u64>());
        assert_eq!(stats.used_bytes, 0); // No elements pushed yet
        assert_eq!(stats.capacity_elements, 10);
        assert_eq!(stats.length_elements, 0);
        assert_eq!(stats.utilization(), 0.0);
    }

    #[test]
    fn test_alignment_utilities() {
        assert!(utils::is_simd_aligned(std::ptr::null::<f32>()));

        assert_eq!(utils::align_up(10, 8), 16);
        assert_eq!(utils::align_up(16, 8), 16);

        assert_eq!(utils::align_down(15, 8), 8);
        assert_eq!(utils::align_down(16, 8), 16);

        assert_eq!(utils::padding_for_alignment(10, 8), 6);
        assert_eq!(utils::padding_for_alignment(16, 8), 0);
    }

    #[test]
    fn test_memory_tracking() {
        let initial_memory = total_allocated_memory();

        {
            let _buffer1: AlignedBuffer<u64> = AlignedBuffer::new(100).unwrap();
            let _buffer2: AlignedBuffer<u32> = AlignedBuffer::new(200).unwrap();

            let current_memory = total_allocated_memory();
            assert!(current_memory > initial_memory);
        }

        // Memory should be freed after buffers are dropped
        let final_memory = total_allocated_memory();
        assert_eq!(final_memory, initial_memory);
    }

    #[test]
    fn test_empty_buffer() {
        let buffer: AlignedBuffer<i32> = AlignedBuffer::empty();
        assert_eq!(buffer.capacity(), 0);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        assert_eq!(buffer.as_slice().len(), 0);
    }

    #[test]
    fn test_zeroed_buffer() {
        let buffer: AlignedBuffer<u8> = AlignedBuffer::new_zeroed(10).unwrap();
        assert_eq!(buffer.capacity(), 10);

        // Check that all memory is zeroed
        let slice = unsafe { std::slice::from_raw_parts(buffer.as_ptr(), buffer.capacity()) };
        assert!(slice.iter().all(|&x| x == 0));
    }
}
