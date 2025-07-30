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
use std::sync::{Arc, Mutex};

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
    /// Total bytes allocated in memory
    pub allocated_bytes: usize,
    /// Bytes currently in use
    pub used_bytes: usize,
    /// Memory alignment in bytes
    pub alignment: usize,
    /// Total capacity in number of elements
    pub capacity_elements: usize,
    /// Current length in number of elements
    pub length_elements: usize,
    /// Available free bytes
    pub free_bytes: usize,
    /// Total number of allocations made
    pub num_allocations: usize,
}

/// Memory statistics for individual memory handles.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryHandleStats {
    /// Size of the allocation in bytes
    pub size_bytes: usize,
    /// Number of elements in the allocation
    pub size_elements: usize,
    /// Size of each element in bytes
    pub element_size: usize,
    /// Timestamp when the allocation was created
    pub allocated_at: std::time::SystemTime,
    /// Age of the allocation in milliseconds
    pub age_ms: u64,
}

/// Type-erased trait for tracking memory handles regardless of their element type.
pub trait AnyMemoryHandle: Send + Sync + std::fmt::Debug {
    /// Get the size of the allocation in bytes
    fn size_bytes(&self) -> usize;

    /// Get the element size in bytes
    fn element_size(&self) -> usize;

    /// Get the time when this handle was allocated
    fn allocated_at(&self) -> std::time::SystemTime;

    /// Get the age of this allocation in milliseconds
    fn age_ms(&self) -> Result<u64>;

    /// Check if this handle represents a valid allocation
    fn is_valid(&self) -> bool;

    /// Get memory usage statistics for this handle
    fn usage_stats(&self) -> MemoryHandleStats;
}

/// Thread-safe collection of active memory handles for tracking purposes.
#[derive(Debug, Default)]
pub struct ActiveHandleTracker {
    handles: Arc<Mutex<Vec<Box<dyn AnyMemoryHandle>>>>,
}

impl ActiveHandleTracker {
    /// Create a new active handle tracker
    pub fn new() -> Self {
        Self {
            handles: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Add a handle to the tracker
    pub fn add_handle(&self, handle: Box<dyn AnyMemoryHandle>) {
        if let Ok(mut handles) = self.handles.lock() {
            handles.push(handle);
        }
    }

    /// Remove and return a handle that matches the given criteria
    pub fn remove_handle(
        &self,
        size_bytes: usize,
        allocated_at: std::time::SystemTime,
    ) -> Option<Box<dyn AnyMemoryHandle>> {
        if let Ok(mut handles) = self.handles.lock() {
            if let Some(pos) = handles
                .iter()
                .position(|h| h.size_bytes() == size_bytes && h.allocated_at() == allocated_at)
            {
                Some(handles.remove(pos))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Get the number of active handles
    pub fn active_count(&self) -> usize {
        self.handles.lock().map(|h| h.len()).unwrap_or(0)
    }

    /// Get total memory usage of all active handles
    pub fn total_memory(&self) -> usize {
        if let Ok(handles) = self.handles.lock() {
            handles.iter().map(|h| h.size_bytes()).sum()
        } else {
            0
        }
    }

    /// Get statistics for all active handles
    pub fn all_handle_stats(&self) -> Vec<MemoryHandleStats> {
        if let Ok(handles) = self.handles.lock() {
            handles.iter().map(|h| h.usage_stats()).collect()
        } else {
            Vec::new()
        }
    }

    /// Clear all handles from the tracker
    pub fn clear(&self) {
        if let Ok(mut handles) = self.handles.lock() {
            handles.clear();
        }
    }
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
#[derive(Debug)]
pub struct MemoryPool<T> {
    available_buffers: Vec<AlignedBuffer<T>>,
    buffer_capacity: usize,
    max_pool_size: usize,
    total_allocated_bytes: usize,
    max_memory_bytes: Option<usize>,
    active_handles: ActiveHandleTracker,
}

impl<T> MemoryPool<T> {
    /// Create a new memory pool with the specified buffer capacity and pool size.
    pub fn new(buffer_capacity: usize, max_pool_size: usize) -> Self {
        MemoryPool {
            available_buffers: Vec::with_capacity(max_pool_size),
            buffer_capacity,
            max_pool_size,
            total_allocated_bytes: 0,
            max_memory_bytes: None,
            active_handles: ActiveHandleTracker::new(),
        }
    }

    /// Create a new memory pool with memory size constraints.
    pub fn with_memory_limit(
        buffer_capacity: usize,
        max_pool_size: usize,
        max_memory_bytes: usize,
    ) -> Self {
        MemoryPool {
            available_buffers: Vec::with_capacity(max_pool_size),
            buffer_capacity,
            max_pool_size,
            total_allocated_bytes: 0,
            max_memory_bytes: Some(max_memory_bytes),
            active_handles: ActiveHandleTracker::new(),
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
    pub fn allocate<U: Send + Sync + std::fmt::Debug + 'static>(
        &mut self,
        size: usize,
    ) -> Result<MemoryHandle<U>> {
        // Check memory constraints
        if let Some(max_memory) = self.max_memory_bytes {
            if self.total_allocated_bytes + size > max_memory {
                return Err(MemoryError::AllocationFailed { size }.into());
            }
        }

        // Create a handle representing the allocation
        let handle = MemoryHandle::new(size);
        self.total_allocated_bytes += size;

        // Track the handle using the type-erased tracking system
        let handle_clone = MemoryHandle::<U> {
            size: handle.size,
            allocated_at: handle.allocated_at,
            phantom: PhantomData,
        };
        self.active_handles.add_handle(Box::new(handle_clone));

        Ok(handle)
    }

    /// Deallocate a buffer handle (for compatibility)
    pub fn deallocate<U: Send + Sync + std::fmt::Debug + 'static>(
        &mut self,
        handle: MemoryHandle<U>,
    ) -> Result<()> {
        // Remove the handle from active tracking
        let size = handle.size_bytes();
        let allocated_at = handle.allocated_at();

        if self
            .active_handles
            .remove_handle(size, allocated_at)
            .is_some()
        {
            // Successfully removed from tracking, update memory accounting
            self.total_allocated_bytes = self.total_allocated_bytes.saturating_sub(size);
        }

        Ok(())
    }

    /// Get current memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.total_allocated_bytes
    }

    /// Get maximum memory limit in bytes (if set)
    pub fn memory_limit(&self) -> Option<usize> {
        self.max_memory_bytes
    }

    /// Get remaining memory capacity in bytes
    pub fn remaining_memory(&self) -> Option<usize> {
        self.max_memory_bytes
            .map(|max| max.saturating_sub(self.total_allocated_bytes))
    }

    /// Check if memory usage is at or near capacity
    pub fn is_memory_pressure(&self, threshold_percent: f64) -> bool {
        if let Some(max_memory) = self.max_memory_bytes {
            let usage_percent = (self.total_allocated_bytes as f64 / max_memory as f64) * 100.0;
            usage_percent >= threshold_percent
        } else {
            false
        }
    }

    /// Get pool statistics
    pub fn stats(&self) -> MemoryStats {
        let pool_allocated = self.max_pool_size * self.buffer_capacity * std::mem::size_of::<T>();
        let pool_used = (self.max_pool_size - self.available_buffers.len())
            * self.buffer_capacity
            * std::mem::size_of::<T>();

        // Include tracked allocations from handles
        let total_allocated = pool_allocated + self.total_allocated_bytes;
        let total_used = pool_used + self.total_allocated_bytes;

        MemoryStats {
            alignment: ALIGNED_SIZE,
            allocated_bytes: total_allocated,
            used_bytes: total_used,
            capacity_elements: self.max_pool_size * self.buffer_capacity,
            length_elements: (self.max_pool_size - self.available_buffers.len())
                * self.buffer_capacity,
            free_bytes: total_allocated.saturating_sub(total_used),
            num_allocations: self.max_pool_size - self.available_buffers.len(),
        }
    }

    /// Get detailed memory usage report
    pub fn memory_report(&self) -> String {
        let stats = self.stats();
        let mut report = format!(
            "Memory Pool Report:\n\
             - Pool Capacity: {} buffers\n\
             - Available Buffers: {}\n\
             - Buffer Size: {} elements\n\
             - Total Allocated: {} bytes\n\
             - Total Used: {} bytes\n\
             - Utilization: {:.2}%\n",
            self.max_pool_size,
            self.available_buffers.len(),
            self.buffer_capacity,
            stats.allocated_bytes,
            stats.used_bytes,
            stats.utilization()
        );

        if let Some(limit) = self.max_memory_bytes {
            report.push_str(&format!(
                " - Memory Limit: {} bytes\n\
                 - Memory Usage: {} bytes\n\
                 - Remaining: {} bytes\n",
                limit,
                self.total_allocated_bytes,
                limit.saturating_sub(self.total_allocated_bytes)
            ));
        }

        report
    }

    /// Get the number of active memory handles
    pub fn active_handle_count(&self) -> usize {
        self.active_handles.active_count()
    }

    /// Get total memory usage from active handles
    pub fn active_handles_memory(&self) -> usize {
        self.active_handles.total_memory()
    }

    /// Get statistics for all active handles
    pub fn active_handle_stats(&self) -> Vec<MemoryHandleStats> {
        self.active_handles.all_handle_stats()
    }

    /// Clear all active handle tracking (for cleanup/reset scenarios)
    pub fn clear_active_handles(&self) {
        self.active_handles.clear();
    }

    /// Get a detailed report including active handle information
    pub fn detailed_memory_report(&self) -> String {
        let stats = self.stats();
        let active_count = self.active_handle_count();
        let active_memory = self.active_handles_memory();

        let mut report = format!(
            "Detailed Memory Pool Report:\n\
             - Pool Capacity: {} buffers\n\
             - Available Buffers: {}\n\
             - Buffer Size: {} elements\n\
             - Total Allocated: {} bytes\n\
             - Total Used: {} bytes\n\
             - Utilization: {:.2}%\n\
             - Active Handles: {}\n\
             - Active Handle Memory: {} bytes\n",
            self.max_pool_size,
            self.available_buffers.len(),
            self.buffer_capacity,
            stats.allocated_bytes,
            stats.used_bytes,
            stats.utilization(),
            active_count,
            active_memory
        );

        if let Some(limit) = self.max_memory_bytes {
            report.push_str(&format!(
                " - Memory Limit: {} bytes\n\
                 - Memory Usage: {} bytes\n\
                 - Remaining: {} bytes\n",
                limit,
                self.total_allocated_bytes,
                limit.saturating_sub(self.total_allocated_bytes)
            ));
        }

        // Add active handle details if any exist
        if active_count > 0 {
            report.push_str("\nActive Handle Details:\n");
            for (i, handle_stats) in self.active_handle_stats().iter().enumerate() {
                report.push_str(&format!(
                    "  Handle {}: {} bytes ({} elements of {} bytes each), age: {}ms\n",
                    i + 1,
                    handle_stats.size_bytes,
                    handle_stats.size_elements,
                    handle_stats.element_size,
                    handle_stats.age_ms
                ));
            }
        }

        report
    }
}

/// Handle representing a memory allocation
#[derive(Debug)]
pub struct MemoryHandle<T> {
    size: usize,
    allocated_at: std::time::SystemTime,
    phantom: std::marker::PhantomData<T>,
}

impl<T> MemoryHandle<T> {
    /// Create a new memory handle with the specified size
    pub fn new(size: usize) -> Self {
        Self {
            size,
            allocated_at: std::time::SystemTime::now(),
            phantom: std::marker::PhantomData,
        }
    }

    /// Get the size of the memory allocation in bytes
    pub fn size_bytes(&self) -> usize {
        self.size
    }

    /// Get the size of the memory allocation in elements
    pub fn size_elements(&self) -> usize {
        self.size / std::mem::size_of::<T>()
    }

    /// Get the time when this handle was allocated
    pub fn allocated_at(&self) -> std::time::SystemTime {
        self.allocated_at
    }

    /// Get the age of this allocation in milliseconds
    pub fn age_ms(&self) -> Result<u64> {
        self.allocated_at
            .elapsed()
            .map(|duration| duration.as_millis() as u64)
            .map_err(|_| MemoryError::AllocationFailed { size: 0 }.into())
    }

    /// Check if this handle represents a valid allocation
    pub fn is_valid(&self) -> bool {
        self.size > 0
    }

    /// Get memory usage statistics for this handle
    pub fn usage_stats(&self) -> MemoryHandleStats {
        MemoryHandleStats {
            size_bytes: self.size,
            size_elements: self.size_elements(),
            element_size: std::mem::size_of::<T>(),
            allocated_at: self.allocated_at,
            age_ms: self.age_ms().unwrap_or(0),
        }
    }
}

impl<T: Send + Sync + std::fmt::Debug + 'static> AnyMemoryHandle for MemoryHandle<T> {
    fn size_bytes(&self) -> usize {
        self.size
    }

    fn element_size(&self) -> usize {
        std::mem::size_of::<T>()
    }

    fn allocated_at(&self) -> std::time::SystemTime {
        self.allocated_at
    }

    fn age_ms(&self) -> Result<u64> {
        self.allocated_at
            .elapsed()
            .map(|duration| duration.as_millis() as u64)
            .map_err(|_| MemoryError::AllocationFailed { size: 0 }.into())
    }

    fn is_valid(&self) -> bool {
        self.size > 0
    }

    fn usage_stats(&self) -> MemoryHandleStats {
        MemoryHandleStats {
            size_bytes: self.size,
            size_elements: self.size_elements(),
            element_size: std::mem::size_of::<T>(),
            allocated_at: self.allocated_at,
            age_ms: self.age_ms().unwrap_or(0),
        }
    }
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

    #[test]
    fn test_memory_handle_creation() {
        let handle: MemoryHandle<i32> = MemoryHandle::new(1024);
        assert_eq!(handle.size_bytes(), 1024);
        assert_eq!(handle.size_elements(), 1024 / std::mem::size_of::<i32>());
        assert!(handle.is_valid());
        assert!(handle.age_ms().unwrap() >= 0);
    }

    #[test]
    fn test_memory_handle_stats() {
        let handle: MemoryHandle<f64> = MemoryHandle::new(2048);
        let stats = handle.usage_stats();
        assert_eq!(stats.size_bytes, 2048);
        assert_eq!(stats.size_elements, 2048 / std::mem::size_of::<f64>());
        assert_eq!(stats.element_size, std::mem::size_of::<f64>());
        assert!(stats.age_ms >= 0);
    }

    #[test]
    fn test_memory_pool_with_limits() {
        let mut pool: MemoryPool<u8> = MemoryPool::with_memory_limit(100, 5, 1000);
        assert_eq!(pool.memory_usage(), 0);
        assert_eq!(pool.memory_limit(), Some(1000));
        assert_eq!(pool.remaining_memory(), Some(1000));
        assert!(!pool.is_memory_pressure(50.0));

        // Allocate some memory
        let handle1 = pool.allocate::<u8>(500).unwrap();
        assert_eq!(pool.memory_usage(), 500);
        assert_eq!(pool.remaining_memory(), Some(500));
        assert!(pool.is_memory_pressure(50.0));

        // Try to allocate too much
        let result = pool.allocate::<u8>(600);
        assert!(result.is_err());

        // Deallocate and check
        pool.deallocate(handle1).unwrap();
        assert_eq!(pool.memory_usage(), 0);
        assert_eq!(pool.remaining_memory(), Some(1000));
    }

    #[test]
    fn test_memory_pool_reporting() {
        let pool: MemoryPool<i32> = MemoryPool::with_memory_limit(50, 2, 500);
        let report = pool.memory_report();
        assert!(report.contains("Memory Pool Report"));
        assert!(report.contains("Memory Limit: 500 bytes"));
    }

    #[test]
    fn test_memory_handle_age() {
        let handle: MemoryHandle<u16> = MemoryHandle::new(256);

        // Sleep for a short time to test age calculation
        std::thread::sleep(std::time::Duration::from_millis(10));

        let age = handle.age_ms().unwrap();
        assert!(age >= 10);
    }

    #[test]
    fn test_active_handle_tracking() {
        let mut pool: MemoryPool<u8> = MemoryPool::new(100, 5);

        // Initially no active handles
        assert_eq!(pool.active_handle_count(), 0);
        assert_eq!(pool.active_handles_memory(), 0);

        // Allocate some handles
        let handle1 = pool.allocate::<i32>(128).unwrap();
        let handle2 = pool.allocate::<f64>(256).unwrap();

        // Should now have 2 active handles
        assert_eq!(pool.active_handle_count(), 2);
        assert_eq!(pool.active_handles_memory(), 128 + 256);

        // Get handle statistics
        let stats = pool.active_handle_stats();
        assert_eq!(stats.len(), 2);

        // Deallocate one handle
        pool.deallocate(handle1).unwrap();
        assert_eq!(pool.active_handle_count(), 1);
        assert_eq!(pool.active_handles_memory(), 256);

        // Deallocate the other handle
        pool.deallocate(handle2).unwrap();
        assert_eq!(pool.active_handle_count(), 0);
        assert_eq!(pool.active_handles_memory(), 0);
    }

    #[test]
    fn test_active_handle_tracker() {
        let tracker = ActiveHandleTracker::new();

        // Initially empty
        assert_eq!(tracker.active_count(), 0);
        assert_eq!(tracker.total_memory(), 0);

        // Add some handles
        let handle1: MemoryHandle<i32> = MemoryHandle::new(1024);
        let handle2: MemoryHandle<f64> = MemoryHandle::new(2048);

        let allocated_at1 = handle1.allocated_at();
        let _allocated_at2 = handle2.allocated_at();

        tracker.add_handle(Box::new(handle1));
        tracker.add_handle(Box::new(handle2));

        assert_eq!(tracker.active_count(), 2);
        assert_eq!(tracker.total_memory(), 3072);

        // Remove one handle
        let removed = tracker.remove_handle(1024, allocated_at1);
        assert!(removed.is_some());
        assert_eq!(tracker.active_count(), 1);
        assert_eq!(tracker.total_memory(), 2048);

        // Get all stats
        let all_stats = tracker.all_handle_stats();
        assert_eq!(all_stats.len(), 1);
        assert_eq!(all_stats[0].size_bytes, 2048);

        // Clear all
        tracker.clear();
        assert_eq!(tracker.active_count(), 0);
    }

    #[test]
    fn test_any_memory_handle_trait() {
        let handle: MemoryHandle<u32> = MemoryHandle::new(512);

        // Test AnyMemoryHandle trait methods
        assert_eq!(handle.size_bytes(), 512);
        assert_eq!(handle.element_size(), std::mem::size_of::<u32>());
        assert!(handle.is_valid());
        assert!(handle.age_ms().unwrap() >= 0);

        let stats = handle.usage_stats();
        assert_eq!(stats.size_bytes, 512);
        assert_eq!(stats.element_size, std::mem::size_of::<u32>());
    }

    #[test]
    fn test_memory_pool_detailed_reporting() {
        let mut pool: MemoryPool<i64> = MemoryPool::with_memory_limit(50, 2, 1000);

        // Allocate some handles
        let _handle1 = pool.allocate::<u8>(200).unwrap();
        let _handle2 = pool.allocate::<u16>(300).unwrap();

        let report = pool.detailed_memory_report();
        assert!(report.contains("Detailed Memory Pool Report"));
        assert!(report.contains("Active Handles: 2"));
        assert!(report.contains("Active Handle Memory: 500 bytes"));
        assert!(report.contains("Active Handle Details:"));
    }

    #[test]
    fn test_memory_pool_clear_active_handles() {
        let mut pool: MemoryPool<f32> = MemoryPool::new(100, 5);

        // Allocate some handles
        let _handle1 = pool.allocate::<i32>(128).unwrap();
        let _handle2 = pool.allocate::<f64>(256).unwrap();

        assert_eq!(pool.active_handle_count(), 2);

        // Clear active handles
        pool.clear_active_handles();
        assert_eq!(pool.active_handle_count(), 0);

        // Memory usage should still be tracked in total_allocated_bytes
        assert!(pool.memory_usage() > 0);
    }
}
