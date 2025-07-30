/*!
 * Copyright (c) 2023 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

use std::sync::atomic::{AtomicI32, Ordering};

/// Global maximum number of threads for LightGBM operations
static LGBM_MAX_NUM_THREADS: AtomicI32 = AtomicI32::new(-1);

/// Global default number of threads for LightGBM operations
static LGBM_DEFAULT_NUM_THREADS: AtomicI32 = AtomicI32::new(-1);

/// Returns the number of threads to use for parallel operations.
/// 
/// This function determines the appropriate number of threads based on:
/// 1. LightGBM-specific default if set
/// 2. OpenMP/Rayon global configuration otherwise
/// 3. Respects the maximum thread limit if set
pub fn omp_num_threads() -> i32 {
    let mut default_num_threads = 1;

    let lgbm_default = LGBM_DEFAULT_NUM_THREADS.load(Ordering::Relaxed);
    
    if lgbm_default > 0 {
        // if LightGBM-specific default has been set, ignore OpenMP-global config
        default_num_threads = lgbm_default;
    } else {
        // otherwise, default to Rayon's thread pool size (equivalent to OpenMP's omp_get_max_threads)
        default_num_threads = rayon::current_num_threads() as i32;
    }

    let lgbm_max = LGBM_MAX_NUM_THREADS.load(Ordering::Relaxed);
    
    // ensure that if lgbm_set_max_threads() was ever called, LightGBM doesn't
    // use more than that many threads
    if lgbm_max > 0 && default_num_threads > lgbm_max {
        return lgbm_max;
    }

    default_num_threads
}

/// Sets the default number of threads for LightGBM operations.
/// 
/// # Arguments
/// 
/// * `num_threads` - The number of threads to use as default. 
///                   If <= 0, resets to use global configuration.
pub fn omp_set_num_threads(num_threads: i32) {
    if num_threads <= 0 {
        LGBM_DEFAULT_NUM_THREADS.store(-1, Ordering::Relaxed);
    } else {
        LGBM_DEFAULT_NUM_THREADS.store(num_threads, Ordering::Relaxed);
    }
}

/// Sets the maximum number of threads for LightGBM operations.
/// 
/// # Arguments
/// 
/// * `max_threads` - The maximum number of threads allowed.
///                   If <= 0, removes the limit.
pub fn lgbm_set_max_threads(max_threads: i32) {
    if max_threads <= 0 {
        LGBM_MAX_NUM_THREADS.store(-1, Ordering::Relaxed);
    } else {
        LGBM_MAX_NUM_THREADS.store(max_threads, Ordering::Relaxed);
    }
}

/// Gets the current maximum thread limit.
pub fn lgbm_get_max_threads() -> i32 {
    LGBM_MAX_NUM_THREADS.load(Ordering::Relaxed)
}

/// Gets the current default thread count.
pub fn lgbm_get_default_threads() -> i32 {
    LGBM_DEFAULT_NUM_THREADS.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_behavior() {
        // Reset to defaults
        omp_set_num_threads(-1);
        lgbm_set_max_threads(-1);
        
        // Should return at least 1 thread
        assert!(omp_num_threads() >= 1);
    }

    #[test]
    fn test_set_default_threads() {
        // Set default to 4 threads
        omp_set_num_threads(4);
        assert_eq!(omp_num_threads(), 4);
        
        // Reset
        omp_set_num_threads(-1);
    }

    #[test]
    fn test_max_threads_limit() {
        // Set default to 8 and max to 4
        omp_set_num_threads(8);
        lgbm_set_max_threads(4);
        
        // Should be limited to 4
        assert_eq!(omp_num_threads(), 4);
        
        // Reset
        omp_set_num_threads(-1);
        lgbm_set_max_threads(-1);
    }

    #[test]
    fn test_negative_values() {
        // Setting negative values should reset to defaults
        omp_set_num_threads(-5);
        assert_eq!(lgbm_get_default_threads(), -1);
        
        lgbm_set_max_threads(-10);
        assert_eq!(lgbm_get_max_threads(), -1);
    }
}