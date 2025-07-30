//! OpenMP wrapper compatibility module using Rayon
//!
//! - Dynamic creation and reconstruction of thread pools
//! - Panic capture and re-panic (exception handling)
//! - Single-thread fallback controlled by feature flag

use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Mutex;

// Manage a global thread pool with mutex for synchronization
lazy_static::lazy_static! {
    static ref GLOBAL_POOL: Mutex<ThreadPool> = {
        // Default: use all logical CPU cores
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_cpus::get())
            .build()
            .expect("Failed to build default Rayon thread pool");
        Mutex::new(pool)
    };
}

/// Equivalent to OpenMP's omp_set_num_threads: rebuild the pool with the given thread count
pub fn omp_set_num_threads(num_threads: usize) {
    let new_pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Failed to build Rayon thread pool");
    *GLOBAL_POOL.lock().unwrap() = new_pool;
}

/// Equivalent to OpenMP's omp_num_threads: return the current number of threads in the pool
pub fn omp_num_threads() -> usize {
    GLOBAL_POOL.lock().unwrap().current_num_threads()
}

/// Helper to capture and rethrow panics from parallel execution
#[derive(Debug)]
pub struct ThreadExceptionHelper {
    panic_payload: Option<Box<dyn std::any::Any + Send + 'static>>,
}

impl ThreadExceptionHelper {
    /// Create a new helper
    pub fn new() -> Self {
        ThreadExceptionHelper {
            panic_payload: None,
        }
    }

    /// Capture a panic payload once
    pub fn capture(&mut self, payload: Box<dyn std::any::Any + Send + 'static>) {
        if self.panic_payload.is_none() {
            self.panic_payload = Some(payload);
        }
    }

    /// Re-panic with the captured payload, if any
    pub fn rethrow(self) {
        if let Some(payload) = self.panic_payload {
            panic::resume_unwind(payload);
        }
    }
}

/// Execute a parallel loop over an iterator, capturing any panics and rethrowing after completion
pub fn omp_parallel_for<I, F>(iter: I, mut func: F)
where
    I: IntoIterator + Send,
    I::Item: Send,
    F: FnMut(I::Item) + Send,
{
    let mut helper = ThreadExceptionHelper::new();
    // Use the global pool for parallel execution
    GLOBAL_POOL.lock().unwrap().install(|| {
        iter.into_iter().for_each(|item| {
            // Equivalent to OMP_LOOP_EX_BEGIN / OMP_LOOP_EX_END
            let result = panic::catch_unwind(AssertUnwindSafe(|| {
                func(item);
            }));
            if let Err(payload) = result {
                helper.capture(payload);
            }
        });
    });
    // Equivalent to OMP_THROW_EX
    helper.rethrow();
}

// -- If built with feature "no_omp", force single-threaded mode --
#[cfg(feature = "no_omp")]
pub fn init_single_thread() {
    omp_set_num_threads(1);
}

/// Initialize single-threaded mode (no-op when OpenMP support is enabled)
#[cfg(not(feature = "no_omp"))]
pub fn init_single_thread() {
    // No-op when OpenMP support is enabled
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn test_thread_count_change() {
        let default = omp_num_threads();
        omp_set_num_threads(2);
        assert_eq!(omp_num_threads(), 2);
        omp_set_num_threads(default);
        assert_eq!(omp_num_threads(), default);
    }

    #[test]
    #[should_panic(expected = "explicit panic")]
    fn test_parallel_for_panic() {
        // Ensure that a panic inside the parallel loop is propagated
        let v = vec![1, 2, 3];
        omp_parallel_for(v, |x| {
            if x == 2 {
                panic!("explicit panic");
            }
        });
    }

    #[test]
    fn test_parallel_for_success() {
        // Verify correct behavior when no panics occur
        let v = vec![1, 2, 3, 4];
        let sum = Mutex::new(0);
        omp_parallel_for(v, |x| {
            let mut guard = sum.lock().unwrap();
            *guard += x;
        });
        assert_eq!(*sum.lock().unwrap(), 10);
    }
}
