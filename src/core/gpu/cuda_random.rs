/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

use cubecl::prelude::*;

/// A wrapper for random generator - Rust implementation of LightGBM's CUDARandom
/// This is a direct port of the C++ CUDA random number generator that maintains
/// semantic equivalence with the original implementation.
#[derive(Clone, Copy, Debug)]
pub struct CudaRandom {
    /// Internal state variable, initialized to 123456789 to match C++ version
    x: u32,
}

impl CudaRandom {
    /// Create a new CudaRandom instance with default seed
    pub fn new() -> Self {
        Self { x: 123456789 }
    }

    /// Create a new CudaRandom instance with specific seed
    pub fn with_seed(seed: i32) -> Self {
        Self { x: seed as u32 }
    }

    /// Set specific seed
    /// Equivalent to C++ `__device__ void SetSeed(int seed)`
    pub fn set_seed(&mut self, seed: i32) {
        self.x = seed as u32;
    }

    /// Generate random integer, int16 range. [0, 65536]
    /// Returns the random integer between [lower_bound, upper_bound)
    /// Equivalent to C++ `__device__ inline int NextShort(int lower_bound, int upper_bound)`
    pub fn next_short(&mut self, lower_bound: i32, upper_bound: i32) -> i32 {
        (self.rand_int16()) % (upper_bound - lower_bound) + lower_bound
    }

    /// Generate random integer, int32 range
    /// Returns the random integer between [lower_bound, upper_bound)
    /// Equivalent to C++ `__device__ inline int NextInt(int lower_bound, int upper_bound)`
    pub fn next_int(&mut self, lower_bound: i32, upper_bound: i32) -> i32 {
        (self.rand_int32()) % (upper_bound - lower_bound) + lower_bound
    }

    /// Generate random float data
    /// Returns the random float between [0.0, 1.0)
    /// Equivalent to C++ `__device__ inline float NextFloat()`
    pub fn next_float(&mut self) -> f32 {
        // get random float in [0,1)
        (self.rand_int16() as f32) / 32768.0f32
    }

    /// Private method equivalent to C++ `__device__ inline int RandInt16()`
    /// Uses Linear Congruential Generator: x = (214013 * x + 2531011)
    /// Returns 15-bit random integer (masked to 0x7FFF)
    fn rand_int16(&mut self) -> i32 {
        self.x = self.x.wrapping_mul(214013).wrapping_add(2531011);
        ((self.x >> 16) & 0x7FFF) as i32
    }

    /// Private method equivalent to C++ `__device__ inline int RandInt32()`
    /// Uses Linear Congruential Generator: x = (214013 * x + 2531011)
    /// Returns 31-bit random integer (masked to 0x7FFFFFFF)
    fn rand_int32(&mut self) -> i32 {
        self.x = self.x.wrapping_mul(214013).wrapping_add(2531011);
        (self.x & 0x7FFFFFFF) as i32
    }
}

impl Default for CudaRandom {
    fn default() -> Self {
        Self::new()
    }
}

// CubeCL GPU kernels for batch random number generation
#[cube(launch_unchecked)]
pub fn cuda_random_generate_shorts(
    output: &mut Array<i32>,
    seed: i32,
    lower_bound: i32,
    upper_bound: i32,
) {
    let index = ABSOLUTE_POS;
    if index < output.len() {
        // Each thread gets its own generator state based on seed + index
        let thread_seed = seed + i32::cast_from(index);
        let mut x = u32::cast_from(thread_seed);

        // Apply the same LCG formula as the C++ version
        x = x * 214013u32 + 2531011u32;
        let rand_val = ((x >> 16) & 0x7FFF) as i32;

        output[index] = rand_val % (upper_bound - lower_bound) + lower_bound;
    }
}

#[cube(launch_unchecked)]
pub fn cuda_random_generate_ints(
    output: &mut Array<i32>,
    seed: i32,
    lower_bound: i32,
    upper_bound: i32,
) {
    let index = ABSOLUTE_POS;
    if index < output.len() {
        // Each thread gets its own generator state based on seed + index
        let thread_seed = seed + i32::cast_from(index);
        let mut x = u32::cast_from(thread_seed);

        // Apply the same LCG formula as the C++ version
        x = x * 214013u32 + 2531011u32;
        let rand_val = (x & 0x7FFFFFFF) as i32;

        output[index] = rand_val % (upper_bound - lower_bound) + lower_bound;
    }
}

#[cube(launch_unchecked)]
pub fn cuda_random_generate_floats(output: &mut Array<f32>, seed: i32) {
    let index = ABSOLUTE_POS;
    if index < output.len() {
        // Each thread gets its own generator state based on seed + index
        let thread_seed = seed + i32::cast_from(index);
        let mut x = u32::cast_from(thread_seed);

        // Apply the same LCG formula as the C++ version
        x = x * 214013u32 + 2531011u32;
        let rand_val = ((x >> 16) & 0x7FFF) as i32;

        // Convert to float in [0.0, 1.0) range exactly like C++ version
        output[index] = (rand_val as f32) / 32768.0f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_random_default_construction() {
        let rng = CudaRandom::new();
        assert_eq!(rng.x, 123456789);
    }

    #[test]
    fn test_cuda_random_with_seed() {
        let rng = CudaRandom::with_seed(42);
        assert_eq!(rng.x, 42);
    }

    #[test]
    fn test_set_seed() {
        let mut rng = CudaRandom::new();
        rng.set_seed(12345);
        assert_eq!(rng.x, 12345);
    }

    #[test]
    fn test_next_short_range() {
        let mut rng = CudaRandom::new();
        for _ in 0..100 {
            let val = rng.next_short(10, 20);
            assert!(
                val >= 10 && val < 20,
                "Value {} is outside range [10, 20)",
                val
            );
        }
    }

    #[test]
    fn test_next_int_range() {
        let mut rng = CudaRandom::new();
        for _ in 0..100 {
            let val = rng.next_int(100, 200);
            assert!(
                val >= 100 && val < 200,
                "Value {} is outside range [100, 200)",
                val
            );
        }
    }

    #[test]
    fn test_next_float_range() {
        let mut rng = CudaRandom::new();
        for _ in 0..100 {
            let val = rng.next_float();
            assert!(
                val >= 0.0 && val < 1.0,
                "Value {} is outside range [0.0, 1.0)",
                val
            );
        }
    }

    #[test]
    fn test_deterministic_behavior() {
        let mut rng1 = CudaRandom::with_seed(42);
        let mut rng2 = CudaRandom::with_seed(42);

        // Both generators should produce the same sequence
        for _ in 0..10 {
            assert_eq!(rng1.next_int(0, 1000), rng2.next_int(0, 1000));
            assert_eq!(rng1.next_short(0, 100), rng2.next_short(0, 100));

            let f1 = rng1.next_float();
            let f2 = rng2.next_float();
            assert!(
                (f1 - f2).abs() < f32::EPSILON,
                "Float values differ: {} vs {}",
                f1,
                f2
            );
        }
    }

    #[test]
    fn test_lcg_formula_equivalence() {
        // Test that our LCG implementation matches the C++ version exactly
        let mut rng = CudaRandom::with_seed(123456789);

        // First call should update x to: (214013 * 123456789 + 2531011)
        // and return ((new_x >> 16) & 0x7FFF)
        let first_rand16 = rng.rand_int16();
        let expected_x_after_first = 123456789u32.wrapping_mul(214013).wrapping_add(2531011);
        let expected_first_rand16 = ((expected_x_after_first >> 16) & 0x7FFF) as i32;

        assert_eq!(first_rand16, expected_first_rand16);
        assert_eq!(rng.x, expected_x_after_first);
    }
}
