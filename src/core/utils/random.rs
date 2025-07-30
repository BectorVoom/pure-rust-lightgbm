/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;

/// A wrapper for random generator
#[derive(Debug)]
pub struct Random {
    x: u32,
}

impl Random {
    /// Constructor, with random seed
    pub fn new() -> Self {
        let mut rng = StdRng::from_entropy();
        let seed: i32 = rng.gen_range(0..=65536);
        Random { x: seed as u32 }
    }

    /// Constructor, with specific seed
    pub fn with_seed(seed: i32) -> Self {
        Random { x: seed as u32 }
    }

    /// Generate random integer, int16 range. [0, 65536]
    /// Returns The random integer between [lower_bound, upper_bound)
    pub fn next_short(&mut self, lower_bound: i32, upper_bound: i32) -> i32 {
        self.rand_int16() % (upper_bound - lower_bound) + lower_bound
    }

    /// Generate random integer, int32 range
    /// Returns The random integer between [lower_bound, upper_bound)
    pub fn next_int(&mut self, lower_bound: i32, upper_bound: i32) -> i32 {
        self.rand_int32() % (upper_bound - lower_bound) + lower_bound
    }

    /// Generate random float data
    /// Returns The random float between [0.0, 1.0)
    pub fn next_float(&mut self) -> f32 {
        // get random float in [0,1)
        (self.rand_int16() as f32) / 32768.0
    }

    /// Sample K data from {0,1,...,N-1}
    /// Returns K Ordered sampled data from {0,1,...,N-1}
    pub fn sample(&mut self, n: i32, k: i32) -> Vec<i32> {
        let mut ret = Vec::new();
        ret.reserve(k as usize);

        if k > n || k <= 0 {
            return ret;
        } else if k == n {
            for i in 0..n {
                ret.push(i);
            }
        } else if k > 1 && (k as f64) > (n as f64 / (k as f64).log2()) {
            for i in 0..n {
                let prob = (k - ret.len() as i32) as f64 / (n - i) as f64;
                if self.next_float() < prob as f32 {
                    ret.push(i);
                }
            }
        } else {
            let mut sample_set = HashSet::new();
            for r in (n - k)..n {
                let v = self.next_int(0, r + 1);
                if !sample_set.insert(v) {
                    sample_set.insert(r);
                }
            }
            let mut sorted_vec: Vec<i32> = sample_set.into_iter().collect();
            sorted_vec.sort();
            ret = sorted_vec;
        }

        ret
    }

    /// Generate random int16
    fn rand_int16(&mut self) -> i32 {
        self.x = self.x.wrapping_mul(214013).wrapping_add(2531011);
        ((self.x >> 16) & 0x7FFF) as i32
    }

    /// Generate random int32
    fn rand_int32(&mut self) -> i32 {
        self.x = self.x.wrapping_mul(214013).wrapping_add(2531011);
        (self.x & 0x7FFFFFFF) as i32
    }
}

impl Default for Random {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_with_seed() {
        let mut rng1 = Random::with_seed(123456789);
        let mut rng2 = Random::with_seed(123456789);

        // Same seed should produce same results
        assert_eq!(rng1.next_int(0, 100), rng2.next_int(0, 100));
        assert_eq!(rng1.next_short(0, 1000), rng2.next_short(0, 1000));
        assert_eq!(rng1.next_float(), rng2.next_float());
    }

    #[test]
    fn test_next_short_bounds() {
        let mut rng = Random::with_seed(42);
        for _ in 0..100 {
            let val = rng.next_short(10, 20);
            assert!(val >= 10 && val < 20);
        }
    }

    #[test]
    fn test_next_int_bounds() {
        let mut rng = Random::with_seed(42);
        for _ in 0..100 {
            let val = rng.next_int(100, 200);
            assert!(val >= 100 && val < 200);
        }
    }

    #[test]
    fn test_next_float_bounds() {
        let mut rng = Random::with_seed(42);
        for _ in 0..100 {
            let val = rng.next_float();
            assert!(val >= 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_sample_edge_cases() {
        let mut rng = Random::with_seed(42);

        // k > n should return empty vector
        assert_eq!(rng.sample(5, 10).len(), 0);

        // k <= 0 should return empty vector
        assert_eq!(rng.sample(5, 0).len(), 0);
        assert_eq!(rng.sample(5, -1).len(), 0);

        // k == n should return all elements
        let result = rng.sample(5, 5);
        assert_eq!(result.len(), 5);
        assert_eq!(result, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_sample_normal_case() {
        let mut rng = Random::with_seed(42);
        let result = rng.sample(10, 3);

        assert_eq!(result.len(), 3);
        // Check all elements are unique
        let mut unique_check = HashSet::new();
        for &val in &result {
            assert!(unique_check.insert(val));
            assert!(val >= 0 && val < 10);
        }

        // Check result is sorted
        let mut sorted_result = result.clone();
        sorted_result.sort();
        assert_eq!(result, sorted_result);
    }
}
