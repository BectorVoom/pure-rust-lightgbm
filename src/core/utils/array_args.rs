///
use crate::core::utils::openmp_wrapper::omp_num_threads;
use crate::core::utils::threading::Threading;
use std::sync::{Arc, Mutex};

/// Contains array operations equivalent to C++ LightGBM::ArrayArgs<VAL_T>
/// Provides operations like ArgMax, TopK, etc.
#[derive(Debug)]
pub struct ArrayArgs;

impl ArrayArgs {
    /// Find index of maximum element using multi-threading for large arrays
    /// Equivalent to C++ ArrayArgs<VAL_T>::ArgMaxMT
    pub fn arg_max_mt<T>(array: &[T]) -> usize
    where
        T: PartialOrd + Copy + Send + Sync,
    {
        let num_threads = omp_num_threads() as i32;
        let arg_maxs = Arc::new(Mutex::new(vec![0usize; num_threads as usize]));
        let arg_maxs_clone = arg_maxs.clone();

        let n_blocks = Threading::for_loop(0usize, array.len(), 1024usize, move |i, start, end| {
            let mut arg_max = start;
            for j in (start + 1)..end {
                if array[j] > array[arg_max] {
                    arg_max = j;
                }
            }
            let mut maxs = arg_maxs_clone.lock().unwrap();
            maxs[i as usize] = arg_max;
        });

        let arg_maxs_final = arg_maxs.lock().unwrap();
        let mut ret = arg_maxs_final[0];
        for i in 1..(n_blocks as usize) {
            if array[arg_maxs_final[i]] > array[ret] {
                ret = arg_maxs_final[i];
            }
        }
        ret
    }

    /// Find index of maximum element in vector (with MT optimization for large arrays)
    /// Equivalent to C++ ArrayArgs<VAL_T>::ArgMax(const std::vector<VAL_T>& array)
    pub fn arg_max_vec<T>(array: &[T]) -> usize
    where
        T: PartialOrd + Copy + Send + Sync,
    {
        if array.is_empty() {
            return 0;
        }
        if array.len() > 1024 {
            Self::arg_max_mt(array)
        } else {
            let mut arg_max = 0;
            for i in 1..array.len() {
                if array[i] > array[arg_max] {
                    arg_max = i;
                }
            }
            arg_max
        }
    }

    /// Find index of minimum element in vector
    /// Equivalent to C++ ArrayArgs<VAL_T>::ArgMin(const std::vector<VAL_T>& array)
    pub fn arg_min_vec<T>(array: &[T]) -> usize
    where
        T: PartialOrd + Copy,
    {
        if array.is_empty() {
            return 0;
        }
        let mut arg_min = 0;
        for i in 1..array.len() {
            if array[i] < array[arg_min] {
                arg_min = i;
            }
        }
        arg_min
    }

    /// Find index of maximum element in slice
    /// Equivalent to C++ ArrayArgs<VAL_T>::ArgMax(const VAL_T* array, size_t n)
    pub fn arg_max_slice<T>(array: &[T]) -> usize
    where
        T: PartialOrd + Copy,
    {
        if array.is_empty() {
            return 0;
        }
        let mut arg_max = 0;
        for i in 1..array.len() {
            if array[i] > array[arg_max] {
                arg_max = i;
            }
        }
        arg_max
    }

    /// Find index of minimum element in slice
    /// Equivalent to C++ ArrayArgs<VAL_T>::ArgMin(const VAL_T* array, size_t n)
    pub fn arg_min_slice<T>(array: &[T]) -> usize
    where
        T: PartialOrd + Copy,
    {
        if array.is_empty() {
            return 0;
        }
        let mut arg_min = 0;
        for i in 1..array.len() {
            if array[i] < array[arg_min] {
                arg_min = i;
            }
        }
        arg_min
    }

    /// Partition function for quickselect algorithm
    /// Equivalent to C++ ArrayArgs<VAL_T>::Partition
    pub fn partition<T>(arr: &mut [T], start: usize, end: usize) -> (usize, usize)
    where
        T: PartialOrd + Copy,
    {
        if start >= end.saturating_sub(1) {
            return (start.saturating_sub(1), end);
        }

        let mut i = start.saturating_sub(1);
        let mut j = end.saturating_sub(1);
        let mut p = i;
        let mut q = j;
        let v = arr[end - 1];

        loop {
            while {
                i = i.saturating_add(1);
                i < arr.len() && arr[i] > v
            } {}

            while v > arr[j] {
                if j == start {
                    break;
                }
                j = j.saturating_sub(1);
            }

            if i >= j {
                break;
            }

            arr.swap(i, j);

            if arr[i] == v {
                p = p.saturating_add(1);
                arr.swap(p, i);
            }
            if v == arr[j] {
                q = q.saturating_sub(1);
                arr.swap(j, q);
            }
        }

        arr.swap(i, end - 1);
        j = i.saturating_sub(1);
        i = i.saturating_add(1);

        let mut k = start;
        while k <= p && j > 0 {
            arr.swap(k, j);
            k = k.saturating_add(1);
            j = j.saturating_sub(1);
        }

        k = end.saturating_sub(2);
        while k >= q && i < arr.len() {
            arr.swap(i, k);
            k = k.saturating_sub(1);
            i = i.saturating_add(1);
        }

        (j, i)
    }

    /// Find k-th largest element using quickselect
    /// Equivalent to C++ ArrayArgs<VAL_T>::ArgMaxAtK
    /// Note: k refers to index here (0-based)
    pub fn arg_max_at_k<T>(arr: &mut [T], start: usize, end: usize, k: usize) -> usize
    where
        T: PartialOrd + Copy,
    {
        if start >= end.saturating_sub(1) {
            return start;
        }

        let (l, r) = Self::partition(arr, start, end);

        // If found or all elements are the same
        if (k > l && k < r) || (l == start.saturating_sub(1) && r == end.saturating_sub(1)) {
            k
        } else if k <= l {
            Self::arg_max_at_k(arr, start, l + 1, k)
        } else {
            Self::arg_max_at_k(arr, r, end, k)
        }
    }

    /// Get top k maximum values
    /// Equivalent to C++ ArrayArgs<VAL_T>::MaxK
    /// Note: k is 1-based (k=3 means top 3 numbers)
    pub fn max_k<T>(array: &[T], k: i32) -> Vec<T>
    where
        T: PartialOrd + Copy,
    {
        let mut out = Vec::new();
        if k <= 0 {
            return out;
        }

        out.extend_from_slice(array);

        if (k as usize) >= array.len() {
            return out;
        }

        let out_len = out.len();
        Self::arg_max_at_k(&mut out, 0, out_len, (k - 1) as usize);
        out.truncate(k as usize);
        out
    }

    /// Fill vector with a value
    /// Equivalent to C++ ArrayArgs<VAL_T>::Assign
    pub fn assign<T>(array: &mut Vec<T>, t: T, n: usize)
    where
        T: Copy,
    {
        array.resize(n, t);
        for i in 0..array.len() {
            array[i] = t;
        }
    }

    /// Check if all elements are zero
    /// Equivalent to C++ ArrayArgs<VAL_T>::CheckAllZero
    pub fn check_all_zero<T>(array: &[T]) -> bool
    where
        T: PartialEq + Default,
    {
        let zero = T::default();
        for item in array {
            if *item != zero {
                return false;
            }
        }
        true
    }

    /// Check if all elements equal a specific value
    /// Equivalent to C++ ArrayArgs<VAL_T>::CheckAll
    pub fn check_all<T>(array: &[T], t: T) -> bool
    where
        T: PartialEq,
    {
        for item in array {
            if *item != t {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arg_max_vec_basic() {
        let array = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        assert_eq!(ArrayArgs::arg_max_vec(&array), 1);

        let array_single = vec![42.0];
        assert_eq!(ArrayArgs::arg_max_vec(&array_single), 0);

        let array_empty: Vec<f64> = vec![];
        assert_eq!(ArrayArgs::arg_max_vec(&array_empty), 0);
    }

    #[test]
    fn test_arg_min_vec_basic() {
        let array = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(ArrayArgs::arg_min_vec(&array), 1);

        let array_single = vec![42.0];
        assert_eq!(ArrayArgs::arg_min_vec(&array_single), 0);

        let array_empty: Vec<f64> = vec![];
        assert_eq!(ArrayArgs::arg_min_vec(&array_empty), 0);
    }

    #[test]
    fn test_arg_max_slice_basic() {
        let array = [1.0, 5.0, 3.0, 2.0, 4.0];
        assert_eq!(ArrayArgs::arg_max_slice(&array), 1);

        let array_single = [42.0];
        assert_eq!(ArrayArgs::arg_max_slice(&array_single), 0);

        let array_empty: [f64; 0] = [];
        assert_eq!(ArrayArgs::arg_max_slice(&array_empty), 0);
    }

    #[test]
    fn test_arg_min_slice_basic() {
        let array = [5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(ArrayArgs::arg_min_slice(&array), 1);

        let array_single = [42.0];
        assert_eq!(ArrayArgs::arg_min_slice(&array_single), 0);

        let array_empty: [f64; 0] = [];
        assert_eq!(ArrayArgs::arg_min_slice(&array_empty), 0);
    }

    #[test]
    fn test_arg_max_mt_large_array() {
        // Test with array larger than 1024 to trigger multi-threading
        let mut array = vec![0.0; 2000];
        array[1500] = 100.0; // Set maximum at index 1500
        assert_eq!(ArrayArgs::arg_max_mt(&array), 1500);

        // Test with maximum at beginning
        array[0] = 200.0;
        assert_eq!(ArrayArgs::arg_max_mt(&array), 0);
    }

    #[test]
    fn test_assign() {
        let mut array = vec![1, 2, 3];
        ArrayArgs::assign(&mut array, 5, 10);
        assert_eq!(array.len(), 10);
        assert!(array.iter().all(|&x| x == 5));

        // Test with zero size
        let mut array2 = vec![1, 2, 3];
        ArrayArgs::assign(&mut array2, 9, 0);
        assert_eq!(array2.len(), 0);
    }

    #[test]
    fn test_check_all_zero() {
        let array_zeros = vec![0, 0, 0, 0];
        assert!(ArrayArgs::check_all_zero(&array_zeros));

        let array_with_nonzero = vec![0, 0, 1, 0];
        assert!(!ArrayArgs::check_all_zero(&array_with_nonzero));

        let array_empty: Vec<i32> = vec![];
        assert!(ArrayArgs::check_all_zero(&array_empty));

        // Test with floats
        let array_float_zeros = vec![0.0, 0.0, 0.0];
        assert!(ArrayArgs::check_all_zero(&array_float_zeros));
    }

    #[test]
    fn test_check_all() {
        let array_fives = vec![5, 5, 5, 5];
        assert!(ArrayArgs::check_all(&array_fives, 5));
        assert!(!ArrayArgs::check_all(&array_fives, 4));

        let array_mixed = vec![5, 5, 3, 5];
        assert!(!ArrayArgs::check_all(&array_mixed, 5));

        let array_empty: Vec<i32> = vec![];
        assert!(ArrayArgs::check_all(&array_empty, 42));
    }

    #[test]
    fn test_max_k() {
        let array = vec![5.0, 2.0, 8.0, 1.0, 9.0, 3.0];

        // Test k=0 (should return empty)
        let result_k0 = ArrayArgs::max_k(&array, 0);
        assert_eq!(result_k0.len(), 0);

        // Test k negative (should return empty)
        let result_neg = ArrayArgs::max_k(&array, -1);
        assert_eq!(result_neg.len(), 0);

        // Test k=3 (should return top 3)
        let result_k3 = ArrayArgs::max_k(&array, 3);
        assert_eq!(result_k3.len(), 3);

        // Test k >= array length (should return entire array)
        let result_large = ArrayArgs::max_k(&array, 10);
        assert_eq!(result_large.len(), array.len());
    }

    #[test]
    fn test_same_inputs_as_cpp() {
        // These test cases should match exactly with C++ version

        // Test case 1: Standard array
        let test1 = vec![1.5, 3.2, 2.1, 4.8, 2.9, 1.1];
        assert_eq!(ArrayArgs::arg_max_vec(&test1), 3); // 4.8 at index 3
        assert_eq!(ArrayArgs::arg_min_vec(&test1), 5); // 1.1 at index 5

        // Test case 2: Array with duplicates at extremes
        let test2 = vec![5.0, 1.0, 3.0, 5.0, 1.0];
        assert_eq!(ArrayArgs::arg_max_vec(&test2), 0); // First occurrence of 5.0
        assert_eq!(ArrayArgs::arg_min_vec(&test2), 1); // First occurrence of 1.0

        // Test case 3: Single element
        let test3 = vec![42.0];
        assert_eq!(ArrayArgs::arg_max_vec(&test3), 0);
        assert_eq!(ArrayArgs::arg_min_vec(&test3), 0);

        // Test case 4: All zeros
        let test4 = vec![0.0, 0.0, 0.0];
        assert!(ArrayArgs::check_all_zero(&test4));
        assert!(ArrayArgs::check_all(&test4, 0.0));

        // Test case 5: MaxK
        let test5 = vec![9.0, 1.0, 5.0, 3.0, 7.0];
        let top3 = ArrayArgs::max_k(&test5, 3);
        assert_eq!(top3.len(), 3);
    }
}
