/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 * 
 * Rust port of LightGBM common utilities
 */

use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use std::time::{Duration, Instant};

use super::json11::{Json, JsonParse};
use super::log::Log;

/// Common utility functions for LightGBM operations.
/// Provides string manipulation, parsing, and various helper functions.
#[derive(Debug)]
pub struct Common;

impl Common {
    
    // String manipulation functions
    
    /// Converts an ASCII character to lowercase.
    /// Only affects ASCII letters A-Z, other characters are returned unchanged.
    pub fn tolower(c: char) -> char {
        if c.is_ascii() && c >= 'A' && c <= 'Z' {
            (c as u8 - b'A' + b'a') as char
        } else {
            c
        }
    }

    /// Trims whitespace and control characters from both ends of a string.
    /// Removes spaces, form feed, newline, carriage return, tab, and vertical tab.
    pub fn trim(s: String) -> String {
        if s.is_empty() {
            return s;
        }
        s.trim_matches(&[' ', '\x0C', '\n', '\r', '\t', '\x0B'][..]).to_string()
    }

    /// Removes surrounding quotation marks (single or double) from a string.
    pub fn remove_quotation_symbol(s: String) -> String {
        if s.is_empty() {
            return s;
        }
        s.trim_matches(&['\'', '"'][..]).to_string()
    }

    /// Checks if a string starts with the given prefix.
    pub fn starts_with(s: &str, prefix: &str) -> bool {
        s.starts_with(prefix)
    }

    // String splitting functions

    /// Splits a string by the given delimiter character.
    /// Returns a vector of strings, excluding empty segments.
    pub fn split(s: &str, delimiter: char) -> Vec<String> {
        let mut result = Vec::new();
        let mut start = 0;
        let chars: Vec<char> = s.chars().collect();
        
        for (i, &ch) in chars.iter().enumerate() {
            if ch == delimiter {
                if start < i {
                    result.push(chars[start..i].iter().collect());
                }
                start = i + 1;
            }
        }
        
        if start < chars.len() {
            result.push(chars[start..].iter().collect());
        }
        
        result
    }

    /// Splits a string by finding content between bracket delimiters.
    /// Returns strings found between left_delimiter and right_delimiter pairs.
    pub fn split_brackets(s: &str, left_delimiter: char, right_delimiter: char) -> Vec<String> {
        let mut result = Vec::new();
        let mut start = 0;
        let mut open = false;
        let chars: Vec<char> = s.chars().collect();
        
        for (i, &ch) in chars.iter().enumerate() {
            if ch == left_delimiter {
                open = true;
                start = i + 1;
            } else if ch == right_delimiter && open {
                if start < i {
                    result.push(chars[start..i].iter().collect());
                }
                open = false;
            }
        }
        
        result
    }

    /// Splits a string into lines, handling both \n and \r line endings.
    /// Consecutive line endings are treated as a single separator.
    pub fn split_lines(s: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut start = 0;
        let chars: Vec<char> = s.chars().collect();
        let mut i = 0;
        
        while i < chars.len() {
            if chars[i] == '\n' || chars[i] == '\r' {
                if start < i {
                    result.push(chars[start..i].iter().collect());
                }
                // Skip line endings
                while i < chars.len() && (chars[i] == '\n' || chars[i] == '\r') {
                    i += 1;
                }
                start = i;
            } else {
                i += 1;
            }
        }
        
        if start < chars.len() {
            result.push(chars[start..].iter().collect());
        }
        
        result
    }

    /// Splits a string by any character found in the delimiters string.
    /// Each character in the delimiters parameter is treated as a potential split point.
    pub fn split_delimiters(s: &str, delimiters: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut start = 0;
        let chars: Vec<char> = s.chars().collect();
        let delimiter_chars: Vec<char> = delimiters.chars().collect();
        
        for (i, &ch) in chars.iter().enumerate() {
            if delimiter_chars.contains(&ch) {
                if start < i {
                    result.push(chars[start..i].iter().collect());
                }
                start = i + 1;
            }
        }
        
        if start < chars.len() {
            result.push(chars[start..].iter().collect());
        }
        
        result
    }

    // JSON config functions

    /// Extracts a string value from a JSON configuration by key.
    /// Returns an error if the JSON is invalid or key is not found.
    pub fn get_from_parser_config(config_str: &str, key: &str) -> Result<String, String> {
        match Json::parse(config_str, JsonParse::Standard) {
            Ok(config_json) => {
                let value = config_json.object_index(key);
                Ok(value.string_value().to_string())
            }
            Err(err) => {
                Log::fatal(&format!("Invalid parser config: {}. Please check if follow json format.", err));
                Err(err)
            }
        }
    }

    /// Saves a key-value pair to a JSON configuration string.
    /// Currently returns the original config as JSON mutation is not fully implemented.
    pub fn save_to_parser_config(config_str: &str, key: &str, value: &str) -> Result<String, String> {
        match Json::parse(config_str, JsonParse::Standard) {
            Ok(config_json) => {
                if !config_json.is_object() {
                    return Err("Config is not a JSON object".to_string());
                }
                // For now, return original config as the JSON implementation doesn't support mutation
                // This would need to be implemented properly in the JSON library
                Ok(config_json.dump())
            }
            Err(err) => {
                Log::fatal(&format!("Invalid parser config: {}. Please check if follow json format.", err));
                Err(err)
            }
        }
    }

    // Numeric parsing functions
    
    /// Parse integer with validation (equivalent to AtoiAndCheck in C++)
    pub fn atoi_and_check(s: &str) -> Option<i32> {
        s.trim().parse::<i32>().ok()
    }
    
    /// Parse double with validation (equivalent to AtofAndCheck in C++)
    pub fn atof_and_check(s: &str) -> Option<f64> {
        s.trim().parse::<f64>().ok()
    }
    

    /// Parses an integer from a string, returning the value and number of characters consumed.
    /// Handles leading whitespace and sign characters.
    pub fn atoi<T>(s: &str) -> Result<(T, usize), std::num::ParseIntError>
    where
        T: FromStr<Err = std::num::ParseIntError> + Default,
    {
        let trimmed = s.trim_start();
        let mut chars = trimmed.chars();
        let mut num_str = String::new();
        let mut sign = 1;
        let mut consumed = 0;
        
        // Skip leading whitespace
        let start_pos = s.len() - trimmed.len();
        
        // Handle sign
        if let Some(first_char) = chars.next() {
            match first_char {
                '-' => {
                    sign = -1;
                    consumed += 1;
                }
                '+' => {
                    consumed += 1;
                }
                c if c.is_ascii_digit() => {
                    num_str.push(c);
                    consumed += 1;
                }
                _ => {
                    return T::from_str("").map(|_| (T::default(), start_pos));
                }
            }
        }
        
        // Parse digits
        for ch in chars {
            if ch.is_ascii_digit() {
                num_str.push(ch);
                consumed += 1;
            } else {
                break;
            }
        }
        
        if num_str.is_empty() {
            return T::from_str("").map(|_| (T::default(), start_pos));
        }
        
        if sign == -1 {
            num_str = format!("-{}", num_str);
        }
        
        T::from_str(&num_str).map(|val| (val, start_pos + consumed))
    }

    /// Parses a floating-point number from a string, returning the value and characters consumed.
    /// Handles special cases like NaN, infinity, and scientific notation.
    pub fn atof(s: &str) -> Result<(f64, usize), String> {
        let trimmed = s.trim_start();
        let start_pos = s.len() - trimmed.len();
        
        // Handle special cases
        let lower_trimmed = trimmed.to_lowercase();
        if lower_trimmed.starts_with("nan") || lower_trimmed.starts_with("na") || lower_trimmed.starts_with("null") {
            return Ok((f64::NAN, start_pos + 3.min(trimmed.len())));
        }
        if lower_trimmed.starts_with("inf") || lower_trimmed.starts_with("infinity") {
            let sign = if trimmed.starts_with('-') { -1.0 } else { 1.0 };
            let consumed = if lower_trimmed.starts_with("infinity") { 8 } else { 3 };
            return Ok((sign * f64::INFINITY, start_pos + consumed));
        }
        
        // Parse number
        let mut chars = trimmed.chars().peekable();
        let mut num_str = String::new();
        let mut consumed = 0;
        
        // Handle sign
        if let Some(&first_char) = chars.peek() {
            if first_char == '+' || first_char == '-' {
                num_str.push(chars.next().unwrap());
                consumed += 1;
            }
        }
        
        // Parse digits, decimal point, and exponent
        let mut has_digits = false;
        while let Some(&ch) = chars.peek() {
            match ch {
                '0'..='9' => {
                    num_str.push(chars.next().unwrap());
                    consumed += 1;
                    has_digits = true;
                }
                '.' => {
                    num_str.push(chars.next().unwrap());
                    consumed += 1;
                }
                'e' | 'E' => {
                    num_str.push(chars.next().unwrap());
                    consumed += 1;
                    // Handle exponent sign
                    if let Some(&exp_sign) = chars.peek() {
                        if exp_sign == '+' || exp_sign == '-' {
                            num_str.push(chars.next().unwrap());
                            consumed += 1;
                        }
                    }
                }
                _ => break,
            }
        }
        
        if !has_digits {
            return Ok((f64::NAN, start_pos));
        }
        
        match num_str.parse::<f64>() {
            Ok(val) => Ok((val, start_pos + consumed)),
            Err(_) => Ok((f64::NAN, start_pos + consumed)),
        }
    }

    /// High-precision floating-point parser (currently delegates to atof).
    /// In a full implementation, this would use specialized high-precision parsing.
    pub fn atof_precise(s: &str) -> Result<(f64, usize), String> {
        // For now, use the same implementation as atof
        // In a full implementation, this would use a high-precision parser
        Self::atof(s)
    }

    /// Validates that a string can be completely parsed as an integer.
    /// Returns true if the entire string represents a valid integer.
    pub fn atoi_and_check_generic<T>(s: &str) -> bool
    where
        T: FromStr<Err = std::num::ParseIntError> + Default,
    {
        match Self::atoi::<T>(s) {
            Ok((_, consumed)) => consumed == s.len(),
            Err(_) => false,
        }
    }

    /// Validates that a string can be completely parsed as a floating-point number.
    /// Returns true if the entire string represents a valid float.
    pub fn atof_and_check_generic(s: &str) -> bool {
        match Self::atof(s) {
            Ok((_, consumed)) => consumed == s.len(),
            Err(_) => false,
        }
    }

    // Skip functions

    /// Removes leading spaces and tab characters from a string.
    pub fn skip_space_and_tab(s: &str) -> &str {
        s.trim_start_matches(&[' ', '\t'][..])
    }

    /// Removes leading newline, carriage return, and space characters from a string.
    pub fn skip_return(s: &str) -> &str {
        s.trim_start_matches(&['\n', '\r', ' '][..])
    }

    // Array utility functions

    /// Converts an array of one type to another using the Into trait.
    pub fn array_cast<T, U>(arr: &[T]) -> Vec<U>
    where
        T: Copy + Into<U>,
    {
        arr.iter().map(|&x| x.into()).collect()
    }

    /// Converts a delimited string to a vector of parsed values.
    /// Uses Default::default() for unparseable values.
    pub fn string_to_array<T>(s: &str, delimiter: char) -> Vec<T>
    where
        T: FromStr + Default,
        T::Err: fmt::Debug,
    {
        Self::split(s, delimiter)
            .into_iter()
            .map(|part| part.parse().unwrap_or_default())
            .collect()
    }

    /// Converts a string with nested bracket-delimited arrays to a vector of vectors.
    /// Parses content between brackets as separate arrays using the specified delimiter.
    pub fn string_to_array_of_arrays<T>(
        s: &str,
        left_bracket: char,
        right_bracket: char,
        delimiter: char,
    ) -> Vec<Vec<T>>
    where
        T: FromStr + Default,
        T::Err: fmt::Debug,
    {
        Self::split_brackets(s, left_bracket, right_bracket)
            .into_iter()
            .map(|part| Self::string_to_array(&part, delimiter))
            .collect()
    }

    /// Converts a space-delimited string to a vector of exactly n elements.
    /// Panics if the number of elements doesn't match n.
    pub fn string_to_array_fixed<T>(s: &str, n: usize) -> Vec<T>
    where
        T: FromStr + Default,
        T::Err: fmt::Debug,
    {
        if n == 0 {
            return Vec::new();
        }
        let parts = Self::split(s, ' ');
        assert_eq!(parts.len(), n, "Expected {} elements, got {}", n, parts.len());
        parts.into_iter().map(|part| part.parse().unwrap_or_default()).collect()
    }

    /// Fast implementation of string to array conversion for exactly n elements.
    /// Currently delegates to string_to_array_fixed.
    pub fn string_to_array_fast<T>(s: &str, n: usize) -> Vec<T>
    where
        T: FromStr + Default,
        T::Err: fmt::Debug,
    {
        // Fast implementation using direct parsing
        Self::string_to_array_fixed(s, n)
    }

    /// Joins array elements into a string with the specified delimiter.
    pub fn join<T: fmt::Display>(arr: &[T], delimiter: &str) -> String {
        if arr.is_empty() {
            return String::new();
        }
        
        let mut result = arr[0].to_string();
        for item in &arr[1..] {
            result.push_str(delimiter);
            result.push_str(&item.to_string());
        }
        result
    }

    /// Joins a range of array elements into a string with the specified delimiter.
    /// Range is clamped to array bounds.
    pub fn join_range<T: fmt::Display>(arr: &[T], start: usize, end: usize, delimiter: &str) -> String {
        if end <= start || start >= arr.len() {
            return String::new();
        }
        
        let actual_start = start.min(arr.len() - 1);
        let actual_end = end.min(arr.len());
        
        if actual_start >= actual_end {
            return String::new();
        }
        
        Self::join(&arr[actual_start..actual_end], delimiter)
    }

    // Mathematical functions

    /// Computes base raised to the power of an integer exponent.
    /// Uses efficient recursive implementation with optimizations for even and divisible-by-3 powers.
    pub fn pow<T>(base: T, power: i32) -> f64
    where
        T: Into<f64> + Copy,
    {
        let base_f64 = base.into();
        if power < 0 {
            1.0 / Self::pow(base_f64, -power)
        } else if power == 0 {
            1.0
        } else if power % 2 == 0 {
            Self::pow(base_f64 * base_f64, power / 2)
        } else if power % 3 == 0 {
            Self::pow(base_f64 * base_f64 * base_f64, power / 3)
        } else {
            base_f64 * Self::pow(base_f64, power - 1)
        }
    }

    /// Rounds up to the next power of 2.
    /// Returns 0 for non-positive input or on overflow.
    pub fn pow2_round_up(x: i64) -> i64 {
        if x <= 0 {
            return 0;
        }
        let mut t = 1i64;
        for _ in 0..64 {
            if t >= x {
                return t;
            }
            t = t.checked_shl(1).unwrap_or(0);
            if t == 0 {
                break;
            }
        }
        0
    }

    /// Applies softmax transformation in-place to the input array.
    /// Uses numerical stability by subtracting the maximum value.
    pub fn softmax(input: &mut [f64]) {
        if input.is_empty() {
            return;
        }
        
        // Find maximum
        let max_val = input.iter().fold(input[0], |max, &x| max.max(x));
        
        // Compute exp(x - max) and sum
        let mut sum = 0.0;
        for val in input.iter_mut() {
            *val = (*val - max_val).exp();
            sum += *val;
        }
        
        // Normalize
        for val in input.iter_mut() {
            *val /= sum;
        }
    }

    /// Applies softmax transformation from input array to output array.
    /// Uses numerical stability by subtracting the maximum value.
    pub fn softmax_copy(input: &[f64], output: &mut [f64]) {
        assert_eq!(input.len(), output.len());
        if input.is_empty() {
            return;
        }
        
        // Find maximum
        let max_val = input.iter().fold(input[0], |max, &x| max.max(x));
        
        // Compute exp(x - max) and sum
        let mut sum = 0.0;
        for (i, &val) in input.iter().enumerate() {
            output[i] = (val - max_val).exp();
            sum += output[i];
        }
        
        // Normalize
        for val in output.iter_mut() {
            *val /= sum;
        }
    }

    /// Clamps f64 values to avoid infinity, replacing NaN with 0.
    /// Values beyond ±1e300 are clamped to that range.
    pub fn avoid_inf_f64(x: f64) -> f64 {
        if x.is_nan() {
            0.0
        } else if x >= 1e300 {
            1e300
        } else if x <= -1e300 {
            -1e300
        } else {
            x
        }
    }

    /// Clamps f32 values to avoid infinity, replacing NaN with 0.
    /// Values beyond ±1e38 are clamped to that range.
    pub fn avoid_inf_f32(x: f32) -> f32 {
        if x.is_nan() {
            0.0
        } else if x >= 1e38 {
            1e38
        } else if x <= -1e38 {
            -1e38
        } else {
            x
        }
    }

    /// Rounds a floating-point number to the nearest integer.
    pub fn round_int(x: f64) -> i32 {
        (x + 0.5) as i32
    }

    /// Returns the sign of a number: 1 for positive, -1 for negative, 0 for zero.
    pub fn sign<T>(x: T) -> i32 
    where
        T: PartialOrd + From<i8>,
    {
        if x > T::from(0) {
            1
        } else if x < T::from(0) {
            -1
        } else {
            0
        }
    }

    /// Computes the natural logarithm safely, returning negative infinity for non-positive values.
    pub fn safe_log<T>(x: T) -> f64
    where
        T: Into<f64> + PartialOrd + From<i8> + Copy,
    {
        if x > T::from(0) {
            x.into().ln()
        } else {
            f64::NEG_INFINITY
        }
    }

    // Bitset functions

    /// Creates an empty bitset capable of holding n bits.
    pub fn empty_bitset(n: usize) -> Vec<u32> {
        let size = (n + 31) / 32;
        vec![0u32; size]
    }

    /// Inserts a value into a bitset, resizing if necessary.
    pub fn insert_bitset<T>(bitset: &mut Vec<u32>, val: T)
    where
        T: Into<usize>,
    {
        let val = val.into();
        let i1 = val / 32;
        let i2 = val % 32;
        
        if bitset.len() <= i1 {
            bitset.resize(i1 + 1, 0);
        }
        
        bitset[i1] |= 1u32 << i2;
    }

    /// Constructs a bitset from an array of values.
    pub fn construct_bitset<T>(vals: &[T]) -> Vec<u32>
    where
        T: Into<usize> + Copy,
    {
        let mut bitset = Vec::new();
        for &val in vals {
            Self::insert_bitset(&mut bitset, val);
        }
        bitset
    }

    /// Checks if a position is set in the bitset.
    pub fn find_in_bitset<T>(bits: &[u32], pos: T) -> bool
    where
        T: Into<usize>,
    {
        let pos = pos.into();
        let i1 = pos / 32;
        if i1 >= bits.len() {
            return false;
        }
        let i2 = pos % 32;
        (bits[i1] >> i2) & 1 != 0
    }

    // Utility functions

    /// Checks if two floating-point numbers are approximately equal using ULP-based comparison.
    pub fn check_double_equal_ordered(a: f64, b: f64) -> bool {
        let upper = f64::from_bits(a.to_bits().wrapping_add(1));
        b <= upper
    }

    /// Gets the next representable floating-point value (ULP + 1).
    pub fn get_double_upper_bound(a: f64) -> f64 {
        f64::from_bits(a.to_bits().wrapping_add(1))
    }

    /// Gets the length of the first line in a string (until null, newline, or carriage return).
    pub fn get_line(s: &str) -> usize {
        s.chars().take_while(|&c| c != '\0' && c != '\n' && c != '\r').count()
    }

    /// Skips past newline characters (\r and \n) at the start of a string.
    pub fn skip_new_line(s: &str) -> &str {
        let mut chars = s.chars();
        if let Some('\r') = chars.as_str().chars().next() {
            chars.next();
        }
        if let Some('\n') = chars.as_str().chars().next() {
            chars.next();
        }
        chars.as_str()
    }

    /// Checks if a string contains only characters allowed in JSON values (no structural characters).
    pub fn check_allowed_json(s: &str) -> bool {
        let forbidden_chars = ['"', ',', ':', '[', ']', '{', '}'];
        !s.chars().any(|c| forbidden_chars.contains(&c))
    }

    // Vector utility functions

    /// Converts a vector of boxed values to a vector of const pointers.
    pub fn const_ptr_in_vector_wrapper<T>(input: &[Box<T>]) -> Vec<*const T> {
        input.iter().map(|boxed| boxed.as_ref() as *const T).collect()
    }

    /// Sorts paired arrays by keys starting from the given index.
    /// If reverse is true, sorts in descending order.
    pub fn sort_for_pair<T1, T2>(keys: &mut [T1], values: &mut [T2], start: usize, reverse: bool)
    where
        T1: Ord + Clone,
        T2: Clone,
    {
        assert_eq!(keys.len(), values.len());
        if start >= keys.len() {
            return;
        }
        
        let mut pairs: Vec<(T1, T2)> = keys[start..]
            .iter()
            .zip(values[start..].iter())
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        
        if reverse {
            pairs.sort_by(|a, b| b.0.cmp(&a.0));
        } else {
            pairs.sort_by(|a, b| a.0.cmp(&b.0));
        }
        
        for (i, (key, value)) in pairs.into_iter().enumerate() {
            keys[start + i] = key;
            values[start + i] = value;
        }
    }

    /// Converts a vector of vectors to a vector of mutable pointers to the inner vectors' data.
    pub fn vector2ptr<T>(data: &mut [Vec<T>]) -> Vec<*mut T> {
        data.iter_mut().map(|vec| vec.as_mut_ptr()).collect()
    }

    /// Returns a vector containing the sizes of each inner vector.
    pub fn vector_size<T>(data: &[Vec<T>]) -> Vec<i32> {
        data.iter().map(|vec| vec.len() as i32).collect()
    }

    // Statistics functions

    /// Computes minimum, maximum, and sum of an array in a single pass.
    /// Uses pairwise comparison optimization for better performance.
    pub fn obtain_min_max_sum<T1, T2>(w: &[T1]) -> (T1, T1, T2)
    where
        T1: Copy + PartialOrd + std::ops::Add<Output = T1>,
        T2: From<T1>,
    {
        if w.is_empty() {
            panic!("Cannot compute min/max/sum of empty array");
        }
        
        let mut min_w = w[0];
        let mut max_w = w[0];
        let mut sum_w = w[0];
        
        let mut i = 1;
        if w.len() % 2 == 0 && w.len() > 1 {
            if w[0] < w[1] {
                min_w = w[0];
                max_w = w[1];
            } else {
                min_w = w[1];
                max_w = w[0];
            }
            sum_w = w[0] + w[1];
            i = 2;
        }
        
        while i + 1 < w.len() {
            let (smaller, larger) = if w[i] < w[i + 1] {
                (w[i], w[i + 1])
            } else {
                (w[i + 1], w[i])
            };
            
            if smaller < min_w {
                min_w = smaller;
            }
            if larger > max_w {
                max_w = larger;
            }
            
            sum_w = sum_w + w[i] + w[i + 1];
            i += 2;
        }
        
        if i < w.len() {
            if w[i] < min_w {
                min_w = w[i];
            }
            if w[i] > max_w {
                max_w = w[i];
            }
            sum_w = sum_w + w[i];
        }
        
        (min_w, max_w, T2::from(sum_w))
    }

    /// Validates that all elements in an array are within the specified closed interval.
    /// Logs fatal error and terminates if any elements are outside the range.
    pub fn check_elements_interval_closed<T>(
        y: &[T],
        y_min: T,
        y_max: T,
        caller_name: &str,
    ) where
        T: PartialOrd + Copy + fmt::Display,
    {
        let mut i = 1;
        while i < y.len() {
            if i == y.len() - 1 {
                // Odd case
                if y[i] < y_min || y[i] > y_max {
                    Log::fatal(&format!(
                        "[{}]: does not tolerate element [#{} = {}] outside [{}, {}]",
                        caller_name, i, y[i], y_min, y_max
                    ));
                }
                break;
            }
            
            let (smaller, larger, smaller_idx, larger_idx) = if y[i - 1] < y[i] {
                (y[i - 1], y[i], i - 1, i)
            } else {
                (y[i], y[i - 1], i, i - 1)
            };
            
            if smaller < y_min {
                Log::fatal(&format!(
                    "[{}]: does not tolerate element [#{} = {}] outside [{}, {}]",
                    caller_name, smaller_idx, smaller, y_min, y_max
                ));
            }
            if larger > y_max {
                Log::fatal(&format!(
                    "[{}]: does not tolerate element [#{} = {}] outside [{}, {}]",
                    caller_name, larger_idx, larger, y_min, y_max
                ));
            }
            
            i += 2;
        }
    }
}

// Timer implementation
/// A timer utility for measuring execution time of different operations.
/// Supports starting/stopping named timers and accumulating durations.
#[derive(Debug)]
pub struct Timer {
    start_times: HashMap<String, Instant>,
    stats: HashMap<String, Duration>,
}

impl Timer {
    /// Creates a new timer instance.
    pub fn new() -> Self {
        Self {
            start_times: HashMap::new(),
            stats: HashMap::new(),
        }
    }

    /// Starts timing for the given operation name.
    pub fn start(&mut self, name: &str) {
        self.start_times.insert(name.to_string(), Instant::now());
    }

    /// Stops timing for the given operation name and accumulates the duration.
    pub fn stop(&mut self, name: &str) {
        if let Some(start_time) = self.start_times.remove(name) {
            let duration = start_time.elapsed();
            *self.stats.entry(name.to_string()).or_insert(Duration::ZERO) += duration;
        }
    }

    /// Prints all accumulated timing statistics.
    pub fn print(&self) {
        for (name, duration) in &self.stats {
            println!("{} costs: {:.6} seconds", name, duration.as_secs_f64());
        }
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        self.print();
    }
}

// Function timer for RAII timing
/// RAII-style function timer that automatically stops timing when dropped.
#[derive(Debug)]
pub struct FunctionTimer {
    name: String,
    timer: *mut Timer,
}

impl FunctionTimer {
    /// Creates a new function timer that starts timing immediately.
    pub fn new(name: &str, timer: &mut Timer) -> Self {
        timer.start(name);
        Self {
            name: name.to_string(),
            timer: timer as *mut Timer,
        }
    }
}

impl Drop for FunctionTimer {
    fn drop(&mut self) {
        unsafe {
            if !self.timer.is_null() {
                (*self.timer).stop(&self.name);
            }
        }
    }
}

lazy_static::lazy_static! {
    /// Global timer instance for application-wide timing measurements.
    pub static ref GLOBAL_TIMER: std::sync::Mutex<Timer> = std::sync::Mutex::new(Timer::new());
}

// CommonC namespace for locale-independent operations
/// Locale-independent common operations, equivalent to C-style functions.
#[derive(Debug)]
pub struct CommonC;

impl CommonC {
    /// Joins array elements into a string with the specified delimiter.
    pub fn join<T: fmt::Display>(arr: &[T], delimiter: &str) -> String {
        Common::join(arr, delimiter)
    }

    /// Joins a range of array elements into a string with the specified delimiter.
    pub fn join_range<T: fmt::Display>(arr: &[T], start: usize, end: usize, delimiter: &str) -> String {
        Common::join_range(arr, start, end, delimiter)
    }

    /// Parses a floating-point number from a string (locale-independent).
    pub fn atof(s: &str) -> Result<(f64, usize), String> {
        Common::atof(s)
    }

    /// Fast string to array conversion for exactly n elements (locale-independent).
    pub fn string_to_array_fast<T>(s: &str, n: usize) -> Vec<T>
    where
        T: FromStr + Default,
        T::Err: fmt::Debug,
    {
        Common::string_to_array_fast(s, n)
    }

    /// Converts a string to array with exactly n elements (locale-independent).
    pub fn string_to_array<T>(s: &str, n: usize) -> Vec<T>
    where
        T: FromStr + Default,
        T::Err: fmt::Debug,
    {
        Common::string_to_array_fixed(s, n)
    }

    /// Converts a delimited string to array (locale-independent).
    pub fn string_to_array_with_delimiter<T>(s: &str, delimiter: char) -> Vec<T>
    where
        T: FromStr + Default,
        T::Err: fmt::Debug,
    {
        Common::string_to_array(s, delimiter)
    }

    /// Converts array elements to string representation with optional high precision.
    pub fn array_to_string<T: fmt::Display>(arr: &[T], n: usize, high_precision: bool) -> String {
        if arr.is_empty() || n == 0 {
            return String::new();
        }
        
        let actual_n = n.min(arr.len());
        if high_precision && std::any::type_name::<T>().contains("f") {
            // High precision for floating point
            arr[..actual_n].iter()
                .map(|x| format!("{:.17}", x))
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            arr[..actual_n].iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tolower() {
        assert_eq!(Common::tolower('A'), 'a');
        assert_eq!(Common::tolower('Z'), 'z');
        assert_eq!(Common::tolower('a'), 'a');
        assert_eq!(Common::tolower('1'), '1');
    }

    #[test]
    fn test_trim() {
        assert_eq!(Common::trim("  hello  ".to_string()), "hello");
        assert_eq!(Common::trim("\t\n hello \r\n".to_string()), "hello");
        assert_eq!(Common::trim("".to_string()), "");
    }

    #[test]
    fn test_split() {
        let result = Common::split("a,b,c", ',');
        assert_eq!(result, vec!["a", "b", "c"]);
        
        let result = Common::split("a,,b", ',');
        assert_eq!(result, vec!["a", "b"]);
    }

    #[test]
    fn test_atoi() {
        let result: Result<(i32, usize), _> = Common::atoi("123");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().0, 123);
        
        let result: Result<(i32, usize), _> = Common::atoi("-456");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().0, -456);
    }

    #[test]
    fn test_atof() {
        let result = Common::atof("123.45");
        assert!(result.is_ok());
        assert!((result.unwrap().0 - 123.45).abs() < 1e-10);
        
        let result = Common::atof("nan");
        assert!(result.is_ok());
        assert!(result.unwrap().0.is_nan());
    }

    #[test]
    fn test_softmax() {
        let mut input = vec![1.0, 2.0, 3.0];
        Common::softmax(&mut input);
        
        let sum: f64 = input.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(input[2] > input[1]);
        assert!(input[1] > input[0]);
    }

    #[test]
    fn test_bitset() {
        let mut bitset = Common::empty_bitset(100);
        Common::insert_bitset(&mut bitset, 5);
        Common::insert_bitset(&mut bitset, 15);
        Common::insert_bitset(&mut bitset, 95);
        
        assert!(Common::find_in_bitset(&bitset, 5));
        assert!(Common::find_in_bitset(&bitset, 15));
        assert!(Common::find_in_bitset(&bitset, 95));
        assert!(!Common::find_in_bitset(&bitset, 6));
    }

    #[test]
    fn test_pow() {
        assert!((Common::pow(2.0, 3) - 8.0).abs() < 1e-10);
        assert!((Common::pow(2.0, -2) - 0.25).abs() < 1e-10);
        assert!((Common::pow(5.0, 0) - 1.0).abs() < 1e-10);
    }
}