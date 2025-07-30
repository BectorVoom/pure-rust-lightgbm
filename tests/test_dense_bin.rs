// //! Tests for dense_bin module equivalence with C++ implementation

// use lightgbm_rust::core::types::*;
// use lightgbm_rust::io::dense_bin::*;

// #[test]
// fn test_dense_bin_8bit_creation() {
//     let bin = DenseBin8Bit::new(100);
//     assert_eq!(bin.num_data, 100);
//     assert_eq!(bin.data.len(), 100);
// }

// #[test]
// fn test_dense_bin_4bit_creation() {
//     let bin = DenseBin4Bit::new(100);
//     assert_eq!(bin.num_data, 100);
//     assert_eq!(bin.data.len(), 50); // (100 + 1) / 2
// }

// #[test]
// fn test_data_access() {
//     let mut bin = DenseBin8Bit::new(10);

//     // Test basic data access
//     let test_data = vec![1u8, 5, 3, 7, 2, 9, 4, 6, 8, 0];
//     for (i, &val) in test_data.iter().enumerate() {
//         bin.data[i] = val;
//     }

//     for (i, &expected) in test_data.iter().enumerate() {
//         assert_eq!(bin.data(i as DataSize), expected);
//     }
// }

// #[test]
// fn test_4bit_data_access() {
//     let mut bin = DenseBin4Bit::new(8);

//     // Simulate pushing 4-bit values
//     let test_values = vec![3u8, 7, 2, 5, 9, 1, 4, 6];

//     // Manually set up the data array for 4-bit packing
//     // Values: [3, 7, 2, 5, 9, 1, 4, 6]
//     // Packed: [0x73, 0x52, 0x19, 0x46]  (low nibble first)
//     bin.data[0] = 0x73; // 3 | (7 << 4)
//     bin.data[1] = 0x52; // 2 | (5 << 4)
//     bin.data[2] = 0x19; // 9 | (1 << 4)
//     bin.data[3] = 0x46; // 4 | (6 << 4)

//     // Test data access
//     for (i, &expected) in test_values.iter().enumerate() {
//         assert_eq!(bin.data(i as DataSize), expected);
//     }
// }

// #[test]
// fn test_iterator_functionality() {
//     let mut bin = DenseBin8Bit::new(5);
//     bin.data = vec![1, 3, 0, 2, 4];

//     let iterator = DenseBinIterator::new(&bin, 0, 4, 0);

//     // Test get method with range checking
//     assert_eq!(iterator.get(0), 1); // 1 - 0 + 1 = 2 (with offset)
//     assert_eq!(iterator.get(1), 4); // 3 - 0 + 1 = 4 (with offset)
//     assert_eq!(iterator.get(2), 0); // most_freq_bin (out of range)
//     assert_eq!(iterator.get(3), 3); // 2 - 0 + 1 = 3 (with offset)
//     assert_eq!(iterator.get(4), 5); // 4 - 0 + 1 = 5 (with offset)

//     // Test raw_get method (no range checking)
//     assert_eq!(iterator.raw_get(0), 1);
//     assert_eq!(iterator.raw_get(1), 3);
//     assert_eq!(iterator.raw_get(2), 0);
//     assert_eq!(iterator.raw_get(3), 2);
//     assert_eq!(iterator.raw_get(4), 4);
// }

// #[test]
// fn test_bin_sizes() {
//     let bin8 = DenseBin8Bit::new(100);
//     let bin16 = DenseBin16Bit::new(100);
//     let bin32 = DenseBin32Bit::new(100);

//     assert_eq!(bin8.data.len(), 100);
//     assert_eq!(bin16.data.len(), 100);
//     assert_eq!(bin32.data.len(), 100);

//     // Check memory layout
//     assert_eq!(std::mem::size_of_val(&bin8.data[0]), 1);
//     assert_eq!(std::mem::size_of_val(&bin16.data[0]), 2);
//     assert_eq!(std::mem::size_of_val(&bin32.data[0]), 4);
// }

// #[test]
// fn test_semantic_equivalence_data_access() {
//     // Test that data access patterns match C++ behavior

//     // Test 1: Regular 8-bit access
//     let mut bin = DenseBin8Bit::new(5);
//     bin.data = vec![10, 20, 30, 40, 50];

//     for i in 0..5 {
//         assert_eq!(bin.data(i as DataSize), bin.data[i]);
//     }

//     // Test 2: 4-bit access patterns
//     let mut bin4 = DenseBin4Bit::new(6);
//     // Simulate C++ 4-bit packing: each byte stores two 4-bit values
//     bin4.data = vec![0x21, 0x43, 0x65]; // [1,2], [3,4], [5,6]

//     assert_eq!(bin4.data(0), 1);
//     assert_eq!(bin4.data(1), 2);
//     assert_eq!(bin4.data(2), 3);
//     assert_eq!(bin4.data(3), 4);
//     assert_eq!(bin4.data(4), 5);
//     assert_eq!(bin4.data(5), 6);
// }

// #[test]
// fn test_memory_alignment() {
//     // Test that memory layout follows C++ expectations
//     let bin = DenseBin8Bit::new(100);

//     // Check that data pointer is properly aligned
//     let ptr = bin.data.as_ptr() as usize;
//     assert_eq!(ptr % std::mem::align_of::<u8>(), 0);

//     // For larger types, check alignment as well
//     let bin16 = DenseBin16Bit::new(100);
//     let ptr16 = bin16.data.as_ptr() as usize;
//     assert_eq!(ptr16 % std::mem::align_of::<u16>(), 0);

//     let bin32 = DenseBin32Bit::new(100);
//     let ptr32 = bin32.data.as_ptr() as usize;
//     assert_eq!(ptr32 % std::mem::align_of::<u32>(), 0);
// }

// #[test]
// fn test_find_in_bitset() {
//     let bin = DenseBin8Bit::new(10);

//     // Test bitset operations
//     let bitset = vec![0b00000101u32, 0b00001010u32]; // bits 0, 2, 33, 35 set

//     assert!(bin.find_in_bitset(&bitset, 64, 0));  // bit 0 set
//     assert!(!bin.find_in_bitset(&bitset, 64, 1)); // bit 1 not set
//     assert!(bin.find_in_bitset(&bitset, 64, 2));  // bit 2 set
//     assert!(!bin.find_in_bitset(&bitset, 64, 3)); // bit 3 not set

//     // Test second word
//     assert!(bin.find_in_bitset(&bitset, 64, 33)); // bit 33 set (bit 1 of word 1)
//     assert!(!bin.find_in_bitset(&bitset, 64, 34)); // bit 34 not set
//     assert!(bin.find_in_bitset(&bitset, 64, 35)); // bit 35 set (bit 3 of word 1)

//     // Test out of bounds
//     assert!(!bin.find_in_bitset(&bitset, 64, 64)); // out of bounds
// }

// #[cfg(test)]
// mod cpp_equivalence_tests {
//     use super::*;

//     /// Test that verifies the same input produces the same output
//     /// as the C++ implementation for basic operations
//     #[test]
//     fn test_cpp_equivalent_data_storage() {
//         // This test simulates the exact same operations that would
//         // happen in the C++ DenseBin constructor and data access

//         let num_data = 1000;
//         let mut bin = DenseBin8Bit::new(num_data);

//         // Simulate C++ push operations
//         for i in 0..num_data {
//             let value = (i % 256) as u8;
//             bin.data[i as usize] = value;
//         }

//         // Verify data access matches expected C++ behavior
//         for i in 0..num_data {
//             let expected = (i % 256) as u8;
//             assert_eq!(bin.data(i), expected,
//                       "Data access mismatch at index {}: expected {}, got {}",
//                       i, expected, bin.data(i));
//         }
//     }

//     #[test]
//     fn test_cpp_equivalent_4bit_storage() {
//         // Test 4-bit storage equivalent to C++ implementation
//         let num_data = 100;
//         let mut bin = DenseBin4Bit::new(num_data);

//         // Simulate the C++ 4-bit packing
//         for i in 0..num_data {
//             let value = (i % 16) as u8;
//             let byte_idx = (i >> 1) as usize;
//             let bit_offset = (i & 1) << 2;

//             if byte_idx < bin.data.len() {
//                 if bit_offset == 0 {
//                     bin.data[byte_idx] = (bin.data[byte_idx] & 0xF0) | value;
//                 } else {
//                     bin.data[byte_idx] = (bin.data[byte_idx] & 0x0F) | (value << 4);
//                 }
//             }
//         }

//         // Verify data access
//         for i in 0..num_data {
//             let expected = (i % 16) as u8;
//             assert_eq!(bin.data(i), expected,
//                       "4-bit data access mismatch at index {}: expected {}, got {}",
//                       i, expected, bin.data(i));
//         }
//     }
// }
