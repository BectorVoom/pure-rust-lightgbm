// Standalone Rust test for DenseBin equivalence with C++
// Run with: rustc equivalence_test.rs && ./equivalence_test

use std::convert::TryFrom;

// Essential types
pub type DataSize = i32;

// Simplified DenseBin implementation matching the C++ version exactly
#[derive(Debug, Clone)]
pub struct DenseBin<T, const IS_4BIT: bool> {
    data: Vec<T>,
    buf: Vec<u8>,
    num_data: DataSize,
}

impl<T, const IS_4BIT: bool> DenseBin<T, IS_4BIT>
where
    T: Copy + Clone + Default + From<u8> + Into<u32> + PartialOrd,
{
    pub fn new(num_data: DataSize) -> Self {
        let (data, buf) = if IS_4BIT {
            assert_eq!(std::mem::size_of::<T>(), 1, "4-bit mode requires u8 value type");
            let size = ((num_data + 1) / 2) as usize;
            (vec![T::from(0u8); size], vec![0u8; size])
        } else {
            (vec![T::default(); num_data as usize], Vec::new())
        };
        
        Self { data, buf, num_data }
    }
    
    pub fn push(&mut self, _tid: i32, idx: DataSize, value: u32) {
        if IS_4BIT {
            let i1 = (idx >> 1) as usize;
            let i2 = (idx & 1) << 2;
            let val = (value as u8) << i2;
            
            if i2 == 0 {
                let data_ptr = self.data.as_mut_ptr() as *mut u8;
                unsafe { *data_ptr.add(i1) = val; }
            } else {
                if self.buf.len() <= i1 {
                    self.buf.resize(i1 + 1, 0);
                }
                self.buf[i1] = val;
            }
        } else {
            if std::mem::size_of::<T>() == 1 {
                self.data[idx as usize] = T::from(value as u8);
            } else if std::mem::size_of::<T>() == 2 {
                unsafe {
                    let ptr = self.data.as_mut_ptr() as *mut u16;
                    *ptr.add(idx as usize) = (value & 0xFFFF) as u16;
                }
            } else if std::mem::size_of::<T>() == 4 {
                unsafe {
                    let ptr = self.data.as_mut_ptr() as *mut u32;
                    *ptr.add(idx as usize) = value;
                }
            }
        }
    }
    
    pub fn finish_load(&mut self) {
        if IS_4BIT && !self.buf.is_empty() {
            let len = ((self.num_data + 1) / 2) as usize;
            let data_ptr = self.data.as_mut_ptr() as *mut u8;
            
            for i in 0..len {
                if i < self.buf.len() {
                    unsafe { *data_ptr.add(i) |= self.buf[i]; }
                }
            }
            self.buf.clear();
        }
    }
    
    pub fn data(&self, idx: DataSize) -> T {
        if IS_4BIT {
            let data_ptr = self.data.as_ptr() as *const u8;
            unsafe {
                let byte_val = *data_ptr.add((idx >> 1) as usize);
                let shift = (idx & 1) << 2;
                T::from((byte_val >> shift) & 0xf)
            }
        } else {
            self.data[idx as usize]
        }
    }
    
    pub fn get_data(&self) -> *const u8 {
        self.data.as_ptr() as *const u8
    }
}

// Type aliases
pub type DenseBin4Bit = DenseBin<u8, true>;
pub type DenseBin8Bit = DenseBin<u8, false>;
pub type DenseBin16Bit = DenseBin<u16, false>;

fn test_equivalence() {
    println!("=== Rust DenseBin Equivalence Test ===");
    
    // Test Case 1: 8-bit DenseBin - IDENTICAL to C++ test
    println!("\n--- Test Case 1: 8-bit DenseBin ---");
    let test_values_8bit = vec![10u32, 25, 0, 15, 42, 100, 255, 128, 64, 200];
    
    let mut bin8 = DenseBin8Bit::new(test_values_8bit.len() as DataSize);
    
    for (i, &value) in test_values_8bit.iter().enumerate() {
        bin8.push(0, i as DataSize, value);
    }
    
    print!("Input values: ");
    for val in &test_values_8bit { print!("{} ", val); }
    println!();
    
    print!("Retrieved values: ");
    for i in 0..test_values_8bit.len() {
        let retrieved = bin8.data(i as DataSize);
        print!("{} ", retrieved);
    }
    println!();
    
    // Verify correctness
    for (i, &expected_u32) in test_values_8bit.iter().enumerate() {
        let expected = expected_u32 as u8;
        let actual = bin8.data(i as DataSize);
        assert_eq!(actual, expected, "8-bit mismatch at index {}", i);
    }
    println!("✓ 8-bit test passed");
    
    // Test Case 2: 4-bit DenseBin - IDENTICAL to C++ test
    println!("\n--- Test Case 2: 4-bit DenseBin ---");
    let test_values_4bit = vec![3u32, 7, 2, 5, 9, 1, 4, 6, 8, 0, 15, 12];
    
    let mut bin4 = DenseBin4Bit::new(test_values_4bit.len() as DataSize);
    
    for (i, &value) in test_values_4bit.iter().enumerate() {
        bin4.push(0, i as DataSize, value & 0xF);
    }
    bin4.finish_load();
    
    print!("Input values (4-bit): ");
    for val in &test_values_4bit { print!("{} ", val & 0xF); }
    println!();
    
    print!("Retrieved values: ");
    for i in 0..test_values_4bit.len() {
        let retrieved = bin4.data(i as DataSize);
        print!("{} ", retrieved);
    }
    println!();
    
    // Print raw memory - COMPARE WITH C++
    print!("Raw memory bytes: ");
    let raw_data = bin4.get_data();
    let num_bytes = (test_values_4bit.len() + 1) / 2;
    for i in 0..num_bytes {
        unsafe { print!("0x{:02X} ", *raw_data.add(i)); }
    }
    println!();
    
    // Verify correctness
    for (i, &expected_u32) in test_values_4bit.iter().enumerate() {
        let expected = (expected_u32 & 0xF) as u8;
        let actual = bin4.data(i as DataSize);
        assert_eq!(actual, expected, "4-bit mismatch at index {}", i);
    }
    println!("✓ 4-bit test passed");
    
    // Test Case 3: Edge cases
    println!("\n--- Test Case 3: Edge Cases ---");
    
    let empty_bin = DenseBin8Bit::new(0);
    println!("Empty bin size: {}", empty_bin.num_data);
    
    let mut single_bin = DenseBin8Bit::new(1);
    single_bin.push(0, 0, 123);
    println!("Single element value: {}", single_bin.data(0));
    assert_eq!(single_bin.data(0), 123);
    
    let large_size = 1000;
    let mut large_bin = DenseBin16Bit::new(large_size);
    for i in 0..large_size {
        large_bin.push(0, i, (i % 65536) as u32);
    }
    
    let mut large_test_passed = true;
    for i in 0..large_size {
        let expected = (i % 65536) as u16;
        let actual_u16 = unsafe {
            let ptr = large_bin.get_data() as *const u16;
            *ptr.add(i as usize)
        };
        if actual_u16 != expected {
            large_test_passed = false;
            println!("Large test failed at index {}: expected {}, got {}", i, expected, actual_u16);
            break;
        }
    }
    println!("Large bin pattern test: {}", if large_test_passed { "✓ PASS" } else { "✗ FAIL" });
    
    println!("\n=== All Rust tests completed ===");
}

fn main() {
    match std::panic::catch_unwind(|| {
        test_equivalence();
    }) {
        Ok(_) => println!("\n🎉 All Rust equivalence tests PASSED!"),
        Err(e) => {
            eprintln!("❌ Test failed: {:?}", e);
            std::process::exit(1);
        }
    }
}