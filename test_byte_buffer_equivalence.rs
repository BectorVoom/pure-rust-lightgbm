/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

use std::fs::File;
use std::io::Write as IoWrite;

use lightgbm_rust::core::utils::byte_buffer::ByteBuffer;
use lightgbm_rust::core::utils::binary_writer::BinaryWriter;

fn main() -> std::io::Result<()> {
    let mut buffer = ByteBuffer::new();
    
    // Test 1: Basic int8_t write (i8 in Rust)
    let int8_val: i8 = 42;
    let int8_bytes = int8_val.to_le_bytes();
    buffer.write(&int8_bytes)?;
    println!("After int8 write - Size: {}", buffer.get_size());
    println!("int8 value at index 0: {}", buffer.get_at(0) as i8);
    
    // Test 2: int16_t write (i16 in Rust)
    let int16_val: i16 = 1337;
    let int16_bytes = int16_val.to_le_bytes();
    buffer.write(&int16_bytes)?;
    println!("After int16 write - Size: {}", buffer.get_size());
    
    // Test 3: int32_t write (i32 in Rust)
    let int32_val: i32 = 123456;
    let int32_bytes = int32_val.to_le_bytes();
    buffer.write(&int32_bytes)?;
    println!("After int32 write - Size: {}", buffer.get_size());
    
    // Test 4: double write (f64 in Rust)
    let double_val: f64 = 3.14159;
    let double_bytes = double_val.to_le_bytes();
    buffer.write(&double_bytes)?;
    println!("After double write - Size: {}", buffer.get_size());
    
    // Test 5: string write
    let str_val = "Hello, World!";
    let str_bytes = str_val.as_bytes();
    buffer.write(str_bytes)?;
    println!("After string write - Size: {}", buffer.get_size());
    
    // Print all bytes for comparison
    print!("All bytes: ");
    for i in 0..buffer.get_size() {
        print!("{} ", buffer.get_at(i));
    }
    println!();
    
    // Test data() method
    let data_ptr = buffer.data();
    unsafe {
        println!("First byte via data(): {}", *data_ptr);
    }
    
    // Test with_initial_size
    let mut reserve_buffer = ByteBuffer::with_initial_size(100);
    reserve_buffer.write(&int8_bytes)?;
    println!("Reserved buffer size after write: {}", reserve_buffer.get_size());
    
    // Save results to file for comparison with C++
    let mut out_file = File::create("rust_byte_buffer_results.txt")?;
    writeln!(out_file, "Size after int8: {}", 1)?;
    writeln!(out_file, "Size after int16: {}", 1 + std::mem::size_of::<i16>())?;
    writeln!(out_file, "Size after int32: {}", 1 + std::mem::size_of::<i16>() + std::mem::size_of::<i32>())?;
    writeln!(out_file, "Size after double: {}", 1 + std::mem::size_of::<i16>() + std::mem::size_of::<i32>() + std::mem::size_of::<f64>())?;
    writeln!(out_file, "Size after string: {}", buffer.get_size())?;
    writeln!(out_file, "First byte: {}", buffer.get_at(0))?;
    writeln!(out_file, "Byte at index 1: {}", buffer.get_at(1))?;
    writeln!(out_file, "Byte at index 2: {}", buffer.get_at(2))?;
    
    // Write all bytes
    write!(out_file, "All bytes: ")?;
    for i in 0..buffer.get_size() {
        write!(out_file, "{} ", buffer.get_at(i))?;
    }
    writeln!(out_file)?;
    
    println!("Rust test completed. Results saved to rust_byte_buffer_results.txt");
    
    Ok(())
}