use lightgbm_rust::core::utils::array_args::ArrayArgs;

fn main() {
    println!("=== Rust ArrayArgs Equivalence Tests ===");
    
    // Test case 1: Standard array
    let test1 = vec![1.5, 3.2, 2.1, 4.8, 2.9, 1.1];
    println!("Test 1 - Standard array: {:?}", test1);
    println!("ArgMax: {}", ArrayArgs::arg_max_vec(&test1));
    println!("ArgMin: {}", ArrayArgs::arg_min_vec(&test1));
    println!("CheckAllZero: {}", ArrayArgs::check_all_zero(&test1) as i32);
    println!("CheckAll(0.0): {}", ArrayArgs::check_all(&test1, 0.0) as i32);
    
    // Test MaxK
    let maxk_result = ArrayArgs::max_k(&test1, 3);
    println!("MaxK(3): {:?}", maxk_result);
    println!();
    
    // Test case 2: Array with duplicates at extremes
    let test2 = vec![5.0, 1.0, 3.0, 5.0, 1.0];
    println!("Test 2 - Array with duplicates: {:?}", test2);
    println!("ArgMax: {}", ArrayArgs::arg_max_vec(&test2));
    println!("ArgMin: {}", ArrayArgs::arg_min_vec(&test2));
    println!();
    
    // Test case 3: Single element
    let test3 = vec![42.0];
    println!("Test 3 - Single element: {:?}", test3);
    println!("ArgMax: {}", ArrayArgs::arg_max_vec(&test3));
    println!("ArgMin: {}", ArrayArgs::arg_min_vec(&test3));
    println!();
    
    // Test case 4: All zeros
    let test4 = vec![0.0, 0.0, 0.0];
    println!("Test 4 - All zeros: {:?}", test4);
    println!("CheckAllZero: {}", ArrayArgs::check_all_zero(&test4) as i32);
    println!("CheckAll(0.0): {}", ArrayArgs::check_all(&test4, 0.0) as i32);
    println!("ArgMax: {}", ArrayArgs::arg_max_vec(&test4));
    println!("ArgMin: {}", ArrayArgs::arg_min_vec(&test4));
    println!();
    
    // Test case 5: Empty array behavior
    let test5: Vec<f64> = vec![];
    println!("Test 5 - Empty array: {:?}", test5);
    println!("ArgMax: {}", ArrayArgs::arg_max_vec(&test5));
    println!("ArgMin: {}", ArrayArgs::arg_min_vec(&test5));
    println!("CheckAllZero: {}", ArrayArgs::check_all_zero(&test5) as i32);
    println!();
    
    // Test case 6: MaxK edge cases
    let test6 = vec![5.0, 2.0, 8.0, 1.0, 9.0, 3.0];
    println!("Test 6 - MaxK tests: {:?}", test6);
    
    // k=0
    let maxk_0 = ArrayArgs::max_k(&test6, 0);
    println!("MaxK(0) size: {}", maxk_0.len());
    
    // k negative
    let maxk_neg = ArrayArgs::max_k(&test6, -1);
    println!("MaxK(-1) size: {}", maxk_neg.len());
    
    // k=3
    let maxk_3 = ArrayArgs::max_k(&test6, 3);
    println!("MaxK(3) size: {} values: {:?}", maxk_3.len(), maxk_3);
    
    // k large
    let maxk_large = ArrayArgs::max_k(&test6, 10);
    println!("MaxK(10) size: {}", maxk_large.len());
    println!();
    
    // Test case 7: Slice versions
    let test7 = [1.0, 5.0, 3.0, 2.0, 4.0];
    println!("Test 7 - Slice: {:?}", test7);
    println!("ArgMax (slice): {}", ArrayArgs::arg_max_slice(&test7));
    println!("ArgMin (slice): {}", ArrayArgs::arg_min_slice(&test7));
    println!();
    
    // Test case 8: Assign function
    let mut test8 = Vec::new();
    ArrayArgs::assign(&mut test8, 5, 10);
    let all_fives = test8.iter().all(|&x| x == 5);
    println!("Test 8 - Assign(5, 10) size: {} all_5s: {}", test8.len(), all_fives as i32);
    println!();
    
    // Test case 9: Integer types
    let test9 = vec![10, 20, 15, 5, 25];
    println!("Test 9 - Integer array: {:?}", test9);
    println!("ArgMax: {}", ArrayArgs::arg_max_vec(&test9));
    println!("ArgMin: {}", ArrayArgs::arg_min_vec(&test9));
    
    let maxk_int = ArrayArgs::max_k(&test9, 2);
    println!("MaxK(2) size: {}", maxk_int.len());
    println!();
    
    // Test case 10: Negative values
    let test10 = vec![-5.0, -1.0, -3.0, -2.0];
    println!("Test 10 - Negative values: {:?}", test10);
    println!("ArgMax: {}", ArrayArgs::arg_max_vec(&test10));  // -1.0 at index 1
    println!("ArgMin: {}", ArrayArgs::arg_min_vec(&test10));  // -5.0 at index 0
    println!();
    
    // Test case 11: Mixed positive/negative
    let test11 = vec![-2.0, 3.0, -1.0, 1.0];
    println!("Test 11 - Mixed values: {:?}", test11);
    println!("ArgMax: {}", ArrayArgs::arg_max_vec(&test11));  // 3.0 at index 1
    println!("ArgMin: {}", ArrayArgs::arg_min_vec(&test11));  // -2.0 at index 0
    println!();
    
    println!("=== All Rust tests completed ===");
    
    // Now run validation against expected C++ outputs
    println!("\n=== VALIDATION AGAINST C++ REFERENCE ===");
    
    let mut all_passed = true;
    
    // Validation Test 1
    if ArrayArgs::arg_max_vec(&test1) != 3 {
        println!("FAIL: Test1 ArgMax expected 3, got {}", ArrayArgs::arg_max_vec(&test1));
        all_passed = false;
    }
    if ArrayArgs::arg_min_vec(&test1) != 5 {
        println!("FAIL: Test1 ArgMin expected 5, got {}", ArrayArgs::arg_min_vec(&test1));
        all_passed = false;
    }
    
    // Validation Test 2
    if ArrayArgs::arg_max_vec(&test2) != 0 {
        println!("FAIL: Test2 ArgMax expected 0, got {}", ArrayArgs::arg_max_vec(&test2));
        all_passed = false;
    }
    if ArrayArgs::arg_min_vec(&test2) != 1 {
        println!("FAIL: Test2 ArgMin expected 1, got {}", ArrayArgs::arg_min_vec(&test2));
        all_passed = false;
    }
    
    // Validation Test 3
    if ArrayArgs::arg_max_vec(&test3) != 0 {
        println!("FAIL: Test3 ArgMax expected 0, got {}", ArrayArgs::arg_max_vec(&test3));
        all_passed = false;
    }
    if ArrayArgs::arg_min_vec(&test3) != 0 {
        println!("FAIL: Test3 ArgMin expected 0, got {}", ArrayArgs::arg_min_vec(&test3));
        all_passed = false;
    }
    
    // Validation Test 4
    if !ArrayArgs::check_all_zero(&test4) {
        println!("FAIL: Test4 CheckAllZero expected true");
        all_passed = false;
    }
    if !ArrayArgs::check_all(&test4, 0.0) {
        println!("FAIL: Test4 CheckAll(0.0) expected true");
        all_passed = false;
    }
    
    // Validation Test 5
    if ArrayArgs::arg_max_vec(&test5) != 0 {
        println!("FAIL: Test5 ArgMax expected 0, got {}", ArrayArgs::arg_max_vec(&test5));
        all_passed = false;
    }
    if ArrayArgs::arg_min_vec(&test5) != 0 {
        println!("FAIL: Test5 ArgMin expected 0, got {}", ArrayArgs::arg_min_vec(&test5));
        all_passed = false;
    }
    if !ArrayArgs::check_all_zero(&test5) {
        println!("FAIL: Test5 CheckAllZero expected true");
        all_passed = false;
    }
    
    // Validation Test 6
    if ArrayArgs::max_k(&test6, 0).len() != 0 {
        println!("FAIL: Test6 MaxK(0) size expected 0, got {}", ArrayArgs::max_k(&test6, 0).len());
        all_passed = false;
    }
    if ArrayArgs::max_k(&test6, -1).len() != 0 {
        println!("FAIL: Test6 MaxK(-1) size expected 0, got {}", ArrayArgs::max_k(&test6, -1).len());
        all_passed = false;
    }
    if ArrayArgs::max_k(&test6, 3).len() != 3 {
        println!("FAIL: Test6 MaxK(3) size expected 3, got {}", ArrayArgs::max_k(&test6, 3).len());
        all_passed = false;
    }
    if ArrayArgs::max_k(&test6, 10).len() != 6 {
        println!("FAIL: Test6 MaxK(10) size expected 6, got {}", ArrayArgs::max_k(&test6, 10).len());
        all_passed = false;
    }
    
    // Validation Test 7
    if ArrayArgs::arg_max_slice(&test7) != 1 {
        println!("FAIL: Test7 ArgMax slice expected 1, got {}", ArrayArgs::arg_max_slice(&test7));
        all_passed = false;
    }
    if ArrayArgs::arg_min_slice(&test7) != 0 {
        println!("FAIL: Test7 ArgMin slice expected 0, got {}", ArrayArgs::arg_min_slice(&test7));
        all_passed = false;
    }
    
    // Validation Test 8
    let mut test8_check = Vec::new();
    ArrayArgs::assign(&mut test8_check, 5, 10);
    if test8_check.len() != 10 {
        println!("FAIL: Test8 Assign size expected 10, got {}", test8_check.len());
        all_passed = false;
    }
    if !test8_check.iter().all(|&x| x == 5) {
        println!("FAIL: Test8 Assign not all values are 5");
        all_passed = false;
    }
    
    // Validation Test 9
    if ArrayArgs::arg_max_vec(&test9) != 4 {
        println!("FAIL: Test9 ArgMax expected 4, got {}", ArrayArgs::arg_max_vec(&test9));
        all_passed = false;
    }
    if ArrayArgs::arg_min_vec(&test9) != 3 {
        println!("FAIL: Test9 ArgMin expected 3, got {}", ArrayArgs::arg_min_vec(&test9));
        all_passed = false;
    }
    if ArrayArgs::max_k(&test9, 2).len() != 2 {
        println!("FAIL: Test9 MaxK(2) size expected 2, got {}", ArrayArgs::max_k(&test9, 2).len());
        all_passed = false;
    }
    
    // Validation Test 10
    if ArrayArgs::arg_max_vec(&test10) != 1 {
        println!("FAIL: Test10 ArgMax expected 1, got {}", ArrayArgs::arg_max_vec(&test10));
        all_passed = false;
    }
    if ArrayArgs::arg_min_vec(&test10) != 0 {
        println!("FAIL: Test10 ArgMin expected 0, got {}", ArrayArgs::arg_min_vec(&test10));
        all_passed = false;
    }
    
    // Validation Test 11
    if ArrayArgs::arg_max_vec(&test11) != 1 {
        println!("FAIL: Test11 ArgMax expected 1, got {}", ArrayArgs::arg_max_vec(&test11));
        all_passed = false;
    }
    if ArrayArgs::arg_min_vec(&test11) != 0 {
        println!("FAIL: Test11 ArgMin expected 0, got {}", ArrayArgs::arg_min_vec(&test11));
        all_passed = false;
    }
    
    if all_passed {
        println!("✅ ALL VALIDATION TESTS PASSED - Rust and C++ implementations are equivalent!");
    } else {
        println!("❌ SOME VALIDATION TESTS FAILED - Check implementation differences");
        std::process::exit(1);
    }
}