use lightgbm_rust::core::utils::common::Common;

fn test_string_functions() {
    println!("\n=== Testing String Functions ===");
    
    // Test tolower
    println!("tolower('A'): {}", Common::tolower('A'));
    println!("tolower('Z'): {}", Common::tolower('Z'));
    println!("tolower('a'): {}", Common::tolower('a'));
    println!("tolower('1'): {}", Common::tolower('1'));
    
    // Test Trim
    println!("Trim('  hello  '): '{}'", Common::trim("  hello  ".to_string()));
    println!("Trim('\\t\\n hello \\r\\n'): '{}'", Common::trim("\t\n hello \r\n".to_string()));
    println!("Trim(''): '{}'", Common::trim("".to_string()));
    
    // Test RemoveQuotationSymbol
    println!("RemoveQuotationSymbol('\"hello\"'): '{}'", Common::remove_quotation_symbol("\"hello\"".to_string()));
    println!("RemoveQuotationSymbol(\"'world'\"): '{}'", Common::remove_quotation_symbol("'world'".to_string()));
    
    // Test StartsWith
    println!("StartsWith('hello world', 'hello'): {}", Common::starts_with("hello world", "hello"));
    println!("StartsWith('hello world', 'world'): {}", Common::starts_with("hello world", "world"));
    
    // Test Split
    let split_result = Common::split("a,b,c", ',');
    print!("Split('a,b,c', ','): ");
    for s in &split_result {
        print!("'{}' ", s);
    }
    println!();
    
    let split_result = Common::split("a,,b", ',');
    print!("Split('a,,b', ','): ");
    for s in &split_result {
        print!("'{}' ", s);
    }
    println!();
    
    // Test SplitLines
    let lines_result = Common::split_lines("line1\nline2\r\nline3");
    print!("SplitLines('line1\\nline2\\r\\nline3'): ");
    for s in &lines_result {
        print!("'{}' ", s);
    }
    println!();
}

fn test_numeric_functions() {
    println!("\n=== Testing Numeric Functions ===");
    
    // Test Atoi
    match Common::atoi::<i32>("123abc") {
        Ok((value, consumed)) => {
            let remaining = &"123abc"[consumed..];
            println!("Atoi('123abc'): value={}, remaining='{}'", value, remaining);
        }
        Err(_) => println!("Atoi('123abc'): failed to parse"),
    }
    
    match Common::atoi::<i32>("-456") {
        Ok((value, consumed)) => {
            let remaining = &"-456"[consumed..];
            println!("Atoi('-456'): value={}, remaining='{}'", value, remaining);
        }
        Err(_) => println!("Atoi('-456'): failed to parse"),
    }
    
    // Test Atof
    match Common::atof("123.45") {
        Ok((value, _)) => println!("Atof('123.45'): value={:.6}", value),
        Err(_) => println!("Atof('123.45'): failed to parse"),
    }
    
    match Common::atof("1.23e-4") {
        Ok((value, _)) => println!("Atof('1.23e-4'): value={:.6e}", value),
        Err(_) => println!("Atof('1.23e-4'): failed to parse"),
    }
    
    match Common::atof("nan") {
        Ok((value, _)) => println!("Atof('nan'): value={}", if value.is_nan() { "NaN".to_string() } else { value.to_string() }),
        Err(_) => println!("Atof('nan'): failed to parse"),
    }
    
    match Common::atof("inf") {
        Ok((value, _)) => println!("Atof('inf'): value={}", if value.is_infinite() { "Inf".to_string() } else { value.to_string() }),
        Err(_) => println!("Atof('inf'): failed to parse"),
    }
    
    // Test AtoiAndCheck
    println!("AtoiAndCheck('123'): {:?}", Common::atoi_and_check("123"));
    println!("AtoiAndCheck('123abc'): {:?}", Common::atoi_and_check("123abc"));
    
    // Test AtofAndCheck
    println!("AtofAndCheck('123.45'): {:?}", Common::atof_and_check("123.45"));
    println!("AtofAndCheck('123.45abc'): {:?}", Common::atof_and_check("123.45abc"));
}

fn test_array_functions() {
    println!("\n=== Testing Array Functions ===");
    
    // Test ArrayCast
    let int_vec = vec![1i32, 2, 3, 4, 5];
    let double_vec: Vec<f64> = Common::array_cast(&int_vec);
    print!("ArrayCast<int, double>([1,2,3,4,5]): ");
    for d in &double_vec {
        print!("{} ", d);
    }
    println!();
    
    // Test StringToArray
    let str_to_array: Vec<i32> = Common::string_to_array("1,2,3,4,5", ',');
    print!("StringToArray<int>('1,2,3,4,5', ','): ");
    for i in &str_to_array {
        print!("{} ", i);
    }
    println!();
    
    // Test Join
    let join_vec = vec![10, 20, 30, 40];
    let joined = Common::join(&join_vec, ", ");
    println!("Join([10,20,30,40], ', '): '{}'", joined);
    
    let joined_range = Common::join_range(&join_vec, 1, 3, " | ");
    println!("Join([10,20,30,40], 1, 3, ' | '): '{}'", joined_range);
}

fn test_math_functions() {
    println!("\n=== Testing Math Functions ===");
    
    // Test Pow
    println!("Pow(2.0, 3): {:.6}", Common::pow(2.0, 3));
    println!("Pow(2.0, -2): {:.6}", Common::pow(2.0, -2));
    println!("Pow(5.0, 0): {:.6}", Common::pow(5.0, 0));
    
    // Test Pow2RoundUp
    println!("Pow2RoundUp(10): {}", Common::pow2_round_up(10));
    println!("Pow2RoundUp(16): {}", Common::pow2_round_up(16));
    println!("Pow2RoundUp(100): {}", Common::pow2_round_up(100));
    
    // Test Softmax
    let mut softmax_vec = vec![1.0, 2.0, 3.0];
    Common::softmax(&mut softmax_vec);
    print!("Softmax([1.0, 2.0, 3.0]): ");
    for d in &softmax_vec {
        print!("{:.6} ", d);
    }
    println!();
    
    // Test AvoidInf
    println!("AvoidInf(1e301): {}", Common::avoid_inf_f64(1e301));
    println!("AvoidInf(-1e301): {}", Common::avoid_inf_f64(-1e301));
    println!("AvoidInf(NaN): {}", Common::avoid_inf_f64(f64::NAN));
    
    // Test RoundInt
    println!("RoundInt(3.7): {}", Common::round_int(3.7));
    println!("RoundInt(3.2): {}", Common::round_int(3.2));
    println!("RoundInt(-2.7): {}", Common::round_int(-2.7));
}

fn test_bitset_functions() {
    println!("\n=== Testing Bitset Functions ===");
    
    // Test EmptyBitset
    let mut bitset = Common::empty_bitset(100);
    println!("EmptyBitset(100) size: {}", bitset.len());
    
    // Test InsertBitset and FindInBitset
    Common::insert_bitset(&mut bitset, 5usize);
    Common::insert_bitset(&mut bitset, 15usize);
    Common::insert_bitset(&mut bitset, 95usize);
    
    println!("After inserting 5, 15, 95:");
    println!("FindInBitset(bitset, 5): {}", Common::find_in_bitset(&bitset, 5usize));
    println!("FindInBitset(bitset, 15): {}", Common::find_in_bitset(&bitset, 15usize));
    println!("FindInBitset(bitset, 95): {}", Common::find_in_bitset(&bitset, 95usize));
    println!("FindInBitset(bitset, 6): {}", Common::find_in_bitset(&bitset, 6usize));
    
    // Test ConstructBitset
    let vals = vec![1usize, 3, 7, 31, 63];
    let constructed_bitset = Common::construct_bitset(&vals);
    println!("ConstructBitset([1,3,7,31,63]) size: {}", constructed_bitset.len());
    println!("Contains 7: {}", Common::find_in_bitset(&constructed_bitset, 7usize));
    println!("Contains 8: {}", Common::find_in_bitset(&constructed_bitset, 8usize));
}

fn test_utility_functions() {
    println!("\n=== Testing Utility Functions ===");
    
    // Test CheckDoubleEqualOrdered
    println!("CheckDoubleEqualOrdered(1.0, 1.0): {}", Common::check_double_equal_ordered(1.0, 1.0));
    println!("CheckDoubleEqualOrdered(1.0, 1.0000000001): {}", Common::check_double_equal_ordered(1.0, 1.0000000001));
    
    // Test GetDoubleUpperBound
    let upper = Common::get_double_upper_bound(1.0);
    println!("GetDoubleUpperBound(1.0): {:.17e}", upper);
    
    // Test GetLine
    println!("GetLine('hello\\nworld'): {}", Common::get_line("hello\nworld"));
    println!("GetLine('hello world'): {}", Common::get_line("hello world"));
    
    // Test CheckAllowedJSON
    println!("CheckAllowedJSON('hello'): {}", Common::check_allowed_json("hello"));
    println!("CheckAllowedJSON('hello{{world}}'): {}", Common::check_allowed_json("hello{world}"));
}

fn test_statistics_functions() {
    println!("\n=== Testing Statistics Functions ===");
    
    // Test ObtainMinMaxSum
    let data = vec![3.5, 1.2, 5.8, 2.1, 4.7];
    let (min_val, max_val, sum_val): (f64, f64, f64) = Common::obtain_min_max_sum(&data);
    
    println!("ObtainMinMaxSum([3.5, 1.2, 5.8, 2.1, 4.7]):");
    println!("  Min: {:.2}", min_val);
    println!("  Max: {:.2}", max_val);
    println!("  Sum: {:.2}", sum_val);
}

fn main() {
    println!("=== Rust LightGBM Common Functions Test ===");
    
    test_string_functions();
    test_numeric_functions();
    test_array_functions();
    test_math_functions();
    test_bitset_functions();
    test_utility_functions();
    test_statistics_functions();
    
    println!("\n=== All tests completed ===");
}