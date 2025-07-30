// Comprehensive equivalence tests for JSON11 Rust implementation
// These tests verify that the Rust implementation produces the same results
// as the original C++ json11 implementation

use lightgbm_rust::core::utilsjson11::{Json, JsonParse, JsonType, JsonObject};

#[test]
fn test_basic_value_equivalence() {
    // Test null
    let null_json = Json::parse("null", JsonParse::Standard).unwrap();
    assert!(null_json.is_null());
    assert_eq!(null_json.dump(), "null");
    
    // Test booleans
    let true_json = Json::parse("true", JsonParse::Standard).unwrap();
    assert!(true_json.is_bool());
    assert_eq!(true_json.bool_value(), true);
    assert_eq!(true_json.dump(), "true");
    
    let false_json = Json::parse("false", JsonParse::Standard).unwrap();
    assert!(false_json.is_bool());
    assert_eq!(false_json.bool_value(), false);
    assert_eq!(false_json.dump(), "false");
    
    // Test integers
    let int_json = Json::parse("42", JsonParse::Standard).unwrap();
    assert!(int_json.is_number());
    assert_eq!(int_json.int_value(), 42);
    assert_eq!(int_json.number_value(), 42.0);
    assert_eq!(int_json.dump(), "42");
    
    // Test negative integers
    let neg_int_json = Json::parse("-123", JsonParse::Standard).unwrap();
    assert!(neg_int_json.is_number());
    assert_eq!(neg_int_json.int_value(), -123);
    assert_eq!(neg_int_json.dump(), "-123");
    
    // Test floats
    let float_json = Json::parse("3.14159", JsonParse::Standard).unwrap();
    assert!(float_json.is_number());
    assert!((float_json.number_value() - 3.14159).abs() < f64::EPSILON);
    
    // Test scientific notation
    let sci_json = Json::parse("1.23e10", JsonParse::Standard).unwrap();
    assert!(sci_json.is_number());
    assert!((sci_json.number_value() - 1.23e10).abs() < f64::EPSILON);
}

#[test]
fn test_string_equivalence() {
    // Basic string
    let str_json = Json::parse("\"hello world\"", JsonParse::Standard).unwrap();
    assert!(str_json.is_string());
    assert_eq!(str_json.string_value(), "hello world");
    assert_eq!(str_json.dump(), "\"hello world\"");
    
    // Empty string
    let empty_str = Json::parse("\"\"", JsonParse::Standard).unwrap();
    assert!(empty_str.is_string());
    assert_eq!(empty_str.string_value(), "");
    assert_eq!(empty_str.dump(), "\"\"");
    
    // String with escapes
    let escaped = Json::parse("\"hello\\nworld\\t!\"", JsonParse::Standard).unwrap();
    assert_eq!(escaped.string_value(), "hello\nworld\t!");
    
    // String with quotes
    let quoted = Json::parse("\"Say \\\"hello\\\"\"", JsonParse::Standard).unwrap();
    assert_eq!(quoted.string_value(), "Say \"hello\"");
    
    // String with backslashes
    let backslash = Json::parse("\"path\\\\to\\\\file\"", JsonParse::Standard).unwrap();
    assert_eq!(backslash.string_value(), "path\\to\\file");
}

#[test]
fn test_array_equivalence() {
    // Empty array
    let empty_array = Json::parse("[]", JsonParse::Standard).unwrap();
    assert!(empty_array.is_array());
    assert_eq!(empty_array.array_items().len(), 0);
    assert_eq!(empty_array.dump(), "[]");
    
    // Simple array
    let simple_array = Json::parse("[1, 2, 3]", JsonParse::Standard).unwrap();
    assert!(simple_array.is_array());
    let items = simple_array.array_items();
    assert_eq!(items.len(), 3);
    assert_eq!(items[0].int_value(), 1);
    assert_eq!(items[1].int_value(), 2);
    assert_eq!(items[2].int_value(), 3);
    assert_eq!(simple_array.dump(), "[1, 2, 3]");
    
    // Mixed type array
    let mixed_array = Json::parse("[1, \"hello\", true, null]", JsonParse::Standard).unwrap();
    assert!(mixed_array.is_array());
    let items = mixed_array.array_items();
    assert_eq!(items.len(), 4);
    assert_eq!(items[0].int_value(), 1);
    assert_eq!(items[1].string_value(), "hello");
    assert_eq!(items[2].bool_value(), true);
    assert!(items[3].is_null());
    
    // Nested arrays
    let nested = Json::parse("[[1, 2], [3, 4]]", JsonParse::Standard).unwrap();
    assert!(nested.is_array());
    let outer = nested.array_items();
    assert_eq!(outer.len(), 2);
    assert!(outer[0].is_array());
    assert!(outer[1].is_array());
    assert_eq!(outer[0].array_items()[0].int_value(), 1);
    assert_eq!(outer[1].array_items()[1].int_value(), 4);
}

#[test]
fn test_object_equivalence() {
    // Empty object
    let empty_obj = Json::parse("{}", JsonParse::Standard).unwrap();
    assert!(empty_obj.is_object());
    assert_eq!(empty_obj.object_items().len(), 0);
    assert_eq!(empty_obj.dump(), "{}");
    
    // Simple object
    let simple_obj = Json::parse("{\"name\": \"Alice\", \"age\": 30}", JsonParse::Standard).unwrap();
    assert!(simple_obj.is_object());
    let obj = simple_obj.object_items();
    assert_eq!(obj.len(), 2);
    assert_eq!(obj.get("name").unwrap().string_value(), "Alice");
    assert_eq!(obj.get("age").unwrap().int_value(), 30);
    
    // Object with various types
    let complex_obj = Json::parse(
        "{\"string\": \"value\", \"number\": 42, \"boolean\": true, \"null_val\": null, \"array\": [1, 2, 3]}",
        JsonParse::Standard
    ).unwrap();
    assert!(complex_obj.is_object());
    let obj = complex_obj.object_items();
    assert_eq!(obj.len(), 5);
    assert_eq!(obj.get("string").unwrap().string_value(), "value");
    assert_eq!(obj.get("number").unwrap().int_value(), 42);
    assert_eq!(obj.get("boolean").unwrap().bool_value(), true);
    assert!(obj.get("null_val").unwrap().is_null());
    assert!(obj.get("array").unwrap().is_array());
    
    // Nested objects
    let nested_obj = Json::parse(
        "{\"user\": {\"name\": \"Bob\", \"details\": {\"age\": 25, \"active\": true}}}",
        JsonParse::Standard
    ).unwrap();
    assert!(nested_obj.is_object());
    let user = nested_obj.object_items().get("user").unwrap();
    assert!(user.is_object());
    let details = user.object_items().get("details").unwrap();
    assert!(details.is_object());
    assert_eq!(details.object_items().get("age").unwrap().int_value(), 25);
}

#[test]
fn test_indexing_equivalence() {
    // Array indexing
    let array = Json::parse("[10, 20, 30]", JsonParse::Standard).unwrap();
    assert_eq!(array.array_index(0).int_value(), 10);
    assert_eq!(array.array_index(1).int_value(), 20);
    assert_eq!(array.array_index(2).int_value(), 30);
    
    // Out of bounds indexing returns null (like C++ version)
    assert!(array.array_index(10).is_null());
    
    // Object indexing
    let obj = Json::parse("{\"a\": 1, \"b\": 2}", JsonParse::Standard).unwrap();
    assert_eq!(obj.object_index("a").int_value(), 1);
    assert_eq!(obj.object_index("b").int_value(), 2);
    
    // Non-existent key returns null (like C++ version)
    assert!(obj.object_index("nonexistent").is_null());
    
    // Indexing wrong types returns null
    let number = Json::from(42);
    assert!(number.array_index(0).is_null());
    assert!(number.object_index("key").is_null());
}

#[test]
fn test_comparison_equivalence() {
    // Same type comparisons
    assert_eq!(Json::from(42), Json::from(42));
    assert_ne!(Json::from(42), Json::from(43));
    assert!(Json::from(42) < Json::from(43));
    
    assert_eq!(Json::from("hello"), Json::from("hello"));
    assert_ne!(Json::from("hello"), Json::from("world"));
    assert!(Json::from("hello") < Json::from("world"));
    
    assert_eq!(Json::from(true), Json::from(true));
    assert_ne!(Json::from(true), Json::from(false));
    assert!(Json::from(false) < Json::from(true));
    
    // Different type comparisons (types are ordered as in C++: null < bool < number < string < array < object)
    assert!(Json::null() < Json::from(false));
    assert!(Json::from(true) < Json::from(42));
    assert!(Json::from(42) < Json::from("hello"));
    assert!(Json::from("hello") < Json::from(vec![Json::from(1)]));
    
    let mut obj = JsonObject::new();
    obj.insert("key".to_string(), Json::from("value"));
    assert!(Json::from(vec![Json::from(1)]) < Json::from(obj));
}

#[test]
fn test_shape_validation_equivalence() {
    let obj = Json::parse(
        "{\"name\": \"Alice\", \"age\": 30, \"active\": true, \"scores\": [1, 2, 3]}",
        JsonParse::Standard
    ).unwrap();
    
    // Valid shape
    let valid_shape = vec![
        ("name".to_string(), JsonType::String),
        ("age".to_string(), JsonType::Number),
        ("active".to_string(), JsonType::Bool),
        ("scores".to_string(), JsonType::Array),
    ];
    assert!(obj.has_shape(&valid_shape).is_ok());
    
    // Invalid shape (wrong type)
    let invalid_shape = vec![("name".to_string(), JsonType::Number)];
    assert!(obj.has_shape(&invalid_shape).is_err());
    
    // Shape validation on non-object fails
    let array = Json::from(vec![Json::from(1)]);
    assert!(array.has_shape(&valid_shape).is_err());
}

#[test]
fn test_error_handling_equivalence() {
    // These should all fail parsing (same as C++ version)
    let invalid_cases = vec![
        "invalid",           // Not JSON
        "{key: value}",      // Unquoted keys
        "[1, 2, 3,]",       // Trailing comma
        "{\"key\": }",      // Missing value
        "\"unclosed string", // Unclosed string
        "[1, 2",            // Unclosed array
        "{\"key\": \"val\"", // Unclosed object
        "nul",              // Incomplete keyword
        "tru",              // Incomplete keyword
        "fals",             // Incomplete keyword
        "123.456.789",      // Invalid number
        "123e",             // Invalid scientific notation
        "00123",            // Leading zeros
    ];
    
    for invalid_json in invalid_cases {
        assert!(Json::parse(invalid_json, JsonParse::Standard).is_err(),
                "Should fail to parse: {}", invalid_json);
    }
}

#[test]
fn test_round_trip_equivalence() {
    // Test that parsing and dumping produces equivalent results
    let test_cases = vec![
        "null",
        "true",
        "false",
        "42",
        "-123",
        "3.14159",
        "\"hello world\"",
        "[]",
        "[1, 2, 3]",
        "{}",
        "{\"key\": \"value\"}",
        "{\"name\": \"Alice\", \"age\": 30, \"active\": true}",
        "[{\"id\": 1, \"data\": [1, 2, 3]}, {\"id\": 2, \"data\": [4, 5, 6]}]",
    ];
    
    for test_case in test_cases {
        let parsed = Json::parse(test_case, JsonParse::Standard).unwrap();
        let dumped = parsed.dump();
        let reparsed = Json::parse(&dumped, JsonParse::Standard).unwrap();
        
        // The values should be equivalent even if formatting differs slightly
        assert_eq!(parsed, reparsed, "Round-trip failed for: {}", test_case);
    }
}

#[test]
fn test_number_precision_equivalence() {
    // Test that number precision matches C++ behavior
    let test_numbers = vec![
        "0",
        "1",
        "-1", 
        "123456789",
        "-123456789",
        "3.14159265358979323846",
        "1.23e10",
        "1.23e-10",
        "2.225073858507201e-308", // Close to min double
        "1.7976931348623157e+308", // Close to max double
    ];
    
    for num_str in test_numbers {
        let parsed = Json::parse(num_str, JsonParse::Standard).unwrap();
        assert!(parsed.is_number());
        
        // Verify that the value is preserved correctly
        let original_value: f64 = num_str.parse().unwrap();
        assert!((parsed.number_value() - original_value).abs() < f64::EPSILON * 10.0,
                "Number precision mismatch for: {}", num_str);
    }
}

#[test]
fn test_whitespace_handling_equivalence() {
    // Test various whitespace scenarios (should all parse to the same result)
    let compact = Json::parse("{\"a\":1,\"b\":[2,3]}", JsonParse::Standard).unwrap();
    
    let with_spaces = Json::parse("{ \"a\" : 1 , \"b\" : [ 2 , 3 ] }", JsonParse::Standard).unwrap();
    let with_newlines = Json::parse("{\n  \"a\": 1,\n  \"b\": [\n    2,\n    3\n  ]\n}", JsonParse::Standard).unwrap();
    let with_tabs = Json::parse("{\t\"a\":\t1,\t\"b\":\t[\t2,\t3\t]\t}", JsonParse::Standard).unwrap();
    
    assert_eq!(compact, with_spaces);
    assert_eq!(compact, with_newlines);
    assert_eq!(compact, with_tabs);
}

#[test] 
fn test_unicode_handling_equivalence() {
    // Test basic Unicode support
    let unicode_json = Json::parse("\"Hello \\u0048\\u0065\\u006C\\u006C\\u006F\"", JsonParse::Standard).unwrap();
    assert_eq!(unicode_json.string_value(), "Hello Hello");
    
    // Test that serialization escapes Unicode line separators correctly
    let line_sep = Json::from("Line\u{2028}Separator");
    assert!(line_sep.dump().contains("\\u2028"));
    
    let para_sep = Json::from("Paragraph\u{2029}Separator");
    assert!(para_sep.dump().contains("\\u2029"));
}

#[test]
fn test_large_structure_equivalence() {
    // Test handling of larger nested structures
    let large_json = r#"
    {
        "users": [
            {
                "id": 1,
                "name": "Alice",
                "profile": {
                    "age": 30,
                    "interests": ["reading", "coding", "music"],
                    "settings": {
                        "theme": "dark",
                        "notifications": true,
                        "privacy": {
                            "public": false,
                            "friends_only": true
                        }
                    }
                }
            },
            {
                "id": 2,
                "name": "Bob",
                "profile": {
                    "age": 25,
                    "interests": ["gaming", "sports"],
                    "settings": {
                        "theme": "light",
                        "notifications": false,
                        "privacy": {
                            "public": true,
                            "friends_only": false
                        }
                    }
                }
            }
        ],
        "metadata": {
            "version": "1.0",
            "created": "2023-01-01",
            "total_users": 2
        }
    }
    "#;
    
    let parsed = Json::parse(large_json, JsonParse::Standard).unwrap();
    assert!(parsed.is_object());
    
    let users = parsed.object_index("users");
    assert!(users.is_array());
    assert_eq!(users.array_items().len(), 2);
    
    let first_user = users.array_index(0);
    assert_eq!(first_user.object_index("name").string_value(), "Alice");
    
    let settings = first_user.object_index("profile").object_index("settings");
    assert_eq!(settings.object_index("theme").string_value(), "dark");
    
    let privacy = settings.object_index("privacy");
    assert_eq!(privacy.object_index("public").bool_value(), false);
    
    // Test that the structure can be serialized and reparsed
    let dumped = parsed.dump();
    let reparsed = Json::parse(&dumped, JsonParse::Standard).unwrap();
    assert_eq!(parsed, reparsed);
}