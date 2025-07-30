// Copyright (c) 2013 Dropbox, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

//! json11
//! 
//! json11 is a tiny JSON library for Rust, providing JSON parsing and
//! serialization.
//! 
//! The core object provided by the library is Json. A Json object
//! represents any JSON value: null, bool, number (int or double), string,
//! array (Vec), or object (std::collections::HashMap).
//! 
//! Json objects act like values: they can be assigned, copied, moved, compared
//! for equality or order, etc. There are also helper methods Json::dump, to
//! serialize a Json to a string, and Json::parse (static) to parse a string
//! as a Json object.

use std::collections::HashMap;
use std::sync::Arc;
use std::fmt;

/// Parse strategy for JSON parsing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonParse {
    /// Standard JSON parsing - strict JSON specification
    Standard,
    /// Extended parsing that allows C-style comments
    Comments,
}

/// JSON value types
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum JsonType {
    /// JSON null value
    Null,
    /// JSON number (integer or floating-point)
    Number,
    /// JSON boolean (true or false)
    Bool,
    /// JSON string
    String,
    /// JSON array
    Array,
    /// JSON object (key-value pairs)
    Object,
}

/// Array and object type aliases
/// Type alias for JSON arrays (vectors of Json values)
pub type JsonArray = Vec<Json>;
/// Type alias for JSON objects (hashmaps with string keys and Json values)
pub type JsonObject = HashMap<String, Json>;

/// Shape type for validation
pub type JsonShape = Vec<(String, JsonType)>;

/// Core JSON value representation
#[derive(Debug, Clone)]
pub struct Json {
    value: Arc<dyn JsonValue>,
}

/// Trait for JSON value types
pub trait JsonValue: Send + Sync + fmt::Debug {
    /// Returns the JSON type of this value
    fn json_type(&self) -> JsonType;
    /// Tests equality with another JSON value
    fn equals(&self, other: &dyn JsonValue) -> bool;
    /// Tests if this value is less than another JSON value
    fn less_than(&self, other: &dyn JsonValue) -> bool;
    /// Serializes this value to a JSON string
    fn dump(&self, out: &mut String);
    
    // Default implementations for type conversion
    /// Returns the numeric value, or 0.0 if not a number
    fn number_value(&self) -> f64 { 0.0 }
    /// Returns the integer value, or 0 if not a number
    fn int_value(&self) -> i32 { 0 }
    /// Returns the boolean value, or false if not a boolean
    fn bool_value(&self) -> bool { false }
    /// Returns the string value, or empty string if not a string
    fn string_value(&self) -> &str { "" }
    /// Returns the array items, or empty array if not an array
    fn array_items(&self) -> &JsonArray { &EMPTY_ARRAY }
    /// Returns the object items, or empty object if not an object
    fn object_items(&self) -> &JsonObject { &EMPTY_OBJECT }
    /// Returns the array element at index i, or null if not an array or index out of bounds
    fn array_index(&self, _i: usize) -> &Json { &NULL_JSON }
    /// Returns the object value for key, or null if not an object or key not found
    fn object_index(&self, _key: &str) -> &Json { &NULL_JSON }
}

lazy_static::lazy_static! {
    static ref EMPTY_ARRAY: JsonArray = Vec::new();
    static ref EMPTY_OBJECT: JsonObject = HashMap::new();
    static ref NULL_JSON: Json = Json::null();
    static ref TRUE_JSON: Json = Json::from(true);
    static ref FALSE_JSON: Json = Json::from(false);
}

// Concrete implementations for each JSON type
#[derive(Debug)]
struct JsonNull;

#[derive(Debug)]
struct JsonNumber(f64);

#[derive(Debug)]
struct JsonInteger(i32);

#[derive(Debug)]
struct JsonBool(bool);

#[derive(Debug)]
struct JsonString(String);

#[derive(Debug)]
struct JsonArrayValue(JsonArray);

#[derive(Debug)]
struct JsonObjectValue(JsonObject);

impl Json {
    // Constructors
    /// Creates a new JSON null value
    pub fn null() -> Self {
        Json {
            value: Arc::new(JsonNull),
        }
    }
    
    /// Creates a new JSON number from an f64 value
    pub fn from_f64(value: f64) -> Self {
        Json {
            value: Arc::new(JsonNumber(value)),
        }
    }
    
    /// Creates a new JSON number from an i32 value
    pub fn from_i32(value: i32) -> Self {
        Json {
            value: Arc::new(JsonInteger(value)),
        }
    }
    
    // Type checking methods
    /// Returns true if this JSON value is null
    pub fn is_null(&self) -> bool {
        self.json_type() == JsonType::Null
    }
    
    /// Returns true if this JSON value is a number
    pub fn is_number(&self) -> bool {
        self.json_type() == JsonType::Number
    }
    
    /// Returns true if this JSON value is a boolean
    pub fn is_bool(&self) -> bool {
        self.json_type() == JsonType::Bool
    }
    
    /// Returns true if this JSON value is a string
    pub fn is_string(&self) -> bool {
        self.json_type() == JsonType::String
    }
    
    /// Returns true if this JSON value is an array
    pub fn is_array(&self) -> bool {
        self.json_type() == JsonType::Array
    }
    
    /// Returns true if this JSON value is an object
    pub fn is_object(&self) -> bool {
        self.json_type() == JsonType::Object
    }
    
    // Accessors
    /// Returns the JSON type of this value
    pub fn json_type(&self) -> JsonType {
        self.value.json_type()
    }
    
    /// Returns the numeric value of this JSON value
    pub fn number_value(&self) -> f64 {
        self.value.number_value()
    }
    
    /// Returns the integer value of this JSON value
    pub fn int_value(&self) -> i32 {
        self.value.int_value()
    }
    
    /// Returns the boolean value of this JSON value
    pub fn bool_value(&self) -> bool {
        self.value.bool_value()
    }
    
    /// Returns the string value of this JSON value
    pub fn string_value(&self) -> &str {
        self.value.string_value()
    }
    
    /// Returns the array items of this JSON value
    pub fn array_items(&self) -> &JsonArray {
        self.value.array_items()
    }
    
    /// Returns the object items of this JSON value
    pub fn object_items(&self) -> &JsonObject {
        self.value.object_items()
    }
    
    // Indexing operators
    /// Returns the array element at the given index
    pub fn array_index(&self, i: usize) -> &Json {
        self.value.array_index(i)
    }
    
    /// Returns the object value for the given key
    pub fn object_index(&self, key: &str) -> &Json {
        self.value.object_index(key)
    }
    
    // Serialization
    /// Serializes this JSON value to a string
    pub fn dump(&self) -> String {
        let mut out = String::new();
        self.value.dump(&mut out);
        out
    }
    
    /// Serializes this JSON value to the given string buffer
    pub fn dump_to(&self, out: &mut String) {
        self.value.dump(out);
    }
}

// From trait implementations for constructors
impl From<()> for Json {
    fn from(_: ()) -> Self {
        Json::null()
    }
}

impl From<f64> for Json {
    fn from(value: f64) -> Self {
        Json::from_f64(value)
    }
}

impl From<i32> for Json {
    fn from(value: i32) -> Self {
        Json::from_i32(value)
    }
}

impl From<bool> for Json {
    fn from(value: bool) -> Self {
        Json {
            value: Arc::new(JsonBool(value)),
        }
    }
}

impl From<String> for Json {
    fn from(value: String) -> Self {
        Json {
            value: Arc::new(JsonString(value)),
        }
    }
}

impl From<&str> for Json {
    fn from(value: &str) -> Self {
        Json {
            value: Arc::new(JsonString(value.to_string())),
        }
    }
}

impl From<JsonArray> for Json {
    fn from(value: JsonArray) -> Self {
        Json {
            value: Arc::new(JsonArrayValue(value)),
        }
    }
}

impl From<JsonObject> for Json {
    fn from(value: JsonObject) -> Self {
        Json {
            value: Arc::new(JsonObjectValue(value)),
        }
    }
}

// JsonValue implementations for each type
impl JsonValue for JsonNull {
    fn json_type(&self) -> JsonType { JsonType::Null }
    fn equals(&self, other: &dyn JsonValue) -> bool {
        other.json_type() == JsonType::Null
    }
    fn less_than(&self, _other: &dyn JsonValue) -> bool { false }
    fn dump(&self, out: &mut String) {
        out.push_str("null");
    }
}

impl JsonValue for JsonNumber {
    fn json_type(&self) -> JsonType { JsonType::Number }
    fn number_value(&self) -> f64 { self.0 }
    fn int_value(&self) -> i32 { self.0 as i32 }
    fn equals(&self, other: &dyn JsonValue) -> bool {
        other.json_type() == JsonType::Number && (self.0 - other.number_value()).abs() < f64::EPSILON
    }
    fn less_than(&self, other: &dyn JsonValue) -> bool {
        other.json_type() == JsonType::Number && self.0 < other.number_value()
    }
    fn dump(&self, out: &mut String) {
        if self.0.is_finite() {
            out.push_str(&format!("{:.17}", self.0));
        } else {
            out.push_str("null");
        }
    }
}

impl JsonValue for JsonInteger {
    fn json_type(&self) -> JsonType { JsonType::Number }
    fn number_value(&self) -> f64 { self.0 as f64 }
    fn int_value(&self) -> i32 { self.0 }
    fn equals(&self, other: &dyn JsonValue) -> bool {
        other.json_type() == JsonType::Number && (self.0 as f64 - other.number_value()).abs() < f64::EPSILON
    }
    fn less_than(&self, other: &dyn JsonValue) -> bool {
        other.json_type() == JsonType::Number && (self.0 as f64) < other.number_value()
    }
    fn dump(&self, out: &mut String) {
        out.push_str(&self.0.to_string());
    }
}

impl JsonValue for JsonBool {
    fn json_type(&self) -> JsonType { JsonType::Bool }
    fn bool_value(&self) -> bool { self.0 }
    fn equals(&self, other: &dyn JsonValue) -> bool {
        other.json_type() == JsonType::Bool && self.0 == other.bool_value()
    }
    fn less_than(&self, other: &dyn JsonValue) -> bool {
        other.json_type() == JsonType::Bool && !self.0 && other.bool_value()
    }
    fn dump(&self, out: &mut String) {
        out.push_str(if self.0 { "true" } else { "false" });
    }
}

impl JsonValue for JsonString {
    fn json_type(&self) -> JsonType { JsonType::String }
    fn string_value(&self) -> &str { &self.0 }
    fn equals(&self, other: &dyn JsonValue) -> bool {
        other.json_type() == JsonType::String && self.0 == other.string_value()
    }
    fn less_than(&self, other: &dyn JsonValue) -> bool {
        other.json_type() == JsonType::String && self.0.as_str() < other.string_value()
    }
    fn dump(&self, out: &mut String) {
        out.push('"');
        for ch in self.0.chars() {
            match ch {
                '\\' => out.push_str("\\\\"),
                '"' => out.push_str("\\\""),
                '\u{08}' => out.push_str("\\b"),
                '\u{0C}' => out.push_str("\\f"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) <= 0x1f => {
                    out.push_str(&format!("\\u{:04x}", c as u32));
                }
                '\u{2028}' => out.push_str("\\u2028"),
                '\u{2029}' => out.push_str("\\u2029"),
                _ => out.push(ch),
            }
        }
        out.push('"');
    }
}

impl JsonValue for JsonArrayValue {
    fn json_type(&self) -> JsonType { JsonType::Array }
    fn array_items(&self) -> &JsonArray { &self.0 }
    fn equals(&self, other: &dyn JsonValue) -> bool {
        if other.json_type() != JsonType::Array {
            return false;
        }
        let other_array = other.array_items();
        if self.0.len() != other_array.len() {
            return false;
        }
        self.0.iter().zip(other_array.iter()).all(|(a, b)| a == b)
    }
    fn less_than(&self, other: &dyn JsonValue) -> bool {
        if other.json_type() != JsonType::Array {
            return false;
        }
        self.0 < *other.array_items()
    }
    fn array_index(&self, i: usize) -> &Json {
        self.0.get(i).unwrap_or(&NULL_JSON)
    }
    fn dump(&self, out: &mut String) {
        out.push('[');
        for (i, item) in self.0.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            item.dump_to(out);
        }
        out.push(']');
    }
}

impl JsonValue for JsonObjectValue {
    fn json_type(&self) -> JsonType { JsonType::Object }
    fn object_items(&self) -> &JsonObject { &self.0 }
    fn equals(&self, other: &dyn JsonValue) -> bool {
        if other.json_type() != JsonType::Object {
            return false;
        }
        let other_object = other.object_items();
        if self.0.len() != other_object.len() {
            return false;
        }
        self.0.iter().all(|(k, v)| {
            other_object.get(k).map_or(false, |other_v| v == other_v)
        })
    }
    fn less_than(&self, other: &dyn JsonValue) -> bool {
        if other.json_type() != JsonType::Object {
            return false;
        }
        // Compare lexicographically by sorted keys and values
        let mut self_items: Vec<_> = self.0.iter().collect();
        let mut other_items: Vec<_> = other.object_items().iter().collect();
        self_items.sort_by_key(|(k, _)| *k);
        other_items.sort_by_key(|(k, _)| *k);
        self_items < other_items
    }
    fn object_index(&self, key: &str) -> &Json {
        self.0.get(key).unwrap_or(&NULL_JSON)
    }
    fn dump(&self, out: &mut String) {
        out.push('{');
        for (i, (key, value)) in self.0.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            Json::from(key.clone()).dump_to(out);
            out.push_str(": ");
            value.dump_to(out);
        }
        out.push('}');
    }
}

// Comparison operators for Json
impl PartialEq for Json {
    fn eq(&self, other: &Self) -> bool {
        if Arc::ptr_eq(&self.value, &other.value) {
            return true;
        }
        if self.json_type() != other.json_type() {
            return false;
        }
        self.value.equals(other.value.as_ref())
    }
}

impl Eq for Json {}

impl PartialOrd for Json {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Json {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if Arc::ptr_eq(&self.value, &other.value) {
            return std::cmp::Ordering::Equal;
        }
        match self.json_type().cmp(&other.json_type()) {
            std::cmp::Ordering::Equal => {
                if self.value.less_than(other.value.as_ref()) {
                    std::cmp::Ordering::Less
                } else if other.value.less_than(self.value.as_ref()) {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            }
            other_ordering => other_ordering,
        }
    }
}

impl Json {
    // Parsing methods (to be implemented)
    /// Parses a JSON string and returns a Json value
    pub fn parse(input: &str, strategy: JsonParse) -> Result<Json, String> {
        JsonParser::new(input, strategy).parse()
    }
    
    /// Parses multiple JSON values from a string
    pub fn parse_multi(input: &str, strategy: JsonParse) -> Result<(Vec<Json>, usize), String> {
        JsonParser::new(input, strategy).parse_multi()
    }
    
    // Shape validation
    /// Validates that this JSON object has the expected shape (field types)
    pub fn has_shape(&self, shape: &JsonShape) -> Result<(), String> {
        if !self.is_object() {
            return Err(format!("Expected JSON object, got {}", self.dump()));
        }
        
        for (key, expected_type) in shape {
            let value = self.object_index(key);
            if value.json_type() != *expected_type {
                return Err(format!("Bad type for {} in {}", key, self.dump()));
            }
        }
        
        Ok(())
    }
}

// JSON Parser implementation
struct JsonParser {
    input: Vec<char>,
    pos: usize,
    strategy: JsonParse,
}

impl JsonParser {
    fn new(input: &str, strategy: JsonParse) -> Self {
        JsonParser {
            input: input.chars().collect(),
            pos: 0,
            strategy,
        }
    }
    
    fn parse(&mut self) -> Result<Json, String> {
        let result = self.parse_value(0)?;
        self.consume_whitespace_and_comments();
        if self.pos != self.input.len() {
            return Err(format!("Unexpected trailing character: {:?}", self.input[self.pos]));
        }
        Ok(result)
    }
    
    fn parse_multi(&mut self) -> Result<(Vec<Json>, usize), String> {
        let mut results = Vec::new();
        let mut last_pos = 0;
        
        while self.pos < self.input.len() {
            results.push(self.parse_value(0)?);
            self.consume_whitespace_and_comments();
            last_pos = self.pos;
        }
        
        Ok((results, last_pos))
    }
    
    fn parse_value(&mut self, depth: usize) -> Result<Json, String> {
        const MAX_DEPTH: usize = 200;
        if depth > MAX_DEPTH {
            return Err("Exceeded maximum nesting depth".to_string());
        }
        
        let ch = self.get_next_token()?;
        
        match ch {
            't' => self.expect_keyword("true", Json::from(true)),
            'f' => self.expect_keyword("false", Json::from(false)),
            'n' => self.expect_keyword("null", Json::null()),
            '"' => Ok(Json::from(self.parse_string()?)),
            '[' => self.parse_array(depth),
            '{' => self.parse_object(depth),
            '-' | '0'..='9' => {
                self.pos -= 1;
                self.parse_number()
            }
            _ => Err(format!("Unexpected character: {:?}", ch)),
        }
    }
    
    fn consume_whitespace_and_comments(&mut self) {
        self.consume_whitespace();
        if self.strategy == JsonParse::Comments {
            loop {
                let comment_found = self.consume_comment();
                if !comment_found {
                    break;
                }
                self.consume_whitespace();
            }
        }
    }
    
    fn consume_whitespace(&mut self) {
        while self.pos < self.input.len() {
            match self.input[self.pos] {
                ' ' | '\r' | '\n' | '\t' => self.pos += 1,
                _ => break,
            }
        }
    }
    
    fn consume_comment(&mut self) -> bool {
        if self.pos >= self.input.len() || self.input[self.pos] != '/' {
            return false;
        }
        
        self.pos += 1;
        if self.pos >= self.input.len() {
            return false;
        }
        
        match self.input[self.pos] {
            '/' => {
                // Single line comment
                self.pos += 1;
                while self.pos < self.input.len() && self.input[self.pos] != '\n' {
                    self.pos += 1;
                }
                true
            }
            '*' => {
                // Multi-line comment
                self.pos += 1;
                while self.pos + 1 < self.input.len() {
                    if self.input[self.pos] == '*' && self.input[self.pos + 1] == '/' {
                        self.pos += 2;
                        return true;
                    }
                    self.pos += 1;
                }
                false // Unterminated comment
            }
            _ => {
                self.pos -= 1; // Back up
                false
            }
        }
    }
    
    fn get_next_token(&mut self) -> Result<char, String> {
        self.consume_whitespace_and_comments();
        if self.pos >= self.input.len() {
            return Err("Unexpected end of input".to_string());
        }
        let ch = self.input[self.pos];
        self.pos += 1;
        Ok(ch)
    }
    
    fn expect_keyword(&mut self, keyword: &str, value: Json) -> Result<Json, String> {
        // Move back one position since we already consumed the first character
        self.pos -= 1;
        
        if self.pos + keyword.len() > self.input.len() {
            return Err(format!("Unexpected end of input while parsing {}", keyword));
        }
        
        let slice: String = self.input[self.pos..self.pos + keyword.len()].iter().collect();
        if slice == keyword {
            self.pos += keyword.len();
            Ok(value)
        } else {
            Err(format!("Expected {}, got {}", keyword, slice))
        }
    }
    
    fn parse_string(&mut self) -> Result<String, String> {
        let mut result = String::new();
        let mut last_escaped_codepoint: Option<u32> = None;
        
        while self.pos < self.input.len() {
            let ch = self.input[self.pos];
            self.pos += 1;
            
            if ch == '"' {
                if let Some(codepoint) = last_escaped_codepoint {
                    self.encode_utf8(codepoint, &mut result);
                }
                return Ok(result);
            }
            
            if (ch as u32) <= 0x1f {
                return Err(format!("Unescaped control character in string: {:?}", ch));
            }
            
            if ch != '\\' {
                if let Some(codepoint) = last_escaped_codepoint {
                    self.encode_utf8(codepoint, &mut result);
                    last_escaped_codepoint = None;
                }
                result.push(ch);
                continue;
            }
            
            // Handle escape sequences
            if self.pos >= self.input.len() {
                return Err("Unexpected end of input in string".to_string());
            }
            
            let escape_ch = self.input[self.pos];
            self.pos += 1;
            
            match escape_ch {
                'u' => {
                    // Unicode escape sequence
                    if self.pos + 4 > self.input.len() {
                        return Err("Incomplete Unicode escape sequence".to_string());
                    }
                    
                    let hex_chars: String = self.input[self.pos..self.pos + 4].iter().collect();
                    let codepoint = u32::from_str_radix(&hex_chars, 16)
                        .map_err(|_| format!("Invalid Unicode escape: \\u{}", hex_chars))?;
                    
                    // Handle surrogate pairs
                    if let Some(high) = last_escaped_codepoint {
                        if (0xD800..=0xDBFF).contains(&high) && (0xDC00..=0xDFFF).contains(&codepoint) {
                            // Valid surrogate pair
                            let combined = 0x10000 + ((high - 0xD800) << 10) + (codepoint - 0xDC00);
                            self.encode_utf8(combined, &mut result);
                            last_escaped_codepoint = None;
                        } else {
                            self.encode_utf8(high, &mut result);
                            last_escaped_codepoint = Some(codepoint);
                        }
                    } else {
                        last_escaped_codepoint = Some(codepoint);
                    }
                    
                    self.pos += 4;
                }
                'b' => {
                    if let Some(codepoint) = last_escaped_codepoint {
                        self.encode_utf8(codepoint, &mut result);
                        last_escaped_codepoint = None;
                    }
                    result.push('\u{08}');
                }
                'f' => {
                    if let Some(codepoint) = last_escaped_codepoint {
                        self.encode_utf8(codepoint, &mut result);
                        last_escaped_codepoint = None;
                    }
                    result.push('\u{0C}');
                }
                'n' => {
                    if let Some(codepoint) = last_escaped_codepoint {
                        self.encode_utf8(codepoint, &mut result);
                        last_escaped_codepoint = None;
                    }
                    result.push('\n');
                }
                'r' => {
                    if let Some(codepoint) = last_escaped_codepoint {
                        self.encode_utf8(codepoint, &mut result);
                        last_escaped_codepoint = None;
                    }
                    result.push('\r');
                }
                't' => {
                    if let Some(codepoint) = last_escaped_codepoint {
                        self.encode_utf8(codepoint, &mut result);
                        last_escaped_codepoint = None;
                    }
                    result.push('\t');
                }
                '"' | '\\' | '/' => {
                    if let Some(codepoint) = last_escaped_codepoint {
                        self.encode_utf8(codepoint, &mut result);
                        last_escaped_codepoint = None;
                    }
                    result.push(escape_ch);
                }
                _ => {
                    return Err(format!("Invalid escape character: \\{}", escape_ch));
                }
            }
        }
        
        Err("Unexpected end of input in string".to_string())
    }
    
    fn encode_utf8(&self, codepoint: u32, result: &mut String) {
        if let Some(ch) = char::from_u32(codepoint) {
            result.push(ch);
        }
    }
    
    fn parse_array(&mut self, depth: usize) -> Result<Json, String> {
        let mut array = Vec::new();
        
        let first_ch = self.get_next_token()?;
        if first_ch == ']' {
            return Ok(Json::from(array));
        }
        
        // Put back the character and parse first element
        self.pos -= 1;
        
        loop {
            array.push(self.parse_value(depth + 1)?);
            
            let ch = self.get_next_token()?;
            if ch == ']' {
                break;
            } else if ch == ',' {
                // Continue to next element
                continue;
            } else {
                return Err(format!("Expected ',' or ']' in array, got '{}'", ch));
            }
        }
        
        Ok(Json::from(array))
    }
    
    fn parse_object(&mut self, depth: usize) -> Result<Json, String> {
        let mut object = JsonObject::new();
        
        let first_ch = self.get_next_token()?;
        if first_ch == '}' {
            return Ok(Json::from(object));
        }
        
        // Put back the character for first key
        self.pos -= 1;
        
        loop {
            let key_ch = self.get_next_token()?;
            if key_ch != '"' {
                return Err(format!("Expected string key in object, got '{}'", key_ch));
            }
            
            let key = self.parse_string()?;
            
            let colon_ch = self.get_next_token()?;
            if colon_ch != ':' {
                return Err(format!("Expected ':' after key in object, got '{}'", colon_ch));
            }
            
            let value = self.parse_value(depth + 1)?;
            object.insert(key, value);
            
            let ch = self.get_next_token()?;
            if ch == '}' {
                break;
            } else if ch == ',' {
                // Continue to next key-value pair
                continue;
            } else {
                return Err(format!("Expected ',' or '}}' in object, got '{}'", ch));
            }
        }
        
        Ok(Json::from(object))
    }
    
    fn parse_number(&mut self) -> Result<Json, String> {
        let start_pos = self.pos;
        
        // Handle negative sign
        if self.pos < self.input.len() && self.input[self.pos] == '-' {
            self.pos += 1;
        }
        
        if self.pos >= self.input.len() {
            return Err("Incomplete number".to_string());
        }
        
        // Integer part
        if self.input[self.pos] == '0' {
            self.pos += 1;
            if self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                return Err("Leading zeros not permitted in numbers".to_string());
            }
        } else if self.input[self.pos].is_ascii_digit() {
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        } else {
            return Err(format!("Invalid number character: '{}'", self.input[self.pos]));
        }
        
        let mut is_integer = true;
        
        // Decimal part
        if self.pos < self.input.len() && self.input[self.pos] == '.' {
            is_integer = false;
            self.pos += 1;
            if self.pos >= self.input.len() || !self.input[self.pos].is_ascii_digit() {
                return Err("At least one digit required after decimal point".to_string());
            }
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        
        // Exponent part
        if self.pos < self.input.len() && (self.input[self.pos] == 'e' || self.input[self.pos] == 'E') {
            is_integer = false;
            self.pos += 1;
            
            if self.pos < self.input.len() && (self.input[self.pos] == '+' || self.input[self.pos] == '-') {
                self.pos += 1;
            }
            
            if self.pos >= self.input.len() || !self.input[self.pos].is_ascii_digit() {
                return Err("At least one digit required in exponent".to_string());
            }
            
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        
        let number_str: String = self.input[start_pos..self.pos].iter().collect();
        
        if is_integer && (self.pos - start_pos) <= 10 {
            // Try to parse as integer first
            if let Ok(int_val) = number_str.parse::<i32>() {
                return Ok(Json::from(int_val));
            }
        }
        
        // Parse as float
        let float_val = number_str.parse::<f64>()
            .map_err(|_| format!("Invalid number: {}", number_str))?;
        Ok(Json::from(float_val))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_creation() {
        let null_json = Json::null();
        assert!(null_json.is_null());
        assert_eq!(null_json.dump(), "null");
    }
    
    #[test]
    fn test_boolean_creation() {
        let true_json = Json::from(true);
        let false_json = Json::from(false);
        
        assert!(true_json.is_bool());
        assert!(false_json.is_bool());
        assert_eq!(true_json.bool_value(), true);
        assert_eq!(false_json.bool_value(), false);
        assert_eq!(true_json.dump(), "true");
        assert_eq!(false_json.dump(), "false");
    }
    
    #[test]
    fn test_number_creation() {
        let int_json = Json::from(42);
        let float_json = Json::from(3.14);
        
        assert!(int_json.is_number());
        assert!(float_json.is_number());
        assert_eq!(int_json.int_value(), 42);
        assert_eq!(float_json.number_value(), 3.14);
    }
    
    #[test]
    fn test_string_creation() {
        let str_json = Json::from("hello");
        let string_json = Json::from("world".to_string());
        
        assert!(str_json.is_string());
        assert!(string_json.is_string());
        assert_eq!(str_json.string_value(), "hello");
        assert_eq!(string_json.string_value(), "world");
    }
    
    #[test]
    fn test_comparison() {
        let a = Json::from(42);
        let b = Json::from(42);
        let c = Json::from(43);
        
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a < c);
    }
    
    #[test]
    fn test_parsing_basic() {
        // Test null
        let null_json = Json::parse("null", JsonParse::Standard).unwrap();
        assert!(null_json.is_null());
        
        // Test boolean
        let true_json = Json::parse("true", JsonParse::Standard).unwrap();
        assert!(true_json.is_bool());
        assert_eq!(true_json.bool_value(), true);
        
        let false_json = Json::parse("false", JsonParse::Standard).unwrap();
        assert!(false_json.is_bool());
        assert_eq!(false_json.bool_value(), false);
        
        // Test numbers
        let int_json = Json::parse("42", JsonParse::Standard).unwrap();
        assert!(int_json.is_number());
        assert_eq!(int_json.int_value(), 42);
        
        let float_json = Json::parse("3.14", JsonParse::Standard).unwrap();
        assert!(float_json.is_number());
        assert!((float_json.number_value() - 3.14).abs() < f64::EPSILON);
        
        // Test string
        let str_json = Json::parse("\"hello\"", JsonParse::Standard).unwrap();
        assert!(str_json.is_string());
        assert_eq!(str_json.string_value(), "hello");
    }
    
    #[test]
    fn test_parsing_array() {
        let array_json = Json::parse("[1, 2, 3]", JsonParse::Standard).unwrap();
        assert!(array_json.is_array());
        
        let array = array_json.array_items();
        assert_eq!(array.len(), 3);
        assert_eq!(array[0].int_value(), 1);
        assert_eq!(array[1].int_value(), 2);
        assert_eq!(array[2].int_value(), 3);
        
        // Test empty array
        let empty_array = Json::parse("[]", JsonParse::Standard).unwrap();
        assert!(empty_array.is_array());
        assert_eq!(empty_array.array_items().len(), 0);
    }
    
    #[test]
    fn test_parsing_object() {
        let obj_json = Json::parse("{\"key\": \"value\", \"num\": 42}", JsonParse::Standard).unwrap();
        assert!(obj_json.is_object());
        
        let obj = obj_json.object_items();
        assert_eq!(obj.len(), 2);
        assert_eq!(obj.get("key").unwrap().string_value(), "value");
        assert_eq!(obj.get("num").unwrap().int_value(), 42);
        
        // Test empty object
        let empty_obj = Json::parse("{}", JsonParse::Standard).unwrap();
        assert!(empty_obj.is_object());
        assert_eq!(empty_obj.object_items().len(), 0);
    }
    
    #[test]
    fn test_serialization() {
        // Test basic types
        assert_eq!(Json::null().dump(), "null");
        assert_eq!(Json::from(true).dump(), "true");
        assert_eq!(Json::from(false).dump(), "false");
        assert_eq!(Json::from(42).dump(), "42");
        assert_eq!(Json::from("hello").dump(), "\"hello\"");
        
        // Test arrays
        let array = Json::from(vec![Json::from(1), Json::from(2), Json::from(3)]);
        assert_eq!(array.dump(), "[1, 2, 3]");
        
        // Test objects
        let mut object = JsonObject::new();
        object.insert("key".to_string(), Json::from("value"));
        object.insert("num".to_string(), Json::from(42));
        let obj_json = Json::from(object);
        let dump = obj_json.dump();
        // Order may vary in HashMap, so check both possibilities
        assert!(dump == "{\"key\": \"value\", \"num\": 42}" || 
                dump == "{\"num\": 42, \"key\": \"value\"}");
    }
    
    #[test]
    fn test_indexing() {
        let array_json = Json::parse("[10, 20, 30]", JsonParse::Standard).unwrap();
        assert_eq!(array_json.array_index(0).int_value(), 10);
        assert_eq!(array_json.array_index(1).int_value(), 20);
        assert_eq!(array_json.array_index(2).int_value(), 30);
        
        // Out of bounds should return null
        assert!(array_json.array_index(5).is_null());
        
        let obj_json = Json::parse("{\"a\": 1, \"b\": 2}", JsonParse::Standard).unwrap();
        assert_eq!(obj_json.object_index("a").int_value(), 1);
        assert_eq!(obj_json.object_index("b").int_value(), 2);
        
        // Non-existent key should return null
        assert!(obj_json.object_index("c").is_null());
    }
    
    #[test]
    fn test_string_escaping() {
        let escaped_json = Json::parse("\"hello\\nworld\\t!\"", JsonParse::Standard).unwrap();
        assert_eq!(escaped_json.string_value(), "hello\nworld\t!");
        
        let quote_json = Json::parse("\"Say \\\"hello\\\"\"", JsonParse::Standard).unwrap();
        assert_eq!(quote_json.string_value(), "Say \"hello\"");
    }
    
    #[test]
    fn test_error_handling() {
        // Invalid JSON should return errors
        assert!(Json::parse("invalid", JsonParse::Standard).is_err());
        assert!(Json::parse("{key: value}", JsonParse::Standard).is_err()); // Missing quotes
        assert!(Json::parse("[1, 2,]", JsonParse::Standard).is_err()); // Trailing comma
        assert!(Json::parse("{\"key\"}", JsonParse::Standard).is_err()); // Missing value
    }
    
    #[test]
    fn test_has_shape() {
        let obj_json = Json::parse("{\"name\": \"Alice\", \"age\": 30, \"active\": true}", JsonParse::Standard).unwrap();
        
        let shape = vec![
            ("name".to_string(), JsonType::String),
            ("age".to_string(), JsonType::Number),
            ("active".to_string(), JsonType::Bool),
        ];
        
        assert!(obj_json.has_shape(&shape).is_ok());
        
        // Wrong type should fail
        let bad_shape = vec![("name".to_string(), JsonType::Number)];
        assert!(obj_json.has_shape(&bad_shape).is_err());
        
        // Non-object should fail
        let array_json = Json::from(vec![Json::from(1)]);
        assert!(array_json.has_shape(&shape).is_err());
    }
}