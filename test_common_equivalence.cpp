#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <iomanip>

// Include the original LightGBM common.h header
#include "../include/LightGBM/utils/common.h"

using namespace LightGBM::Common;

void test_string_functions() {
    std::cout << "\n=== Testing String Functions ===\n";
    
    // Test tolower
    std::cout << "tolower('A'): " << tolower('A') << std::endl;
    std::cout << "tolower('Z'): " << tolower('Z') << std::endl;
    std::cout << "tolower('a'): " << tolower('a') << std::endl;
    std::cout << "tolower('1'): " << tolower('1') << std::endl;
    
    // Test Trim
    std::cout << "Trim('  hello  '): '" << Trim("  hello  ") << "'" << std::endl;
    std::cout << "Trim('\\t\\n hello \\r\\n'): '" << Trim("\t\n hello \r\n") << "'" << std::endl;
    std::cout << "Trim(''): '" << Trim("") << "'" << std::endl;
    
    // Test RemoveQuotationSymbol
    std::cout << "RemoveQuotationSymbol('\"hello\"'): '" << RemoveQuotationSymbol("\"hello\"") << "'" << std::endl;
    std::cout << "RemoveQuotationSymbol(\"'world'\"): '" << RemoveQuotationSymbol("'world'") << "'" << std::endl;
    
    // Test StartsWith
    std::cout << "StartsWith('hello world', 'hello'): " << std::boolalpha << StartsWith("hello world", "hello") << std::endl;
    std::cout << "StartsWith('hello world', 'world'): " << std::boolalpha << StartsWith("hello world", "world") << std::endl;
    
    // Test Split
    std::vector<std::string> split_result = Split("a,b,c", ',');
    std::cout << "Split('a,b,c', ','): ";
    for (const auto& s : split_result) {
        std::cout << "'" << s << "' ";
    }
    std::cout << std::endl;
    
    split_result = Split("a,,b", ',');
    std::cout << "Split('a,,b', ','): ";
    for (const auto& s : split_result) {
        std::cout << "'" << s << "' ";
    }
    std::cout << std::endl;
    
    // Test SplitLines
    std::vector<std::string> lines_result = SplitLines("line1\nline2\r\nline3");
    std::cout << "SplitLines('line1\\nline2\\r\\nline3'): ";
    for (const auto& s : lines_result) {
        std::cout << "'" << s << "' ";
    }
    std::cout << std::endl;
}

void test_numeric_functions() {
    std::cout << "\n=== Testing Numeric Functions ===\n";
    
    // Test Atoi
    int int_result;
    const char* remaining = Atoi("123abc", &int_result);
    std::cout << "Atoi('123abc'): value=" << int_result << ", remaining='" << remaining << "'" << std::endl;
    
    remaining = Atoi("-456", &int_result);
    std::cout << "Atoi('-456'): value=" << int_result << ", remaining='" << remaining << "'" << std::endl;
    
    // Test Atof
    double double_result;
    remaining = Atof("123.45", &double_result);
    std::cout << "Atof('123.45'): value=" << std::fixed << std::setprecision(6) << double_result << std::endl;
    
    remaining = Atof("1.23e-4", &double_result);
    std::cout << "Atof('1.23e-4'): value=" << std::scientific << std::setprecision(6) << double_result << std::endl;
    
    remaining = Atof("nan", &double_result);
    std::cout << "Atof('nan'): value=" << (std::isnan(double_result) ? "NaN" : std::to_string(double_result)) << std::endl;
    
    remaining = Atof("inf", &double_result);
    std::cout << "Atof('inf'): value=" << (std::isinf(double_result) ? "Inf" : std::to_string(double_result)) << std::endl;
    
    // Test AtoiAndCheck
    std::cout << "AtoiAndCheck('123'): " << std::boolalpha << AtoiAndCheck("123", &int_result) << std::endl;
    std::cout << "AtoiAndCheck('123abc'): " << std::boolalpha << AtoiAndCheck("123abc", &int_result) << std::endl;
    
    // Test AtofAndCheck
    std::cout << "AtofAndCheck('123.45'): " << std::boolalpha << AtofAndCheck("123.45", &double_result) << std::endl;
    std::cout << "AtofAndCheck('123.45abc'): " << std::boolalpha << AtofAndCheck("123.45abc", &double_result) << std::endl;
}

void test_array_functions() {
    std::cout << "\n=== Testing Array Functions ===\n";
    
    // Test ArrayCast
    std::vector<int> int_vec = {1, 2, 3, 4, 5};
    std::vector<double> double_vec = ArrayCast<int, double>(int_vec);
    std::cout << "ArrayCast<int, double>([1,2,3,4,5]): ";
    for (double d : double_vec) {
        std::cout << d << " ";
    }
    std::cout << std::endl;
    
    // Test StringToArray
    std::vector<int> str_to_array = StringToArray<int>("1,2,3,4,5", ',');
    std::cout << "StringToArray<int>('1,2,3,4,5', ','): ";
    for (int i : str_to_array) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    
    // Test Join
    std::vector<int> join_vec = {10, 20, 30, 40};
    std::string joined = Join(join_vec, ", ");
    std::cout << "Join([10,20,30,40], ', '): '" << joined << "'" << std::endl;
    
    std::string joined_range = Join(join_vec, 1, 3, " | ");
    std::cout << "Join([10,20,30,40], 1, 3, ' | '): '" << joined_range << "'" << std::endl;
}

void test_math_functions() {
    std::cout << "\n=== Testing Math Functions ===\n";
    
    // Test Pow
    std::cout << "Pow(2.0, 3): " << std::fixed << std::setprecision(6) << Pow(2.0, 3) << std::endl;
    std::cout << "Pow(2.0, -2): " << Pow(2.0, -2) << std::endl;
    std::cout << "Pow(5.0, 0): " << Pow(5.0, 0) << std::endl;
    
    // Test Pow2RoundUp
    std::cout << "Pow2RoundUp(10): " << Pow2RoundUp(10) << std::endl;
    std::cout << "Pow2RoundUp(16): " << Pow2RoundUp(16) << std::endl;
    std::cout << "Pow2RoundUp(100): " << Pow2RoundUp(100) << std::endl;
    
    // Test Softmax
    std::vector<double> softmax_vec = {1.0, 2.0, 3.0};
    Softmax(&softmax_vec);
    std::cout << "Softmax([1.0, 2.0, 3.0]): ";
    for (double d : softmax_vec) {
        std::cout << std::setprecision(6) << d << " ";
    }
    std::cout << std::endl;
    
    // Test AvoidInf
    std::cout << "AvoidInf(1e301): " << AvoidInf(1e301) << std::endl;
    std::cout << "AvoidInf(-1e301): " << AvoidInf(-1e301) << std::endl;
    std::cout << "AvoidInf(NaN): " << AvoidInf(std::numeric_limits<double>::quiet_NaN()) << std::endl;
    
    // Test RoundInt
    std::cout << "RoundInt(3.7): " << RoundInt(3.7) << std::endl;
    std::cout << "RoundInt(3.2): " << RoundInt(3.2) << std::endl;
    std::cout << "RoundInt(-2.7): " << RoundInt(-2.7) << std::endl;
}

void test_bitset_functions() {
    std::cout << "\n=== Testing Bitset Functions ===\n";
    
    // Test EmptyBitset
    std::vector<uint32_t> bitset = EmptyBitset(100);
    std::cout << "EmptyBitset(100) size: " << bitset.size() << std::endl;
    
    // Test InsertBitset and FindInBitset
    InsertBitset(&bitset, 5);
    InsertBitset(&bitset, 15);
    InsertBitset(&bitset, 95);
    
    std::cout << "After inserting 5, 15, 95:" << std::endl;
    std::cout << "FindInBitset(bitset, 5): " << std::boolalpha << FindInBitset(bitset.data(), bitset.size(), 5) << std::endl;
    std::cout << "FindInBitset(bitset, 15): " << std::boolalpha << FindInBitset(bitset.data(), bitset.size(), 15) << std::endl;
    std::cout << "FindInBitset(bitset, 95): " << std::boolalpha << FindInBitset(bitset.data(), bitset.size(), 95) << std::endl;
    std::cout << "FindInBitset(bitset, 6): " << std::boolalpha << FindInBitset(bitset.data(), bitset.size(), 6) << std::endl;
    
    // Test ConstructBitset
    std::vector<int> vals = {1, 3, 7, 31, 63};
    std::vector<uint32_t> constructed_bitset = ConstructBitset(vals.data(), vals.size());
    std::cout << "ConstructBitset([1,3,7,31,63]) size: " << constructed_bitset.size() << std::endl;
    std::cout << "Contains 7: " << std::boolalpha << FindInBitset(constructed_bitset.data(), constructed_bitset.size(), 7) << std::endl;
    std::cout << "Contains 8: " << std::boolalpha << FindInBitset(constructed_bitset.data(), constructed_bitset.size(), 8) << std::endl;
}

void test_utility_functions() {
    std::cout << "\n=== Testing Utility Functions ===\n";
    
    // Test CheckDoubleEqualOrdered
    std::cout << "CheckDoubleEqualOrdered(1.0, 1.0): " << std::boolalpha << CheckDoubleEqualOrdered(1.0, 1.0) << std::endl;
    std::cout << "CheckDoubleEqualOrdered(1.0, 1.0000000001): " << std::boolalpha << CheckDoubleEqualOrdered(1.0, 1.0000000001) << std::endl;
    
    // Test GetDoubleUpperBound
    double upper = GetDoubleUpperBound(1.0);
    std::cout << "GetDoubleUpperBound(1.0): " << std::scientific << std::setprecision(17) << upper << std::endl;
    
    // Test GetLine
    std::cout << "GetLine('hello\\nworld'): " << GetLine("hello\nworld") << std::endl;
    std::cout << "GetLine('hello world'): " << GetLine("hello world") << std::endl;
    
    // Test CheckAllowedJSON
    std::cout << "CheckAllowedJSON('hello'): " << std::boolalpha << CheckAllowedJSON("hello") << std::endl;
    std::cout << "CheckAllowedJSON('hello{world}'): " << std::boolalpha << CheckAllowedJSON("hello{world}") << std::endl;
}

void test_statistics_functions() {
    std::cout << "\n=== Testing Statistics Functions ===\n";
    
    // Test ObtainMinMaxSum
    std::vector<double> data = {3.5, 1.2, 5.8, 2.1, 4.7};
    double min_val, max_val, sum_val;
    ObtainMinMaxSum(data.data(), data.size(), &min_val, &max_val, &sum_val);
    
    std::cout << "ObtainMinMaxSum([3.5, 1.2, 5.8, 2.1, 4.7]):" << std::endl;
    std::cout << "  Min: " << std::fixed << std::setprecision(2) << min_val << std::endl;
    std::cout << "  Max: " << max_val << std::endl;
    std::cout << "  Sum: " << sum_val << std::endl;
}

int main() {
    std::cout << "=== C++ LightGBM Common Functions Test ===\n";
    
    test_string_functions();
    test_numeric_functions();
    test_array_functions();
    test_math_functions();
    test_bitset_functions();
    test_utility_functions();
    test_statistics_functions();
    
    std::cout << "\n=== All tests completed ===\n";
    return 0;
}