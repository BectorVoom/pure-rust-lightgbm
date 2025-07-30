#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <limits>

// Simple implementations of the key common functions for comparison

inline char tolower_simple(char in) {
    if (in <= 'Z' && in >= 'A')
        return in - ('Z' - 'z');
    return in;
}

inline std::string Trim_simple(std::string str) {
    if (str.empty()) {
        return str;
    }
    str.erase(str.find_last_not_of(" \f\n\r\t\v") + 1);
    str.erase(0, str.find_first_not_of(" \f\n\r\t\v"));
    return str;
}

inline std::string RemoveQuotationSymbol_simple(std::string str) {
    if (str.empty()) {
        return str;
    }
    str.erase(str.find_last_not_of("'\"") + 1);
    str.erase(0, str.find_first_not_of("'\""));
    return str;
}

inline bool StartsWith_simple(const std::string& str, const std::string prefix) {
    if (str.substr(0, prefix.size()) == prefix) {
        return true;
    } else {
        return false;
    }
}

inline std::vector<std::string> Split_simple(const char* c_str, char delimiter) {
    std::vector<std::string> ret;
    std::string str(c_str);
    size_t i = 0;
    size_t pos = 0;
    while (pos < str.length()) {
        if (str[pos] == delimiter) {
            if (i < pos) {
                ret.push_back(str.substr(i, pos - i));
            }
            ++pos;
            i = pos;
        } else {
            ++pos;
        }
    }
    if (i < pos) {
        ret.push_back(str.substr(i));
    }
    return ret;
}

inline std::vector<std::string> SplitLines_simple(const char* c_str) {
    std::vector<std::string> ret;
    std::string str(c_str);
    size_t i = 0;
    size_t pos = 0;
    while (pos < str.length()) {
        if (str[pos] == '\n' || str[pos] == '\r') {
            if (i < pos) {
                ret.push_back(str.substr(i, pos - i));
            }
            // skip the line endings
            while (str[pos] == '\n' || str[pos] == '\r') ++pos;
            // new begin
            i = pos;
        } else {
            ++pos;
        }
    }
    if (i < pos) {
        ret.push_back(str.substr(i));
    }
    return ret;
}

template<typename T>
inline const char* Atoi_simple(const char* p, T* out) {
    int sign;
    T value;
    while (*p == ' ') {
        ++p;
    }
    sign = 1;
    if (*p == '-') {
        sign = -1;
        ++p;
    } else if (*p == '+') {
        ++p;
    }
    for (value = 0; *p >= '0' && *p <= '9'; ++p) {
        value = value * 10 + (*p - '0');
    }
    *out = static_cast<T>(sign * value);
    while (*p == ' ') {
        ++p;
    }
    return p;
}

template<typename T>
inline double Pow_simple(T base, int power) {
    if (power < 0) {
        return 1.0 / Pow_simple(base, -power);
    } else if (power == 0) {
        return 1;
    } else if (power % 2 == 0) {
        return Pow_simple(base*base, power / 2);
    } else if (power % 3 == 0) {
        return Pow_simple(base*base*base, power / 3);
    } else {
        return base * Pow_simple(base, power - 1);
    }
}

inline int64_t Pow2RoundUp_simple(int64_t x) {
    int64_t t = 1;
    for (int i = 0; i < 64; ++i) {
        if (t >= x) {
            return t;
        }
        t <<= 1;
    }
    return 0;
}

inline void Softmax_simple(std::vector<double>* p_rec) {
    std::vector<double> &rec = *p_rec;
    double wmax = rec[0];
    for (size_t i = 1; i < rec.size(); ++i) {
        wmax = std::max(rec[i], wmax);
    }
    double wsum = 0.0f;
    for (size_t i = 0; i < rec.size(); ++i) {
        rec[i] = std::exp(rec[i] - wmax);
        wsum += rec[i];
    }
    for (size_t i = 0; i < rec.size(); ++i) {
        rec[i] /= static_cast<double>(wsum);
    }
}

inline double AvoidInf_simple(double x) {
    if (std::isnan(x)) {
        return 0.0;
    } else if (x >= 1e300) {
        return 1e300;
    } else if (x <= -1e300) {
        return -1e300;
    } else {
        return x;
    }
}

inline int RoundInt_simple(double x) {
    return static_cast<int>(x + 0.5f);
}

inline std::vector<uint32_t> EmptyBitset_simple(int n) {
    int size = n / 32;
    if (n % 32 != 0) ++size;
    return std::vector<uint32_t>(size);
}

template<typename T>
inline void InsertBitset_simple(std::vector<uint32_t>* vec, const T val) {
    auto& ref_v = *vec;
    int i1 = val / 32;
    int i2 = val % 32;
    if (static_cast<int>(vec->size()) < i1 + 1) {
        vec->resize(i1 + 1, 0);
    }
    ref_v[i1] |= (1 << i2);
}

template<typename T>
inline bool FindInBitset_simple(const uint32_t* bits, int n, T pos) {
    int i1 = pos / 32;
    if (i1 >= n) {
        return false;
    }
    int i2 = pos % 32;
    return (bits[i1] >> i2) & 1;
}

template<typename T>
inline std::vector<T> ArrayCast_simple(const std::vector<T>& arr) {
    return arr;  // simplified - in real case would cast between types
}

template<typename T>
inline std::string Join_simple(const std::vector<T>& strs, const char* delimiter) {
    if (strs.empty()) {
        return std::string("");
    }
    std::stringstream str_buf;
    str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
    str_buf << strs[0];
    for (size_t i = 1; i < strs.size(); ++i) {
        str_buf << delimiter;
        str_buf << strs[i];
    }
    return str_buf.str();
}

void test_string_functions() {
    std::cout << "\n=== Testing String Functions ===\n";
    
    std::cout << "tolower('A'): " << tolower_simple('A') << std::endl;
    std::cout << "tolower('Z'): " << tolower_simple('Z') << std::endl;
    std::cout << "tolower('a'): " << tolower_simple('a') << std::endl;
    std::cout << "tolower('1'): " << tolower_simple('1') << std::endl;
    
    std::cout << "Trim('  hello  '): '" << Trim_simple("  hello  ") << "'" << std::endl;
    std::cout << "Trim('\\t\\n hello \\r\\n'): '" << Trim_simple("\t\n hello \r\n") << "'" << std::endl;
    std::cout << "Trim(''): '" << Trim_simple("") << "'" << std::endl;
    
    std::cout << "RemoveQuotationSymbol('\"hello\"'): '" << RemoveQuotationSymbol_simple("\"hello\"") << "'" << std::endl;
    std::cout << "RemoveQuotationSymbol(\"'world'\"): '" << RemoveQuotationSymbol_simple("'world'") << "'" << std::endl;
    
    std::cout << "StartsWith('hello world', 'hello'): " << std::boolalpha << StartsWith_simple("hello world", "hello") << std::endl;
    std::cout << "StartsWith('hello world', 'world'): " << std::boolalpha << StartsWith_simple("hello world", "world") << std::endl;
    
    std::vector<std::string> split_result = Split_simple("a,b,c", ',');
    std::cout << "Split('a,b,c', ','): ";
    for (const auto& s : split_result) {
        std::cout << "'" << s << "' ";
    }
    std::cout << std::endl;
    
    split_result = Split_simple("a,,b", ',');
    std::cout << "Split('a,,b', ','): ";
    for (const auto& s : split_result) {
        std::cout << "'" << s << "' ";
    }
    std::cout << std::endl;
    
    std::vector<std::string> lines_result = SplitLines_simple("line1\nline2\r\nline3");
    std::cout << "SplitLines('line1\\nline2\\r\\nline3'): ";
    for (const auto& s : lines_result) {
        std::cout << "'" << s << "' ";
    }
    std::cout << std::endl;
}

void test_numeric_functions() {
    std::cout << "\n=== Testing Numeric Functions ===\n";
    
    int int_result;
    const char* remaining = Atoi_simple("123abc", &int_result);
    std::cout << "Atoi('123abc'): value=" << int_result << ", remaining='" << remaining << "'" << std::endl;
    
    remaining = Atoi_simple("-456", &int_result);
    std::cout << "Atoi('-456'): value=" << int_result << ", remaining='" << remaining << "'" << std::endl;
}

void test_array_functions() {
    std::cout << "\n=== Testing Array Functions ===\n";
    
    std::vector<int> int_vec = {1, 2, 3, 4, 5};
    std::vector<int> double_vec = ArrayCast_simple(int_vec);
    std::cout << "ArrayCast<int, double>([1,2,3,4,5]): ";
    for (int d : double_vec) {
        std::cout << d << " ";
    }
    std::cout << std::endl;
    
    std::vector<int> join_vec = {10, 20, 30, 40};
    std::string joined = Join_simple(join_vec, ", ");
    std::cout << "Join([10,20,30,40], ', '): '" << joined << "'" << std::endl;
}

void test_math_functions() {
    std::cout << "\n=== Testing Math Functions ===\n";
    
    std::cout << "Pow(2.0, 3): " << std::fixed << std::setprecision(6) << Pow_simple(2.0, 3) << std::endl;
    std::cout << "Pow(2.0, -2): " << Pow_simple(2.0, -2) << std::endl;
    std::cout << "Pow(5.0, 0): " << Pow_simple(5.0, 0) << std::endl;
    
    std::cout << "Pow2RoundUp(10): " << Pow2RoundUp_simple(10) << std::endl;
    std::cout << "Pow2RoundUp(16): " << Pow2RoundUp_simple(16) << std::endl;
    std::cout << "Pow2RoundUp(100): " << Pow2RoundUp_simple(100) << std::endl;
    
    std::vector<double> softmax_vec = {1.0, 2.0, 3.0};
    Softmax_simple(&softmax_vec);
    std::cout << "Softmax([1.0, 2.0, 3.0]): ";
    for (double d : softmax_vec) {
        std::cout << std::setprecision(6) << d << " ";
    }
    std::cout << std::endl;
    
    std::cout << "AvoidInf(1e301): " << AvoidInf_simple(1e301) << std::endl;
    std::cout << "AvoidInf(-1e301): " << AvoidInf_simple(-1e301) << std::endl;
    std::cout << "AvoidInf(NaN): " << AvoidInf_simple(std::numeric_limits<double>::quiet_NaN()) << std::endl;
    
    std::cout << "RoundInt(3.7): " << RoundInt_simple(3.7) << std::endl;
    std::cout << "RoundInt(3.2): " << RoundInt_simple(3.2) << std::endl;
    std::cout << "RoundInt(-2.7): " << RoundInt_simple(-2.7) << std::endl;
}

void test_bitset_functions() {
    std::cout << "\n=== Testing Bitset Functions ===\n";
    
    std::vector<uint32_t> bitset = EmptyBitset_simple(100);
    std::cout << "EmptyBitset(100) size: " << bitset.size() << std::endl;
    
    InsertBitset_simple(&bitset, 5);
    InsertBitset_simple(&bitset, 15);
    InsertBitset_simple(&bitset, 95);
    
    std::cout << "After inserting 5, 15, 95:" << std::endl;
    std::cout << "FindInBitset(bitset, 5): " << std::boolalpha << FindInBitset_simple(bitset.data(), bitset.size(), 5) << std::endl;
    std::cout << "FindInBitset(bitset, 15): " << std::boolalpha << FindInBitset_simple(bitset.data(), bitset.size(), 15) << std::endl;
    std::cout << "FindInBitset(bitset, 95): " << std::boolalpha << FindInBitset_simple(bitset.data(), bitset.size(), 95) << std::endl;
    std::cout << "FindInBitset(bitset, 6): " << std::boolalpha << FindInBitset_simple(bitset.data(), bitset.size(), 6) << std::endl;
}

int main() {
    std::cout << "=== C++ LightGBM Common Functions Test (Simplified) ===\n";
    
    test_string_functions();
    test_numeric_functions();
    test_array_functions();
    test_math_functions();
    test_bitset_functions();
    
    std::cout << "\n=== All tests completed ===\n";
    return 0;
}