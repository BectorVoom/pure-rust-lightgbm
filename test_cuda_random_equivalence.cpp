/*!
 * C++ test program to verify CUDA random equivalence
 * This tests the original C++ CUDARandom implementation
 */

#include <iostream>
#include <vector>
#include <iomanip>

// Simplified version of the original CUDA random for testing
class CUDARandom {
public:
    void SetSeed(int seed) {
        x = seed;
    }
    
    int NextShort(int lower_bound, int upper_bound) {
        return (RandInt16()) % (upper_bound - lower_bound) + lower_bound;
    }
    
    int NextInt(int lower_bound, int upper_bound) {
        return (RandInt32()) % (upper_bound - lower_bound) + lower_bound;
    }
    
    float NextFloat() {
        return static_cast<float>(RandInt16()) / (32768.0f);
    }

private:
    int RandInt16() {
        x = (214013 * x + 2531011);
        return static_cast<int>((x >> 16) & 0x7FFF);
    }
    
    int RandInt32() {
        x = (214013 * x + 2531011);
        return static_cast<int>(x & 0x7FFFFFFF);
    }
    
    unsigned int x = 123456789;
};

int main() {
    std::cout << "C++ CUDA Random Equivalence Test" << std::endl;
    std::cout << "=================================" << std::endl;
    
    CUDARandom rng;
    
    // Test 1: Default seed behavior
    std::cout << "\nTest 1: Default seed (123456789)" << std::endl;
    std::cout << "First 10 NextShort(0, 100):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << rng.NextShort(0, 100) << " ";
    }
    std::cout << std::endl;
    
    // Reset and test NextInt
    rng.SetSeed(123456789);
    std::cout << "First 10 NextInt(0, 1000):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << rng.NextInt(0, 1000) << " ";
    }
    std::cout << std::endl;
    
    // Reset and test NextFloat
    rng.SetSeed(123456789);
    std::cout << "First 10 NextFloat():" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < 10; i++) {
        std::cout << rng.NextFloat() << " ";
    }
    std::cout << std::endl;
    
    // Test 2: Custom seed
    std::cout << "\nTest 2: Custom seed (42)" << std::endl;
    rng.SetSeed(42);
    std::cout << "First 5 NextShort(10, 50):" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << rng.NextShort(10, 50) << " ";
    }
    std::cout << std::endl;
    
    // Test 3: Large range
    rng.SetSeed(12345);
    std::cout << "\nTest 3: Seed 12345, NextInt(0, 1000000):" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << rng.NextInt(0, 1000000) << " ";
    }
    std::cout << std::endl;
    
    // Test 4: Verify deterministic behavior
    std::cout << "\nTest 4: Deterministic behavior verification" << std::endl;
    CUDARandom rng1, rng2;
    rng1.SetSeed(9999);
    rng2.SetSeed(9999);
    
    bool deterministic = true;
    for (int i = 0; i < 20; i++) {
        if (rng1.NextInt(0, 100) != rng2.NextInt(0, 100)) {
            deterministic = false;
            break;
        }
    }
    std::cout << "Deterministic: " << (deterministic ? "PASS" : "FAIL") << std::endl;
    
    // Test 5: Raw LCG values for verification
    std::cout << "\nTest 5: Raw LCG internal states (seed=1)" << std::endl;
    CUDARandom rng_raw;
    rng_raw.SetSeed(1);
    std::cout << "First 5 NextShort(0, 65536) with seed=1:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << rng_raw.NextShort(0, 65536) << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\nC++ equivalence test completed." << std::endl;
    return 0;
}