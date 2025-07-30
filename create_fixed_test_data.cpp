#include <iostream>
#include <vector>
#include <fstream>

// Create fixed test data for equivalence testing
int main() {
    // Fixed test data - same for C++ and Rust
    std::vector<std::vector<uint32_t>> rows = {
        {10, 25, 100},    // Row 0
        {5, 15, 50, 75},  // Row 1  
        {1, 30},          // Row 2
        {20, 40, 60},     // Row 3
        {2, 8, 90}        // Row 4
    };
    
    std::vector<float> gradients = {0.5, -0.3, 1.2, -0.8, 0.1};
    std::vector<float> hessians = {0.6, 0.4, 0.9, 0.7, 0.5};
    
    // Export test data
    std::ofstream file("fixed_test_data.txt");
    file << "# Fixed test data for equivalence testing\n";
    file << "# Format: row_data_count value1 value2 ... gradient hessian\n";
    
    for (size_t i = 0; i < rows.size(); ++i) {
        file << rows[i].size();
        for (uint32_t val : rows[i]) {
            file << " " << val;
        }
        file << " " << gradients[i] << " " << hessians[i] << "\n";
    }
    
    file.close();
    std::cout << "Fixed test data exported to fixed_test_data.txt" << std::endl;
    
    return 0;
}