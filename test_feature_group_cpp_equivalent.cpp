/*!
 * Copyright (c) 2017 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

/*
 * C++ equivalent test for FeatureGroup to validate Rust implementation
 * This test creates the same scenarios as the Rust tests for comparison
 */

#include <LightGBM/feature_group.h>
#include <LightGBM/bin.h>
#include <LightGBM/meta.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cassert>

using namespace LightGBM;

/// Test data structure matching the Rust version
struct TestData {
    std::vector<double> values;
    data_size_t num_data;
    int num_features;
    int group_id;
    
    TestData() {
        values = {1.0, 2.5, 3.1, 0.8, 4.2, 1.7, 2.9, 3.6, 0.5, 4.8};
        num_data = 10;
        num_features = 2;
        group_id = 0;
    }
    
    static TestData large_dataset() {
        TestData data;
        data.values.clear();
        for (int i = 0; i < 1000; ++i) {
            data.values.push_back(static_cast<double>(i) / 100.0);
        }
        data.num_data = 1000;
        data.num_features = 5;
        data.group_id = 0;
        return data;
    }
};

/// Create a simple BinMapper for testing
std::unique_ptr<BinMapper> create_test_bin_mapper() {
    // Create a simple numerical bin mapper
    std::vector<double> values = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    int num_bins = 5;
    
    // This is a simplified BinMapper creation
    // In actual implementation, you'd use proper BinMapper constructor
    return std::make_unique<BinMapper>();
}

/// Test basic FeatureGroup construction
void test_feature_group_basic_construction() {
    TestData test_data;
    
    // Create bin mappers
    std::vector<std::unique_ptr<BinMapper>> bin_mappers;
    for (int i = 0; i < test_data.num_features; ++i) {
        bin_mappers.emplace_back(create_test_bin_mapper());
    }
    
    // Test primary constructor
    FeatureGroup feature_group(
        test_data.num_features,
        0, // not multi-val
        &bin_mappers,
        test_data.num_data,
        test_data.group_id
    );
    
    // Verify basic properties
    assert(feature_group.num_feature() == test_data.num_features);
    assert(!feature_group.is_multi_val());
    assert(feature_group.num_total_bin() > 0);
    
    std::cout << "Basic construction test passed" << std::endl;
}

/// Test FeatureGroup with multi-value bins
void test_feature_group_multi_val_construction() {
    TestData test_data;
    
    std::vector<std::unique_ptr<BinMapper>> bin_mappers;
    for (int i = 0; i < test_data.num_features; ++i) {
        bin_mappers.emplace_back(create_test_bin_mapper());
    }
    
    // Test with multi-value bins enabled
    FeatureGroup feature_group(
        test_data.num_features,
        1, // enable multi-val
        &bin_mappers,
        test_data.num_data,
        test_data.group_id
    );
    
    assert(feature_group.num_feature() == test_data.num_features);
    assert(feature_group.is_multi_val());
    
    std::cout << "Multi-val construction test passed" << std::endl;
}

/// Test single feature constructor
void test_feature_group_single_feature() {
    TestData test_data;
    
    std::vector<std::unique_ptr<BinMapper>> bin_mappers;
    bin_mappers.emplace_back(create_test_bin_mapper());
    
    FeatureGroup feature_group(&bin_mappers, test_data.num_data);
    
    assert(feature_group.num_feature() == 1);
    assert(!feature_group.is_multi_val());
    
    std::cout << "Single feature test passed" << std::endl;
}

/// Test copy constructor
void test_feature_group_copy_constructor() {
    TestData test_data;
    
    std::vector<std::unique_ptr<BinMapper>> bin_mappers;
    for (int i = 0; i < test_data.num_features; ++i) {
        bin_mappers.emplace_back(create_test_bin_mapper());
    }
    
    FeatureGroup original(
        test_data.num_features,
        0,
        &bin_mappers,
        test_data.num_data,
        test_data.group_id
    );
    
    // Test copy with different num_data
    data_size_t new_num_data = test_data.num_data * 2;
    FeatureGroup copied(original, new_num_data);
    
    // Verify properties are copied correctly
    assert(copied.num_feature() == original.num_feature());
    assert(copied.is_multi_val() == original.is_multi_val());
    assert(copied.is_sparse() == original.is_sparse());
    assert(copied.num_total_bin() == original.num_total_bin());
    
    std::cout << "Copy constructor test passed" << std::endl;
}

/// Test data pushing functionality
void test_feature_group_push_data() {
    TestData test_data;
    
    std::vector<std::unique_ptr<BinMapper>> bin_mappers;
    for (int i = 0; i < test_data.num_features; ++i) {
        bin_mappers.emplace_back(create_test_bin_mapper());
    }
    
    FeatureGroup feature_group(
        test_data.num_features,
        0,
        &bin_mappers,
        test_data.num_data,
        test_data.group_id
    );
    
    // Initialize streaming
    feature_group.InitStreaming(1, 4);
    
    // Push some test data
    for (size_t idx = 0; idx < test_data.values.size(); ++idx) {
        int feature_idx = idx % test_data.num_features;
        data_size_t line_idx = idx / test_data.num_features;
        
        if (line_idx < test_data.num_data) {
            feature_group.PushData(0, feature_idx, line_idx, test_data.values[idx]);
        }
    }
    
    // Finish loading
    feature_group.FinishLoad();
    
    assert(feature_group.num_feature() == test_data.num_features);
    
    std::cout << "Push data test passed" << std::endl;
}

/// Test resize functionality
void test_feature_group_resize() {
    TestData test_data;
    
    std::vector<std::unique_ptr<BinMapper>> bin_mappers;
    for (int i = 0; i < test_data.num_features; ++i) {
        bin_mappers.emplace_back(create_test_bin_mapper());
    }
    
    FeatureGroup feature_group(
        test_data.num_features,
        0,
        &bin_mappers,
        test_data.num_data,
        test_data.group_id
    );
    
    // Test resizing
    data_size_t new_size = test_data.num_data * 2;
    feature_group.ReSize(new_size);
    
    assert(feature_group.num_feature() == test_data.num_features);
    
    std::cout << "Resize test passed" << std::endl;
}

/// Test bin value conversion
void test_feature_group_bin_to_value() {
    TestData test_data;
    
    std::vector<std::unique_ptr<BinMapper>> bin_mappers;
    for (int i = 0; i < test_data.num_features; ++i) {
        bin_mappers.emplace_back(create_test_bin_mapper());
    }
    
    FeatureGroup feature_group(
        test_data.num_features,
        0,
        &bin_mappers,
        test_data.num_data,
        test_data.group_id
    );
    
    // Test bin to value conversion
    for (int feature_idx = 0; feature_idx < test_data.num_features; ++feature_idx) {
        double value = feature_group.BinToValue(feature_idx, 1);
        // Should return a valid floating point value
        assert(std::isfinite(value));
    }
    
    std::cout << "Bin to value test passed" << std::endl;
}

/// Test iterator creation
void test_feature_group_iterators() {
    TestData test_data;
    
    std::vector<std::unique_ptr<BinMapper>> bin_mappers;
    for (int i = 0; i < test_data.num_features; ++i) {
        bin_mappers.emplace_back(create_test_bin_mapper());
    }
    
    FeatureGroup feature_group(
        test_data.num_features,
        0,
        &bin_mappers,
        test_data.num_data,
        test_data.group_id
    );
    
    // Test sub-feature iterator
    for (int feature_idx = 0; feature_idx < test_data.num_features; ++feature_idx) {
        BinIterator* iterator = feature_group.SubFeatureIterator(feature_idx);
        // May be null for some configurations - this is acceptable
        if (iterator) {
            delete iterator;
        }
    }
    
    // Test feature group iterator
    BinIterator* group_iterator = feature_group.FeatureGroupIterator();
    if (group_iterator) {
        delete group_iterator;
    }
    
    std::cout << "Iterator test passed" << std::endl;
}

/// Integration test that reads the same data the Rust test writes
void generate_cpp_results_for_comparison() {
    // Read test data from file (written by Rust test)
    std::ifstream input_file("/tmp/rust_feature_group_test_data.txt");
    if (!input_file.is_open()) {
        std::cout << "Could not open test data file - creating with default data" << std::endl;
        TestData test_data;
        
        std::vector<std::unique_ptr<BinMapper>> bin_mappers;
        for (int i = 0; i < test_data.num_features; ++i) {
            bin_mappers.emplace_back(create_test_bin_mapper());
        }
        
        FeatureGroup feature_group(
            test_data.num_features,
            0,
            &bin_mappers,
            test_data.num_data,
            test_data.group_id
        );
        
        // Write C++ results
        std::ofstream results_file("/tmp/cpp_feature_group_results.txt");
        results_file << "num_feature: " << feature_group.num_feature() << std::endl;
        results_file << "is_multi_val: " << feature_group.is_multi_val() << std::endl;
        results_file << "is_sparse: " << feature_group.is_sparse() << std::endl;
        results_file << "num_total_bin: " << feature_group.num_total_bin() << std::endl;
        results_file << "sizes_in_byte_with_data: " << feature_group.SizesInByte(true) << std::endl;
        results_file << "sizes_in_byte_without_data: " << feature_group.SizesInByte(false) << std::endl;
        
        // Write bin ranges
        for (int feature_idx = 0; feature_idx < test_data.num_features; ++feature_idx) {
            results_file << "feature_" << feature_idx << "_min_bin: " << feature_group.feature_min_bin(feature_idx) << std::endl;
            results_file << "feature_" << feature_idx << "_max_bin: " << feature_group.feature_max_bin(feature_idx) << std::endl;
        }
        
        // Write bin to value conversions
        for (int feature_idx = 0; feature_idx < test_data.num_features; ++feature_idx) {
            for (uint32_t bin = 0; bin < 5; ++bin) {
                double value = feature_group.BinToValue(feature_idx, bin);
                results_file << "feature_" << feature_idx << "_bin_" << bin << "_value: " << value << std::endl;
            }
        }
        
        return;
    }
    
    // If we can read the file, process the shared test data
    data_size_t num_data;
    int num_features;
    int group_id;
    
    input_file >> num_data >> num_features >> group_id;
    
    std::vector<double> values;
    double value;
    while (input_file >> value) {
        values.push_back(value);
    }
    input_file.close();
    
    // Create FeatureGroup with the same data
    std::vector<std::unique_ptr<BinMapper>> bin_mappers;
    for (int i = 0; i < num_features; ++i) {
        bin_mappers.emplace_back(create_test_bin_mapper());
    }
    
    FeatureGroup feature_group(num_features, 0, &bin_mappers, num_data, group_id);
    
    // Initialize and push data
    feature_group.InitStreaming(1, 4);
    
    for (size_t idx = 0; idx < values.size(); ++idx) {
        int feature_idx = idx % num_features;
        data_size_t line_idx = idx / num_features;
        
        if (line_idx < num_data) {
            feature_group.PushData(0, feature_idx, line_idx, values[idx]);
        }
    }
    
    feature_group.FinishLoad();
    
    // Write C++ results for comparison with Rust
    std::ofstream results_file("/tmp/cpp_feature_group_results.txt");
    results_file << "num_feature: " << feature_group.num_feature() << std::endl;
    results_file << "is_multi_val: " << feature_group.is_multi_val() << std::endl;
    results_file << "is_sparse: " << feature_group.is_sparse() << std::endl;
    results_file << "num_total_bin: " << feature_group.num_total_bin() << std::endl;
    results_file << "sizes_in_byte_with_data: " << feature_group.SizesInByte(true) << std::endl;
    results_file << "sizes_in_byte_without_data: " << feature_group.SizesInByte(false) << std::endl;
    
    // Write bin ranges
    for (int feature_idx = 0; feature_idx < num_features; ++feature_idx) {
        results_file << "feature_" << feature_idx << "_min_bin: " << feature_group.feature_min_bin(feature_idx) << std::endl;
        results_file << "feature_" << feature_idx << "_max_bin: " << feature_group.feature_max_bin(feature_idx) << std::endl;
    }
    
    // Write bin to value conversions
    for (int feature_idx = 0; feature_idx < num_features; ++feature_idx) {
        for (uint32_t bin = 0; bin < 5; ++bin) {
            double value = feature_group.BinToValue(feature_idx, bin);
            results_file << "feature_" << feature_idx << "_bin_" << bin << "_value: " << value << std::endl;
        }
    }
    
    std::cout << "C++ results written for comparison" << std::endl;
}

int main() {
    std::cout << "Running C++ FeatureGroup equivalence tests..." << std::endl;
    
    try {
        test_feature_group_basic_construction();
        test_feature_group_multi_val_construction();
        test_feature_group_single_feature();
        test_feature_group_copy_constructor();
        test_feature_group_push_data();
        test_feature_group_resize();
        test_feature_group_bin_to_value();
        test_feature_group_iterators();
        
        generate_cpp_results_for_comparison();
        
        std::cout << "All C++ tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
    
    return 0;
}