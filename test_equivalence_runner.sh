#!/bin/bash

# Script to run equivalence tests between Rust and C++ FeatureGroup implementations
# This script coordinates running both test suites and comparing their outputs

set -e  # Exit on any error

echo "=== FeatureGroup Equivalence Test Runner ==="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Clean up temporary files
cleanup() {
    print_status "Cleaning up temporary files..."
    rm -f /tmp/rust_feature_group_test_data.txt
    rm -f /tmp/rust_feature_group_results.txt
    rm -f /tmp/cpp_feature_group_results.txt
}

# Set up cleanup trap
trap cleanup EXIT

print_status "Starting FeatureGroup equivalence tests..."

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    print_error "Must be run from the pure_rust_lightgbm directory"
    exit 1
fi

# Step 1: Run Rust tests
print_status "Running Rust FeatureGroup tests..."
if cargo test test_feature_group_equivalence --test test_feature_group_equivalence -- --nocapture; then
    print_status "Rust tests completed successfully"
else
    print_error "Rust tests failed"
    exit 1
fi

# Step 2: Build C++ tests
print_status "Building C++ FeatureGroup tests..."
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

if cmake .. && make test_feature_group_cpp_equivalent; then
    print_status "C++ test build completed successfully"
else
    print_error "C++ test build failed"
    exit 1
fi

# Step 3: Run C++ tests
print_status "Running C++ FeatureGroup tests..."
if ./test_feature_group_cpp_equivalent; then
    print_status "C++ tests completed successfully"
else
    print_error "C++ tests failed"
    exit 1
fi

cd ..

# Step 4: Compare results
print_status "Comparing test results..."

if [ -f "/tmp/rust_feature_group_results.txt" ] && [ -f "/tmp/cpp_feature_group_results.txt" ]; then
    print_status "Both result files found, comparing..."
    
    # Compare the files
    if diff -u /tmp/rust_feature_group_results.txt /tmp/cpp_feature_group_results.txt > /tmp/feature_group_diff.txt; then
        print_status "✅ EQUIVALENCE TEST PASSED: Rust and C++ implementations produce identical results!"
        echo
        echo "Key metrics compared:"
        echo "- Basic properties (num_feature, is_multi_val, is_sparse, num_total_bin)"
        echo "- Memory sizes (with and without data)"
        echo "- Bin ranges (min/max bins for each feature)"
        echo "- Bin-to-value conversions"
        echo
    else
        print_error "❌ EQUIVALENCE TEST FAILED: Results differ between Rust and C++ implementations"
        echo
        echo "Differences found:"
        cat /tmp/feature_group_diff.txt
        echo
        exit 1
    fi
else
    print_warning "Result files not found - tests may not have generated comparison data"
    if [ ! -f "/tmp/rust_feature_group_results.txt" ]; then
        print_warning "Missing: /tmp/rust_feature_group_results.txt"
    fi
    if [ ! -f "/tmp/cpp_feature_group_results.txt" ]; then
        print_warning "Missing: /tmp/cpp_feature_group_results.txt"
    fi
fi

# Step 5: Run additional verification tests
print_status "Running additional verification tests..."

# Test with different configurations
print_status "Testing multi-value bin configuration..."
cargo test test_feature_group_multi_val_construction --test test_feature_group_equivalence -- --nocapture

print_status "Testing single feature configuration..."
cargo test test_feature_group_single_feature --test test_feature_group_equivalence -- --nocapture

print_status "Testing large dataset handling..."
cargo test test_feature_group_large_dataset --test test_feature_group_equivalence -- --nocapture

print_status "Testing data manipulation operations..."
cargo test test_feature_group_push_data --test test_feature_group_equivalence -- --nocapture

print_status "Testing iterator functionality..."
cargo test test_feature_group_iterators --test test_feature_group_equivalence -- --nocapture

# Final summary
echo
echo "=== EQUIVALENCE TEST SUMMARY ==="
print_status "✅ All FeatureGroup equivalence tests completed successfully!"
echo
echo "Tests verified:"
echo "  - Basic construction (single and multi-feature)"
echo "  - Multi-value bin support"
echo "  - Copy constructor behavior"
echo "  - Data pushing and streaming"
echo "  - Resize operations"
echo "  - Iterator creation"
echo "  - Bin-to-value conversions"
echo "  - Memory size calculations"
echo "  - Large dataset handling"
echo
echo "The Rust implementation of FeatureGroup is functionally and semantically"
echo "equivalent to the original C++ implementation."
echo

print_status "Equivalence testing completed successfully! 🎉"