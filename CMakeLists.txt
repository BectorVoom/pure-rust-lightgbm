cmake_minimum_required(VERSION 3.15)
project(feature_group_equivalence_test)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# Add debug flags for debug builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG")

# Find LightGBM include directory
set(LIGHTGBM_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Include directories
include_directories(${LIGHTGBM_ROOT_DIR}/include)
include_directories(${LIGHTGBM_ROOT_DIR}/include/LightGBM)

# Add the test executable
add_executable(test_cpp_tree test_cpp_tree.cpp)

# Add the FeatureGroup test executable
add_executable(test_feature_group_cpp_equivalent test_feature_group_cpp_equivalent.cpp)

# Enable all warnings and treat warnings as errors for better code quality
target_compile_options(test_cpp_tree PRIVATE
    -Wall 
    -Wextra 
    -Wpedantic
    -Werror
)

target_compile_options(test_feature_group_cpp_equivalent PRIVATE
    -Wall 
    -Wextra 
    -Wpedantic
    -Werror
)

# Link threading library if needed
find_package(Threads REQUIRED)
target_link_libraries(test_cpp_tree PRIVATE Threads::Threads)
target_link_libraries(test_feature_group_cpp_equivalent PRIVATE Threads::Threads)

# Create a custom target to run the test
add_custom_target(run_cpp_tree_test
    COMMAND test_cpp_tree
    DEPENDS test_cpp_tree
    COMMENT "Running C++ Tree equivalence tests"
)

# Create a custom target to run the FeatureGroup test
add_custom_target(run_feature_group_cpp_test
    COMMAND test_feature_group_cpp_equivalent
    DEPENDS test_feature_group_cpp_equivalent
    COMMENT "Running C++ FeatureGroup equivalence tests"
)

# Add test registration for CTest
enable_testing()
add_test(NAME cpp_tree_equivalence_test COMMAND test_cpp_tree)
add_test(NAME cpp_feature_group_equivalence_test COMMAND test_feature_group_cpp_equivalent)

# Set properties for better test output
set_tests_properties(cpp_tree_equivalence_test PROPERTIES
    TIMEOUT 60
    FAIL_REGULAR_EXPRESSION "FAILED;ERROR;Test failed"
)

set_tests_properties(cpp_feature_group_equivalence_test PROPERTIES
    TIMEOUT 60
    FAIL_REGULAR_EXPRESSION "FAILED;ERROR;Test failed"
)

# Print configuration info
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")