# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a pure Rust implementation of the LightGBM gradient boosting framework, designed for high performance, memory safety, and seamless integration with the Rust ecosystem. The project is currently in active development with a complete architectural foundation, comprehensive E2E test suite, and defined APIs, but core ML algorithms are still being implemented.

## Development Commands

### Build and Test
- Build project: `cargo build`
- Run all tests: `cargo test`
- Run E2E tests specifically: `cargo test --test comprehensive_e2e_tests`
- Run specific test binary: `cargo run --bin test_complete_workflow`
- Format code: `cargo fmt`
- Lint code: `cargo clippy`
- Run with output: `cargo test -- --nocapture`

### Binary Executables
The project includes several test binaries defined in Cargo.toml:
- `test_missing_values`, `test_feature_importance`, `debug_split_finding`
- `test_ensemble`, `test_ensemble_simple`, `test_hyperopt`
- `test_complete_workflow`

Run with: `cargo run --bin <binary_name>`

### Development Setup
- Requires Rust 1.75 or later
- Optional GPU acceleration requires CUDA/OpenCL drivers
- Uses feature flags: `default = ["cpu", "polars", "csv"]`
- Other features: `gpu`, `async`, `python`, `full`

## Architecture and Code Organization

### Core Module Structure
```
src/
├── core/           # Fundamental types, error handling, traits, constants
├── config/         # Configuration management with validation
├── dataset/        # Data loading, preprocessing, binning
│   ├── binning/    # Categorical and numerical binning
│   ├── loader/     # Data loading utilities
│   └── preprocessing/ # Missing value handling
├── boosting/       # Gradient boosting algorithms (GBDT)
├── tree/           # Decision tree learning and construction
│   ├── histogram/  # Histogram-based operations with SIMD
│   ├── learner/    # Serial, parallel, feature-parallel learners
│   └── split/      # Split finding and evaluation
├── prediction/     # Model inference pipeline
├── metrics/        # Evaluation metrics (classification, regression, ranking)
└── io/             # Model serialization (bincode, JSON, LightGBM format)
```

### Key Components
- **Memory Management**: 32-byte aligned memory allocation for SIMD optimization
- **Parallel Processing**: Multi-threaded training/prediction using Rayon
- **DataFrame Support**: Native Polars integration alongside ndarray
- **Configuration System**: Type-safe config builder with validation
- **Comprehensive Testing**: 18 E2E scenarios covering all workflows

## Current Implementation Status

### ✅ Complete (Working)
- Core infrastructure, types, error handling
- Configuration system with validation
- Dataset structure and basic data access
- Model interfaces and API definitions
- Comprehensive E2E test suite (18 scenarios)
- Memory management and alignment

### 🚧 In Progress (Critical for v0.1.0)
- **Core Training Algorithms**: GBDT implementation
- **Tree Learning**: Histogram-based tree construction
- **Prediction Pipeline**: Model inference and tree traversal
- **CSV Loading Bug**: Feature count inconsistency issue

### Test Results
- **18 E2E tests implemented**: 15 passed (infrastructure), 3 failed (core algorithms not implemented)
- Test data available in `tests/data/` directory
- Common test utilities in `tests/common/mod.rs`

## Important Files and Patterns

### Key Files to Understand
- `src/lib.rs` - Main library interface and API exports
- `src/core/mod.rs` - Fundamental types and trait definitions
- `src/boosting.rs` - Main GBDT training logic (implementation pending)
- `tests/common/mod.rs` - Comprehensive test utilities and fixtures
- `pure_rust_detailed_lightgbm_design_document.md` - Complete system design

### Code Patterns
- Uses ndarray::Array2<f32> for feature matrices, Array1<f32> for vectors
- Error handling via `anyhow::Result<T>` and custom error types
- Builder pattern for configuration (ConfigBuilder)
- RAII and ownership patterns for memory safety
- Feature-gated functionality for optional dependencies

## Development Guidelines

### Testing Strategy
- Always run E2E tests: `cargo test --test comprehensive_e2e_tests`
- Use test fixtures from `tests/common/mod.rs` for consistent test data
- Validate against original LightGBM behavior when possible
- Include performance tests for critical paths

### Performance Considerations
- Maintain 32-byte memory alignment for SIMD operations
- Use Rayon for parallel processing where appropriate
- Minimize allocations in hot paths (tree learning, prediction)
- Profile critical code paths during implementation

### Code Style
- Follow existing patterns in the codebase
- Use meaningful names reflecting ML/LightGBM terminology
- Add comprehensive documentation for public APIs
- Include inline comments for complex algorithms

## Dependencies and Features

### Core Dependencies
- `ndarray` (with rayon, serde features) - Numerical computing
- `rayon` - Parallel processing
- `anyhow`, `thiserror` - Error handling
- `serde`, `serde_json`, `bincode` - Serialization
- `polars` (optional) - DataFrame support
- `csv` (optional) - CSV processing

### Optional Features
- `cubecl` - GPU acceleration (planned)
- `tokio` - Async support
- `pyo3` - Python bindings

## Critical Implementation Notes

### Data Layout
- Features stored as row-major Array2<f32>
- Labels as Array1<f32>
- Weights and categorical indices as optional Vec<usize>
- Missing values represented as f32::NAN

### Memory Alignment
- Uses 32-byte aligned memory for SIMD optimization
- Critical for performance in histogram operations and tree learning

### Configuration
- Type-safe configuration via ConfigBuilder
- Validation occurs at build time
- Supports all LightGBM parameters with Rust naming conventions

When implementing core algorithms, refer to the comprehensive design document and E2E tests to understand expected behavior and interfaces.