# Contributing to Pure Rust LightGBM

Thank you for your interest in contributing to Pure Rust LightGBM! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites
- Rust 1.75 or later
- Git
- Basic familiarity with machine learning concepts
- Knowledge of the original LightGBM framework is helpful but not required

### Development Setup
1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/pure-rust-lightgbm.git`
3. Create a development branch: `git checkout -b feature/your-feature-name`
4. Install dependencies: `cargo build`
5. Run tests to ensure everything works: `cargo test`

## Types of Contributions

### High Priority (v0.1.0)
These are critical for basic functionality:

1. **Core Algorithm Implementation** - Gradient boosting training loop
2. **Tree Learning** - Decision tree construction and splitting
3. **Prediction Pipeline** - Model inference and tree traversal
4. **Bug Fixes** - Critical bugs like CSV loading issues

### Medium Priority (v0.2.0+)
Important for production readiness:

1. **Missing Value Handling** - Data preprocessing improvements
2. **Model Persistence** - Save/load functionality
3. **Feature Importance** - Model interpretability
4. **Performance Optimization** - SIMD and memory optimizations

### Low Priority (v0.3.0+)
Advanced features and optimizations:

1. **GPU Acceleration** - CubeCL integration
2. **Hyperparameter Optimization** - AutoML features
3. **Model Ensembles** - Advanced ML techniques

## Development Guidelines

### Code Style
- Follow standard Rust formatting: `cargo fmt`
- Use meaningful variable and function names
- Add documentation comments for public APIs
- Follow the existing code organization and patterns

### Testing Requirements
- All new functionality must include tests
- Run the comprehensive E2E test suite: `cargo test --test comprehensive_e2e_tests`
- Ensure all existing tests pass
- Add integration tests for major features
- Include performance tests for critical paths

### Documentation
- Update relevant documentation for API changes
- Add examples for new functionality
- Keep the design document updated for architectural changes
- Include inline code comments for complex algorithms

### Performance Considerations
- Profile critical code paths
- Use SIMD operations where beneficial
- Maintain memory alignment for performance
- Consider parallel processing opportunities
- Minimize memory allocations in hot paths

## Pull Request Process

### Before Submitting
1. Ensure your code compiles without warnings: `cargo build`
2. Run the full test suite: `cargo test`
3. Format your code: `cargo fmt`
4. Run clippy for additional checks: `cargo clippy`
5. Update documentation as needed

### Pull Request Template
Use the provided PR template and ensure you:
- [ ] Describe the changes clearly
- [ ] Reference related issues
- [ ] Include test coverage
- [ ] Note any breaking changes
- [ ] Verify CI passes

### Review Process
- All PRs require review before merging
- Address reviewer feedback promptly
- Maintain backward compatibility when possible
- Ensure documentation is updated

## Issue Guidelines

### Reporting Bugs
- Use the bug report template
- Include steps to reproduce
- Provide system information (OS, Rust version)
- Include relevant error messages and logs
- Test with the latest version

### Suggesting Features
- Check existing issues first
- Clearly describe the use case
- Consider implementation complexity
- Discuss API design implications
- Reference the design document when relevant

### Working on Issues
- Comment on issues you want to work on
- Ask questions if requirements are unclear
- Keep the scope focused and manageable
- Update issue status regularly

## Architecture Overview

### Key Concepts
- **Modular Design**: Each component is self-contained
- **Interface-First**: APIs are defined before implementation
- **Test-Driven**: Comprehensive E2E tests guide development
- **Performance-Oriented**: Optimized for speed and memory efficiency

### Important Files
- `src/lib.rs` - Main library interface
- `src/core/` - Fundamental types and infrastructure
- `src/boosting/` - Core GBDT algorithm
- `src/tree/` - Tree learning and construction
- `tests/comprehensive_e2e_tests.rs` - E2E test suite
- `pure_rust_detailed_lightgbm_design_document.md` - System design

### Development Workflow
1. Review the design document for context
2. Check E2E tests for expected behavior
3. Implement functionality following existing patterns
4. Ensure tests pass and add new tests
5. Update documentation

## Code Organization

### Module Structure
```
src/
├── core/           # Fundamental types and infrastructure
├── config/         # Configuration management
├── dataset/        # Data loading and preprocessing
├── boosting/       # Gradient boosting algorithms
├── tree/           # Tree learning and construction
├── prediction/     # Model inference
├── metrics/        # Evaluation metrics
└── io/             # Model serialization
```

### Naming Conventions
- Use descriptive names for functions and variables
- Follow Rust naming conventions (snake_case, CamelCase)
- Prefix internal functions with underscore if needed
- Use domain-specific terminology from ML/LightGBM

## Testing Strategy

### Test Levels
1. **Unit Tests** - Individual function testing
2. **Integration Tests** - Module interaction testing
3. **E2E Tests** - Complete workflow validation
4. **Performance Tests** - Benchmarking and profiling

### Test Data
- Use deterministic test data with fixed seeds
- Include edge cases and boundary conditions
- Test with various data sizes and types
- Validate against known-good results when possible

## Release Process

### Version Management
- Follow semantic versioning (SemVer)
- Tag releases appropriately
- Maintain changelog
- Document breaking changes

### Quality Gates
- All tests must pass
- No compiler warnings
- Documentation is up to date
- Performance benchmarks pass
- Security review for significant changes

## Getting Help

### Resources
- [Design Document](pure_rust_detailed_lightgbm_design_document.md)
- [E2E Test Analysis](E2E_TEST_RESULTS_ANALYSIS.md)
- [Issue Tracker](https://github.com/BectorVoom/pure-rust-lightgbm/issues)
- [Original LightGBM Documentation](https://lightgbm.readthedocs.io/)

### Communication
- Create issues for questions and discussions
- Be respectful and constructive
- Help others when possible
- Share knowledge and insights

## Recognition

Contributors will be acknowledged in:
- Release notes
- Documentation
- Contributor list
- Git commit history

Thank you for contributing to Pure Rust LightGBM! Your efforts help make machine learning in Rust more accessible and powerful.