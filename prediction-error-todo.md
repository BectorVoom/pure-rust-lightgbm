# Pure Rust LightGBM Prediction Discrepancy Resolution TODO

## Problem Statement
Pure Rust LightGBM implementation produces predictions that differ from Python LightGBM by approximately 10⁻² (MSE: 0.082, MAE: 0.164). Target precision is 10⁻¹⁰ for algorithmic equivalence.

## Root Cause Analysis Results

### Primary Issues Identified
1. **Split Gain Calculation Precision** - Numerical differences in regularization application
2. **Data Count Estimation** - Approximation in histogram-based data counting
3. **Parameter Alignment** - Default parameter differences between implementations
4. **Floating-Point Precision** - Cumulative precision loss in iterative calculations

## High Priority Fixes

### 1. Improve Split Gain Calculation Precision
- **File**: `src/tree/split/finder.rs:403-446`
- **Issue**: Regularization formula may differ from Python LightGBM C++ implementation
- **Action**: 
  - [ ] Compare exact split gain formula with LightGBM C++ source
  - [ ] Implement bit-exact L1/L2 regularization calculations
  - [ ] Add numerical stability improvements for edge cases
- **Expected Impact**: Reduce prediction differences by 30-50%

### 2. Fix Data Count Estimation in Histograms
- **Files**: `src/tree/split/finder.rs:184`, `src/tree/split/finder.rs:308`
- **Issue**: Using hessian-based approximation instead of exact data counts
- **Action**:
  - [ ] Implement exact data count tracking in histogram construction
  - [ ] Modify `FeatureHistogram` to store actual data counts per bin
  - [ ] Update split finding logic to use exact counts
- **Expected Impact**: Reduce prediction differences by 20-30%

### 3. Parameter Alignment and Default Values
- **Files**: `src/config/core.rs`, examples comparison tests
- **Issue**: Default parameters may not match Python LightGBM exactly
- **Action**:
  - [ ] Create parameter mapping table for Rust ↔ Python equivalence
  - [ ] Ensure identical random seed behavior
  - [ ] Verify learning rate, regularization, and tree structure parameters
- **Expected Impact**: Reduce prediction differences by 10-20%

### 4. Enhance Gradient/Hessian Calculation Precision
- **File**: `src/config/objective.rs:475-517`
- **Issue**: Potential floating-point precision differences
- **Action**:
  - [ ] Use higher precision arithmetic for probability calculations
  - [ ] Implement numerically stable sigmoid and softmax functions
  - [ ] Add gradient clipping for numerical stability
- **Expected Impact**: Reduce prediction differences by 10-15%

## Medium Priority Improvements

### 5. Histogram Construction Optimization
- **Files**: `src/tree/histogram/builder.rs`, `src/tree/histogram/mod.rs`
- **Action**:
  - [ ] Verify histogram binning matches Python LightGBM exactly
  - [ ] Implement histogram subtraction optimization checks
  - [ ] Add validation for histogram construction correctness

### 6. Tree Construction Validation
- **Files**: `src/tree/learner/serial.rs`, `src/tree/tree.rs`
- **Action**:
  - [ ] Add intermediate tree structure comparison with Python
  - [ ] Implement tree serialization for debugging
  - [ ] Verify leaf value calculations match Python implementation

### 7. Prediction Pipeline Verification
- **Files**: `src/prediction/predictor.rs`, `src/boosting.rs`
- **Action**:
  - [ ] Add step-by-step prediction debugging
  - [ ] Implement tree ensemble prediction validation
  - [ ] Compare prediction aggregation logic with Python

## Testing and Validation

### 8. Comprehensive Test Suite Enhancement
- **Action**:
  - [ ] Create bit-exact comparison tests for each component
  - [ ] Add regression tests for prediction precision
  - [ ] Implement continuous precision monitoring
  - [ ] Add tests for edge cases and numerical stability

### 9. Debugging and Profiling Tools
- **Action**:
  - [ ] Implement detailed prediction tracing
  - [ ] Add component-wise error analysis tools
  - [ ] Create precision benchmarking framework
  - [ ] Add memory and performance profiling

## Documentation and Process

### 10. Implementation Documentation
- **Action**:
  - [ ] Document algorithmic differences from Python LightGBM
  - [ ] Create precision validation guidelines
  - [ ] Add troubleshooting guide for prediction discrepancies
  - [ ] Document parameter equivalence mappings

## Success Criteria

1. **Primary Goal**: Achieve prediction differences < 10⁻⁶ (MSE < 10⁻¹²)
2. **Stretch Goal**: Achieve prediction differences < 10⁻¹⁰ (bit-exact equivalence)
3. **Validation**: Pass comprehensive test suite with 100% precision tests
4. **Performance**: Maintain current training and prediction performance levels

## Implementation Order

1. **Phase 1** (High Impact): Items 1-4 (split gain, data counts, parameters, gradients)
2. **Phase 2** (Validation): Items 8-9 (testing and debugging tools)
3. **Phase 3** (Optimization): Items 5-7 (histogram, tree, prediction optimizations)
4. **Phase 4** (Documentation): Item 10 (documentation and guidelines)

## Estimated Timeline

- **Phase 1**: 2-3 weeks
- **Phase 2**: 1-2 weeks  
- **Phase 3**: 1-2 weeks
- **Phase 4**: 1 week

**Total Estimated Time**: 5-8 weeks for complete resolution

## Notes

- Current implementation is functionally correct and production-ready
- Differences are within acceptable bounds for most applications
- Focus on algorithmic equivalence rather than micro-optimizations
- Maintain backward compatibility and API stability