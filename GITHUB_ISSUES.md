# GitHub Issues for Pure Rust LightGBM Implementation Gaps

Based on comprehensive E2E testing, the following GitHub issues should be created to track implementation gaps and bugs discovered.

## Critical Implementation Gaps

### Issue #1: Implement Core Gradient Boosting Training Algorithm
**Priority:** Critical  
**Labels:** enhancement, core-algorithm, high-priority  
**Milestone:** v0.1.0 - Core Functionality  

#### Description
The core gradient boosting training algorithm is not implemented. Currently, `LGBMRegressor::fit()` and `LGBMClassifier::fit()` methods return `NotImplemented` errors.

#### Requirements
- Implement GBDT training loop as specified in design document Section 2.3
- Support for regression, binary classification, and multiclass classification objectives
- Gradient and hessian computation for different loss functions
- Integration with tree learner subsystem

#### Acceptance Criteria
- [ ] `LGBMRegressor::fit()` successfully trains regression models
- [ ] `LGBMClassifier::fit()` successfully trains classification models  
- [ ] Training metrics are computed and logged
- [ ] Early stopping works when configured
- [ ] All existing E2E tests pass

#### Files to Modify
- `src/boosting/gbdt.rs`
- `src/boosting/objective/*.rs`
- `src/lib.rs` (remove NotImplemented returns)

---

### Issue #2: Implement Tree Learning Subsystem
**Priority:** Critical  
**Labels:** enhancement, tree-learning, high-priority  
**Milestone:** v0.1.0 - Core Functionality

#### Description
The tree learning subsystem is not implemented. This is required for the gradient boosting algorithm to function.

#### Requirements
- Implement histogram-based tree construction as specified in design document Section 2.4
- Support for serial and parallel tree learners
- Split finding algorithms with gain calculation
- Tree node construction and management

#### Acceptance Criteria
- [ ] Trees can be built from gradient/hessian data
- [ ] Histogram construction works efficiently
- [ ] Split finding produces optimal splits
- [ ] Tree prediction works correctly
- [ ] Memory usage is optimized

#### Files to Modify
- `src/tree/learner/*.rs`
- `src/tree/split/*.rs` 
- `src/tree/histogram/*.rs`
- `src/tree/tree.rs`

---

### Issue #3: Implement Model Prediction Pipeline
**Priority:** Critical  
**Labels:** enhancement, prediction, high-priority  
**Milestone:** v0.1.0 - Core Functionality

#### Description
Model prediction functionality is not implemented. Currently, `predict()` methods return `NotImplemented` errors.

#### Requirements
- Implement tree traversal for prediction as specified in design document Section 2.6
- Support for batch and single prediction
- Raw scores and probability prediction modes
- Efficient memory usage for large prediction sets

#### Acceptance Criteria
- [ ] `predict()` methods return correct predictions
- [ ] Batch prediction works efficiently
- [ ] Prediction consistency across multiple calls
- [ ] Support for different prediction types (raw, probability)
- [ ] Performance meets benchmarks (>1000 predictions/sec for small models)

#### Files to Modify
- `src/prediction.rs`
- `src/lib.rs` (remove NotImplemented returns)
- `src/tree/tree.rs` (prediction methods)

---

## High Priority Bugs

### Issue #4: Fix CSV Dataset Feature Count Bug
**Priority:** High  
**Labels:** bug, dataset-loading, data-pipeline  
**Milestone:** v0.1.0 - Core Functionality

#### Description
CSV dataset loading incorrectly counts target and weight columns as features. E2E test `test_e2e_data_pipeline_csv_loading` fails because `dataset.num_features()` returns 6 instead of expected 5.

#### Root Cause
The CSV loader is including the target column and weight column in the feature count when it should exclude them.

#### Expected Behavior
- Target column should not be counted as a feature
- Weight column should not be counted as a feature
- `num_features()` should return only the actual feature count

#### Steps to Reproduce
1. Create CSV with 5 features + 1 target + 1 weight column
2. Load using `DatasetFactory::from_csv()`
3. Call `dataset.num_features()`
4. Observe: returns 6, expected: 5

#### Files to Fix
- `src/dataset/loader/csv.rs`
- `src/dataset/dataset.rs` (possibly)

---

### Issue #5: Implement Missing Value Detection and Handling
**Priority:** High  
**Labels:** enhancement, missing-values, data-preprocessing  
**Milestone:** v0.2.0 - Data Processing

#### Description
Missing value detection and handling is incomplete. The `has_missing_values()` method exists but behavior is not properly implemented.

#### Requirements
- Detect missing values (NaN, null, empty strings)
- Implement missing value imputation strategies
- Support for different missing value types as specified in design document Section 2.2.2

#### Acceptance Criteria
- [ ] `has_missing_values()` correctly detects missing values
- [ ] Support for NaN, Zero, and custom missing value representations
- [ ] Missing value imputation works (mean, median, mode)
- [ ] E2E test `test_e2e_data_pipeline_missing_values` passes

#### Files to Modify
- `src/dataset/preprocessing/missing.rs`
- `src/dataset/dataset.rs`

---

## Medium Priority Implementation Gaps

### Issue #6: Implement Model Serialization and Persistence
**Priority:** Medium  
**Labels:** enhancement, serialization, model-persistence  
**Milestone:** v0.2.0 - Model Management

#### Description
Model save/load functionality is not implemented. This is required for production model deployment.

#### Requirements
- Save trained models to disk in multiple formats (bincode, JSON, LightGBM-compatible)
- Load saved models and restore prediction capability
- Version compatibility and migration support

#### Acceptance Criteria
- [ ] `save_model()` methods work for all model types
- [ ] `load_model()` methods restore complete model state
- [ ] Saved models are portable across systems
- [ ] Compatible with original LightGBM format where possible

#### Files to Modify
- `src/io/serialization/*.rs`
- `src/lib.rs` (remove NotImplemented returns)

---

### Issue #7: Implement Feature Importance Calculation
**Priority:** Medium  
**Labels:** enhancement, interpretability, feature-importance  
**Milestone:** v0.3.0 - Advanced Features

#### Description
Feature importance calculation is not implemented. This is important for model interpretability.

#### Requirements
- Split-based feature importance
- Gain-based feature importance  
- SHAP value computation
- Integration with trained models

#### Acceptance Criteria
- [ ] `feature_importance()` returns correct importance scores
- [ ] Support for different importance types
- [ ] SHAP values can be computed
- [ ] Performance is reasonable for large models

#### Files to Modify
- `src/prediction.rs` (feature importance methods)
- `src/tree/tree.rs` (tree-level importance)

---

### Issue #8: Implement Hyperparameter Optimization
**Priority:** Medium  
**Labels:** enhancement, hyperparameter-optimization, automl  
**Milestone:** v0.3.0 - Advanced Features

#### Description
Hyperparameter optimization functionality is not implemented. This is important for automated model tuning.

#### Requirements
- Support for multiple optimization algorithms (grid search, random search, Bayesian optimization)
- Cross-validation integration
- Parameter space definition and validation
- Progress tracking and early termination

#### Acceptance Criteria
- [ ] `optimize_hyperparameters()` function works
- [ ] Multiple optimization strategies available
- [ ] Integration with cross-validation
- [ ] Reasonable optimization performance

#### Files to Modify
- `src/hyperopt.rs`
- New files for optimization algorithms

---

## Long-term Implementation Gaps

### Issue #9: Implement GPU Acceleration Framework
**Priority:** Low  
**Labels:** enhancement, gpu-acceleration, performance  
**Milestone:** v0.4.0 - Performance Optimization

#### Description
GPU acceleration through CubeCL is not implemented. This is important for large-scale training performance.

#### Requirements
- CubeCL integration for histogram construction
- GPU tree learning algorithms
- Memory management for GPU operations
- Fallback to CPU when GPU unavailable

#### Files to Modify
- `src/gpu/*.rs`
- Integration with existing training pipeline

---

### Issue #10: Implement Model Ensemble Methods
**Priority:** Low  
**Labels:** enhancement, ensemble-methods, advanced-ml  
**Milestone:** v0.4.0 - Advanced Features

#### Description
Model ensemble functionality is not implemented. This provides improved prediction accuracy through model combination.

#### Requirements
- Multiple ensemble strategies (voting, averaging, stacking)
- Weight optimization for ensemble members
- Cross-validation for ensemble validation

#### Files to Modify
- `src/ensemble.rs`
- Integration with existing model interfaces

---

## Test and Infrastructure Issues

### Issue #11: Add Performance Benchmarks to E2E Tests
**Priority:** Medium  
**Labels:** testing, performance, benchmarks  
**Milestone:** v0.2.0 - Quality Assurance

#### Description
E2E tests should include concrete performance benchmarks with pass/fail thresholds.

#### Requirements
- Training time benchmarks
- Prediction throughput benchmarks
- Memory usage benchmarks
- Accuracy benchmarks against reference datasets

#### Files to Modify
- `tests/comprehensive_e2e_tests.rs`
- New benchmark test files

---

### Issue #12: Fix Compiler Warnings
**Priority:** Low  
**Labels:** code-quality, maintenance  
**Milestone:** v0.1.1 - Code Quality

#### Description
There are numerous compiler warnings related to hidden lifetime parameters and unused code that should be addressed.

#### Requirements
- Fix lifetime parameter warnings in trait definitions
- Remove or properly annotate unused code
- Ensure clean compilation with no warnings

#### Files to Modify
- `src/core/traits.rs`
- Various files with unused code

---

## Issue Template

For creating actual GitHub issues, use this template:

```markdown
**Priority:** [Critical/High/Medium/Low]
**Component:** [core-algorithm/dataset/prediction/etc.]
**Type:** [bug/enhancement/documentation]

### Description
[Clear description of the issue]

### Requirements
- [ ] Requirement 1
- [ ] Requirement 2

### Acceptance Criteria
- [ ] Criteria 1
- [ ] Criteria 2

### Files to Modify
- `path/to/file1.rs`
- `path/to/file2.rs`

### Related Issues
- Depends on: #X
- Blocks: #Y

### Additional Context
[Any additional context, screenshots, or links]
```

## Prioritization Summary

1. **Must Have (v0.1.0):** Issues #1-5 (Core training, tree learning, prediction, CSV bug, missing values)
2. **Should Have (v0.2.0):** Issues #6-7, #11 (Model persistence, feature importance, benchmarks)
3. **Could Have (v0.3.0):** Issues #8 (Hyperparameter optimization)
4. **Won't Have Initially (v0.4.0+):** Issues #9-10, #12 (GPU acceleration, ensembles, warnings)

This prioritization ensures a functional LightGBM implementation in v0.1.0 with progressive enhancement in subsequent versions.