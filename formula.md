# LightGBM Boosting Algorithms: Mathematical Formulas and Implementation

This document provides a comprehensive analysis of the mathematical formulas and algorithms implemented in the LightGBM boosting framework, extracted from the source code analysis.

## Table of Contents
1. [Core Gradient Boosting Decision Trees (GBDT)](#core-gradient-boosting-decision-trees-gbdt)
2. [GOSS (Gradient One-Side Sampling)](#goss-gradient-one-side-sampling)
3. [DART (Dropouts meet Multiple Additive Regression Trees)](#dart-dropouts-meet-multiple-additive-regression-trees)
4. [Random Forest](#random-forest)
5. [Bagging Strategy](#bagging-strategy)
6. [Prediction Mechanisms](#prediction-mechanisms)
7. [Implementation Details](#implementation-details)

---

## Core Gradient Boosting Decision Trees (GBDT)

### Fundamental Boosting Formula

The core gradient boosting update rule implemented in `gbdt.cpp:TrainOneIter()`:

```
F_{m+1}(x) = F_m(x) + η · h_m(x)
```

Where:
- `F_m(x)` is the model prediction after m iterations
- `η` is the learning rate (shrinkage_rate_)
- `h_m(x)` is the new tree at iteration m

### Gradient and Hessian Computation

From `gbdt.cpp:Boosting()`, the objective function computes:

```
g_i = ∂L(y_i, F(x_i)) / ∂F(x_i)    (First-order gradients)
h_i = ∂²L(y_i, F(x_i)) / ∂F(x_i)²  (Second-order gradients/Hessians)
```

### Newton-Raphson Method for Leaf Values

The optimal leaf value calculation uses Newton-Raphson method:

```
w_j = -Σ_{i∈I_j} g_i / (Σ_{i∈I_j} h_i + λ)
```

Where:
- `I_j` is the set of samples in leaf j
- `λ` is the regularization parameter
- `g_i`, `h_i` are gradients and hessians for sample i

### Split Finding Criterion

The gain formula for finding optimal splits:

```
Gain = (1/2) · [G_L²/(H_L + λ) + G_R²/(H_R + λ) - G²/(H + λ)] - γ
```

Where:
- `G_L = Σ_{i∈I_L} g_i` (sum of gradients in left child)
- `G_R = Σ_{i∈I_R} g_i` (sum of gradients in right child)
- `H_L = Σ_{i∈I_L} h_i` (sum of hessians in left child)
- `H_R = Σ_{i∈I_R} h_i` (sum of hessians in right child)
- `γ` is the minimum split gain threshold

### Multi-class Extension

For multi-class classification with K classes:

```
F_m^{(k)}(x) = F_{m-1}^{(k)}(x) + η · h_m^{(k)}(x)
```

Where k ∈ {1, 2, ..., K} and each class has its own tree per iteration.

---

## GOSS (Gradient One-Side Sampling)

From `goss.hpp`, GOSS implements gradient-based sampling:

### Gradient Magnitude Calculation

```
|grad_i| = |Σ_{k=1}^K g_i^{(k)} · h_i^{(k)}|
```

### Two-stage Sampling Process

1. **Top-k Selection**: Select samples with largest gradients
   ```
   A = {i : |grad_i| ≥ threshold}
   ```
   Where threshold is the (top_rate × n)-th largest gradient

2. **Random Sampling**: From remaining samples
   ```
   B = {random sample from remaining with probability other_rate}
   ```

### Weight Compensation

For samples in B, apply weight multiplication:

```
w_i = w_i × (1 - top_rate) / other_rate
```

This compensates for the undersampling of small gradient samples.

### GOSS Algorithm Implementation

```
Input: Training data D, top_rate a, other_rate b
1. Sort samples by |grad_i| in descending order
2. A = top a × |D| samples
3. B = random sample of b × |D| from remaining samples
4. For each sample i in B: multiply gradients and hessians by (1-a)/b
5. Return A ∪ B
```

---

## DART (Dropouts meet Multiple Additive Regression Trees)

From `dart.hpp`, DART implements dropout regularization:

### Dropout Probability

```
P(drop tree i) = drop_rate × weight_i × (number_of_trees / sum_of_all_weights)
```

### Normalization Process

After dropping k trees, the normalization involves:

1. **Drop Phase**: Set dropped tree weights to -1
2. **Validation Update**: 
   ```
   tree_weight = tree_weight × (1 / (k + 1))
   ```
3. **Training Update**:
   ```
   tree_weight = tree_weight × (-k)
   ```

### Learning Rate Adjustment

```
η_effective = η / (1 + number_of_dropped_trees)
```

In XGBoost mode:
```
η_effective = η / (η + number_of_dropped_trees)
```

### DART Algorithm

```
For each iteration:
1. Select trees to drop based on dropout probability
2. Temporarily remove selected trees from ensemble
3. Train new tree on modified ensemble
4. Normalize weights of dropped trees
5. Add new tree with adjusted learning rate
```

---

## Random Forest

From `rf.hpp`, Random Forest implements:

### Averaging Formula

```
F(x) = (1/M) × Σ_{m=1}^M T_m(x)
```

Where M is the number of trees and averaging is enabled (`average_output_ = true`).

### No Shrinkage

```
η = 1.0  (shrinkage_rate_ = 1.0)
```

### Bootstrap Sampling

Each tree is trained on a bootstrap sample of the data with replacement.

### Score Update for RF

```
Score_new = (Score_old × (m-1) + Tree_prediction) / m
```

Where m is the current tree number.

---

## Bagging Strategy

From `bagging.hpp`, the bagging implementation:

### Sample Selection Probability

```
P(select sample i) = bagging_fraction
```

### Balanced Bagging

For imbalanced datasets:

```
P(select positive sample) = pos_bagging_fraction
P(select negative sample) = neg_bagging_fraction
```

### Query-based Bagging

For ranking problems:
```
1. Select queries with probability bagging_fraction
2. Include all documents in selected queries
```

---

## Prediction Mechanisms

From `gbdt_prediction.cpp`:

### Raw Prediction

```
F(x) = Σ_{m=start}^{end} Σ_{k=1}^K T_m^{(k)}(x)
```

### Final Prediction

```
Output = ObjectiveFunction.ConvertOutput(F(x))
```

### Early Stopping in Prediction

The prediction process can be stopped early if:
```
early_stop_callback(current_prediction) == true
```

---

## Implementation Details

### Memory Layout

- **Gradients/Hessians**: Stored with offset per tree/class
  ```
  offset = tree_id × num_data
  gradient_ptr = gradients + offset
  ```

### Parallel Processing

- **OpenMP**: Used for parallel gradient computation
- **Thread Safety**: Mutex protection for model initialization

### GPU Support

- **CUDA**: Conditional compilation for GPU acceleration
- **Memory Management**: Separate host/device memory buffers

### Regularization Parameters

- **L1 Regularization**: Applied in tree construction
- **L2 Regularization**: λ parameter in leaf value calculation
- **Minimum Split Gain**: γ threshold for split acceptance

### Optimization Techniques

1. **Histogram-based Learning**: Efficient feature value discretization
2. **Leaf-wise Growth**: Grows trees leaf by leaf rather than level by level
3. **Categorical Features**: Native support for categorical variables
4. **Memory Efficiency**: Gradient buffer reuse and subset handling

---

## Algorithm Complexity

### Time Complexity
- **Training**: O(n × d × log(n)) per iteration
- **Prediction**: O(log(n)) per sample for single tree
- **GOSS**: O(n × log(n)) for sorting + O(n) for sampling

### Space Complexity
- **Gradients/Hessians**: O(n × K) where K is number of classes
- **Trees**: O(number_of_leaves × number_of_trees)
- **Bagging**: O(n) for indices storage

---

## Key Innovations in LightGBM

1. **Gradient One-Side Sampling (GOSS)**: Reduces computation while maintaining accuracy
2. **Exclusive Feature Bundling (EFB)**: Reduces feature space
3. **Leaf-wise Growth**: More efficient than level-wise growth
4. **Histogram-based Learning**: Faster than pre-sorted algorithms
5. **Network Communication**: Efficient distributed training protocols

---

## References

This analysis is based on the LightGBM source code in the `/src/boosting/` directory:
- `gbdt.cpp`, `gbdt.h` - Core GBDT implementation
- `goss.hpp` - GOSS sampling strategy
- `dart.hpp` - DART regularization
- `rf.hpp` - Random Forest implementation
- `bagging.hpp` - Bagging strategy
- `gbdt_prediction.cpp` - Prediction mechanisms

The mathematical formulations are extracted directly from the algorithmic implementations in the source code, providing the exact formulas used in the LightGBM framework.