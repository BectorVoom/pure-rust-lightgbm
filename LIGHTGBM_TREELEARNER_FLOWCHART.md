# LightGBM TreeLearner Implementation Flowchart

## Overview
This document provides a comprehensive flowchart illustrating the implementation logic of the **TreeLearner** component in LightGBM's C++ codebase, specifically focusing on the `SerialTreeLearner` class located in `C_lightgbm/treelearner/`.

## Legend

### Flowchart Symbols
- **Rectangle** ⬜: Process/Computation step
- **Diamond** ◇: Decision point/Conditional branch
- **Rounded Rectangle** ⚪: Start/End/Data structures
- **Parallelogram** ⬟: Input/Output operations
- **Hexagon** ⬡: Mathematical computation

### Data Type Abbreviations
- `int`: 32-bit integer
- `data_size_t`: Platform-specific integer for data sizes
- `score_t`: Floating-point type for gradients/hessians (typically `double`)
- `hist_t`: Histogram data type (typically `double`)
- `uint32_t`: 32-bit unsigned integer
- `std::vector<T>`: Dynamic array of type T
- `std::unique_ptr<T>`: Smart pointer to type T

## Core Data Structures

### SplitInfo Structure
```cpp
struct SplitInfo {
    int feature = -1;                    // Feature index for split
    uint32_t threshold = 0;              // Split threshold value
    data_size_t left_count = 0;          // Data count in left child
    data_size_t right_count = 0;         // Data count in right child
    double left_output = 0.0;            // Left leaf output value
    double right_output = 0.0;           // Right leaf output value
    double gain = kMinScore;             // Split gain value
    double left_sum_gradient = 0;        // Sum of gradients in left child
    double left_sum_hessian = 0;         // Sum of hessians in left child
    double right_sum_gradient = 0;       // Sum of gradients in right child
    double right_sum_hessian = 0;        // Sum of hessians in right child
    bool default_left = true;            // Default direction for missing values
    int8_t monotone_type = 0;           // Monotonic constraint type
    std::vector<uint32_t> cat_threshold; // Categorical split thresholds
};
```

### LeafSplits Structure
```cpp
class LeafSplits {
    int leaf_index_;                     // Current leaf index
    data_size_t num_data_in_leaf_;      // Number of data points in leaf
    double sum_gradients_;               // Sum of gradients in leaf
    double sum_hessians_;               // Sum of hessians in leaf
    const data_size_t* data_indices_;   // Indices of data in leaf
    double weight_;                     // Leaf weight for regularization
};
```

## Main Algorithm Flow

```mermaid
flowchart TD
    Start([🟢 Start: SerialTreeLearner::Train]) --> Init[⬜ Initialize Training Parameters]
    
    Init --> BeforeTrain[⬜ BeforeTrain(): Setup Data Structures]
    BeforeTrain --> CreateTree[⬜ Create Tree with num_leaves capacity]
    CreateTree --> SetRoot[⬜ Set Root Leaf Output]
    
    SetRoot --> ForcedSplits{◇ Forced Splits Defined?}
    ForcedSplits -->|Yes| ApplyForced[⬜ ForceSplits(): Apply JSON-defined splits]
    ForcedSplits -->|No| SplitLoop[⬜ Initialize split=0, left_leaf=0, right_leaf=-1]
    ApplyForced --> SplitLoop
    
    SplitLoop --> CheckSplits{◇ split < num_leaves - 1?}
    CheckSplits -->|No| QuantizedCheck{◇ Use Quantized Gradients?}
    CheckSplits -->|Yes| BeforeFindBest[⬜ BeforeFindBestSplit()]
    
    BeforeFindBest --> DepthCheck{◇ Max Depth Exceeded?}
    DepthCheck -->|Yes| SkipLeaf[⬜ Set gain = kMinScore]
    DepthCheck -->|No| DataCheck{◇ Sufficient Data in Leaves?}
    DataCheck -->|No| SkipLeaf
    DataCheck -->|Yes| SetupHistograms[⬜ Setup Histogram Arrays]
    
    SetupHistograms --> FindBestSplits[⬜ FindBestSplits(): Core splitting algorithm]
    FindBestSplits --> SelectBest[⬜ Select leaf with maximum gain]
    SelectBest --> ValidGain{◇ Gain > 0?}
    ValidGain -->|No| EndSplitting[🔴 End: No positive gain]
    ValidGain -->|Yes| SplitTree[⬜ Split(): Partition tree and data]
    
    SplitTree --> IncrementSplit[⬜ split++, update left_leaf, right_leaf]
    IncrementSplit --> SplitLoop
    
    QuantizedCheck -->|Yes| RenewLeaf[⬜ Renew leaf outputs with quantized gradients]
    QuantizedCheck -->|No| ReturnTree[🟢 Return trained tree]
    RenewLeaf --> ReturnTree
    
    SkipLeaf --> SplitLoop
```

## Detailed Split Finding Algorithm

```mermaid
flowchart TD
    FindStart([🟢 Start: FindBestSplits]) --> FeatureLoop[⬜ Parallel loop over features]
    
    FeatureLoop --> CheckUsed{◇ Feature used by tree?}
    CheckUsed -->|No| SkipFeature[⬜ Continue to next feature]
    CheckUsed -->|Yes| CheckSplittable{◇ Parent histogram splittable?}
    CheckSplittable -->|No| SetUnsplittable[⬜ Mark feature as unsplittable]
    CheckSplittable -->|Yes| ConstructHist[⬜ ConstructHistograms()]
    
    SetUnsplittable --> SkipFeature
    ConstructHist --> HistogramReady[⬜ Histograms constructed for smaller/larger leaves]
    HistogramReady --> FindFromHist[⬜ FindBestSplitsFromHistograms()]
    
    FindFromHist --> ThreadLoop[⬜ Parallel thread loop over features]
    ThreadLoop --> FixHistogram[⬜ Fix histogram boundaries and missing values]
    FixHistogram --> ComputeBest[⬜ ComputeBestSplitForFeature()]
    
    ComputeBest --> SubtractHist{◇ Use histogram subtraction?}
    SubtractHist -->|Yes| SubtractOp[⬜ Subtract smaller from parent histogram]
    SubtractHist -->|No| ConstructLarger[⬜ Construct larger leaf histogram directly]
    
    SubtractOp --> ComputeLarger[⬜ ComputeBestSplitForFeature() for larger leaf]
    ConstructLarger --> ComputeLarger
    ComputeLarger --> UpdateBest[⬜ Update best splits per leaf]
    
    UpdateBest --> SkipFeature
    SkipFeature --> EndFeatures{◇ All features processed?}
    EndFeatures -->|No| FeatureLoop
    EndFeatures -->|Yes| ReturnBest[🟢 Return best splits]
```

## Mathematical Formulas

### 1. Leaf Output Calculation

The leaf output is calculated using the Newton-Raphson method with regularization:

**Formula:**
```
leaf_output = -ThresholdL1(sum_gradients, λ₁) / (sum_hessians + λ₂)
```

**With L1 regularization:**
```
ThresholdL1(s, λ₁) = sign(s) × max(0, |s| - λ₁)
```

**Variables:**
- `sum_gradients` (double): Sum of first-order gradients in leaf
- `sum_hessians` (double): Sum of second-order gradients (Hessians) in leaf
- `λ₁` (double): L1 regularization parameter (`config->lambda_l1`)
- `λ₂` (double): L2 regularization parameter (`config->lambda_l2`)

**With path smoothing:**
```
smoothed_output = output × (n/α) / (n/α + 1) + parent_output / (n/α + 1)
```
- `α` (double): Path smoothing parameter (`config->path_smooth`)
- `n` (data_size_t): Number of data points in leaf
- `parent_output` (double): Output of parent node

### 2. Split Gain Calculation

The gain from a split is computed as the improvement in objective function:

**Formula:**
```
gain = G_left²/(H_left + λ₂) + G_right²/(H_right + λ₂) - G_parent²/(H_parent + λ₂)
```

**With L1 regularization:**
```
gain = ThresholdL1(G_left, λ₁)²/(H_left + λ₂) + ThresholdL1(G_right, λ₁)²/(H_right + λ₂) - ThresholdL1(G_parent, λ₁)²/(H_parent + λ₂)
```

**Variables:**
- `G_left`, `G_right`, `G_parent` (double): Sum of gradients in left child, right child, and parent
- `H_left`, `H_right`, `H_parent` (double): Sum of hessians in left child, right child, and parent

### 3. Histogram Construction

For each feature bin, gradients and hessians are accumulated:

**Formula:**
```
histogram[bin_idx].gradient += gradient[data_idx]
histogram[bin_idx].hessian += hessian[data_idx]
```

**With quantized gradients:**
```
int_gradient = gradient / grad_scale
int_hessian = hessian / hess_scale
packed_value = (int_gradient << 32) | int_hessian
```

**Variables:**
- `histogram` (hist_t[]): Array of histogram bins for feature
- `bin_idx` (int): Bin index for current data point
- `data_idx` (data_size_t): Index of current data point
- `grad_scale`, `hess_scale` (double): Scaling factors for quantization

### 4. Monotonic Constraints

When monotonic constraints are enabled, the split gain is penalized:

**Formula:**
```
constrained_gain = gain × monotone_penalty
```

**Monotonic constraint check:**
```
if (monotone_type > 0 && left_output > right_output) return 0;
if (monotone_type < 0 && left_output < right_output) return 0;
```

**Variables:**
- `monotone_penalty` (double): Penalty factor (`config->monotone_penalty`)
- `monotone_type` (int8_t): Constraint type (-1: decreasing, 0: none, 1: increasing)

## Algorithm Components Detail

### BeforeTrain() Process
```mermaid
flowchart TD
    BeforeStart([🟢 Start: BeforeTrain]) --> ResetPool[⬜ Reset histogram pool]
    ResetPool --> SampleFeatures[⬜ Column sampler: Reset features by tree]
    SampleFeatures --> InitTrain[⬜ Initialize training data with selected features]
    InitTrain --> InitPartition[⬜ DataPartition::Init() - Set all data to root leaf]
    InitPartition --> ResetConstraints[⬜ Reset monotonic constraints]
    ResetConstraints --> ResetSplits[⬜ Reset best_split_per_leaf_ array]
    ResetSplits --> ComputeRoot[⬡ Compute root leaf statistics]
    
    ComputeRoot --> UseBagging{◇ Using bagging?}
    UseBagging -->|Yes| BaggingInit[⬜ smaller_leaf_splits_->Init(leaf=0, partition, gradients, hessians)]
    UseBagging -->|No| FullInit[⬜ smaller_leaf_splits_->Init(gradients, hessians)]
    
    BaggingInit --> InitLarger[⬜ larger_leaf_splits_->Init()]
    FullInit --> InitLarger
    InitLarger --> CEGBCheck{◇ Cost-effective gradient boosting enabled?}
    CEGBCheck -->|Yes| CEGBInit[⬜ cegb_->BeforeTrain()]
    CEGBCheck -->|No| QuantBitsCheck{◇ Quantized gradients enabled?}
    CEGBInit --> QuantBitsCheck
    QuantBitsCheck -->|Yes| SetHistBits[⬜ Set histogram bits for root leaf]
    QuantBitsCheck -->|No| BeforeEnd[🟢 End: BeforeTrain]
    SetHistBits --> BeforeEnd
```

### ComputeBestSplitForFeature() Process
```mermaid
flowchart TD
    ComputeStart([🟢 Start: ComputeBestSplitForFeature]) --> CheckNumerical{◇ Numerical feature?}
    CheckNumerical -->|Yes| CheckMonotone{◇ Monotonic constraints enabled?}
    CheckNumerical -->|No| CategoricalSplit[⬡ FindBestThresholdCategorical()]
    
    CheckMonotone -->|Yes| RecomputeConstraints[⬜ Recompute constraints for feature]
    CheckMonotone -->|No| NumericalSplit[⬡ FindBestThreshold()]
    RecomputeConstraints --> NumericalSplit
    
    NumericalSplit --> CheckQuantized{◇ Using quantized gradients?}
    CheckQuantized -->|Yes| QuantizedThreshold[⬡ FindBestThresholdInt()]
    CheckQuantized -->|No| SetFeature[⬜ new_split.feature = real_feature_index]
    QuantizedThreshold --> SetFeature
    CategoricalSplit --> SetFeature
    
    SetFeature --> CEGBCheck{◇ CEGB enabled?}
    CEGBCheck -->|Yes| ApplyCEGB[⬡ new_split.gain -= cegb_->DeltaGain()]
    CEGBCheck -->|No| MonotoneCheck{◇ Monotonic split?}
    ApplyCEGB --> MonotoneCheck
    
    MonotoneCheck -->|Yes| ApplyPenalty[⬡ new_split.gain *= monotone_penalty]
    MonotoneCheck -->|No| CheckBetter{◇ new_split > best_split && feature_used?}
    ApplyPenalty --> CheckBetter
    
    CheckBetter -->|Yes| UpdateBest[⬜ *best_split = new_split]
    CheckBetter -->|No| ComputeEnd[🟢 End: ComputeBestSplitForFeature]
    UpdateBest --> ComputeEnd
```

### Split() Process - Tree and Data Partitioning
```mermaid
flowchart TD
    SplitStart([🟢 Start: Split Tree and Data]) --> GetSplitInfo[⬜ Get best_split_info for selected leaf]
    GetSplitInfo --> GetInnerFeature[⬜ inner_feature_index = train_data_->InnerFeatureIndex()]
    GetInnerFeature --> CEGBUpdate{◇ CEGB enabled?}
    CEGBUpdate -->|Yes| UpdateCEGB[⬜ cegb_->UpdateLeafBestSplits()]
    CEGBUpdate -->|No| SetLeftLeaf[⬜ left_leaf = best_leaf]
    UpdateCEGB --> SetLeftLeaf
    
    SetLeftLeaf --> GetNextId[⬜ next_leaf_id = tree->NextLeafId()]
    GetNextId --> BeforeConstraints[⬜ constraints_->BeforeSplit()]
    BeforeConstraints --> CheckNumerical{◇ Numerical split?}
    
    CheckNumerical -->|Yes| NumericalSplit[⬜ Handle numerical split]
    CheckNumerical -->|No| CategoricalSplit[⬜ Handle categorical split]
    
    NumericalSplit --> GetThreshold[⬜ threshold_double = train_data_->RealThreshold()]
    GetThreshold --> PartitionData[⬜ data_partition_->Split()]
    PartitionData --> UpdateCounts{◇ Update counts needed?}
    UpdateCounts -->|Yes| UpdateDataCounts[⬜ Update left_count, right_count from data_partition_]
    UpdateCounts -->|No| TreeSplitNum[⬜ right_leaf = tree->Split() for numerical]
    UpdateDataCounts --> TreeSplitNum
    
    CategoricalSplit --> ConstructBitset[⬜ Construct categorical bitset from thresholds]
    ConstructBitset --> PartitionCat[⬜ data_partition_->Split() with bitset]
    PartitionCat --> UpdateCountsCat{◇ Update counts needed?}
    UpdateCountsCat -->|Yes| UpdateDataCountsCat[⬜ Update left_count, right_count]
    UpdateCountsCat -->|No| TreeSplitCat[⬜ right_leaf = tree->SplitCategorical()]
    UpdateDataCountsCat --> TreeSplitCat
    
    TreeSplitNum --> InitLeafSplits[⬜ Initialize smaller_leaf_splits_ and larger_leaf_splits_]
    TreeSplitCat --> InitLeafSplits
    
    InitLeafSplits --> CheckQuantGrad{◇ Quantized gradients?}
    CheckQuantGrad -->|Yes| QuantizedInit[⬜ Init with quantized gradient/hessian sums]
    CheckQuantGrad -->|No| RegularInit[⬜ Init with regular gradient/hessian sums]
    QuantizedInit --> SetHistBits[⬜ Set histogram bits for new leaves]
    RegularInit --> DebugCheck{◇ Debug mode?}
    SetHistBits --> DebugCheck
    
    DebugCheck -->|Yes| ValidateSplit[⬜ CheckSplit() - Validate split correctness]
    DebugCheck -->|No| UpdateConstraints[⬜ constraints_->Update()]
    ValidateSplit --> UpdateConstraints
    
    UpdateConstraints --> RecomputeLeaves[⬜ Recompute outputs for constraint-affected leaves]
    RecomputeLeaves --> SplitEnd[🟢 End: Split completed]
```

## Key Variables and Their Purposes

### Training Data Variables
| Variable | Type | Purpose |
|----------|------|---------|
| `num_data_` | `data_size_t` | Total number of training samples |
| `num_features_` | `int` | Number of features in dataset |
| `train_data_` | `const Dataset*` | Pointer to training dataset |
| `gradients_` | `const score_t*` | First-order gradients for current iteration |
| `hessians_` | `const score_t*` | Second-order gradients (Hessians) for current iteration |

### Tree Structure Variables
| Variable | Type | Purpose |
|----------|------|---------|
| `data_partition_` | `std::unique_ptr<DataPartition>` | Manages data assignment to tree leaves |
| `best_split_per_leaf_` | `std::vector<SplitInfo>` | Best split information for each leaf |
| `smaller_leaf_splits_` | `std::unique_ptr<LeafSplits>` | Statistics for leaf with fewer data points |
| `larger_leaf_splits_` | `std::unique_ptr<LeafSplits>` | Statistics for leaf with more data points |

### Histogram Variables
| Variable | Type | Purpose |
|----------|------|---------|
| `histogram_pool_` | `HistogramPool` | Memory pool for histogram caching |
| `parent_leaf_histogram_array_` | `FeatureHistogram*` | Histograms for parent leaf |
| `smaller_leaf_histogram_array_` | `FeatureHistogram*` | Histograms for smaller child leaf |
| `larger_leaf_histogram_array_` | `FeatureHistogram*` | Histograms for larger child leaf |

### Configuration Variables
| Variable | Type | Purpose |
|----------|------|---------|
| `config_` | `const Config*` | Training configuration parameters |
| `col_sampler_` | `ColSampler` | Feature sampling for each tree/node |
| `constraints_` | `std::unique_ptr<LeafConstraintsBase>` | Monotonic and interaction constraints |
| `forced_split_json_` | `const Json*` | JSON specification for forced splits |

## Algorithmic References

### 1. Gradient Boosting Decision Trees (GBDT)
The core algorithm implements the GBDT framework as described in:
- **Reference**: "Greedy Function Approximation: A Gradient Boosting Machine" by Jerome H. Friedman
- **Key equation**: Leaf output optimization using Newton-Raphson method

### 2. Histogram-based Split Finding
LightGBM uses histogram-based split finding for efficiency:
- **Reference**: "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" by Guolin Ke et al.
- **Algorithm**: Pre-compute feature histograms and use subtraction trick
- **Optimization**: `histogram_parent = histogram_left + histogram_right`

### 3. Leaf-wise Tree Growth
Unlike level-wise growth, LightGBM grows trees leaf-wise:
- **Strategy**: Always split the leaf with maximum loss reduction
- **Benefit**: More efficient use of leaf budget for complex trees
- **Control**: `max_depth` parameter prevents overfitting

### 4. Gradient Quantization (Optional)
When enabled, gradients are quantized to reduce memory usage:
- **Method**: Linear quantization with scaling factors
- **Formula**: `quantized_grad = round(gradient / grad_scale)`
- **Benefit**: Reduced memory footprint and potential speedup

### 5. Cost-Effective Gradient Boosting (CEGB)
Advanced optimization technique for feature selection:
- **Reference**: "Cost-Effective Gradient Boosting" algorithm
- **Purpose**: Reduce computational cost by smart feature selection
- **Implementation**: Modifies split gain based on feature usage history

## Performance Optimizations

### 1. Histogram Subtraction
Instead of computing histograms for both children, compute one and subtract:
```cpp
// More efficient than computing both histograms separately
larger_leaf_histogram = parent_histogram - smaller_leaf_histogram;
```

### 2. Memory Pool for Histograms
Reuse memory across different tree nodes using `HistogramPool`:
- Reduces memory allocation overhead
- Implements LRU-style caching for histogram arrays

### 3. Parallel Feature Processing
Split finding is parallelized across features:
```cpp
#pragma omp parallel for schedule(static) num_threads(num_threads)
for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    // Compute best split for this feature in parallel
}
```

### 4. SIMD Optimizations
Histogram construction uses SIMD instructions where available for:
- Gradient/hessian accumulation
- Bin assignment operations
- Arithmetic operations on histogram data

This comprehensive flowchart captures the complete tree learning process in LightGBM, from initialization through split finding to final tree construction, including all major mathematical formulations and algorithmic optimizations.