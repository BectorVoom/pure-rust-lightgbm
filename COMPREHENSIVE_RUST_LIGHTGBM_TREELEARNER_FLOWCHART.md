# Comprehensive Pure Rust LightGBM TreeLearner Implementation Flowchart

## Overview

This document provides a comprehensive technical flowchart illustrating the **TreeLearner** implementation logic for the Pure Rust LightGBM framework. This flowchart serves as both documentation and implementation guide, incorporating the corrected mathematical formulas from validation results and the actual implementation patterns found in the codebase.

**🎯 Key Corrections Applied:**
- **Issue #110 Resolution**: Max-gain leaf selection algorithm
- **Mathematical Formula Validation**: Exact formulas matching C++ LightGBM
- **BeforeTrain Process**: Comprehensive initialization workflow
- **Histogram Subtraction**: Optimized construction methods

## Legend & Data Types

### Flowchart Symbols
- **🟢 Circle**: Start/End points
- **⬜ Rectangle**: Process/Computation step  
- **◇ Diamond**: Decision point/Conditional branch
- **⬡ Hexagon**: Mathematical computation with formula
- **⬟ Parallelogram**: Input/Output operations
- **🔄 Process Loop**: Iterative operations

### Rust Data Types
```rust
// Core type aliases from src/core/types.rs
type DataSize = u32;           // Platform-optimized data size
type FeatureIndex = usize;     // Feature index type
type NodeIndex = usize;        // Tree node index
type BinIndex = u32;           // Histogram bin index
type Score = f32;              // Prediction score type
type Hist = f64;               // Histogram accumulator type

// Primary data structures
type Array1<T> = ndarray::Array1<T>;    // 1D array
type Array2<T> = ndarray::Array2<T>;    // 2D matrix
type Vec<T> = std::vec::Vec<T>;         // Dynamic vector
```

## Core Data Structures

### 1. Complete SplitInfo Structure
```rust
/// Split information with full C++ LightGBM compatibility
#[derive(Debug, Clone, PartialEq)]
pub struct SplitInfo {
    // Basic split parameters
    pub feature: FeatureIndex,          // Split feature index
    pub threshold_bin: BinIndex,         // Bin threshold index
    pub threshold_value: f64,            // Actual threshold value
    pub gain: f64,                       // Split gain improvement
    
    // Left child statistics
    pub left_sum_gradient: f64,          // ∑(gradients) in left child
    pub left_sum_hessian: f64,           // ∑(hessians) in left child  
    pub left_count: DataSize,            // Data count in left child
    pub left_output: Score,              // Predicted output for left child
    
    // Right child statistics  
    pub right_sum_gradient: f64,         // ∑(gradients) in right child
    pub right_sum_hessian: f64,          // ∑(hessians) in right child
    pub right_count: DataSize,           // Data count in right child
    pub right_output: Score,             // Predicted output for right child
    
    // Missing value handling
    pub default_left: bool,              // Default direction for NaN values
    
    // Extended features (Issues #102, #103)
    pub monotone_type: i8,               // Monotonic constraint (-1/0/1)
    pub cat_threshold: Vec<u32>,         // Categorical bitset threshold
}
```

### 2. NodeInfo Structure
```rust
/// Information about a tree node during construction
#[derive(Debug, Clone)]
struct NodeInfo {
    pub node_index: NodeIndex,           // Node index in tree
    pub data_indices: Vec<DataSize>,     // Data points in this node
    pub depth: usize,                    // Current node depth
    pub sum_gradients: f64,              // Sum of gradients
    pub sum_hessians: f64,               // Sum of hessians
    pub path_features: Vec<FeatureIndex>, // Features used in path to node
}
```

### 3. FeatureHistogram Structure  
```rust
/// Optimized histogram with interleaved storage
#[derive(Debug, Clone)]
pub struct FeatureHistogram {
    data: Vec<Hist>,                     // [grad₀, hess₀, grad₁, hess₁, ...]
    num_bins: usize,                     // Number of bins
    feature_index: FeatureIndex,         // Associated feature index
}

impl FeatureHistogram {
    #[inline]
    pub fn get_gradient(&self, bin: usize) -> Hist {
        self.data[bin << 1]              // data[bin * 2] with bit shift
    }
    
    #[inline] 
    pub fn get_hessian(&self, bin: usize) -> Hist {
        self.data[(bin << 1) + 1]        // data[bin * 2 + 1] with bit shift
    }
}
```

### 4. Dataset Structure
```rust
/// Training dataset with binning information
#[derive(Debug, Clone)]
pub struct Dataset {
    pub features: Array2<f32>,           // Feature matrix (n_samples × n_features)
    pub num_data: DataSize,              // Number of data samples
    pub num_features: usize,             // Number of features
    pub bin_mappers: Vec<BinMapper>,     // Feature discretization mappers
}
```

## Main Algorithm Flow

```mermaid
flowchart TD
    Start([🟢 Start: SerialTreeLearner::train<br/>dataset: &Dataset<br/>gradients: &ArrayView1<Score><br/>hessians: &ArrayView1<Score><br/>iteration: usize]) --> ValidateInput[⬜ validate_input()<br/>Check dataset dimensions<br/>Validate finite gradients/hessians]
    
    ValidateInput --> BeforeTrain[⬜ before_train()<br/>🔧 Issue #111: Complete initialization<br/>Reset pools, constraints, cache]
    
    BeforeTrain --> InitTree[⬜ Initialize Tree<br/>tree = Tree::with_capacity(max_leaves, learning_rate)<br/>Set shrinkage factor]
    
    InitTree --> CalcRootStats[⬡ Calculate Root Statistics<br/>total_gradients: f64 = Σ(gradients)<br/>total_hessians: f64 = Σ(hessians)<br/>all_data_indices: Vec<DataSize> = [0..num_data]]
    
    CalcRootStats --> SetRootOutput[⬡ Calculate Root Output<br/>root_output = calculate_leaf_output(<br/>  total_gradients, total_hessians,<br/>  lambda_l1, lambda_l2)<br/>Formula: -ThresholdL1(G, λ₁) / (H + λ₂)]
    
    SetRootOutput --> InitCandidates[⬜ Initialize Candidate Leaves<br/>candidate_leaves = Vec::new()<br/>candidate_leaves.push(root_node_info)<br/>Clear histogram cache]
    
    InitCandidates --> MainLoop{◇ Main Tree Growing Loop<br/>tree.num_leaves() < max_leaves<br/>AND !candidate_leaves.is_empty()}
    
    MainLoop -->|No| FinalizeOutputs[⬜ finalize_tree_outputs()<br/>Set leaf outputs for all leaf nodes<br/>using L1/L2 regularization]
    
    MainLoop -->|Yes| FindBestSplits[⬜ Find Best Splits<br/>🔧 Issue #110: Max-gain leaf selection<br/>For each candidate leaf:<br/>  best_split = find_best_split()]
    
    FindBestSplits --> SelectMaxGain[⬜ Select Maximum Gain Leaf<br/>max_gain = min_split_gain<br/>best_leaf_index = None<br/>For each candidate:<br/>  if split.gain > max_gain:<br/>    max_gain = split.gain<br/>    best_leaf_index = candidate_index]
    
    SelectMaxGain --> ValidGain{◇ Valid Split Found?<br/>best_leaf_index.is_some()<br/>AND max_gain > min_split_gain}
    
    ValidGain -->|No| FinalizeOutputs
    
    ValidGain -->|Yes| CheckDepth{◇ Check Maximum Depth<br/>selected_leaf.depth >= max_depth}
    
    CheckDepth -->|Yes| FinalizeOutputs
    
    CheckDepth -->|No| ApplySplit[⬜ apply_split()<br/>Partition data based on split<br/>Create left and right child nodes<br/>Update tree structure]
    
    ApplySplit --> CacheHistograms{◇ Use Histogram Subtraction?<br/>config.use_histogram_subtraction<br/>AND cache optimization available}
    
    CacheHistograms -->|Yes| CacheParent[⬜ Cache Parent Histograms<br/>Store histograms for subtraction<br/>Enable parent - sibling optimization]
    
    CacheHistograms -->|No| UpdateCandidates[⬜ Update Candidate Leaves<br/>Remove selected leaf<br/>Add left and right children<br/>Update candidate_leaves vector]
    
    CacheParent --> UpdateCandidates
    
    UpdateCandidates --> MainLoop
    
    FinalizeOutputs --> QuantizedCheck{◇ Use Quantized Gradients?<br/>🔧 Issue #109: Not implemented<br/>config.use_quantized_gradients}
    
    QuantizedCheck -->|Yes| RenewLeaf[⬜ renew_tree_output_by_histogram()<br/>Recalculate outputs with quantized data<br/>📝 Not implemented in current version]
    
    QuantizedCheck -->|No| ReturnTree[🟢 Return Trained Tree<br/>tree: Tree]
    
    RenewLeaf --> ReturnTree
```

## BeforeTrain Process (Issue #111 Resolution)

```mermaid
flowchart TD
    BeforeStart([🟢 Start: before_train()<br/>dataset: &Dataset<br/>iteration: usize]) --> ResetPool[⬜ Reset Histogram Pool<br/>histogram_pool.lock().unwrap().reset()<br/>Clear cached histograms for reuse]
    
    ResetPool --> SampleFeatures[⬜ Column Sampler Reset<br/>feature_sampler.reset()<br/>Reset per-tree feature sampling<br/>🔧 Issue #106: Feature sampling by tree]
    
    SampleFeatures --> SampleTreeFeatures[⬜ Sample Features for Tree<br/>sampled_features = feature_sampler<br/>  .sample_features(num_features, iteration)<br/>Column sampling per tree iteration]
    
    SampleTreeFeatures --> InitDataset[⬜ Initialize Training Data<br/>histogram_builder.initialize_for_dataset(<br/>  dataset.num_features,<br/>  dataset.num_data as usize)<br/>Prepare histogram construction]
    
    InitDataset --> InitPartition[⬜ DataPartition::Init()<br/>📝 Conceptual: Set all data to root leaf<br/>all_data_indices = [0..dataset.num_data]<br/>Initialize data partitioning structures]
    
    InitPartition --> ResetConstraints[⬜ Reset Monotonic Constraints<br/>constraint_manager.reset_for_tree()<br/>🔧 Issue #102: Monotonic constraints<br/>Clear per-tree constraint state]
    
    ResetConstraints --> ResetSplits[⬜ Reset Split Arrays<br/>📝 Conceptual: Reset best_split_per_leaf<br/>Clear previous split information<br/>Prepare for new tree construction]
    
    ResetSplits --> ComputeRoot[⬡ Compute Root Statistics<br/>total_gradients = Σ(gradients)<br/>total_hessians = Σ(hessians)<br/>Initialize root node state]
    
    ComputeRoot --> BaggingCheck{◇ Using Data Bagging?<br/>🔧 Issue #108: Not implemented<br/>config.bagging_fraction < 1.0}
    
    BaggingCheck -->|Yes| BaggingInit[⬜ Initialize Bagging<br/>📝 Not implemented<br/>smaller_leaf_splits_.Init(<br/>  leaf=0, partition, gradients, hessians)<br/>Setup data sampling]
    
    BaggingCheck -->|No| FullInit[⬜ Full Data Initialization<br/>📝 Conceptual implementation<br/>smaller_leaf_splits_.Init(<br/>  gradients, hessians)<br/>Use all training data]
    
    BaggingInit --> InitLarger[⬜ Initialize Larger Splits<br/>📝 Not implemented<br/>larger_leaf_splits_.Init()<br/>Prepare for dual histogram approach]
    
    FullInit --> InitLarger
    
    InitLarger --> CEGBCheck{◇ Cost-Effective Gradient Boosting?<br/>🔧 Issue #108: Not implemented<br/>config.use_cegb}
    
    CEGBCheck -->|Yes| CEGBInit[⬜ CEGB Initialization<br/>📝 Not implemented<br/>cegb_.BeforeTrain()<br/>Setup cost-effective optimization]
    
    CEGBCheck -->|No| QuantBitsCheck{◇ Quantized Gradients?<br/>🔧 Issue #109: Not implemented<br/>config.use_quantized_gradients}
    
    CEGBInit --> QuantBitsCheck
    
    QuantBitsCheck -->|Yes| SetHistBits[⬜ Set Histogram Bits<br/>📝 Not implemented<br/>Set quantization bits for root leaf<br/>Initialize quantized histogram storage]
    
    QuantBitsCheck -->|No| ClearCache[⬜ Clear Histogram Cache<br/>node_histograms.clear()<br/>Prepare cache for new tree]
    
    SetHistBits --> ClearCache
    
    ClearCache --> BeforeEnd[🟢 End: before_train()<br/>✅ Initialization complete]
```

## Split Finding Algorithm (Issues #102, #103 Corrections)

```mermaid
flowchart TD
    FindStart([🟢 Start: find_best_split()<br/>dataset: &Dataset<br/>gradients: &ArrayView1<Score><br/>hessians: &ArrayView1<Score><br/>node_info: &NodeInfo]) --> CheckMinData{◇ Sufficient Data Check<br/>node_info.data_indices.len() >=<br/>2 * min_data_in_leaf}
    
    CheckMinData -->|No| ReturnNone[🔴 return Ok(None)<br/>Cannot split - insufficient data]
    
    CheckMinData -->|Yes| CheckMinHessian{◇ Sufficient Hessian Check<br/>node_info.sum_hessians >=<br/>2.0 * min_sum_hessian_in_leaf}
    
    CheckMinHessian -->|No| ReturnNone
    
    CheckMinHessian -->|Yes| SampleFeatures[⬜ FeatureSampler::sample_features()<br/>sampled_features = feature_sampler<br/>  .sample_features(num_features, 0)<br/>🔧 Issue #106: Column sampling]
    
    SampleFeatures --> CheckSampled{◇ Sampled Features Available?<br/>!sampled_features.is_empty()}
    
    CheckSampled -->|No| ReturnNone
    
    CheckSampled -->|Yes| FilterConstraints[⬜ ConstraintManager::filter_candidate_features()<br/>allowed_features = constraint_manager<br/>  .filter_candidate_features(<br/>    &sampled_features,<br/>    &node_info.path_features)<br/>🔧 Issue #102: Monotonic constraints]
    
    FilterConstraints --> CheckFiltered{◇ Allowed Features Available?<br/>!allowed_features.is_empty()}
    
    CheckFiltered -->|No| ReturnNone
    
    CheckFiltered -->|Yes| ConstructHistograms[⬜ construct_node_histograms()<br/>histograms = construct_node_histograms(<br/>  dataset, gradients, hessians,<br/>  &node_info.data_indices,<br/>  &allowed_features)<br/>Build feature histograms]
    
    ConstructHistograms --> CacheForSubtraction{◇ Cache for Subtraction?<br/>config.use_histogram_subtraction}
    
    CacheForSubtraction -->|Yes| CacheHists[⬜ Cache Histograms<br/>cache_histograms(<br/>  node_info.node_index,<br/>  &histograms,<br/>  &allowed_features)<br/>Store for parent-sibling optimization]
    
    CacheForSubtraction -->|No| InitBestSplit[⬜ Initialize Best Split Search<br/>best_split: Option<SplitInfo> = None<br/>Prepare to iterate through features]
    
    CacheHists --> InitBestSplit
    
    InitBestSplit --> FeatureLoop[🔄 For each allowed feature<br/>for (i, &feature_idx) in allowed_features.iter().enumerate()]
    
    FeatureLoop --> GetBinMapper[⬜ Get BinMapper<br/>bin_mapper = dataset.bin_mapper(feature_idx)<br/>Get feature discretization information]
    
    GetBinMapper --> CheckFeatureType{◇ Check Feature Type<br/>bin_mapper.feature_type()}
    
    CheckFeatureType -->|Numerical| CheckMonotone{◇ Monotonic Constraints?<br/>🔧 Issue #102: Check monotonic type<br/>split.monotone_type != 0}
    
    CheckFeatureType -->|Categorical| CategoricalSplit[⬡ FindBestThresholdCategorical()<br/>🔧 Issue #103: Categorical splits<br/>Use bitset threshold representation<br/>split.cat_threshold = categorical_bitset]
    
    CheckMonotone -->|Yes| RecomputeConstraints[⬜ Recompute Monotonic Constraints<br/>📝 Not fully implemented<br/>Update constraint bounds for feature<br/>Apply monotonic penalty if needed]
    
    CheckMonotone -->|No| NumericalSplit[⬡ find_best_split_for_feature()<br/>split_candidate = split_finder<br/>  .find_best_split_for_feature(<br/>    feature_idx, &histograms[i],<br/>    node_info.sum_gradients,<br/>    node_info.sum_hessians,<br/>    node_info.data_indices.len(),<br/>    bin_mapper.bin_upper_bounds())]
    
    RecomputeConstraints --> NumericalSplit
    
    NumericalSplit --> CheckQuantized{◇ Using Quantized Gradients?<br/>🔧 Issue #109: Not implemented<br/>config.use_quantized_gradients}
    
    CheckQuantized -->|Yes| QuantizedThreshold[⬡ FindBestThresholdInt()<br/>📝 Not implemented<br/>Use integer histogram for quantized data<br/>Optimize with integer arithmetic]
    
    CheckQuantized -->|No| SetFeature[⬜ Set Feature Information<br/>if let Some(mut split) = split_candidate {<br/>  split.feature = feature_idx<br/>  // Continue processing<br/>}]
    
    QuantizedThreshold --> SetFeature
    CategoricalSplit --> SetFeature
    
    SetFeature --> CEGBCheck{◇ CEGB Enabled?<br/>🔧 Issue #108: Not implemented<br/>config.use_cegb}
    
    CEGBCheck -->|Yes| ApplyCEGB[⬡ Apply CEGB Delta<br/>📝 Not implemented<br/>split.gain -= cegb.delta_gain()<br/>Reduce gain by cost-effective penalty]
    
    CEGBCheck -->|No| MonotoneCheck{◇ Monotonic Split Validation?<br/>🔧 Issue #102: Constraint validation<br/>split.has_monotonic_constraint()}
    
    ApplyCEGB --> MonotoneCheck
    
    MonotoneCheck -->|Yes| ApplyPenalty[⬡ Apply Monotonic Penalty<br/>if !split.validate_monotonic_constraint() {<br/>  continue // Skip invalid split<br/>}<br/>Check left_output vs right_output ordering]
    
    MonotoneCheck -->|No| ValidateConstraints[⬜ ConstraintManager::validate_split()<br/>validation_result = constraint_manager<br/>  .validate_split(<br/>    &split, node_info.depth,<br/>    0.0, &node_info.path_features)<br/>General constraint validation]
    
    ApplyPenalty --> ValidateConstraints
    
    ValidateConstraints --> CheckSplitValid{◇ Split Valid and Better?<br/>validation_result.is_valid()<br/>AND (best_split.is_none()<br/>  OR split.gain > best_split.gain)}
    
    CheckSplitValid -->|Yes| UpdateBest[⬜ Update Best Split<br/>best_split = Some(split)<br/>Store current best split candidate]
    
    CheckSplitValid -->|No| NextFeature[⬜ Continue to Next Feature<br/>Process next feature in allowed_features]
    
    UpdateBest --> NextFeature
    
    NextFeature --> CheckMoreFeatures{◇ More Features?<br/>More features in allowed_features}
    
    CheckMoreFeatures -->|Yes| FeatureLoop
    
    CheckMoreFeatures -->|No| ReturnBest[🟢 Return Best Split<br/>Ok(best_split)]
```

## Histogram Construction Process

```mermaid
flowchart TD
    HistStart([🟢 Start: construct_node_histograms()<br/>dataset: &Dataset<br/>gradients: &ArrayView1<Score><br/>hessians: &ArrayView1<Score><br/>data_indices: &[DataSize]<br/>feature_indices: &[FeatureIndex]]) --> CheckParallel{◇ Use Parallel Construction?<br/>feature_indices.len() > 4<br/>AND data_indices.len() > 1000}
    
    CheckParallel -->|Yes| ParallelConstruct[⬜ Parallel Feature Construction<br/>parallel_histograms: Result<Vec<_>, _> =<br/>  feature_indices.par_iter().map(|&feature_idx| {<br/>    // Build histogram for each feature in parallel<br/>  }).collect()]
    
    CheckParallel -->|No| SequentialConstruct[⬜ Sequential Feature Construction<br/>for &feature_idx in feature_indices {<br/>  histogram = construct_feature_histogram(...)<br/>  histograms.push(histogram)<br/>}]
    
    ParallelConstruct --> FeatureHist[⬜ For Each Feature: construct_feature_histogram()<br/>feature_column = dataset.features.column(feature_idx)<br/>bin_mapper = dataset.bin_mapper(feature_idx)]
    
    SequentialConstruct --> FeatureHist
    
    FeatureHist --> CheckSIMD{◇ Use SIMD Optimization?<br/>config.use_simd<br/>AND data_indices.len() > 256<br/>AND sufficient_data_for_simd}
    
    CheckSIMD -->|Yes| SIMDConstruct[⬜ construct_feature_histogram_simd()<br/>Process in parallel chunks<br/>data_indices.par_chunks(chunk_size)<br/>Each thread maintains local histogram]
    
    CheckSIMD -->|No| ScalarConstruct[⬜ construct_feature_histogram_scalar()<br/>Sequential processing<br/>Single-threaded histogram accumulation]
    
    SIMDConstruct --> InitLocalHist[⬜ Initialize Local Histograms<br/>For each thread chunk:<br/>local_histogram = Array1::zeros(num_bins * 2)<br/>Initialize thread-local storage]
    
    InitLocalHist --> SIMDLoop[🔄 SIMD Processing Loop<br/>for &data_idx in chunk {<br/>  // Process data point with SIMD optimization<br/>}]
    
    SIMDLoop --> GetBin[⬡ Bin Mapping<br/>feature_value = feature_values[data_idx]<br/>bin = bin_mapper.value_to_bin(feature_value)<br/>Map continuous value to discrete bin]
    
    GetBin --> AccumulateLocal[⬡ Local Accumulation<br/>gradient = gradients[data_idx] as Hist<br/>hessian = hessians[data_idx] as Hist<br/>local_histogram[bin * 2] += gradient<br/>local_histogram[bin * 2 + 1] += hessian<br/>📝 Interleaved storage format]
    
    AccumulateLocal --> CheckMoreDataSIMD{◇ More Data in Chunk?<br/>More data points in current chunk}
    
    CheckMoreDataSIMD -->|Yes| SIMDLoop
    
    CheckMoreDataSIMD -->|No| AtomicMerge[⬜ Atomic Merge to Main Histogram<br/>📝 Note: Current implementation needs<br/>proper atomic operations for thread safety<br/>for (main_val, local_val) in<br/>  histogram.iter_mut().zip(local_histogram.iter()) {<br/>    *main_val += *local_val<br/>  }]
    
    AtomicMerge --> HistReady[⬜ Feature Histogram Ready<br/>histogram: Array1<Hist><br/>✅ Completed histogram construction]
    
    ScalarConstruct --> ScalarLoop[🔄 Scalar Processing Loop<br/>for &data_idx in data_indices {<br/>  // Sequential data processing<br/>}]
    
    ScalarLoop --> GetBinScalar[⬡ Bin Mapping (Scalar)<br/>feature_value = feature_values[data_idx]<br/>bin = bin_mapper.value_to_bin(feature_value)<br/>Single-threaded bin mapping]
    
    GetBinScalar --> AccumulateScalar[⬡ Direct Accumulation<br/>gradient = gradients[data_idx] as Hist<br/>hessian = hessians[data_idx] as Hist<br/>histogram[bin * 2] += gradient<br/>histogram[bin * 2 + 1] += hessian<br/>📝 Direct histogram update]
    
    AccumulateScalar --> CheckMoreScalar{◇ More Data Points?<br/>More data points to process}
    
    CheckMoreScalar -->|Yes| ScalarLoop
    
    CheckMoreScalar -->|No| HistReady
    
    HistReady --> CheckSubtraction{◇ Use Histogram Subtraction?<br/>config.use_histogram_subtraction<br/>AND optimization_available}
    
    CheckSubtraction -->|Yes| TrySubtraction[⬜ try_histogram_subtraction()<br/>Check if parent and sibling histograms<br/>are available in cache<br/>Enable parent - sibling optimization]
    
    CheckSubtraction -->|No| ReturnHistograms[🟢 Return Constructed Histograms<br/>Ok(histograms: Vec<Array1<Hist>>)]
    
    TrySubtraction --> SubtractionAvailable{◇ Cached Histograms Available?<br/>parent_histograms.is_some()<br/>AND sibling_histograms.is_some()}
    
    SubtractionAvailable -->|Yes| PerformSubtraction[⬡ Histogram Subtraction<br/>📝 Core LightGBM optimization<br/>current_histogram[i] = parent_histogram[i] - sibling_histogram[i]<br/>Element-wise subtraction for each bin<br/>Reduces computation by ~50%]
    
    SubtractionAvailable -->|No| ReturnHistograms
    
    PerformSubtraction --> ReturnHistograms
```

## Single Feature Split Finding

```mermaid
flowchart TD
    SingleStart([🟢 Start: find_best_split_for_feature()<br/>feature_index: FeatureIndex<br/>histogram: &ArrayView1<Hist><br/>total_sum_gradient: f64<br/>total_sum_hessian: f64<br/>total_count: DataSize<br/>bin_boundaries: &[f64]]) --> CheckBins{◇ Sufficient Bins?<br/>num_bins = histogram.len() / 2<br/>num_bins >= 2}
    
    CheckBins -->|No| NoSplit[🔴 return None<br/>Insufficient bins for splitting]
    
    CheckBins -->|Yes| InitBestSplit[⬜ Initialize Best Split<br/>best_split = SplitInfo::new()<br/>best_split.feature = feature_index<br/>Prepare split information structure]
    
    InitBestSplit --> InitAccumulators[⬜ Initialize Accumulators<br/>left_sum_gradient = 0.0: f64<br/>left_sum_hessian = 0.0: f64<br/>left_count = 0: DataSize<br/>Initialize left child statistics]
    
    InitAccumulators --> BinLoop[🔄 Bin Iteration Loop<br/>for bin in 0..num_bins-1 {<br/>  // Try each possible split point<br/>}]
    
    BinLoop --> AccumulateLeft[⬡ Accumulate Left Statistics<br/>grad_idx = bin * 2<br/>hess_idx = bin * 2 + 1<br/>left_sum_gradient += histogram[grad_idx]<br/>left_sum_hessian += histogram[hess_idx]<br/>📝 Interleaved histogram access]
    
    AccumulateLeft --> AccumulateCount[⬡ Approximate Count Accumulation<br/>🔧 TODO #99: Implement exact count tracking<br/>left_count += histogram[hess_idx].round() as DataSize<br/>📝 Using hessian as count approximation]
    
    AccumulateCount --> CalcRight[⬡ Calculate Right Statistics<br/>right_sum_gradient = total_sum_gradient - left_sum_gradient<br/>right_sum_hessian = total_sum_hessian - left_sum_hessian<br/>right_count = total_count - left_count<br/>Complement statistics from totals]
    
    CalcRight --> CheckConstraints{◇ Constraint Validation<br/>left_count >= min_data_in_leaf<br/>AND right_count >= min_data_in_leaf<br/>AND left_sum_hessian >= min_sum_hessian_in_leaf<br/>AND right_sum_hessian >= min_sum_hessian_in_leaf}
    
    CheckConstraints -->|No| NextBin[⬜ Continue to Next Bin<br/>Skip this split point<br/>Try next bin threshold]
    
    CheckConstraints -->|Yes| CalcGain[⬡ Calculate Split Gain<br/>📝 Mathematical Formula 2 (Validated)<br/>gain = calculate_split_gain(<br/>  total_sum_gradient, total_sum_hessian,<br/>  left_sum_gradient, left_sum_hessian,<br/>  right_sum_gradient, right_sum_hessian)]
    
    CalcGain --> CheckGain{◇ Gain Improvement Check<br/>gain > best_split.gain<br/>AND gain > min_split_gain}
    
    CheckGain -->|Yes| UpdateSplit[⬜ Update Best Split<br/>best_split.threshold_bin = bin as BinIndex<br/>best_split.threshold_value = bin_boundaries[bin]<br/>best_split.gain = gain<br/>best_split.left_sum_gradient = left_sum_gradient<br/>best_split.left_sum_hessian = left_sum_hessian<br/>best_split.left_count = left_count<br/>best_split.right_sum_gradient = right_sum_gradient<br/>best_split.right_sum_hessian = right_sum_hessian<br/>best_split.right_count = right_count]
    
    CheckGain -->|No| NextBin
    
    UpdateSplit --> SetDefaultDir[⬡ Set Default Direction<br/>📝 Mathematical Formula: Default Direction<br/>best_split.default_left = <br/>  left_sum_hessian >= right_sum_hessian<br/>Missing values go to side with larger hessian sum]
    
    SetDefaultDir --> NextBin
    
    NextBin --> CheckMoreBins{◇ More Bins to Process?<br/>More bins in range 0..num_bins-1}
    
    CheckMoreBins -->|Yes| BinLoop
    
    CheckMoreBins -->|No| CheckValid{◇ Valid Split Found?<br/>best_split.is_valid()<br/>gain > 0.0 AND left_count > 0 AND right_count > 0}
    
    CheckValid -->|Yes| CalcOutputs[⬡ Calculate Leaf Outputs<br/>📝 Mathematical Formula 1 (Validated)<br/>best_split.calculate_outputs(lambda_l1, lambda_l2)<br/>left_output = calculate_leaf_output(<br/>  left_sum_gradient, left_sum_hessian, λ₁, λ₂)<br/>right_output = calculate_leaf_output(<br/>  right_sum_gradient, right_sum_hessian, λ₁, λ₂)]
    
    CheckValid -->|No| NoValidSplit[🔴 return None<br/>No valid split found for feature]
    
    CalcOutputs --> ReturnSplit[🟢 return Some(best_split)<br/>Valid split with calculated outputs]
```

## Mathematical Formulas (Validated)

### 1. Leaf Output Calculation (Formula 1)

**Newton-Raphson optimization with L1/L2 regularization:**

```
leaf_output = -ThresholdL1(sum_gradients, λ₁) / (sum_hessians + λ₂)
```

**L1 Thresholding Function:**
```
ThresholdL1(g, λ₁) = {
  g - λ₁     if g > λ₁
  g + λ₁     if g < -λ₁  
  0          if |g| ≤ λ₁
}
```

**Rust Implementation (Validated):**
```rust
fn calculate_leaf_output(
    sum_gradient: f64, 
    sum_hessian: f64, 
    lambda_l1: f64, 
    lambda_l2: f64
) -> Score {
    if sum_hessian <= 0.0 {
        return 0.0;
    }

    let numerator = if lambda_l1 > 0.0 {
        if sum_gradient > lambda_l1 {
            sum_gradient - lambda_l1
        } else if sum_gradient < -lambda_l1 {
            sum_gradient + lambda_l1
        } else {
            0.0
        }
    } else {
        sum_gradient
    };

    (-numerator / (sum_hessian + lambda_l2)) as Score
}
```

**Variables:**
- `sum_gradients` (f64): Sum of first-order gradients in node
- `sum_hessians` (f64): Sum of second-order gradients (Hessians) in node  
- `λ₁` (f64): L1 regularization parameter (`lambda_l1`)
- `λ₂` (f64): L2 regularization parameter (`lambda_l2`)

### 2. Split Gain Calculation (Formula 2)

**Bit-exact C++ LightGBM implementation formula:**

```
gain = GetLeafGain(G_left, H_left, λ₁, λ₂) + 
       GetLeafGain(G_right, H_right, λ₁, λ₂) - 
       GetLeafGain(G_parent, H_parent, λ₁, λ₂)
```

**Individual Leaf Gain (Validated):**
```
GetLeafGain(G, H, λ₁, λ₂) = {
  0                                    if H + λ₂ ≤ 0
  0                                    if |G| ≤ λ₁
  ThresholdL1(G, λ₁)² / (2(H + λ₂))   otherwise
}
```

**Rust Implementation (Validated):**
```rust
fn calculate_leaf_gain_exact(
    sum_gradient: f64, 
    sum_hessian: f64, 
    lambda_l1: f64, 
    lambda_l2: f64
) -> f64 {
    if sum_hessian + lambda_l2 <= 0.0 {
        return 0.0;
    }

    let abs_sum_gradient = sum_gradient.abs();
    
    if abs_sum_gradient <= lambda_l1 {
        0.0
    } else {
        let numerator = if sum_gradient > 0.0 {
            sum_gradient - lambda_l1
        } else {
            sum_gradient + lambda_l1
        };
        (numerator * numerator) / (2.0 * (sum_hessian + lambda_l2))
    }
}

fn calculate_split_gain(
    &self,
    total_sum_gradient: f64,
    total_sum_hessian: f64,
    left_sum_gradient: f64,
    left_sum_hessian: f64,
    right_sum_gradient: f64,
    right_sum_hessian: f64,
) -> f64 {
    let gain_left = self.calculate_leaf_gain_exact(
        left_sum_gradient, left_sum_hessian, 
        self.config.lambda_l1, self.config.lambda_l2
    );
    let gain_right = self.calculate_leaf_gain_exact(
        right_sum_gradient, right_sum_hessian, 
        self.config.lambda_l1, self.config.lambda_l2
    );
    let gain_parent = self.calculate_leaf_gain_exact(
        total_sum_gradient, total_sum_hessian, 
        self.config.lambda_l1, self.config.lambda_l2
    );
    
    gain_left + gain_right - gain_parent
}
```

**Variables:**
- `G_left`, `G_right`, `G_parent` (f64): Sum of gradients in left, right, parent
- `H_left`, `H_right`, `H_parent` (f64): Sum of hessians in left, right, parent
- `λ₁` (f64): L1 regularization parameter (`lambda_l1`)
- `λ₂` (f64): L2 regularization parameter (`lambda_l2`)

### 3. Histogram Storage Format (Formula 3)

**Interleaved storage for cache efficiency (Validated):**

```
histogram = [grad_bin₀, hess_bin₀, grad_bin₁, hess_bin₁, ..., grad_binₙ, hess_binₙ]
```

**Optimized Access (Validated):**
```rust
impl FeatureHistogram {
    #[inline]
    pub fn get_gradient(&self, bin: usize) -> Hist {
        self.data[bin << 1]  // Equivalent to data[bin * 2]
    }
    
    #[inline] 
    pub fn get_hessian(&self, bin: usize) -> Hist {
        self.data[(bin << 1) + 1]  // Equivalent to data[bin * 2 + 1]
    }
    
    #[inline]
    pub fn add_sample(&mut self, bin: usize, gradient: Hist, hessian: Hist) {
        let base_idx = bin << 1;
        self.data[base_idx] += gradient;
        self.data[base_idx + 1] += hessian;
    }
}
```

**Accumulation Formula:**
```rust
// For each data point
let bin = bin_mapper.value_to_bin(feature_value);
histogram[bin * 2] += gradient;      // Gradient accumulation
histogram[bin * 2 + 1] += hessian;   // Hessian accumulation
```

### 4. Histogram Subtraction Optimization (Formula 4)

**Core LightGBM optimization formula:**

```
histogram_child = histogram_parent - histogram_sibling
```

**Element-wise Implementation (Validated):**
```rust
pub fn subtract_from(
    &mut self, 
    parent: &FeatureHistogram, 
    sibling: &FeatureHistogram
) {
    debug_assert_eq!(self.num_bins, parent.num_bins);
    debug_assert_eq!(self.num_bins, sibling.num_bins);
    
    for i in 0..self.data.len() {
        self.data[i] = parent.data[i] - sibling.data[i];
    }
}
```

**Strategy:**
- Build histogram for smaller child directly (fewer data points)
- Use subtraction to get larger child: `larger_child = parent - smaller_child`
- Reduces computation by approximately 50% for sibling nodes

### 5. Default Direction Formula (Validated)

**Missing value handling:**

```
default_left = (left_sum_hessian >= right_sum_hessian)
```

**Rationale:** Direct missing values to the child with higher confidence (larger hessian sum).

**Rust Implementation (Validated):**
```rust
// During split evaluation
best_split.default_left = left_sum_hessian >= right_sum_hessian;

// During prediction
if feature_value.is_nan() {
    if split_info.default_left {
        // Go to left child
    } else {
        // Go to right child  
    }
}
```

### 6. Categorical Feature Bitset (Formula 5)

**For categorical features with bitset representation:**

```
categorical_goes_left(category, threshold) = (threshold & (1 << (category % 32))) != 0
```

**Multi-word Implementation (Issue #103):**
```rust
pub fn categorical_goes_left(&self, category: u32) -> bool {
    if self.cat_threshold.is_empty() {
        return false;
    }
    
    let word_index = (category / 32) as usize;
    let bit_index = category % 32;
    
    if word_index >= self.cat_threshold.len() {
        return false;
    }
    
    (self.cat_threshold[word_index] & (1u32 << bit_index)) != 0
}
```

**Example:** Categories 0, 2, 3 go left → bitset: `0b00001101` → threshold: `[13u32]`

## Key Configuration Variables

### SerialTreeLearnerConfig Structure
```rust
#[derive(Debug, Clone)]
pub struct SerialTreeLearnerConfig {
    // Tree structure constraints
    pub max_leaves: usize,              // Maximum leaves (default: 31)
    pub max_depth: usize,               // Maximum depth (default: 6)
    pub min_data_in_leaf: DataSize,     // Min data points per leaf (default: 20)
    pub min_sum_hessian_in_leaf: f64,   // Min hessian sum per leaf (default: 1e-3)
    
    // Regularization parameters  
    pub lambda_l1: f64,                 // L1 regularization (default: 0.0)
    pub lambda_l2: f64,                 // L2 regularization (default: 0.0)
    pub min_split_gain: f64,            // Min gain threshold (default: 0.0)
    
    // Feature discretization
    pub max_bin: usize,                 // Max histogram bins (default: 255)
    
    // Sampling and optimization
    pub feature_sampling: FeatureSamplingConfig,  // Column sampling config
    pub histogram_config: HistogramBuilderConfig, // Histogram construction config
    pub use_histogram_subtraction: bool,          // Enable subtraction optimization
    pub learning_rate: f64,                       // Tree shrinkage factor (default: 0.1)
}
```

## Performance Optimizations

### 1. Memory Layout Optimizations
- **Interleaved Storage**: `[grad₀, hess₀, grad₁, hess₁, ...]` for cache efficiency
- **SIMD-friendly Access**: Bit-shift operations for index calculations
- **Memory Pool**: Reuse histogram allocations across nodes

### 2. Parallel Processing
- **Feature-level Parallelization**: Build histograms for different features in parallel
- **SIMD Optimization**: Process data in chunks for vectorization
- **Thread-local Accumulation**: Avoid contention with local histograms

### 3. Algorithmic Optimizations
- **Histogram Subtraction**: 50% reduction in histogram construction
- **Max-gain Leaf Selection**: Always split the most promising leaf
- **Early Stopping**: Skip splits below minimum gain threshold

## Algorithmic References

### 1. Gradient Boosting Decision Trees (GBDT)
- **Reference**: "Greedy Function Approximation: A Gradient Boosting Machine" by Jerome H. Friedman (2001)
- **Newton-Raphson**: Second-order optimization using Hessian information
- **Leaf-wise Growth**: Split leaves with maximum loss reduction first

### 2. LightGBM Histogram-based Algorithm  
- **Reference**: "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" by Guolin Ke et al. (2017)
- **Histogram Optimization**: Pre-sorted feature discretization
- **Subtraction Trick**: `child = parent - sibling` optimization

### 3. Regularization Techniques
- **L1 Regularization**: Soft thresholding for sparsity
- **L2 Regularization**: Ridge penalty in denominator
- **Early Stopping**: Minimum gain threshold prevents overfitting

## Issues Status & Missing Features

### 🔧 Resolved Issues
- **Issue #110**: ✅ Max-gain leaf selection implemented
- **Mathematical Formulas**: ✅ Validated against C++ LightGBM
- **Histogram Subtraction**: ✅ Working optimization

### ❌ Missing Critical Features
- **Issue #102**: Monotonic constraints (placeholder implementation)
- **Issue #103**: Categorical features (bitset logic defined, not integrated)
- **Issue #108**: Cost-Effective Gradient Boosting (CEGB)
- **Issue #109**: Quantized gradients for memory efficiency

### ⚠️ Incomplete Features  
- **Issue #105**: Path smoothing regularization
- **Issue #106**: Feature sampling (basic framework exists)
- **Issue #99**: Exact data count tracking in histograms

This comprehensive flowchart serves as both documentation and implementation guide for the Pure Rust LightGBM TreeLearner component, providing validated mathematical formulas, detailed algorithmic flows, and clear indicators of current implementation status versus the target C++ LightGBM functionality.