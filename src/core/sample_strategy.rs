//! Sample strategy module for LightGBM
//!
//! This module contains sampling strategies for data and features during training.

// From #include <LightGBM/cuda/cuda_utils.hu>:

//   - CUDAVector - GPU memory vector container (used as CUDAVector<data_size_t>)

//   From #include <LightGBM/utils/random.h>:

//   - Random class - Random number generator with methods:
//     - NextShort(), NextInt(), NextFloat() - Generate random numbers
//     - Sample() - Sample K data from {0,1,...,N-1}

//   From #include <LightGBM/utils/common.h>:

//   - Common::AlignmentAllocator<T, kAlignedSize> - Memory-aligned allocator
//   - kAlignedSize constant - Memory alignment size (32 bytes)
//   - Common namespace utilities (string processing, memory management)

//   From #include <LightGBM/utils/threading.h>:

//   - ParallelPartitionRunner<INDEX_T, TWO_BUFFER> - Template class for parallel data partitioning
//   - Threading class - Static methods for parallel processing:
//     - BlockInfo() - Calculate thread block information
//     - For() - Parallel for loop execution

//   From #include <LightGBM/config.h>:

//   - Config struct - Configuration parameters for sampling strategies

//   From #include <LightGBM/dataset.h>:

//   - Dataset class - Training data container
//   - Metadata class - Labels, weights, query information

//   From #include <LightGBM/tree_learner.h>:

//   - TreeLearner class - Interface for tree learning algorithms

//   From #include <LightGBM/objective_function.h>:

//   - ObjectiveFunction class - Loss function interface with methods:
//     - GetGradients() - Calculate gradients and hessians
//     - IsConstantHessian() - Check if hessian is constant

//   From #include <LightGBM/meta.h> (inherited):

//   - data_size_t - typedef for int32_t, used for data indices
//   - score_t - typedef for float/double, used for gradients/scores

//   Key Variables Used in SampleStrategy:

//   - bag_data_indices_ - Vector with aligned allocator for bagging indices
//   - bagging_rands_ - Vector of Random objects for parallel random generation
//   - bagging_runner_ - ParallelPartitionRunner for parallel bagging operations
//   - cuda_bag_data_indices_ - GPU memory buffer for bagging indices
