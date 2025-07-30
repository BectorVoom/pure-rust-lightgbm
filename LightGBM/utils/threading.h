#ifndef LIGHTGBM_UTILS_THREADING_H_
#define LIGHTGBM_UTILS_THREADING_H_

#include <algorithm>

namespace LightGBM {
namespace Threading {

// Calculate block information for parallel processing
template<typename T>
inline void BlockInfo(int num_threads, T num_data, T min_block_size, int* n_block, T* block_size) {
  if (num_threads <= 1 || num_data < min_block_size) {
    *n_block = 1;
    *block_size = num_data;
  } else {
    *n_block = std::min(num_threads, static_cast<int>(num_data / min_block_size));
    *block_size = (num_data + *n_block - 1) / *n_block;
  }
}

}  // namespace Threading
}  // namespace LightGBM

#endif  // LIGHTGBM_UTILS_THREADING_H_