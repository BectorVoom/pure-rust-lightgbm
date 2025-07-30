#ifndef LIGHTGBM_BIN_H_
#define LIGHTGBM_BIN_H_

#include <cstdint>
#include <vector>
#include <memory>

namespace LightGBM {

// Type definitions matching LightGBM
typedef int32_t data_size_t;
typedef float score_t;
typedef double hist_t;

// Mock MultiValBin base class
class MultiValBin {
 public:
  virtual ~MultiValBin() = default;
  
  virtual data_size_t num_data() const = 0;
  virtual int num_bin() const = 0;
  virtual double num_element_per_row() const = 0;
  virtual const std::vector<uint32_t>& offsets() const = 0;
  
  virtual void PushOneRow(int tid, data_size_t idx, const std::vector<uint32_t>& values) = 0;
  virtual void FinishLoad() = 0;
  virtual bool IsSparse() = 0;
  
  // Histogram construction methods
  virtual void ConstructHistogram(const data_size_t* data_indices, data_size_t start,
                                  data_size_t end, const score_t* gradients,
                                  const score_t* hessians, hist_t* out) const = 0;
  
  virtual void ConstructHistogram(data_size_t start, data_size_t end,
                                  const score_t* gradients, const score_t* hessians,
                                  hist_t* out) const = 0;
  
  virtual void ConstructHistogramOrdered(const data_size_t* data_indices,
                                         data_size_t start, data_size_t end,
                                         const score_t* gradients,
                                         const score_t* hessians,
                                         hist_t* out) const = 0;
  
  // Integer histogram methods
  virtual void ConstructHistogramInt32(const data_size_t* data_indices, data_size_t start,
                                       data_size_t end, const score_t* gradients,
                                       const score_t* hessians, hist_t* out) const = 0;
  
  virtual void ConstructHistogramInt32(data_size_t start, data_size_t end,
                                       const score_t* gradients, const score_t* hessians,
                                       hist_t* out) const = 0;
  
  virtual void ConstructHistogramOrderedInt32(const data_size_t* data_indices,
                                              data_size_t start, data_size_t end,
                                              const score_t* gradients,
                                              const score_t* hessians,
                                              hist_t* out) const = 0;
  
  virtual void ConstructHistogramInt16(const data_size_t* data_indices, data_size_t start,
                                       data_size_t end, const score_t* gradients,
                                       const score_t* hessians, hist_t* out) const = 0;
  
  virtual void ConstructHistogramInt16(data_size_t start, data_size_t end,
                                       const score_t* gradients, const score_t* hessians,
                                       hist_t* out) const = 0;
  
  virtual void ConstructHistogramOrderedInt16(const data_size_t* data_indices,
                                              data_size_t start, data_size_t end,
                                              const score_t* gradients,
                                              const score_t* hessians,
                                              hist_t* out) const = 0;
  
  virtual void ConstructHistogramInt8(const data_size_t* data_indices, data_size_t start,
                                      data_size_t end, const score_t* gradients,
                                      const score_t* hessians, hist_t* out) const = 0;
  
  virtual void ConstructHistogramInt8(data_size_t start, data_size_t end,
                                      const score_t* gradients, const score_t* hessians,
                                      hist_t* out) const = 0;
  
  virtual void ConstructHistogramOrderedInt8(const data_size_t* data_indices,
                                             data_size_t start, data_size_t end,
                                             const score_t* gradients,
                                             const score_t* hessians,
                                             hist_t* out) const = 0;
  
  // Copy methods
  virtual MultiValBin* CreateLike(data_size_t num_data, int num_bin, int num_feature,
                                  double estimate_element_per_row,
                                  const std::vector<uint32_t>& offsets) const = 0;
  
  virtual void ReSize(data_size_t num_data, int num_bin, int num_feature,
                      double estimate_element_per_row, const std::vector<uint32_t>& offsets) = 0;
  
  virtual void CopySubrow(const MultiValBin* full_bin, const data_size_t* used_indices,
                          data_size_t num_used_indices) = 0;
  
  virtual void CopySubcol(const MultiValBin* full_bin, const std::vector<int>& used_feature_index,
                          const std::vector<uint32_t>& lower,
                          const std::vector<uint32_t>& upper,
                          const std::vector<uint32_t>& delta) = 0;
  
  virtual void CopySubrowAndSubcol(const MultiValBin* full_bin,
                                   const data_size_t* used_indices,
                                   data_size_t num_used_indices,
                                   const std::vector<int>& used_feature_index,
                                   const std::vector<uint32_t>& lower,
                                   const std::vector<uint32_t>& upper,
                                   const std::vector<uint32_t>& delta) = 0;
  
  #ifdef USE_CUDA
  virtual const void* GetRowWiseData(uint8_t* bit_type,
                                     size_t* total_size,
                                     bool* is_sparse,
                                     const void** out_data_ptr,
                                     uint8_t* data_ptr_bit_type) const = 0;
  #endif  // USE_CUDA
};

}  // namespace LightGBM

#endif  // LIGHTGBM_BIN_H_