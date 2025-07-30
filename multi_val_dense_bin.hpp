// Simplified MultiValDenseBin header for testing

#pragma once
#include <iostream>
#include <vector>
#include <cstdint>

typedef int32_t data_size_t;
typedef double score_t;  
typedef double hist_t;

class MultiValBin {
public:
    virtual ~MultiValBin() = default;
    virtual data_size_t num_data() const = 0;
    virtual int32_t num_bin() const = 0;
    virtual double num_element_per_row() const = 0;
    virtual const std::vector<uint32_t>& offsets() const = 0;
    virtual void PushOneRow(int tid, data_size_t idx, const std::vector<uint32_t>& values) = 0;
    virtual void FinishLoad() = 0;
    virtual bool IsSparse() = 0;
    
    virtual void ConstructHistogram(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                  const score_t* gradients, const score_t* hessians, hist_t* out) const = 0;
    virtual void ConstructHistogram(data_size_t start, data_size_t end,
                                  const score_t* gradients, const score_t* hessians, hist_t* out) const = 0;
    virtual void ConstructHistogramOrdered(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                         const score_t* gradients, const score_t* hessians, hist_t* out) const = 0;
    
    virtual void ConstructHistogramInt32(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                        const score_t* gradients, const score_t* hessians, hist_t* out) const = 0;
    virtual void ConstructHistogramInt32(data_size_t start, data_size_t end,
                                        const score_t* gradients, const score_t* hessians, hist_t* out) const = 0;
    virtual void ConstructHistogramOrderedInt32(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                               const score_t* gradients, const score_t* hessians, hist_t* out) const = 0;
    
    virtual void ConstructHistogramInt16(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                        const score_t* gradients, const score_t* hessians, hist_t* out) const = 0;
    virtual void ConstructHistogramInt16(data_size_t start, data_size_t end,
                                        const score_t* gradients, const score_t* hessians, hist_t* out) const = 0;
    virtual void ConstructHistogramOrderedInt16(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                               const score_t* gradients, const score_t* hessians, hist_t* out) const = 0;
    
    virtual void ConstructHistogramInt8(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                       const score_t* gradients, const score_t* hessians, hist_t* out) const = 0;
    virtual void ConstructHistogramInt8(data_size_t start, data_size_t end,
                                       const score_t* gradients, const score_t* hessians, hist_t* out) const = 0;
    virtual void ConstructHistogramOrderedInt8(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                              const score_t* gradients, const score_t* hessians, hist_t* out) const = 0;
    
    virtual MultiValBin* CreateLike(data_size_t num_data, int num_bin, int num_feature, double, const std::vector<uint32_t>& offsets) const = 0;
    virtual void ReSize(data_size_t num_data, int num_bin, int num_feature, double, const std::vector<uint32_t>& offsets) = 0;
    virtual void CopySubrow(const MultiValBin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices) = 0;
    virtual void CopySubcol(const MultiValBin* full_bin, const std::vector<int>& used_feature_index,
                           const std::vector<uint32_t>&, const std::vector<uint32_t>&, const std::vector<uint32_t>&) = 0;
    virtual void CopySubrowAndSubcol(const MultiValBin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices,
                                    const std::vector<int>& used_feature_index, const std::vector<uint32_t>&, 
                                    const std::vector<uint32_t>&, const std::vector<uint32_t>&) = 0;
    virtual MultiValBin* Clone() = 0;
};

template <typename VAL_T>
class MultiValDenseBin : public MultiValBin {
 public:
  explicit MultiValDenseBin(data_size_t num_data, int num_bin, int num_feature,
    const std::vector<uint32_t>& offsets)
    : num_data_(num_data), num_bin_(num_bin), num_feature_(num_feature),
      offsets_(offsets) {
    data_.resize(static_cast<size_t>(num_data_) * num_feature_, static_cast<VAL_T>(0));
  }

  ~MultiValDenseBin() {}

  data_size_t num_data() const override {
    return num_data_;
  }

  int32_t num_bin() const override {
    return num_bin_;
  }

  double num_element_per_row() const override { return num_feature_; }

  const std::vector<uint32_t>& offsets() const override { return offsets_; }

  void PushOneRow(int, data_size_t idx, const std::vector<uint32_t>& values) override {
    auto start = RowPtr(idx);
    for (auto i = 0; i < num_feature_; ++i) {
      data_[start + i] = static_cast<VAL_T>(values[i]);
    }
  }

  void FinishLoad() override {}

  bool IsSparse() override {
    return false;
  }

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* gradients, const score_t* hessians, hist_t* out) const override {
    for (data_size_t i = start; i < end; ++i) {
      const auto idx = data_indices[i];
      const auto j_start = RowPtr(idx);
      const auto gradient = gradients[idx];
      const auto hessian = hessians[idx];
      
      for (int j = 0; j < num_feature_; ++j) {
        const uint32_t bin = static_cast<uint32_t>(data_[j_start + j]);
        const auto ti = (bin + offsets_[j]) << 1;
        out[ti] += gradient;
        out[ti + 1] += hessian;
      }
    }
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
                          const score_t* gradients, const score_t* hessians,
                          hist_t* out) const override {
    for (data_size_t i = start; i < end; ++i) {
      const auto j_start = RowPtr(i);
      const auto gradient = gradients[i];
      const auto hessian = hessians[i];
      
      for (int j = 0; j < num_feature_; ++j) {
        const uint32_t bin = static_cast<uint32_t>(data_[j_start + j]);
        const auto ti = (bin + offsets_[j]) << 1;
        out[ti] += gradient;
        out[ti + 1] += hessian;
      }
    }
  }

  void ConstructHistogramOrdered(const data_size_t* data_indices,
                                 data_size_t start, data_size_t end,
                                 const score_t* gradients,
                                 const score_t* hessians,
                                 hist_t* out) const override {
    for (data_size_t i = start; i < end; ++i) {
      const auto idx = data_indices[i];
      const auto j_start = RowPtr(idx);
      const auto gradient = gradients[i]; // Note: using i instead of idx for ordered
      const auto hessian = hessians[i];
      
      for (int j = 0; j < num_feature_; ++j) {
        const uint32_t bin = static_cast<uint32_t>(data_[j_start + j]);
        const auto ti = (bin + offsets_[j]) << 1;
        out[ti] += gradient;
        out[ti + 1] += hessian;
      }
    }
  }

  // Simplified integer implementations
  void ConstructHistogramInt32(const data_size_t* data_indices, data_size_t start, data_size_t end,
                              const score_t* gradients, const score_t*, hist_t* out) const override {
    ConstructHistogram(data_indices, start, end, gradients, gradients, out);
  }

  void ConstructHistogramInt32(data_size_t start, data_size_t end,
                              const score_t* gradients, const score_t*, hist_t* out) const override {
    ConstructHistogram(start, end, gradients, gradients, out);
  }

  void ConstructHistogramOrderedInt32(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                     const score_t* gradients, const score_t*, hist_t* out) const override {
    ConstructHistogramOrdered(data_indices, start, end, gradients, gradients, out);
  }

  void ConstructHistogramInt16(const data_size_t* data_indices, data_size_t start, data_size_t end,
                              const score_t* gradients, const score_t*, hist_t* out) const override {
    ConstructHistogram(data_indices, start, end, gradients, gradients, out);
  }

  void ConstructHistogramInt16(data_size_t start, data_size_t end,
                              const score_t* gradients, const score_t*, hist_t* out) const override {
    ConstructHistogram(start, end, gradients, gradients, out);
  }

  void ConstructHistogramOrderedInt16(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                     const score_t* gradients, const score_t*, hist_t* out) const override {
    ConstructHistogramOrdered(data_indices, start, end, gradients, gradients, out);
  }

  void ConstructHistogramInt8(const data_size_t* data_indices, data_size_t start, data_size_t end,
                             const score_t* gradients, const score_t*, hist_t* out) const override {
    ConstructHistogram(data_indices, start, end, gradients, gradients, out);
  }

  void ConstructHistogramInt8(data_size_t start, data_size_t end,
                             const score_t* gradients, const score_t*, hist_t* out) const override {
    ConstructHistogram(start, end, gradients, gradients, out);
  }

  void ConstructHistogramOrderedInt8(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                    const score_t* gradients, const score_t*, hist_t* out) const override {
    ConstructHistogramOrdered(data_indices, start, end, gradients, gradients, out);
  }

  MultiValBin* CreateLike(data_size_t num_data, int num_bin, int num_feature, double,
    const std::vector<uint32_t>& offsets) const override {
    return new MultiValDenseBin<VAL_T>(num_data, num_bin, num_feature, offsets);
  }

  void ReSize(data_size_t num_data, int num_bin, int num_feature,
              double, const std::vector<uint32_t>& offsets) override {
    num_data_ = num_data;
    num_bin_ = num_bin;
    num_feature_ = num_feature;
    offsets_ = offsets;
    size_t new_size = static_cast<size_t>(num_feature_) * num_data_;
    if (data_.size() < new_size) {
      data_.resize(new_size, 0);
    }
  }

  void CopySubrow(const MultiValBin* full_bin, const data_size_t* used_indices,
                  data_size_t num_used_indices) override {
    const auto other_bin = reinterpret_cast<const MultiValDenseBin<VAL_T>*>(full_bin);
    for (data_size_t i = 0; i < num_used_indices; ++i) {
      const auto j_start = RowPtr(i);
      const auto other_j_start = other_bin->RowPtr(used_indices[i]);
      for (int j = 0; j < num_feature_; ++j) {
        data_[j_start + j] = other_bin->data_[other_j_start + j];
      }
    }
  }

  void CopySubcol(const MultiValBin* full_bin,
                  const std::vector<int>& used_feature_index,
                  const std::vector<uint32_t>&,
                  const std::vector<uint32_t>&,
                  const std::vector<uint32_t>&) override {
    const auto other_bin = reinterpret_cast<const MultiValDenseBin<VAL_T>*>(full_bin);
    for (data_size_t i = 0; i < num_data_; ++i) {
      const auto j_start = RowPtr(i);
      const auto other_j_start = other_bin->RowPtr(i);
      for (int j = 0; j < num_feature_; ++j) {
        if (other_bin->data_[other_j_start + used_feature_index[j]] > 0) {
          data_[j_start + j] = other_bin->data_[other_j_start + used_feature_index[j]];
        } else {
          data_[j_start + j] = 0;
        }
      }
    }
  }

  void CopySubrowAndSubcol(const MultiValBin* full_bin,
                           const data_size_t* used_indices,
                           data_size_t num_used_indices,
                           const std::vector<int>& used_feature_index,
                           const std::vector<uint32_t>&,
                           const std::vector<uint32_t>&,
                           const std::vector<uint32_t>&) override {
    const auto other_bin = reinterpret_cast<const MultiValDenseBin<VAL_T>*>(full_bin);
    for (data_size_t i = 0; i < num_used_indices; ++i) {
      const auto j_start = RowPtr(i);
      const auto other_j_start = other_bin->RowPtr(used_indices[i]);
      for (int j = 0; j < num_feature_; ++j) {
        if (other_bin->data_[other_j_start + used_feature_index[j]] > 0) {
          data_[j_start + j] = other_bin->data_[other_j_start + used_feature_index[j]];
        } else {
          data_[j_start + j] = 0;
        }
      }
    }
  }

  inline size_t RowPtr(data_size_t idx) const {
    return static_cast<size_t>(idx) * num_feature_;
  }

  MultiValDenseBin<VAL_T>* Clone() override {
    return new MultiValDenseBin<VAL_T>(*this);
  }

 private:
  data_size_t num_data_;
  int num_bin_;
  int num_feature_;
  std::vector<uint32_t> offsets_;
  std::vector<VAL_T> data_;

  MultiValDenseBin(const MultiValDenseBin<VAL_T>& other)
    : num_data_(other.num_data_), num_bin_(other.num_bin_), num_feature_(other.num_feature_),
      offsets_(other.offsets_), data_(other.data_) {
  }
};