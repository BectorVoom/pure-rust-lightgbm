#ifndef LIGHTGBM_UTILS_COMMON_H_
#define LIGHTGBM_UTILS_COMMON_H_

#include <vector>
#include <memory>
#include <cstdlib>
#include <cassert>

namespace LightGBM {
namespace Common {

// Mock global timer for timing functionality
extern void* global_timer;

// Function timer class
class FunctionTimer {
 public:
  FunctionTimer(const char* name, void* timer) : name_(name), timer_(timer) {
    // Mock implementation - in real LightGBM this would start timing
  }
  
  ~FunctionTimer() {
    // Mock implementation - in real LightGBM this would stop timing
  }
  
 private:
  const char* name_;
  void* timer_;
};

// Aligned memory allocator
template<typename T, size_t Alignment = 32>
class AlignmentAllocator {
 public:
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  
  template<typename U>
  struct rebind {
    typedef AlignmentAllocator<U, Alignment> other;
  };
  
  AlignmentAllocator() = default;
  
  template<typename U>
  AlignmentAllocator(const AlignmentAllocator<U, Alignment>&) {}
  
  pointer allocate(size_type n) {
    size_t size = n * sizeof(T);
    void* ptr = nullptr;
    
#ifdef _MSC_VER
    ptr = _aligned_malloc(size, Alignment);
#else
    if (posix_memalign(&ptr, Alignment, size) != 0) {
      ptr = nullptr;
    }
#endif
    
    if (!ptr) {
      throw std::bad_alloc();
    }
    return static_cast<pointer>(ptr);
  }
  
  void deallocate(pointer ptr, size_type) {
#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }
  
  template<typename U, typename... Args>
  void construct(U* ptr, Args&&... args) {
    new(ptr) U(std::forward<Args>(args)...);
  }
  
  template<typename U>
  void destroy(U* ptr) {
    ptr->~U();
  }
  
  bool operator==(const AlignmentAllocator&) const { return true; }
  bool operator!=(const AlignmentAllocator&) const { return false; }
};

}  // namespace Common
}  // namespace LightGBM

// CHECK macro for assertions
#define CHECK_EQ(a, b) assert((a) == (b))
#define CHECK_NE(a, b) assert((a) != (b))
#define CHECK_GT(a, b) assert((a) > (b))
#define CHECK_GE(a, b) assert((a) >= (b))
#define CHECK_LT(a, b) assert((a) < (b))
#define CHECK_LE(a, b) assert((a) <= (b))

#endif  // LIGHTGBM_UTILS_COMMON_H_