#ifndef LIGHTGBM_UTILS_OPENMP_WRAPPER_H_
#define LIGHTGBM_UTILS_OPENMP_WRAPPER_H_

#ifdef _OPENMP
#include <omp.h>
#else
// Mock OpenMP functions when OpenMP is not available
inline int omp_get_thread_num() { return 0; }
inline int omp_get_num_threads() { return 1; }
inline int omp_get_max_threads() { return 1; }
inline void omp_set_num_threads(int num_threads) { (void)num_threads; }
#endif

// OpenMP wrapper function
inline int OMP_NUM_THREADS() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

// Prefetch macros
#ifdef __GNUC__
  #define PREFETCH_T0(addr) __builtin_prefetch(addr, 0, 3)
#elif defined(_MSC_VER)
  #include <intrin.h>
  #define PREFETCH_T0(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#else
  #define PREFETCH_T0(addr) // No-op for other compilers
#endif

#endif  // LIGHTGBM_UTILS_OPENMP_WRAPPER_H_