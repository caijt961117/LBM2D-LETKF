#ifndef CU_ALLOCATOR_HPP_
#define CU_ALLOCATOR_HPP_
#include <new>
#include <stdexcept>
#include <cuda_runtime.h>
#include "cuda_safe_call.hpp"

namespace util {

template<class T> 
struct cu_managed_allocator {
  using value_type = T;
  cu_managed_allocator() {}
  template<class U> cu_managed_allocator(const cu_managed_allocator<U>&) {}
  static T* allocate(std::size_t n) {
    T* ret = nullptr;
    if(n > 0) { CUDA_SAFE_CALL(cudaMallocManaged(&ret, n*sizeof(T))); }
    return ret;
  }
  static void deallocate(T* p, std::size_t n = 0) {
    if(n > 0) { CUDA_SAFE_CALL(cudaFree(p)); }
  }
};
template<class T, class U> bool operator == (const cu_managed_allocator<T>&, const cu_managed_allocator<U>&) { return true; }
template<class T, class U> bool operator != (const cu_managed_allocator<T>&, const cu_managed_allocator<U>&) { return false; }

template<class T> 
struct cu_device_allocator {
  using value_type = T;
  cu_device_allocator() {}
  template<class U> cu_device_allocator(const cu_device_allocator<U>&) {}
  static T* allocate(std::size_t n) {
    T* ret = nullptr;
    if(n > 0) { CUDA_SAFE_CALL(cudaMalloc(&ret, n*sizeof(T))); }
    return ret;
  }
  static void deallocate(T* p, std::size_t n = 0) {
    if(n > 0) { CUDA_SAFE_CALL(cudaFree(p)); }
  }
};
template<class T, class U> bool operator == (const cu_device_allocator<T>&, const cu_device_allocator<U>&) { return true; }
template<class T, class U> bool operator != (const cu_device_allocator<T>&, const cu_device_allocator<U>&) { return false; }

}

#endif
