#ifndef CU_UNIQUE_PTR_HPP_
#define CU_UNIQUE_PTR_HPP_

#include <memory>
#include <type_traits>
#include <cuda_runtime_api.h>

#include "cuda-safe_call.hpp"

namespace util {
// origin: fixstars
// このコードは、CC0 1.0 全世界（パブリックドメイン）としますので、ご自由にコピペしてお使いください https://creativecommons.org/publicdomain/zero/1.0/deed.ja
// This code is licensed under CC0 1.0 Universal (Public Domain). You can use this without any limitation. https://creativecommons.org/publicdomain/zero/1.0/deed.en
struct deleter
{
  void operator()(void* p) const
  {
    CUDA_SAFE_CALL(::cudaFree(p));
  }
};
template<typename T>
using cu_unique_ptr = std::unique_ptr<T, deleter>;

// auto array = cuda::make_unique<float[]>(n);
// ::cudaMemcpy(array.get(), src_array, sizeof(float)*n, ::cudaMemcpyHostToDevice);
template<typename T>
typename std::enable_if<std::is_array<T>::value, cu_unique_ptr<T>>::type make_unique(const std::size_t n)
{
  using U = typename std::remove_extent<T>::type;
  U* p;
  CHECK_CUDA_ERROR(::cudaMalloc(reinterpret_cast<void**>(&p), sizeof(U) * n));
  return cu_unique_ptr<T>{p};
}

// auto value = cuda::make_unique<my_class>();
// ::cudaMemcpy(value.get(), src_value, sizeof(my_class), ::cudaMemcpyHostToDevice);
template<typename T>
cu_unique_ptr<T> make_unique()
{
  T* p;
  CUDA_SAFE_CALL(::cudaMalloc(reinterpret_cast<void**>(&p), sizeof(T)));
  return cu_unique_ptr<T>{p};
}

}

#endif

