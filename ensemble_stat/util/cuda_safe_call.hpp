#ifndef CUDA_SAFE_CALL_HPP_
#define CUDA_SAFE_CALL_HPP_

#include "runtime_error.hpp"
#include <string>
#include <cuda_runtime_api.h>

#ifndef NODEBUG

#define CUDA_SAFE_CALL_FAILED( error ) \
  { \
    throw STD_RUNTIME_ERROR(std::string("CUDA failed: ") + cudaGetErrorString(error)); \
  }


#define CUDA_SAFE_CALL( ... ) \
  { \
    cudaError_t error = __VA_ARGS__; \
    if(error != cudaSuccess) { \
      CUDA_SAFE_CALL_FAILED(error); \
    } \
  }

#define CUCHECK( ... ) \
  { \
    __VA_ARGS__; \
    cudaError_t error = cudaGetLastError(); \
    if(error != cudaSuccess) { \
      CUDA_SAFE_CALL_FAILED(error); \
    } \
  }

#else

#define CUDA_SAFE_CALL( ... ) __VA_ARGS__
#define CUCHECK( ... ) __VA_ARGS__

#endif // ifndef NODEBUG
  
#endif // ifndef CUDA_SAFE_CALL_HPP_

