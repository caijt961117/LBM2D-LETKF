#ifndef CUDA_SAFE_CALL_HPP_
#define CUDA_SAFE_CALL_HPP_

#include <string>
#include <cuda_runtime_api.h>
#include <curand.h>

#include "runtime_error.hpp"
#include "curandGetErrorString.hpp"
#include "cublasGetErrorString.hpp"
#include "cusolverGetErrorString.hpp"

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

#define CURAND_SAFE_CALL_FAILED( error ) \
  { \
    throw STD_RUNTIME_ERROR(std::string("CURAND failed: ") + curandGetErrorString(error)); \
  }

#define CURAND_SAFE_CALL( ... ) \
  { \
    curandStatus_t error = __VA_ARGS__; \
    if(error != CURAND_STATUS_SUCCESS) { \
      CURAND_SAFE_CALL_FAILED(error); \
    } \
  }

#define CUBLAS_SAFE_CALL_FAILED( error ) \
  { \
    throw STD_RUNTIME_ERROR(std::string("CUBLAS failed: ") + cublasGetErrorString(error)); \
  }

#define CUBLAS_SAFE_CALL( ... ) \
  { \
    cublasStatus_t error = __VA_ARGS__; \
    if(error != CUBLAS_STATUS_SUCCESS) { \
      CUBLAS_SAFE_CALL_FAILED(error); \
    } \
  }

#define CUSOLVER_SAFE_CALL_FAILED( error ) \
  { \
    throw STD_RUNTIME_ERROR(std::string("CUSOLVER failed: ") + cusolverGetErrorString(error)); \
  }

#define CUSOLVER_SAFE_CALL( ... ) \
  { \
    cusolverStatus_t error = __VA_ARGS__; \
    if(error != CUSOLVER_STATUS_SUCCESS) { \
      CUSOLVER_SAFE_CALL_FAILED(error); \
    } \
  }

static inline void cusolverSanity(cusolverStatus_t error, int info) {
    if(error != CUSOLVER_STATUS_SUCCESS) {
        throw STD_RUNTIME_ERROR(std::string("cusolver failed: `") + cusolverGetErrorString(error) + "`"
                + "; info: `" + std::to_string(info) + "`"); \
    }
}


#else

#define CUDA_SAFE_CALL( ... ) __VA_ARGS__
#define CUCHECK( ... ) __VA_ARGS__
#define CURAND_SAFE_CALL( ... ) __VA_ARGS__

#endif // ifndef NODEBUG
  
#endif // ifndef CUDA_SAFE_CALL_HPP_

