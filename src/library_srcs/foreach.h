#ifndef FOREACH_H_
#define FOREACH_H_

#include <cstdint>
#include <cassert>
#include <type_traits>
#include <string>
#include "defineCal.h"

#ifdef USE_NVCC
#include "cuda_safe_call.hpp"
#endif


#if defined(USE_INTEL)
  #define PRAGMA_FOR_SIMD  _Pragma("ivdep")

  #define ASSUME_ALIGNED64(VAL) __assume_aligned(VAL, 64)
  #define ASSUME64(VAL) __assume(VAL%64 == 0)

#elif defined(USE_A64FX)
  #define PRAGMA_FOR_SIMD  _Pragma("loop prefetch_sequential soft")

  #define ASSUME_ALIGNED64(VAL)  
  #define ASSUME64(VAL)  
#else
  #define PRAGMA_FOR_SIMD  

  #define ASSUME_ALIGNED64(VAL)  
  #define ASSUME64(VAL)  
#endif


// macro
#ifdef __CUDA_ARCH__
    #define FOR_EACH1D_BLOCKIDX(L, NL) \
        const auto L = blockIdx.x;

    #define FOR_EACH1D_BLOCKIDY(LY, NLY) \
        const auto LY = blockIdx.y;

    #define FOR_EACH3D(I, J, K, NX, NY, NZ) \
        const auto I = threadIdx.x; \
        const auto J = threadIdx.y; \
        const auto K = threadIdx.z; 

    #define FOR_EACH1D_XX(IJK, NN) \
        const auto IJK = threadIdx.x + blockDim.x * blockIdx.x; \
        if(IJK >= NN) { return; }

    #define SKIP_FOR() return
#else
    #define FOR_EACH1D_BLOCKIDX(L, NL) \
        PRAGMA_FOR_SIMD \
        _Pragma("omp parallel for") \
        for(int L=0; L<NL; L++) 

    #define FOR_EACH1D_BLOCKIDY(LY, NLY) \
        for(int LY=0; LY<NLY; LY++) 

    #define FOR_EACH3D(I,J,K, NX,NY,NZ) \
        PRAGMA_FOR_SIMD \
        for(int K=0; K<NZ; K++) \
        PRAGMA_FOR_SIMD \
        for(int J=0; J<NY; J++) \
        PRAGMA_FOR_SIMD \
        for(int I=0; I<NX; I++) 

    #define FOR_EACH1D_XX(IJK, NN) \
        PRAGMA_FOR_SIMD \
        for(int IJK=0; IJK<NN; IJK++) 

    #define SKIP_FOR() continue
#endif


namespace foreach {

struct backend {};

struct openmp    : backend {};
struct cuda      : backend {};


#ifdef USE_NVCC
using opti = cuda;
#else
using opti = openmp;
#endif


inline void sync()
{
#ifdef USE_NVCC
    cudaDeviceSynchronize();
#else
#endif
}

namespace helper {
    #ifdef USE_NVCC
    template<class Func, class... Args> 
    __global__ void exec_gpu(
        Func    func,
        Args... args
        ) 
    {
        func(args...);
    } 
    #endif

    template<class Func, class... Args> 
    void exec_cpu(
        Func    func,
        Args... args
        )
    {
        func(args...);
    }
} // namespace helper


template<class ExecutionBackend, int NX, class Func, class... Args>
void exec_block_amr(
    const int num_tasks,
    Func    func,
    Args... args
    )
{
    if (std::is_same<ExecutionBackend, cuda>::value) {
#ifdef USE_NVCC
        helper::exec_gpu<Func, Args...> <<<
                dim3(num_tasks, 1, 1), 
                dim3(NX, NX, NX)
            >>> (func, args...);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
    }
    else if (std::is_same<ExecutionBackend, openmp>::value) {
        helper::exec_cpu<Func, Args...>(func, args...);
    }
    else {
        static_assert(std::is_base_of<backend, ExecutionBackend>::value, "undefined execution backend");
    }
}


template<class ExecutionBackend, int NX, class Func, class... Args>
void exec_block_amr_L2F(
    const int num_tasks,
    Func    func,
    Args... args
    )
{
    if (std::is_same<ExecutionBackend, cuda>::value) {
#ifdef USE_NVCC
        helper::exec_gpu<Func, Args...> <<<
                dim3(num_tasks, 8, 1), 
                dim3(NX, NX, NX)
            >>> (func, args...);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
    }
    else if (std::is_same<ExecutionBackend, openmp>::value) {
        helper::exec_cpu<Func, Args...>(func, args...);
    }
    else {
        static_assert(std::is_base_of<backend, ExecutionBackend>::value, "unexpected execution backend");
    }
}


template<class ExecutionBackend, int NX, class Func, class... Args>
void exec_block_amr_F2L(
    const int num_tasks,
    Func    func,
    Args... args
    )
{
    if (std::is_same<ExecutionBackend, cuda>::value) {
#ifdef USE_NVCC
        helper::exec_gpu<Func, Args...> <<<
                dim3(num_tasks, 1, 1), 
                dim3(NX/2, NX/2, NX/2)
            >>> (func, args...);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
    }
    else if (std::is_same<ExecutionBackend, openmp>::value) {
        helper::exec_cpu<Func, Args...>(func, args...);
    }
    else {
        static_assert(std::is_base_of<backend, ExecutionBackend>::value, "unexpected execution backend");
    }
}


template<class ExecutionBackend, class Func, class... Args>
void exec_1d(
    const int nn_max,
    Func    func,
    Args... args
    )
{
    if (std::is_same<ExecutionBackend, cuda>::value) {
#ifdef USE_NVCC
        constexpr auto nth = 256;
        const auto nb = (nn_max + nth - 1)/nth;
        helper::exec_gpu<Func, Args...> <<< nb, nth >>> (func, args...);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
    }
    else if (std::is_same<ExecutionBackend, openmp>::value) {
        helper::exec_cpu<Func, Args...>(func, args...);
    }
    else {
        static_assert(std::is_base_of<backend, ExecutionBackend>::value, "unexpected execution backend");
    }
}

} // namespace foreach


#endif
