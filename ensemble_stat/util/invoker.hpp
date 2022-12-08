#ifndef UTIL_INVOKER_HPP_
#define UTIL_INVOKER_HPP_

// usage: i.e.
//      util::invoke_device<<<N/256, 256>>>(
//        []__device__(real* p) {
//          const long tid = threadIdx.x + blockIdx.x*blockDim.x;
//          p[tid] = real(tid) / real(blockDim.x*gridDim.x);
//        }, hoge_pointer
//      );

namespace util {
template<class Func, class... Args> __global__ void invoke_device(Func func, Args... args) { func(args...); }
}

#endif

