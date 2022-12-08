#ifndef CUDA_WRAPPER_HPP_
#define CUDA_WRAPPER_HPP_

#include <cuda_runtime.h>
#include <cstdint>

namespace util {
struct cuda_wrapper {

static size_t mem_usage() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return size_t(std::ptrdiff_t(total) - std::ptrdiff_t(free));
}

};
}

#endif
