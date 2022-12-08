#ifndef CURAND_WRAPPER_HPP_
#define CURAND_WRAPPER_HPP_

#include <stdexcept>
#include <curand.h>
#include "cuda_safe_call.hpp"
#include "device_memory.hpp"

namespace util {

class curand_wrapper {
private:
    curandGenerator_t randgen_;
    util::device_memory<float> randbuf_;
    size_t cnt_used_;

public:
    using value_type = float;
    using this_type = curand_wrapper;
    static constexpr size_t default_seed = 5489u;
    static constexpr size_t default_bufsize = 1024u;

public: // initializer
    curand_wrapper(size_t seed=default_seed, size_t bufsize=default_bufsize) {
        reset(seed, bufsize);
    }
    void reset(size_t seed=default_seed, size_t bufsize=default_bufsize) {
        CURAND_SAFE_CALL(( curandCreateGenerator(&randgen_, CURAND_RNG_PSEUDO_MTGP32) ));
        CURAND_SAFE_CALL(( curandSetPseudoRandomGeneratorSeed(randgen_, seed) ));
        randbuf_.reallocate(bufsize);
        cnt_used_ = bufsize;
    }
    curand_wrapper(const this_type&) = delete;

    ~curand_wrapper() { curandDestroyGenerator(randgen_); }

public: // getters
    size_t size_total() const noexcept { return randbuf_.size(); }
    ssize_t size_avail() const noexcept { return size_total() - ssize_t(cnt_used_); }
    float* pop_ptr(size_t pop_cnt) { 
        float* ret_ptr = randbuf_.ptr() + cnt_used_;
        cnt_used_ += pop_cnt;
        if(size_avail() < 0) { throw std::out_of_range("randbuf_"); }
        return ret_ptr;
    }

public: // updators
    // normal distribution N(mean, stdev)
    void gen_rand_normal(float mean, float stdev) {
        CURAND_SAFE_CALL(( curandGenerateNormal(randgen_, randbuf_.ptr(), randbuf_.size(), mean, stdev) ));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        cnt_used_ = 0u;
    }

    // uniform distribution [0, 1)
    void gen_rand_uniform() {
        CURAND_SAFE_CALL(( curandGenerateUniform(randgen_, randbuf_.ptr(), randbuf_.size()) ));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        cnt_used_ = 0u;
    }
};
} // namespace util

#endif
