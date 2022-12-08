#ifndef DEVICE_MEMORY_HPP_
#define DEVICE_MEMORY_HPP_

#include <zlib.h>

#include <vector>
#include <string>

#include "cuda_safe_call.hpp"

namespace util {

template<class T>
class device_memory {
public:
    using value_type = T;

private:
    size_t size_ = 0;
    T* ptr_ = nullptr;
    bool is_weak_ = false;

public:
    // initialize as primary pointer
    device_memory<T>(size_t size = 0): size_(size) {
        CUDA_SAFE_CALL(cudaMalloc(&ptr_, sizeof(T) * size_));
    }
    device_memory<T>(const std::vector<T>& v) {
        size_ = v.size();
        CUDA_SAFE_CALL(cudaMalloc(&ptr_, sizeof(T) * size_));
        CUDA_SAFE_CALL(cudaMemcpy(ptr_, v.data(), sizeof(T) * size_, cudaMemcpyHostToDevice));
    }
    device_memory<T>(const device_memory<T>& dv) {
        size_ = dv.size_;
        CUDA_SAFE_CALL(cudaMalloc(&ptr_, sizeof(T) * size_));
        CUDA_SAFE_CALL(cudaMemcpy(ptr_, dv.ptr_, sizeof(T) * size_, cudaMemcpyDeviceToDevice));
    }
    ~device_memory<T>() {
        if(!is_weak_) { cudaFree(ptr_); }
    }

    void reallocate(size_t size) {
        if(ptr_ != nullptr) { cudaFree(ptr_); }
        size_ = size;
        CUDA_SAFE_CALL(cudaMalloc(&ptr_, sizeof(T) * size));
        is_weak_ = false;
    }

    // initialize as weak pointer
    void init_as_weak_ptr(T* ptr, size_t size) {
        size_ = size;
        ptr_ = ptr;
        is_weak_ = true;
    }
    device_memory<T>(T* ptr, size_t size) {
        init_as_weak_ptr(size, ptr);
    }

public:
          T* ptr()       { return ptr_; }
    const T* ptr() const { return ptr_; }

    operator       T*()       { return ptr_; }
    operator const T*() const { return ptr_; }

    auto size() const { return size_; }

public:
    void from_host(const std::vector<T>& v) {
        RUNTIME_ASSERT(size_ >= v.size());
        CUDA_SAFE_CALL(cudaMemcpy(ptr_, v.data(), sizeof(T) * size_, cudaMemcpyHostToDevice));
    }
    auto to_host() const {
        std::vector<T> ret(size_);
        CUDA_SAFE_CALL(cudaMemcpy(ret.data(), ptr_, sizeof(T)*size_, cudaMemcpyDeviceToHost));
        return ret;
    }

public:
    void swap(device_memory<T>& rhs) {
        T* tmp = ptr_;
        ptr_ = rhs.ptr_;
        rhs.ptr_ = tmp;
    }

}; // class

} // namespace

#endif
