#include <cuda_runtime.h>
#include "cuda_safe_call.hpp"
#include "runtime_error.hpp"

namespace util {

template<class T>
class cu_device_ptr {
public:
    using this_type = cu_device_ptr<T>;
    using value_type = T;
protected:
    T* data_ = nullptr;
    std::size_t size_ = 0;

public:
    cu_device_ptr() = delete;
    cu_device_ptr(const cu_device_ptr&) = delete;
    template<class U> void operator=(U) = delete;
    cu_device_ptr(const std::size_t& size): size_(size) { CUDA_SAFE_CALL(cudaMalloc(&data_, size * sizeof(T))); }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    void swap(this_type& ptr) {
        RUNTIME_ASSERT(size() == ptr.size());
        T* tmp = data_;
        data_ = ptr.data_;
        ptr.data_ = tmp;
    }

    ~cu_device_ptr() { reset(); }
    void reset() {
        if(size_ != 0) {
            CUDA_SAFE_CALL(cudaFree(data_));
            size_ = 0;
        }
    }
};

}
