#ifndef CU_MATRIX_HPP_
#define CU_MATRIX_HPP_

#include <hdf5.h>
#include <hdf5_hl.h>

#include <array>
#include <string>

#include "device_memory.hpp"
#include "runtime_error.hpp"
#include "invoker.hpp"
#include "port.hpp"
#include "cuda_safe_call.hpp"
#include "typename.hpp"
#include "h5_wrapper.hpp"
#include "range.hpp"

namespace util {
namespace cu_matrix {

template<class T, class Impl_>
class array3dx {
    static constexpr size_t align_stride = 1; // stride restriction in cublasSgemmStridedBatched
    size_t n_rows_, n_cols_, stride_, n_batch_;
    device_memory<T> mem_;

public:
    using value_type = T;
    using impl_type = Impl_;

protected: // initializers //
    void init(size_t n_rows, size_t n_cols, size_t n_batch) {
        n_rows_ = n_rows;
        n_cols_ = n_cols;
        stride_ = ((n_cols_ * n_rows_ + align_stride - 1) / align_stride) * align_stride;
        n_batch_ = n_batch;
        mem_.reallocate(stride_ * n_batch_);
    }

public: // assign //
    void set_ones(T scale=1) {
        auto* ptr = mem_.ptr();
        auto n_rows = n_rows_;
        auto n_cols = n_cols_;
        auto stride = stride_;
        auto n_batch = n_batch_;
        util::invoke_device<<< dim3(n_cols, n_batch), n_rows >>>(
            [=] __device__ () {
                const auto i = threadIdx.x;
                const auto j = blockIdx.x;
                const auto b = blockIdx.y;
                const auto idx = i + n_rows*j + stride*b;
                ptr[idx] = scale;
            });
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    void set_zeros() { set_ones(0); }

public: // getter/setters //
          auto& mem()       { return mem_; }
    const auto& mem() const { return mem_; }
    auto size() const { return mem_.size(); }

    T* ptr(size_t i=0, size_t j=0, size_t i_batch=0) {
        return mem().ptr() + i + j * n_rows() + stride()*i_batch;
    }
    const T* ptr(size_t i=0, size_t j=0, size_t i_batch=0) const {
        return mem().ptr() + i + j * n_rows() + stride()*i_batch;
    }

    auto n_cols() const { return n_cols_; }
    auto n_rows() const { return n_rows_; }
    auto stride() const { return stride_; }
    auto n_batch() const { return n_batch_; }

public: // elementwise compound assinment operations //
    #define ECA(op) \
        void operator op(const impl_type& rhs) { \
            auto* ptr = mem().ptr(); \
            auto* ptr_rhs = rhs.mem().ptr(); \
            util::port::pfor3d<util::port::cuda>( \
                util::port::thread3d(mem().size()), \
                [=] __device__ (util::port::thread3d, int i, int, int) { \
                    ptr[i] op ptr_rhs[i]; \
                } \
            );\
        }
    ECA(=)
    ECA(+=)
    ECA(-=)
    ECA(*=)
    ECA(/=)
    #undef ECA

public:
    std::string filename(std::string prefix, std::string label, int idx) const { 
        std::string fname = "";
        fname += prefix;
        fname += "/";
        fname += label;
        fname += "." + util::typenameof<T>() + "_" + std::to_string(n_rows_) + "_" + std::to_string(n_cols_) + "_" + std::to_string(n_batch_);
        if(idx >= 0) { fname += "." + std::to_string(idx); }
        fname += ".h5";
        return fname;
    }

    void save(std::string prefix, std::string label, int idx=0) const {
        static_assert(align_stride == 1);
        std::vector<T> a = mem_.to_host();
        std::array<hsize_t, 3> dims = {{ n_rows_, n_cols_, n_batch_ }};
        hid_t h5fid = H5Fcreate(filename(prefix, label, idx).c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        runtime_assert(h5fid > 0, "could not open file: " + filename(prefix, label, idx));
        herr_t herr = H5LTmake_dataset(h5fid, "data", dims.size(), dims.data(), h5typeof(T()), a.data());
        runtime_assert(herr >= 0, "could not write h5 data");
        H5Fclose(h5fid);
    }

    void load(std::string prefix, std::string label, int idx=0) {
        static_assert(align_stride == 1);
        std::vector<T> a(n_cols_ * n_rows_ * n_batch_);
        hid_t h5fid = H5Fopen(filename(prefix, label, idx).c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        runtime_assert(h5fid > 0, "could not open file: " + filename(prefix, label, idx));
        herr_t herr = H5LTread_dataset(h5fid, "data", h5typeof(T()), a.data());
        runtime_assert(herr >= 0, "could not load data");
        H5Fclose(h5fid);
        CUDA_SAFE_CALL(cudaMemcpy(mem_.ptr(), a.data(), sizeof(T)*n_cols_*n_rows_*n_batch_, cudaMemcpyHostToDevice));
    }

};

template<class T>
class scalar_x : public array3dx<T, scalar_x<T>> {
    using this_type = scalar_x<T>;
    using base_type = array3dx<T, scalar_x<T>>;
public:
    scalar_x<T>(size_t n_batch=0) { this->this_type::init(n_batch); }
    void init(size_t n_batch) { this->base_type::init(1, 1, n_batch); }
    void n_rows() = delete;
    void n_cols() = delete;
    void stride() = delete;
};

template<class T>
class vector_x : public array3dx<T, vector_x<T>> {
    using this_type = vector_x<T>;
    using base_type = array3dx<T, vector_x<T>>;
public:
    vector_x<T>(size_t n=0, size_t n_batch=0) { this->this_type::init(n, n_batch); }
    void init(size_t n=0, size_t n_batch=0) { this->base_type::init(n, 1, n_batch); }
    auto n() const noexcept { return this->base_type::n_rows(); }
    void n_rows() = delete;
    void n_cols() = delete;

public: // batched operation w/ reduction //
    void export_inner_product_self(scalar_x<T>& a) {
        runtime_assert(this->n_batch() == a.n_batch(), "batch size mismatch");
        auto n = this->n();
        auto n_batch = this->n_batch();
        auto stride = this->stride();
        auto* ptr_vec = this->mem().ptr();
        auto* ptr_scalar = a.mem().ptr();
        util::port::pfor3d<util::port::cuda>(
            util::port::thread3d(n_batch),
            [=] __device__ (util::port::thread3d, int i_batch, int, int) {
                T sum2 = 0;
                for(int i=0; i<n; i++) {
                    auto val = ptr_vec[i + i_batch*stride];
                    sum2 += val*val;
                }
                ptr_scalar[i_batch] = sum2;
            }
        );
    }

};


template<class T>
class matrix_x : public array3dx<T, matrix_x<T>> {
    using this_type = matrix_x<T>;
    using base_type = array3dx<T, matrix_x<T>>;
public:
    matrix_x<T>(size_t n_rows=0, size_t n_cols=0, size_t n_batch=0) { this->this_type::init(n_rows, n_cols, n_batch); }
    void init(size_t n_rows=0, size_t n_cols=0, size_t n_batch=0) { this->base_type::init(n_rows, n_cols, n_batch); }

public: // assign diagonal matrix //
    void set_identity(T scale=1) {
        runtime_assert(this->n_rows() == this->n_cols(), "matrices should be square");
        auto* ptr = this->mem().ptr();
        auto n = this->n_rows();
        auto stride = this->stride();
        auto n_batch = this->n_batch();
        util::invoke_device<<< dim3(n, n_batch), n >>>(
            [=] __device__ () {
                const auto i = threadIdx.x;
                const auto j = blockIdx.x;
                const auto b = blockIdx.y;
                const auto idx = i + n*j + stride*b;
                const T s = (i == j ? scale : 0);
                ptr[idx] = s;
            });
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    void set_diag(const vector_x<T>& v, T scale=1, T power=1) {
        runtime_assert(this->n_rows() == this->n_cols(), "matrices should be square");
        auto* ptr = this->mem().ptr();
        auto n = this->n_rows();
        auto stride = this->stride();
        auto n_batch = this->n_batch();
        runtime_assert(v.n() == this->n_rows(), "vector/matrix size mismatch");
        auto* vp = v.ptr();
        auto v_stride = v.stride();
        util::invoke_device<<< dim3(n, n_batch), n>>>(
            [=] __device__ () {
                const auto i = threadIdx.x;
                const auto j = blockIdx.x;
                const auto b = blockIdx.y;
                const auto idx = i + n*j + stride*b;
                if (i == j) { // diagonal
                    const auto v_idx = i + v_stride*b;
                    ptr[idx] = scale * pow(vp[v_idx], power);
                } else {
                    ptr[idx] = 0;
                }
            });
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

public: // batched operations //
    void columnwise_add(const vector_x<T>& v, T scale=1) {
        auto* ptr = this->mem().ptr();
        auto n_rows = this->n_rows();
        auto n_cols = this->n_cols();
        auto stride = this->stride();
        auto n_batch = this->n_batch();
        runtime_assert(v.n() == this->n_rows(), "vector/matrix size mismatch");
        auto* vp = v.ptr();
        auto v_stride = v.stride();
        util::invoke_device<<< dim3(n_cols, n_batch), n_rows>>>(
            [=] __device__ () {
                const auto i = threadIdx.x;
                const auto j = blockIdx.x;
                const auto b = blockIdx.y;
                const auto idx = i + n_rows*j + stride*b;
                const auto v_idx = i + v_stride*b;
                ptr[idx] += scale*vp[v_idx];
            });
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    void scalar_mul(const scalar_x<T>& a) {
        auto* ptr = this->mem().ptr();
        auto n_rows = this->n_rows();
        auto n_cols = this->n_cols();
        auto stride = this->stride();
        auto n_batch = this->n_batch();
        runtime_assert(a.n_batch() == n_batch, "batch size mismatch");
        auto* sp = a.ptr();
        util::port::pfor3d<util::port::cuda>(
            util::port::thread3d(n_rows, n_cols, n_batch),
            [=] __device__ (util::port::thread3d, int i, int j, int k) {
                const auto idx = i + n_rows*j + stride*k;
                const auto s_idx = k;
                ptr[idx] *= sp[s_idx];
            }
        );
    }

public: // batched operations w/ reduction //
    void export_column_mean(vector_x<T>& v) const {
        runtime_assert(this->n_rows() == v.n(), "vector/matrix size mismatch");
        runtime_assert(this->n_batch() == v.n_batch(), "batch size mismatch");

        // this pointer
        auto n_cols = this->n_cols();
        auto n_rows = this->n_rows();
        auto n_batch = this->n_batch();
        auto stride = this->stride();

        auto* vp = v.ptr();
        auto v_stride = v.stride();

        util::invoke_device<<<n_batch, n_rows>>>([=] __device__ () {
            vp[threadIdx.x + v_stride*blockIdx.x] = 0;
            });
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        for(auto&& j: util::range(n_cols)) {
            auto* ptr = this->ptr();
            util::invoke_device<<<n_batch, n_rows>>>([=] __device__ () {
                const auto i = threadIdx.x;
                const auto b = blockIdx.x;
                const auto idx = i + n_rows*j + stride*b;
                const auto v_idx = i + v_stride*b;
                vp[v_idx] += ptr[idx]/n_cols;
            });
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
        }
    }

    void export_frobenius_inner_product_self(scalar_x<T>& a, T scale=1) {
        runtime_assert(this->n_batch() == a.n_batch(), "batch size mismatch");
        auto n_cols = this->n_cols();
        auto n_rows = this->n_rows();
        auto n_batch = this->n_batch();
        auto stride = this->stride();
        auto* ptr_mat = this->mem().ptr();
        auto* ptr_scalar = a.mem().ptr();
        util::port::pfor3d<util::port::cuda>(
            util::port::thread3d(n_batch),
            [=] __device__ (util::port::thread3d, int i_batch, int, int) {
                T sum2 = 0;
                for(int i_col=0; i_col<n_cols; i_col++) {
                for(int i_row=0; i_row<n_rows; i_row++) {
                    auto val = ptr_mat[i_row + i_col*n_rows + i_batch*stride];
                    sum2 += val*val * scale;
                }
                }
                ptr_scalar[i_batch] = sum2;
            }
        );
    }

    void export_trace(scalar_x<T>& a, bool inverse=false, T eps=1e-30) {
        runtime_assert(this->n_batch() == a.n_batch(), "batch size mismatch");
        auto n_cols = this->n_cols();
        auto n_rows = this->n_rows();
        runtime_assert(n_rows == n_cols, "matrices should be square");
        auto n_batch = this->n_batch();
        auto stride = this->stride();
        auto* ptr_mat = this->mem().ptr();
        auto* ptr_scalar = a.mem().ptr();
        util::port::pfor3d<util::port::cuda>(
            util::port::thread3d(n_batch),
            [=] __device__ (util::port::thread3d, int i_batch, int, int) {
                T sum = 0;
                for(int i=0; i<n_rows; i++) {
                    auto val = ptr_mat[i + i*n_rows + i_batch*stride];
                    if(val < eps) { continue; }
                    if(inverse) { val = (T)1. / val; }
                    sum += val;
                }
                ptr_scalar[i_batch] = sum;
            }
        );
    }

};

// assume fp32
using vector = vector_x<float>;
using matrix = matrix_x<float>;
using scalar = scalar_x<float>;

}} // namespace
#endif
