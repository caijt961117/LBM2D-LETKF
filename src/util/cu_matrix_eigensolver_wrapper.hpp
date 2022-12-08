#ifndef UTIL_CU_MATRIX_EIGENSOLVER_WRAPPER_HPP_
#define UTIL_CU_MATRIX_EIGENSOLVER_WRAPPER_HPP_

#ifdef USE_EIGENG_BATCHED
#include "eigen_GPU_batch.hpp"
#else
#include <cusolverDn.h>
#endif

#include "cu_matrix.hpp"

namespace util {
namespace cu_matrix {

#ifdef USE_EIGENG_BATCHED

// EigenG wrapper for eigenvalue decomposition ()
//template<class T=float>
class eigensolver_wrapper {
    bool is_initialized_ = false;
    bool accuracy_test = false;

    int L_  = 0;
    int nm_ = 0;
    int n_  = 0;
    int m_  = 0;
    cudaStream_t stream_;
    device_memory<float> workspace_;

    public:
    eigensolver_wrapper() {  }
    ~eigensolver_wrapper(){ cudaStreamDestroy( stream_ ); }

    void init(const matrix& mat, const vector& vec) {
        L_  = vec.n_batch();
        nm_ = mat.n_rows();
        n_  = vec.n();
        m_  = mat.n_cols();

        size_t worksize = 0;
        cudaDeviceSynchronize();
        eigen_GPU_batch_BufferSize(
                L_,
                nm_,
                n_,
                m_,
                mat.mem().ptr(),
                vec.mem().ptr(),
                &worksize);
        cudaDeviceSynchronize();
        worksize = ( worksize / sizeof(float) ) + 1;
        workspace_.reallocate(worksize);
        cudaStreamCreate( &stream_ );
        is_initialized_ = true;
    }

    void solve(matrix& mat, vector& vec) {
        if(!is_initialized_) { init(mat, vec); }
        cudaDeviceSynchronize();
        eigen_GPU_batch(
                L_,
                nm_,
                n_,
                m_,
                mat.mem().ptr(),
                vec.mem().ptr(),
                workspace_.ptr(),
                stream_);
        cudaDeviceSynchronize();
    } 
};

#else

// cusolverDn wrapper for eigenvalue decomposition (Jacobi method, batched)
class eigensolver_wrapper {
    cusolverDnHandle_t handle_;
    syevjInfo_t params_;
    device_memory<float> workspace_;
    device_memory<int> info_;
    bool is_initialized_ = false;

    // Jacobi params
    static constexpr float eps = 1e-6;
    static constexpr int max_sweeps = 100;

public:
    eigensolver_wrapper() { CUSOLVER_SAFE_CALL(cusolverDnCreate(&handle_)); }
    ~eigensolver_wrapper() { cusolverDnDestroy(handle_); }

    void init(const matrix& mat, const vector& vec) {
        // Jacobi parameter
        CUSOLVER_SAFE_CALL(cusolverDnCreateSyevjInfo(&params_));
        CUSOLVER_SAFE_CALL(cusolverDnXsyevjSetTolerance(params_, eps));
        CUSOLVER_SAFE_CALL(cusolverDnXsyevjSetMaxSweeps(params_, max_sweeps));
        CUSOLVER_SAFE_CALL(cusolverDnXsyevjSetSortEig(params_, 0));

        // workspace
        int worksize=0;
        CUSOLVER_SAFE_CALL(cusolverDnSsyevjBatched_bufferSize(
                    handle_,
                    CUSOLVER_EIG_MODE_VECTOR,
                    CUBLAS_FILL_MODE_LOWER,
                    mat.n_rows(), mat.mem().ptr(),
                    vec.n(), vec.mem().ptr(),
                    &worksize,
                    params_,
                    vec.n_batch()
                    ));
        workspace_.reallocate(worksize);
        info_.reallocate(1);
        is_initialized_ = true;
    }

   void solve(matrix& mat, vector& vec) {
        if(!is_initialized_) { init(mat, vec); }
        auto status = cusolverDnSsyevjBatched(
                handle_,
                CUSOLVER_EIG_MODE_VECTOR,
                CUBLAS_FILL_MODE_LOWER,
                mat.n_rows(), mat.mem().ptr(),
                vec.n(), vec.mem().ptr(),
                workspace_.ptr(), workspace_.size(),
                info_.ptr(),
                params_,
                vec.n_batch()
                );
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        cusolverSanity(status, info_.to_host()[0]);
    }
};
#endif // ifdef USE_EIGENG_BATCHED

}} // namespace

#endif
