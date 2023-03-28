#ifndef UTIL_CU_MATRIX_CUBLAS_WRAPPER_HPP_
#define UTIL_CU_MATRIX_CUBLAS_WRAPPER_HPP_

#include <cublas.h>
#include <cublas_v2.h>

#include "cu_matrix.hpp"

namespace util {
namespace cu_matrix {

namespace cublas_api_wrapper__ {
    template<typename T> inline cublasStatus_t
    copy(cublasHandle_t handle, int n, const T* x, int incx, T* y, int incy);
    template<> inline cublasStatus_t
    copy(cublasHandle_t handle, int n, const float* x, int incx, float* y, int incy) {
        return cublasScopy(handle, n, x, incx, y, incy);
    }
    template<> inline cublasStatus_t
    copy(cublasHandle_t handle, int n, const double* x, int incx, double* y, int incy) {
        return cublasDcopy(handle, n, x, incx, y, incy);
    }

    template<typename T> inline cublasStatus_t
    axpy(cublasHandle_t handle, int n, const T* alpha, const T* x, int incx, T* y, int incy);
    template<> inline cublasStatus_t
    axpy(cublasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy) {
        return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
    }
    template<> inline cublasStatus_t
    axpy(cublasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy) {
        return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
    }

    template<typename T> inline cublasStatus_t
    nrm2(cublasHandle_t handle, int n, const T* x, int incx, T* result);
    template<> inline cublasStatus_t
    nrm2(cublasHandle_t handle, int n, const float* x, int incx, float* result) {
        return cublasSnrm2(handle, n, x, incx, result);
    }
    template<> inline cublasStatus_t
    nrm2(cublasHandle_t handle, int n, const double* x, int incx, double* result) {
        return cublasDnrm2(handle, n, x, incx, result);
    }

    template<typename T> inline cublasStatus_t
    gemmStridedBatched(
        cublasHandle_t handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m, int n, int k,
        const T* alpha,
        const T* A, int lda, long long int strideA,
        const T* B, int ldb, long long int strideB,
        const T* beta,
        T* C, int ldc, long long int strideC,
        int batchCount
    );
    template<> inline cublasStatus_t
    gemmStridedBatched(
        cublasHandle_t handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m, int n, int k,
        const float* alpha,
        const float* A, int lda, long long int strideA,
        const float* B, int ldb, long long int strideB,
        const float* beta,
        float* C, int ldc, long long int strideC,
        int batchCount
    ) {
        return cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    }
    template<> inline cublasStatus_t
    gemmStridedBatched(
        cublasHandle_t handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m, int n, int k,
        const double* alpha,
        const double* A, int lda, long long int strideA,
        const double* B, int ldb, long long int strideB,
        const double* beta,
        double* C, int ldc, long long int strideC,
        int batchCount
    ) {
        return cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    }
}

template<typename T>
class cublas_wrapper {
public:
    using real = T;
    using vector = vector_x<real>;
    using matrix = matrix_x<real>;

private:
    cublasHandle_t handle_;

public:
    cublas_wrapper() { CUBLAS_SAFE_CALL(cublasCreate(&handle_)); }
    ~cublas_wrapper() { cublasDestroy(handle_); }

    // copy wrapper;; y = x
    void copy(const vector& x, vector& y) {
        runtime_assert(x.n() == y.n(), "vector size mismatch");
        runtime_assert(x.stride() == y.stride(), "vector size mismatch");
        runtime_assert(x.n_batch() == y.n_batch(), "batch size mismatch");
        CUBLAS_SAFE_CALL(cublas_api_wrapper__::copy(
            handle_,
            x.stride() * x.n_batch(),
            x.ptr(), 1,
            y.ptr(), 1
            ));
    }
    void copy(const matrix& x, matrix& y) {
        runtime_assert(x.n_rows() == y.n_rows(), "matrix size mismatch");
        runtime_assert(x.n_cols() == y.n_cols(), "matrix size mismatch");
        runtime_assert(x.stride() == y.stride(), "matrix size mismatch");
        runtime_assert(x.n_batch() == y.n_batch(), "batch size mismatch");
        CUBLAS_SAFE_CALL(cublas_api_wrapper__::copy(
            handle_,
            x.stride() * x.n_batch(),
            x.ptr(), 1,
            y.ptr(), 1
            ));
    }

    // axpy wrapper;; y = a*x + y
    void axpy(real alpha, const vector& x, vector& y) {
        runtime_assert(x.n() == y.n(), "vector size mismatch");
        runtime_assert(x.stride() == y.stride(), "vector size mismatch");
        runtime_assert(x.n_batch() == y.n_batch(), "batch size mismatch");
        CUBLAS_SAFE_CALL(cublas_api_wrapper__::axpy(
            handle_,
            x.stride() * x.n_batch(),
            &alpha,
            x.ptr(), 1,
            y.ptr(), 1
            ));
    }

    // nrm2 wrapper;; a = L2norm(x)
    void nrm2(const vector& x, real* a) {
        CUBLAS_SAFE_CALL(cublas_api_wrapper__::nrm2(
            handle_,
            x.stride() * x.n_batch(),
            x.ptr(), 1,
            a
            ));
    }

    // nrm2 wrapper;; result = L2norm(matA)
    void nrm2(const matrix& matA, real* result) {
        CUBLAS_SAFE_CALL(cublas_api_wrapper__::nrm2(
            handle_,
            matA.stride() * matA.n_batch(),
            matA.ptr(), 1,
            result
            ));
    }

    // gemm wrapper
    // ;; C = alpha * Ao * Bo + beta * C;; Ao = A or A.T as per transa;; Bo is similary
    // ;; default arguments: C = A * B
    void gemm(matrix& matC,
            const matrix& matA,
            const matrix& matB,
            cublasOperation_t transa=CUBLAS_OP_N,
            cublasOperation_t transb=CUBLAS_OP_N,
            real alpha=1,
            real beta=0
    ) {
        const auto Cm = matC.n_rows();
        const auto Am = (transa == CUBLAS_OP_N ? matA.n_rows() : matA.n_cols());
        runtime_assert(Cm == Am, "matrix size mismatch: m: n_rows");
        const auto Cn = matC.n_cols();
        const auto Bn = (transb == CUBLAS_OP_N ? matB.n_cols() : matB.n_rows());
        runtime_assert(Cn == Bn, "matrix size mismatch: n: n_cols");
        const auto Ak = (transa == CUBLAS_OP_N ? matA.n_cols() : matA.n_rows());
        const auto Bk = (transb == CUBLAS_OP_N ? matB.n_rows() : matB.n_cols());
        runtime_assert(Ak == Bk, "matrix size mismatch: k: reduction");
        CUBLAS_SAFE_CALL(cublas_api_wrapper__::gemmStridedBatched(
                    handle_,
                    transa, transb,
                    Cm, Cn, Ak,
                    &alpha,
                    matA.mem().ptr(), matA.n_rows(), matA.stride(),
                    matB.mem().ptr(), matB.n_rows(), matB.stride(),
                    &beta,
                    matC.mem().ptr(), matC.n_rows(), matC.stride(),
                    matC.n_batch()
                    ));
    }

    // gemv wrapper;; default arguments: y = A * x
    // ! to batch, use gemmBatched instead of gemv
    void gemv(vector& vecy,
            const matrix& matA,
            const vector& vecx,
            cublasOperation_t transa=CUBLAS_OP_N,
            real alpha=1,
            real beta=0
    ) {
        const auto yn = vecy.n();
        const auto An = (transa == CUBLAS_OP_N ? matA.n_rows() : matA.n_cols());
        runtime_assert(yn == An, "matrix-vector size mismatch: n: result vector");
        const auto xk = vecx.n();
        const auto Ak = (transa == CUBLAS_OP_N ? matA.n_cols() : matA.n_rows());
        runtime_assert(xk == Ak, "matrix-vector size mismatch: k: reduction");
        CUBLAS_SAFE_CALL(cublas_api_wrapper__::gemmStridedBatched(
                    handle_,
                    transa, CUBLAS_OP_N,
                    yn, 1, Ak,
                    &alpha,
                    matA.mem().ptr(), matA.n_rows(), matA.stride(),
                    vecx.mem().ptr(), vecx.n(), vecx.stride(),
                    &beta,
                    vecy.mem().ptr(), vecy.n(), vecy.stride(),
                    vecy.n_batch()
                    ));
    }
};

}} // namespace
#endif
