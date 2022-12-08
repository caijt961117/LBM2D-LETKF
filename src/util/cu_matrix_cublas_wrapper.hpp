#ifndef UTIL_CU_MATRIX_CUBLAS_WRAPPER_HPP_
#define UTIL_CU_MATRIX_CUBLAS_WRAPPER_HPP_

#include <cublas.h>
#include <cublas_v2.h>

#include "cu_matrix.hpp"

namespace util {
namespace cu_matrix {

//
class cublas_wrapper {
    cublasHandle_t handle_;

public:
    cublas_wrapper() { CUBLAS_SAFE_CALL(cublasCreate(&handle_)); }
    ~cublas_wrapper() { cublasDestroy(handle_); }

    // copy wrapper;; y = x
    void copy(const vector& x, vector& y) {
        runtime_assert(x.n() == y.n(), "vector size mismatch");
        runtime_assert(x.stride() == y.stride(), "vector size mismatch");
        runtime_assert(x.n_batch() == y.n_batch(), "batch size mismatch");
        CUBLAS_SAFE_CALL(cublasScopy(
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
        CUBLAS_SAFE_CALL(cublasScopy(
            handle_,
            x.stride() * x.n_batch(),
            x.ptr(), 1,
            y.ptr(), 1
            ));
    }

    // axpy wrapper;; y = a*x + y
    void axpy(float alpha, const vector& x, vector& y) {
        runtime_assert(x.n() == y.n(), "vector size mismatch");
        runtime_assert(x.stride() == y.stride(), "vector size mismatch");
        runtime_assert(x.n_batch() == y.n_batch(), "batch size mismatch");
        CUBLAS_SAFE_CALL(cublasSaxpy(
            handle_,
            x.stride() * x.n_batch(),
            &alpha,
            x.ptr(), 1,
            y.ptr(), 1
            ));
    }

    // nrm2 wrapper;; a = L2norm(x)
    void nrm2(const vector& x, float* a) {
        CUBLAS_SAFE_CALL(cublasSnrm2(
            handle_,
            x.stride() * x.n_batch(),
            x.ptr(), 1,
            a
            ));
    }

    // nrm2 wrapper;; result = L2norm(matA)
    void nrm2(const matrix& matA, float* result) {
        CUBLAS_SAFE_CALL(cublasSnrm2(
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
            float alpha=1,
            float beta=0
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
        CUBLAS_SAFE_CALL(cublasSgemmStridedBatched(
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
            float alpha=1,
            float beta=0
    ) {
        const auto yn = vecy.n();
        const auto An = (transa == CUBLAS_OP_N ? matA.n_rows() : matA.n_cols());
        runtime_assert(yn == An, "matrix-vector size mismatch: n: result vector");
        const auto xk = vecx.n();
        const auto Ak = (transa == CUBLAS_OP_N ? matA.n_cols() : matA.n_rows());
        runtime_assert(xk == Ak, "matrix-vector size mismatch: k: reduction");
        CUBLAS_SAFE_CALL(cublasSgemmStridedBatched(
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
