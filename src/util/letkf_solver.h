#ifndef LETKF_SOLVER_H_
#define LETKF_SOLVER_H_

// local ensemble transform Kalman filter
// formulation by [Y. Zeng et al., RMetS, 2016]

#include <cmath>
#include <iostream>

#include "cu_matrix.hpp"
#include "cu_matrix_cublas_wrapper.hpp"
#include "cu_matrix_eigensolver_wrapper.hpp"
#include "invoker.hpp"
#include "port.hpp"
#include "mpi_safe_call.hpp"
#include "range.hpp"
#include "timer.hpp"

#define DEBUG_POINT() std::cout << "[D] " << __FILE__ << ": " << __LINE__ << std::endl

namespace util {

class letkf_solver {
public:
    using matrix_t = util::cu_matrix::matrix;
    using vector_t = util::cu_matrix::vector;

private:
    // libs
    util::cu_matrix::cublas_wrapper cublas;
    util::cu_matrix::eigensolver_wrapper eigensolver;
    util::timer timer;
    std::vector<MPI_Request> mpi_requests_xk, mpi_requests_yk;

    // sizes
    size_t n_ens, n_stt, n_obs, n_batch; // ensemble, state vector, observation vector, batch

    // input matrix / vector
    matrix_t X;   // [n_stt, n_ens] ensemble of cal
    std::vector<float> X_host;
    matrix_t Y;   // [n_obs, n_ens] Y = H(X);
    vector_t yo;  // [n_obs] observations 

    // temporary variables for ensemble statistics
    vector_t xbar; // [n_stt] emsemble mean of X
    vector_t ybar; // [n_obs] ensemble mean of Y
    matrix_t dX;   // [n_stt, n_ens] X - xbar
    matrix_t dY;   // [n_obs, n_ens] Y - ybar
    vector_t dyo;  // [n_obs] yo - ybar

    // hyper paramter
    float beta;    // covariance inflation factor
    matrix_t rR;  // [n_obs, n_obs] (rho o R)^-1

    // eigen of letkf
    matrix_t Q;   // [n_ens, n_ens] the matrix to be engenvalue-decomposed
    matrix_t V;   // [n_ens, n_ens] eigenvectors matrix
    vector_t d;   // [n_ens] eigenvalues
    matrix_t D;   // [n_ens, n_ens] D = diag(d)
    matrix_t tmp_ee; // [n_ens, n_ens] temporal var
    matrix_t tmp_oe; // [n_obs, n_ens] temporal var
    vector_t tmp_o; // [n_obs] temporal var
    vector_t tmp_e; // [n_ens] temporal var

    // solutions in ensemble space
    matrix_t P;   // [n_ens, n_ens] covariance matrix
    vector_t w;   // [n_ens] ensemble mean
    matrix_t W;   // [n_ens, n_ens] ensemble perturbation

    // solution in float space
    matrix_t Xsol;   // [n_stt, n_ens]

public:
    void init_geo(size_t n_ens, size_t n_stt, size_t n_obs, size_t n_batch) {
        this->n_ens = n_ens;
        this->n_stt = n_stt;
        this->n_obs = n_obs;
        this->n_batch = n_batch;
        X.init(n_stt, n_ens, n_batch);
        X_host.resize(X.size());
        Y.init(n_obs, n_ens, n_batch);
        yo.init(n_obs, n_batch);
        xbar.init(n_stt, n_batch);
        ybar.init(n_obs, n_batch);
        dyo.init(n_obs, n_batch);
        dX.init(n_stt, n_ens, n_batch);
        dY.init(n_obs, n_ens, n_batch);
        beta = 1;
        rR.init(n_obs, n_obs, n_batch);
        rR.set_identity();
        Q.init(n_ens, n_ens, n_batch);
        V.init(n_ens, n_ens, n_batch);
        d.init(n_ens, n_batch);
        D.init(n_ens, n_ens, n_batch);
        tmp_ee.init(n_ens, n_ens, n_batch);
        tmp_oe.init(n_obs, n_ens, n_batch);
        tmp_o.init(n_obs, n_batch);
        tmp_e.init(n_ens, n_batch);
        P.init(n_ens, n_ens, n_batch);
        w.init(n_ens, n_batch);
        W.init(n_ens, n_ens, n_batch);
        Xsol.init(n_stt, n_ens, n_batch);
    }

    void inspect() {
        size_t n_total = X.size() + Y.size() + yo.size() + rR.size()
                + xbar.size() + ybar.size() + dX.size() + dY.size() + dyo.size()
                + Q.size() + V.size() + d.size() + D.size()
                + tmp_ee.size() + tmp_oe.size() + tmp_o.size() + tmp_e.size()
                + P.size() + w.size() + W.size()
                + Xsol.size();
        double b_total = n_total * sizeof(float) / 1024. / 1024. / 1024.;
        std::cout << "letkf_solver info:" << std::endl
            << " typeof: float32" << std::endl
            << " size of ensemble: " << n_ens << std::endl
            << " size of state vector: " << n_stt << std::endl
            << " size of observation vector: " << n_obs << std::endl
            << " size of batches: " << n_batch << std::endl
            << " total number of vector/matrix components: " << n_total
            << " (" << b_total << " GiB)" << std::endl;
    }

    /// collect xk; in case that all k are confined in self MPI process
    template<class Func> void set_xk(size_t k_ens, size_t i_batch, dim3 nb, dim3 nt, Func func) {
        auto* xk = X.ptr(0, k_ens, i_batch);
        util::invoke_device<<<nb, nt>>>(func, xk, X.stride());
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    /// collect xk; in case that each k is distributed into each MPI process;
    void reorder_xk(const matrix_t& src, matrix_t& dst) {
        /// reorder [ens, batch, stt] --> [batch, ens, stt]
        const size_t n_batch = this->n_batch;
        const size_t n_ens = this->n_ens;
        const float* X_ebs = src.ptr();
        float* X_bes = dst.ptr();
        const size_t n_stt = this->n_stt;
        util::port::pfor3d<util::port::cuda>(
            util::port::thread3d(n_stt, n_ens, n_batch),
            [=] __device__ (util::port::thread3d, int i_stt, int i_ens, int i_batch) {
                X_bes[i_stt + i_ens*n_stt + i_batch*n_stt*n_ens]
                = X_ebs[i_stt + i_batch*n_stt + i_ens*n_stt*n_batch];
            }
        );
    }
    void mpi_allgather_xk(MPI_Comm comm, const float* xk) {
        const size_t size = n_stt*n_batch;
        MPI_SAFE_CALL(MPI_Allgather(xk, size, MPI_FLOAT, 
                    dX.ptr(), size, MPI_FLOAT, comm));
        reorder_xk(dX, X);
    }

    /// collect xk; int case distibuted batch
    void mpi_ialltoall_xk(MPI_Comm comm, MPI_Request* req, const float* xk) {
        const size_t size = n_stt * n_batch;
        MPI_Ialltoall(xk, size, MPI_FLOAT, dX.ptr(), size, MPI_FLOAT, comm, req);
    }
    void mpi_ialltoall_xk_host(MPI_Comm comm, MPI_Request* req, const float* xk) {
        const size_t size = n_stt * n_batch;
        MPI_Ialltoall(xk, size, MPI_FLOAT, X_host.data(), size, MPI_FLOAT, comm, req);
    }
    void copy_xk_host_to_device() {
        CUDA_SAFE_CALL(cudaMemcpy(dX.ptr(), X_host.data(), dX.size()*sizeof(float), cudaMemcpyHostToDevice));
    }
    void reorder_xk_after_ialltoall() {
        reorder_xk(dX, X);
    }

    /// collect yk; in case that all k are confined in self MPI process
    template<class Func> void set_yk(size_t k_ens, size_t i_batch, dim3 nb, dim3 nt, Func func) {
        auto* Yptr = Y.ptr(0, k_ens, i_batch);
        util::invoke_device<<<nb, nt>>>(func, Yptr, Y.stride());
        CUCHECK();
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    /// collect yk; in case that each k is distributed into each MPI process;
    void reorder_yk(const matrix_t& src, matrix_t& dst) {
        /// reorder [ens, batch, stt] --> [batch, ens, stt]
        const size_t n_batch = this->n_batch;
        const size_t n_ens = this->n_ens;
        const size_t n_obs = this->n_obs;
        const float* Y_ebs = src.ptr();
        float* Y_bes = dst.ptr();
        util::port::pfor3d<util::port::cuda>(
            util::port::thread3d(n_obs, n_ens, n_batch),
            [=] __device__ __host__ (util::port::thread3d, int i_obs, int i_ens, int i_batch) {
                Y_bes[i_obs + i_ens*n_obs + i_batch*n_obs*n_ens]
                = Y_ebs[i_obs + i_batch*n_obs + i_ens*n_obs*n_batch];
            }
        );
    }
    void mpi_allgather_yk(MPI_Comm comm, const float* yk) {
        const size_t size = n_obs*n_batch;
        MPI_SAFE_CALL(MPI_Allgather(yk, size, MPI_FLOAT, 
                    dY.ptr(), size, MPI_FLOAT, comm));
        reorder_yk(dY, Y);
    }

    /// collect yk; in case distributed batch
    void mpi_alltoall_yk(MPI_Comm comm, const float* yk) {
        const size_t size = n_obs * n_batch;
        MPI_Alltoall(yk, size, MPI_FLOAT, dY.ptr(), size, MPI_FLOAT, comm);
    }
    void mpi_ialltoall_yk(MPI_Comm comm, MPI_Request* req, const float* yk) {
        const size_t size = n_obs * n_batch;
        MPI_Ialltoall(yk, size, MPI_FLOAT, dY.ptr(), size, MPI_FLOAT, comm, req);
    }
    void reorder_yk_after_ialltoall() {
        reorder_yk(dY, Y);
    }

    /// set observation data
    template<class Func> void set_yo(size_t i_batch, dim3 nb, dim3 nt, Func func) {
        auto* y = this->yo.ptr(0, i_batch);
        util::invoke_device<<<nb, nt>>>(func, y, yo.stride());
        CUCHECK();
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    /// set error covariance R^-1
    template<class Func> void set_rR(dim3 nb, dim3 nt, Func func) {
        util::invoke_device<<<nb, nt>>>(func, rR.ptr(), rR.n_rows(), rR.stride());
        CUCHECK();
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    void set_rR_scalar(float scale) { rR.set_identity(scale); }

    void set_covariance_inflation(float scale) { beta = scale; }

    void solve_evd(int verbose=0, std::string save_prefix="", int t=0) {
        timer.start("BLAS-like");
        // save packed arrays
        if(save_prefix.size() > 0) {
            timer.transit("savemat");
            Y.save(save_prefix, "_Y", t);
            timer.transit("BLAS-like");
        }

        // prepare ybar, dY, innovation
        Y.export_column_mean(ybar);
        cublas.copy(Y, dY);
        dY.columnwise_add(ybar, -1);
        cublas.copy(yo, dyo);
        cublas.axpy(-1, ybar, dyo);

        // Q := (Ne-1)I/beta + dYT * rR * dY
        Q.set_identity((n_ens-1)/beta);
        cublas.gemm(tmp_oe, rR, dY);
        cublas.gemm(Q, dY, tmp_oe, CUBLAS_OP_T, CUBLAS_OP_N, 1, 1);

        // Q =: V * diag(d) * V^T
        cublas.copy(Q, V);
        timer.transit("EVD");
        eigensolver.solve(V, d);
        if(save_prefix.size() > 0) { 
            timer.transit("savemat");
            Q.save(save_prefix, "_Q", t);
            V.save(save_prefix, "_Qsol", t);
            timer.transit("BLAS-like");
        }
        timer.stop_and_ignore_latter();
    }

    void solve_xa(int verbose=0, std::string save_prefix="", int t=0) {
        timer.start("BLAS-like");

        // save packed arrays
        if(save_prefix.size() > 0) {
            timer.transit("savemat");
            X.save(save_prefix, "_X", t);
            timer.transit("BLAS-like");
        }
        // prepare xbar, dX
        X.export_column_mean(xbar);
        cublas.copy(X, dX);
        dX.columnwise_add(xbar, -1);

        // P := V * D^-1 * VT
        D.set_diag(d, 1, -1);
        cublas.gemm(tmp_ee, D, V, CUBLAS_OP_N, CUBLAS_OP_T);
        cublas.gemm(P, V, tmp_ee);

        // w := P * dYT * rR * dyo
        cublas.gemv(tmp_o, rR, dyo);
        cublas.gemv(tmp_e, dY, tmp_o, CUBLAS_OP_T);
        cublas.gemv(w, P, tmp_e);

        if(save_prefix.size() > 0) { 
            timer.transit("savemat");
            P.save(save_prefix, "_Pa", t);
            w.save(save_prefix, "_wa", t);
            rR.save(save_prefix, "_rR", t);
            dY.save(save_prefix, "_dY", t);
            dyo.save(save_prefix, "_dyo", t);
            yo.save(save_prefix, "_yo", t);
            W.save(save_prefix, "_dW", t);
            timer.transit("BLAS-like");
        }

        // W := sqrt(Ne - 1) * sqrtm(P)
        D.set_diag(d, 1, -0.5);
        cublas.gemm(tmp_ee, D, V, CUBLAS_OP_N, CUBLAS_OP_T);
        cublas.gemm(W, V, tmp_ee, CUBLAS_OP_N, CUBLAS_OP_N, sqrt(n_ens-1));

        // Xsol = xbar*1.T + dX*(wbar*1.T + dW)
        W.columnwise_add(w);
        cublas.gemm(Xsol, dX, W);
        Xsol.columnwise_add(xbar);

        if(save_prefix.size() > 0) {
            timer.transit("savemat");

            // Xsol = ...
            Xsol.save(save_prefix, "_Xsol", t);

            // Z := W - I
            D.set_identity();
            cublas.gemm(W, D, D, CUBLAS_OP_N, CUBLAS_OP_N, -1, 1);
            W.save(save_prefix, "_Z", t);
            timer.transit("BLAS-like");
        }
        timer.stop_and_ignore_latter();
    }

    /// sanity check; Xsol[...] is not expected to be nan
    void sanity() const {
        auto* Xsol_ptr = Xsol.ptr();
        util::port::pfor3d<util::port::cuda>(
            util::port::thread3d(Xsol.size()),
            [=] __device__ (util::port::thread3d, int i, int, int) {
                auto x = Xsol_ptr[i];
                if( x != x ) {
                    printf("%s:%d: fatal: Xsol[%d] is nan\n", __FILE__, __LINE__, i);
                    *((int*)NULL) = -1;
                }
            }
        );
    }

    /// output result
    template<class Func> void update_xk(size_t k_ens, size_t i_batch, dim3 nb, dim3 nt, Func func) {
        sanity();
        const auto* xk = Xsol.ptr(0, k_ens, i_batch);
        util::invoke_device<<<nb, nt>>>(func, xk, Xsol.stride());
        CUCHECK();
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    void reorder_Xsol(const matrix_t& src, matrix_t& dst) {
        /// reorder [batch, ens, stt] --> [ens, batch, stt]
        const size_t n_batch = this->n_batch;
        const size_t n_ens = this->n_ens;
        const float* X_bes = src.ptr();
        float* X_ebs = dst.ptr();
        const size_t n_stt = this->n_stt;
        util::port::pfor3d<util::port::cuda>(
            util::port::thread3d(n_stt, n_ens, n_batch),
            [=] __device__ (util::port::thread3d, int i_stt, int i_ens, int i_batch) {
                X_ebs[i_stt + i_batch*n_stt + i_ens*n_stt*n_batch]
                 = X_bes[i_stt + i_ens*n_stt + i_batch*n_stt*n_ens];
            }
        );
    }
    void mpi_alltoall_Xsol(MPI_Comm comm) {
        reorder_Xsol(Xsol, X);
        const size_t size = n_stt * n_batch;
        MPI_SAFE_CALL(MPI_Alltoall(X.ptr(), size, MPI_FLOAT,
                    Xsol.ptr(), size, MPI_FLOAT, comm));
    }

public:
    const auto& get_timer() const { return timer; }
    void clear_timer() { timer.clear(); }

};

} // namespace

#endif
