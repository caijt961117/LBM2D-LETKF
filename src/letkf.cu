#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>

#include <sys/stat.h> // mkdir

#include "assimilate.h"
#include "lbm.h"
#include "data.h"
#include "config.h"
#include "util/mpi_safe_call.hpp"
#include "util/cuda_safe_call.hpp"
#include "util/invoker.hpp"
#include "util/runtime_error.hpp"

#include "util/port.hpp"

#define DEBUG_PRINT() //if (mpi.rank() == 0) { std::cout << "[D] " << __PRETTY_FUNCTION__ << ": " << __LINE__ << std::endl; }

#ifdef DA_LETKF

// config of letkf
static constexpr auto xyprune = config::da_xyprune;
static_assert(config::nx % xyprune == 0);
static constexpr auto timeprune = config::daprune;
static constexpr real beta = LETKF_COVINF; // covariance inflation factor
static constexpr bool need_save_mat = false;

static constexpr int n_batch0 = config::nx * config::ny;
int n_batch_sp(const util::mpi& mpi) { runtime_assert(n_batch0 % mpi.size() == 0, "batch vs. mpi undivisable"); return n_batch0 / mpi.size(); }
int i_batch_sp_begin(const util::mpi& mpi) { return n_batch_sp(mpi) * mpi.rank(); }

static constexpr int ns_per_grid = config::Q; // lbm

static constexpr int no_per_grid = 3; // rho, u, v

static constexpr int n_stt = ns_per_grid;

static constexpr real d_local = xyprune==2 ? 1.5 : xyprune - 1; 
static constexpr int n_obs_local = xyprune == 1 ? /* pointwise: */ 0 : /* r-localization c=p-1: */ 2;
static constexpr int obs_offset  = xyprune == 1 ? /* pointwise: */ 0 : /* r-localization c=p-1: */ 1;
static constexpr int n_obs_x     = xyprune == 1 ? /* pointwise: */ 1 : /* r-localization c=p-1: */ 4;
static constexpr int n_obs = n_obs_x * n_obs_x * no_per_grid;

void DataAssimilator::init_letkf(const util::mpi& mpi) {
    xk.reallocate(n_batch0 * n_stt);
    xk_host.resize(n_batch0 * n_stt);
    yk.reallocate(n_batch0 * n_obs);

    const int n_batch = n_batch_sp(mpi);
    solver.init_geo(mpi.size(), n_stt, n_obs, n_batch);
    solver.set_rR(dim3(1, 1, n_batch), dim3(n_obs_x, n_obs_x),
        [=] __device__ (real* e, int n_rows, int stride) {
            // obs grid
            const int i = threadIdx.x;
            const int j = threadIdx.y;
            const int ij = i + n_obs_x*j;
            // indices of the batch (== cal grid)
            const int b  = blockIdx.z;
            const int ib = b % config::nx;
            const int jb = b / config::nx;

            // obs error cov
            const int k = ij + ij*n_rows + b*stride;
            const int kofs = n_obs_x*n_obs_x + (n_obs_x*n_obs_x)*n_rows;
            /// u
            e[k] = powf(config::u_ref * config::obs_error, real(-2));
            /// v
            e[k + kofs] = powf(config::u_ref * config::obs_error, real(-2));
            /// rho
            e[k + 2*kofs] = powf(config::rho_ref * config::obs_error, real(-2));

            // R-localization
            if(xyprune > 1) {
                constexpr real cutoff = d_local + 1e-6;
                const int io = (int(ib/xyprune) + i - (n_obs_local-obs_offset))*xyprune;
                const int jo = (int(jb/xyprune) + j - (n_obs_local-obs_offset))*xyprune;
                const real di = ib - io;
                const real dj = jb - jo;
                const real d = sqrtf(di*di + dj*dj) / cutoff;
                const real gaspari_cohn = 
                    (d <= 0) ? 1:
                    (d <= 1) ? -d*d*d*d*d/4. + d*d*d*d/2. + d*d*d*5./8. - d*d*5./3. + 1:
                    (d <= 2) ? d*d*d*d*d/12. - d*d*d*d/2. + d*d*d*5./8. + d*d*5./3. - d*5 + 4 - 2./3./d:
                    0;
                e[k] *= gaspari_cohn;
                e[k + kofs] *= gaspari_cohn;
                e[k + 2*kofs] *= gaspari_cohn;
            }
        });
    solver.set_covariance_inflation(beta);
    if(mpi.rank() == 0) { solver.inspect(); }
    mpi.barrier();
}

void DataAssimilator::assimilate_letkf_uv(data& dat, const util::mpi& mpi, const int t) {
    timer.stop_and_ignore_latter();
    auto step = t / config::iiter;
    if(step % timeprune != 0) { return; }
    if(config::verbose >= 1 && mpi.rank() == 0)  { std::cout << __PRETTY_FUNCTION__ << ": t=" << t << std::endl; }

    const auto* f = dat.d().f.data();
    const auto* rho = dat.d().r.data();
    const auto* u = dat.d().u.data();
    const auto* v = dat.d().v.data();
    auto* ro = this->obse.data();
    auto* uo = this->obse.data() + config::nx * config::ny;
    auto* vo = this->obse.data() + 2*config::nx * config::ny;
    MPI_Request mpi_requests[4]; // load_obs[3]+yk[1], xk[1]


    timer.transit("pack_xk");
    auto* xk = this->xk.ptr();
    util::invoke_device<<< dim3(config::ny, config::Q), config::nx >>>(
        [=] __device__ () {
            const int i = threadIdx.x + blockDim.x * blockIdx.x;
            const int q = blockIdx.y;
            const int i_lbm = config::ijq(i, q);
            const int i_mat = q + config::Q * i;
            xk[i_mat] = f[i_lbm];
        }
    );
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(xk_host.data(), xk, xk_host.size()*sizeof(real), cudaMemcpyDeviceToHost));

    timer.transit("pack_yk");
    auto* yk = this->yk.ptr();
    util::invoke_device<<< dim3(config::ny, config::nx), dim3(n_obs_x, n_obs_x) >>>(
        [=] __device__ () {
            // indices of the rows in the batch
            const int ir = threadIdx.x;
            const int jr = threadIdx.y;
            const int r = ir + jr * n_obs_x;
            // index of the batch
            const int it = blockIdx.x;
            const int jt = blockIdx.y;
            const int bt = it + config::nx * jt;
            // indices of the grid (offset by batch)
            const int ig = it;
            const int jg = jt;
            // indices of the grid
            const int io = (int(ig/xyprune) + ir - (n_obs_local-obs_offset))*xyprune;
            const int jo = (int(jg/xyprune) + jr - (n_obs_local-obs_offset))*xyprune;
            const int di = abs(ig - io);
            const int dj = abs(jg - jo);

            const int iop = (io + config::nx) % config::nx;
            const int jop = (jo + config::ny) % config::ny;

            // pack u, v
            real fc[config::Q];
            for(int q=0; q<config::Q; q++) {
                const auto i = config::ijq(iop, jop, q);
                fc[q] = f[i];
            }
            auto rc = rf(fc);
            auto uc = uf(fc, rc);
            auto vc = vf(fc, rc);
            if(di > n_obs_local*xyprune || dj > n_obs_local*xyprune) { uc=0; vc=0; rc=0; }
            const auto k = r + n_obs*bt;
            yk[k] = uc;
            yk[k + n_obs_x*n_obs_x] = vc;
            yk[k + 2*n_obs_x*n_obs_x] = rc;
        }
    );
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    timer.transit("mpi_yk");
    solver.mpi_ialltoall_yk(mpi.comm(), &mpi_requests[3], yk);

    // comm overlap: load_obs@master&MPI_Ibcast(obse), MPI_Alltoall(yk)
    timer.transit("load_obs");
    if(mpi.rank() == 0) { /// load observation r
        const auto fname = "io/observed/0/rho_" + std::to_string(step) + ".dat";
        auto file = std::ifstream(fname, std::ios::binary);
        runtime_assert(file.is_open(), "IOError: could not open file: " + fname);
        file.read(reinterpret_cast<char*>(ro), sizeof(real) * config::nx * config::ny);
    }
    MPI_Ibcast(ro, /* cnt= */config::nx*config::ny, MPI_FLOAT, /* root= */0, mpi.comm(), &mpi_requests[0]);
    if(mpi.rank() == 0) { /// load observation u
        const auto fname = "io/observed/0/u_" + std::to_string(step) + ".dat";
        auto file = std::ifstream(fname, std::ios::binary);
        runtime_assert(file.is_open(), "IOError: could not open file: " + fname);
        file.read(reinterpret_cast<char*>(uo), sizeof(real) * config::nx * config::ny);
    }
    MPI_Ibcast(uo, /* cnt= */config::nx*config::ny, MPI_FLOAT, /* root= */0, mpi.comm(), &mpi_requests[1]);
    if(mpi.rank() == 0) { /// load observation v
        const auto fname = "io/observed/0/v_" + std::to_string(step) + ".dat";
        auto file = std::ifstream(fname, std::ios::binary);
        runtime_assert(file.is_open(), "IOError: could not open file: " + fname);
        file.read(reinterpret_cast<char*>(vo), sizeof(real) * config::nx * config::ny);
    }
    MPI_Ibcast(vo, /* cnt= */config::nx*config::ny, MPI_FLOAT, /* root= */0, mpi.comm(), &mpi_requests[2]);
    MPI_Waitall(4, mpi_requests, MPI_STATUSES_IGNORE);
    mpi.barrier();

    // finalize yk after comm
    timer.transit("pack_yk");
    solver.reorder_yk_after_ialltoall();

    timer.transit("set_yo");
    runtime_assert(config::ny % mpi.size() == 0, "grid number vs. mpi is undivisible");
    const int yo_n_batch_y = config::ny / mpi.size();
    const int yo_j_batch_offset = i_batch_sp_begin(mpi) / config::nx;
    solver.set_yo(0, dim3(config::nx, yo_n_batch_y), dim3(n_obs_x, n_obs_x),
        [=] __device__ (float* yo, size_t stride) {
            // indices of the rows in the batch
            const int ir = threadIdx.x;
            const int jr = threadIdx.y;
            const int r = ir + jr * n_obs_x;
            // index of the batch
            const int it = blockIdx.x;
            const int jt = blockIdx.y;
            const int bt = it + config::nx * jt;
            // indices of the grid (offset by batch)
            const int ig = it;
            const int jg = jt + yo_j_batch_offset;
            // indices of the grid
            const int io = (int(ig/xyprune) + ir - (n_obs_local-obs_offset))*xyprune;
            const int jo = (int(jg/xyprune) + jr - (n_obs_local-obs_offset))*xyprune;
            const int di = abs(ig - io);
            const int dj = abs(jg - jo);
            const int iop = (io + config::nx) % config::nx;
            const int jop = (jo + config::ny) % config::ny;

            // pack u, v
            const int i = config::ij(iop, jop);
            const int k = r + n_obs*bt;
            auto uc = uo[i];
            auto vc = vo[i];
            auto rc = ro[i];
            if(di > n_obs_local*xyprune || dj > n_obs_local*xyprune) { uc=0; vc=0; rc=0; }
            yo[k] = uc;
            yo[k + n_obs_x*n_obs_x] = vc;
            yo[k + 2*n_obs_x*n_obs_x] = rc;
        } // [=] __device__
    ); // solver.ser_yo

    timer.transit("mpi_xk");
    solver.mpi_ialltoall_xk_host(mpi.comm(), &mpi_requests[0], xk_host.data());

    { /// solver part with overlapped communication of mpi_ialltoall_xk
        timer.transit("solve_comm_ovlp");
        std::string mat_save_prefix = 
            need_save_mat
            ? config::prefix + "/letkf/mat/ens" + std::to_string(mpi.rank())
            : "";
        ::mkdir(mat_save_prefix.c_str(), 0755);
        solver.solve_evd(0, mat_save_prefix, t);
    }

    #ifndef LETKF_NO_MPI_BARRIER
    timer.transit("mpi_barrier");
    mpi.barrier();
    #endif
    timer.transit("mpi_xk");
    MPI_Wait(&mpi_requests[0], MPI_STATUSES_IGNORE);
    timer.transit("pack_xk");
    solver.copy_xk_host_to_device();
    solver.reorder_xk_after_ialltoall();

    { /// non overlapped solver part
        timer.transit("solve");
        std::string mat_save_prefix = 
            need_save_mat
            ? config::prefix + "/letkf/mat/ens" + std::to_string(mpi.rank())
            : "";
        ::mkdir(mat_save_prefix.c_str(), 0755);
        solver.solve_xa(0, mat_save_prefix, t);
    }

    #ifndef LETKF_NO_MPI_BARRIER
    timer.transit("mpi_barrier");
    mpi.barrier();
    #endif
    timer.transit("mpi_xsol");
    solver.mpi_alltoall_Xsol(mpi.comm());

    // update f by LETKF sol
    timer.transit("update_xk");
    auto* fn = dat.dn().f.data();
    solver.update_xk(
        0, // id of ens (dummy)
        0, // i_batch_0
        dim3(config::ny, config::Q), config::nx,
        [=] __device__ (const float* Xk, size_t stride) {
            const auto i = threadIdx.x + blockDim.x * blockIdx.x;
            const auto q = blockIdx.y;
            const auto idx_lbm = config::ijq(i, q);
            const auto idx_xk = q + n_stt*i;
            fn[idx_lbm] = Xk[idx_xk];
        }
    );
    DEBUG_PRINT();

    // update macro by fn
    timer.transit("lbm2euler");
    auto* rn = dat.dn().r.data();
    auto* un = dat.dn().u.data();
    auto* vn = dat.dn().v.data();
    util::invoke_device<<<config::nx, config::ny>>>(
        [=] __device__ () {
            const auto i = threadIdx.x + blockDim.x * blockIdx.x;
            // lbm
            real fc[config::Q];
            #pragma unroll
            for(int q=0; q<config::Q; q++) {
                fc[q] = fn[config::ijq(i, q)];
            }
            const real rnc = rf(fc);
            const real unc = uf(fc, rnc);
            const real vnc = vf(fc, rnc);
            rn[i] = rnc;
            un[i] = unc;
        }
    );
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    dat.swap();

    timer.stop_and_ignore_latter();
}

#endif // ifdef DA_LETKF
