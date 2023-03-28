#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>

#include <curand.h>

#include "data.h"
#include "config.h"
#include "lbm.h"
#include "util/range.hpp"
#include "util/mpi_wrapper.hpp"
#include "util/port.hpp"
#include "util/cuda_safe_call.hpp"
#include "util/invoker.hpp"

namespace port = util::port;
#ifdef PORT_CUDA
using backend = port::cuda;
#else
using backend = port::openmp;
#endif

void stdata::init_lbm(int ensemble_id_orig) {
    // const
    constexpr auto nx = config::nx;
    constexpr auto ny = config::ny;
    constexpr auto Q = config::Q;

    // data assimilation: different random seed from nature run (we should not know the nature)
    int ensemble_id = ensemble_id_orig;
    #ifdef DATA_ASSIMILATION
    ensemble_id += 334;
    #endif

    // random for initial velocity
    if(config::verbose >= 0) {
        std::cout << "ensemble " << ensemble_id_orig << ": initial velocity with random_seed = " << ensemble_id << std::endl;
    }
    auto rand_engine = std::mt19937(ensemble_id);
    auto rand_dist = std::uniform_real_distribution<real>(-1, 1);
    auto rand = [&]() { return rand_dist(rand_engine); };

    // alloc
    f.resize(nx * ny * Q);
    r.resize(nx * ny);
    u.resize(nx * ny);
    v.resize(nx * ny);
    nus.resize(nx * ny);

    // tmp val for stream function
    constexpr real p_amp = 0.01;
    constexpr int nk = nx/2;
    constexpr real k0 = config::kf;
    constexpr real dk = 10;
    constexpr real sigma = 5;
    util::cu_vector<real> p(nk*nk), theta(nk*nk);
    for(const auto kj: util::range(nk)) {
        for(const auto ki: util::range(nk)) {
            const auto kij = ki + nk * kj;
            const auto k = std::sqrt(real(ki*ki + kj*kj));
            //const auto pk = k*k*std:0.1:exp(-k*k/2.); /// non-forced
            const auto pk = (k0-dk <= k and k <= k0+dk)
                ? std::exp( - (k-k0)*(k-k0) / sigma )
                : 0; /// forced maltrud91
            //const auto pk = ((ki == k0 or kj == k0) and ki <= k0 and kj <= k0)
            //    ? 1
            //    : 0; /// forced lilly69
            p.at(kij) = pk;
            theta.at(kij) = rand() * M_PI;
        }
    }

    // ini. val.
    port::pfor3d<backend>(
        port::thread3d(config::nx, config::ny, 1),
        [] __host__ __device__ (port::thread3d, int i, int j, int,
            const int nx,
            const int ny,
            const int nki,
            const int nkj,
            const real* p,
            const real* theta,
            real* r,
            real* u,
            real* v
        ) {
            const auto ij = config::ij(i, j);

            // fluid
            const real r0 = config::rho_ref;
            real utmp = 0;
            real vtmp = 0;
            for(int kj=0; kj<nkj; kj++) {
                const auto ky = 2 * M_PI * kj / ny;
                for(int ki=0; ki<nki; ki++) {
                    const auto kx = 2 * M_PI * ki / nx;
                    const auto kij = ki + nki*kj;
                    utmp += p[kij] *  ky * cos(kx * i + ky * j + theta[kij]);
                    vtmp += p[kij] * -kx * cos(kx * i + ky * j + theta[kij]);
                }
            }
            r[ij] = r0;
            u[ij] = utmp;
            v[ij] = vtmp;

        }, // __host__ __device__
        nx, ny, nk, nk, p.data(), theta.data(), r.data(), u.data(), v.data()
    );

    /// modify u, v by max. vs cfl
    real vmax = 1e-30 * config::u_ref;
    for(const auto ij: util::range(nx*ny)) {
        vmax = std::max(vmax, std::sqrt(u.at(ij)*u.at(ij) + v.at(ij)*v.at(ij)));
    }
    for(const auto ij: util::range(nx*ny)) {
        u.at(ij) *= config::u_ref / vmax * p_amp;
        v.at(ij) *= config::u_ref / vmax * p_amp;
    }

    #if 0 // if no initital pertubation
    // ini. val.
    for(const auto ij: util::range(nx*ny)) {
        r[ij] = config::rho_ref;
        u[ij] = 0;
        v[ij] = 0;
    }
    #endif


    // finally f
    for(const auto j: util::range(ny)) {
        for(const auto i: util::range(nx)) {
            const auto ij = config::ij(i, j);
            for(const auto q: util::range(Q) ){
                f.at(config::ijq(i, j, q)) = feq(r.at(ij), u.at(ij), v.at(ij), q);
            }
        }
    }
}

void stdata::init_force(int ensemble_id) {
    // const
    constexpr auto nx = config::nx;
    constexpr auto ny = config::ny;

    // alloc
    force_x.resize(nx * ny);
    force_y.resize(nx * ny);
    for(auto&& ij: util::range(nx * ny)) {
        force_x.at(ij) = 0;
        force_y.at(ij) = 0;
    }

    init_forces(ensemble_id);

    update_forces(ensemble_id);
}

void stdata::init_forces(int ensemble_id) {
    force_kx.clear();
    force_ky.clear();
    force_amp.clear();

    // initilize spectrum of force
    constexpr int max_search_nmode = 10000;
    constexpr real kf = config::kf;
    constexpr real amp_scale = config::fkf;
    constexpr real sigma = 1; // gaussian stdev
    constexpr real dk = 2; // clipping width
    constexpr int kmax_search = int(kf+dk);
    long double asum = 0;
    for(int ky=0; ky<kmax_search; ky++) { for(int kx=0; kx<kmax_search; kx++) {
        if(kx == 0 and ky == 0) { continue; }
        const auto k = std::sqrt(ky*ky + kx*kx);
        if(kf-dk <= k and k <= kf+dk) {  /// maltrud91
        //if(((kf-dk <= kx and kx <= kf+dk) or (kf-dk <= ky and ky <= kf+dk)) and kx <= kf+dk and ky <= kf+dk) { /// lilly69
        //if ((kx == kf and ky == 0) or (ky == kf and kx == 0)) { /// few modes
            const real amp = std::exp( - (k-kf)*(k-kf) / sigma ); /// Gaussian decay by |k-kf|
            runtime_assert(amp == amp, "InternalError: amplitude is nan");
            asum += amp;
            force_kx.push_back(kx);
            force_ky.push_back(ky);
            force_amp.push_back(amp);
            //if(ensemble_id == 0) { std::cout << "forces wavenumber: " << k << " (" << kx << ", " << ky << ")" << std::endl; }
        }
        if(force_kx.size() >= max_search_nmode) { break; }
    }}

    // prune modes of force
    constexpr size_t max_nmode = 100;
    if(force_kx.size() > max_nmode) {
        auto tmp_kx = force_kx;
        auto tmp_ky = force_ky;
        auto tmp_am = force_amp;
        std::vector<int> ps(max_nmode);
        std::mt19937 engine(force_kx.size());
        for(auto& p: ps) {
            std::uniform_int_distribution<int> dist(0, max_nmode);
            p = dist(engine);
        }

        force_kx.clear();
        force_ky.clear();
        force_amp.clear();
        for(auto p: ps) {
            force_kx.push_back(tmp_kx.at(p));
            force_ky.push_back(tmp_ky.at(p));
            force_amp.push_back(tmp_am.at(p));
        }
    }

    if(config::verbose >= 1 && ensemble_id == 0) {
        std::cout << "force: " << "n=" << int(force_kx.size()) << ", sum=" << asum << std::endl;
    }

    constexpr real config_t0__ = 1.;
    constexpr real force_total_amp = amp_scale * config::u_ref / config_t0__;
    for(real& amp: force_amp) {
        amp *= force_total_amp / asum;
        runtime_assert(amp == amp, "InternalError: amplitude is nan");
    }

    // initialize random for force_theta
    /// initialize random seed
    CURAND_SAFE_CALL(( curandCreateGenerator(&randgen, CURAND_RNG_PSEUDO_MTGP32) ));
    CURAND_SAFE_CALL(( curandSetPseudoRandomGeneratorSeed(randgen, 1991ul) ));

    /// malloc rand buffer
    n_rand_buf = force_amp.size() * 4096;
    i_rand_outdated = n_rand_buf;
    CUDA_SAFE_CALL(( cudaMalloc(&force_theta_d, n_rand_buf * sizeof(float)) ));

    // first
    update_forces(ensemble_id);
}

void stdata::update_forces_watanabe97(int /*ensemble_id*/) {
    // update ramd buf
    i_rand_outdated += force_amp.size() * 4;
    if( i_rand_outdated + force_amp.size() * 4 >= n_rand_buf ) {
        constexpr float force_mean = 0, force_stdev = 0.84089641525;
        CURAND_SAFE_CALL(( curandGenerateNormal(randgen, force_theta_d, n_rand_buf, force_mean, force_stdev) ));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        i_rand_outdated = 0;
    }

    //
    constexpr auto nx = config::nx, ny = config::ny;

    const auto* force_rand = &force_theta_d[i_rand_outdated];
    const auto* force_kx = this->force_kx.data();
    const auto* force_ky = this->force_ky.data();
    const auto* force_amp = this->force_amp.data();
    const auto force_n = this->force_amp.size();
    auto* force_x = this->force_x.data();
    auto* force_y = this->force_y.data();


//#define USE_SHARED
//#ifdef USE_SHARED
//    const int shared_size = sizeof(real)*7*force_n;
//    constexpr int nthx = std::min(256, nx);
//    constexpr int nbx = nx/nthx, nby = ny;
//    static_assert(nx % nthx == 0);
//
//    util::invoke_device<<<dim3(nbx, nby), nthx, shared_size>>>([=]__device__() {
//        const int i = threadIdx.x + blockDim.x * blockIdx.x;
//        const int j = threadIdx.y + blockDim.y * blockIdx.x;
//
//        extern __shared__ real s_force[];
//        { // load shared
//            const int kloop = (force_n + blockDim.x-1)/blockDim.x;
//            for(int nn=0; nn<kloop; nn++) {
//                const int n = kloop*nn + threadIdx.x;
//                if(n >= force_n) { continue; }
//                s_force[n] = force_amp[n];
//                s_force[n + 1*force_n] = force_kx[n];
//                s_force[n + 2*force_n] = force_ky[n];
//                s_force[n + 3*force_n] = force_rand[n + 0*force_n];
//                s_force[n + 4*force_n] = force_rand[n + 1*force_n];
//                s_force[n + 5*force_n] = force_rand[n + 2*force_n];
//                s_force[n + 6*force_n] = force_rand[n + 3*force_n];
//            }
//            __syncthreads();
//        }
//
//        const real x = (i - nx/2) / (real)nx;
//        const real y = (j - ny/2) / (real)ny;
//        real f_x = 0;
//        real f_y = 0;
//        for(int n=0; n<force_n; n++) {
//            const real kx = s_force[n + 1*force_n];
//            const real ky = s_force[n + 2*force_n];
//            const real theta = 2*M_PI*(kx*x + ky*y);
//            const real sine = sin(theta);
//            const real cosi = cos(theta);
//            const real r0 = s_force[n + 3*force_n];
//            const real r1 = s_force[n + 4*force_n];
//            const real r2 = s_force[n + 5*force_n];
//            const real r3 = s_force[n + 6*force_n];
//            const real amp = s_force[n];
//            const real fek = (r0*kx+r1*ky)*sine + (r2*kx+r3*ky)*cosi;
//            f_x += amp*2*M_PI*ky    * fek;
//            f_y += amp*2*M_PI*(-kx) * fek;
//        }
//        const auto ij = i + nx * j;
//        constexpr real R = 0.;
//        const real fx = force_x[ij], fxn = f_x;
//        const real fy = force_y[ij], fyn = f_y;
//        force_x[ij] = R * fx + sqrt(1-R*R) * fxn;
//        force_y[ij] = R * fy + sqrt(1-R*R) * fyn;
//    });
//
//#else
    port::pfor3d<backend>(
        port::thread3d(nx, ny, 1),
        [=] __host__ __device__ (port::thread3d, int i, int j, int) {
            const real x = (i - nx/2) / (real)nx;
            const real y = (j - ny/2) / (real)ny;
            real f_x = 0;
            real f_y = 0;
            for(int n=0; n<force_n; n++) {
                const real kx = force_kx[n];
                const real ky = force_ky[n];
                const real theta = 2*M_PI*(kx*x + ky*y);
                const real sine = sin(theta);
                const real cosi = cos(theta);
                const real r[4] = {
                    force_rand[n + 0*force_n],
                    force_rand[n + 1*force_n],
                    force_rand[n + 2*force_n],
                    force_rand[n + 3*force_n]
                    };
                const real amp = force_amp[n];
                f_x += amp*2*M_PI*ky    * ( (r[0]*kx+r[1]*ky)*sine + (r[2]*kx+r[3]*ky)*cosi );
                f_y += amp*2*M_PI*(-kx) * ( (r[0]*kx+r[1]*ky)*sine + (r[2]*kx+r[3]*ky)*cosi );
            }

            const auto ij = i + nx * j;
            constexpr real R = 0.;
            const real fx = force_x[ij], fxn = f_x;
            const real fy = force_y[ij], fyn = f_y;
            force_x[ij] = R * fx + sqrt(1-R*R) * fxn;
            force_y[ij] = R * fy + sqrt(1-R*R) * fyn;
        }
    );
//#endif
}

void stdata::update_forces_maltrud91(int /*ensemble_id*/) {
    // update ramd buf
    i_rand_outdated += force_amp.size() * 2;
    if( i_rand_outdated + force_amp.size() * 2 >= n_rand_buf ) {
        CURAND_SAFE_CALL(( curandGenerateUniform(randgen, force_theta_d, n_rand_buf) ));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        i_rand_outdated = 0;
    }

    //
    constexpr auto nx = config::nx, ny = config::ny;

    const auto* force_theta = &force_theta_d[i_rand_outdated];
    const auto* force_kx = this->force_kx.data();
    const auto* force_ky = this->force_ky.data();
    const auto* force_amp = this->force_amp.data();
    const auto force_n = this->force_amp.size();
    auto* force_x = this->force_x.data();
    auto* force_y = this->force_y.data();

    port::pfor3d<backend>(
        port::thread3d(nx, ny, 1),
        [=] __host__ __device__ (port::thread3d, int i, int j, int) {
            const real x = (i - nx/2) / (real)nx;
            const real y = (j - ny/2) / (real)ny;
            real f_x = 0;
            real f_y = 0;
            for(int n=0; n<force_n; n++) {
                const real kx = force_kx[n];
                const real ky = force_ky[n];
                const real theta = real(2.*M_PI) * (force_theta[n] - 0.5f);
                const real coskxy = cos( real(2.*M_PI) *(kx*x + ky*y) + theta);
                const real amp = force_amp[n];
                const real ax = force_theta[n + force_n] + 1, ay = 2-ax;
                //f_x += ky < 0.5 ? 0 : ax * -amp/ky * coskxy;
                //f_y += kx < 0.5 ? 0 : ay *  amp/kx * coskxy;
                f_x += ax * amp * coskxy;
                f_y += ay * amp * coskxy;
            }

            const auto ij = i + nx * j;
            constexpr real R = 0;
            const real fx = force_x[ij], fxn = f_x;
            const real fy = force_y[ij], fyn = f_y;
            force_x[ij] = R * fx + sqrt(1-R*R) * fxn;
            force_y[ij] = R * fy + sqrt(1-R*R) * fyn;
        }
    );
}

void stdata::update_forces_xia14(int /*ensemble_id*/) {

    //
    constexpr auto nx = config::nx, ny = config::ny;

    auto* force_x = this->force_x.data();
    auto* force_y = this->force_y.data();

    constexpr real lf = 1./16;
    constexpr real f0 = 3;

    port::pfor3d<backend>(
        port::thread3d(nx, ny, 1),
        [=] __host__ __device__ (port::thread3d, int i, int j, int) {
            const int ij = i + nx*j;
            const real x = i / (real)nx - 0.5;
            const real y = j / (real)ny - 0.5;
            const real xx = (x*x + y*y);
            force_x[ij] = f0*lf*exp(-(xx/lf/lf)/2);
            force_y[ij] = 0;
        }
    );
}

void stdata::purturbulate(int ensemble_id) {
    #ifdef LYAPNOV
    // const
    constexpr auto nx = config::nx;
    constexpr auto ny = config::ny;
    constexpr auto Q = config::Q;
    constexpr auto epsilon = LY_EPSILON;
    ensemble_id += 132333;
    if(config::verbose >= 1 && ensemble_id == 0) {
        std::cout << "purturbulation: random_seed = " << ensemble_id << std::endl;
    }
    auto rand_engine = std::mt19937(ensemble_id);
    auto rand_dist = std::normal_distribution<real>(0, 1);
    auto randn = [&]() { return rand_dist(rand_engine); };

    // f with purturbulation
    //     purturbulation using the context of newtonian nudging
    for(const auto j: util::range(ny)) {
        for(const auto i: util::range(nx)) {
            const auto ij = config::ij(i, j);
            const auto urand = config::u_ref * randn();
            const auto vrand = config::u_ref * randn();
            for(const auto q: util::range(Q) ){
                const auto ff = f.at(config::ijq(i, j, q));
                const auto fo = feq(r.at(config::ij(i, j)), urand, vrand, q);
                f.at(config::ijq(i, j, q)) = epsilon*fo + (1-epsilon)*ff;
            }
        }
    }
    #endif
}

void stdata::inspect() const {
    if(config::verbose >= 100) {
        std::cout << "stdata:: " << std::endl;
        std::cout << "  u: " << *std::min_element(u.begin(), u.end())  << " " << *std::max_element(u.begin(), u.end()) << std::endl;
        std::cout << "  v: " << *std::min_element(v.begin(), v.end())  << " " << *std::max_element(v.begin(), v.end()) << std::endl;
        std::cout << "  r: " << *std::min_element(r.begin(), r.end())  << " " << *std::max_element(r.begin(), r.end()) << std::endl;
        std::cout << "  nus: " << *std::min_element(nus.begin(), nus.end())  << " " << *std::max_element(nus.begin(), nus.end()) << std::endl;
    }

    if(config::verbose >= 10) {
        std::cout << std::scientific << std::setprecision(16) << std::flush;
        real momentum_x_total = 0, momentum_y_total = 0;
        real energy = 0;
        real enstrophy = 0;
        real nus_total = 0;
        real mass = 0;
        real divu2 = 0;
        real divu = 0;
        real maxdivu = 0;
        real vel2 = 0;
        real maxvel2 = 0;
        for(int j=0; j<config::ny; j++) {
            for(int i=0; i<config::nx; i++) {
                const auto ij = config::ij(i, j);
                if(!config::is_boundary(i, j)) {
                    momentum_x_total += r.at(ij) * u.at(ij);
                    momentum_y_total += r.at(ij) * v.at(ij);
                    energy += (u.at(ij) * u.at(ij) + v.at(ij) * v.at(ij));
                    nus_total += nus.at(ij);
                    mass += r.at(ij);

                    // derivatives
                    const int iw = (i-1+config::nx)%config::nx;
                    const int ie = (i+1)%config::nx;
                    const int js = (j-1+config::ny)%config::ny;
                    const int jn = (j+1)%config::ny;
                    const real uc = u.at(config::ij(i, j));
                    const real vc = v.at(config::ij(i, j));
                    const real ue = u.at(config::ij(ie, j));
                    const real uw = u.at(config::ij(iw, j));
                    const real us = u.at(config::ij(i, js));
                    const real un = u.at(config::ij(i, jn));
                    const real ux = (ue - uw) / (2*config::dx);
                    const real uy = (un - us) / (2*config::dx);
                    const real vw = v.at(config::ij(iw, j));
                    const real ve = v.at(config::ij(ie, j));
                    const real vs = v.at(config::ij(i, js));
                    const real vn = v.at(config::ij(i, jn));
                    const real vx = (ve - vw) / (2*config::dx);
                    const real vy = (vn - vs) / (2*config::dx);
                    enstrophy += (vx - uy) * (vx - uy) / 2;
                    enstrophy += (vx - uy) * (vx - uy) / 2;
                    divu2 += (ux + vy) * (ux + vy);
                    divu += (ux + vy);
                    maxdivu = std::max(maxdivu, std::abs(ux + vy));
                    vel2 += uc*uc + vc*vc;
                    maxvel2 = std::max(maxvel2, uc*uc + vc*vc);
                }
            }
        }
        momentum_x_total /= (config::nx * config::ny);
        momentum_y_total /= (config::nx * config::ny);
        energy           /= (config::nx * config::ny);
        enstrophy        /= (config::nx * config::ny);
        nus_total        /= (config::nx * config::ny);
        mass             /= (config::nx * config::ny);
        divu2            /= (config::nx * config::ny);
        divu             /= (config::nx * config::ny);
        vel2             /= (config::nx * config::ny);

        //std::cout << " mean mx: " << momentum_x_total << " [kg m/s]" << std::endl;
        //std::cout << " mean my: " << momentum_y_total << " [kg m/s]" << std::endl;
        std::cout << " RMS, max speed: " << std::sqrt(vel2) << ", " << std::sqrt(maxvel2) << " [m/s]" << std::endl;
        //std::cout << " mean energy: " << energy << " [m2/s2]" << std::endl;
        //std::cout << " mean enstrophy: " << enstrophy << " [/s2]" << std::endl;
        std::cout << " mean les visc: " << nus_total << " [m2/s]" << std::endl;
        std::cout << " mean mass: " << mass << " [kg]" << std::endl;
        std::cout << " delta_rho: " << *std::max_element(r.begin(), r.end()) - *std::min_element(r.begin(), r.end()) << " [kg/m3]" << std::endl;
        std::cout << " max vel divergence (times dx, devided by uref): " << maxdivu * config::dx / config::u_ref << " []" << std::endl;


        std::cout << std::resetiosflags(std::ios_base::floatfield);
        std::cout << std::endl;
    }
}
