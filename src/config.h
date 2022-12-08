#ifndef CONFIG_H_
#define CONFIG_H_

#define PORT_CUDA

#define LES // numerical viscosity by les
//#define ICLBM // feq with imcompressible approx.

#define EXIT_IF_NAN
#ifdef DA_LETKF
//#define NO_OUTPUT
#define DROP_FORMER_HALF_TIMER
//#define LETKF_NO_MPI_BARRIER
#endif


// test mode
//#define OBSERVE
//#define LYAPNOV

//#define DATA_ASSIMILATION
//
//

#if defined(LYAPNOV_OBSERVE) && !defined(OBSERVE)
#define OBSERVE
#endif

#if defined(DA_LETKF) || defined(DA_NUDGING)
#ifndef DATA_ASSIMILATION
#define DATA_ASSIMILATION
#endif
#endif

#ifdef DATA_ASSIMILATION
#if !defined(DA_LETKF) && !defined(DA_NUDGING)
#error select DA_LETKF or DA_NUDGING
#endif

//#define DA_OBS_UV
#define DA_OBS_RUV

#ifdef DA_LETKF
#define ENSEMBLE_STAT
#ifndef LETKF_COVINF
#define LETKF_COVINF 1
#endif
#endif

#endif // DATA_ASSIMILATION

#ifndef DAPRUNE
#define DAPRUNE 10 // default: per 100 steps at nx128-noles, whose error doubling time ~= 900 steps
#endif

#ifndef DA_XYPRUNE
#define DA_XYPRUNE 1
#endif

#ifndef DA_QUADRA
#define DA_QUADRA 0
#endif

//#define DA_DUMMY // no DA; just for debug
#ifdef DA_DUMMY
#define DATA_ASSIMILATION
#endif

//
#define SPINUP

// 
#define FORCE_TURB

// observation error via lbm or uv
#define OBSERVE_ERROR_RUV // in RUV mode, observation of f[q] is unavailable
#define OBS_XYPRUNE_NAN // set nan on pruned data

// log level
#define VERBOSE 100

// mesh resol
#define SCALER 2 // by which nx = 64 times
#define NX (64 * SCALER)
#ifndef IOPRUNE
#define IOPRUNE 10
#endif

// check defines
#if defined(LY_EPSILON) && !defined(LYAPNOV)
    #define LYAPNOV
#elif defined(LYAPNOV) && !defined(LY_EPSILON)
    #define LY_EPSILON (1e-8)
#endif

#if ((defined(OBSERVE) && defined(DATA_ASSIMILATION))) || \
(defined(DATA_ASSIMILATION) && defined(LYAPNOV))
    #error defining more than two of OBSERVE/DATA_ASSIMILATION/LYAPNOV are contradicted
#endif

#if (!defined(OBSERVE) && !defined(LYAPNOV) && !defined(DATA_ASSIMILATION))
    #error one of OBSERVE/DATA_ASSIMILATION/LYAPNOV should be defined to determine run-mode
#endif

// precision of floating point vars
//#define DOUBLE_PRECISION
#ifdef DOUBLE_PRECISION
    using real = double;
    #define MPI_REAL_T MPI_DOUBLE
#else
    using real = float;
    #define MPI_REAL_T MPI_FLOAT
#endif


#include <iostream>
#include <string>

#include "util/cuda_hostdevice.hpp"

namespace config {

constexpr auto verbose = VERBOSE; // verbose level

// god
//constexpr real Re = infty;
constexpr int ny = NX;
constexpr int nx = ny;
constexpr int scaler = SCALER;
constexpr int iiter = 5 * scaler; // output interval
#ifdef SPINUP
constexpr int spinup = 100000 * iiter; // cold start duration
#endif
constexpr int ioprune = IOPRUNE; // sosika
constexpr int daprune = DAPRUNE; // sosika

constexpr int iter_main = iiter * 2000;
#if defined(LYAPNOV) || defined(LYAPNOV_OBSERVE)
constexpr int iter_total = iter_main * 5; // test longer
#elif defined(DA_DUMMY) || defined(OBSERVE)
constexpr int iter_total = iter_main + 1; // da buffer
#else
constexpr int iter_total = iter_main; // final
#endif

constexpr real cfl = 0.05;

// phys
constexpr real rho_ref = 1;
constexpr real nu = 1e-7;;//2.5e-4; //1/Re;
constexpr real h_ref = 1; // reference length
constexpr real kf = 8; // injected wavenumber

constexpr real u_ref = 1;

// data assimilation
constexpr real obs_error = 0.08;
constexpr real da_nud_rate = std::min(real(0.99), real(0.01 * DAPRUNE)); // by EDT, alpha = DA_interval / EDT; by LETKF, alpha = sigma_x^2 / (sigma_x^2 + sigma_y^2)
constexpr int da_xyprune = DA_XYPRUNE;
constexpr int da_quadra = DA_QUADRA;

// cfd
inline __host__ __device__ constexpr int ij(int i, int j) { return i + nx*j; }
inline __host__ __device__ constexpr int ij_periodic(int i, int j) { return ij((i+nx)%nx, (j+ny)%ny); }
inline __host__ __device__ constexpr int ijq(int i, int j, int q) { return ij(i,j) + nx*ny*q; }
inline __host__ __device__ constexpr int ijq(int ij, int q) { return ij + nx*ny*q; }
inline __host__ __device__ constexpr bool is_boundary(int i, int j) {
    //return (i == 0 || i == config::nx-1) || (j == 0 || j == config::ny-1);
    //return (j == 0 || j == config::ny-1);
    return false; // 2d burgers?
}


constexpr real c = u_ref / cfl; // lbm c_ref

constexpr real dx = h_ref / (ny-1);
constexpr real dt = dx / c;

constexpr real tau = 0.5 + 3. * nu / c/c / dt;
constexpr real omega = 1./tau;

// lbm d2q9
constexpr int Q = 9;
constexpr real w0 = 4./9., w1 = 1./9., w2 = 1./36.;
constexpr real friction_rate = 2.5e-4/dt; // artificial friction force for low-k energy dissipation. see [Xia and Qian @ Phys. Rev. E 2014]

#if 1 // default
inline __host__ __device__ constexpr int  qe(int ex, int ey) { return ex+1 + 3*(ey+1); }
inline __host__ __device__ constexpr int  ex(int q) { return (q%3)-1; }
inline __host__ __device__ constexpr int  ey(int q) { return (q/3)-1; }
#else
inline __host__ __device__ constexpr int  qe(int ex, int ey) { return -ex+1 + 3*(-ey+1); }
inline __host__ __device__ constexpr int  ex(int q) { return -((q%3)-1); }
inline __host__ __device__ constexpr int  ey(int q) { return -((q/3)-1); }
#endif

inline __host__ __device__ constexpr real wq(int q) { 
    int ee = ex(q) * ex(q) + ey(q) * ey(q);
    return ee == 0 ? w0 : ee == 1 ? w1 : w2;
}
inline __host__ __device__ constexpr int qr(int q) { return (Q-1-q); }

inline void inspect() {
    if(config::verbose >= 0) {
        std::cout << "config.h::" << std::endl
            //<< "  Re = " << Re << std::endl
            << "  nu = " << nu << " m2/s" << std::endl
            << "  u_ref = " << u_ref << " m/s" << std::endl
            << "  h_ref = " << h_ref << " meter" << std::endl
            << "  omega = " << omega << std::endl
            << "  dx = " << dx << " meter" << std::endl
            << "  dt = " << dt << " sec" << std::endl
            << "  iiter = " << iiter << std::endl
            << "  dt_io = " << dt*iiter << std::endl
            << "  c = " << c << " m/s" << std::endl
            << "  Area = " << dx*dx*nx*ny << std::endl
            ;
    }
}

// output 
const std::string prefix = "io";

} // namespace config

#endif

