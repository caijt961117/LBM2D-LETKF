#ifndef LBM_H_
#define LBM_H_

#include <iostream>
#include <cstdlib>

#include "config.h"
#include "data.h"
#include "util/cuda_hostdevice.hpp"

void prepare_force(data&, int);
void les(data&, int);
void lbm_streaming_macro(data&, int);
void lbm_collision_force(data&, int);

inline void update(data& dat, int t) {
    #ifdef FORCE_TURB
    prepare_force(dat, t);
    #endif

    lbm_streaming_macro(dat, t);
    #ifdef LES
    les(dat, t);
    #endif

    lbm_collision_force(dat, t);

    dat.swap();
}

inline __host__ __device__ real feq(real rho, real u, real v, int q) {
    real uc0 = (u * config::ex(q) + v * config::ey(q))/config::c;
    real uu0 = (u*u + v*v)/(config::c*config::c);
    #ifdef ICLBM
    return config::rho_ref * config::wq(q) * (rho/config::rho_ref + real(3.)*uc0 + real(4.5)*uc0*uc0 - real(1.5)*uu0);
    #else
    return rho * config::wq(q) * (real(1.) + real(3.)*uc0 + real(4.5)*uc0*uc0 - real(1.5)*uu0);
    #endif
}

inline __host__ __device__ real force_acc(
real fx_, real fy_,
int q,
real rho=config::rho_ref, real u=0, real v=0,
real omega=config::omega
) {
    const auto cx = config::u_ref * config::ex(q);
    const auto cy = config::u_ref * config::ey(q);
    const auto wq = config::wq(q);
    constexpr auto cs2 = config::c * config::c / 3;
    constexpr auto dt = config::dt;
    const auto fx = (fx_ - config::friction_rate/dt * u) * dt ;
    const auto fy = (fy_ - config::friction_rate/dt * v) * dt ;
    real f1 = (cx*fx + cy*fy) / cs2; // first-order term
    real ret = f1;
    #ifdef FORCE_SECOND_ORDER
    constexpr auto cs4 = cs2 * cs2;
    real f2 = real(1.) / cs4 * (
              fx*u * (cx*cx - cs2)
            + (fx*v + fy*u)*cx*cy
            + fy*v * (cy*cy - cs2)
            );
    ret += f2;
    #endif
    #ifdef VELOCITY_FORCE_CORRECTION
    ret *= (real(1.) - real(0.5)*omega);
    #endif
    ret *= wq * rho;
    return ret;
}

inline __host__ __device__ real rf(const real (&f)[config::Q]) {
    real ret = 0;
    #pragma unroll
    for(int q=0; q<config::Q; q++) { ret += f[q]; }
    return ret;
}

inline __host__ __device__ real uf(const real (&f)[config::Q], const real rho, const real force_x=0) {
    real ret = 0;
    #pragma unroll
    for(int q=0; q<config::Q; q++) { ret += f[q] * config::c * config::ex(q); }

    #ifdef VELOCITY_FORCE_CORRECTION
    ret += force_x * config::dt / 2;
    #endif

    #ifndef ICLBM
    ret /= rho;
    #endif

    return ret;
}

inline __host__ __device__ real vf(const real (&f)[config::Q], const real rho, const real force_y=0) {
    real ret = 0;
    #pragma unroll
    for(int q=0; q<config::Q; q++) { ret += f[q] * config::c * config::ey(q); }

    #ifdef VELOCITY_FORCE_CORRECTION
    ret += force_y * config::dt / 2;
    #endif

    #ifndef ICLBM
    ret /= rho;
    #endif

    return ret;
}

#endif
