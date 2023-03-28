#ifndef LBM_COLLISION_MRT_H__
#define LBM_COLLISION_MRT_H__

#include "util/cuda_hostdevice.hpp"
#include "config.h"
#include "lbm.h"

/// ordinate of q:
/// inline __host__ __device__ constexpr int  qe(int ex, int ey) { return ex+1 + 3*(ey+1); }
/// inline __host__ __device__ constexpr int  ex(int q) { return (q%3)-1; }
/// inline __host__ __device__ constexpr int  ey(int q) { return (q/3)-1; }

inline __host__ __device__
void lbm_collision_mrt(real (&fc)[config::Q], real omega, real rho, real u, real v, int verbose=0) {
    constexpr int Q  = config::Q;
    constexpr int Qm = config::Q;
    static constexpr real M[Qm][Q] = {
        {  1,  1,  1,  1,  1,  1,  1,  1,  1 }, // rho
        {  2, -1,  2, -1, -4, -1,  2, -1,  2 }, // e
        {  1, -2,  1, -2,  4, -2,  1, -2,  1 }, // ep
        { -1,  0,  1, -1,  0,  1, -1,  0,  1 }, // jx
        { -1,  0,  1,  2,  0, -2, -1,  0,  1 }, // qx
        { -1, -1, -1,  0,  0,  0,  1,  1,  1 }, // jy
        { -1,  2, -1,  0,  0,  0,  1, -2,  1 }, // qy
        {  0, -1,  0,  1,  0,  1,  0, -1,  0 }, // pxx
        {  1,  0, -1,  0,  0,  0, -1,  0,  1 }  // pxy
    };
    static constexpr real mt[Qm] = {
        1./9., 1./36., 1./36., 1./6., 1./12., 1./6., 1./12., 1./4., 1./4.
    }; /// where, M_inverse = diag(mt) . M_transpose
    //const real s[Qm] = { 0, 1.4, 1.4, 0, 1.2, 0, 1.2, omega, omega }; /// optim hyperviscousity
    //const real s[Qm] = { 0, 1, 1, 0, 1, 0, 1, omega, omega }; /// stable
    //const real s[Qm] = { omega, omega, omega, omega, omega, omega, omega, omega, omega }; /// degrading into SRT
    const real s[Qm] = { 2, 2, 2, 2, 2, 2, 2, omega, omega }; /// [debug] unstable??
    //const real s[Qm] = { 0, 0, 0, 0, 0, 0, 0, omega, omega }; /// [debug] unstable??
    //const real om = 1 / (1/omega + 0.001);
    //const real s[Qm] = { om, om, om, om, om, om, om, omega, omega }; /// [debug]

    real fneq[Q] = {NAN,};
    for(int q=0; q<Q; q++) {
        fneq[q] = fc[q] - feq(rho, u, v, q);
    }

    for(int q=0; q<Q; q++) {
        for(int qm=0; qm<Qm; qm++) {
            fc[q] -= mt[qm]*M[qm][q] * s[qm] * M[qm][q] * fneq[q];
        }
    }
}

#endif
