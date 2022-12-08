#ifndef LBM_COLLISION_SRT_H__
#define LBM_COLLISION_SRT_H__

#include "util/cuda_hostdevice.hpp"
#include "config.h"
#include "lbm.h"

inline __host__ __device__
void lbm_collision_srt(real (&fc)[config::Q], real omega, real rho, real u, real v) {
    #pragma unroll
    for(int q=0; q<config::Q; q++) {
        const real feqq = feq(rho, u, v, q);
        const real fneq = fc[q] - feqq;
        fc[q] -= omega*fneq;
    }
}

#endif
