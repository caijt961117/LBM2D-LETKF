#include <cstdlib>
#include <iostream>

#include "lbm.h"
#include "lbm_collision_srt.h"
#include "lbm_collision_mrt.h"
#include "data.h"
#include "config.h"
#include "util/port.hpp"

namespace port = util::port;
#ifdef PORT_CUDA
using backend = port::cuda;
#define PORT_CONTINUE return
#else
using backend = port::openmp;
#define PORT_CONTINUE continue
#endif

void prepare_force(data& dat, int) {
    dat.d0().update_forces(-1);
}

void les(data& dat, int t) {
    #ifdef LES
    const real* fn = dat.dn().f.data();
    const real* rn = dat.dn().r.data();
    const real* un = dat.dn().u.data();
    const real* vn = dat.dn().v.data();
    real* nus = dat.d().nus.data();
    port::pfor3d<backend>(
        port::thread3d(config::nx, config::ny, 1),
        [=] __host__ __device__ (port::thread3d, int i, int j, int) {
            if(config::is_boundary(i, j)) { PORT_CONTINUE; }
            const auto ij = config::ij(i, j);

            const auto ije = config::ij_periodic(i+1, j);
            const auto ijw = config::ij_periodic(i-1, j);
            const auto ijn = config::ij_periodic(i, j+1);
            const auto ijs = config::ij_periodic(i, j-1);

            const auto dx = config::dx;

            const auto ux = (un[ije] - un[ijw]) / (2*dx);
            const auto uy = (un[ijn] - un[ijs]) / (2*dx);
            const auto vx = (vn[ije] - vn[ijw]) / (2*dx);
            const auto vy = (vn[ijn] - vn[ijs]) / (2*dx);
            const auto s00 = ux;
            const auto s01 = real(.5) * (uy + vx);
            const auto s10 = s01;
            const auto s11 = vy;
            const auto ss = sqrt(2 * (s00*s00 + s01*s10 + s10*s01 + s11*s11));
            #ifdef LES_CSM
            const auto qq = -real(.5) * (ux*ux + uy*vx + vx*uy + vy*vy);
            const auto ee = real(.5) * (ux*ux + uy*uy + vx*vx + vy*vy);
            const auto fcs = qq/ee;
            nus[ij] = 0.05 * pow(abs(fcs), real(1.5)) * dx*dx * ss;
            #else
            const auto cs = 0.2; /// 0.1 -- 0.2 ??
            nus[ij] = cs*cs * dx*dx * ss;
            #endif
        } // [=]__host__ __device__()
    ); // port::pfor3d()
    #endif
}

void lbm_streaming_macro(data& dat, int t) {
    const real* f = dat.d().f.data();
    const real* force_x = dat.d0().force_x.data();
    const real* force_y = dat.d0().force_y.data();
    real* fn = dat.dn().f.data();
    real* rn = dat.dn().r.data();
    real* un = dat.dn().u.data();
    real* vn = dat.dn().v.data();

    port::pfor3d<backend>(
        port::thread3d(config::nx, config::ny, 1),
        [=] __host__ __device__ (port::thread3d, int i, int j, int) {
            const int ij = config::ij(i, j);
            if(config::is_boundary(i, j)) { PORT_CONTINUE; }
            real fc[config::Q];
            #pragma unroll
            for(int q=0; q<config::Q; q++) {
                int iu = (i - config::ex(q) + config::nx) % config::nx;
                int ju = (j - config::ey(q) + config::ny) % config::ny;
                fc[q] = f[config::ijq(iu, ju, q)];
            }
            #pragma unroll
            for(int q=0; q<config::Q; q++) { fn[config::ijq(i, j, q)] = fc[q]; }

            /// macro
            const real rnc = rf(fc);
            const real fxc = force_x[ij];
            const real fyc = force_y[ij];
            const real unc = uf(fc, rnc, fxc);
            const real vnc = vf(fc, rnc, fyc);
            rn[ij] = rnc;
            un[ij] = unc;
            vn[ij] = vnc;
        } // [=]__host__ __device__()
    ); // port::pfor3d()

}

void lbm_collision_force(data& dat, int t) {
    #ifdef LES
    const real* nus = dat.d().nus.data();
    #endif
    const real* force_x = dat.d0().force_x.data();
    const real* force_y = dat.d0().force_y.data();

    real* fn = dat.dn().f.data();
    const real* rn = dat.dn().r.data();
    const real* un = dat.dn().u.data();
    const real* vn = dat.dn().v.data();

    port::pfor3d<backend>(
        port::thread3d(config::nx, config::ny, 1),
        [=] __host__ __device__ (port::thread3d, int i, int j, int) {
            const int ij = config::ij(i, j);
            if(config::is_boundary(i, j)) { PORT_CONTINUE; }
            real fc[config::Q];
            #pragma unroll
            for(int q=0; q<config::Q; q++) {
                fc[q] = fn[config::ijq(i, j, q)];
            }
            const real rnc = rn[ij];
            const real unc = un[ij];
            const real vnc = vn[ij];

            /// visc
            #ifdef LES
            const auto nu_sgs = nus[ij];
            const real nu = config::nu + nu_sgs;
            const real c = config::c;
            const real dt = config::dt;
            const real tau = 0.5 + 3. * nu / c/c / dt;
            const real omega = 1./tau;
            #else
            const real omega = config::omega;
            #endif

            /// collision
            lbm_collision_srt(fc, omega, rnc, unc, vnc);

            /// forcing
            const real fxc = force_x[ij];
            const real fyc = force_y[ij];
            #pragma unroll
            for(int q=0; q<config::Q; q++) {
                fc[q] += force_acc(fxc, fyc, q, rnc, unc, vnc, omega); // XXX
            }

            /// final write
            #pragma unroll
            for(int q=0; q<config::Q; q++) { fn[config::ijq(i, j, q)] = fc[q]; }
        } // [=]__host__ __device__()
    ); // port::pfor3d()

}
