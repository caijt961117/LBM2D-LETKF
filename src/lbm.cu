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
#else
using backend = port::openmp;
#endif

void prepare_force(data& dat, int) {
    dat.d0().update_forces(-1);
}

void les(data& dat, int t) {
    #ifdef LES
    const real* f = dat.d().f.data();
    const real* r = dat.d().r.data();
    const real* u = dat.d().u.data();
    const real* v = dat.d().v.data();
    real* nus = dat.d().nus.data();
    port::pfor3d<backend>(
        port::thread3d(config::nx, config::ny, 1),
        [=] __host__ __device__ (port::thread3d, int i, int j, int) {
            if(!config::is_boundary(i, j)) {
                const auto ij = config::ij(i, j);

                const auto ije = config::ij_periodic(i+1, j);
                const auto ijw = config::ij_periodic(i-1, j);
                const auto ijn = config::ij_periodic(i, j+1);
                const auto ijs = config::ij_periodic(i, j-1);

                const auto dx = config::dx; 

                const auto ux = (u[ije] - u[ijw]) / (2*dx);
                const auto uy = (u[ijn] - u[ijs]) / (2*dx);
                const auto vx = (v[ije] - v[ijw]) / (2*dx);
                const auto vy = (v[ijn] - v[ijs]) / (2*dx);
                const auto s00 = ux;
                const auto s01 = real(.5) * (uy + vx);
                const auto s10 = s01;
                const auto s11 = vy;
                const auto ss = sqrt(2 * (s00*s00 + s01*s10 + s10*s01 + s11*s11));
                /// traditional Smagorinky model
                //nus[ij] = 0.1 * dx*dx * ss;
                /// CSM (Kobayashi, 2003)
                const auto qq = -real(.5) * (ux*ux + uy*vx + vx*uy + vy*vy);
                const auto ee = real(.5) * (ux*ux + uy*uy + vx*vx + vy*vy);
                const auto fcs = qq/ee;
                nus[ij] = 0.05 * pow(abs(fcs), real(1.5)) * dx*dx * ss;
            } else {
            } // if is_boundary
        } // [=]__host__ __device__()
    ); // port::pfor3d()

    //auto rand_engine = std::mt19937(t);
    //auto rand_dist = std::uniform_real_distribution<real>(0, 1);
    //auto rand = [&]() { return rand_dist(rand_engine); };
    //for( auto ij: util::irange(config::nx * config::ny)) {
    //    nus[ij] = config::nu * rand();
    //}
    #endif
}

void lbm_core(data& dat, int t) {
    const real* f = dat.d().f.data();
    const real* r = dat.d().r.data();
    const real* u = dat.d().u.data();
    const real* v = dat.d().v.data();
    #ifdef LES
    const real* nus = dat.d().nus.data();
    #endif
    const real* force_x = dat.d0().force_x.data();
    const real* force_y = dat.d0().force_y.data();

    real* fn = dat.dn().f.data();
    real* rn = dat.dn().r.data();
    real* un = dat.dn().u.data();
    real* vn = dat.dn().v.data();

    // update by order: f --> macro
    port::pfor3d<backend>(
        port::thread3d(config::nx, config::ny, 1),
        [=] __host__ __device__ (port::thread3d, int i, int j, int) {
            const int ij = config::ij(i, j);
            if(!config::is_boundary(i, j)) {
                // lbm
                real fc[config::Q];
                /// streaming
                #pragma unroll
                for(int q=0; q<config::Q; q++) {
                    int iu = (i - config::ex(q) + config::nx) % config::nx;
                    int ju = (j - config::ey(q) + config::ny) % config::ny;
                    fc[q] = f[config::ijq(iu, ju, q)];
                }

                /// macro
                real rnc = rf(fc);
                real unc = uf(fc, rnc);
                real vnc = vf(fc, rnc);

                /// macro output after streaming
                rn[ij] = rnc;
                un[ij] = unc;
                vn[ij] = vnc;

                ///LES
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

                /// force
                const real fxc = force_x[ij];
                const real fyc = force_y[ij];
                #pragma unroll
                for(int q=0; q<config::Q; q++) {
                    fc[q] += force_acc(fxc, fyc, q, rnc, unc, vnc, omega);
                    //fc[q] += force_acc(fxc, fyc, q, config::rho_ref, unc, vnc, omega);
                }
                /// macro
                rnc = rf(fc);
                unc = uf(fc, rnc);
                vnc = vf(fc, rnc);

                /// collision
                lbm_collision_srt(fc, omega, rnc, unc, vnc);
                //lbm_collision_mrt(fc, omega, rnc, unc, vnc);


                /// final write
                #pragma unroll
                for(int q=0; q<config::Q; q++) { fn[config::ijq(i, j, q)] = fc[q]; }
            } else {
            } // if is_boundary
        } // [=]__host__ __device__()
    ); // port::pfor3d()

}
