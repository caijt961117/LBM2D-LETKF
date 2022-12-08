#ifndef ASSIMILATE_H_
#define ASSIMILATE_H_

#include <iostream>

#include "config.h"
#include "data.h"
#include "util/letkf_solver.h"
#include "util/timer.hpp"
#include "util/device_memory.hpp"

class DataAssimilator {
private:
    #ifdef PORT_CUDA
    using fp = util::cu_vector<real>;
    #else
    using fp = std::vector<real>;
    #endif
    fp obse; // I/O buffer of observation

    util::letkf_solver solver;
    util::device_memory<real> xk, yk; // LETKF buffer: k-th column of X, Y
    std::vector<real> xk_host; // LETKF buffer for MPI (overlapped)
    util::timer timer;
public:
    DataAssimilator(const util::mpi& mpi) { init(mpi); }

private:
    void init(const util::mpi& mpi) {
        #ifdef DA_OBS_UV
        obse.resize(config::nx * config::ny * 2); // observe u, v
        #else
        obse.resize(config::nx * config::ny * config::Q); // observe f
        #endif

        #ifdef DA_LETKF
        init_letkf(mpi);
        #endif
    }

public:
    void init_letkf(const util::mpi& mpi); // public for cuda_device_lambda

public:
    void assimilate_nudging_lbm(data& dat, const int t);
    void assimilate_nudging_uv(data& dat, const int t);
    void assimilate_letkf_lbm(data& dat, const util::mpi& mpi, const int t);
    void assimilate_letkf_uv(data& dat, const util::mpi& mpi, const int t);

public: 
    // wrapper of assimilate_*()
    void assimilate(data& dat, const util::mpi& mpi, const int t) {
        #ifdef DA_LETKF
            #if defined(DA_OBS_UV) || defined(DA_OBS_RUV)
            assimilate_letkf_uv(dat, mpi, t);
            #else
            assimilate_letkf_lbm(dat, mpi, t);
            #endif
        #else
            #if defined(DA_OBS_UV) || defined(DA_OBS_RUV)
            assimilate_nudging_uv(dat, t);
            #else
            assimilate_nudging_lbm(dat, t);
            #endif
        #endif
    }

    // getter
    const auto& get_timer() const { return timer; }
    const auto& get_letkf_timer() const { return solver.get_timer(); }
    void clear_all_timer() { timer.clear(); solver.clear_timer(); }

};

#endif

