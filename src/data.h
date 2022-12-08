#ifndef DATA_H_
#define DATA_H_

#include <curand.h>

#include "config.h"
#include "util/cu_vector.hpp"
#include "util/cuda_hostdevice.hpp"
#include "util/mpi_wrapper.hpp"
#include "util/property.hpp"

struct stdata {
    #ifdef PORT_CUDA
    using fp = util::cu_vector<real>;
    #else
    using fp = std::vector<real>;
    #endif

    fp f, r, u, v, nus; // lbm

    fp force_x, force_y;
    fp force_kx, force_ky, force_amp; // forcing spectrum by Maltrud 1991

    fp beta; // LETKF adaptive covariance inflation

    // randoms
    curandGenerator_t randgen;
    int n_rand_buf, i_rand_outdated;
    float* force_theta_d;

    stdata() = delete;
    stdata(int ensemble_id) { init(ensemble_id); }
    void init(int ensemble_id) { init_lbm(ensemble_id); init_force(ensemble_id); }
    void init_lbm(int ensemble_id);
    void init_force(int ensemble_id);
    void init_forces(int ensemble_id);
    void update_forces_maltrud91(int ensemble_id);
    void update_forces_xia14(int ensemble_id);
    void update_forces_watanabe97(int ensemble_id);
    void update_forces(int ensemble_id) { update_forces_watanabe97(ensemble_id); }
    void purturbulate(int ensemble_id); // for lyapnov test
    void inspect() const;
};

struct data : util::property_enabled<data> {
public:
    property<int> ensemble_id;

private:
    stdata data_0_, data_1_;
    bool swap_d_ { false };

public:
    data() = delete;
    data(int ensemble_id): ensemble_id(ensemble_id), data_0_(ensemble_id), data_1_(ensemble_id) {}
    void init_heavy();
          auto& d()        { return (swap_d_? data_1_ : data_0_); }
    const auto& d()  const { return (swap_d_? data_1_ : data_0_); }
          auto& dn()       { return (swap_d_? data_0_ : data_1_); }
    const auto& dn() const { return (swap_d_? data_0_ : data_1_); }
          auto& d0()       { return data_0_; }
    const auto& d0() const { return (swap_d_? data_1_ : data_0_); }
    void swap() { swap_d_ = !swap_d_; }
};

#endif

