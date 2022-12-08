#include <iostream>
#include <cstdlib>

#include "unit.h"
#include "config.h"
#include "data.h"
#include "lbm.h"

void unit_test_feq_relevant() {
    return ;;
    std::cout << "unit test feq_relevant::" << std::endl;
    const real rho = config::rho_ref;
    const real u = -0.1 * config::c;
    const real v = 0.03 * config::c;

    real f[config::Q];
    
    for(int q=0; q<config::Q; q++) {
        f[q] = feq(rho, u, v, q);
    }
    const real rhoq = rf(f);
    const real uq = uf(f, rhoq);
    const real vq = vf(f, rhoq);

    std::cout << "  rho: " << rho << ", " << rhoq << "; error = " << rho-rhoq << std::endl;
    std::cout << "  u: " << u << ", " << uq << "; error = " << u-uq << std::endl;
    std::cout << "  v: " << v << ", " << vq << "; error = " << v-vq << std::endl;
    std::exit(0);
}

