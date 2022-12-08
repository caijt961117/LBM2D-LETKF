// bindiff:
// calc rmse between dir<args[0], args[1]>, for each time, {u,v}_time.dat

#include <iostream>
#include <fstream>
#include <cmath> // nan
#include <cstdio>
#include <array>

#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>

#include "util/argstr.hpp"
#include "util/range.hpp"
#include "util/runtime_error.hpp"

using real = float;
using byte = char;

constexpr int nx = 128*128;
constexpr int bufsize = nx*2; // u, v
constexpr int skip = 10; // ioprune
using buffer_t = std::array<real, bufsize>;

#ifndef N_ENS
#define N_ENS
#endif
constexpr int n_ens = N_ENS;

int main(int argc, char** argv) try {
    // args: dir_left, dir_right, t
    const auto args = util::argstr(argc, argv, true);
    const auto prefix_nature = args.at(0);
    const auto prefix_ensemble = args.at(1);
    std::cout << "dirinfo: nature=" << prefix_nature << " -- " 
        << "ensemble=" << prefix_ensemble << std::endl;

    // buf
    buffer_t buf_nature, buf_ens_mean, buf_ens_stdev;
    std::array<buffer_t, n_ens> buf_ensemble;

    // for each time
    const auto t_iter = boost::lexical_cast<int>(args.at(2));
    for(const auto t: util::irange(t_iter)) {
        if(t % skip != 0) { continue; }
        // load nature
        {
            buf_nature.fill(std::nan("0"));
            const std::string fu = prefix_nature + "/u_"  + std::to_string(t) + ".dat";
            const std::string fv = prefix_nature + "/v_"  + std::to_string(t) + ".dat";
            std::ifstream u(fu, std::ios::binary), v(fv, std::ios::binary);
            u.read((byte*)buf_nature.data(), nx*sizeof(real));
            v.read((byte*)(buf_nature.data() + nx), nx*sizeof(real));
            //runtime_assert(u.good() && v.good());
            //for(const auto& b: buf_nature) {
            //    runtime_assert(!std::isnan(b));
            //}
        }
        // load emsemble files
        for(const auto k: util::irange(n_ens)) {
            buf_ensemble[k].fill(std::nan("0"));
            const std::string fu = prefix_ensemble + "/" + std::to_string(k) + "/u_"  + std::to_string(t) + ".dat";
            const std::string fv = prefix_ensemble + "/" + std::to_string(k) + "/v_"  + std::to_string(t) + ".dat";
            std::ifstream u(fu, std::ios::binary), v(fv, std::ios::binary);
            u.read((byte*)buf_ensemble[k].data(), nx*sizeof(real));
            v.read((byte*)(buf_ensemble[k].data() + nx), nx*sizeof(real));
            //runtime_assert(u.good() && v.good());
            //for(const auto& b: buf_ensemble[k]) {
            //    runtime_assert(!std::isnan(b));
            //}
        }
        // calc ensemble stat
        {
            for(const auto i: util::irange(bufsize)) {
                long double sum = 0, ssum = 0;
                for(const auto k: util::irange(n_ens)) {
                    const real& b = buf_ensemble[k][i];
                    sum += b;
                    ssum += b*b;
                }
                const auto mean = sum / n_ens;
                buf_ens_mean[i] = mean;
                buf_ens_stdev[i] = std::sqrt(ssum/n_ens - mean*mean);
            }
        }
        // calc error to nature
        {        
            long double mean = 0, spread = 0, rmse = 0;
            for(const auto i: util::irange(bufsize)) {
                mean += buf_ens_mean[i];
                spread += std::pow(buf_ens_stdev[i], real(2));
                rmse += std::pow(buf_ens_mean[i] - buf_nature[i], real(2));
            }
            mean = mean / bufsize;
            spread = std::sqrt(spread / bufsize);
            rmse = std::sqrt(rmse / bufsize);
            std::cout << "ensemble_mean, spread, rmse = " << mean << ", " << spread << ", " << rmse << ", " << std::endl;
            std::cerr << mean << "," << spread << "," << rmse << std::endl;
        }
    }

} catch (std::runtime_error e) {
    std::cout << e.what() << std::endl;
    return 1;
} catch (std::exception e) {
    std::cout << e.what() << std::endl;
    return 2;
} catch (...) {
    std::cout << "unknown error" << std::endl;
    return 89;
} // main


