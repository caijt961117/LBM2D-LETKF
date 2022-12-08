// bindiff:
// calc rmse between dir<args[0], args[1]>, for each time, {u,v}_time.dat

#include <iostream>
#include <fstream>
#include <cstdint> // std::intptr_t
#include <cmath> // nan
#include <cstdio>
#include <algorithm> // max,min,abs

#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>

#include "util/runtime_error.hpp"
#include "util/argstr.hpp"
#include "util/range.hpp"

using real = float;
constexpr int skip = 10; // ioprune

int main(int argc, char** argv) {
    try {
        // args: dir_left, dir_right, t
        const auto args = util::argstr(argc, argv);
        const auto prefix0 = args.at(0);
        const auto prefix1 = args.at(1);

        // for each time
        const auto t_iter = boost::lexical_cast<int>(args.at(2));
        for(const auto t: util::irange(t_iter)) {
            if(t % skip != 0) { continue; }
            long long int count = 0;
            long double sum_sq_error = 0;
            long double sum_sq_nature = 0;
            for(const std::string fname: {"u", "v"}) {
            //for(const std::string fname: {"f"}) {
                const auto fu = fname + "_" + std::to_string(t) + ".dat";
                auto u0 = std::ifstream(prefix0 + "/" + fu, std::ifstream::binary);
                auto u1 = std::ifstream(prefix1 + "/" + fu, std::ifstream::binary);
                auto buf0 = real(std::nan(""));
                auto buf1 = real(std::nan(""));
                while( u0.read(reinterpret_cast<char*>(&buf0), sizeof(real)).good()
                    && u1.read(reinterpret_cast<char*>(&buf1), sizeof(real)).good()
                ) {
                    const auto error = (buf0 - buf1);
                    sum_sq_error += error*error;
                    sum_sq_nature += buf0*buf0;
                    count ++;
                }
            }
            //runtime_assert(count > 0);
            const auto rmse = std::sqrt(sum_sq_error / count);
            std::cerr << t << ": " << rmse << " (count = " << count << ")" << std::endl;
            std::cout << t << "," << rmse << std::endl;
        }

    } catch (std::runtime_error e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (std::exception e) {
        std::cerr << e.what() << std::endl;
        return 2;
    } catch (...) {
        std::cerr << "unknown error" << std::endl;
        return 89;
    }

} // main()

