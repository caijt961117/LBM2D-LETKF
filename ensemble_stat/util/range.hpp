#ifndef UTIL_RANGE_HPP_
#define UTIL_RANGE_HPP_
// range.hpp
// bash-like seq / python-like range function

#include <vector>
#include <array>
#include <cstdint>
#include "runtime_error.hpp"

namespace util {

inline std::vector<std::intptr_t> bash_seq(std::intptr_t start, std::intptr_t interval, std::intptr_t stop) {
    RUNTIME_ASSERT("invalid argument" && (stop-start)*interval >= 0);
    std::vector<std::intptr_t> ret;
    for(std::intptr_t i = start; i <= stop; i += interval) {
        ret.push_back(i);
    }
    return ret;
}

inline std::vector<std::intptr_t> bash_seq(std::intptr_t start, std::intptr_t stop) { return bash_seq(start, 1, stop); }

// python-like
inline std::vector<std::intptr_t> irange(std::intptr_t end) { return bash_seq(0, 1, end-1); }
inline std::vector<std::intptr_t> irange(std::intptr_t begin, std::intptr_t end) { return bash_seq(begin, 1, end-1); }

// 3D
inline std::vector<std::array<std::intptr_t, 3>> irange3d(std::intptr_t nx, std::intptr_t ny, std::intptr_t nz) {
    RUNTIME_ASSERT("invalid argument" && nx >= 0 && ny >= 0 && nz >= 0);
    std::vector<std::array<std::intptr_t, 3>> ret;
    for(std::intptr_t k=0; k<nz; k++) { for(std::intptr_t j=0; j<ny; j++) { for(std::intptr_t i=0; i<nx; i++) {
        ret.push_back({i, j, k});
    }}}
    return ret;
}

}

#endif

