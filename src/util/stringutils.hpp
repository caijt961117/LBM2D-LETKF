#ifndef UTIL_STRINGUTILS_HPP_
#define UTIL_STRINGUTILS_HPP_
#include <string>
#include <sstream>
#include <iomanip>
#include "runtime_error.hpp"
#include "range.hpp"

namespace util {
inline std::string to_string_aligned(int t, int digits=4) { 
    auto pow = [=](int e, int x) {
        int ret=1;
        for(auto _: util::range(x)) { ret *= e; }
        return ret;
    };
    runtime_assert(0 <= t && t < pow(10, digits), "to_string_aligned: value_out_of_range within given digits");
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(digits) << t;
    return ss.str();
}
}

#endif
