#ifndef UTIL_ARGSTR_HPP_
#define UTIL_ARGSTR_HPP_

#include <string>
#include <vector>
#include <type_traits>
#include "range.hpp"

namespace util {

auto argstr(int argc, char** argv) {
    auto v_ = std::vector<std::string> {};
    for(int i=1; i<argc; i++) { // skip basename
        v_.emplace_back(argv[i]);
    }
    return v_;
}

}

#endif
