#ifndef UTIL_ARGSTR_HPP_
#define UTIL_ARGSTR_HPP_

#include <string>
#include <vector>
#include <type_traits>
#include <iostream>
#include "range.hpp"

namespace util {

auto argstr(int argc, char** argv, bool verbose=false) {
    if(verbose) {
        std::cout << "argstr: " << std::flush;
        for(int i=0; i<argc; i++) {
            std::cout << argv[i] << ' ' << std::flush;
        }
        std::cout << std::endl;
    }
    auto v_ = std::vector<std::string> {};
    for(int i=1; i<argc; i++) { // skip basename
        v_.emplace_back(argv[i]);
    }
    return v_;
}

}

#endif
