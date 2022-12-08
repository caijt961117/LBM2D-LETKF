#ifndef UTIL_DIVISOR_HPP_
#define UTIL_DIVISOR_HPP_
#include <vector>
#include <type_traits>

namespace util {
    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, std::vector<T>>::type
    divisor(const T& n) {
        std::vector<T> ret;
        for(T i=1; i*i <= n; i++) {
            if(n % i == 0) { ret.push_back(i); }
        }
        ret.push_back(n);
        return ret;
    }
}

#endif

