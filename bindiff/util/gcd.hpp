#ifndef UTIL_GCD_HPP_
#define UTIL_GCD_HPP_

// under C++11 or 14
#include <type_traits>
namespace std {

template<typename T, typename U> 
constexpr typename ::std::common_type<T, U>::type 
gcd(const T& a, const U& b) {
    return b==0 ? a : gcd(b, a%b);
}

template<typename T, typename U>
constexpr typename ::std::common_type<T, U>::type
max(const T& a, const U& b) {
    return a<b ? b : a;
}

template<typename T, typename U>
constexpr typename ::std::common_type<T, U>::type
min(const T& a, const U& b) {
    return a<b ? a : b;
}

} // namespace std

#endif
