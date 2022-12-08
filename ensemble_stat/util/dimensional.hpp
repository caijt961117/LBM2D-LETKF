// やろうと思ったらもうあった。
// https://www.ruche-home.net/boyaki/2013-12-28/Carray
#ifndef UTIL_DIMENSIONAL_HPP_
#define UTIL_DIMENSIONAL_HPP_

#include <array>

namespace util {

namespace util_impl_ {

template<typename T, std::size_t Sz, std::size_t... Szs>
struct dimensional_impl_ {
  using type = std::array<typename dimensional_impl_<T, Szs...>::type, Sz>;
};

template<typename T, std::size_t Sz_tail>
struct dimensional_impl_<T, Sz_tail> {
  using type = std::array<T, Sz_tail>;
};

}

template<typename T, std::size_t Sz, std::size_t... Szs>
using dimensional = typename util_impl_::dimensional_impl_<T, Sz, Szs...>::type;

}

#endif

