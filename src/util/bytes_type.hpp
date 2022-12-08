#ifndef UTIL_BYTESTYPE_HPP_
#define UTIL_BYTESTYPE_HPP_

#include <cstdint>
#include <type_traits>

namespace util {

template<typename T> using bytes_type = 
    typename std::conditional<sizeof(T) == sizeof(std::int8_t ), std::int8_t ,
    typename std::conditional<sizeof(T) == sizeof(std::int16_t), std::int16_t,
    typename std::conditional<sizeof(T) == sizeof(std::int32_t), std::int32_t,
    typename std::conditional<sizeof(T) == sizeof(std::int64_t), std::int64_t,
    void
    >::type
    >::type
    >::type
    >::type;

template<typename T>
union bytes_union {
    T value;
    bytes_type<T> b;
};

template<typename T>
bytes_type<T> bytes_from_value(T value) {
    bytes_union<T> tmp;
    tmp.value = value;
    return tmp.b;
}

template<typename T>
T value_from_bytes(bytes_type<T> b) {
    bytes_union<T> tmp;
    tmp.b = b;
    return tmp.value;
}

}

#endif

