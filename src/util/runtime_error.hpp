// runtime_error.hpp
// wrapper for std::runtime_error

#include <stdexcept>
#include <string>

#ifndef NODEBUG

#define STD_RUNTIME_ERROR(error_str) \
    std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": [error] assertion failed: " + std::string(error_str) + "; funcname: " + std::string(__PRETTY_FUNCTION__))

#define runtime_assert(con, comment) \
    if(!(con)) { throw STD_RUNTIME_ERROR(std::string(comment)); }

#else

#define runtime_assert(con, comment)

#endif
