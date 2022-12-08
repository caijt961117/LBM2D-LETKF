#ifndef TYPENAME_HPP_
#define TYPENAME_HPP_
namespace util {
template<typename U> std::string typenameof();
template<> inline std::string typenameof<float >() { return "float"; }
template<> inline std::string typenameof<double>() { return "double"; }
}
#endif
