#ifndef UTIL_CONDITIONAL_HPP_
#define UTIL_CONDITIONAL_HPP_

namespace util {
template<bool TrueFalse, class Then, class Else_> struct conditional { using type = Then; };
template<class Then_, class Else> struct conditional<false, Then_, Else> { using type = Else; };
}

#endif

