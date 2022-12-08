#ifndef UTIL_PROPERTY_HPP_
#define UTIL_PROPERTY_HPP_
// C#-like property as:
// // public Type t { get; private set; };
//
// usage:
// // struct Hoge : public util::property_enabled<Hoge> {
// //     rwro<int> val;
// // };

#include <type_traits>

namespace util {

template<class Owner>
struct property_enabled { // CRTP

    template<typename T>
    struct property { /// private read/write, public readonly
        protected: T _; // entity & operator dot accessor
    
        public: template<class... Args> property(Args&&... args): _(args...) {}

        /// getters
        public: operator const T&() const noexcept { return _; } // by cast
        public: const T& operator ()() const noexcept { return _; } // citylbm

        /// setters
        protected: T& operator =(T&& t) { return (_ = t); }
    
        friend typename std::enable_if<std::is_class<Owner>::value, Owner>::type;
    };

};

// not implemented proxy-property
// like:
// // protected int i_;
// // public int i { get { return hogehoge; } set { fugafuga(); } }

}

#endif
