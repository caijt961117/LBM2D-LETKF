#ifndef H5_WRAPPER_HPP
#define H5_WRAPPER_HPP

#include <hdf5.h>
#include <hdf5_hl.h>

namespace util {
template<typename T> hid_t h5typeof(T);
template<> inline hid_t h5typeof<float> (float ) { return H5T_NATIVE_FLOAT; }
template<> inline hid_t h5typeof<double>(double) { return H5T_NATIVE_DOUBLE; }
};

#endif
