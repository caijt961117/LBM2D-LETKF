#ifndef CUSOLVER_GET_ERROR_STRING_HPP_
#define CUSOLVER_GET_ERROR_STRING_HPP_

#include <string>
#include <cusolver_common.h>

static inline std::string
cusolverGetErrorString(cusolverStatus_t error) {
    switch (error) {
        #define CASE(x) case x: return #x;
        CASE(CUSOLVER_STATUS_SUCCESS);
        CASE(CUSOLVER_STATUS_NOT_INITIALIZED);
        CASE(CUSOLVER_STATUS_ALLOC_FAILED);
        CASE(CUSOLVER_STATUS_INVALID_VALUE);
        CASE(CUSOLVER_STATUS_ARCH_MISMATCH);
        CASE(CUSOLVER_STATUS_MAPPING_ERROR);
        CASE(CUSOLVER_STATUS_EXECUTION_FAILED);
        CASE(CUSOLVER_STATUS_INTERNAL_ERROR);
        CASE(CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
        CASE(CUSOLVER_STATUS_NOT_SUPPORTED);
        CASE(CUSOLVER_STATUS_ZERO_PIVOT);
        CASE(CUSOLVER_STATUS_INVALID_LICENSE);
        #undef CASE
    }

    return std::string("unknown error: `") + std::to_string(int(error)) + "`";
}

#endif
