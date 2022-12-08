#ifndef CUBLAS_GET_ERROR_STRING_HPP_
#define CUBLAS_GET_ERROR_STRING_HPP_

#include <string>

#include <cublas.h>

static inline std::string
cublasGetErrorString(cublasStatus_t error) {
    switch (error) {
        #define CASE(x) case x: return #x;
        CASE(CUBLAS_STATUS_SUCCESS);         
        CASE(CUBLAS_STATUS_NOT_INITIALIZED);
        CASE(CUBLAS_STATUS_ALLOC_FAILED);    
        CASE(CUBLAS_STATUS_INVALID_VALUE);   
        CASE(CUBLAS_STATUS_ARCH_MISMATCH);   
        CASE(CUBLAS_STATUS_MAPPING_ERROR);   
        CASE(CUBLAS_STATUS_EXECUTION_FAILED);
        CASE(CUBLAS_STATUS_INTERNAL_ERROR);  
        CASE(CUBLAS_STATUS_NOT_SUPPORTED);   
        CASE(CUBLAS_STATUS_LICENSE_ERROR);   
        #undef CASE
    }

    return std::string("unknown error: `") + std::to_string(int(error)) + "`";
}

#endif
