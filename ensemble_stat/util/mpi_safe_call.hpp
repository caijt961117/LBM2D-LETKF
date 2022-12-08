#ifndef MPI_SAFE_CALL_HPP_
#define MPI_SAFE_CALL_HPP_

#include "runtime_error.hpp"
#include <mpi.h>
#include <string>

#ifndef NODEBUG

#define MPI_SAFE_CALL( ... ) \
  { \
    const int error = __VA_ARGS__; \
    if(error != MPI_SUCCESS) { \
      char error_chars[MPI_MAX_ERROR_STRING+1] = ""; \
      int error_len = 0; \
      MPI_Error_string(error, error_chars, &error_len); \
      error_chars[error_len] = '\0'; \
      STD_RUNTIME_ERROR(std::string("MPI failed: ") + std::string(error_chars)); \
    } \
  }

#else

#define MPI_SAFE_CALL( ... ) __VA_ARGS__

#endif // ifndef NODEBUG

#endif // ifndef MPI_SAFE_CALL_HPP_

