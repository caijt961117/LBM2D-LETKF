#ifndef IO_H_
#define IO_H_

#include "data.h"
#include "util/mpi_wrapper.hpp"

void output(const data& dat, util::mpi mpi, int t);

#endif
