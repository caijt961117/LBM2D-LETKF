#ifndef MPI_WRAPPER_HPP_
#define MPI_WRAPPER_HPP_

#include <mpi.h>
#include <iostream>
#include "range.hpp"
#include "mpi_safe_call.hpp"

namespace util {

/// struct mpi
struct mpi {
public:
    using int_type = int; // mpi int
private:
    const MPI_Comm comm_;
    int_type rank_;
    int_type size_;

public:
    // setup
    mpi() = delete;
    mpi(const MPI_Comm comm): comm_(comm) {
        MPI_SAFE_CALL(MPI_Comm_rank(comm_, &rank_));
        MPI_SAFE_CALL(MPI_Comm_size(comm_, &size_));
    }
    // basic
    void barrier() const { MPI_SAFE_CALL(MPI_Barrier(comm_)); }
    void abort(int exit_code) const { MPI_SAFE_CALL(MPI_Abort(comm_, exit_code)); }

    // info
    auto rank() const { return rank_; }
    auto size() const { return size_; }
    const auto& comm() const { return comm_; }

    // reduction
    template<typename T> struct MPItypename { static MPI_Datatype name(); }; 
    template<typename T> T reduce_sum(T t) const { T ret = 0; MPI_Allreduce(&t, &ret, 1, MPItypename<T>::name(), MPI_SUM, comm_); return ret; }
    template<typename T> T reduce_max(T t) const { T ret = 0; MPI_Allreduce(&t, &ret, 1, MPItypename<T>::name(), MPI_MAX, comm_); return ret; }
    template<typename T> T reduce_min(T t) const { T ret = 0; MPI_Allreduce(&t, &ret, 1, MPItypename<T>::name(), MPI_MIN, comm_); return ret; }

    // utility
    template<class Func> void for_each_rank(Func&& func) const {
        for(const auto& i: irange(size())) {
            barrier();
            if(rank() == i) { func(); }
        }
        barrier();
    }
    template<class Func> void on_master(Func&& func) const {
        barrier();
        if(rank() == 0) { func(); }
        barrier();
    }
}; // struct mpi

template<> struct mpi::MPItypename<float      > { static MPI_Datatype name() { return MPI_FLOAT      ; } };
template<> struct mpi::MPItypename<double     > { static MPI_Datatype name() { return MPI_DOUBLE     ; } };
template<> struct mpi::MPItypename<long double> { static MPI_Datatype name() { return MPI_LONG_DOUBLE; } };
template<> struct mpi::MPItypename<int        > { static MPI_Datatype name() { return MPI_INT        ; } };
template<> struct mpi::MPItypename<long       > { static MPI_Datatype name() { return MPI_LONG       ; } };
template<> struct mpi::MPItypename<long long  > { static MPI_Datatype name() { return MPI_LONG_LONG  ; } };

}

#endif

