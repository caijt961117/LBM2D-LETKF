# An example of 2D LBM-LETKF

<img width=480 src=doc/header.png />

This is a microbenchmark code of 2D lattice Boltzmann method-local ensemble transform Kalman filter (LBM-LETKF) system for ensemble data assimilation of turbulent flow.
Computation is fully implemented in GPU by using CUDA.
Problem size of the test is typically 128 x 128 grid points with {4,16,64} ensembles, where the required number of GPUs is equal to the ensemble size (4, 16, or 64 GPUs).

## Usage

### Preperation

Firstly, the following external libraries are required:

- GCC >= 7.4.0
- CUDA >= 11.0
- CUDA-aware MPI (e.g. module `ompi-cuda` at Wisteria-A at ITC UTokyo, `mpt/2.23-ga` at HPE SGI8600 at JAEA)
- HDF5 (ver 1.8.2 and 1.12.0 are confirmed)
- Boost C++

**optional:** As eigenvalue decomposition solver for LETKF, while `cuSOLVER` is used as default, [`EigenG-Batched`](https://www.r-ccs.riken.jp/labs/lpnctrt/projects/eigengbatched/index.html), an open source code released from RIKEN, is also available.
(Due to performance reason, `EigenG-Batched` is highly recommended for the larger ensemble size M>32)

To use `EigenG-Batched`, execute
```
./enable_EigenG-Batched.sh
```

### Compile

Select one of test modes from bolows:

| Test Mode | Description |
| -- | -- |
| `NATURE` | Compute a groundtruth |
| `OBSERVE` | Create observation data from nature run |
| `LYAPNOV` | Test lyapunov exponent of the system *(TBD)* |
| `DA_NUDGING` | Compute data assimilation experiment using nudging method |
| `DA_LETKF` | Compute data assimilation experiment using LETKF |
| `DA_DUMMY` | *dummy* |

For example, if you want to compute a groundtruth,
```bash
make clean
TEST=NATURE make
```

Besides, to check compilability of every test mode,

```bash
make test
```

### Run for performance evaluation

To proceed performance evaluation of the LETKF, one needs to run two test modes:

1. `OBSERVE` to create observation data.
2. `DA_LETKF` to run the LETKF and measure its performance.

For more detail of runs, please refer sample job scripts, [`run/wistimer`](run/wistimer) (for Wisteria-A) or [`run/jstimer`](run/jstimer) (for HPE SGI8600).

After runs, measured elapsed time can be observed in `io/elapsed_time_rank*.csv` and `io/letkf_elapsed_time_rank*.csv`.

## Publications

```bibtex
@inproceedings{Hasegawa2022-scalah22,
    title = {{GPU Optimization of Lattice Boltzmann Method with Local Ensemble Transform Kalman Filter}},
    year = {2022},
    booktitle = {ScalAH22: 13th Workshop on Latest Advances in Scalable Algorithms for Large-Scale Heterogeneous Systems held in conjunction with SC22: The International Conference for High Performance Computing, Networking, Storage and Analysis  (in press)},
    author = {
        Hasegawa, Yuta
        and Imamura, Toshiyuki
        and Ina, Takuya
        and Onodera, Naoyuki
        and Asahi, Yuuichi
        and Idomura, Yasuhiro
    }
}
```
