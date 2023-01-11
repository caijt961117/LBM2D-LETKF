There are two examples of the results of performance tests:
- `4ens_cusolver`: LETKF with 4 ensembles (4 GPUs); eigenvalue decomposition (EVD) by `cuSOLVER`
- `64ens_eigeng`: LETKF with 64 ensembles (64 GPUs); EVD by `EigenG-Batched`

Test infomation:
- date: 2022/12/26--28
- environment: [Wisteria-A](https://www.cc.u-tokyo.ac.jp/supercomputer/wisteria/service/) at ITC UTokyo (NVIDIA A100 GPU x8, InfiniBand HDR(200Gbps) x4)
  * modules: 
    - `gcc/8.3.1`
    - `cuda/11.2`
    - `ompi-cuda/4.1.1-11.2`
    - `hdf5/1.12.0`
  * environmental variables: 
    - `UCX_MEMTYPE_CACHE=n`
    - `UCX_IB_GPU_DIRECT_RDMA=no`
