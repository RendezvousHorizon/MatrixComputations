# MatrixComputations
Implementation of parallel algorithms in *Matrix Computations 4th* book.
## Parallel Matmul
*matmul_mpi/matmul_mpi_cannon.c*: MPI impl of parallel matmul algorithm in Section 1.6.
## Parallel LU with Partial Pivoting
*lu/mpi*: MPI impl of parallel LU factorization in Section 3.6. 
## Matmul Neon & OpenCL
See *matmul_neon_cl*.

Config:
* Matrix size: M=N=K=1024. A is col major. B and C are row major.
* Device: MBP 14 2021.

Results:

| Method          | Precision | Duration | MFLOPS |
| ------          | --------- | -------- | ------ |
| Naive(ijk)      | FP32      | 3.91s    | 548    |
| Naive(ijk)(-O2) | FP32      | 1.85s    | 1160   |
| Block(kij)      | FP32      | 2.21s    | 970    |
| Block(kij)(-O2) | FP32      | 0.15s    | 13951  |
| Neon8x8         | FP32      | 0.59s    | 3622   |
| Neon8x8(-O2)    | FP32      | 0.03s    | 72387  |
| CL(Naive)       | FP32      | 0.009s   | 247743 |
| CL(ROW)         | FP32      | 0.27s    | 8107   |
| CL(Row+Lmem)    | FP32      | 0.29s    | 7348   |
| CL(no-mem-rw)   | FP32      | 0.64ms   | 3.37T  |

The GPU has no graphics memory, so using local memory in OpenCL doesn't help.
GPU's theoretical max FLOPS is 4.6T. A fake toy kernel that only do MADD on local variables (CL(no-mem-rw)) achieves 3.37T FLOPS, 73% of the maximum.