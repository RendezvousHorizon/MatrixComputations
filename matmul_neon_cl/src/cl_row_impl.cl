/*
For thread i:
Calculates row C[i].
Row A[i] is stored in private memory.
Col B[:, j] is stored in local memory.
Ref: https://handsonopencl.github.io/
*/
__kernel void matmul(
    __global float *a,
    __global float *b,
    __global float *c,
    const int M,
    const int N,
    const int K){

    int i = get_global_id(0);

    for (int j = 0; j < N; j++) {
        float tmp = 0;

        for (int k = 0; k < K; k++)
            tmp += a[i + k * M] * b[k * N + j];

        c[i * N + j] = tmp;
    }
}