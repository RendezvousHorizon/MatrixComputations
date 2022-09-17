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
    const int K,
    __local float *bcol) {

    int i = get_global_id(0);
    int li = get_local_id(0);
    int ln = get_local_size(0);

    // Copy row a[i, :] to private memory
    // TODO: How to create a private dynamic array?
    float arow[1024];
    for (int k = 0; k < K; k++)
        arow[k] = a[i + k * M];
    
    for (int j = 0; j < N; j++) {
        // Copy col b[:, j] to local memory to share with other work items inside the group.
        for (int k = li; k < K; k+=ln)
            bcol[k] = b[k * N + j];

        barrier(CLK_LOCAL_MEM_FENCE);

        float tmp = 0;
        // Dot product
        for (int k = 0; k < K; k++)
            tmp += arow[k] * bcol[k];

        c[i * N + j] = tmp;

        // barrier(CLK_LOCAL_MEM_FENCE);
    }
}