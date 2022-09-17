__kernel void matmul(__global float* a,
                        __global float* b,
                        __global float* c,
                        const int M, 
                        const int N, 
                        const int K){
    
    int i = get_global_id(0);
    int j = get_global_id(1);
    int index = (i * N) + j;

    float sum = 0;
    for(int k = 0; k < K; k++){
        sum += a[i + k * M] * b[k * N + j];
    }
    c[index] = sum;
}
