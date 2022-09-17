#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

const int DEFAULT_M = 100, DEFAULT_N = 100, DEFAULT_K = 100;
const double E = 1e-7;

double drand(double low, double high)
{
    return ((double)rand() * (high - low)) / (double)RAND_MAX + low;
}

// calculate C = C + AB
void matmul_ijk(double *A, double *B, double *C, int M, int N, int K) 
{
    for (int i = 0; i < M; i++) 
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < K; k++)
            {
                C[i * N + j] = C[i * N + j] + A[i * K + k] * B[k * N + j];
            }
        }
    }
}

void matmul_kij(double *A, double *B, double *C, int M, int N, int K) 
{
    for (int k = 0; k < K; k++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int i = 0; i < M; i++) 
            {
                C[i * N + j] = C[i * N + j] + A[i * K + k] * B[k * N + j];
            }
        }
    }
}

int main(int argc, char **argv)
{
    srand(time(NULL));

    // initial m, n, k
    int m, n, k;
    if (argc < 4)
    {
        m = DEFAULT_M;
        n = DEFAULT_N;
        k = DEFAULT_K;
    }
    else
    {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    }

    // initial matrix
    double *A = (double *)malloc(m * k * sizeof(double));
    double *B = (double *)malloc(k * n * sizeof(double));
    double *C = (double *)malloc(m * n * sizeof(double));
    double *D = (double *)malloc(m * n * sizeof(double));

    for (int i = 0; i < m * k; i++)
    {
        A[i] = drand(-1, 1);
    }
    for (int i = 0; i < k * n; i++)
    {
        B[i] = drand(-1, 1);
    }
    for (int i = 0; i < m * n; i++)
    {
        C[i] = D[i] = 0;
    }

    // preform matmul
    matmul_ijk(A, B, C, m, n, k);

    struct timeval begin, end;
    gettimeofday(&begin, 0);

    matmul_kij(A, B, D, m, n, k);
    
    gettimeofday(&end, 0);

    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds * 1e-6;

    // validation
    int rv = 0;
    for (int i = 0; i < m * n; i++)
    {
        if (fabs(C[i] - D[i]) > E)
        {
            printf("Validation failed.\n");
            rv = -1;
            break;
        }
    }

    printf("Elapsed %.6f seconds.\n", elapsed);

    // free allocated space
    free(A);
    free(B);
    free(C);
    free(D);

    return rv;
}