#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "local_domain.h"
#include "lu.h"
#include "pivot.h"

#define EPSILON 1e-5

double getL(double *L, int n, int i, int j)
{
    if (i > j)
        return L[i * n + j];
    else if (i == j)
        return 1;
    else 
        return 0;
}

double getU(double *U, int n, int i, int j)
{
    if (i <= j)
        return U[i * n + j];
    else 
        return 0;
}

double drand(double low, double high)
{
    return ((double)rand() * (high - low)) / (double)RAND_MAX + low;
}

int main(int argc, char **argv)
{
    srand(time(NULL));

    MPI_Init(&argc, &argv);
    
    int lrank, nrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &lrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);

    int N, B, P, Q;
    if (argc != 5)
    {
        printf("Usage: %s <N> <B> <P> <Q>\n", argv[0]);
        return -1;
    }
    N = atoi(argv[1]);
    B = atoi(argv[2]);
    P = atoi(argv[3]);
    Q = atoi(argv[4]);

    local_domain_t ldomain;
    grid_t grid;
    double *globalA;

    //init globalA
    if (lrank == 0)
    {
        globalA = (double *)malloc(N * N * sizeof(double));
        for (int i = 0; i < N * N; i++)
            globalA[i] = drand(-1, 1);
    }

    init_grid(&grid, MPI_COMM_WORLD, P, Q);
    init_local_domain(globalA, N, B, &grid, &ldomain);

    // initial pivot
    pivot_final_t pivot;
    init_pivot(&pivot, N);

    // LU factorization
    luf_pivoted(&ldomain, &pivot);
    #ifdef DEBUG
    print_global_matrix(&ldomain, "Final LU");
    #endif
    
    // check correctness
    double *LU;
    if (lrank == 0)
    {
        LU = (double *)malloc(N * N * sizeof(double));
        gather_local_domain(LU, &ldomain);
        double *C = (double *)calloc(N * N, sizeof(double));

        // calculate LU
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                    C[i * N + j] += getL(LU, N, i, k) * getU(LU, N, k, j);
        
        #ifdef DEBUG
        printf("Matmul(L, U):\n");
        print_matrix(C, N);
        printf("Pivot:\n");
        for (int i = 0; i < N; i++)
            printf("%d ", pivot.data[i]);
        printf("\n");
        printf("A:\n");
        print_matrix(globalA, N);
        #endif
        // compare LU with PA
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                double pa = globalA[pivot.data[i] * N + j];
                double lu = C[i * N + j];
                assert(ABS(pa - lu) < EPSILON);
                #ifdef DEBUG
                printf("%f ", ABS(pa - lu));
                #endif
            }
            #ifdef DEBUG
            printf("\n");
            #endif
        }

        printf("Result correct.\n");

        free(LU);
        free(C);
        free(globalA);
    }
    else 
    {
        gather_local_domain(LU, &ldomain);
    }
    MPI_Finalize();

    return 0;
}