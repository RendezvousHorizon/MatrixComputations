#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "local_domain.h"

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
        #ifdef DEBUG
        for (int i = 0; i < N * N; i++)
            globalA[i] = i;
        #else
        for (int i = 0; i < N * N; i++)
            globalA[i] = drand(-1, 1);
        #endif
    }

    init_grid(&grid, MPI_COMM_WORLD, P, Q);
    init_local_domain(globalA, N, B, &grid, &ldomain);

    MPI_Finalize();
}