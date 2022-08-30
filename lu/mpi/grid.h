#ifndef GRID_H
#define GRID_H

#include <mpi.h>
#include <stdio.h>

typedef struct grid_s
{
    MPI_Comm        all_comm;
    MPI_Comm        row_comm;
    MPI_Comm        col_comm;
    int P;
    int Q;
    int size;
    int lrank;
    int myrow;
    int mycol;

} grid_t;

#define RANK_FROM_PQ(grid, p, q) (p * grid->Q + q)

void init_grid(grid_t *grid, MPI_Comm comm, int P, int Q);

#endif