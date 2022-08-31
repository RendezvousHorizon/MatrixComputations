#include "grid.h"

void init_grid(grid_t *grid, MPI_Comm comm, int P, int Q)
{
    grid->all_comm = comm;
    grid->P = P;
    grid->Q = Q;
    MPI_Comm_rank(comm, &grid->lrank);
    grid->size = P * Q;
    grid->myrow = (int)(grid->lrank / Q);
    grid->mycol = grid->lrank % Q;

    MPI_Comm_split(comm, grid->myrow, grid->mycol, &grid->row_comm);
    MPI_Comm_split(comm, grid->mycol, grid->myrow, &grid->col_comm);
    #ifdef DEBUG
    printf("Rank %d: Done init grid, P=%d, Q=%d, myrow=%d, mycol=%d.\n", grid->lrank, P, Q, grid->myrow, grid->mycol);
    #endif
}