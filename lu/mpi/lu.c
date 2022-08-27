
#include "lu.h"


static int find_pivot(ludata_t *A, int k)
{
    int B = A->ldomain->B;
    int P = A->grid->P;
    int bk = int(k / B);
    k = k % B;

    int lbi = PANEL_BROW_START(A->ldomain, bk);
    int lbj = PANEL_BCOL(A->ldomain, bk);

    // calculate local max
    int pivot_row = -1;
    double lmax = -DBL_MAX;
    while (lbi < A->ldomain->NBP)
    {
        int i = 0;
        if (G_BROW(A->ldomain, lbi) == bk) // this block is in the diagonal
            i = k;
        
        double *blk = BLK(A->ldomain, lbi, lbj);
        while (i < B)
        {
            double v = BLK_ELE(blk, B, i, k);
            if (v > lmax)
            {
                lmax = v;
                pivot_row = G_ROW(A->ldomain, lbi, i);
            }
            i++;
        }
        lbi++;
    }

    // gather & broadcast
    int root = COL_ROOT_RANK(A->grid);
    double *lmaxs;
    int *pivot_rows;
    if (A->grid->lrank == root)
    {
        lmaxs = (double *) malloc(P * sizeof(double));
        pivot_rows = (int *) malloc(P * sizeof(int));
    }
    MPI_Gather(&lmax, 1, MPI_DOUBLE, lmaxs, P, MPI_DOUBLE, root, A->grid->col_comm);
    MPI_Gather(&pivot_row, 1, MPI_INT, pivot_rows, P, MPI_INT, root, A->grid->col_comm);

    if (A->grid->lrank == root)
    {
        for (int i = 0; i < P; i++)
        {
            if (lmaxs[i] > lmax)
            {
                lmax = lmaxs[i];
                pivot_row = pivot_rows[i];
            }
        }
    }
    MPI_Bcast(&pivot_row, 1, MPI_INT, root, A->grid->col_comm);
    return pivot_row;

}
static void luf_panel(ludata_t *A, int bstart, pivot_t *P)
{
    int B = A->ldomain->B;
    int P = A->grid->P;
    int Q = A->grid->Q;
    int grid_row = bstart % P;
    int grid_col = bstart % Q;
    
    if (A->grid->mycol == grid_col) 
    {
        // Factorize only on one column
        for (int k = 0; k < B; k++)
        {
            int pivot_row = find_pivot(A, bstart * B + k);
            if (pivot_row != bstart * B + k)
                swap(P, k, pivot_row - bstart * B);
        }
    }
    else
    {

    }

}
static void luf_pivoted_recursive(ludata_t *A, int bstart, pivot_t *P)
{    
    int B = A->ldomain->B;
    int N = A->ldomain->N;
    // Step1: LU left panel: P[A11T, A12T]T = [L11T, L21T]T * U11
    luf_panel(A, bstart, P);

    // Recursion stop threshold
    if (gsij + B >= N)
        return;

    // Step2: permute [A12T, A22T]T with P. Get A12', A22'
    permute_panel_right(A, bstart, P);

    // Step3: solve A12' = L11[U12] 
    solve_U12(A, bstart);

    // Step4: A22' -= L21 * U12
    subtact_A22();
    matrix_t *A22 = get_submatrix(A);

    // Step5: recursively LU A22'
    pivot_t P2(A22->n);
    luf_pivoted_blocked(A22, P2);

    // Step6: permute L21 according to P2
    permute_L21();
}