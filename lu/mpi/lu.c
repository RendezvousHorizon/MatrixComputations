
#include "lu.h"

// rv = argmax_i(A[k:, k])
static int find_pivot(local_domain_t *ldomain, int k)
{
    int B = ldomain->B;
    int P = ldomain->grid->P;
    int bk = (int)(k / B);
    k = k % B;

    int lbi = PANEL_L_BROW_START(ldomain, bk);
    int lbj = PANEL_L_BCOL(ldomain, bk);

    // calculate local max
    int pivot_row = -1;
    double lmax = -DBL_MAX;
    for (;lbi < ldomain->NBP; lbi++)
    {
        int i = 0;
        if (G_BROW(ldomain, lbi) == bk) // this block is in the diagonal
            i = k;
        
        double *blk = BLK(ldomain, lbi, lbj);
        for (; i < B; i++)
        {
            double v = BLK_ELE(blk, B, i, k);
            if (v > lmax)
            {
                lmax = v;
                pivot_row = G_ROW(ldomain, lbi, i);
            }
        }
    }
    
    #ifdef DEBUG
    printf("Rank %d: (Before bcast) pivot_row=%d, lmax=%f\n", ldomain->grid->lrank, pivot_row, lmax);
    #endif 

    // gather & broadcast
    int root = 0;
    double *lmaxs;
    int *pivot_rows;
    if (MY_ROW == root)
    {
        lmaxs = (double *) malloc(P * sizeof(double));
        pivot_rows = (int *) malloc(P * sizeof(int));
    }
    MPI_Gather(&lmax, 1, MPI_DOUBLE, lmaxs, 1, MPI_DOUBLE, root, ldomain->grid->col_comm);
    MPI_Gather(&pivot_row, 1, MPI_INT, pivot_rows, 1, MPI_INT, root, ldomain->grid->col_comm);

    if (MY_ROW == root)
    {
        for (int i = 0; i < P; i++)
        {
            if (lmaxs[i] > lmax)
            {
                lmax = lmaxs[i];
                pivot_row = pivot_rows[i];
            }
        }
        free(lmaxs);
        free(pivot_rows);
    }
    
    MPI_Bcast(&pivot_row, 1, MPI_INT, root, ldomain->grid->col_comm);
    #ifdef DEBUG
    printf("Rank %d: (After bcast) pivot_row=%d\n", ldomain->grid->lrank, pivot_row);
    #endif
    return pivot_row;
}


static void daxpy(double *y, double a, double *x, int n)
{
    for (int i = 0; i < n; i++)
        y[i] += a * x[i];
}

static void luf_panel(local_domain_t *ldomain, int bstart, pivot_swap_t *pivot)
{
    int B = ldomain->B;
    int P = ldomain->grid->P;
    int Q = ldomain->grid->Q;
    int panel_row = bstart % P;
    int panel_col = bstart % Q;
    
    if (MY_COL == panel_col) 
    {
        int prime_row = P_FROM_G_ROW(ldomain, bstart * B);
        // Factorize panel in one grid column
        for (int k = 0; k < B; k++)
        {
            int gk = bstart * B + k;
            int pivot_row = find_pivot(ldomain, gk);
            pivot->data[k] = pivot_row;
            // swap two row: gk and pivot_row
            if (pivot_row != gk)
            {
                int p1 = prime_row;
                int p2 = P_FROM_G_ROW(ldomain, pivot_row);
                int len = B - k;
                double *buf1, *buf2;
                if (p1 == MY_ROW && p2 == MY_ROW)
                {
                    buf1 = BLK(ldomain, L_BROW(ldomain, gk), L_BCOL(ldomain, gk)) + k * B + k;
                    buf2 = BLK(ldomain, L_BROW(ldomain, pivot_row), L_BCOL(ldomain, gk)) + (pivot_row % B) * B + k;
                    double *tmp = (double *)malloc(len * sizeof(double));
                    memcpy(tmp, buf1, len * sizeof(double));
                    memcpy(buf1, buf2, len * sizeof(double));
                    memcpy(buf2, tmp, len * sizeof(double));
                    free(tmp);
                }
                else if (p1 == MY_ROW)
                {
                    buf1 = BLK(ldomain, L_BROW(ldomain, gk), L_BCOL(ldomain, gk)) + k * B + k;
                    MPI_Sendrecv_replace(buf1, len, MPI_DOUBLE, p2, 0, p2, 0, ldomain->grid->col_comm, MPI_STATUS_IGNORE);
                }
                else if (p2 == MY_ROW)
                {         
                    buf2 = BLK(ldomain, L_BROW(ldomain, pivot_row), L_BCOL(ldomain, gk)) + (pivot_row % B) * B + k;
                    MPI_Sendrecv_replace(buf2, len, MPI_DOUBLE, p1, 0, p1, 0, ldomain->grid->col_comm, MPI_STATUS_IGNORE);
                }
            }

            // Panel[gk+1:] -= Panel[gk] * Panel[gk+1:,0] / Panel[gk, 0]
            double *row_gk;
            int len = B - k - 1;
            if (len == 0)
                continue;;
            if (MY_ROW == prime_row)
            {
                row_gk = BLK(ldomain, L_BROW(ldomain, gk), L_BCOL(ldomain, gk)) + k * B + k;
            }
            else
            {
                row_gk = (double *)malloc(len * sizeof(double));
            }
            MPI_Bcast(row_gk, len, MPI_DOUBLE, prime_row, ldomain->grid->col_comm);
            int lbi = PANEL_L_BROW_START(ldomain, bstart);
            int lbj = PANEL_L_BCOL(ldomain, bstart);
            for(; lbi < ldomain->NBP; lbi++)
            {
                int i = 0;
                if (MY_ROW == prime_row && lbi == PANEL_L_BROW_START(ldomain, bstart))
                    i = k + 1;
                double *blk = BLK(ldomain, lbi, lbj);
                for (; i < B; i++)
                {
                    BLK_ELE(blk, B, i, k) /= row_gk[0];
                    // y, a, x, n: y[:n] = a * x[:n] + y
                    daxpy(&BLK_ELE(blk, B, i, k + 1), -BLK_ELE(blk, B, i, k), row_gk + 1, len);
                }
            }
            if (MY_ROW != prime_row)
                free(row_gk);

        }
    }
    // broadcast pivot along the row
    MPI_Bcast(pivot->data, pivot->n, MPI_INT, panel_col, ldomain->grid->row_comm);
    #ifdef DEBUG
    print_global_matrix(ldomain, "After swap pivot");
    #endif
}

static void luf_pivoted_recursive(local_domain_t *ldomain, pivot_final_t *pivot, int bstart)
{    
    int B = ldomain->B;
    int N = ldomain->N;

    pivot_swap_t swap_pivot;
    init_pivot(&swap_pivot, B);
    // Step1: LU left panel: P[A11T, A12T]T = [L11T, L21T]T * U11
    luf_panel(ldomain, bstart, &swap_pivot);

    // Recursion stop threshold
    if (bstart * B + B >= N)
        return;

    // Step2: permute [A12T, A22T]T with P. Get A12', A22'
    permute_ldomain(ldomain, &swap_pivot, bstart * B, bstart * B + B, N);
/*
    // Step3: solve A12' = L11[U12] 
    solve_U12(ldomain, bstart);

    // Step4: A22' -= L21 * U12
    subtact_A22();
    matrix_t *A22 = get_submatrix(ldomain);

    // Step5: recursively LU A22'
    pivot_t P2(A22->n);
    luf_pivoted_blocked(A22, P2);

    // Step6: permute L21 according to P2
    permute_L21();
*/
}

void luf_pivoted(local_domain_t *ldomain, pivot_final_t *pivot)
{
    luf_pivoted_recursive(ldomain, pivot, 0);
}
