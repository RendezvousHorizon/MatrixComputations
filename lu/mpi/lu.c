
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
    double lmax = 0;
    for (;lbi < ldomain->NBP; lbi++)
    {
        int i = 0;
        if (G_BROW(ldomain, lbi) == bk) // this block is in the diagonal
            i = k;
        
        double *blk = BLK(ldomain, lbi, lbj);
        for (; i < B; i++)
        {
            double v = BLK_ELE(blk, B, i, k);
            if (ABS_GT(v, lmax))
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
            if (ABS_GT(lmaxs[i], lmax))
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
            pivot->data[k] = pivot_row - bstart * B;
            // swap two row: gk and pivot_row
            if (pivot_row != gk)
               swap_row(ldomain, gk, pivot_row, bstart * B, bstart * B + B);

            // Panel[gk+1:] -= Panel[gk] * Panel[gk+1:,0] / Panel[gk, 0]
            double *row_gk;
            int len = B - k - 1;
            if (MY_ROW == prime_row)
            {
                row_gk = BLK(ldomain, L_BROW(ldomain, gk), L_BCOL(ldomain, gk)) + k * B + k;
            }
            else
            {
                row_gk = (double *)malloc((len + 1) * sizeof(double));
            }
            MPI_Bcast(row_gk, len + 1, MPI_DOUBLE, prime_row, ldomain->grid->col_comm);
            #ifdef DEBUG
            printf("[DEBUG][lu.c:luf_panel][Rank %d]:row_gk[0]=%f\n", MY_RANK, row_gk[0]);
            #endif
            int lbi = PANEL_L_BROW_START(ldomain, bstart);
            int lbj = PANEL_L_BCOL(ldomain, bstart);
            #ifdef DEBUG
            printf("[DEBUG][lu.c:luf_panel][Rank %d]:lbi=%d, lbj=%d\n", MY_RANK, lbi, lbj);
            #endif
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

// solve L x = A and store the results back to A. L, x and A are nxn matrixes.
static void forward_substitution(double *L, double *A, int n)
{
    for (int k = 0; k < n; k++)
        for (int i = k + 1; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i * n + j] -= L[i * n + k] * A[k * n + j];
}

static void solve_U12(local_domain_t *ldomain, int bstart)
{
    int B = ldomain->B;
    if (MY_ROW == P_FROM_G_ROW(ldomain, bstart * B))
    {
        double *L11;
        int lbi = L_BROW(ldomain, bstart * B);
        int lbj = L_BCOL_START(ldomain, bstart * B + B);
        int prime_col = Q_FROM_G_COL(ldomain, bstart * B); 
        if (MY_COL == prime_col)
            L11 = BLK(ldomain, lbi, L_BCOL(ldomain, bstart * B));
        else
            L11 = (double *)malloc(B * B * sizeof(double));
        MPI_Bcast(L11, B * B, MPI_DOUBLE, prime_col, ldomain->grid->row_comm);

        for (;lbj < ldomain->NBQ; lbj++)
        {
            double *blk = BLK(ldomain, lbi, lbj);
            // solve L11 * U = blk
            forward_substitution(L11, blk, B);
        }
        if (MY_COL != prime_col)
            free(L11);    
    }
}

static void C_minus_A_matmul_B(double *C, double *A, double *B, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                BLK_ELE(C, n, i, j) -= BLK_ELE(A, n, i, k) * BLK_ELE(B, n, k, j);
}

static void subtract_A22(local_domain_t *ldomain, int bstart)
{
    int B = ldomain->B;
    int lbi = L_BROW_START(ldomain, bstart * B + B);
    int lbj = L_BCOL_START(ldomain, bstart * B + B);
    int nbp = ldomain->NBP - lbi;
    int nbq = ldomain->NBQ - lbj;
    int prime_row = P_FROM_G_ROW(ldomain, bstart * B);
    int prime_col = Q_FROM_G_COL(ldomain, bstart * B);

    double **u_blks, **l_blks;
    u_blks = (double **)malloc(nbp * sizeof(double *));
    l_blks = (double **)malloc(nbq * sizeof(double *));

    // bcast U12
    if (MY_ROW == prime_row)
    {
        for (int j = 0; j < nbq; j++)
            u_blks[j] = BLK(ldomain, L_BROW(ldomain, bstart * B), lbj + j);
    }
    else
    {
        for (int j = 0; j < nbq; j++)
            u_blks[j] = (double *)malloc(B * B * sizeof(double));
    }
    for (int j = 0; j < nbq; j++)
        MPI_Bcast(u_blks[j],  B * B, MPI_DOUBLE, prime_row, ldomain->grid->col_comm);

    // bcast L21
    if (MY_COL == prime_col)
    {
        for (int i = 0; i < nbp; i++)
            l_blks[i] = BLK(ldomain, lbi + i, L_BCOL(ldomain, bstart * B));
    }
    else
    {
        for (int i = 0; i < nbp; i++)
            l_blks[i] = (double *)malloc(B * B * sizeof(double));
    }
    for (int i = 0; i < nbp; i++)
        MPI_Bcast(l_blks[i], B * B, MPI_DOUBLE, prime_col, ldomain->grid->row_comm);

    // perform Aij -= Lik * Ukj locally
    for (int i = 0; i < nbp; i++)
        for (int j = 0; j < nbq; j++)
        {
            C_minus_A_matmul_B(BLK(ldomain, lbi + i, lbj + j), l_blks[i], u_blks[j], B);
        }

    // free malloced memory
    if (MY_ROW != prime_row)
    {
        for (int j = 0; j < nbq; j++)
            free(u_blks[j]);
    }
    if (MY_COL != prime_col)
    {
        for (int i = 0; i < nbp; i++)
            free(l_blks[i]);
    }
    free(l_blks);
    free(u_blks);
}

static void luf_pivoted_recursive(local_domain_t *ldomain, pivot_final_t *pivot, int bstart)
{    
    int B = ldomain->B;
    int N = ldomain->N;

    // Recursion stop.
    if (bstart * B == N)
        return;

    pivot_swap_t swap_pivot;
    init_pivot(&swap_pivot, B);
    // Step 1: LU left panel: P[A11T, A12T]T = [L11T, L21T]T * U11
    luf_panel(ldomain, bstart, &swap_pivot);

    // Step 2: permute [A12T, A22T]T with P. Get A12', A22'. And permute L in the left of the panel.
    #ifdef DEBUG
    fflush(stdout);
    for (int i = 0; i < ldomain->grid->P * ldomain->grid->Q; i++)
    {
        if (MY_RANK == i)
        {
            if (i == 0)
                printf("swap pivot:\n");
            printf("rank %d:", MY_RANK);
            for (int j = 0; j < swap_pivot.n; j++)
                printf("%d ",swap_pivot.data[j]);
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    #endif
    permute_ldomain(ldomain, &swap_pivot, bstart * B, bstart * B + B, N);
    permute_ldomain(ldomain, &swap_pivot, bstart * B, 0, bstart * B);

    // Step 3: solve A12' = L11[U12] 
    solve_U12(ldomain, bstart);
    // Step 4: A22' -= L21 * U12
    subtract_A22(ldomain, bstart);

    // Step 5: recursively LU A22'
    pivot_final_t sub_pivot;
    init_pivot(&sub_pivot, pivot->n - B);
    luf_pivoted_recursive(ldomain, &sub_pivot, bstart + 1);

    // Step 6: update final pivot
    pivot_apply_swap_to_final(pivot, &swap_pivot);
    pivot_apply_final_to_final(pivot, &sub_pivot);

    free_pivot(&swap_pivot);
    free_pivot(&sub_pivot);
}

void luf_pivoted(local_domain_t *ldomain, pivot_final_t *pivot)
{
    luf_pivoted_recursive(ldomain, pivot, 0);
}
