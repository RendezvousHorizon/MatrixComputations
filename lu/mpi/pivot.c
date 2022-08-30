#include "pivot.h"

void init_pivot(pivot_t *pivot, int n)
{
    pivot->n = n;
    pivot->data = (int *)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++)
        pivot->data[i] = i;
}

void free_pivot(pivot_t *pivot)
{
    free(pivot->data);
}

static void swap_row(local_domain_t *ldomain, int grow1, int grow2, int gj_start, int gj_end)
{   
    #ifdef DEBUG
    printf("[DEBUG][pivot.c:swap_row][rank(%d)]:Start. (grow1, grow2, gj_start, gj_end)=(%d, %d, %d, %d)\n", MY_RANK, grow1, grow2, gj_start, gj_end);
    #endif
    if (grow1 == grow2 || gj_start == gj_end)
        return;
    int B = ldomain->B;

    assert(gj_start % B == 0 && gj_end % B == 0);

    int gbj_start = (int) (gj_start / B);
    int gbj_end = (int) (gj_end / B);
    int num_block_rows = (int)(gbj_end / ldomain->grid->Q) + (MY_COL < (gbj_end % ldomain->grid->Q)) - (int)(gbj_start / ldomain->grid->Q) - (MY_COL < gbj_start % ldomain->grid->Q);
    #ifdef DEBUG
    printf("[DEBUG][pivot.c:swap_row][rank(%d)]: num_block_rows=%d\n", MY_RANK, num_block_rows);
    #endif
    if (num_block_rows == 0)
        return;

    MPI_Datatype row_type;
    MPI_Type_vector(num_block_rows, B, B * B, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);

    int p1 = P_FROM_G_ROW(ldomain, grow1);
    int p2 = P_FROM_G_ROW(ldomain, grow2);

    double *buf1, *buf2;
    if (p1 == MY_ROW && p2 == MY_ROW)
    {
        buf1 = BLK(ldomain, L_BROW(ldomain, grow1), L_BCOL(ldomain, gj_start) + (MY_COL < Q_FROM_G_COL(ldomain, gj_start))) + (grow1 % B) * B;
        buf2 = BLK(ldomain, L_BROW(ldomain, grow2), L_BCOL(ldomain, gj_start) + (MY_COL < Q_FROM_G_COL(ldomain, gj_start))) + (grow2 % B) * B;
        double *tmp = (double *)malloc(num_block_rows * B * sizeof(double));
        for (int i = 0; i < num_block_rows; i++)
        {
            memcpy(tmp, buf1, B * sizeof(double));
            memcpy(buf1, buf2, B * sizeof(double));
            memcpy(buf2, tmp, B * sizeof(double));
        }
        free(tmp);
    }
    else if (p1 == MY_ROW)
    {
        buf1 = BLK(ldomain, L_BROW(ldomain, grow1), L_BCOL(ldomain, gj_start) + (MY_COL < Q_FROM_G_COL(ldomain, gj_start))) + (grow1 % B) * B;
        MPI_Sendrecv_replace(buf1, 1, row_type, p2, 0, p2, 0, ldomain->grid->col_comm, MPI_STATUS_IGNORE);
    }
    else if (p2 == MY_ROW)
    {
        buf2 = BLK(ldomain, L_BROW(ldomain, grow2), L_BCOL(ldomain, gj_start) + (MY_COL < Q_FROM_G_COL(ldomain, gj_start))) + (grow2 % B) * B;
        MPI_Sendrecv_replace(buf2, 1, row_type, p1, 0, p1, 0, ldomain->grid->col_comm, MPI_STATUS_IGNORE);
    }

    MPI_Type_free(&row_type);

}
void permute_ldomain(local_domain_t *ldomain, pivot_swap_t *pivot, int gi_start, int gj_start, int gj_end)
{
    for (int i = 0; i < pivot->n; i++)
    {
        int grow1 = pivot->data[i];
        int grow2 = gi_start + i;
        swap_row(ldomain, grow1, grow2, gj_start, gj_end);
    }
}