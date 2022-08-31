#ifndef LOCAL_DOMAIN_H
#define LOCAL_DOMAIN_H

#include <stdlib.h>
#include "grid.h"

typedef struct local_domain_s
{
    double *data; // store local domain
    const grid_t *grid; // grid
    int N; // size of square global matrix A
    int B; // block size
    int NBP; // number of blocks per rank along the row
    int NBQ; // number of blocks per rank along the column
    int NUM_BLOCKS; // number of blocks per rank
} local_domain_t;

#define PANEL_L_BCOL(ldomain, bk) ((int)((bk) / ldomain->grid->Q))
#define PANEL_L_BROW_START(ldomain, bk) ((int)((bk) / ldomain->grid->P) + (int)((bk) % ldomain->grid->P > ldomain->grid->myrow))

// get global block idx or element idx from ldomain idx in rank (p, q)
#define G_BROW_P(ldomain, lbrow, p) ((lbrow) * ldomain->grid->P + p)
#define G_BCOL_Q(ldmain, lbcol, q) ((lbcol) * ldomain->grid->Q + q)
#define G_ROW_P(ldomain, lbrow, p, i) (G_BROW_P(ldomain, lbrow, p) * ldomain->B + i)
#define G_COL_Q(ldomain, lbcol, q, j) (G_BCOL_Q(ldomain, lbcol, q) * ldomain->B + j)

// get global block idx or element idx from ldomain idx in rank self
#define G_BROW(ldomain, lbrow) G_BROW_P(ldomain, lbrow, ldomain->grid->myrow)
#define G_BCOL(ldomain, lbcol) G_BROW_Q(ldomain, lbcol, ldomain->grid->mycol)
#define G_ROW(ldomain, lbrow, i) G_ROW_P(ldomain, lbrow, ldomain->grid->myrow, i)
#define G_COL(ldomain, lbcol, j) G_COL_Q(ldomain, lbcol, ldomain->grid->mycol, j)

// global idx to local idx
#define L_BROW(ldomain, gi) ((int)((gi) / ldomain->B / ldomain->grid->P))
#define L_BCOL(ldomain, gj) ((int)((gj) / ldomain->B / ldomain->grid->Q))

#define BNUM(ldomain, gi) ((int)((gi) / ldomain->B))
#define L_BROW_START(ldomain, gi) ((int)(BNUM(ldomain, gi) / ldomain->grid->P) + (int)(BNUM(ldomain, gi) % ldomain->grid->P > ldomain->grid->myrow))
#define L_BCOL_START(ldomain, gj) ((int)(BNUM(ldomain, gj) / ldomain->grid->Q) + (int)(BNUM(ldomain, gj) % ldomain->grid->Q > ldomain->grid->mycol))

// get rank from global idx
#define P_FROM_G_ROW(ldomain, gi) (((int)((gi) / ldomain->B)) % ldomain->grid->P)
#define Q_FROM_G_COL(ldomain, gj) (((int)((gj) / ldomain->B)) % ldomain->grid->Q)
#define RANK_FROM_G_ROW_COL(ldomain, gi, gj) RANK_FROM_PQ(P_FROM_G_ROW(ldomain, gi), Q_FROM_G_COL(ldomain->gj))

#define BLK_ELE(data, B, i, j) (data[(i) * B + (j)])
#define BLK(ldomain, lbrow, lbcol) (ldomain->data + ((lbrow) * ldomain->NBQ + lbcol) * B * B)

#define MY_RANK (ldomain->grid->lrank)
#define MY_ROW (ldomain->grid->myrow)
#define MY_COL (ldomain->grid->mycol)

void init_local_domain(const double *globalA, int N, int B, const grid_t *grid, local_domain_t *ldomain);
void gather_local_domain(double *globalA, local_domain_t *ldomain);
void print_matrix(double *A, int n);
void print_global_matrix(local_domain_t *ldomain, char *msg);
#endif