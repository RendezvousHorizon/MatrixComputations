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

#define PANEL_L_BCOL(ldomain, bk) (int(bk / ldomain->grid->Q))
#define PANEL_L_BROW_START(ldomain, bk) (int(bk / ldomain->grid->P) + int(bk % ldomain->grid->P > ldomain->grid->myrow))

// get global block idx or element idx from ldomain idx in rank (p, q)
#define G_BROW_P(ldomain, lbrow, p) (lbrow * ldomain->grid->P + p)
#define G_BCOL_Q(ldmain, lbcol, q) (lbcol * ldomain->grid->Q + q)
#define G_ROW_P(ldomain, lbrow, p, i) (G_BROW_P(ldomain, lbrow, p) * ldomain->B + i)
#define G_COL_Q(ldomain, lbcol, q, j) (G_BCOL_Q(ldomain, lbcol, q) * ldomain->B + j)

// get global block idx or element idx from ldomain idx in rank self
#define G_BROW(ldomain, lbrow) G_BROW_P(ldomain, lbrow, ldomain->grid->myrow)
#define G_BCOL(ldomain, lbcol) G_BROW_Q(ldomain, lbcol, ldomain->grid->mycol)
#define G_ROW(ldomain, lbrow, i) G_ROW_P(ldomain, lbrow, ldomain->grid->myrow, i)
#define G_COL(ldomain, lbcol, j) G_COL_Q(ldomain, lbcol, ldomain->grid->mycol, j)

#define BLK_ELE(data, B, i, j) (data[i * B + j])
#define BLK(ldomain, lbrow, lbcol) (ldomain->data + (lbrow * ldomain->NBQ + lbcol) * B * B)


void init_local_domain(const double *globalA, int N, int B, const grid_t *grid, local_domain_t *ldomain);

#endif