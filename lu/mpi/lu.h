#ifndef LU_H
#define LU_H

#include <mpi.h>
#include <float.h>
#include "grid.h"
#include "local_domain.h"

typedef int coord_t[2];

typedef struct ludata_s
{
    grid_t *grid;
    local_domain_t *ldomain;
}ludata_t;


void luf_pivoted(ludata_t *A, pivot_t *P);

#endif