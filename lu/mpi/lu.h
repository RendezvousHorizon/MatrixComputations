#ifndef LU_H
#define LU_H

#include <mpi.h>
#include <float.h>
#include <string.h>
#include "local_domain.h"
#include "pivot.h"

typedef int coord_t[2];

void luf_pivoted(local_domain_t *ldomain, pivot_final_t *pivot);

#endif