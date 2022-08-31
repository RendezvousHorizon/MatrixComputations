#ifndef PIVOT_H
#define PIVOT_H

#include <string.h>
#include <assert.h>
#include "local_domain.h"

typedef struct pivot_s
{
    int *data;
    int n;
}pivot_t;

typedef pivot_t pivot_swap_t;
typedef pivot_t pivot_final_t;

#define CEIL_DIV(x, m) ((int)((x + m - 1) / (m)))
#define CEIL(x, m)(CEIL_DIV(x, m) * (m))
#define MIN(a, b) ((a < b) ? (a) : (b))
#define ABS(a) ((a > 0) ? (a) : -(a))
#define ABS_GT(a, b) (ABS(a) > ABS(b))

void init_pivot(pivot_t *pivot, int n);
void free_pivot(pivot_t *pivot);
void swap_row(local_domain_t *ldomain, int grow1, int grow2, int gj_start, int gj_end);
void permute_ldomain(local_domain_t *ldomain, pivot_swap_t *pivot, int gi_start, int gj_start, int gj_end);
pivot_final_t *convert_swap_to_final_t(pivot_swap_t *pivot);
void pivot_apply_swap_to_final(pivot_final_t *final_pivot, pivot_swap_t *swap_pivot);
void pivot_apply_final_to_final(pivot_final_t *final_pivot, pivot_final_t *sub_pivot);
#endif