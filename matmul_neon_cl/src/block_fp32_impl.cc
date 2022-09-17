#include "impl.h"

void BlockFP32Impl::operator()(int m, int n, int k, float *a, float *b, float *c) {
    int bm = 32;
    int bn = 32;
    for (int i = 0; i < m; i+=bm)
        for (int j = 0; j < n; j+=bn)
            for (int p = 0; p < k; p++)
                for (int ii = i; ii < i + bm; ii++)
                    for (int jj = j; jj < j + bn; jj++)
                        c[ii * n + jj] += a[ii + p * m] * b[p * n + jj];
}