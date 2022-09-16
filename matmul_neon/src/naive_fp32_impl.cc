#include "impl.h"

void NaiveFP32Impl::operator()(int m, int n, int k, float *a, float *b, float *c) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int p = 0; p < k; p++)
                c[i * n + j] += a[i + p * m] * b[p * n + j];
}