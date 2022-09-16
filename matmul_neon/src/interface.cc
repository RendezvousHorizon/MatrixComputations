#include "interface.h"

template <typename T, typename Impl>
Interface<T, Impl>::Interface(int m, int n, int k): _m(m), _n(n), _k(k) {
    _a = _b = _c = nullptr;
}

template <typename T, typename Impl>
Interface<T, Impl>::~Interface() {
    if (_a) free(_a);
    if (_b) free(_b);
    if (_c) free(_c);
}

template <typename T>
static void rand_init(T *a, int n) {
    for (int i = 0; i < n; i++)
        a[i] = static_cast<T>(rand() / RAND_MAX - 0.5);
}
template <typename T, typename Impl>
void Interface<T, Impl>::init() {
    _a = new T[_m * _k];
    _b = new T[_k * _n];
    _c = new T[_m * _n];
    rand_init<T>(_a, _m * _k);
    rand_init<T>(_b, _k * _n);
    validate_impl();
}

template <typename T, typename Impl>
void Interface<T, Impl>::naive_matmul(T* a, T *b, T *c) {
    for (int i = 0; i < _m; i++)
        for (int j = 0; j < _n; j++)
            for (int p = 0; p < _k; p++)
                _c[i * _n + j] += _a[i + p * _m] * _b[p * _n + j];
}

template <typename T, typename Impl>
void Interface<T, Impl>::validate_impl() {
    T *d = new T[_m * _n];
    memset(d, 0, _m * _n * sizeof(T));
    memset(_c, 0, _m * _n * sizeof(T));
    naive_matmul(_a, _b, _c);
    Impl()(_m, _n, _k, _a, _b, d);

    for (int i = 0; i < _m; i++)
        for (int j = 0; j < _n; j++)
            if (ABS(_c[i * _n + j] - d[i * _n + j]) > 1e-3) {
                std::cout << "Validation failed. Exit.\n";
                exit(-1);
            }

    delete []d;
}