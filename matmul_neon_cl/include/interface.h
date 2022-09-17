#ifndef __INTERFACE_H
#define __INTERFACE_H

#include <iostream>
#include <string>
#include <cstdlib>
#define ABS(a) ((a) > 0 ? (a) : -(a))

template <typename T, typename Impl>
class Interface {
public:
    Interface(int m, int n, int k);
    ~Interface();
    void init();
    void run_once() {
        Impl()(_m, _n, _k, _a, _b, _c);
    };
    void validate_impl();

private:
    void naive_matmul(T *a, T *b, T *c);

private:
    int _m;
    int _n;
    int _k;
    T *_a; // m * k matrix, col major
    T *_b; // k * n matrix, row major
    T *_c; // m * n matrix, row major
};

#include "interface.cc" // avoid template class function link error
#endif