#ifndef __IMPL_H
#define __IMPL_H

class NeonFP32Impl {
public:
    void operator()(int m, int n, int k, float *a, float *b, float *c);
};

class NaiveFP32Impl {
public:
    void operator()(int m, int n, int k, float *a, float *b, float *c);   
};

class BlockFP32Impl {
public:
    void operator()(int m, int n, int k, float *a, float *b, float *c);     
};
#endif