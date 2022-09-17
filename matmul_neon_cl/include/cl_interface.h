#ifndef __CL_INTERFACE_H
#define __CL_INTERFACE_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdio>
#include "cl.hpp"

#define ABS(a) ((a) > 0 ? (a) : -(a))

class CLInterface{
public:
    CLInterface(int m, int n, int k);
    ~CLInterface();
    void init();
    void run_once();
    void validate_impl();
private: 
    void naive_matmul(float *a, float *b, float *c);

private:
    int _m;
    int _n;
    int _k;
    float *_a;
    float *_b;
    float *_c;
    cl::Program _program;    // The program that will run on the device.    
    cl::Context _context;    // The context which holds the device.    
    cl::Device _device;      // The device where the kernel will run.
};

#endif