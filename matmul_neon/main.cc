#include <iostream>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <ctime>
#include <cstdlib>

#include "interface.h"
#include "impl.h"

#define MIN(a, b) (a < b ? a : b)
int M = 1024;
int N = 1024;
int K = 1024;
const double MAX_SECONDS = 10;
int REPEAT = 100;

template <typename T, typename Impl>
int get_repeat(Interface<T, Impl> &interface) {
    auto start = std::chrono::high_resolution_clock::now();
    interface.run_once();
    auto finish = std::chrono::high_resolution_clock::now();
    double dur = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
    return MIN((int)(MAX_SECONDS * 1e6 / dur), REPEAT);
}

int main()
{
    srand(time(NULL));
    #if defined(NEON_FP32)
        Interface<float, NeonFP32Impl> interface(M, N, K);
    #elif defined(BLOCK_FP32)
        Interface<float, BlockFP32Impl> interface(M, N, K);
    #else 
        Interface<float, NaiveFP32Impl> interface(M, N, K);
    #endif
    
    interface.init();
    interface.validate_impl();
    REPEAT = get_repeat(interface);
    printf("Num runs=%d\n", REPEAT);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < REPEAT; i++)
        interface.run_once();
    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> fs = finish - start;
    double dur = std::chrono::duration_cast<std::chrono::microseconds>(fs).count() * 1.0 / REPEAT;
    double mflops = 2. * M * N * K / dur;
    printf("Duration: %.2fus, MFLOPs: %.2f.\n", dur, mflops);
    return 0;

}