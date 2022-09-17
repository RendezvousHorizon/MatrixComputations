function cl_fp32() {
    clang++ main.cc src/cl_interface.cc -o cl_fp32.exe -I include -DCL_FP32 -DCL_HPP_TARGET_OPENCL_VERSION=110 -DCL_HPP_MINIMUM_OPENCL_VERSION=100 -std=c++0x -DCL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY -framework OpenCL
}

function neon_fp32() {
    clang++ main.cc src/**impl.cc -o neon_fp32.exe -I include -I src -std=c++11 -DNEON_FP32
}
$1 ""