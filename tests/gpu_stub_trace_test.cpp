#include "decodeforge/gpu.hpp"
#include <exception>
#include <iostream>
#include <stdexcept>
using namespace decodeforge;
int main() {
    // Taking the overload address forces CPU-only linkage without constructing
    // an invalid GpuSession or claiming any CUDA work occurred.
    auto volatile overload=static_cast<Kernel (GpuSession::*)(int,int,AutotuneTrace&)>(&GpuSession::autotune);
    if(overload==nullptr || cuda_compiled()) return 1;
    try {
        GpuSession gpu(make_random_inputs({1,1,1,16},1));
    } catch(const std::runtime_error&) {
        std::cout<<"CPU-only GPU stub rejected session construction\n";
        return 0;
    }
    return 1;
}
