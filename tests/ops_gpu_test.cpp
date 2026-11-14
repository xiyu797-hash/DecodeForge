#include "decodeforge/ops.hpp"
#include "decodeforge/core.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace decodeforge;
namespace {
int checks=0;
void compare(const std::vector<float>& ref,const std::vector<float>& actual,const std::string& name) {
    const auto stats=compare_outputs(ref,actual,0.01,0.007);
    if(stats.mismatched)throw std::runtime_error(name+": "+std::to_string(stats.mismatched)+" mismatches, max_abs="+std::to_string(stats.max_abs));
    ++checks;
}
}
int main(int argc,char**argv) {
    try {
        int device=0;
        if(argc==3&&std::string(argv[1])=="--device") device=std::stoi(argv[2]);
        else if(argc!=1)throw std::invalid_argument("usage: decodeforge_ops_gpu_tests [--device 0]");
        {
            auto in=make_gemm_input(2,3,4,DType::FP16,8);
            GemmSession gpu(in,device); bool rejected=false;
            try {(void)gpu.download();} catch(const std::logic_error&) {rejected=true;}
            if(!rejected) throw std::runtime_error("uninitialized GEMM output was downloadable");
            gpu.benchmark_ms(GemmKernel::Tiled,0,1);
            compare(gemm_cpu_reference(in),gpu.download(),"GEMM benchmark output");
        }
        {
            auto in=make_norm_input(2,3,9);
            NormSession gpu(in,device); bool rejected=false;
            try {(void)gpu.download();} catch(const std::logic_error&) {rejected=true;}
            if(!rejected) throw std::runtime_error("uninitialized Norm output was downloadable");
            gpu.benchmark_ms(NormKernel::Fused,0,1);
            compare(norm_cpu_reference(in),gpu.download(),"Norm benchmark output");
            // First TwoPass benchmark prepares its workspace, the next reuses it.
            for(int i=0;i<2;++i) {
                const float ms=gpu.benchmark_ms(NormKernel::TwoPass,0,1);
                if(!std::isfinite(ms) || ms<0) throw std::runtime_error("invalid Norm zero-warmup timing");
                compare(norm_cpu_reference(in),gpu.download(),"Norm TwoPass zero-warmup benchmark output");
            }
        }
        for(DType dt:{DType::FP16,DType::BF16}) for(int k:{1,15,16,17,63,127,129}) {
            auto in=make_gemm_input(17,33,k,dt,static_cast<std::uint64_t>(k*3+static_cast<int>(dt)));
            auto reference=gemm_cpu_reference(in);
            GemmSession gpu(in,device);
            for(GemmKernel kernel:{GemmKernel::Naive,GemmKernel::Tiled,GemmKernel::Cublas}) {
                gpu.run(kernel);
                compare(reference,gpu.download(),std::string(dtype_name(dt))+"/"+gemm_kernel_name(kernel));
            }
            std::cout<<"[OPS GPU] GEMM "<<dtype_name(dt)<<" M=17 N=33 K="<<k<<" PASS\n";
        }
        for(int cols:{1,15,16,17,31,32,33,128,257,1025,4096}) {
            auto in=make_norm_input(7,cols,cols*29);
            auto ref=norm_cpu_reference(in);
            NormSession gpu(in,device);
            for(NormKernel k:{NormKernel::Fused,NormKernel::TwoPass}) {
                gpu.run(k);compare(ref,gpu.download(),std::string("norm/")+norm_kernel_name(k));
                gpu.run(k);compare(ref,gpu.download(),std::string("norm repeated/")+norm_kernel_name(k));
            }
            std::cout<<"[OPS GPU] RMSNorm cols="<<cols<<" PASS\n";
        }
        std::cout<<"[OPS GPU] ALL PASS; checks="<<checks<<"\n";
        return EXIT_SUCCESS;
    }catch(const std::exception& e){std::cerr<<"[OPS GPU] FAIL: "<<e.what()<<"\n";return EXIT_FAILURE;}
}
