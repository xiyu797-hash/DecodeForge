#include "decodeforge/core.hpp"
#include "decodeforge/gpu.hpp"
#include "decodeforge/ops.hpp"
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace decodeforge;
namespace {
struct Opt {
    std::string command, csv;
    int m=256,n=256,k=256,rows=256,cols=1024,device=0,warmup=10,repeats=50;
    DType dtype=DType::FP16;
};
Opt parse(int argc,char** argv){
    if(argc<2)throw std::invalid_argument("usage: decodeforge_ops_cli gemm-check|gemm-bench|norm-check|norm-bench [options]");
    Opt o;o.command=argv[1];
    for(int i=2;i<argc;++i){
        const std::string key=argv[i];
        if(++i>=argc)throw std::invalid_argument("option value missing for "+key);
        const std::string value=argv[i];
        if(key=="--csv")o.csv=value;
        else if(key=="--dtype"){
            if(value=="fp16")o.dtype=DType::FP16;
            else if(value=="bf16")o.dtype=DType::BF16;
            else throw std::invalid_argument("dtype only fp16 or bf16");
        }else{
            std::size_t consumed=0;
            const int v=std::stoi(value,&consumed);
            if(consumed!=value.size()) throw std::invalid_argument("invalid numeric value for "+key);
            if(key=="--m")o.m=v;
            else if(key=="--n")o.n=v;
            else if(key=="--k")o.k=v;
            else if(key=="--rows")o.rows=v;
            else if(key=="--cols")o.cols=v;
            else if(key=="--device")o.device=v;
            else if(key=="--repeats")o.repeats=v;
            else if(key=="--warmup")o.warmup=v;
            else throw std::invalid_argument("unknown option "+key);
        }
    }
    return o;
}
void check_equal(const std::vector<float>& ref,const std::vector<float>& actual,const char* what){
    const auto e=compare_outputs(ref,actual,0.01,0.007);
    if(e.mismatched)throw std::runtime_error(std::string(what)+" failed, mismatches="+std::to_string(e.mismatched));
}
void csv_write(const std::string& file,const std::string& header,const std::string& data){
    if(file.empty())return;
    std::ofstream f(file,std::ios::trunc);
    if(!f)throw std::runtime_error("cannot open CSV: "+file);
    f<<header<<'\n'<<data<<'\n';
    if(!f)throw std::runtime_error("cannot write CSV: "+file);
    std::cout<<"[REPORT] "<<file<<'\n';
}
}
int main(int argc,char**argv){
    try{
        if(argc==2&&std::string(argv[1])=="--help"){
            std::cout<<"Usage: decodeforge_ops_cli gemm-check|gemm-bench --m 256 --n 256 --k 256 --dtype fp16|bf16\n"
                        "       decodeforge_ops_cli norm-check|norm-bench --rows 256 --cols 1024\n"
                        "       --device 0 --repeats 50 --warmup 10 --csv path.csv\n";
            return 0;
        }
        auto o=parse(argc,argv);
        const bool is_bench=o.command=="gemm-bench"||o.command=="norm-bench";
        if(o.command=="gemm-check"||o.command=="gemm-bench"){
            auto in=make_gemm_input(o.m,o.n,o.k,o.dtype,20261106);
            auto ref=gemm_cpu_reference(in);
            if(!cuda_compiled())throw std::runtime_error("not built with CUDA; GPU ops check not available");
            GemmSession gpu(in,o.device);
            for(auto kernel:{GemmKernel::Naive,GemmKernel::Tiled,GemmKernel::Cublas}){
                gpu.run(kernel);check_equal(ref,gpu.download(),gemm_kernel_name(kernel));
                std::cout<<"[GEMM] "<<gemm_kernel_name(kernel)<<" numeric PASS\n";
            }
            if(is_bench){
                const auto naive=gpu.benchmark_ms(GemmKernel::Naive,o.warmup,o.repeats);
                const auto tiled=gpu.benchmark_ms(GemmKernel::Tiled,o.warmup,o.repeats);
                const auto blas=gpu.benchmark_ms(GemmKernel::Cublas,o.warmup,o.repeats);
                std::cout<<std::setprecision(6)<<"GEMM="<<dtype_name(o.dtype)<<" M="<<o.m<<" N="<<o.n<<" K="<<o.k
                          <<" naive="<<naive<<"ms tiled="<<tiled<<"ms cublas="<<blas<<"ms\n";
                csv_write(o.csv,"dtype,m,n,k,naive_ms,tiled_ms,cublas_ms",
                          std::string(dtype_name(o.dtype))+","+std::to_string(o.m)+","+std::to_string(o.n)+","+std::to_string(o.k)+","+
                          std::to_string(naive)+","+std::to_string(tiled)+","+std::to_string(blas));
            }
        }else if(o.command=="norm-check"||o.command=="norm-bench"){
            auto in=make_norm_input(o.rows,o.cols,20261106);
            auto ref=norm_cpu_reference(in);
            if(!cuda_compiled())throw std::runtime_error("not built with CUDA; GPU ops check not available");
            NormSession gpu(in,o.device);
            for(auto kernel:{NormKernel::Fused,NormKernel::TwoPass}){
                gpu.run(kernel);check_equal(ref,gpu.download(),norm_kernel_name(kernel));
                std::cout<<"[RMSNORM] "<<norm_kernel_name(kernel)<<" numeric PASS\n";
            }
            if(is_bench){
                const auto fused=gpu.benchmark_ms(NormKernel::Fused,o.warmup,o.repeats);
                const auto two=gpu.benchmark_ms(NormKernel::TwoPass,o.warmup,o.repeats);
                std::cout<<std::setprecision(6)<<"RMSNorm rows="<<o.rows<<" cols="<<o.cols
                          <<" fused="<<fused<<"ms two_pass="<<two<<"ms\n";
                csv_write(o.csv,"rows,cols,fused_ms,two_pass_ms",
                          std::to_string(o.rows)+","+std::to_string(o.cols)+","+
                          std::to_string(fused)+","+std::to_string(two));
            }
        }else throw std::invalid_argument("unknown command: "+o.command);
        std::cout<<"[OPS] ALL ENABLED CHECKS PASS\n";return EXIT_SUCCESS;
    }catch(const std::exception&e){std::cerr<<"[OPS] FAIL: "<<e.what()<<'\n';return EXIT_FAILURE;}
}
