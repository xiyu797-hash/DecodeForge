#include "decodeforge/ops.hpp"
#include "decodeforge/bfloat16.hpp"
#include "decodeforge/half.hpp"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace decodeforge;
namespace {
int checks=0;
void ok(bool condition,const char* message) {
    ++checks;if(!condition) throw std::runtime_error(message);
}
template<typename F>void should_throw(F f) {bool fail=false;try{f();}catch(const std::exception&){fail=true;}ok(fail,"expected invalid input to throw");}
void test_bfloat16() {
    for(std::uint32_t raw=0;raw<65536;++raw) {
        const auto x=static_cast<std::uint16_t>(raw);
        const auto y=float_to_bfloat16(bfloat16_to_float(x));
        if((raw & 0x7f80u)==0x7f80u && (raw&0x007fu))
            ok((y&0x7f80u)==0x7f80u && (y&0x007fu)!=0,"bf16 NaN must remain NaN");
        else ok(x==y,"bf16 round-trip");
    }
    ok(float_to_bfloat16(1.f)==0x3f80u,"bf16 one");
    ok(float_to_bfloat16(-1.f)==0xbf80u,"bf16 negative one");
    ok(float_to_bfloat16(1.f+1.f/256.f)==0x3f80u,"bf16 tie-even");
}
void test_gemm_golden(DType type) {
    GemmInput in;
    in.m=2;in.n=2;in.k=3;in.dtype=type;
    for(float v:{1.f,2.f,3.f,4.f,5.f,6.f})in.a.push_back(encode_scalar(v,type));
    for(float v:{7.f,8.f,9.f,10.f,11.f,12.f})in.b.push_back(encode_scalar(v,type));
    auto y=gemm_cpu_reference(in);
    ok(y.size()==4,"gemm size");
    ok(y[0]==58.f && y[1]==64.f && y[2]==139.f && y[3]==154.f,"gemm golden");
}
void test_shapes() {
    for(DType type:{DType::FP16,DType::BF16}) for(int m:{1,3,13,17})
        for(int n:{1,5,16,33})for(int k:{1,7,16,17,33}) {
            auto in=make_gemm_input(m,n,k,type,static_cast<std::uint64_t>(m*31+n*71+k));
            const auto y=gemm_cpu_reference(in);
            ok(y.size()==static_cast<std::size_t>(m)*n,"gemm shape");
            for(float x:y)ok(std::isfinite(x),"gemm finite");
            auto in2=make_gemm_input(m,n,k,type,static_cast<std::uint64_t>(m*31+n*71+k));
            ok(in.a==in2.a && in.b==in2.b,"gemm deterministic seed");
        }
    for(int row:{1,3,31})for(int col:{1,2,3,16,17,63,64,129,1025}) {
        const auto in=make_norm_input(row,col,static_cast<std::uint64_t>(row*500+col));
        auto y=norm_cpu_reference(in);
        ok(y.size()==static_cast<std::size_t>(row)*col,"norm shape");
        for(float v:y)ok(std::isfinite(v),"norm finite");
    }
}
void test_norm_golden() {
    NormInput in;
    in.rows=1;in.cols=2;in.epsilon=1e-5f;
    in.x={float_to_half(3.f),float_to_half(4.f)};
    in.residual={float_to_half(0.f),float_to_half(0.f)};
    in.gamma={float_to_half(1.f),float_to_half(2.f)};
    auto out=norm_cpu_reference(in);
    const double inv=1.0/std::sqrt(12.5+1e-5);
    ok(std::abs(out[0]-3*inv)<1e-6,"norm golden first");
    ok(std::abs(out[1]-8*inv)<1e-6,"norm golden second");
    in.epsilon=0;
    should_throw([&]{in.validate();});
    in.epsilon=1e-5f;
    in.gamma.clear();
    should_throw([&]{in.validate();});
    should_throw([&]{make_norm_input(0,1,1);});
    should_throw([&]{make_gemm_input(-1,1,1,DType::FP16,1);});
}
}
int main() {
    try {
        test_bfloat16();std::cout<<"[PASS] BF16 exhaustive 65,536 bit-pattern roundtrip\n";
        test_gemm_golden(DType::FP16);
        test_gemm_golden(DType::BF16);
        std::cout<<"[PASS] manual FP16/BF16 GEMM golden values\n";
        test_shapes();std::cout<<"[PASS] 160 GEMM shapes and 27 RMSNorm shapes\n";
        test_norm_golden();std::cout<<"[PASS] RMSNorm manual golden and invalid-argument guards\n";
        std::cout<<"[OPS CPU] ALL PASS; assertions="<<checks<<"\n";
        return EXIT_SUCCESS;
    }catch(const std::exception& e) {std::cerr<<"[OPS CPU] FAIL: "<<e.what()<<"\n";return EXIT_FAILURE;}
}
