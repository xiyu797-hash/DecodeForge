#include "decodeforge/ops.hpp"
#include "decodeforge/half.hpp"
#include "decodeforge/bfloat16.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace decodeforge {
const char* dtype_name(DType type) noexcept { return type==DType::FP16?"fp16":"bf16"; }
float decode_scalar(std::uint16_t bits,DType dtype) noexcept {
    return dtype==DType::FP16?half_to_float(bits):bfloat16_to_float(bits);
}
std::uint16_t encode_scalar(float x,DType dtype) noexcept {
    return dtype==DType::FP16?float_to_half(x):float_to_bfloat16(x);
}
void GemmInput::validate() const {
    if(m<1||n<1||k<1||m>16384||n>16384||k>16384) throw std::invalid_argument("GEMM dimensions must be 1..16384");
    if(dtype!=DType::FP16 && dtype!=DType::BF16) throw std::invalid_argument("invalid GEMM dtype");
    if(a.size()!=static_cast<std::size_t>(m)*static_cast<std::size_t>(k) || b.size()!=static_cast<std::size_t>(k)*static_cast<std::size_t>(n))
        throw std::invalid_argument("GEMM buffer length mismatch");
    for(auto v:a) if(!std::isfinite(decode_scalar(v,dtype))) throw std::invalid_argument("GEMM A contains Inf or NaN");
    for(auto v:b) if(!std::isfinite(decode_scalar(v,dtype))) throw std::invalid_argument("GEMM B contains Inf or NaN");
}
GemmInput make_gemm_input(int m,int n,int k,DType dtype,std::uint64_t seed) {
    if(m<1||n<1||k<1||m>16384||n>16384||k>16384) throw std::invalid_argument("invalid GEMM shape");
    if(dtype!=DType::FP16&&dtype!=DType::BF16) throw std::invalid_argument("invalid dtype");
    GemmInput in; in.m=m;in.n=n;in.k=k;in.dtype=dtype;
    in.a.resize(static_cast<std::size_t>(m)*static_cast<std::size_t>(k));
    in.b.resize(static_cast<std::size_t>(k)*static_cast<std::size_t>(n));
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> u(-0.25f,0.25f);
    for(auto& x:in.a) x=encode_scalar(u(rng),dtype);
    for(auto& x:in.b) x=encode_scalar(u(rng),dtype);
    in.validate();
    return in;
}
std::vector<float> gemm_cpu_reference(const GemmInput& in) {
    in.validate();
    std::vector<float> output(static_cast<std::size_t>(in.m)*static_cast<std::size_t>(in.n));
    for(int row=0;row<in.m;++row) for(int col=0;col<in.n;++col) {
        double sum=0;
        for(int kk=0;kk<in.k;++kk) {
            const double a=decode_scalar(in.a[static_cast<std::size_t>(row)*static_cast<std::size_t>(in.k)+static_cast<std::size_t>(kk)],in.dtype);
            const double b=decode_scalar(in.b[static_cast<std::size_t>(kk)*static_cast<std::size_t>(in.n)+static_cast<std::size_t>(col)],in.dtype);
            sum+=a*b;
        }
        output[static_cast<std::size_t>(row)*static_cast<std::size_t>(in.n)+static_cast<std::size_t>(col)]=static_cast<float>(sum);
    }
    return output;
}
void NormInput::validate() const {
    if(rows<1||cols<1||rows>16384||cols>16384) throw std::invalid_argument("RMSNorm shape must be 1..16384");
    if(!(epsilon>0.f) || !std::isfinite(epsilon)) throw std::invalid_argument("RMSNorm epsilon must be finite positive");
    const auto count=static_cast<std::size_t>(rows)*static_cast<std::size_t>(cols);
    if(x.size()!=count || residual.size()!=count || gamma.size()!=static_cast<std::size_t>(cols))
        throw std::invalid_argument("RMSNorm buffer length mismatch");
    for(auto h:x) if(!std::isfinite(half_to_float(h))) throw std::invalid_argument("RMSNorm x Inf/NaN");
    for(auto h:residual) if(!std::isfinite(half_to_float(h))) throw std::invalid_argument("RMSNorm residual Inf/NaN");
    for(auto h:gamma) if(!std::isfinite(half_to_float(h))) throw std::invalid_argument("RMSNorm gamma Inf/NaN");
}
NormInput make_norm_input(int rows,int cols,std::uint64_t seed) {
    if(rows<1||cols<1||rows>16384||cols>16384) throw std::invalid_argument("invalid RMSNorm shape");
    NormInput in;in.rows=rows;in.cols=cols;
    in.x.resize(static_cast<std::size_t>(rows)*static_cast<std::size_t>(cols));
    in.residual.resize(static_cast<std::size_t>(rows)*static_cast<std::size_t>(cols));
    in.gamma.resize(static_cast<std::size_t>(cols));
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> u(-1.f,1.f);
    for(auto& h:in.x) h=float_to_half(u(rng));
    for(auto& h:in.residual) h=float_to_half(u(rng));
    for(auto& h:in.gamma) h=float_to_half(1.f+0.1f*u(rng));
    in.validate();return in;
}
std::vector<float> norm_cpu_reference(const NormInput& in) {
    in.validate();
    std::vector<float> result(static_cast<std::size_t>(in.rows)*static_cast<std::size_t>(in.cols));
    for(int row=0;row<in.rows;++row) {
        double sum=0;
        for(int col=0;col<in.cols;++col) {
            const auto idx=static_cast<std::size_t>(row)*static_cast<std::size_t>(in.cols)+static_cast<std::size_t>(col);
            const double v=half_to_float(in.x[idx])+half_to_float(in.residual[idx]);
            sum+=v*v;
        }
        const double inv=1.0/std::sqrt(sum/in.cols+static_cast<double>(in.epsilon));
        for(int col=0;col<in.cols;++col) {
            const auto idx=static_cast<std::size_t>(row)*static_cast<std::size_t>(in.cols)+static_cast<std::size_t>(col);
            const double v=half_to_float(in.x[idx])+half_to_float(in.residual[idx]);
            result[idx]=static_cast<float>(v*inv*half_to_float(in.gamma[static_cast<std::size_t>(col)]));
        }
    }
    return result;
}
const char* gemm_kernel_name(GemmKernel k) noexcept {
    switch(k) {case GemmKernel::Naive:return "naive";case GemmKernel::Tiled:return "shared_mem_tiled";case GemmKernel::Cublas:return "cublas";}
    return "unknown";
}
const char* norm_kernel_name(NormKernel k) noexcept {
    switch(k) {case NormKernel::Fused:return "fused";case NormKernel::TwoPass:return "two_pass";}
    return "unknown";
}
}
