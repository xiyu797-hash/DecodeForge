#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace decodeforge {
enum class DType { FP16, BF16 };
const char* dtype_name(DType type) noexcept;
float decode_scalar(std::uint16_t bits,DType type) noexcept;
std::uint16_t encode_scalar(float x,DType type) noexcept;

// GEMM: A(M×K)×B(K×N)→FP32 C(M×N)，全部行优先。
struct GemmInput {
    int m=0,n=0,k=0;
    DType dtype=DType::FP16;
    std::vector<std::uint16_t> a;
    std::vector<std::uint16_t> b;
    void validate() const;
};
GemmInput make_gemm_input(int m,int n,int k,DType dtype,std::uint64_t seed);
std::vector<float> gemm_cpu_reference(const GemmInput& in);
enum class GemmKernel { Naive, Tiled, Cublas };
const char* gemm_kernel_name(GemmKernel k) noexcept;
class GemmSession {
public:
    explicit GemmSession(const GemmInput& input,int device=0);
    ~GemmSession();
    GemmSession(const GemmSession&)=delete;
    GemmSession& operator=(const GemmSession&)=delete;
    void run(GemmKernel kernel);
    std::vector<float> download();
    float benchmark_ms(GemmKernel kernel,int warmup=10,int repetitions=100);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Residual+RMSNorm: y=(x+residual) * rsqrt(mean((x+residual)^2)+eps) * gamma。
// 输入与gamma为FP16，累加与输出FP32；并非FP16输出的完整LLM推理层。
struct NormInput {
    int rows=0,cols=0;
    float epsilon=1.e-5f;
    std::vector<std::uint16_t> x;
    std::vector<std::uint16_t> residual;
    std::vector<std::uint16_t> gamma;
    void validate() const;
};
NormInput make_norm_input(int rows,int cols,std::uint64_t seed);
std::vector<float> norm_cpu_reference(const NormInput& in);
enum class NormKernel { Fused, TwoPass };
const char* norm_kernel_name(NormKernel k) noexcept;
class NormSession {
public:
    explicit NormSession(const NormInput& input,int device=0);
    ~NormSession();
    NormSession(const NormSession&)=delete;
    NormSession& operator=(const NormSession&)=delete;
    void run(NormKernel kernel);
    std::vector<float> download();
    float benchmark_ms(NormKernel kernel,int warmup=10,int repetitions=100);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace decodeforge
