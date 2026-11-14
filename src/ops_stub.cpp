#include "decodeforge/ops.hpp"
#include <stdexcept>
namespace decodeforge {
struct GemmSession::Impl {};
struct NormSession::Impl {};
GemmSession::GemmSession(const GemmInput&,int){ throw std::runtime_error("CUDA backend not built"); }
GemmSession::~GemmSession()=default;
void GemmSession::run(GemmKernel){throw std::runtime_error("CUDA backend not built");}
std::vector<float> GemmSession::download(){throw std::runtime_error("CUDA backend not built");}
float GemmSession::benchmark_ms(GemmKernel,int,int){throw std::runtime_error("CUDA backend not built");}
NormSession::NormSession(const NormInput&,int){throw std::runtime_error("CUDA backend not built");}
NormSession::~NormSession()=default;
void NormSession::run(NormKernel){throw std::runtime_error("CUDA backend not built");}
std::vector<float> NormSession::download(){throw std::runtime_error("CUDA backend not built");}
float NormSession::benchmark_ms(NormKernel,int,int){throw std::runtime_error("CUDA backend not built");}
}
