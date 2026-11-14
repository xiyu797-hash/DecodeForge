#include "decodeforge/gpu.hpp"
#include <stdexcept>
namespace decodeforge {
const char* kernel_name(Kernel k) noexcept {
    switch(k) { case Kernel::Direct: return "direct"; case Kernel::Grouped: return "grouped_paired"; case Kernel::GroupedScalar: return "grouped_scalar"; case Kernel::TwoPassCublas: return "two_pass_cublas"; }
    return "unknown";
}
bool cuda_compiled() noexcept { return false; }
int visible_gpu_count() { return 0; }
std::string gpu_description(int) { return "CUDA backend not built (configure -DDF_ENABLE_CUDA=ON)"; }
struct GpuSession::Impl {};
GpuSession::GpuSession(const Inputs&, int) { throw std::runtime_error("CUDA backend not built; never reports fake GPU results"); }
GpuSession::~GpuSession() = default;
void GpuSession::run(Kernel) { throw std::runtime_error("CUDA not built"); }
Kernel GpuSession::autotune(int,int) { throw std::runtime_error("CUDA not built"); }
Kernel GpuSession::autotune(int,int,AutotuneTrace&) { throw std::runtime_error("CUDA not built"); }
void GpuSession::run_auto() { throw std::runtime_error("CUDA not built"); }
std::vector<float> GpuSession::download() { throw std::runtime_error("CUDA not built"); }
float GpuSession::benchmark_ms(Kernel,int,int) { throw std::runtime_error("CUDA not built"); }
void GpuSession::capture_graph(Kernel) { throw std::runtime_error("CUDA not built"); }
void GpuSession::launch_graph(int) { throw std::runtime_error("CUDA not built"); }
float GpuSession::benchmark_graph_ms(int,int) { throw std::runtime_error("CUDA not built"); }
}
