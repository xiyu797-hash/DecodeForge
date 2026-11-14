#pragma once
#include "decodeforge/core.hpp"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace decodeforge {

enum class Kernel { Direct, Grouped, GroupedScalar, TwoPassCublas };
const char* kernel_name(Kernel k) noexcept;
struct AutotuneSample { float direct_ms=0, grouped_ms=0; bool direct_first=false; };
struct AutotuneTrace {
    std::array<AutotuneSample,9> samples{};
    int requested_warmup=0, requested_repetitions=0;
    int actual_warmup=0, actual_repetitions=0;
    Kernel selected=Kernel::Direct;
    double wall_ms=0;
};

// CUDA不可用时，所有GPU入口明确抛出异常；不会以CPU运算伪装GPU通过。
bool cuda_compiled() noexcept;
int visible_gpu_count();
std::string gpu_description(int device = 0);

class GpuSession {
public:
    explicit GpuSession(const Inputs& input, int device = 0);
    ~GpuSession();
    GpuSession(const GpuSession&) = delete;
    GpuSession& operator=(const GpuSession&) = delete;
    GpuSession(GpuSession&&) = delete;
    GpuSession& operator=(GpuSession&&) = delete;

    // Grouped uses packed-pair INT4 decoding; GroupedScalar preserves the
    // original scalar-nibble algorithm for controlled ablation experiments.
    // The autotuner chooses only Direct vs Grouped (not the scalar baseline).
    void run(Kernel kernel);
    // 每个GpuSession独立缓存；交替测量9组并比较配对耗时比中位数。
    // warmup 和 repetitions 为下限，分别至少执行10次和50次。
    Kernel autotune(int warmup = 10, int repetitions = 50);
    // Explicit diagnostic overload. Trace is published only after success;
    // this records the actual internal measurements, not a later benchmark.
    Kernel autotune(int warmup,int repetitions,AutotuneTrace& trace);
    void run_auto();
    std::vector<float> download();
    // 在CUDA stream上以Event计时；排除CPU生成数据、上传和下载。
    float benchmark_ms(Kernel kernel, int warmup = 10, int repetitions = 100);
    // CUDA Graph仅对固定Shape、固定Device指针的Fused Kernel捕获。
    void capture_graph(Kernel kernel);
    void launch_graph(int repetitions = 1);
    float benchmark_graph_ms(int warmup = 10, int repetitions = 100);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    template<bool Capture> Kernel autotune_impl(int warmup,int repetitions,AutotuneTrace* trace);
};

} // namespace decodeforge
