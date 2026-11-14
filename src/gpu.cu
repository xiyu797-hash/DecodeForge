#include "decodeforge/gpu.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace decodeforge {
namespace {
void cuda_check(cudaError_t status, const char* expr, int line) {
    if (status != cudaSuccess) {
        std::ostringstream ss;
        ss << "CUDA " << expr << " at line " << line << ": " << cudaGetErrorString(status);
        throw std::runtime_error(ss.str());
    }
}
void blas_check(cublasStatus_t status, const char* expr, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream ss;
        ss << "cuBLAS " << expr << " at line " << line << ": status " << static_cast<int>(status);
        throw std::runtime_error(ss.str());
    }
}
#define DF_CUDA(x) cuda_check((x), #x, __LINE__)
#define DF_BLAS(x) blas_check((x), #x, __LINE__)

__device__ __forceinline__ int extract_s4(const unsigned char byte, int k) {
    const int v = (k & 1) ? (byte >> 4) & 15 : byte & 15;
    return v >= 8 ? v - 16 : v;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// 每个CTA计算一个 (batch, output_row)，CTA独立读取权重。
// B=1/2时有较多CTA可供并行；这是独立候选，并非对所有Shape最佳。
template<bool UseShift>
__global__ void fused_direct_kernel(const unsigned char* __restrict__ q,
                                     const half* __restrict__ scales,
                                     const half* __restrict__ activation,
                                     float* __restrict__ out,
                                     int n, int k, int batch, int group, int group_shift, int groups, int row_bytes) {
    const int row = static_cast<int>(blockIdx.x) % n;
    const int b = static_cast<int>(blockIdx.x) / n;
    if (b >= batch) return;
    const int tid = static_cast<int>(threadIdx.x);
    float accum = 0.0f;
    for (int col = tid; col < k; col += static_cast<int>(blockDim.x)) {
        const unsigned char data = q[static_cast<std::size_t>(row) * row_bytes + col / 2];
        int scale_col;
        if constexpr (UseShift) scale_col = col >> group_shift;
        else scale_col = col / group;
        const float scale = __half2float(scales[static_cast<std::size_t>(row) * groups + scale_col]);
        const float act = __half2float(activation[static_cast<std::size_t>(b) * k + col]);
        accum = fmaf(static_cast<float>(extract_s4(data, col)) * scale, act, accum);
    }
    accum = warp_sum(accum);
    __shared__ float partial[4]; // 固定blockDim=128 -> 4 warps
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) partial[warp] = accum;
    __syncthreads();
    if (warp == 0) {
        float v = lane < 4 ? partial[lane] : 0.f;
        v = warp_sum(v);
        if (lane == 0) out[static_cast<std::size_t>(b) * n + row] = v;
    }
}

// Ablation baseline: original scalar-nibble grouped kernel; this is not autotuned.
// It permits an apples-to-apples performance comparison with paired decoding.
template<int B>
__global__ void fused_grouped_scalar_kernel(const unsigned char* __restrict__ q,
                                            const half* __restrict__ scales,
                                            const half* __restrict__ activation,
                                            float* __restrict__ out,
                                            int n, int k, int group, int groups, int row_bytes) {
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    float acc[B];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.f;
    for (int col = tid; col < k; col += static_cast<int>(blockDim.x)) {
        const unsigned char data = q[static_cast<std::size_t>(row) * row_bytes + col / 2];
        const float w = static_cast<float>(extract_s4(data, col)) *
                        __half2float(scales[static_cast<std::size_t>(row) * groups + col / group]);
#pragma unroll
        for (int b = 0; b < B; ++b)
            acc[b] = fmaf(w, __half2float(activation[static_cast<std::size_t>(b) * k + col]), acc[b]);
    }
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = warp_sum(acc[b]);
    __shared__ float partial[B][4];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
#pragma unroll
        for (int b = 0; b < B; ++b) partial[b][warp] = acc[b];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int b = 0; b < B; ++b) {
            float v = lane < 4 ? partial[b][lane] : 0.f;
            v = warp_sum(v);
            if (lane == 0) out[static_cast<std::size_t>(b) * n + row] = v;
        }
    }
}

// 每个CTA计算同一output_row的所有B个输出，复用每次量化权重读取。
// B取编译期常量1/2/4/8，避免动态局部数组导致的显存溢出。
template<int B>
__global__ void fused_grouped_kernel(const unsigned char* __restrict__ q,
                                      const half* __restrict__ scales,
                                      const half* __restrict__ activation,
                                      float* __restrict__ out,
                                      int n, int k, int group, int groups, int row_bytes) {
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    float acc[B];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.f;
    // Pairwise INT4 decode: one packed byte supplies two weights.
    // All supported group sizes are even; pairs do not cross group boundaries.
    for (int col = tid * 2; col < k; col += static_cast<int>(blockDim.x) * 2) {

        const std::size_t packed_idx =
            static_cast<std::size_t>(row) * row_bytes + col / 2;

        const unsigned char data = q[packed_idx];

        const float scale = __half2float(
            scales[static_cast<std::size_t>(row) * groups + col / group]
        );

        const float w0 =
            static_cast<float>(extract_s4(data, col)) * scale;

#pragma unroll
        for (int b = 0; b < B; ++b) {
            const float a0 = __half2float(
                activation[static_cast<std::size_t>(b) * k + col]
            );

            acc[b] = fmaf(w0, a0, acc[b]);
        }

        // Handle odd K without reading beyond the input.
        if (col + 1 < k) {

            const float w1 =
                static_cast<float>(extract_s4(data, col + 1)) * scale;

#pragma unroll
            for (int b = 0; b < B; ++b) {
                const float a1 = __half2float(
                    activation[static_cast<std::size_t>(b) * k + col + 1]
                );

                acc[b] = fmaf(w1, a1, acc[b]);
            }
        }
    }
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = warp_sum(acc[b]);
    __shared__ float partial[B][4];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
#pragma unroll
        for (int b = 0; b < B; ++b) partial[b][warp] = acc[b];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int b = 0; b < B; ++b) {
            float v = lane < 4 ? partial[b][lane] : 0.f;
            v = warp_sum(v);
            if (lane == 0) out[static_cast<std::size_t>(b) * n + row] = v;
        }
    }
}

__global__ void dequant_fp32_kernel(const unsigned char* __restrict__ q,
                                     const half* __restrict__ scales,
                                     float* __restrict__ dense,
                                     int n, int k, int group, int groups, int row_bytes) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(n) * k;
    if (i >= count) return;
    const int row = static_cast<int>(i / k);
    const int col = static_cast<int>(i % k);
    const unsigned char byte = q[static_cast<std::size_t>(row) * row_bytes + col / 2];
    dense[i] = static_cast<float>(extract_s4(byte, col)) *
               __half2float(scales[static_cast<std::size_t>(row) * groups + col / group]);
}
__global__ void activation_fp32_kernel(const half* __restrict__ activation, float* __restrict__ dense, std::size_t size) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < size) dense[idx] = __half2float(activation[idx]);
}

class Event {
public:
    Event() { DF_CUDA(cudaEventCreate(&event_)); }
    ~Event() { if (event_) cudaEventDestroy(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    cudaEvent_t get() const { return event_; }
private:
    cudaEvent_t event_ = nullptr;
};
}

const char* kernel_name(Kernel k) noexcept {
    switch(k) { case Kernel::Direct: return "direct"; case Kernel::Grouped: return "grouped_paired"; case Kernel::GroupedScalar: return "grouped_scalar"; case Kernel::TwoPassCublas: return "two_pass_cublas"; }
    return "unknown";
}
bool cuda_compiled() noexcept { return true; }
int visible_gpu_count() { int count = 0; DF_CUDA(cudaGetDeviceCount(&count)); return count; }
std::string gpu_description(int device) {
    cudaDeviceProp p{};
    DF_CUDA(cudaGetDeviceProperties(&p, device));
    std::ostringstream ss;
    ss << p.name << " (sm_" << p.major << p.minor << ", VRAM "
       << static_cast<long long>(p.totalGlobalMem / (1024 * 1024)) << " MiB)";
    return ss.str();
}

struct GpuSession::Impl {
    Shape s;
    int device;
    int group_shift = 0;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    unsigned char* q = nullptr;
    half* scales = nullptr;
    half* activations = nullptr;
    float* y = nullptr;
    float* dense_w = nullptr;
    float* dense_a = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    std::optional<Kernel> cached_kernel;
    bool output_ready = false;

    Impl(const Inputs& in, int d) : s(in.shape), device(d) {
        in.validate();
        // Shape::validate restricts group to 16/32/64/128.
        if (s.group<=0 || (s.group & (s.group-1))!=0)
            throw std::logic_error("shift kernels require power-of-two group");
        for (int g=s.group; g>1; g>>=1) ++group_shift;
        if (d < 0 || d >= visible_gpu_count()) throw std::invalid_argument("invalid CUDA device index");
        DF_CUDA(cudaSetDevice(d));
        try {
            DF_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            DF_BLAS(cublasCreate(&blas));
            DF_BLAS(cublasSetStream(blas, stream));
            DF_CUDA(cudaMalloc(reinterpret_cast<void**>(&q), in.packed_weight.size()));
            DF_CUDA(cudaMalloc(reinterpret_cast<void**>(&scales), in.scale_fp16.size()*sizeof(std::uint16_t)));
            DF_CUDA(cudaMalloc(reinterpret_cast<void**>(&activations), in.activation_fp16.size()*sizeof(std::uint16_t)));
            DF_CUDA(cudaMalloc(reinterpret_cast<void**>(&y), static_cast<std::size_t>(s.batch)*s.n*sizeof(float)));
            // 只有调用TwoPassCublas时才分配FP32工作区。常规融合路径不占用该空间。
            DF_CUDA(cudaMemcpyAsync(q,in.packed_weight.data(),in.packed_weight.size(),cudaMemcpyHostToDevice,stream));
            DF_CUDA(cudaMemcpyAsync(scales,in.scale_fp16.data(),in.scale_fp16.size()*sizeof(std::uint16_t),cudaMemcpyHostToDevice,stream));
            DF_CUDA(cudaMemcpyAsync(activations,in.activation_fp16.data(),in.activation_fp16.size()*sizeof(std::uint16_t),cudaMemcpyHostToDevice,stream));
            DF_CUDA(cudaStreamSynchronize(stream));
        } catch(...) { cleanup(); throw; }
    }
    ~Impl() { cudaSetDevice(device); cleanup(); }
    void cleanup() noexcept {
        if (graph_exec) { cudaGraphExecDestroy(graph_exec); graph_exec = nullptr; }
        if (graph) { cudaGraphDestroy(graph); graph = nullptr; }
        if (dense_a) { cudaFree(dense_a); dense_a = nullptr; }
        if (dense_w) { cudaFree(dense_w); dense_w = nullptr; }
        if (y) { cudaFree(y); y = nullptr; }
        if (activations) { cudaFree(activations); activations = nullptr; }
        if (scales) { cudaFree(scales); scales = nullptr; }
        if (q) { cudaFree(q); q = nullptr; }
        if (blas) { cublasDestroy(blas); blas = nullptr; }
        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
    }
    void prepare_workspace(Kernel kernel) {
        if (kernel!=Kernel::TwoPassCublas) return;
        const std::size_t nw=static_cast<std::size_t>(s.n)*s.k;
        const std::size_t na=static_cast<std::size_t>(s.batch)*s.k;
        if (!dense_w) DF_CUDA(cudaMalloc(reinterpret_cast<void**>(&dense_w),nw*sizeof(float)));
        if (!dense_a) DF_CUDA(cudaMalloc(reinterpret_cast<void**>(&dense_a),na*sizeof(float)));
    }
    void launch(Kernel kernel) {
        // 设备在外层API入口设置；捕获CUDA Graph期间不可在此切换设备。
        constexpr int t=128;
        if (kernel==Kernel::Direct) {
            // Exploratory threshold, not a specialization for a full shape.
            if (s.k>=1024)
                fused_direct_kernel<true><<<static_cast<unsigned>(s.n*s.batch),t,0,stream>>>(q,scales,activations,y,
                                   s.n,s.k,s.batch,s.group,group_shift,s.groups_per_row(),s.row_bytes());
            else
                fused_direct_kernel<false><<<static_cast<unsigned>(s.n*s.batch),t,0,stream>>>(q,scales,activations,y,
                                   s.n,s.k,s.batch,s.group,group_shift,s.groups_per_row(),s.row_bytes());
            DF_CUDA(cudaGetLastError());
        } else if (kernel==Kernel::Grouped) {
            const unsigned grid=static_cast<unsigned>(s.n);
            switch(s.batch) {
                case 1: fused_grouped_kernel<1><<<grid,t,0,stream>>>(q,scales,activations,y,s.n,s.k,s.group,s.groups_per_row(),s.row_bytes()); break;
                case 2: fused_grouped_kernel<2><<<grid,t,0,stream>>>(q,scales,activations,y,s.n,s.k,s.group,s.groups_per_row(),s.row_bytes()); break;
                case 4: fused_grouped_kernel<4><<<grid,t,0,stream>>>(q,scales,activations,y,s.n,s.k,s.group,s.groups_per_row(),s.row_bytes()); break;
                case 8: fused_grouped_kernel<8><<<grid,t,0,stream>>>(q,scales,activations,y,s.n,s.k,s.group,s.groups_per_row(),s.row_bytes()); break;
                default: throw std::logic_error("unsupported batch");
            }
            DF_CUDA(cudaGetLastError());
        } else if (kernel==Kernel::GroupedScalar) {
            const unsigned grid=static_cast<unsigned>(s.n);
            switch(s.batch) {
                case 1: fused_grouped_scalar_kernel<1><<<grid,t,0,stream>>>(q,scales,activations,y,s.n,s.k,s.group,s.groups_per_row(),s.row_bytes()); break;
                case 2: fused_grouped_scalar_kernel<2><<<grid,t,0,stream>>>(q,scales,activations,y,s.n,s.k,s.group,s.groups_per_row(),s.row_bytes()); break;
                case 4: fused_grouped_scalar_kernel<4><<<grid,t,0,stream>>>(q,scales,activations,y,s.n,s.k,s.group,s.groups_per_row(),s.row_bytes()); break;
                case 8: fused_grouped_scalar_kernel<8><<<grid,t,0,stream>>>(q,scales,activations,y,s.n,s.k,s.group,s.groups_per_row(),s.row_bytes()); break;
                default: throw std::logic_error("unsupported batch");
            }
            DF_CUDA(cudaGetLastError());
        } else if (kernel==Kernel::TwoPassCublas) {
            const std::size_t nw=static_cast<std::size_t>(s.n)*s.k;
            const std::size_t na=static_cast<std::size_t>(s.batch)*s.k;
            prepare_workspace(kernel);
            dequant_fp32_kernel<<<static_cast<unsigned>((nw+255)/256),256,0,stream>>>(q,scales,dense_w,s.n,s.k,s.group,s.groups_per_row(),s.row_bytes());
            activation_fp32_kernel<<<static_cast<unsigned>((na+255)/256),256,0,stream>>>(activations,dense_a,na);
            DF_CUDA(cudaGetLastError());
            const float alpha=1.f,beta=0.f;
            // W row-major N×K -> column-major K×N, op(T) -> N×K。
            // A row-major B×K -> column-major K×B，输出 column-major N×B == row-major B×N。
            DF_BLAS(cublasSgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,s.n,s.batch,s.k,
                                &alpha,dense_w,s.k,dense_a,s.k,&beta,y,s.n));
        } else throw std::invalid_argument("unknown kernel");
    }
    float time(Kernel kernel, int warmup, int repetitions) {
        if (warmup<0 || repetitions<1 || repetitions>100000) throw std::invalid_argument("invalid timing count");
        output_ready = false;
        DF_CUDA(cudaSetDevice(device));
        // Allocation is setup, even when the caller requests zero warmup launches.
        prepare_workspace(kernel);
        for(int i=0;i<warmup;++i) launch(kernel);
        DF_CUDA(cudaStreamSynchronize(stream));
        Event start,stop;
        DF_CUDA(cudaEventRecord(start.get(),stream));
        for(int i=0;i<repetitions;++i) launch(kernel);
        DF_CUDA(cudaEventRecord(stop.get(),stream));
        DF_CUDA(cudaEventSynchronize(stop.get()));
        output_ready = true;
        float elapsed=0;
        DF_CUDA(cudaEventElapsedTime(&elapsed,start.get(),stop.get()));
        return elapsed / repetitions;
    }
    float time_graph(int warmup,int repetitions) {
        if (!graph_exec) throw std::logic_error("capture graph before benchmarking");
        if (warmup<0 || repetitions<1 || repetitions>100000) throw std::invalid_argument("invalid timing count");
        output_ready = false;
        DF_CUDA(cudaSetDevice(device));
        for(int i=0;i<warmup;++i) DF_CUDA(cudaGraphLaunch(graph_exec,stream));
        DF_CUDA(cudaStreamSynchronize(stream));
        Event start,stop;
        DF_CUDA(cudaEventRecord(start.get(),stream));
        for(int i=0;i<repetitions;++i) DF_CUDA(cudaGraphLaunch(graph_exec,stream));
        DF_CUDA(cudaEventRecord(stop.get(),stream));
        DF_CUDA(cudaEventSynchronize(stop.get()));
        output_ready = true;
        float elapsed=0;
        DF_CUDA(cudaEventElapsedTime(&elapsed,start.get(),stop.get()));
        return elapsed/repetitions;
    }
};

GpuSession::GpuSession(const Inputs& input,int device):impl_(std::make_unique<Impl>(input,device)) {}
GpuSession::~GpuSession()=default;
void GpuSession::run(Kernel k) {
    impl_->output_ready = false;
    DF_CUDA(cudaSetDevice(impl_->device));
    impl_->launch(k);
    DF_CUDA(cudaStreamSynchronize(impl_->stream));
    impl_->output_ready = true;
}
template<bool Capture>
Kernel GpuSession::autotune_impl(int warmup,int repetitions,AutotuneTrace* destination) {
    // Paired, alternating-order samples reduce drift and fixed-order bias.
    if(warmup<0 || repetitions<1 || repetitions>100000) throw std::invalid_argument("invalid autotune count");
    const int actual_warmup=std::max(warmup,10), actual_repetitions=std::max(repetitions,50);
    [[maybe_unused]] std::conditional_t<Capture,AutotuneTrace,unsigned char> captured{};
    [[maybe_unused]] std::chrono::steady_clock::time_point started{};
    if constexpr(Capture) {
        captured.requested_warmup=warmup;
        captured.requested_repetitions=repetitions;
        captured.actual_warmup=actual_warmup;
        captured.actual_repetitions=actual_repetitions;
        started=std::chrono::steady_clock::now();
    }
    std::array<float,9> ratios{};
    for(std::size_t i=0;i<ratios.size();++i) {
        float direct=0, grouped=0;
        if(i%2==0) {
            direct=impl_->time(Kernel::Direct,actual_warmup,actual_repetitions);
            grouped=impl_->time(Kernel::Grouped,actual_warmup,actual_repetitions);
        } else {
            grouped=impl_->time(Kernel::Grouped,actual_warmup,actual_repetitions);
            direct=impl_->time(Kernel::Direct,actual_warmup,actual_repetitions);
        }
        if(!(direct>0.f) || !(grouped>0.f) || !std::isfinite(direct) || !std::isfinite(grouped))
            throw std::runtime_error("invalid autotune timing");
        ratios[i]=direct/grouped;
        if constexpr(Capture) captured.samples[i]={direct,grouped,i%2==0};
    }
    std::sort(ratios.begin(),ratios.end());
    impl_->cached_kernel=ratios[ratios.size()/2]<=1.f?Kernel::Direct:Kernel::Grouped;
    if constexpr(Capture) {
        captured.selected=*impl_->cached_kernel;
        captured.wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        *destination=captured;
    }
    return *impl_->cached_kernel;
}
Kernel GpuSession::autotune(int warmup,int repetitions) {return autotune_impl<false>(warmup,repetitions,nullptr);}
Kernel GpuSession::autotune(int warmup,int repetitions,AutotuneTrace& trace) {return autotune_impl<true>(warmup,repetitions,&trace);}
void GpuSession::run_auto() {
    if(!impl_->cached_kernel) autotune();
    run(*impl_->cached_kernel);
}
std::vector<float> GpuSession::download() {
    if(!impl_->output_ready) throw std::logic_error("output unavailable before execution");
    DF_CUDA(cudaSetDevice(impl_->device));
    std::vector<float> result(static_cast<std::size_t>(impl_->s.n)*impl_->s.batch);
    DF_CUDA(cudaMemcpyAsync(result.data(),impl_->y,result.size()*sizeof(float),cudaMemcpyDeviceToHost,impl_->stream));
    DF_CUDA(cudaStreamSynchronize(impl_->stream));
    return result;
}
float GpuSession::benchmark_ms(Kernel k,int warmup,int repetitions) { return impl_->time(k,warmup,repetitions); }
void GpuSession::capture_graph(Kernel k) {
    if (k==Kernel::TwoPassCublas) throw std::invalid_argument("graph capture supports fused kernels only");
    DF_CUDA(cudaSetDevice(impl_->device));
    if (impl_->graph_exec) { DF_CUDA(cudaGraphExecDestroy(impl_->graph_exec)); impl_->graph_exec=nullptr; }
    if (impl_->graph) { DF_CUDA(cudaGraphDestroy(impl_->graph)); impl_->graph=nullptr; }
    // Capture之前确保输入上传完成，固定形状、固定device pointers。
    DF_CUDA(cudaStreamSynchronize(impl_->stream));
    DF_CUDA(cudaStreamBeginCapture(impl_->stream,cudaStreamCaptureModeGlobal));
    try { impl_->launch(k); }
    catch(...) { cudaGraph_t aborted=nullptr; cudaStreamEndCapture(impl_->stream,&aborted); if(aborted) cudaGraphDestroy(aborted); throw; }
    DF_CUDA(cudaStreamEndCapture(impl_->stream,&impl_->graph));
    DF_CUDA(cudaGraphInstantiate(&impl_->graph_exec,impl_->graph,nullptr,nullptr,0));
}
void GpuSession::launch_graph(int repetitions) {
    if (!impl_->graph_exec) throw std::logic_error("graph has not been captured");
    if (repetitions<1 || repetitions>100000) throw std::invalid_argument("invalid graph repetitions");
    impl_->output_ready = false;
    DF_CUDA(cudaSetDevice(impl_->device));
    for(int i=0;i<repetitions;++i) DF_CUDA(cudaGraphLaunch(impl_->graph_exec,impl_->stream));
    DF_CUDA(cudaStreamSynchronize(impl_->stream));
    impl_->output_ready = true;
}
float GpuSession::benchmark_graph_ms(int warmup,int repetitions) { return impl_->time_graph(warmup,repetitions); }
}
