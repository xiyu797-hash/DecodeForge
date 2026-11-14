#include "decodeforge/ops.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace decodeforge {
namespace {
void check_cuda(cudaError_t e,const char* s,int line) {
    if(e!=cudaSuccess) {
        std::ostringstream ss;ss<<"CUDA "<<s<<":"<<line<<": "<<cudaGetErrorString(e);
        throw std::runtime_error(ss.str());
    }
}
void check_cublas(cublasStatus_t e,const char* s,int line) {
    if(e!=CUBLAS_STATUS_SUCCESS) {
        std::ostringstream ss;ss<<"cuBLAS "<<s<<":"<<line<<": status="<<static_cast<int>(e);
        throw std::runtime_error(ss.str());
    }
}
#define CHECK_CUDA(v) check_cuda((v),#v,__LINE__)
#define CHECK_CUBLAS(v) check_cublas((v),#v,__LINE__)

class CudaEvents {
public:
    CudaEvents(){
        CHECK_CUDA(cudaEventCreate(&start));
        try { CHECK_CUDA(cudaEventCreate(&stop)); }
        catch (...) { cudaEventDestroy(start); start=nullptr; throw; }
    }
    ~CudaEvents(){if(stop)cudaEventDestroy(stop);if(start)cudaEventDestroy(start);}
    cudaEvent_t start=nullptr,stop=nullptr;
};

template<typename T> __device__ float scalar_to_float(T v);
template<> __device__ float scalar_to_float<half>(half v) {return __half2float(v);}
template<> __device__ float scalar_to_float<__nv_bfloat16>(__nv_bfloat16 v) {return __bfloat162float(v);}

template<typename T>
__global__ void naive_gemm_kernel(const T* a,const T* b,float* y,int m,int n,int k) {
    const int row=static_cast<int>(blockIdx.y)*16+threadIdx.y;
    const int col=static_cast<int>(blockIdx.x)*16+threadIdx.x;
    if(row>=m||col>=n) return;
    float acc=0.f;
    for(int kk=0;kk<k;++kk)
        acc=fmaf(scalar_to_float(a[static_cast<std::size_t>(row)*k+kk]),
                 scalar_to_float(b[static_cast<std::size_t>(kk)*n+col]),acc);
    y[static_cast<std::size_t>(row)*n+col]=acc;
}
template<typename T>
__global__ void tiled_gemm_kernel(const T* a,const T* b,float* y,int m,int n,int k) {
    constexpr int TS=16;
    const int row=static_cast<int>(blockIdx.y)*TS+threadIdx.y;
    const int col=static_cast<int>(blockIdx.x)*TS+threadIdx.x;
    __shared__ float lhs[TS][TS];
    __shared__ float rhs[TS][TS];
    float acc=0;
    for(int tile=0;tile<(k+TS-1)/TS;++tile) {
        const int ka=tile*TS+threadIdx.x;
        const int kb=tile*TS+threadIdx.y;
        lhs[threadIdx.y][threadIdx.x]=(row<m&&ka<k)?scalar_to_float(a[static_cast<std::size_t>(row)*k+ka]):0.f;
        rhs[threadIdx.y][threadIdx.x]=(kb<k&&col<n)?scalar_to_float(b[static_cast<std::size_t>(kb)*n+col]):0.f;
        __syncthreads();
#pragma unroll
        for(int t=0;t<TS;++t) acc=fmaf(lhs[threadIdx.y][t],rhs[t][threadIdx.x],acc);
        __syncthreads();
    }
    if(row<m&&col<n) y[static_cast<std::size_t>(row)*n+col]=acc;
}

__device__ __forceinline__ float warp_reduce(float v) {
    for(int delta=16;delta>=1;delta>>=1) v+=__shfl_down_sync(0xffffffff,v,delta);
    return v;
}
__global__ void residual_add(const half* x,const half* r,float* t,std::size_t count) {
    const auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count) t[i]=__half2float(x[i])+__half2float(r[i]);
}
// 两阶段参考：先写FP32 Residual中间矩阵，再读中间矩阵计算RMSNorm。
__global__ void norm_two_pass_kernel(const float* t,const half* gamma,float* y,int cols,float eps) {
    const int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    float accum=0;
    for(int col=tid;col<cols;col+=blockDim.x) {
        const float v=t[static_cast<std::size_t>(row)*cols+col];
        accum=fmaf(v,v,accum);
    }
    accum=warp_reduce(accum);
    __shared__ float sums[8];
    __shared__ float inv;
    if(lane==0) sums[warp]=accum;
    __syncthreads();
    if(warp==0) {
        float v=lane<8?sums[lane]:0.f;
        v=warp_reduce(v);
        if(lane==0) inv=rsqrtf(v/static_cast<float>(cols)+eps);
    }
    __syncthreads();
    for(int col=tid;col<cols;col+=blockDim.x)
        y[static_cast<std::size_t>(row)*cols+col]=t[static_cast<std::size_t>(row)*cols+col]*inv*__half2float(gamma[col]);
}
// 融合方案：不在显存里写Residual中间矩阵，减少一次全矩阵写入/读取。
__global__ void norm_fused_kernel(const half* x,const half* r,const half* gamma,float* y,int cols,float eps) {
    const int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    float accum=0;
    for(int col=tid;col<cols;col+=blockDim.x) {
        const auto i=static_cast<std::size_t>(row)*cols+col;
        const float v=__half2float(x[i])+__half2float(r[i]);
        accum=fmaf(v,v,accum);
    }
    accum=warp_reduce(accum);
    __shared__ float sums[8];
    __shared__ float inv;
    if(lane==0) sums[warp]=accum;
    __syncthreads();
    if(warp==0) {
        float v=lane<8?sums[lane]:0.f;
        v=warp_reduce(v);
        if(lane==0) inv=rsqrtf(v/static_cast<float>(cols)+eps);
    }
    __syncthreads();
    for(int col=tid;col<cols;col+=blockDim.x) {
        const auto i=static_cast<std::size_t>(row)*cols+col;
        const float v=__half2float(x[i])+__half2float(r[i]);
        y[i]=v*inv*__half2float(gamma[col]);
    }
}
}

struct GemmSession::Impl {
    int m,n,k,device;
    DType dtype;
    cudaStream_t stream=nullptr;
    cublasHandle_t blas=nullptr;
    void *a=nullptr,*b=nullptr;
    float* y=nullptr;
    bool output_ready=false;
    Impl(const GemmInput& in,int d):m(in.m),n(in.n),k(in.k),device(d),dtype(in.dtype) {
        in.validate();
        int count=0;CHECK_CUDA(cudaGetDeviceCount(&count));
        if(d<0||d>=count) throw std::invalid_argument("invalid GPU device");
        CHECK_CUDA(cudaSetDevice(d));
        try {
            CHECK_CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            CHECK_CUBLAS(cublasCreate(&blas));
            CHECK_CUBLAS(cublasSetStream(blas,stream));
            CHECK_CUDA(cudaMalloc(&a,in.a.size()*sizeof(std::uint16_t)));
            CHECK_CUDA(cudaMalloc(&b,in.b.size()*sizeof(std::uint16_t)));
            CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&y),static_cast<std::size_t>(m)*n*sizeof(float)));
            CHECK_CUDA(cudaMemcpyAsync(a,in.a.data(),in.a.size()*sizeof(std::uint16_t),cudaMemcpyHostToDevice,stream));
            CHECK_CUDA(cudaMemcpyAsync(b,in.b.data(),in.b.size()*sizeof(std::uint16_t),cudaMemcpyHostToDevice,stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));
        }catch(...){cleanup();throw;}
    }
    ~Impl(){cudaSetDevice(device);cleanup();}
    void cleanup() noexcept {
        if(y){cudaFree(y);y=nullptr;}
        if(b){cudaFree(b);b=nullptr;}
        if(a){cudaFree(a);a=nullptr;}
        if(blas){cublasDestroy(blas);blas=nullptr;}
        if(stream){cudaStreamDestroy(stream);stream=nullptr;}
    }
    void launch(GemmKernel which) {
        const dim3 block(16,16),grid(static_cast<unsigned>((n+15)/16),static_cast<unsigned>((m+15)/16));
        if(which==GemmKernel::Naive) {
            if(dtype==DType::FP16) naive_gemm_kernel<<<grid,block,0,stream>>>(static_cast<half*>(a),static_cast<half*>(b),y,m,n,k);
            else naive_gemm_kernel<<<grid,block,0,stream>>>(static_cast<__nv_bfloat16*>(a),static_cast<__nv_bfloat16*>(b),y,m,n,k);
            CHECK_CUDA(cudaGetLastError());
        } else if(which==GemmKernel::Tiled) {
            if(dtype==DType::FP16) tiled_gemm_kernel<<<grid,block,0,stream>>>(static_cast<half*>(a),static_cast<half*>(b),y,m,n,k);
            else tiled_gemm_kernel<<<grid,block,0,stream>>>(static_cast<__nv_bfloat16*>(a),static_cast<__nv_bfloat16*>(b),y,m,n,k);
            CHECK_CUDA(cudaGetLastError());
        } else if(which==GemmKernel::Cublas) {
            const auto cudaType=dtype==DType::FP16?CUDA_R_16F:CUDA_R_16BF;
            const float alpha=1.f,beta=0.f;
            // Row-major C(M×N) == Col-major C^T(N×M)。B视作Col-major(N×K),A视作Col-major(K×M)。
            CHECK_CUBLAS(cublasGemmEx(blas,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,
                      &alpha,b,cudaType,n,a,cudaType,k,&beta,y,CUDA_R_32F,n,
                      CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
        } else throw std::invalid_argument("invalid GEMM kernel");
    }
    float time(GemmKernel which,int warmup,int repetitions) {
        if(warmup<0||repetitions<1||repetitions>100000) throw std::invalid_argument("bad GEMM benchmark count");
        output_ready=false;
        CHECK_CUDA(cudaSetDevice(device));
        for(int i=0;i<warmup;++i) launch(which);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CudaEvents events;
        CHECK_CUDA(cudaEventRecord(events.start,stream));
        for(int i=0;i<repetitions;++i) launch(which);
        CHECK_CUDA(cudaEventRecord(events.stop,stream));
        CHECK_CUDA(cudaEventSynchronize(events.stop));
        output_ready=true;
        float ms=0;CHECK_CUDA(cudaEventElapsedTime(&ms,events.start,events.stop));
        return ms/repetitions;
    }
};
GemmSession::GemmSession(const GemmInput& input,int device):impl_(std::make_unique<Impl>(input,device)){}
GemmSession::~GemmSession()=default;
void GemmSession::run(GemmKernel kernel){impl_->output_ready=false;CHECK_CUDA(cudaSetDevice(impl_->device));impl_->launch(kernel);CHECK_CUDA(cudaStreamSynchronize(impl_->stream));impl_->output_ready=true;}
std::vector<float> GemmSession::download(){
    if(!impl_->output_ready)throw std::logic_error("output unavailable before execution");
    CHECK_CUDA(cudaSetDevice(impl_->device));
    std::vector<float> data(static_cast<std::size_t>(impl_->m)*impl_->n);
    CHECK_CUDA(cudaMemcpyAsync(data.data(),impl_->y,data.size()*sizeof(float),cudaMemcpyDeviceToHost,impl_->stream));
    CHECK_CUDA(cudaStreamSynchronize(impl_->stream));
    return data;
}
float GemmSession::benchmark_ms(GemmKernel which,int warmup,int reps){return impl_->time(which,warmup,reps);}

struct NormSession::Impl {
    int rows,cols,device;
    float epsilon;
    cudaStream_t stream=nullptr;
    half *x=nullptr,*residual=nullptr,*gamma=nullptr;
    float *tmp=nullptr,*y=nullptr;
    bool output_ready=false;
    Impl(const NormInput& in,int d):rows(in.rows),cols(in.cols),device(d),epsilon(in.epsilon) {
        in.validate();int count=0;CHECK_CUDA(cudaGetDeviceCount(&count));
        if(d<0||d>=count) throw std::invalid_argument("invalid GPU device");
        CHECK_CUDA(cudaSetDevice(d));
        try {
            CHECK_CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&x),in.x.size()*sizeof(std::uint16_t)));
            CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&residual),in.residual.size()*sizeof(std::uint16_t)));
            CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&gamma),in.gamma.size()*sizeof(std::uint16_t)));
            CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&y),in.x.size()*sizeof(float)));
            CHECK_CUDA(cudaMemcpyAsync(x,in.x.data(),in.x.size()*sizeof(std::uint16_t),cudaMemcpyHostToDevice,stream));
            CHECK_CUDA(cudaMemcpyAsync(residual,in.residual.data(),in.residual.size()*sizeof(std::uint16_t),cudaMemcpyHostToDevice,stream));
            CHECK_CUDA(cudaMemcpyAsync(gamma,in.gamma.data(),in.gamma.size()*sizeof(std::uint16_t),cudaMemcpyHostToDevice,stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));
        }catch(...){cleanup();throw;}
    }
    ~Impl(){cudaSetDevice(device);cleanup();}
    void cleanup() noexcept {
        if(tmp){cudaFree(tmp);tmp=nullptr;}
        if(y){cudaFree(y);y=nullptr;}
        if(gamma){cudaFree(gamma);gamma=nullptr;}
        if(residual){cudaFree(residual);residual=nullptr;}
        if(x){cudaFree(x);x=nullptr;}
        if(stream){cudaStreamDestroy(stream);stream=nullptr;}
    }
    void prepare_workspace(NormKernel which) {
        if(which==NormKernel::TwoPass && !tmp) {
            const auto count=static_cast<std::size_t>(rows)*cols;
            CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&tmp),count*sizeof(float)));
        }
    }
    void launch(NormKernel which) {
        if(which==NormKernel::Fused) {
            norm_fused_kernel<<<rows,256,0,stream>>>(x,residual,gamma,y,cols,epsilon);
            CHECK_CUDA(cudaGetLastError());
        } else if(which==NormKernel::TwoPass) {
            const auto count=static_cast<std::size_t>(rows)*cols;
            prepare_workspace(which);
            residual_add<<<static_cast<unsigned>((count+255)/256),256,0,stream>>>(x,residual,tmp,count);
            norm_two_pass_kernel<<<rows,256,0,stream>>>(tmp,gamma,y,cols,epsilon);
            CHECK_CUDA(cudaGetLastError());
        } else throw std::invalid_argument("invalid Norm kernel");
    }
    float time(NormKernel which,int warmup,int repetitions) {
        if(warmup<0||repetitions<1||repetitions>100000) throw std::invalid_argument("bad Norm benchmark count");
        output_ready=false;
        CHECK_CUDA(cudaSetDevice(device));
        // Exclude lazy allocation from CUDA event timings even with zero warmup.
        prepare_workspace(which);
        for(int i=0;i<warmup;++i) launch(which);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CudaEvents events;
        CHECK_CUDA(cudaEventRecord(events.start,stream));
        for(int i=0;i<repetitions;++i) launch(which);
        CHECK_CUDA(cudaEventRecord(events.stop,stream));
        CHECK_CUDA(cudaEventSynchronize(events.stop));
        output_ready=true;
        float ms=0;CHECK_CUDA(cudaEventElapsedTime(&ms,events.start,events.stop));
        return ms/repetitions;
    }
};
NormSession::NormSession(const NormInput& input,int device):impl_(std::make_unique<Impl>(input,device)){}
NormSession::~NormSession()=default;
void NormSession::run(NormKernel kernel){impl_->output_ready=false;CHECK_CUDA(cudaSetDevice(impl_->device));impl_->launch(kernel);CHECK_CUDA(cudaStreamSynchronize(impl_->stream));impl_->output_ready=true;}
std::vector<float> NormSession::download(){
    if(!impl_->output_ready)throw std::logic_error("output unavailable before execution");
    CHECK_CUDA(cudaSetDevice(impl_->device));
    std::vector<float> result(static_cast<std::size_t>(impl_->rows)*impl_->cols);
    CHECK_CUDA(cudaMemcpyAsync(result.data(),impl_->y,result.size()*sizeof(float),cudaMemcpyDeviceToHost,impl_->stream));
    CHECK_CUDA(cudaStreamSynchronize(impl_->stream));
    return result;
}
float NormSession::benchmark_ms(NormKernel which,int warmup,int reps){return impl_->time(which,warmup,reps);}
}
