#include "decodeforge/gpu.hpp"
#include "decodeforge/half.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace decodeforge;

namespace {
int checked=0;
void verify(const std::vector<float>& a,const std::vector<float>& b,const std::string& name) {
    auto error=compare_outputs(a,b,0.006,0.003);
    if(error.mismatched) throw std::runtime_error(name+": "+std::to_string(error.mismatched)+" mismatches / "+
                 std::to_string(error.compared)+", max abs="+std::to_string(error.max_abs));
    ++checked;
}
void check_one(const Shape& s,int device,std::uint64_t seed) {
    auto input=make_random_inputs(s,seed);
    auto ref=cpu_reference(input);
    GpuSession session(input,device);
    for(Kernel k : {Kernel::Direct,Kernel::Grouped,Kernel::GroupedScalar,Kernel::TwoPassCublas}) {
        session.run(k);
        verify(ref,session.download(),describe_shape(s)+" / "+kernel_name(k));
        // repeat-run: 每轮必须重新覆盖输出，杜绝仅一次写入的假阳性。
        session.run(k);
        verify(ref,session.download(),describe_shape(s)+" / "+kernel_name(k)+" repeated");
    }
    // Timed autotune is checked separately on representative B=1 and B=8 cases.
    // Avoid re-running 900+ profiling launches for every shape under Racecheck.
}
void check_boundary_golden(int device) {
    // Sparse independently calculated goldens at group transitions and odd tail.
    const Shape s{257,2049,2,64};
    Inputs in;
    in.shape=s;
    in.packed_weight.assign(static_cast<std::size_t>(s.n)*s.row_bytes(),0);
    in.scale_fp16.assign(static_cast<std::size_t>(s.n)*s.groups_per_row(),float_to_half(1.f));
    in.activation_fp16.assign(static_cast<std::size_t>(s.batch)*s.k,float_to_half(0.f));
    const int cols[] = {63,64,127,128,2048};
    const int weights[] = {-8,7,-8,7,-8};
    for(int row=0;row<s.n;++row) {
        for(int j=0;j<5;++j) set_int4(in.packed_weight,s,row,cols[j],weights[j]);
        const auto base=static_cast<std::size_t>(row)*s.groups_per_row();
        in.scale_fp16[base+1]=float_to_half(2.f);
        in.scale_fp16[base+2]=float_to_half(4.f);
        in.scale_fp16[base+32]=float_to_half(16.f);
    }
    for(int col:cols) {
        in.activation_fp16[col]=float_to_half(1.f);
        in.activation_fp16[static_cast<std::size_t>(s.k)+col]=float_to_half(-1.f);
    }
    in.validate();
    std::vector<float> expected(static_cast<std::size_t>(s.batch)*s.n,-110.f);
    for(int row=0;row<s.n;++row) expected[static_cast<std::size_t>(s.n)+row]=110.f;
    // -8*1 + 7*2 - 8*2 + 7*4 - 8*16 = -110.
    GpuSession gpu(in,device);
    for(Kernel k : {Kernel::Direct,Kernel::Grouped,Kernel::GroupedScalar,Kernel::TwoPassCublas}) {
        gpu.run(k);
        verify(expected,gpu.download(),std::string("handcrafted group/tail golden /")+kernel_name(k));
    }
}
}
int main(int argc,char** argv) {
    try {
        int device=0;
        if(argc==3 && std::string(argv[1])=="--device") device=std::stoi(argv[2]);
        else if(argc!=1) throw std::invalid_argument("Usage: decodeforge_gpu_tests [--device 0]");
        if(!cuda_compiled() || visible_gpu_count()<=0) throw std::runtime_error("NVIDIA GPU required; not a skip");
        {
            auto input=make_random_inputs({3,17,1,16},77);
            GpuSession fresh(input,device);
            bool rejected=false;
            try {(void)fresh.download();} catch(const std::logic_error&) {rejected=true;}
            if(!rejected) throw std::runtime_error("uninitialized W4A16 output was downloadable");
            fresh.capture_graph(Kernel::Direct);
            rejected=false;
            try {(void)fresh.download();} catch(const std::logic_error&) {rejected=true;}
            if(!rejected) throw std::runtime_error("graph capture must not mark output ready");
            fresh.launch_graph();
            verify(cpu_reference(input),fresh.download(),"graph output after first launch");
        }
        {
            auto input=make_random_inputs({3,17,1,16},78);
            const auto ref=cpu_reference(input);
            GpuSession gpu(input,device);
            // First call allocates the workspace; the second reuses it.
            for(int i=0;i<2;++i) {
                const float ms=gpu.benchmark_ms(Kernel::TwoPassCublas,0,1);
                if(!std::isfinite(ms) || ms<0) throw std::runtime_error("invalid TwoPass zero-warmup timing");
                verify(ref,gpu.download(),"TwoPass zero-warmup benchmark output");
            }
        }
        std::cout<<"[GPU VERIFY] "<<gpu_description(device)<<"\n";
        // 30组，覆盖B、非对齐K、group>k、大小维度、group边界、尾部、独立实现。
        const std::vector<Shape> cases{
           {1,1,1,16},{3,7,2,16},{7,15,4,16},{11,16,8,16},
           {13,17,1,16},{5,31,2,32},{31,32,4,32},{9,33,8,32},
           {5,63,1,64},{17,64,2,64},{11,65,4,64},{33,127,8,64},
           {64,128,1,128},{13,129,2,128},{37,255,4,128},{7,257,8,128},
           {128,512,1,64},{65,513,2,64},{32,1023,4,128},{64,1024,8,128},
           {7,3,8,128},{8,111,4,16},{9,259,2,32},{31,513,1,64},
           {33,2049,8,128},{128,256,4,32},{3,9,1,128},{17,70,2,64},
           {91,1001,4,128},{64,2048,8,64}
        };
        for(std::size_t i=0;i<cases.size();++i) {
            check_one(cases[i],device,0xDEADBEEFull+i);
            std::cout<<"[GPU VERIFY] "<<(i+1)<<"/"<<cases.size()<<" PASS "<<describe_shape(cases[i])<<"\n";
        }
        // Both sides of the exploratory Direct and Paired dispatch thresholds,
        // plus the preceding loop-work boundary; cover odd K and every B/G.
        std::uint64_t edge_seed=0x7A1100;
        for(int k : {511,512,513,1023,1024,1025,2047,2048,2049})
            for(int batch : {1,2,4,8})
                for(int group : {16,32,64,128})
                    check_one({7,k,batch,group},device,edge_seed++);
        std::cout<<"[GPU VERIFY] experimental dispatch boundary shapes PASS (144)\n";
        check_boundary_golden(device);
        // Internal trace describes this exact tuning invocation, not a later
        // noisy remeasurement. Invalid calls must not publish a partial trace.
        {
            const Shape s{65,1025,1,64};
            const auto input=make_random_inputs(s,0x5511);
            const auto ref=cpu_reference(input);
            GpuSession gpu(input,device);
            AutotuneTrace trace{};
            const auto selected=gpu.autotune(0,1,trace);
            if(trace.requested_warmup!=0 || trace.requested_repetitions!=1 ||
               trace.actual_warmup!=10 || trace.actual_repetitions!=50 ||
               trace.selected!=selected || !std::isfinite(trace.wall_ms) || trace.wall_ms<0)
                throw std::runtime_error("autotune trace metadata mismatch");
            std::array<float,9> ratios{};
            for(std::size_t i=0;i<trace.samples.size();++i) {
                const auto& sample=trace.samples[i];
                if(sample.direct_first!=(i%2==0) || !std::isfinite(sample.direct_ms) ||
                   !std::isfinite(sample.grouped_ms) || sample.direct_ms<=0 || sample.grouped_ms<=0)
                    throw std::runtime_error("invalid internal autotune sample");
                ratios[i]=sample.direct_ms/sample.grouped_ms;
            }
            std::sort(ratios.begin(),ratios.end());
            const auto expected=ratios[ratios.size()/2]<=1.f?Kernel::Direct:Kernel::Grouped;
            if(selected!=expected) throw std::runtime_error("trace does not reproduce tuner decision");
            gpu.run_auto();
            verify(ref,gpu.download(),"autotune trace cached run");
            trace.wall_ms=-17.;
            bool rejected=false;
            try {gpu.autotune(-1,1,trace);} catch(const std::invalid_argument&) {rejected=true;}
            if(!rejected || trace.wall_ms!=-17.) throw std::runtime_error("failed tuner changed trace");
            gpu.run_auto();
            verify(ref,gpu.download(),"cached run after failed explicit tune");
        }
        // Exercise Graph capture on both sides of each experimental threshold.
        for(const Shape s : {Shape{65,1023,1,16},Shape{65,1024,2,32},Shape{65,1025,4,64},
                             Shape{65,2047,8,128},Shape{65,2048,1,64},Shape{65,2049,2,32}}) {
            auto input=make_random_inputs(s,0xABC000u+static_cast<unsigned>(s.k));
            const auto ref=cpu_reference(input);
            GpuSession gpu(input,device);
            for(Kernel k : {Kernel::Direct,Kernel::Grouped}) {
                gpu.capture_graph(k);
                gpu.launch_graph(3);
                verify(ref,gpu.download(),describe_shape(s)+" graph / "+kernel_name(k));
            }
        }
        // Each tune measures both candidates; the subsequent dispatch must be correct.
        for(const Shape s : {Shape{65,1025,1,16},Shape{65,2049,8,128}}) {
            auto input=make_random_inputs(s,0xABC100u+static_cast<unsigned>(s.k));
            const auto ref=cpu_reference(input);
            GpuSession gpu(input,device);
            const auto selected=gpu.autotune(0,1);
            if(selected!=Kernel::Direct && selected!=Kernel::Grouped)
                throw std::runtime_error("shift autotune returned invalid kernel");
            gpu.run_auto();
            verify(ref,gpu.download(),describe_shape(s)+" shift autotune");
        }
        // Two representative dispatcher cases exercise both batch-launch layouts.
        for(const Shape tune_shape : {Shape{81,1001,1,64}, Shape{81,1001,8,64}}) {
            auto input=make_random_inputs(tune_shape,2026);
            GpuSession tuned(input,device);
            const auto reference=cpu_reference(input);
            const auto selected=tuned.autotune(0,1);
            if(selected!=Kernel::Direct && selected!=Kernel::Grouped)
                throw std::runtime_error("autotune returned invalid kernel");
            tuned.run_auto();
            verify(reference,tuned.download(),"autotuned dispatcher");
        }
        auto in=make_random_inputs({81,1001,4,64},2026);
        GpuSession session(in,device);
        auto reference=cpu_reference(in);
        for(Kernel k:{Kernel::Direct,Kernel::Grouped,Kernel::GroupedScalar}) {
            session.capture_graph(k);
            session.launch_graph(3);
            verify(reference,session.download(),std::string("CUDA graph ")+kernel_name(k));
        }
        const float t=session.benchmark_graph_ms(2,10);
        if(!(t>0)) throw std::runtime_error("invalid CUDA Graph timing");
        std::cout<<"[GPU VERIFY] CUDA graph capture/replay PASS, "<<t<<" ms/replay\n";
        std::cout<<"[GPU VERIFY] ALL PASS; output checks="<<checked<<"\n";
        return EXIT_SUCCESS;
    } catch(const std::exception& e) {
        std::cerr<<"[GPU VERIFY] FAIL: "<<e.what()<<"\n";
        return EXIT_FAILURE;
    }
}
