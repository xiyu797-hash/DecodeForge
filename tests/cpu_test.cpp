#include "decodeforge/core.hpp"
#include "decodeforge/half.hpp"
#include "decodeforge/benchmark_samples.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace decodeforge;
namespace {
int assertions=0;
void require(bool ok,const std::string& what) {
    ++assertions;
    if(!ok) throw std::runtime_error("FAIL: "+what);
}
template<class F> void requires_throw(F f, const std::string& name) {
    bool thrown=false;
    try { f(); } catch(const std::exception&) { thrown=true; }
    require(thrown,"should throw: "+name);
}
void test_ablation_sample_csv() {
    std::array<AblationSample,7> samples{};
    for(std::size_t i=0;i<samples.size();++i)
        samples[i]={static_cast<float>(i+1),static_cast<float>(i+2),i%2==0};
    std::ostringstream output;
    write_ablation_sample_header(output);
    write_ablation_sample_rows(output,Shape{2,17,1,16},samples);
    const std::string expected=
        "n,k,batch,group,trial,first_kernel,paired_ms,scalar_ms,scalar_over_paired\n"
        "2,17,1,16,1,grouped_paired,1,2,2\n"
        "2,17,1,16,2,grouped_scalar,2,3,1.5\n";
    require(output.str().find(expected)==0,"raw paired rows preserve original timing and order");
    samples[2].paired_ms=0.f;
    requires_throw([&]{std::ostringstream bad;write_ablation_sample_rows(bad,Shape{2,17,1,16},samples);},"invalid raw sample timing");
    samples[2].paired_ms=std::numeric_limits<float>::quiet_NaN();
    requires_throw([&]{std::ostringstream bad;write_ablation_sample_rows(bad,Shape{2,17,1,16},samples);},"non-finite raw sample timing");
    const auto path=std::filesystem::current_path()/("df-samples-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".csv");
    write_new_ablation_sample_file(path,"original\n");
    try {
        requires_throw([&]{write_new_ablation_sample_file(path,"replacement\n");},"existing raw report must not be overwritten");
        std::ifstream input(path,std::ios::binary);
        const std::string actual((std::istreambuf_iterator<char>(input)),std::istreambuf_iterator<char>());
        require(actual=="original\n","existing raw report content remains intact");
    } catch(...) { std::filesystem::remove(path); throw; }
    std::filesystem::remove(path);
    const auto unicode_path=std::filesystem::current_path()/std::filesystem::u8path(u8"df-样本-🚀-");
    const auto unique_unicode=std::filesystem::path(unicode_path.native()+std::filesystem::path(std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".csv").native());
    write_new_ablation_sample_file(unique_unicode,"unicode\n");
    try {
        std::ifstream unicode_input(unique_unicode,std::ios::binary);
        const std::string actual((std::istreambuf_iterator<char>(unicode_input)),std::istreambuf_iterator<char>());
        require(actual=="unicode\n","Unicode raw report path roundtrip");
        requires_throw([&]{write_new_ablation_sample_file(unique_unicode,"replacement\n");},"Unicode report is protected");
    } catch(...) { std::filesystem::remove(unique_unicode); throw; }
    std::filesystem::remove(unique_unicode);
}
void test_choice_sample_csv() {
    std::array<ChoiceSample,9> samples{};
    for(std::size_t i=0;i<samples.size();++i)
        samples[i]={static_cast<float>(i+1),static_cast<float>(i+2),i%2==0};
    std::ostringstream output;
    write_choice_sample_header(output);
    write_choice_sample_rows(output,Shape{2,17,1,16},samples,true,12.25,3,15,100);
    const std::string expected=
        "n,k,batch,group,run,trial,first_kernel,direct_ms,paired_ms,direct_over_paired,selected,autotune_wall_ms,warmup,repeats\n"
        "2,17,1,16,3,1,direct,1,2,0.5,direct,12.25,15,100\n"
        "2,17,1,16,3,2,grouped_paired,2,3,0.666666687,direct,12.25,15,100\n";
    require(output.str().find(expected)==0,"choice rows preserve timing, selection, order and tuning wall time");
    samples[0].direct_ms=std::numeric_limits<float>::quiet_NaN();
    requires_throw([&]{std::ostringstream bad;write_choice_sample_rows(bad,Shape{2,17,1,16},samples,true,12.25,3,15,100);},"non-finite direct timing");
    samples[0].direct_ms=1.f;
    requires_throw([&]{std::ostringstream bad;write_choice_sample_rows(bad,Shape{2,17,1,16},samples,true,-1.,3,15,100);},"negative tuning wall time");
}
void test_autotune_trace_csv() {
    AutotuneTrace trace{};
    trace.requested_warmup=0; trace.requested_repetitions=1;
    trace.actual_warmup=10; trace.actual_repetitions=50;
    trace.selected=Kernel::Direct; trace.wall_ms=2.25;
    for(std::size_t i=0;i<trace.samples.size();++i)
        trace.samples[i]={static_cast<float>(i+1),static_cast<float>(i+2),i%2==0};
    std::ostringstream out;
    write_autotune_trace_header(out);
    write_autotune_trace_rows(out,Shape{2,17,1,16},trace,3);
    const std::string expected=
        "n,k,batch,group,run,trial,first_kernel,direct_ms,grouped_ms,direct_over_grouped,selected,requested_warmup,requested_repetitions,actual_warmup,actual_repetitions,autotune_wall_ms\n"
        "2,17,1,16,3,1,direct,1,2,0.5,direct,0,1,10,50,2.25\n"
        "2,17,1,16,3,2,grouped_paired,2,3,0.666666687,direct,0,1,10,50,2.25\n";
    require(out.str().find(expected)==0,"tuner samples preserve original order, timings and metadata");
    trace.samples[2].grouped_ms=std::numeric_limits<float>::quiet_NaN();
    std::ostringstream bad;
    requires_throw([&]{write_autotune_trace_rows(bad,Shape{2,17,1,16},trace,3);},"nonfinite internal sample");
    require(bad.str().empty(),"invalid trace writes no partial row");
    trace.samples[2].grouped_ms=3.f;
    trace.actual_repetitions=49;
    requires_throw([&]{std::ostringstream bad_counts;write_autotune_trace_rows(bad_counts,Shape{2,17,1,16},trace,3);},"inconsistent actual tuning counts");
}
void test_fp16() {
    // 精确遍历所有65536种binary16位型，普通有限值都必须往返无损。
    for(std::uint32_t h=0;h<65536;++h) {
        const auto original=static_cast<std::uint16_t>(h);
        const auto result=float_to_half(half_to_float(original));
        if(((h>>10)&31u)==31u && (h&1023u)) {
            require((result&0x7c00u)==0x7c00u && (result&0x03ffu)!=0,"NaN roundtrip");
        } else require(result==original,"half roundtrip at "+std::to_string(h));
    }
    require(float_to_half(1.f)==0x3c00u,"one maps to 3c00");
    require(float_to_half(-1.f)==0xbc00u,"minus one");
    require(float_to_half(0.f)==0u,"positive zero");
    require(float_to_half(-0.f)==0x8000u,"negative zero");
    require(float_to_half(std::numeric_limits<float>::infinity())==0x7c00u,"inf");
    require(float_to_half(65504.f)==0x7bffu,"half max finite");
    require(float_to_half(65536.f)==0x7c00u,"half overflow");
    // tie-to-even at 1.0 (midpoint between 1 and 1 + 1/1024)
    require(float_to_half(1.f+1.f/2048.f)==0x3c00u,"tie to even");
}
void test_quantization() {
    const Shape s{3,17,2,16};
    auto in=make_random_inputs(s,17);
    require(in.packed_weight.size()==27,"packed row padding");
    require(in.scale_fp16.size()==6,"group ceiling");
    require(in.activation_fp16.size()==34,"activation size");
    for(int n=0;n<s.n;++n) for(int k=0;k<s.k;++k) {
        const int old=get_int4(in,n,k);
        require(old>=-8 && old<=7,"int4 signed range");
        set_int4(in.packed_weight,s,n,k,old);
        require(get_int4(in,n,k)==old,"int4 set/get roundtrip");
    }
    in.validate();
    require(get_int4(in,0,16)>=-8,"tail nibble");
    auto ref=cpu_reference(in);
    auto two=cpu_two_pass_reference(in);
    auto stats=compare_outputs(ref,two,1e-4,1e-4);
    require(stats.mismatched==0,"two CPU references agree");
    in.packed_weight[8] |= 0x80u;
    requires_throw([&]{in.validate();},"nonzero odd-K high padding");
    in.packed_weight[8] &= 0x0fu;
    in.scale_fp16[0]=0;
    requires_throw([&]{in.validate();},"zero scale");
    in.scale_fp16[0]=float_to_half(.1f);
    in.activation_fp16[0]=float_to_half(std::numeric_limits<float>::infinity());
    requires_throw([&]{in.validate();},"non-finite activation");
}
void test_handcrafted_golden() {
    // 手工金标准：排除“两个实现都调用同一个错误解码函数”的共同缺陷。
    const Shape shape{2,4,2,16};
    Inputs in;
    in.shape=shape;
    in.packed_weight.assign(4,0);
    const int q[2][4]={{-8,-1,0,7},{7,-8,7,-8}};
    for(int n=0;n<2;++n) for(int k=0;k<4;++k) set_int4(in.packed_weight,shape,n,k,q[n][k]);
    in.scale_fp16={float_to_half(1.f),float_to_half(.5f)};
    in.activation_fp16={float_to_half(2.f),float_to_half(-3.f),float_to_half(4.f),float_to_half(-1.f),
                        float_to_half(-1.f),float_to_half(2.f),float_to_half(0.f),float_to_half(3.f)};
    in.validate();
    const auto y=cpu_reference(in);
    require(y.size()==4,"golden result size");
    require(y[0]==-20.f,"golden B0 N0");
    require(y[1]==37.f,"golden B0 N1");
    require(y[2]==27.f,"golden B1 N0");
    require(y[3]==-23.5f,"golden B1 N1");
}
void test_input_validation() {
    requires_throw([&]{Shape{0,128,1,32}.validate();},"zero N");
    requires_throw([&]{Shape{128,-1,1,32}.validate();},"negative K");
    requires_throw([&]{Shape{32,64,3,32}.validate();},"unsupported B");
    requires_throw([&]{Shape{32,64,1,3}.validate();},"unsupported group");
    requires_throw([&]{Shape{65537,128,1,32}.validate();},"oversized N");
    requires_throw([&]{Shape{65536,65536,1,32}.validate();},"packed overflow");
    const Shape s{1,1,1,16};
    requires_throw([&]{quantize_inputs(s,{},{});},"wrong input lengths");
    std::vector<std::uint8_t> empty;
    requires_throw([&]{set_int4(empty,s,0,0,8);},"invalid int4 value");
    auto in=make_random_inputs(s,7);
    requires_throw([&]{get_int4(in,0,1);},"out of range read");
    requires_throw([&]{compare_outputs({1.f},{},.1,.1);},"mismatched output lengths");
    requires_throw([&]{compare_outputs({1.f},{1.f},std::numeric_limits<double>::quiet_NaN(),.1);},"NaN tolerance");
    requires_throw([&]{compare_outputs({1.f},{1.f},.1,std::numeric_limits<double>::infinity());},"Inf tolerance");
    auto broken=in;
    broken.packed_weight.clear();
    requires_throw([&]{broken.validate();},"truncated weights");
    requires_throw([&]{get_int4(broken,0,0);},"truncated public int4 accessor");
    Inputs invalid_shape;
    invalid_shape.shape=Shape{1,std::numeric_limits<int>::max(),1,16};
    invalid_shape.packed_weight={0};
    requires_throw([&]{(void)get_int4(invalid_shape,0,0);},"public accessor rejects invalid shape before arithmetic");
    require(Shape{1,std::numeric_limits<int>::max(),1,16}.row_bytes()==1073741824,"row bytes do not overflow");
    require(Shape{1,std::numeric_limits<int>::max(),1,16}.groups_per_row()==134217728,"group count does not overflow");
    requires_throw([&]{(void)Shape{1,1,1,0}.groups_per_row();},"zero group rejected");
    require(signed_int4(0xffu)==-1,"nibble decode must mask high bits");
    requires_throw([&]{
        quantize_inputs(Shape{1,16,1,16}, std::vector<float>(16,1.e30f),
                        std::vector<float>(16,1.f));
    },"unrepresentable scale");
    // 零权重组应输出全0，不产生NaN或无效scale。
    auto zeros=quantize_inputs(Shape{2,7,1,16},std::vector<float>(14,0),std::vector<float>(7,1));
    for(auto v:cpu_reference(zeros)) require(v==0,"zero weights");
}
void test_shapes() {
    const std::vector<int> ks{1,2,7,15,16,17,31,32,33,63,64,65,127,128,129,257,511};
    const std::vector<int> batches{1,2,4,8};
    for(int group : {16,32,64,128}) for(int batch : batches) for(int k:ks) {
        const Shape s{7,k,batch,group};
        auto in=make_random_inputs(s,static_cast<std::uint64_t>(12345+k+group*4+batch*32));
        // Regression for the CUDA pairwise decoder: both nibbles of a packed
        // byte must decode correctly and (with even group sizes) share a scale.
        for(int row=0; row<s.n; ++row) {
            for(int col=0; col<s.k; col+=2) {
                const auto byte=in.packed_weight[static_cast<std::size_t>(row)*s.row_bytes()+col/2];
                require(get_int4(in,row,col)==signed_int4(byte&15u),"paired low nibble");
                if(col+1<s.k) {
                    require(get_int4(in,row,col+1)==signed_int4(byte>>4),"paired high nibble");
                    require(col/s.group==(col+1)/s.group,"paired scale group");
                } else {
                    require((byte&0xf0u)==0,"odd-K padding nibble");
                }
            }
        }
        const auto a=cpu_reference(in);
        const auto b=cpu_two_pass_reference(in);
        const auto stats=compare_outputs(a,b,1e-4,1e-5);
        require(stats.mismatched==0,"shape CPU equivalence "+describe_shape(s));
        require(a.size()==static_cast<std::size_t>(s.n*s.batch),"result layout");
        // repeatability: bit-identical reference with same seed
        auto in2=make_random_inputs(s,static_cast<std::uint64_t>(12345+k+group*4+batch*32));
        require(in.packed_weight==in2.packed_weight && in.scale_fp16==in2.scale_fp16 &&
                in.activation_fp16==in2.activation_fp16,"reproducible generator");
    }
}
}
int main() {
    try {
        test_ablation_sample_csv();
        test_choice_sample_csv();
        test_autotune_trace_csv();
        test_fp16();
        std::cout<<"[PASS] exhaustive binary16 round-trip (65,536 bit patterns)\n";
        test_quantization(); std::cout<<"[PASS] INT4 pack, odd-K, scales, two-pass CPU reference\n";
        test_handcrafted_golden(); std::cout<<"[PASS] handcrafted signed-INT4 golden values\n";
        test_input_validation(); std::cout<<"[PASS] invalid shapes, bounds, NaN/Inf, padding, zero weights\n";
        test_shapes(); std::cout<<"[PASS] 272 shape/group/batch combinations and reproducibility\n";
        std::cout<<"[CPU VERIFY] ALL PASS; assertions="<<assertions<<"\n";
        return EXIT_SUCCESS;
    } catch(const std::exception& e) {
        std::cerr<<"[CPU VERIFY] FAIL: "<<e.what()<<"\n";
        return EXIT_FAILURE;
    }
}
