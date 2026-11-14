#include "decodeforge/core.hpp"
#include "decodeforge/gpu.hpp"
#include "decodeforge/benchmark_samples.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

using namespace decodeforge;
namespace {
struct Options {
    std::string command;
    int n=1024,k=2048,batch=1,group=64,device=0,repeats=100,warmup=15;
    std::uint64_t seed=20261106;
    std::string csv, samples_csv, choice_samples_csv, tune_samples_csv;
    int sample_run=0;
};
void help() {
    std::cout << "DecodeForge v1.0.0-rc1 | C++17/CUDA W4A16 -> FP32 (no Python)\n"
              << "Usage:\n"
              << "  decodeforge_cli cpu-check [--n 23 --k 129 --batch 4 --group 64]\n"
              << "  decodeforge_cli device\n"
              << "  decodeforge_cli gpu-check [--n ... --k ... --batch ... --group ... --device 0]\n"
              << "  decodeforge_cli bench [--n 1024 --k 2048 --batch 1 --group 64] [--repeats 100]"
                 " [--warmup 15] [--device 0] [--csv file.csv] [--samples file.csv] [--choice-samples file.csv] [--tune-samples file.csv] [--sample-run 0]\n"
              << "  decodeforge_cli bench-all [--device 0] [--csv report.csv]\n"
              << "  --batch: 1/2/4/8; --group: 16/32/64/128; default CUDA_ARCHITECTURES=89 (override via CMake)\n";
}
int positive_int(const std::string& text,const char* label,int max=100000) {
    std::size_t pos=0;
    const auto n=std::stoll(text,&pos,10);
    if(pos!=text.size() || n<0 || n>max) throw std::invalid_argument(std::string("invalid ")+label);
    return static_cast<int>(n);
}
Options parse(int argc,char** argv) {
    if(argc<2) { help(); throw std::invalid_argument("command required"); }
    Options o; o.command=argv[1];
    for(int i=2;i<argc;++i) {
        const std::string key(argv[i]);
        if(i+1>=argc) throw std::invalid_argument("missing value for "+key);
        const std::string value(argv[++i]);
        if(key=="--n") o.n=positive_int(value,"n",65536);
        else if(key=="--k") o.k=positive_int(value,"k",65536);
        else if(key=="--batch") o.batch=positive_int(value,"batch",100);
        else if(key=="--group") o.group=positive_int(value,"group",1000);
        else if(key=="--device") o.device=positive_int(value,"device",16);
        else if(key=="--repeats") o.repeats=positive_int(value,"repeats");
        else if(key=="--warmup") o.warmup=positive_int(value,"warmup");
        else if(key=="--seed") {std::size_t consumed=0; o.seed=std::stoull(value,&consumed,10); if(consumed!=value.size()) throw std::invalid_argument("invalid seed");}
        else if(key=="--csv") o.csv=value;
        else if(key=="--samples") o.samples_csv=value;
        else if(key=="--choice-samples") o.choice_samples_csv=value;
        else if(key=="--tune-samples") o.tune_samples_csv=value;
        else if(key=="--sample-run") o.sample_run=positive_int(value,"sample-run",1000000000);
        else throw std::invalid_argument("unknown argument: "+key);
    }
    if(o.repeats<1 || o.repeats>100000) throw std::invalid_argument("--repeats must be 1..100000");
    return o;
}
void require_equal(const std::vector<float>& a,const std::vector<float>& b,const std::string& what) {
    const auto err=compare_outputs(a,b,0.006,0.003);
    if(err.mismatched) throw std::runtime_error(what+": "+std::to_string(err.mismatched)+" values differ; max_abs="+std::to_string(err.max_abs));
}
struct Result { Shape shape; float baseline,direct,grouped,grouped_scalar,paired_speedup,graph; Kernel best; double e2e_direct,e2e_grouped,e2e_baseline,host_selected,host_graph; std::array<AblationSample,7> samples; std::array<ChoiceSample,9> choices; double tune_wall_ms; AutotuneTrace tune_trace; };
Result bench_one(const Shape& s,const Options& o) {
    const auto in=make_random_inputs(s,o.seed);
    const auto expected=cpu_reference(in);
    GpuSession gpu(in,o.device);
    for(Kernel k:{Kernel::Direct,Kernel::Grouped,Kernel::GroupedScalar,Kernel::TwoPassCublas}) {
        gpu.run(k);
        require_equal(expected,gpu.download(),kernel_name(k));
    }
    const float a=gpu.benchmark_ms(Kernel::Direct,o.warmup,o.repeats);
    // Paired alternating-order experiment: the median of seven per-trial
    // ratios is less sensitive to drift than one fixed-order A/B sample.
    std::array<float,7> packed_samples{},scalar_samples{},paired_ratios{};
    std::array<AblationSample,7> samples{};
    for(std::size_t i=0;i<paired_ratios.size();++i) {
        if(i%2==0) {
            packed_samples[i]=gpu.benchmark_ms(Kernel::Grouped,o.warmup,o.repeats);
            scalar_samples[i]=gpu.benchmark_ms(Kernel::GroupedScalar,o.warmup,o.repeats);
        } else {
            scalar_samples[i]=gpu.benchmark_ms(Kernel::GroupedScalar,o.warmup,o.repeats);
            packed_samples[i]=gpu.benchmark_ms(Kernel::Grouped,o.warmup,o.repeats);
        }
        if(!(packed_samples[i]>0.f) || !(scalar_samples[i]>0.f))
            throw std::runtime_error("nonpositive ablation timing");
        paired_ratios[i]=scalar_samples[i]/packed_samples[i];
        samples[i]={packed_samples[i],scalar_samples[i],i%2==0};
    }
    std::sort(packed_samples.begin(),packed_samples.end());
    std::sort(scalar_samples.begin(),scalar_samples.end());
    std::sort(paired_ratios.begin(),paired_ratios.end());
    const float b=packed_samples[packed_samples.size()/2];
    const float scalar=scalar_samples[scalar_samples.size()/2];
    const float paired_speedup=paired_ratios[paired_ratios.size()/2];
    const float base=gpu.benchmark_ms(Kernel::TwoPassCublas,o.warmup,o.repeats);
    AutotuneTrace tune_trace{};
    const auto tune_start=std::chrono::steady_clock::now();
    const int tune_repetitions=std::max(5,std::min(o.repeats,25));
    const Kernel best=o.tune_samples_csv.empty()?gpu.autotune(2,tune_repetitions):gpu.autotune(2,tune_repetitions,tune_trace);
    const double tune_wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tune_start).count();
    gpu.run_auto();
    require_equal(expected,gpu.download(),"autotune path");
    std::array<ChoiceSample,9> choices{};
    if(!o.choice_samples_csv.empty()) {
        for(std::size_t i=0;i<choices.size();++i) {
            const bool direct_first=(i+static_cast<std::size_t>(o.sample_run))%2==0;
            float direct=0.f,paired=0.f;
            if(direct_first) {
                direct=gpu.benchmark_ms(Kernel::Direct,o.warmup,o.repeats);
                paired=gpu.benchmark_ms(Kernel::Grouped,o.warmup,o.repeats);
            } else {
                paired=gpu.benchmark_ms(Kernel::Grouped,o.warmup,o.repeats);
                direct=gpu.benchmark_ms(Kernel::Direct,o.warmup,o.repeats);
            }
            choices[i]={direct,paired,direct_first};
        }
    }
    gpu.capture_graph(best);
    gpu.launch_graph(2);
    require_equal(expected,gpu.download(),"graph correctness");
    const float graph=gpu.benchmark_graph_ms(o.warmup,o.repeats);
    // Wall time includes session setup, H2D, one execution, D2H and teardown.
    // Input generation and CPU reference are deliberately outside the interval.
    auto e2e = [&](Kernel k) {
        constexpr int iterations=5;
        std::vector<float> output;
        const auto start=std::chrono::steady_clock::now();
        for(int i=0;i<iterations;++i) {
            GpuSession fresh(in,o.device);
            fresh.run(k);
            output=fresh.download();
        }
        const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/iterations;
        require_equal(expected,output,"end-to-end path");
        return elapsed;
    };
    const double e2e_direct=e2e(Kernel::Direct);
    const double e2e_grouped=e2e(Kernel::Grouped);
    const double e2e_baseline=e2e(Kernel::TwoPassCublas);
    // Compare synchronized host launch overhead with identical resident buffers;
    // graph capture and input/output transfers are excluded from both paths.
    auto host_time = [&](bool use_graph) {
        const auto start=std::chrono::steady_clock::now();
        for(int i=0;i<o.repeats;++i) {
            if(use_graph) gpu.launch_graph(1); else gpu.run(best);
        }
        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/o.repeats;
    };
    const double host_selected=host_time(false), host_graph=host_time(true);
    std::cout<<describe_shape(s) <<" | two_pass_fp32="<<base<<"ms"
             <<" direct="<<a<<"ms grouped_paired="<<b<<"ms grouped_scalar="<<scalar<<"ms"
             <<" paired_vs_scalar="<<paired_speedup<<"x selected="<<kernel_name(best)<<" graph="<<graph<<"ms"
             <<" relative_to_unfused="<<base/(best==Kernel::Direct?a:b)<<"x\n";
    return {s,base,a,b,scalar,paired_speedup,graph,best,e2e_direct,e2e_grouped,e2e_baseline,host_selected,host_graph,samples,choices,tune_wall_ms,tune_trace};
}
void write_samples_csv(const std::string& path,const std::vector<Result>& results) {
    if(path.empty()) return;
    std::ostringstream content;
    write_ablation_sample_header(content);
    for(const auto& r:results) write_ablation_sample_rows(content,r.shape,r.samples);
    write_new_ablation_sample_file(std::filesystem::u8path(path),content.str());
    std::cout<<"[RAW SAMPLES] "<<path<<"\n";
}
void write_choice_csv(const Options& o,const std::vector<Result>& results) {
    if(o.choice_samples_csv.empty()) return;
    std::ostringstream content;
    write_choice_sample_header(content);
    for(const auto& r:results)
        write_choice_sample_rows(content,r.shape,r.choices,r.best==Kernel::Direct,r.tune_wall_ms,o.sample_run,o.warmup,o.repeats);
    write_new_ablation_sample_file(std::filesystem::u8path(o.choice_samples_csv),content.str());
    std::cout<<"[CHOICE SAMPLES] "<<o.choice_samples_csv<<"\n";
}
void write_tune_csv(const Options& o,const std::vector<Result>& results) {
    if(o.tune_samples_csv.empty()) return;
    std::ostringstream content;
    write_autotune_trace_header(content);
    for(const auto& r:results) write_autotune_trace_rows(content,r.shape,r.tune_trace,o.sample_run);
    try { write_new_ablation_sample_file(std::filesystem::u8path(o.tune_samples_csv),content.str()); }
    catch(const std::exception& e) {throw std::runtime_error(std::string("autotuning completed, but internal trace write failed: ")+e.what());}
    std::cout<<"[TUNE SAMPLES] "<<o.tune_samples_csv<<"\n";
}
void write_csv(const std::string& path, const std::vector<Result>& results) {
    if(path.empty()) return;
    std::ofstream f(std::filesystem::u8path(path),std::ios::binary|std::ios::trunc);
    if(!f) throw std::runtime_error("cannot open CSV: "+path);
    f<<"n,k,batch,group,two_pass_fp32_ms,direct_fused_ms,grouped_fused_ms,grouped_scalar_ms,paired_speedup_vs_scalar,selected_kernel,selected_ms,graph_ms,speedup_vs_two_pass_fp32,e2e_direct_ms,e2e_grouped_ms,e2e_two_pass_ms,host_sync_selected_ms,host_sync_graph_ms\n";
    f<<std::fixed<<std::setprecision(6);
    for(const auto& r:results) {
        const float best=r.best==Kernel::Direct?r.direct:r.grouped;
        f<<r.shape.n<<','<<r.shape.k<<','<<r.shape.batch<<','<<r.shape.group<<','
         <<r.baseline<<','<<r.direct<<','<<r.grouped<<','<<r.grouped_scalar<<','<<r.paired_speedup<<','<<kernel_name(r.best)<<','<<best<<','<<r.graph<<','<<r.baseline/best<<','
         <<r.e2e_direct<<','<<r.e2e_grouped<<','<<r.e2e_baseline<<','<<r.host_selected<<','<<r.host_graph<<'\n';
    }
    if(!f) throw std::runtime_error("CSV write failed: "+path);
    std::cout<<"[REPORT] "<<path<<"\n";
}
}
int cli_main(int argc,char** argv) {
    try {
        if(argc==2 && (std::string(argv[1])=="--help" || std::string(argv[1])=="help")) {help();return 0;}
        const auto opt=parse(argc,argv);
        if((!opt.samples_csv.empty() || !opt.choice_samples_csv.empty() || !opt.tune_samples_csv.empty()) && opt.command!="bench" && opt.command!="bench-all")
            throw std::invalid_argument("raw sample options are valid only for bench or bench-all");
        for(const auto& raw:{opt.samples_csv,opt.choice_samples_csv,opt.tune_samples_csv}) {
            if(raw.empty()) continue;
            if(std::filesystem::exists(std::filesystem::u8path(raw)))
                throw std::invalid_argument("raw sample CSV already exists: "+raw);
        }
        const std::array<std::string,4> paths={opt.csv,opt.samples_csv,opt.choice_samples_csv,opt.tune_samples_csv};
        for(std::size_t i=0;i<paths.size();++i) for(std::size_t j=i+1;j<paths.size();++j)
            if(!paths[i].empty() && !paths[j].empty() &&
               std::filesystem::weakly_canonical(std::filesystem::u8path(paths[i]))==
               std::filesystem::weakly_canonical(std::filesystem::u8path(paths[j])))
                throw std::invalid_argument("CSV output paths must name different files");
        Shape s{opt.n,opt.k,opt.batch,opt.group};
        if(opt.command=="cpu-check") {
            s.validate();
            const auto in=make_random_inputs(s,opt.seed);
            const auto direct=cpu_reference(in),baseline=cpu_two_pass_reference(in);
            require_equal(direct,baseline,"CPU two-pass");
            std::cout<<"[CPU] PASS "<<describe_shape(s)<<"\n";
            return 0;
        }
        if(opt.command=="device") {
            if(!cuda_compiled()) throw std::runtime_error("CUDA backend disabled; rebuild with -DDF_ENABLE_CUDA=ON");
            const auto count=visible_gpu_count();
            std::cout<<"Visible GPU count: "<<count<<"\n";
            if(count==0) throw std::runtime_error("no NVIDIA GPUs visible");
            for(int i=0;i<count;++i) std::cout<<i<<": "<<gpu_description(i)<<'\n';
            return 0;
        }
        if(opt.command!="gpu-check" && opt.command!="bench" && opt.command!="bench-all") {
            help(); throw std::invalid_argument("unknown command: "+opt.command);
        }
        if(!cuda_compiled()) throw std::runtime_error("CUDA backend disabled; GPU verification cannot be replaced with CPU fallback");
        if(visible_gpu_count()<=opt.device) throw std::runtime_error("GPU device index not visible; check CUDA_VISIBLE_DEVICES");
        std::cout<<"[DEVICE] "<<gpu_description(opt.device)<<"\n";
        std::cout<<"[NOTE] CUDA-event device execution timing only; excludes H2D/D2H;"
                    " two_pass_fp32 is an unfused algorithmic baseline, NOT an optimized W4A16 library.\n";
        if(opt.command=="gpu-check") {
            s.validate();
            auto in=make_random_inputs(s,opt.seed);
            auto reference=cpu_reference(in);
            GpuSession gpu(in,opt.device);
            for(Kernel k:{Kernel::Direct,Kernel::Grouped,Kernel::GroupedScalar,Kernel::TwoPassCublas}) {
                gpu.run(k); require_equal(reference,gpu.download(),kernel_name(k));
                std::cout<<"[GPU] PASS "<<kernel_name(k)<<"\n";
            }
            gpu.autotune(2,15);
            gpu.run_auto();
            require_equal(reference,gpu.download(),"autotuned run");
            std::cout<<"[GPU] PASS autotune dispatcher\n";
            gpu.capture_graph(Kernel::Grouped);
            gpu.launch_graph(2);
            require_equal(reference,gpu.download(),"CUDA graph");
            std::cout<<"[GPU] PASS CUDA graph\n[GPU] ALL PASS\n";
            return 0;
        }
        std::vector<Result> r;
        std::cout<<std::fixed<<std::setprecision(5);
        if(opt.command=="bench") {
            s.validate(); r.push_back(bench_one(s,opt));
        } else {
            for(const int batch:{1,2,4,8}) {
                for(const int k:{256,1024,4096}) {
                    const int n=k==4096?2048:1024;
                    r.push_back(bench_one(Shape{n,k,batch,64},opt));
                }
            }
        }
        write_samples_csv(opt.samples_csv,r);
        write_choice_csv(opt,r);
        write_tune_csv(opt,r);
        // On case-insensitive filesystems or through directory links, distinct
        // spellings can still refer to the same file.  Do not let the legacy
        // summary writer truncate the newly created raw sample report.
        for(const auto& raw:{opt.samples_csv,opt.choice_samples_csv,opt.tune_samples_csv})
            if(!raw.empty() && !opt.csv.empty() && std::filesystem::exists(std::filesystem::u8path(opt.csv)) &&
               std::filesystem::equivalent(std::filesystem::u8path(raw),std::filesystem::u8path(opt.csv)))
                throw std::invalid_argument("raw sample and summary CSV refer to the same file");
        write_csv(opt.csv,r);
        std::cout<<"[GPU BENCH] PASS (includes GPU/CPU numerical correctness checks)\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"[ERROR] "<<e.what()<<"\n";
        return 1;
    }
}
#ifdef _WIN32
int wmain(int argc,wchar_t** argv) {
    std::vector<std::string> utf8_args;
    utf8_args.reserve(static_cast<std::size_t>(argc));
    for(int i=0;i<argc;++i) {
        const int size=WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,argv[i],-1,nullptr,0,nullptr,nullptr);
        if(size<=0) { std::cerr<<"[ERROR] invalid Windows argument encoding\n"; return 1; }
        std::string value(static_cast<std::size_t>(size),'\0');
        if(!WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,argv[i],-1,value.data(),size,nullptr,nullptr)) return 1;
        value.pop_back();
        utf8_args.push_back(std::move(value));
    }
    std::vector<char*> args;
    args.reserve(utf8_args.size());
    for(auto& arg:utf8_args) args.push_back(arg.data());
    return cli_main(argc,args.data());
}
#else
int main(int argc,char** argv) { return cli_main(argc,argv); }
#endif
