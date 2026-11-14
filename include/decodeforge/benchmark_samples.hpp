#pragma once
#include "decodeforge/core.hpp"
#include "decodeforge/gpu.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace decodeforge {
inline void write_autotune_trace_header(std::ostream& out) {
    out << "n,k,batch,group,run,trial,first_kernel,direct_ms,grouped_ms,direct_over_grouped,selected,requested_warmup,requested_repetitions,actual_warmup,actual_repetitions,autotune_wall_ms\n";
    if(!out) throw std::runtime_error("autotune trace CSV header write failed");
}
inline void write_autotune_trace_rows(std::ostream& out,const Shape& shape,const AutotuneTrace& trace,int run) {
    shape.validate();
    if(run<0 || trace.requested_warmup<0 || trace.requested_repetitions<1 || trace.requested_repetitions>100000 ||
       trace.actual_warmup!=std::max(trace.requested_warmup,10) ||
       trace.actual_repetitions!=std::max(trace.requested_repetitions,50) ||
       (trace.selected!=Kernel::Direct && trace.selected!=Kernel::Grouped) ||
       !std::isfinite(trace.wall_ms) || trace.wall_ms<0)
        throw std::invalid_argument("invalid autotune trace metadata");
    for(const auto& sample:trace.samples)
        if(!std::isfinite(sample.direct_ms) || !std::isfinite(sample.grouped_ms) ||
           sample.direct_ms<=0 || sample.grouped_ms<=0)
            throw std::invalid_argument("invalid autotune trace timing");
    const auto flags=out.flags(); const auto precision=out.precision();
    for(std::size_t i=0;i<trace.samples.size();++i) {
        const auto& s=trace.samples[i];
        out << shape.n << ',' << shape.k << ',' << shape.batch << ',' << shape.group << ',' << run << ',' << (i+1) << ','
            << (s.direct_first?"direct":"grouped_paired") << ',' << std::defaultfloat
            << std::setprecision(std::numeric_limits<float>::max_digits10)
            << s.direct_ms << ',' << s.grouped_ms << ',' << s.direct_ms/s.grouped_ms << ','
            << (trace.selected==Kernel::Direct?"direct":"grouped_paired") << ','
            << trace.requested_warmup << ',' << trace.requested_repetitions << ','
            << trace.actual_warmup << ',' << trace.actual_repetitions << ','
            << std::setprecision(std::numeric_limits<double>::max_digits10) << trace.wall_ms << '\n';
    }
    out.flags(flags); out.precision(precision);
    if(!out) throw std::runtime_error("autotune trace CSV write failed");
}
struct AblationSample {
    float paired_ms=0;
    float scalar_ms=0;
    bool paired_first=false;
};
struct ChoiceSample {
    float direct_ms=0;
    float paired_ms=0;
    bool direct_first=false;
};

inline void write_choice_sample_header(std::ostream& out) {
    out << "n,k,batch,group,run,trial,first_kernel,direct_ms,paired_ms,direct_over_paired,selected,autotune_wall_ms,warmup,repeats\n";
    if(!out) throw std::runtime_error("choice sample CSV header write failed");
}
inline void write_choice_sample_rows(std::ostream& out,const Shape& shape,
    const std::array<ChoiceSample,9>& samples,bool selected_direct,double autotune_wall_ms,
    int run,int warmup,int repeats) {
    shape.validate();
    if(!std::isfinite(autotune_wall_ms) || autotune_wall_ms<0 || run<0 || warmup<0 || repeats<1)
        throw std::invalid_argument("invalid choice sample metadata");
    for(const auto& sample:samples)
        if(!std::isfinite(sample.direct_ms) || !std::isfinite(sample.paired_ms) ||
           !(sample.direct_ms>0.f) || !(sample.paired_ms>0.f) ||
           !std::isfinite(sample.direct_ms/sample.paired_ms))
            throw std::invalid_argument("invalid choice sample timing");
    const auto flags=out.flags();
    const auto precision=out.precision();
    for(std::size_t i=0;i<samples.size();++i) {
        const auto& s=samples[i];
        out << shape.n << ',' << shape.k << ',' << shape.batch << ',' << shape.group << ','
            << run << ',' << (i+1) << ',' << (s.direct_first?"direct":"grouped_paired") << ','
            << std::defaultfloat << std::setprecision(std::numeric_limits<float>::max_digits10)
            << s.direct_ms << ',' << s.paired_ms << ',' << s.direct_ms/s.paired_ms << ','
            << (selected_direct?"direct":"grouped_paired") << ','
            << std::setprecision(std::numeric_limits<double>::max_digits10) << autotune_wall_ms << ','
            << warmup << ',' << repeats << '\n';
    }
    out.flags(flags); out.precision(precision);
    if(!out) throw std::runtime_error("choice sample CSV write failed");
}

inline void write_ablation_sample_header(std::ostream& out) {
    out << "n,k,batch,group,trial,first_kernel,paired_ms,scalar_ms,scalar_over_paired\n";
    if(!out) throw std::runtime_error("raw sample CSV header write failed");
}

inline void write_ablation_sample_rows(std::ostream& out, const Shape& shape,
                                       const std::array<AblationSample,7>& samples) {
    shape.validate();
    // Validate all rows before writing so bad timings never produce partial rows.
    for(const auto& sample:samples)
        if(!std::isfinite(sample.paired_ms) || !std::isfinite(sample.scalar_ms) ||
           !(sample.paired_ms>0.f) || !(sample.scalar_ms>0.f) ||
           !std::isfinite(sample.scalar_ms/sample.paired_ms))
            throw std::invalid_argument("invalid raw ablation timing");
    const auto old_flags=out.flags();
    const auto old_precision=out.precision();
    out << std::defaultfloat << std::setprecision(std::numeric_limits<float>::max_digits10);
    for(std::size_t i=0;i<samples.size();++i) {
        const auto& sample=samples[i];
        out << shape.n << ',' << shape.k << ',' << shape.batch << ',' << shape.group << ',' << (i+1) << ','
            << (sample.paired_first ? "grouped_paired" : "grouped_scalar") << ','
            << sample.paired_ms << ',' << sample.scalar_ms << ',' << sample.scalar_ms/sample.paired_ms << '\n';
    }
    out.flags(old_flags);
    out.precision(old_precision);
    if(!out) throw std::runtime_error("raw sample CSV write failed");
}
// Create exclusively: never truncate a report from an earlier experiment.
inline void write_new_ablation_sample_file(const std::filesystem::path& path,const std::string& bytes) {
    std::FILE* file=nullptr;
#ifdef _WIN32
    const auto native=path.wstring();
    const auto error=_wfopen_s(&file,native.c_str(),L"wbx");
    if(error!=0 || !file) throw std::runtime_error("cannot create new raw sample CSV");
#else
    file=std::fopen(path.c_str(),"wbx");
    if(!file) throw std::runtime_error("cannot create new raw sample CSV");
#endif
    const bool written=std::fwrite(bytes.data(),1,bytes.size(),file)==bytes.size();
    const bool closed=std::fclose(file)==0;
    if(!written || !closed) throw std::runtime_error("raw sample CSV write or close failed");
}
} // namespace decodeforge
