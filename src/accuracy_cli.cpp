#include "decodeforge/core.hpp"
#include "decodeforge/gpu.hpp"
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>

// This report uses exactly the same seed and CPU reference as the benchmark.
// CSV goes to stdout; diagnostics go to stderr and failures return nonzero.
int main() {
    using namespace decodeforge;
    try {
        if(!cuda_compiled() || visible_gpu_count()<1)
            throw std::runtime_error("a real CUDA device is required");
        constexpr double atol=0.006,rtol=0.003;
        constexpr std::uint64_t seed=20261106;
        std::cerr << "Device: " << gpu_description(0) << '\n';
        std::cout << "n,k,batch,group,seed,path,atol,rtol,compared,mismatched,max_abs,max_relative_floor_1e-6\n";
        std::cout << std::setprecision(17);
        std::size_t total_bad=0;
        for(const int group:{16,32,64,128}) for(const int batch:{1,2,4,8}) {
            for(const Shape s:{Shape{65,129,batch,group},Shape{256,256,batch,group},
                               Shape{1024,1024,batch,group},Shape{2048,4096,batch,group}}) {
                const auto input=make_random_inputs(s,seed);
                const auto expected=cpu_reference(input);
                GpuSession gpu(input,0);
                for(const Kernel k:{Kernel::Direct,Kernel::Grouped,Kernel::GroupedScalar,Kernel::TwoPassCublas}) {
                    gpu.run(k);
                    const auto a=compare_outputs(expected,gpu.download(),atol,rtol);
                    std::cout << s.n << ',' << s.k << ',' << s.batch << ',' << s.group << ',' << seed << ','
                              << kernel_name(k) << ',' << atol << ',' << rtol << ',' << a.compared << ','
                              << a.mismatched << ',' << a.max_abs << ',' << a.max_relative << '\n';
                    if(!std::cout) throw std::runtime_error("accuracy CSV write failed");
                    total_bad+=a.mismatched;
                }
            }
        }
        std::cerr << "Accuracy report complete; mismatches=" << total_bad << '\n';
        return total_bad ? EXIT_FAILURE : EXIT_SUCCESS;
    } catch(const std::exception& e) {
        std::cerr << "Accuracy report failed: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
