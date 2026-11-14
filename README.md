# DecodeForge v1.0.0-rc1 — C++17/CUDA 量化融合算子与可复现性能实验

## Windows 11 / Visual Studio 2026

下方 Windows 示例以当前工程 `E:\c++\项目\DecodeForge_v1.0.0-rc1` 为例；换用其他目录时替换 `cd` 即可。每次验证均创建新的构建和日志目录。

在 PowerShell 中运行 `./scripts/verify_windows.ps1 -CudaToolkitRoot <CUDA工具包目录>`，脚本使用 VS 2026 x64 和 Ninja 新建独立 Release 构建，执行 CPU/GPU 测试、基准，并在存在 Compute Sanitizer 时执行 memcheck 和 racecheck。仅 CPU 验证使用 `./scripts/verify_windows.ps1 -CpuOnly`。可以加 `-SkipSanitizer` 先编译、做数值与普通基准，再用 `scripts/verify_windows_sanitizer.ps1` 对相同构建运行4项完整Sanitizer。`-SkipSanitizer` **不代表安全验收通过**。每次运行生成单独的 `build/windows-*` 与 `validation/windows-*` 目录。需要 CUDA 与 MSVC 版本受 NVIDIA 支持；GPU测试不能由CPU fallback代替。

本机现有工具链为 `E:\DecodeForge-rc1-toolkit`，脚本可以自动检测；不改变系统执行策略的前台命令为 `pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify_windows.ps1`。其他机器可显式传入 `-CudaToolkitRoot`。

### Windows 本机：两阶段验证 + 消融实验

```powershell
cd 'E:\c++\项目\DecodeForge_v1.0.0-rc1'
$cuda='E:\DecodeForge-rc1-toolkit' # 本机现有 CUDA 13.4 工具包路径；其他电脑请替换
.\scripts\verify_windows.ps1 -CudaToolkitRoot $cuda -SkipSanitizer
# 将脚本 configure 输出中的实际 build/windows-* 路径复制到这里：
$build='<上一步打印的构建目录>'
# 使用这个已验证的新构建进行完整Sanitizer和12 Shape消融
.\scripts\verify_windows_sanitizer.ps1 -BuildDirectory $build -CudaToolkitRoot $cuda
.\scripts\benchmark_windows_ablation.ps1 -BuildDirectory $build -CudaToolkitRoot $cuda
```

如racecheck中断或报错，**不允许声明通过**。新消融CSV有 `grouped_scalar_ms`、`grouped_fused_ms`、`paired_speedup_vs_scalar`，后者表示旧Scalar与新Paired对应配对采样耗时比的中位数，**大于1才表示该Shape上优化版有优势**。空闲GPU测量，多次重跑检查波动。

`benchmark_windows_ablation.ps1` 保存每次运行的汇总；如需每对 Scalar/Paired 的原始记录，使用 `benchmark_windows_matrix.ps1` 或 CLI 的 `--samples`。汇总中的 Direct 和 Paired 采用不同时间窗口，不应直接用于声称调优器的准确率。

W4A16 CSV 中 `*_ms` 的 kernel/graph 数值是 CUDA Event 测得的设备流耗时；`e2e_*` 是新建会话、输入上传、执行、输出下载和会话销毁的主机墙钟时间（不含输入生成及 CPU 参考）；`host_sync_*` 比较已驻留数据时逐次同步的普通执行和 Graph 重放，不含 Graph 捕获及数据传输。端到端测试每个路径重复 5 次，主机同步测试的次数由 `--repeats` 决定。性能结果依赖设备负载、温度和频率，不能当作跨机器保证。

`bench` 和 `bench-all` 可额外指定 `--samples <新文件.csv>` 保存每个 shape 的 7 组原始 Scalar/Paired 数据。每行依次为 `n,k,batch,group,trial,first_kernel,paired_ms,scalar_ms,scalar_over_paired`；`trial` 是从 1 开始的采样顺序，`first_kernel` 标明该组先测的 kernel，时间单位为毫秒。该文件用排他创建，已有文件会报错，不会覆盖；`--csv` 汇总格式和既有行为不变。原始采样不包括 Direct 或自动调优器的九组数据，也不能单独证明调优准确性。

可额外指定 `--choice-samples <新文件.csv> --sample-run <非负编号>`，在同一 GPU 会话调优完成后独立交替采集九对 Direct/Grouped Paired；偶数 run 首对先测 Direct，奇数 run 首对先测 Paired。字段为 `n,k,batch,group,run,trial,first_kernel,direct_ms,paired_ms,direct_over_paired,selected,autotune_wall_ms,warmup,repeats`；时间单位为毫秒，调优墙钟时间只包围 `autotune` 调用，成对 CUDA Event 测量包含可能的主机提交间隙。这些是调优后的独立参考采样，并非调优器内部九对样本。输出文件排他创建，不覆盖已有文件；实验仍须独立重复并记录 GPU 负载、频率及运行环境。既有汇总 `selected_ms` 的测量口径未改变。

可选 `--tune-samples <新文件.csv>` 记录调优器内部实际九对采样、执行顺序、请求和实际预热／重复次数、选择及墙钟耗时；它与调优后的 `--choice-samples` 是不同数据。只有显式指定时采集，默认调优策略不变。记录文件独占创建，写入失败可能留下不完整的新文件；多份报告不是事务式写入。`--sample-run` 只是记录标签，不改变内部调优的首测顺序。

**版本状态：Release Candidate，不是已认证的正式 v1.0.0。** 当前验收状态、构建版本和原始证据见 [`docs/RESUME_READINESS.md`](docs/RESUME_READINESS.md)。性能依赖设备负载、温度和频率，不能将一次烟测视作稳定加速。

可随源码展示的 RTX 4060 数值与基准实验摘要和目标 Shape 原始采样位于 [`docs/evidence/rtx4060-windows-20261113`](docs/evidence/rtx4060-windows-20261113/README.md)。完整本机日志位于默认不纳入仓库的 `validation/`；这里的未融合 FP32/cuBLAS 两阶段只作为项目内部参考，不能代表对成熟量化算子的比较。

新增 `decodeforge_accuracy_cli` 对 64 组形状和四条 GPU 路径输出实际最大误差及不匹配数；可用 `pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\accuracy_windows.ps1 -BuildDirectory <新GPU构建目录> -CudaToolkitRoot E:\DecodeForge-rc1-toolkit` 保存原始 CSV。CPU-only 可执行文件会明确失败，不会伪装为 GPU 测试。

专注可变 Batch 的 INT4 权重 × FP16 激活计算，CPU/GPU 均使用 **FP32 输出**。不使用 Python、Agent 或模型推理服务，不依赖 PyTorch/Triton。

本工程是算子优化技术项目，不是完整 LLM 推理框架，也不是生产级推理库。Linux 和其他 GPU 的结果须以各环境中重新执行的验收为准。

## 设计与功能

| 模块 | 实际内容 | 验证方式 |
|---|---|---|
| CPU W4A16格式 | signed INT4 二补码打包；奇数K行填充；分组FP16 Scale；FP16激活；FP32输出 | CPU CTest |
| FP16转换 | 完整 IEEE binary16 65536 位型测试，round-to-nearest/ties-to-even | CPU CTest |
| CPU双重参考 | 在线反量化计算，与预展开FP32权重的独立实现比较 | CPU CTest |
| Direct Fused Kernel | 每个CTA处理一个(B,N)输出；INT4就地解包、Warp Reduction | GPU CTest |
| Grouped Fused Kernel | 每个CTA处理一行权重及B个输出，一次解码相邻2个INT4 | GPU CTest |
| Grouped Scalar消融对照 | 保存原始逐nibble的同形状CUDA Kernel，供双INT4优化前后直接比较 | GPU CTest和配对基准 |
| Two-Pass FP32基线 | dequantize kernel + 激活转换 + cuBLAS SGEMM | GPU CTest |
| Shape autotune | 9组交替顺序的Direct/Grouped配对样本，取耗时比中位数；会话内缓存，Scalar基线不参与选择 | GPU CTest；稳定性见验收状态 |
| CUDA Graph | 固定Shape与缓冲区地址的Kernel捕获、重复启动 | GPU CTest |
| FP16/BF16 GEMM | Naive、16×16 shared-memory tile、cuBLAS三种路径（FP32输出） | CPU/GPU数值测试 |
| Residual＋RMSNorm | FP16输入＋FP32输出，融合一核与两阶段对照 | CPU/GPU数值测试 |
| CSV基准 | 12组Shape（4种B×3种K），增加7组交替采样的Paired/Scalar消融与中位数加速比 | GPU实测生成 |
| CUDA Compute Sanitizer | memcheck和racecheck脚本 | GPU真机测试 |

详细格式和正确性：[`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md)。

## Linux 4090 构建（可选，不影响Windows主线）

只选确实空闲、且经授权使用的GPU。显存低不等于GPU无负载。

```bash
cd ~/DecodeForge                  # 切换到项目根目录
nvidia-smi
nvcc --version

# 只让程序看见物理GPU 1；此时CUDA逻辑设备编号是0
export CUDA_VISIBLE_DEVICES=1

# 先跑CPU核心与工具链验证
# 一次执行CPU、CUDA正确性、Graph、GEMM、RMSNorm、性能与Sanitizer全套验收
bash scripts/verify_all_gpu.sh
```

**如果CUDA 12.0与系统gcc版本不兼容：** 首先运行 `g++ --version` 和 `ls /usr/bin/g++-*`，选存在的 CUDA Toolkit 支持的主机编译器，例如环境中装有 `g++-12` 时使用：

```bash
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12
bash scripts/verify_gpu.sh
```

这些设置只影响当前终端。不要安装 `--allow-unsupported-compiler` 来掩盖兼容性问题。

## 直接编译命令

若使用 `--csv benchmarks/...`，先运行 `mkdir -p benchmarks`。

**仅 CPU：**

```bash
cmake -S . -B build/cpu -DDF_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu -j 8
ctest --test-dir build/cpu --output-on-failure
./build/cpu/decodeforge_cli cpu-check --n 23 --k 129 --batch 4 --group 64
```

**GPU：**

```bash
cmake -S . -B build/gpu -DDF_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/gpu -j 8
ctest --test-dir build/gpu --output-on-failure
./build/gpu/decodeforge_cli device
./build/gpu/decodeforge_cli gpu-check --n 133 --k 1025 --batch 8 --group 64
./build/gpu/decodeforge_cli bench --n 4096 --k 4096 --batch 1 --group 64 --csv benchmarks/b1.csv
./build/gpu/decodeforge_cli bench-all --csv benchmarks/all.csv
./build/gpu/decodeforge_ops_cli gemm-bench --m 256 --n 256 --k 512 --dtype fp16 --csv benchmarks/gemm_fp16.csv
./build/gpu/decodeforge_ops_cli gemm-bench --m 256 --n 256 --k 512 --dtype bf16 --csv benchmarks/gemm_bf16.csv
./build/gpu/decodeforge_ops_cli norm-bench --rows 128 --cols 1024 --csv benchmarks/norm.csv
```

**CPU构建运行gpu-check会以非零返回码报错**，不会伪装成GPU测试通过。

## RTX 4060 Laptop（Windows WSL2 Ubuntu）

使用 WSL2 的 Ubuntu、NVIDIA Windows 驱动以及 WSL2 内的 CUDA Toolkit（不要在WSL中另外安装Linux GPU驱动）。执行上述GPU构建命令，RTX 4060与4090均为SM 8.9。具体CUDA Toolkit、Visual Studio/主机编译器与驱动兼容性按NVIDIA官方文档安装与验证。

## 基准结果怎么解释？

`two_pass_fp32` 是 **INT4→FP32解包、FP16→FP32激活转换 + cuBLAS SGEMM** 的未融合参考；不是经过Tensor Core优化的W4A16工业基线，不可以用其加速倍数宣称超过 TensorRT-LLM/FlashInfer/cuBLASLt。

`direct/grouped`是两个自研候选Kernel；GPU Event耗时排除数据生成、H2D/D2H、GPU内存分配，但事件间可能包含Host发射空档；CUDA Graph数值单列。优化收益在同一GPU、同一Shape、同一功耗和负载状态下比较；报告具体GPU型号、时钟/温度/功耗及编译版本。4090 GPU0繁忙时不要使用它做性能结论。

项目只在实测后填写数值，**不会附带伪造的4090/4060跑分**。参见[`docs/TEST_PLAN.md`](docs/TEST_PLAN.md)。

## 目录

```text
DecodeForge/
  CMakeLists.txt
  include/decodeforge/half.hpp, core.hpp, gpu.hpp
  src/core.cpp             FP16、INT4、量化与CPU参考
  src/gpu.cu               真CUDA Kernel、cuBLAS、CUDA Graph、事件计时
  src/gpu_stub.cpp         CPU-only模式下的显式失败桩
  src/main.cpp             CLI、正确性检查、CSV benchmark
  src/ops_cpu.cpp          BF16、FP16/BF16 GEMM、RMSNorm CPU参考
  src/ops_gpu.cu           Shared-memory GEMM、cuBLAS、RMSNorm融合Kernel
  src/ops_cli.cpp          额外算子的正确性测试与基准CLI
  tests/cpu_test.cpp       65,536 FP16位模式+272参数组合+金标准
  tests/ops_cpu_test.cpp   65,536 BF16位模式+160 GEMM+27 RMSNorm
  tests/gpu_test.cpp       30组GPU形状+候选Kernel与Graph+动态选择器
  tests/ops_gpu_test.cpp   FP16/BF16 GEMM、Residual+RMSNorm真实GPU验证
  scripts/verify_cpu.sh    CPU交付验收
  scripts/verify_gpu.sh    4090/4060真机验收
  scripts/verify_cpu_sanitizers.sh
  scripts/verify_gpu_compute_sanitizer.sh
  docs/DATA_FORMAT.md, TEST_PLAN.md, DESIGN.md, RESUME.md
```

**系统要求：** CMake 3.21+、C++17、CUDA Toolkit 12.0+（GPU构建）、NVIDIA Ada SM89显卡。


## GPU最终验收（重要）

参阅 [`docs/RESUME_READINESS.md`](docs/RESUME_READINESS.md)，其中区分已通过检查和未完成的验收。`scripts/verify_all_gpu.sh` 对W4A16、FP16/BF16 GEMM、融合RMSNorm及Compute Sanitizer做单次完整验收，并记录设备环境。

如使用CPU-only构建，CUDA接口会显式返回非零退出码，不会冒充成功。仓库包含 `.github/workflows/cpu-ci.yml`，仅提供普通GitHub runner可运行的CPU持续集成。
