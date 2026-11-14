# 2026-11-12 W4A16 实验记录

本报告记录实际运行结果，便于复查，不代表正式 GPU 资格验收通过。实验比较的是 DecodeForge 内部实现；未测量 CUTLASS、TensorRT-LLM 或其他工业 W4A16 kernel，不能声称领先行业实现。两阶段 FP32/cuBLAS 是本项目非融合的算法参考，其数据类型、计算和实现与优化的 W4A16 实现不同。

## 环境与原始文件

- Windows，NVIDIA GeForce RTX 4060 Laptop GPU，8 GB，SM89；驱动 616.56，CUDA 13.4.92。20 次实验与 64 shape 实验使用 `build/windows-20261112-143110-3019/decodeforge_cli.exe`，工具链路径记录为 `E:\DecodeForge-rc1-toolkit`。具体版本、环境快照见各自 `environment.txt`。
- 显卡未锁频，顺序执行；开始时分别记录 47°C、45°C 和 GPU 利用率 0%，这只是开始时的快照，并未连续监控频率、温度、功耗或其他进程。无法排除后续干扰和动态时钟影响。
- 单一目标 shape 的 20 轮：[`validation/matrix-raw-20261112-143249-5106099`](../validation/matrix-raw-20261112-143249-5106099)，原始 `*-choice.csv`、`*-samples.csv`、各次日志及 `summary.csv` 均保留。统计文件：[`choice-analysis-20261112-143307-9054332.csv`](../validation/matrix-raw-20261112-143249-5106099/choice-analysis-20261112-143307-9054332.csv)。
- 64 shape 各两轮：[`validation/matrix-raw-20261112-143315-5233629`](../validation/matrix-raw-20261112-143315-5233629)，统计文件：[`choice-analysis-20261112-143458-3427182.csv`](../validation/matrix-raw-20261112-143315-5233629/choice-analysis-20261112-143458-3427182.csv)。两个实验各自的原始日志是计算依据。

## 计时口径

对每次调用，同一组随机输入用于数值校验与测量，shape 的默认种子为 20261106。Direct/Paired 独立验证在调优之后、同一个 GPU session 内进行 9 对，每对轮换顺序；通过 run 编号轮换首对顺序。每个候选的验证计时使用 20 次 warmup 和 100 次重复，CUDA Event 跨整段重复调用计时再除以重复次数；它不包括 H2D/D2H，但可能包含主机提交不及时造成的设备空隙，不能直接等同纯 kernel 指令执行时间。

`*-choice.csv` 是**调优后独立测量**，不是调优器内部 9 对原始计时。当前自动调优本身是另一次测量，实际使用 10 次 warmup、50 次重复和 9 对交替采样，以 Direct/Paired 耗时比中位数选择；内部 9 对原始样本未保存。独立参考同样以九个 `direct_ms / paired_ms` 比值的中位数判定胜者；所选路径与独立参考相同记一次一致。其独立性指不使用调优阶段采样，不表示对噪声免疫或代表绝对真实最优。

若选择不一致，观察损失定义为相对于独立参考胜者的延迟增幅：中位比为 R、选择 Direct 时为 `(R-1)*100%`，选择 Paired 时为 `(1/R-1)*100%`；一致时记 0。此口径由 [`scripts/analyze_windows_choice.ps1`](../scripts/analyze_windows_choice.ps1) 实现，表示该次独立测量的观察量，不是对未来损失的上界。1% 接近持平标记仅是描述阈值，不是经过校准的测量置信区间。

`*-samples.csv` 保存另一项独立的 7 对 Grouped Scalar/Grouped Paired 测量，交替顺序，汇总列 `paired_speedup_vs_scalar` 是每对 `scalar_ms/paired_ms` 比值的中位数，小于 1 表示该次采样的 Paired 较慢。原汇总 CSV 中 `selected_ms` 复用分别采集的 Direct 或 Paired 结果，并不是重新计时的 selected，也不应用作成对测量真值。

Graph host 对照是同一已驻留 GPU 会话的同步主机计时，分别对普通选择路径与 graph replay 调用按次数取平均；graph 捕获及输入输出传输均排除。该口径不等于设备端 Event 时间或真实推理端到端吞吐。

## 目标 shape：20 次独立进程

N=1024、K=256、Batch=1、Group=64；每次独立进程的九对样本均在 `*-choice.csv` 中。调优器选择与调优后独立参考一致 **11/20**；调优器墙钟耗时中位数 **10.5354 ms**，该时间仅覆盖 `autotune` 调用，不包含独立参考采样。最大观察损失 **14.53%**，发生在第 8 次（记录的精确值约 14.526761%，所选 Direct，独立参考 Paired）。

九对中 19/20 次同时出现过 Direct 和 Paired 各自更快的单对；14/20 次的独立中位比落在 `[1/1.05, 1.05]`。这说明当前采样下局部胜负与微小差距波动明显；它不是估算出的置信区间，也不能证明全部不一致都由测量噪声造成。第 8 次九对均为 Paired 更快，但调优内部未保留样本，无法判断该次差异究竟是时钟漂移、不同重复次数、运行干扰还是其他原因。6/20 次位于脚本定义的 1% 接近持平区间。20 次未锁频结果不证明长期准确率或最大损失上限。

## 64 shape：两轮，共 128 次观测

参数为 Batch 1/2/4/8、Group 16/32/64/128，以及 `(N,K)` = `(65,129)`、`(256,256)`、`(1024,1024)`、`(2048,4096)`，包括奇数 K。每次独立进程均记录了 9 对选择参考及 7 对 Scalar/Paired 样本。

- 自动选择与随后独立九对中位参考相符 **109/128**；6/128 的参考中位比落在脚本的 1% 接近持平区间。由于每个 shape 只有两轮，未锁频且参考本身有噪声，不能将 109/128 当作准确率的可靠总体估计。
- **36/128** 次的 Scalar/Paired 比值中位数小于 1，即该次实验中 Paired 相对于 Scalar 观察到退化。两轮只是少量观测，不能证明所有这些 shape 持续退化。
- **74/128** 次的 Graph 同步主机平均耗时小于普通选择路径；这是每次实验的一次对照，不说明稳定收益，也不能外推到图捕获成本、实际端到端场景或所有 shape。

以上数字由该目录 `summary.csv` 和 `choice-analysis-20261112-143458-3427182.csv` 统计；各 shape 的独立 CSV 才是原始数据。不能把这 128 次视为同一 shape 的 128 次重复。

## Nsight Compute：旧构建的单次重放

[`validation/final-audit-20261112-105622/nsight-full-notes.md`](../validation/final-audit-20261112-105622/nsight-full-notes.md) 保存的是更早的 `build/windows-20261112-110201-8224`、N=4096、K=4096、Batch=1、Group=64、Direct 和 Paired 分进程的 `--set full` 单次采集，Nsight Compute 2026.3.1。采集有 kernel replay，不能与本次非 profiling 计时直接比较，也不代表当前构建的结果：

| 指标 | Direct | Grouped Paired |
|---|---:|---:|
| duration（µs） | 183.360 | 104.288 |
| SM throughput（% peak sustained elapsed） | 74.563 | 73.832 |
| DRAM throughput（% peak sustained elapsed） | 19.170 | 35.770 |
| achieved occupancy（% active） | 95.783 | 94.305 |
| registers/thread | 24 | 19 |
| L1 / L2 sector hit rate（%） | 86.248 / 4.794 | 88.992 / 19.020 |
| executed instructions | 24,739,840 | 15,347,712 |
| long scoreboard stalled warps / active issue | 11.434 | 9.669 |

最后一行是每个 active issue 对应的平均 stalled warp 比值，不是百分比或直接的时间占比。一次重放和不同进程采集不足以证明性能瓶颈、因果关系或稳定加速。后续独立源码快照的前后对照见 [`OPTIMIZATION_20261112.md`](OPTIMIZATION_20261112.md)。

## 复现命令

以下命令从项目根目录前台执行，分别创建新的带时间戳的 `validation/matrix-raw-*` 目录，不覆盖上述实验。实验前请保证使用编译了当前参数的 CUDA Release 程序并检查 GPU 是否空闲；这些脚本不会锁频，也不会持续记录 GPU 状态。可使用当时二进制作为历史复跑，不能因此把后续源码与该二进制视为同一版本。

```powershell
cd 'E:\c++\项目\DecodeForge_v1.0.0-rc1'
.\scripts\benchmark_windows_matrix.ps1 -BuildDirectory '.\build\windows-20261112-143110-3019' -CudaToolkitRoot 'E:\DecodeForge-rc1-toolkit' -Rounds 20 -Repeats 100 -Warmup 20 -FocusShape
.\scripts\benchmark_windows_matrix.ps1 -BuildDirectory '.\build\windows-20261112-143110-3019' -CudaToolkitRoot 'E:\DecodeForge-rc1-toolkit' -Rounds 2 -Repeats 100 -Warmup 20
.\scripts\analyze_windows_choice.ps1 -ExperimentDirectory '.\validation\matrix-raw-20261112-143249-5106099'
.\scripts\analyze_windows_choice.ps1 -ExperimentDirectory '.\validation\matrix-raw-20261112-143315-5233629'
```

分析脚本在对应实验目录新建时间戳命名的分析 CSV；复跑请将 `-ExperimentDirectory` 改为脚本输出的新目录。完整 GPU 波动记录和调优器内部原始样本仍需后续实验；不据本报告直接修改默认选型策略。
