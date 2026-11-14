# W4A16 Scale 索引实验（2026-11-12）

## 问题与假设

Direct 与 Grouped Paired 在设备代码内根据 `col / group` 计算 Scale 索引。公开 Shape 契约只允许 group 16/32/64/128，均为 2 的幂。实验候选在主机端计算对应位移，将这两条路径的设备端除法改为右移；Grouped Scalar 保持历史实现。实验不改变数值格式或浮点累加顺序，也不将特定测试 shape 写入 autotuner。

## 已执行的实验

- 基线和候选是 `validation/scale-shift-ab-20261112-143454-d806c2ee` 下独立源码快照和 Release 构建，CTest 各 5/5；原始正式源码暂未改动。
- `validation/build-ab-20261112-143700-6114359`：65 个 shape（含目标 shape）× 2 轮，每轮基线和候选各在独立进程测量；AB/BA 首后顺序按运行轮换。每进程保存 9 对调优后 Direct/Paired 原始 Event 样本以及汇总、日志和环境信息。基线/候选分别取各进程 9 次中位数之比；数据并非同一事件区间的交叉配对。
- 130 个 shape-轮次中，Direct 基线/候选比值的中位数约 1.212；Grouped Paired 约 1.009。以每次比值 <1 计，候选 Direct 有 32/130 次较慢、Paired 有 48/130 次较慢。按 K=4096 分组时两条路径中位比值分别约 1.41、1.24（32 次观察）；K=129 时两者都约 1.00。两个总体数值不是各 workload 等权的生产吞吐。
- `validation/build-ab-20261112-144013-3500304`：针对两轮均出现超过 5% 退化的 3 个 shape 各重复 10 轮。N65/K129/B4/G32 的 Direct 与 Paired 比值中位数约 0.92、0.89，但逐轮波动宽、两者分别 7/10 和 6/10 次候选较慢。该观察应公开，不能声称所有 shape 均有收益；短核波动也不能证明退化完全由此代码引起。
- GPU 时钟未锁定，测量前记录设备状态，但没有连续监视 CPU/GPU 干扰；两次大矩阵轮次不足以给出长期置信区间或行业性能排名。

## Nsight Compute 机制观察

`validation/scale-shift-ab-20261112-143454-d806c2ee/profile-ab-20261112-144056-ceffb0e5` 包含 4 个完整原始报告及对应 CSV。同一 N4096/K4096/B1/G64、单次 full replay，同一 kernel 的基线→候选：

| 指标 | Direct 基线 → 候选 | Paired 基线 → 候选 |
|---|---:|---:|
| Kernel duration | 183.296 → 158.656 µs | 103.936 → 87.328 µs |
| Executed instructions | 24,739,840 → 18,268,160 | 15,347,712 → 11,268,096 |
| Registers/thread | 24 → 20 | 19 → 17 |
| Long scoreboard ratio | 11.44 → 16.47 | 9.66 → 13.89 |

Long scoreboard 是每个活跃 issue 的平均停滞 warp 比值，不是时间百分比；它上升不意味着绝对执行时间上升。Nsight 多重重放的单次指标用于分析，不替代稳定状态计时或证明因果。

## 当前决定与复现

全局位移候选没有直接合入：长 K 有潜在收益，小 K 的退化和波动尚未解释。正在探索按一般循环工作量选择旧核/位移核的方案；只有经过阈值边界、未参与选择的 shape、数值和完整 Sanitizer 复验后才能决定是否保留。

### 探索性混合候选与持出性能

独立 `hybrid` 快照保留短 K 旧核，Direct 的 K≥1024 和 Paired 的 K≥2048 使用位移核。这是基于前述实验设定的工作量阈值，不是对特定完整 shape 的自动调优；不同 GPU 的适用性没有验证。增加了阈值两侧 144 组 Batch/Group/奇偶 K GPU 数值检查，实验快照 CTest 5/5。测试实现的 Graph/自动调优覆盖和内核重复代码仍需完善，因此不直接合入。

`validation/build-ab-20261112-144502-4361837` 为此前没有用于选择阈值的 56 组参数各两轮；N=513/1537、K=1023/1025/1536/2047/2048/2049/3073、B=1/4、Group=32/128。旧脚本按全局序号安排先后，因每轮 shape 数为偶数，同一 shape 的先后顺序两轮相同，存在顺序混杂；不能将其视为平衡的 AB/BA。112 次基线/候选比值总体中位 Direct 为 1.368、Paired 为 1.038；分别有 7/112 和 37/112 次候选较慢。每个 K 仅 16 次观察，分别按 K 汇总的 Direct/Paired 中位比值约为：1023：1.00/1.01；1025：1.32/1.00；1536：1.36/1.01；2047：1.37/1.00；2048：1.37/1.06；2049：1.37/1.12；3073：1.40/1.12。小核仍有单次退化和噪声，未锁频，不能将上述数字当成性能保证；完整逐次记录与每次的 9 对原始数据保存在该目录。

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\benchmark_windows_ab_builds.ps1 `
  -BaselineBuild '.\validation\scale-shift-ab-20261112-143454-d806c2ee\build-baseline' `
  -CandidateBuild '.\validation\scale-shift-ab-20261112-143454-d806c2ee\build-candidate' `
  -CudaToolkitRoot 'E:\DecodeForge-rc1-toolkit' -Rounds 2 -Repeats 100 -Warmup 20
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\analyze_windows_ab_builds.ps1 -ExperimentDirectory '<新生成的 validation\build-ab-*>'
```

Two-pass FP32 是本项目内部未融合参考，以上比较并非对 CUTLASS、TensorRT-LLM 等成熟量化核的对比。

## 2026-11-13：重构候选复验

`validation/scale-shift-ab-20261112-143454-d806c2ee/hybrid-refactor` 将原/位移核改为共享的编译期模板，并补充 144 组阈值边界、阈值两侧 Graph、自动调优和 N257 的组边界/奇数尾手工金标准；独立 Release 构建的 CTest 5/5。实测执行的模板分支见同目录 `nsight-dispatch-20261113-090237-9990f312a78a41d1a22d51b4c36de25c/dispatch-summary.csv`；数值正确本身不证明性能稳定。

- `validation/build-ab-20261113-085628-2895448`：与基线对比 65 shape × 2 轮，130 次基线/候选比值 Direct 中位数 1.245、Paired 1.019；分别有 34 和 46 次候选较慢。K4096 按 B=1/2/4/8 的 Paired 中位比值约 1.32/1.28/1.16/1.00，Direct 约 1.40–1.42。不同 shape 不能合并描述成端到端吞吐。
- `validation/build-ab-20261113-085917-2330700`：56 组持出 shape × 2 轮，Direct/Paired 比值中位数约 1.366/1.024；分别有 12/112 和 45/112 次候选较慢。旧脚本的偶数 shape 轮次未对同一 shape 轮换先后，结果有顺序混杂。
- 对第一组实验中重复出现退化的 6 个 K256 shape 分别做 10 轮，原始数据在 `validation/build-ab-20261113-090121-7462812`。目标 N1024/K256/B1/G64 中 Direct/Paired 的基线/重构候选比值中位数约 0.82/0.81，候选较慢分别 6/10 和 7/10；样本明显波动。旧脚本在此组中每轮目标 shape 都先运行基线，再运行候选，存在严重的顺序混杂，不足以判断是否由代码导致。对保留原短核的独立 hybrid 再做相同 10 轮（`validation/build-ab-20261113-090301-5328239`），目标 shape 两条路径的中位比值均约 0.99，同样存在顺序混杂。两组测试在不同时间执行，不可将差异直接归因于源码。脚本随后已修改为按每个 shape 跨轮切换顺序，并增加 schedule.csv；后续需要同一二进制 A/A 控制及重新测量。
- 新一轮同 shape N4096/K4096/B1/G64 的 full Nsight 报告在 `validation/scale-shift-ab-20261112-143454-d806c2ee/nsight-full-hybrid-ab-20261113-090406-db167213715a43eaa1c60a929dc408b2`。Direct 基线→候选 183.104→158.464 µs、指令 24,739,840→18,268,160、寄存器 24→20；Paired 为 103.616→87.264 µs、15,347,712→11,268,096、19→17。单次重放只能提供机制线索。

以上未锁频，未持续监控 GPU/CPU 状态；部分测量期间本机还执行了轻量的 CPU 日志分析或反汇编操作，影响未量化。原始采样及环境文件均保留。经过后续 A/A 与平衡顺序 A/B 复测，2026-11-13 将 Direct-only 候选与可选调优内部采样合入主线；未采用 Grouped 位移候选。短 K 仍存在未能量化的波动和潜在退化，当前也未证明调优策略的选择稳定性得到改善。

重构候选的完整 Compute Sanitizer 日志在 `validation/sanitizer-20261113-090455-3243-33bae9d4`：两个 GPU 测试的 memcheck 均零错误、racecheck 均零隐患。该次受检二进制来自实验快照 `build-hybrid-refactor`，主 GPU 测试执行 1,418 次输出检查；它不等于正式源码已经采用候选，也不证明覆盖未执行的输入。该次没有重复运行故意越界负控；相同版本工具的较早负控见 `validation/sanitizer-final-20261112-142016`。

修正顺序后的同二进制 A/A 控制在 `validation/build-ab-20261113-091234-9014999`：6 个短 K shape 各 20 轮，记录的 `schedule.csv` 已核对每个 shape 均 10 次 baseline 标记先、10 次 candidate 标记先。两个标记实际运行的是同一个基线可执行文件。目标 N1024/K256/B1/G64 的 Direct 中位比 1.07、范围 0.79–1.34，Paired 中位比 1.04、范围 0.72–1.79；这反映了本测量流程中的较大进程间波动，不是两个不同实现的性能差异。设备时钟未锁定且未连续记录，不能只据此量化置信区间。

相同平衡顺序下的 20 轮候选 A/B 位于 `validation/build-ab-20261113-091416-0551241`。目标 shape 的 Direct/Paired 中位比约 1.01/1.00；6 shape 合计 120 对的中位比约 0.991/0.992。该结果没有复现先前未平衡顺序的 0.82/0.81，但样本波动大，不证明候选对所有短 K 都没有退化。先前两组顺序混杂的观察保留以便审计，不能作为代码回退的定论。

平衡顺序的 56 shape 持出 A/B 保存于 `validation/build-ab-20261113-091559-9345249`，每个 shape 两轮均记录并验证一轮 AB、一轮 BA。112 对的 Direct/Paired 中位比约 1.366/1.013，候选分别有 12/112、46/112 次较慢；K2047 的 Paired 中位比约 0.95。7 个 shape 的 Paired 在两轮都超过 5% 观察退化，仍需进一步复测和同文件 A/A，不能只报告整体中位数。原始汇总出现例如 N513/K1536/B4/G128 的 Direct 单次异常值约 0.764 ms，保留未删；分析文件使用每进程独立九次采样的中位数而非这个单次汇总。

对上述 7 个边界 shape 的平衡 10 轮同文件 A/A 与候选 A/B，分别见 `validation/build-ab-20261113-091834-0389387` 和 `validation/build-ab-20261113-091939-3517115`。每 shape 均核验两种顺序各 5 次。候选 A/B 中，N1537/K2047/B1/G32、N513/K1536/B1/G128、N513/K1536/B4/G128 的 Paired 中位比约 0.95、0.95、0.93，分别 8/10、7/10、7/10 次候选较慢；对应 A/A 中位比约 1.01、0.96、0.99，短核数据仍有明显噪声。由于这三组的回退信号尚未排除，混合候选不直接合入，改为进一步研究只优化 Direct 并保留原 Grouped 的独立候选。

Direct-only 候选源码和构建在 `validation/scale-shift-ab-20261112-143454-d806c2ee/direct-only-20261113-092125-4271468`，CTest 5/5。其 `sass-grouped-20261113-092501-2218447` 对两个 EXE 内的四个 SM89 Grouped Paired 特化逐条比较编码指令，B=1/2/4/8 分别为 120/168/232/392 条且全部一致；匿名命名空间符号前缀已排除。此比较不覆盖整个 cubin、常量区、寄存器或 launch 元数据、host 调度，也不能证明性能相同。CUDA 13.4 portable 未找到 cuobjdump，因此使用本机 CUDA 12.9.26 版本，两次退出码均为 0，均产生切换格式的 nvdisasm warning，原始 stderr 已保留。

Direct-only 目标及短 K 的 6 个 shape、各 20 轮平衡 A/B 在 `validation/build-ab-20261113-092425-6711970`，每个 shape 的实际先后均为 10/10；目标 N1024/K256/B1/G64 的基线/候选中位比 Direct 约 0.98、Paired 约 1.00。7 个边界 shape 各 10 轮在 `validation/build-ab-20261113-092302-8302548`，多个 Grouped 的单轮结果仍波动，不能仅因机器码相同就证明端到端时间相同。其余持出集和候选最终安全验证继续单独记录。

Direct-only 的平衡顺序持出集 `validation/build-ab-20261113-092611-0798969`：56 shape 各两轮，112 对 Direct/Paired 中位比约 1.367/1.000；两个路径分别有 15/112、58/112 次候选较慢。N513/K1025/B1/G128 两轮 Direct 均观察变慢；随后在每次 Event 计时增加至 500 次重复、50 次预热的平衡顺序 A/A 与 A/B 中各重复 10 轮（`validation/build-ab-20261113-092915-7498029` 与 `validation/build-ab-20261113-093003-5538549`），该 shape Direct A/B 比值中位数约 1.18，但单轮范围仍为 0.81–1.82。此数据不证明该 shape 不会退化；先前两次变慢也不应简单归因于代码。

完整 65 shape × 2 轮的平衡 Direct-only A/B 见 `validation/build-ab-20261113-093156-5863681`；130 对 Direct/Paired 总中位比约 1.137/1.001，Direct 有 32/130 次候选较慢。按 K 分类，Direct 中位比 K129≈0.98、K256≈1.01、K1024≈1.29、K4096≈1.41；两个短 K shape 的 Direct 在两轮均有超过 5% 观察退化，相关数据保留并继续复验。

同一 N4096/K4096/B1/G64 的 Direct-only 版 Nsight full 原始数据在 `validation/scale-shift-ab-20261112-143454-d806c2ee/direct-only-20261113-092125-4271468/nsight-direct-full-ab-20261113-093507-ac78beff572e42b3b60016ab3171a9b3`。基线→候选：duration 183.552→158.368 µs、SM throughput 74.48→62.68%、DRAM throughput 19.01→22.04%、achieved occupancy 95.80→95.54%、寄存器 24→20、L1 hit 86.25→86.26%、L2 hit 4.71→4.72%、执行指令 24,739,840→18,268,160、long scoreboard ratio 11.41→16.46。单次多轮 replay 的 ratio 不是百分比，不能将其直接解释为时间损失或普遍加速。
