# DecodeForge v1.0.0-rc1：交付与验收状态（2026-11-09）

## 2026-11-12 后续更新

当前验收状态以 [`RESUME_READINESS.md`](RESUME_READINESS.md) 为准。2026-11-13 主线 Direct-only 优化和调优 trace 合入后，Windows CUDA Release CTest 为 5/5，CPU-only CTest 为 4/4；最终构建配套的 memcheck/racecheck 两套测试共四项零错误或隐患，见 `validation/sanitizer-20261113-164747-5197-c9ab6860`。较早同版本工具的人为越界负控检出错误，见 `validation/sanitizer-final-20261112-142016`；本轮未重复负控。原始项目路径为何注入失败仍未精确定位，脚本使用临时别名。版本继续为 rc1。

后续 Release 构建 `build/windows-20261112-142802-6323` 的 CTest 为 5/5，包含零预热工作区和输出状态回归；见 `validation/windows-20261112-142802-6323`。该构建运行时显式跳过 Sanitizer，不能复用旧二进制的安全结果来声明此构建通过。项目仍是 rc1。

## 2026-11-12 本机复验

**CUDA 13.4 配套 Sanitizer 未通过验收：**官方 13.4.3 独立组件校验后，对两个 GPU 测试的 memcheck/racecheck 四项均报注入失败并退出 -1。故下方旧版工具的零错误结果不能替代此项。人为越界负控被旧工具检出，而新工具同样注入失败；根因尚未确定。原始记录见 `validation/sanitizer-134-20261112-102921/diagnostic.md`。

Windows 11、RTX 4060 Laptop、驱动 616.56、CUDA 13.4.92、MSVC 19.51 的新 Release 构建 CTest 4/4 通过，日志位于 `validation/windows-20261112-100802-6775`。实际对两套 GPU 测试分别运行 Compute Sanitizer 2025.2（CUDA 12.9 附带版本）的 memcheck 与 racecheck，四次均退出0、摘要零错误，原始日志位于 `validation/audit-20261112-100802`。该 Sanitizer 版本早于编译器工具包；这些结果不等同于 CUDA 13.4 配套工具验收，也不是全部输入的安全保证。

RTX 4060 同条件三轮、每轮 12 个 shape 的 Scalar/Paired 消融 CSV 分别位于 `validation/ablation-20261112-101108-4997`、`validation/ablation-20261112-101116-7374`、`validation/ablation-20261112-101124-8321`。测量前 GPU 利用率0%、温度44°C，CPU 总使用率约12%；未锁频，未连续监控干扰。2048×4096、B=1/2/4/8 的三轮 paired 耗时比中位数约1.58/1.55/1.51/1.24；部分较小 shape 出现收益波动甚至退化。该比值仅对照本项目 Scalar，不是 CUTLASS 等工业实现，不能外推端到端模型吞吐。父进程控制台汇总日志因重定向语法失败未保存，但各 shape CSV 与环境文件完整。

本机为 CUDA 工具链建立了 `E:\DecodeForge-rc1-toolkit` junction，指向旧 DecodeForge 目录中已有的 CUDA 13.4 文件；此次构建的 CMake 缓存引用该路径。它并非项目自包含依赖，迁移或删除前需指定有效工具链路径并重新配置新构建目录。

## 本次代码改动（实际已完成）

- 新增 `GroupedScalar` 消融参考Kernel，保存单nibble的原始Grouped路径；新 `Grouped` 保留相邻2个INT4共享压缩byte/scale的Paired方案。
- CUDA Graph和GPU数值测试覆盖 Scalar/Paired/Direct；原有30组GPU Shape保留。自动调优只在Direct/Paired之间选择，在两种代表Batch上测试Dispatcher，减少在Compute Sanitizer中重复运行的采样量。
- 7组交替顺序的Scalar/Paired计时，新增paired耗时比的中位数CSV；这只是改善试验设计，**未证实加速**。
- 自动调优依旧采用9组配对计时和中位数决策（来自上传源码，原有功能，并非本次发明）。
- CMake尊重用户设置的`CMAKE_CUDA_ARCHITECTURES`，默认SM89；修复CUDA Event部分创建失败情况下的资源泄漏；清理重复Sanitizer调用。
- Windows提供“构建/普通测试”与“4项完整Sanitizer”、“12组消融基准”独立脚本，记录环境与日志。
- CPU回归新增签名INT4低/高nibble、奇数K padding、成对Scale分组不变式的覆盖。

## 原始发行包测试快照（历史环境；新结果以顶部本机复验为准）

| 验证 | 结果 | 说明 |
|---|---|---|
| Linux GCC14 C++17 Release | PASS | CPU-only编译+CTest 2/2 |
| Linux Clang17 C++17 Release | PASS | CPU-only编译+CTest 2/2 |
| Linux Clang17 ASan/UBSan Debug | PASS | CPU-only编译+CTest 2/2 |
| CPU quantization, GEMM, RMSNorm, FP16/BF16 | PASS | CPU测试集，细节参见测试源码 |
| GPU CUDA编译/数值测试（本次源码） | NOT RUN | 本执行环境没有nvcc、NVIDIA GPU |
| Memcheck/Racecheck（本次源码） | NOT RUN | Windows需要再次运行新二进制 |
| Paired相对Scalar性能提升 | UNKNOWN | 不能根据算法想象编造倍率 |
| 4090服务器测试（本次源码） | NOT RUN | 需同步后再测 |

**此前用户报告的 Windows 4060 4/4 CTest、2项memcheck PASS，属于本次修改之前的可执行文件，不能代替最终源码复验；此前完整racecheck中断，因此不是PASS。**

## 符合正式简历数据的最低门槛

1. 新编译通过+CTEST 4/4+所有GPU数值验证，所有异常退出须定位。
2. `memcheck`和`racecheck`对两个GPU测试程序均结束、退出0、日志有无错误摘要；中断或超时记录未完成。
3. 相同GPU、功耗/频率与Shape，重复12组Paired/Scalar配对计时。提供均值与中位数或每次原始报告；若没有稳定优势，不写百分比或倍数。
4. README与简历仅写源码、测试和已证实结果。`two_pass_fp32`是未融合参考，不能当作优化过的W4A16库。

## 注意

本项目是一个学习/作品集级CUDA算子工程，并不包含实际LLM Token吞吐、工业级INT4 Tensor Core算子、模型集成、完整多GPU训练、生产部署认证。不能据此声称具备这些能力。
