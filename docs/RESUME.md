# DecodeForge：用于简历的准确表述（v1.0.0-rc1）

**项目名称：** DecodeForge — C++17/CUDA 低比特量化与融合算子性能实验平台

## 当前可以描述的工作（基于源码实现）

- 实现 signed INT4权重+FP16激活的group-wise对称反量化融合计算，提供Direct、Grouped Paired、Grouped Scalar三条CUDA Kernel以及FP32非融合cuBLAS参考；覆盖Batch=1/2/4/8、Group=16/32/64/128及奇数K。
- 设计跨Batch权重复用、相邻双INT4字节复用与Warp归约；保留Scalar版消融Kernel，使用交替顺序的配对CUDA Event计时比较优化效果。
- 将 Direct 路径的长 K Scale 索引改为按合法 Group 的位移计算，保留短 K 原路径；通过 A/A 噪声对照、顺序平衡 A/B 与 Nsight 记录性能及退化范围，结果限已测 RTX 4060 环境。
- 实现Shape内Direct/Paired自动选择（9组配对样本的中位数决策）及固定Shape CUDA Graph捕获；实现FP16/BF16 GEMM与融合Residual+RMSNorm并提供CPU数值参考和GPU测试。
- 建立CMake跨平台编译、GCC/Clang CPU CI、边界输入回归、CUDA Compute Sanitizer脚本和CSV/环境信息记录流程。

## 已有本机证据（限所列源码与测试范围）

- Windows RTX 4060、CUDA 13.4：最终 GPU 构建 CTest 5/5；安全检查以 `RESUME_READINESS.md` 中针对该构建的最新状态为准。过往相同版本工具的故意越界负控曾确认插桩生效，只能说明相应用例的结果。
- 同GPU的Scalar/Paired重复消融与Nsight指标可作为项目内部实验；部分Shape退化，需同时说明条件和结果，不能代表行业实现对照。
- 4090服务器的当前源码复验、端到端模型性能尚未完成，不能写成实测成果。

## 面试可解释的问题

1. INT4 两个有符号nibble如何解码，奇数K尾部如何避免越界，Scale为何跨相邻两列共享？
2. Direct按 `(batch,row)` 分配CTA，Grouped按行复用权重；比较访存、CTA数量和寄存器代价。
3. 为什么CUDA Event只测流上阶段而会受到主机发射空档影响？端到端为什么包含传输、分配、销毁？
4. 自动调优只缓存到Session，九组交错测量至少发射多少kernel？为何一次CSV赢家不是选择准确率真值？
5. 配套Sanitizer在原路径失效时，为什么越界负控必不可少？临时ASCII路径的成功又不能证明哪个具体字符是根因？
6. 哪些输入/输出契约不兼容GPTQ/AWQ，为什么不能直接拿FP32两阶段基线宣传领先工业量化库？

## 禁止夸大

本项目没有实现完整LLM端到端推理、Tensor Core专用INT4 GEMM、线上工业吞吐或优于FlashInfer/CUTLASS的行业结论。非融合FP32两阶段参考只可作为本项目内部算法基线。
