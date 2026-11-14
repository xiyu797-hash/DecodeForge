# DecodeForge 简历就绪与验收状态

状态含义：PASS 为所列范围内有日志；FAIL 为实测失败；INCOMPLETE 为已开始但缺覆盖或存在已知限制；NOT RUN 为未执行。版本继续保持 v1.0.0-rc1，不建议升级正式版。

| 检查项 | 状态 | 证据与边界 |
|---|---|---|
| Windows VS2026 / CUDA 13.4 Release | PASS | 新增误差报告后的 `build/windows-20261113-231410-7158-95b45a28`，`validation/windows-20261113-231410-7158-95b45a28/tests.log`：CTest 5/5；CPU-only `validation/windows-20261113-231332-4991/tests.log` 为 4/4。 |
| 公共 Shape、INT4、CLI 和未初始化输出检查 | PASS | `validation/final-audit-20261112-105622` 中 red/green 日志；输出守卫针对 W4A16/GEMM/RMSNorm，Graph capture 本身不产生可下载输出。 |
| Compute Sanitizer 13.4 memcheck/racecheck | INCOMPLETE | `validation/sanitizer-20261113-164747-5197-c9ab6860`：较早的 `windows-20261113-164723-9598` 构建两项 memcheck 零错误、两项 racecheck 零隐患；其后新增误差报告程序与 CMake 目标，GPU 算子源码未改，但新构建尚未重新执行 Sanitizer。较早轮次同版本工具的故意越界负控检出错误。 |
| 4060 GPU 数值误差记录 | PASS | `validation/accuracy-20261113-2318/accuracy.csv`：64 shape ×4 路径，每路径 203,580 个输出，均满足现有容差。相对误差近零时会放大，定义和逐 shape 数据见 `docs/evidence/rtx4060-windows-20261113`。 |
| Group/Batch/奇数 K 性能矩阵 | INCOMPLETE | 最终构建 `validation/matrix-raw-20261113-165142-6825242`：64 shape ×2 轮，128 次均完成数值检查及原始时序记录；固定 shape 顺序、未锁频，只有两轮不能证明长期稳定性。 |
| Nsight Compute 指标 | PASS | 限较早构建的单次 N4096 K4096 B1、两种 kernel full 重放，见 `validation/final-audit-20261112-105622/nsight-full-notes.md`。非稳态计时和因果证明。 |
| 算子性能优化 A/B | INCOMPLETE | 新增 Direct 按 K 使用 Scale 位移索引，实验构建对长 K 观察到收益；65 shape ×2 轮及短 K 复测、同文件 A/A、退化记录见 `docs/OPTIMIZATION_20261112.md`。未锁频，部分 shape 有波动或退化；跨硬件尚未验证。 |
| 自动调优稳定性、缓存开销 | INCOMPLETE | 默认九对策略未改变，新增可选内部原始采样；最终构建目标 shape 的 20 次运行与事后独立参考相符 18/20，最大观察损失约 19.33%，调优墙钟中位数约 9.63 ms。参考本身波动，非真实准确率或最大损失上限。20/100 候选在另一次 20 轮 A/B 中同为 18/20，耗时中位约翻倍，未合入；见 `FINAL_EXPERIMENTS_20261113.md`。 |
| 等价工业 W4A16 基线 | NOT RUN | 两阶段 FP32 cuBLAS 仅为未融合算法参考，不是 CUTLASS/TensorRT 的量化 kernel。 |
| Linux 当前源码验收 | NOT RUN | 当前机器没有已安装的 WSL Linux 发行版；历史不同版本结果不能替代。 |
| 端到端模型/真实生产负载 | NOT RUN | 单算子与会话级计时不等于 LLM 推理、模型精度或生产服务。 |

## 接口与使用边界

单个 GPU session 的 stream、缓冲区和缓存均可变，不提供并发线程安全保证；调用者必须对同一实例的访问串行化。`download()` 仅在执行、计时或 graph replay 成功后可用；仅 capture graph 不会产生输出。

README/DESIGN 中如果有历史状态，以本文件对应构建的证据为准。旧版与当前源码的性能结果不要混写。完成最终复验、长期可靠性和严谨对照前，不应宣称工业级或全面领先成熟实现。
