> 此文件保留v0.9.1时的测试方案/旧记录。**本次v1.0.0-rc1实时状态以 `docs/RELEASE_STATUS.md` 为准；旧PASS不能证明本版CUDA通过。**

# 发行验收计划、证据和未解决项

## 当前交付环境真实结论

- CPU GCC Release编译、CTest测试：PASS。
- CPU Clang Release编译、CTest测试：PASS。
- CPU Clang ASan+UBSan：PASS。
- FP16/BF16穷举各65,536位模式、272种W4A16组合、160种GEMM组合、27种RMSNorm组合、手写金标准、错误输入：PASS。
- CUDA编译：**未执行（本环境nvcc不存在）**。
- 4090/4060 CUDA正确性：**未执行（本环境GPU不存在）**。
- GPU Benchmark、Nsight、Compute Sanitizer：**未执行**。

## 真机验收项（一次完成，不可跳过）

1. 在RTX 4090空闲卡设置 `CUDA_VISIBLE_DEVICES=1`，检查 `nvcc --version`、`g++ --version` 与 `nvidia-smi`；后续CUDA逻辑设备号为0。
2. `bash scripts/verify_cpu.sh` — CPU数学、pack、变形大小正确性。
3. `bash scripts/verify_gpu.sh` — 从零构建真实CUDA、CTest，至少30种W4A16输入Shape和3条执行路径验证、Graph重放；FP16/BF16 GEMM 3路径、RMSNorm 2路径验证；Benchmark CSV。
4. `bash scripts/verify_gpu_compute_sanitizer.sh` — 对所有算子执行memcheck、racecheck，不应有非法访存或竞争；如显卡驱动或Toolkit不支持，需记录原因，不得视为PASS。
5. 手工测试`--batch=1/2/4/8`、`--k=65/4096`、`--group=16/64/128`的正确性与benchmark，对比压力负载。
6. 在RTX 4060同架构显卡重复；在报告中注明笔记本功耗限制、GPU温度、驱动、Toolkit版本。
7. 运行Nsight Systems/Compute分析最慢Shape，保留 ncu/nsys 导出结果，不应只提交加速倍数。

## 真实通行的CUDA核测试命令

```bash
export CUDA_VISIBLE_DEVICES=1
bash scripts/verify_gpu.sh
bash scripts/verify_gpu_compute_sanitizer.sh
./build/gpu/decodeforge_cli gpu-check --n 4096 --k 4096 --batch 4 --group 128
./build/gpu/decodeforge_cli bench --n 4096 --k 4096 --batch 1 --group 64 --repeats 200 --warmup 30 --csv benchmarks/4090_batch1.csv
```

如果任何步骤失败：**保留全部日志，修复源码，重新从零运行对应验收**。切勿将CPU测试PASS描述为GPU实现PASS。

## 基准公平性

计时仅含GPU计算流上的执行及可能发射间隔，不含CPU数据构造、内存分配、H2D/D2H；如果想比较完整LLM推理速度，必须另外加入端到端的权重加载、算子组合和CPU调度测试。CUDA Events时间包含可能的Host空档，CUDA Graph可帮助区分这部分开销。CPU参考为double累积，不用于真实速度对比。

## v0.9.1新增验证覆盖

- BF16全65536位型、FP16/BF16 GEMM 160个形状、RMSNorm 27个形状，在CPU路径运行通过。
- GPU侧另有FP16/BF16 GEMM 14组矩阵和Residual+RMSNorm 11组行长度的正确性对照入口；需要真实4090或4060执行。
- `scripts/verify_all_gpu.sh` 将环境元数据写入 `benchmarks/device_environment.txt`，顺序检查CPU、GPU完整正确性、W4A16基准、GEMM和RMSNorm基准、Compute Sanitizer。脚本失败即非零退出。
- 跨CPU/GPU公平性：所有CPU参考以double累加再输出FP32；GPU以FP32累加，使用绝对与相对容差。GEMM cuBLAS可能采用不同数学路径，需记录真实误差。
