# W4A16 格式契约与准确性

本项目实现的是一类明确的数据格式：**group-wise symmetric signed INT4 weights，FP16 activations，FP32 accumulation/output**；称为 W4A16→FP32。不直接兼容GPTQ、AWQ等不同pack/zero-point/重排格式。

1. `W[n,k]` 为 N×K 矩阵，行优先；每个权重 [-8,7]，在一字节中以二补码nibble存储。**偶数K坐标在低四位，奇数K坐标在高四位**。
2. 奇数 K 时，每行最后一个字节高四位强制补0，避免把下一行的首字节错误拼入本行。
3. `scale[n, floor(k/G)]` 为正且有限的FP16，G∈{16,32,64,128}，按行分组，非整数倍K时最后一组允许不满。
4. `A[b,k]` 为 B×K 行优先 FP16，B∈{1,2,4,8}。
5. `Y[b,n] = sum_k float(signed4(W[n,k])) × float(scale[n,k/G]) × float(A[b,k])`，存储为 FP32 B×N 行优先。
6. CPU golden 先使用double进行求和，再转换FP32。CUDA fused核在FP32 FMA逐级归约，数值和参考允许少量舍入偏差；GPU验证阈值为 abs≤0.006+0.003×|ref|。
7. quantize_inputs 是为了生成可复现的基准数据：每组scale约为max(abs(weights))/7的可表示FP16值，零组保留正scale。其默认量化算法**不保证任何模型精度**，也不声称可直接加载具体LLM权重。

CPU半精度转化遵守round-to-nearest, ties-to-even。GPU端激活和scale按半精度直接读取，反量化在Float32完成。

## 参考手算案例

W行0：`[-8, -1, 0, 7]`，scale=1；W行1：`[7, -8, 7, -8]`，scale=0.5；激活行0：`[2,-3,4,-1]`；激活行1：`[-1,2,0,3]`。

预期FP32输出两行：`[-20,37]` 和 `[27,-23.5]`。在 `cpu_test.cpp` 中作为独立金标准。
