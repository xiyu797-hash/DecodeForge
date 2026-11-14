#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace decodeforge {

struct Shape {
    int n = 0;               // 输出通道数（权重矩阵行数）
    int k = 0;               // 输入通道数（权重矩阵列数）
    int batch = 0;           // 支持 1、2、4、8
    int group = 0;           // group-wise 对称量化，支持 16、32、64、128
    int groups_per_row() const {
        if(k<=0 || group<=0) throw std::invalid_argument("K and group must be positive");
        return k/group + (k%group != 0);
    }
    int row_bytes() const {
        if(k<=0) throw std::invalid_argument("K must be positive");
        return k/2 + (k%2 != 0);
    }
    void validate() const;
};

// W4A16 格式约定：
// - 权重 N x K，行优先，每个 int4 为有符号二补码 [-8,7]，低 nibble 为偶数列。
// - 量化 scale 为 IEEE754 FP16；每行每 group 元素一个 scale。
// - 激活 B x K，行优先，IEEE754 FP16；输出 B x N，行优先，FP32 累加与存储。
// - 非对齐的 K 由 row_bytes() 行填充；每行最后一个高 nibble 必须为 0。
// - 无 zero-point（对称量化），不等价于 GPTQ/AWQ 全部权重格式。
struct Inputs {
    Shape shape;
    std::vector<std::uint8_t> packed_weight;
    std::vector<std::uint16_t> scale_fp16;
    std::vector<std::uint16_t> activation_fp16;
    void validate() const;
};

int signed_int4(std::uint8_t nibble) noexcept;
void set_int4(std::vector<std::uint8_t>& out, const Shape& s, int row, int col, int value);
int get_int4(const Inputs& in, int row, int col);

Inputs make_random_inputs(const Shape& shape, std::uint64_t seed);
Inputs quantize_inputs(const Shape& shape, const std::vector<float>& weights,
                       const std::vector<float>& activation);
std::vector<float> cpu_reference(const Inputs& in);
std::vector<float> cpu_two_pass_reference(const Inputs& in);

struct Accuracy {
    double max_abs = 0;
    double max_relative = 0;
    std::size_t mismatched = 0;
    std::size_t compared = 0;
};
Accuracy compare_outputs(const std::vector<float>& expected, const std::vector<float>& actual,
                         double atol = 0.002, double rtol = 0.002);
std::string describe_shape(const Shape& s);

} // namespace decodeforge
