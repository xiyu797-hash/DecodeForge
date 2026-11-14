#include "decodeforge/core.hpp"
#include "decodeforge/half.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

namespace decodeforge {
namespace {
// 仅用于已经验证非负且范围安全的形状和索引。
std::size_t u(int v) noexcept { return static_cast<std::size_t>(v); }
std::size_t offset(int row, int width, int col) noexcept {
    return u(row) * u(width) + u(col);
}
std::size_t checked_mul(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::invalid_argument("size overflow");
    return a * b;
}
// 内部快速路径：在 shape、vector 容量与位置已经核验后使用。
void write_nibble_unchecked(std::vector<std::uint8_t>& dest, const Shape& s,
                            int row, int col, int value) noexcept {
    const auto i = offset(row, s.row_bytes(), col / 2);
    const auto nib = static_cast<std::uint8_t>(value & 15);
    dest[i] = ((col & 1) == 0)
        ? static_cast<std::uint8_t>((dest[i] & 0xf0u) | nib)
        : static_cast<std::uint8_t>((dest[i] & 0x0fu) | (static_cast<unsigned>(nib) << 4));
}
int read_nibble_unchecked(const Inputs& in, int row, int col) noexcept {
    const auto byte = in.packed_weight[offset(row, in.shape.row_bytes(), col / 2)];
    return signed_int4(static_cast<std::uint8_t>((col & 1) ? (byte >> 4) : (byte & 15u)));
}
}

void Shape::validate() const {
    if (n <= 0 || k <= 0) throw std::invalid_argument("N and K must be positive");
    if (batch != 1 && batch != 2 && batch != 4 && batch != 8)
        throw std::invalid_argument("Batch must be 1, 2, 4 or 8");
    if (group != 16 && group != 32 && group != 64 && group != 128)
        throw std::invalid_argument("Group must be 16, 32, 64 or 128");
    if (n > 65536 || k > 65536) throw std::invalid_argument("N/K too large (limit 65536)");
    if (checked_mul(u(n), u((k + 1) / 2)) >= (1ull << 31))
        throw std::invalid_argument("packed weights would exceed 2 GiB");
}

void Inputs::validate() const {
    shape.validate();
    if (packed_weight.size() != checked_mul(u(shape.n), u(shape.row_bytes())) ||
        scale_fp16.size() != checked_mul(u(shape.n), u(shape.groups_per_row())) ||
        activation_fp16.size() != checked_mul(u(shape.batch), u(shape.k)))
        throw std::invalid_argument("invalid packed/scales/activations buffer length");
    if (shape.k % 2) {
        for (int row = 0; row < shape.n; ++row) {
            const auto i = offset(row, shape.row_bytes(), shape.row_bytes() - 1);
            if ((packed_weight[i] & 0xf0u) != 0)
                throw std::invalid_argument("odd-K padding nibble must be zero");
        }
    }
    for (auto scale : scale_fp16) {
        const float sf = half_to_float(scale);
        if (!std::isfinite(sf) || sf <= 0.f)
            throw std::invalid_argument("scales must be positive finite FP16 values");
    }
    for (auto act : activation_fp16)
        if (!std::isfinite(half_to_float(act)))
            throw std::invalid_argument("activations must be finite");
}

int signed_int4(std::uint8_t nibble) noexcept {
    const int v = static_cast<int>(nibble & 15u);
    return v >= 8 ? v - 16 : v;
}

void set_int4(std::vector<std::uint8_t>& out, const Shape& s, int row, int col, int value) {
    s.validate();
    if (row < 0 || row >= s.n || col < 0 || col >= s.k || value < -8 || value > 7)
        throw std::out_of_range("invalid int4 position/value");
    if (offset(row, s.row_bytes(), col / 2) >= out.size())
        throw std::out_of_range("packed buffer too short");
    write_nibble_unchecked(out, s, row, col, value);
}

int get_int4(const Inputs& in, int row, int col) {
    in.shape.validate();
    if (row < 0 || row >= in.shape.n || col < 0 || col >= in.shape.k)
        throw std::out_of_range("invalid int4 index");
    const auto i = offset(row, in.shape.row_bytes(), col / 2);
    if (i >= in.packed_weight.size()) throw std::out_of_range("packed buffer too short");
    return read_nibble_unchecked(in,row,col);
}

Inputs quantize_inputs(const Shape& shape, const std::vector<float>& weights,
                       const std::vector<float>& activation) {
    shape.validate();
    if (weights.size() != checked_mul(u(shape.n), u(shape.k)) ||
        activation.size() != checked_mul(u(shape.batch), u(shape.k)))
        throw std::invalid_argument("wrong FP32 input lengths");
    Inputs out;
    out.shape = shape;
    out.packed_weight.resize(checked_mul(u(shape.n), u(shape.row_bytes())), 0);
    out.scale_fp16.resize(checked_mul(u(shape.n), u(shape.groups_per_row())));
    out.activation_fp16.resize(activation.size());
    for (std::size_t i = 0; i < activation.size(); ++i) {
        if (!std::isfinite(activation[i])) throw std::invalid_argument("activation must be finite");
        out.activation_fp16[i] = float_to_half(activation[i]);
    }
    for (int row = 0; row < shape.n; ++row) {
        for (int grp = 0; grp < shape.groups_per_row(); ++grp) {
            const int start = grp * shape.group;
            const int stop = std::min(shape.k, start + shape.group);
            float max_abs = 0.f;
            for (int col = start; col < stop; ++col) {
                const float w = weights[offset(row, shape.k, col)];
                if (!std::isfinite(w)) throw std::invalid_argument("weight must be finite");
                max_abs = std::max(max_abs, std::abs(w));
            }
            // 零权重组仍必须有严格正的FP16 scale；防止后续除以0。
            const float raw_scale = std::max(max_abs / 7.f, 0.00006103515625f);
            const auto scale_bits = float_to_half(raw_scale);
            out.scale_fp16[offset(row, shape.groups_per_row(), grp)] = scale_bits;
            const float scale = half_to_float(scale_bits);
            if (!std::isfinite(scale) || scale <= 0.f)
                throw std::invalid_argument("weight scale exceeds FP16 range");
            for (int col = start; col < stop; ++col) {
                const float w = weights[offset(row, shape.k, col)];
                const float ratio = w / scale;
                const int q = static_cast<int>(std::nearbyint(std::max(-8.f, std::min(7.f, ratio))));
                write_nibble_unchecked(out.packed_weight, shape, row, col, q);
            }
        }
    }
    out.validate();
    return out;
}

Inputs make_random_inputs(const Shape& shape, std::uint64_t seed) {
    shape.validate();
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> w(checked_mul(u(shape.n), u(shape.k)));
    std::vector<float> a(checked_mul(u(shape.batch), u(shape.k)));
    for (auto& v : w) v = dist(rng);
    for (auto& v : a) v = dist(rng);
    return quantize_inputs(shape, w, a);
}

std::vector<float> cpu_reference(const Inputs& in) {
    in.validate();
    const auto& s = in.shape;
    std::vector<float> result(checked_mul(u(s.batch), u(s.n)), 0.f);
    for (int b = 0; b < s.batch; ++b) for (int row = 0; row < s.n; ++row) {
        double acc = 0;
        for (int k = 0; k < s.k; ++k) {
            const float scale = half_to_float(in.scale_fp16[offset(row, s.groups_per_row(), k / s.group)]);
            const float act = half_to_float(in.activation_fp16[offset(b, s.k, k)]);
            acc += static_cast<double>(read_nibble_unchecked(in, row, k)) * scale * act;
        }
        result[offset(b, s.n, row)] = static_cast<float>(acc);
    }
    return result;
}

std::vector<float> cpu_two_pass_reference(const Inputs& in) {
    in.validate();
    const auto& s = in.shape;
    std::vector<float> weight(checked_mul(u(s.n), u(s.k)));
    for (int row = 0; row < s.n; ++row) for (int k = 0; k < s.k; ++k)
        weight[offset(row, s.k, k)] = static_cast<float>(read_nibble_unchecked(in,row,k)) *
            half_to_float(in.scale_fp16[offset(row, s.groups_per_row(), k / s.group)]);
    std::vector<float> result(checked_mul(u(s.batch), u(s.n)), 0.f);
    for (int b = 0; b < s.batch; ++b) for (int row = 0; row < s.n; ++row) {
        double acc = 0;
        for (int k = 0; k < s.k; ++k)
            acc += static_cast<double>(weight[offset(row,s.k,k)]) *
                half_to_float(in.activation_fp16[offset(b,s.k,k)]);
        result[offset(b,s.n,row)] = static_cast<float>(acc);
    }
    return result;
}

Accuracy compare_outputs(const std::vector<float>& expected, const std::vector<float>& actual,
                         double atol, double rtol) {
    if (expected.size() != actual.size() || !std::isfinite(atol) || !std::isfinite(rtol) ||
        atol < 0 || rtol < 0)
        throw std::invalid_argument("invalid compare parameters");
    Accuracy stats;
    stats.compared = expected.size();
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double a = expected[i], b = actual[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            ++stats.mismatched;
            stats.max_abs = std::numeric_limits<double>::infinity();
            continue;
        }
        const double diff = std::abs(a - b);
        stats.max_abs = std::max(stats.max_abs, diff);
        stats.max_relative = std::max(stats.max_relative, diff / std::max(1e-6, std::abs(a)));
        if (diff > atol + rtol * std::abs(a)) ++stats.mismatched;
    }
    return stats;
}

std::string describe_shape(const Shape& s) {
    std::ostringstream ss;
    ss << "N=" << s.n << ",K=" << s.k << ",B=" << s.batch << ",G=" << s.group;
    return ss.str();
}
}
