#pragma once
#include <cstdint>
#include <cstring>

namespace decodeforge {

// IEEE-754 binary16 <-> binary32。独立于 CUDA、无需第三方库。
// float->half 使用 round-to-nearest, ties-to-even；涵盖次正规数、Inf 和 NaN。
inline std::uint16_t float_to_half(float f) noexcept {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    const std::uint16_t sign = static_cast<std::uint16_t>((u >> 16) & 0x8000u);
    const std::uint32_t exp = (u >> 23) & 0xffu;
    const std::uint32_t frac = u & 0x7fffffu;
    if (exp == 255) {
        if (!frac) return static_cast<std::uint16_t>(sign | 0x7c00u);
        std::uint16_t payload = static_cast<std::uint16_t>(frac >> 13);
        return static_cast<std::uint16_t>(sign | 0x7c00u | payload | 0x0200u);
    }
    const int e = static_cast<int>(exp) - 127;
    if (e > 15) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (e < -25) return sign;
    if (e < -14) {
        const std::uint32_t m = frac | 0x800000u;
        const int shift = -e - 1;
        const std::uint32_t base = m >> shift;
        const std::uint32_t rem = m & ((1u << shift) - 1u);
        const std::uint32_t tie = 1u << (shift - 1);
        return static_cast<std::uint16_t>(sign | (base + (rem > tie || (rem == tie && (base & 1u)))));
    }
    std::uint32_t he = static_cast<std::uint32_t>(e + 15);
    std::uint32_t hm = frac >> 13;
    const std::uint32_t rem = frac & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) {
        ++hm;
        if (hm == 0x400u) { hm = 0; ++he; }
    }
    if (he >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    return static_cast<std::uint16_t>(sign | (he << 10) | hm);
}

inline float half_to_float(std::uint16_t h) noexcept {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h & 0x8000u) << 16);
    const std::uint32_t e = (h >> 10) & 0x1fu;
    std::uint32_t mant = h & 0x3ffu;
    std::uint32_t out = 0;
    if (e == 0) {
        if (!mant) out = sign;
        else {
            int ex = -14;
            while ((mant & 0x400u) == 0) { mant <<= 1; --ex; }
            mant &= 0x3ffu;
            out = sign | (static_cast<std::uint32_t>(ex + 127) << 23) | (mant << 13);
        }
    } else if (e == 31) out = sign | 0x7f800000u | (mant << 13);
    else out = sign | ((e + 112u) << 23) | (mant << 13);
    float f = 0;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}
}
