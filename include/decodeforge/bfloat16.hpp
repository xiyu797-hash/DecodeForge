#pragma once
#include <cstdint>
#include <cstring>

namespace decodeforge {
// BF16保留float32指数与高7位尾数，round-to-nearest/ties-to-even。
inline std::uint16_t float_to_bfloat16(float v) noexcept {
    std::uint32_t bits=0;
    std::memcpy(&bits,&v,sizeof(v));
    // NaN必须仍是NaN，避免舍入时丢掉NaN负载。
    if((bits&0x7f800000u)==0x7f800000u && (bits&0x007fffffu)!=0)
        return static_cast<std::uint16_t>((bits>>16)|0x0040u);
    const std::uint32_t bias=0x7fffu + ((bits>>16)&1u);
    return static_cast<std::uint16_t>((bits+bias)>>16);
}
inline float bfloat16_to_float(std::uint16_t bits) noexcept {
    std::uint32_t out=static_cast<std::uint32_t>(bits)<<16;
    float v=0;
    std::memcpy(&v,&out,sizeof(v));
    return v;
}
}
