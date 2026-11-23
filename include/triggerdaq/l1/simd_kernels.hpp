#pragma once

#include <cstdint>

#include "triggerdaq/core/event_frame.hpp"

namespace triggerdaq {

struct L1Features {
    uint32_t total_et{0};
    uint32_t max_cluster4_et{0};
    uint16_t max_muon_pt{0};
    uint16_t muon_hit_count{0};
    uint16_t peak_cell_idx{0};
};

// Portable C++20 scalar reference kernel
[[nodiscard]] L1Features extract_l1_features_scalar(
    const RawEventFrame& frame,
    uint16_t muon_pt_threshold) noexcept;

#if defined(__AVX2__)
// AVX2-vectorized kernel guaranteed bit-identical to extract_l1_features_scalar
[[nodiscard]] L1Features extract_l1_features_avx2(
    const RawEventFrame& frame,
    uint16_t muon_pt_threshold) noexcept;
#endif

// Compile-time dispatched L1 feature extraction kernel
[[nodiscard]] inline L1Features extract_l1_features(
    const RawEventFrame& frame,
    uint16_t muon_pt_threshold) noexcept {
#if defined(__AVX2__)
    return extract_l1_features_avx2(frame, muon_pt_threshold);
#else
    return extract_l1_features_scalar(frame, muon_pt_threshold);
#endif
}

[[nodiscard]] inline constexpr bool has_avx2_kernel() noexcept {
#if defined(__AVX2__)
    return true;
#else
    return false;
#endif
}

} // namespace triggerdaq
