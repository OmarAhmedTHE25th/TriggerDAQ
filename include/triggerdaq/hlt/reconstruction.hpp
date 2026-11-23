#pragma once

#include "triggerdaq/core/event_frame.hpp"

namespace triggerdaq {

struct HltFeatures {
    float cluster_centroid{0.0f};
    float cluster_width{0.0f};
    float isolation_ratio{0.0f};
    float dimuon_inv_mass{0.0f};
    float total_et_norm{0.0f};
    float peak_muon_norm{0.0f};
};

// Pure, zero-allocation calorimeter cluster + di-muon invariant mass reconstruction.
[[nodiscard]] HltFeatures reconstruct_event(const RawEventFrame& frame) noexcept;

} // namespace triggerdaq
