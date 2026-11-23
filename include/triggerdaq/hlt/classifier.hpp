#pragma once

#include <cstdint>

#include "triggerdaq/hlt/reconstruction.hpp"

namespace triggerdaq {

struct ClassifierOutput {
    uint16_t score{0};         // Signal confidence in [0, 1000]
    uint8_t predicted_class{0}; // 0=Background/Pileup, 1=DiMuon, 2=LocalizedJet
};

// Deterministic, zero-allocation Quantized Int8 MLP (6 -> 16 -> 3).
class Int8MlpClassifier {
public:
    [[nodiscard]] static ClassifierOutput predict(const HltFeatures& features) noexcept;
};

} // namespace triggerdaq
