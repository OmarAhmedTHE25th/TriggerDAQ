#include "triggerdaq/hlt/classifier.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace triggerdaq {

namespace {

// Layer 1: 6 -> 16 (int8 weights, int32 biases)
// Inputs q[0..5] in [0, 127]:
//   q0 = cluster_centroid / 64
//   q1 = cluster_width / 25
//   q2 = isolation_ratio
//   q3 = dimuon_inv_mass / 150
//   q4 = total_et_norm (total_et / 4000)
//   q5 = peak_muon_norm (peak_muon / 80)
constexpr int8_t W1[16][6] = {
    // Neurons 0..4: DiMuon signal detectors
    {  0,  -4,   0,  40, -10,  32},
    {  0,   0,  -4,  48, -12,  24},
    {  0,  -2,   0,  36,  -8,  36},
    {  0,   0,   0,  44,  -6,  20},
    {  0,  -4,  -2,  32, -10,  40},
    // Neurons 5..9: LocalizedJet signal detectors
    {  0, -24,  44,  -8,  16,  -4},
    {  0, -28,  48, -10,  12,  -4},
    {  0, -20,  40,  -6,  20,  -6},
    {  0, -32,  50, -12,  10,   0},
    {  0, -18,  38,  -8,  18,  -4},
    // Neurons 10..15: Background & PileupNoise detectors
    {  0,  22, -36, -32,  -8, -20},
    {  0,  28, -42, -36,  24, -16}, // PileupNoise specialist (wide + high ET + low iso)
    {  0,  24, -40, -30,  28, -18}, // PileupNoise specialist
    {  0,  18, -30, -36, -12, -24}, // Low-energy Background specialist
    {  0,  20, -34, -28,  18, -14},
    {  0,  26, -38, -34,  10, -20},
};

constexpr int32_t B1[16] = {
    // DiMuon biases (require genuine di-muon mass + pT)
    -1400, -1600, -1500, -1300, -1550,
    // LocalizedJet biases (require high isolation & narrow width)
    -1200, -1300, -1100, -1250, -1050,
    // Background/Pileup biases
      600,  -200,  -300,   900,   100,   400,
};

// Layer 2: 16 -> 3 ([Background/Pileup, DiMuon, LocalizedJet])
constexpr int8_t W2[3][16] = {
    // Class 0: Background / Pileup
    {-26, -28, -24, -22, -26,  -24, -26, -22, -28, -20,   24,  30,  30,  22,  24,  26},
    // Class 1: DiMuon
    { 34,  36,  32,  30,  34,  -12, -14, -10, -14, -10,  -20, -22, -22, -18, -18, -20},
    // Class 2: LocalizedJet
    {-12, -14, -10, -10, -12,   34,  36,  32,  36,  30,  -20, -26, -26, -18, -22, -22},
};

constexpr int32_t B2[3] = {
    1200, -800, -800
};

inline int8_t quantize_u01(float v) noexcept {
    const float clamped = std::clamp(v, 0.0f, 1.0f);
    return static_cast<int8_t>(clamped * 127.0f + 0.5f);
}

} // namespace

ClassifierOutput Int8MlpClassifier::predict(const HltFeatures& features) noexcept {
    int8_t q[6];
    q[0] = quantize_u01(features.cluster_centroid / 64.0f);
    q[1] = quantize_u01(features.cluster_width / 25.0f);
    q[2] = quantize_u01(features.isolation_ratio);
    q[3] = quantize_u01(features.dimuon_inv_mass / 150.0f);
    q[4] = quantize_u01(features.total_et_norm);
    q[5] = quantize_u01(features.peak_muon_norm);

    // Layer 1: Int8 MAC -> ReLU -> requantize to [0, 127]
    int8_t h[16];
    for (int i = 0; i < 16; ++i) {
        int32_t acc = B1[i];
        for (int j = 0; j < 6; ++j) {
            acc += static_cast<int32_t>(W1[i][j]) * static_cast<int32_t>(q[j]);
        }
        if (acc <= 0) {
            h[i] = 0;
        } else {
            const int32_t scaled = acc >> 6; // / 64
            h[i] = static_cast<int8_t>(scaled > 127 ? 127 : scaled);
        }
    }

    // Layer 2: Int8 MAC -> 3 logits
    int32_t logits[3];
    for (int c = 0; c < 3; ++c) {
        int32_t acc = B2[c];
        for (int i = 0; i < 16; ++i) {
            acc += static_cast<int32_t>(W2[c][i]) * static_cast<int32_t>(h[i]);
        }
        logits[c] = acc;
    }

    // Softmax with temperature scaling for smooth confidence score in [0, 1000]
    constexpr float INV_TEMP = 1.0f / 2500.0f;
    const float max_l = static_cast<float>(std::max({logits[0], logits[1], logits[2]})) * INV_TEMP;
    const float e0 = std::exp(static_cast<float>(logits[0]) * INV_TEMP - max_l);
    const float e1 = std::exp(static_cast<float>(logits[1]) * INV_TEMP - max_l);
    const float e2 = std::exp(static_cast<float>(logits[2]) * INV_TEMP - max_l);
    const float sum = e0 + e1 + e2;

    const float p_signal = (e1 + e2) / sum;
    const uint16_t score = static_cast<uint16_t>(std::clamp(p_signal * 1000.0f + 0.5f, 0.0f, 1000.0f));

    uint8_t pred = 0;
    if (logits[1] >= logits[0] && logits[1] >= logits[2]) {
        pred = 1;
    } else if (logits[2] >= logits[0] && logits[2] >= logits[1]) {
        pred = 2;
    }

    return ClassifierOutput{score, pred};
}

} // namespace triggerdaq
