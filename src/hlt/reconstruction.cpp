#include "triggerdaq/hlt/reconstruction.hpp"

#include <cmath>
#include <cstdint>

namespace triggerdaq {

HltFeatures reconstruct_event(const RawEventFrame& frame) noexcept {
    HltFeatures out{};

    // 1. Calorimeter total energy, max 4-cell sliding window, centroid, and cluster width
    uint32_t total_et = 0;
    double weighted_pos_sum = 0.0;
    for (int i = 0; i < 64; ++i) {
        const uint32_t e = frame.calorimeter_cells[i];
        total_et += e;
        weighted_pos_sum += static_cast<double>(i) * static_cast<double>(e);
    }

    uint32_t max_cluster4 = 0;
    int best_win_start = 0;
    for (int i = 0; i <= 60; ++i) {
        const uint32_t s = static_cast<uint32_t>(frame.calorimeter_cells[i]) +
                           static_cast<uint32_t>(frame.calorimeter_cells[i + 1]) +
                           static_cast<uint32_t>(frame.calorimeter_cells[i + 2]) +
                           static_cast<uint32_t>(frame.calorimeter_cells[i + 3]);
        if (s > max_cluster4) {
            max_cluster4 = s;
            best_win_start = i;
        }
    }

    if (total_et > 0) {
        // Compute energy-weighted centroid within the peak 6-cell neighborhood if localized,
        // and global energy-weighted RMS width around the peak window centroid.
        const int lo = best_win_start > 0 ? best_win_start - 1 : 0;
        const int hi = best_win_start + 4 < 63 ? best_win_start + 4 : 63;
        double local_e = 0.0;
        double local_pos = 0.0;
        for (int i = lo; i <= hi; ++i) {
            const double e = static_cast<double>(frame.calorimeter_cells[i]);
            local_e += e;
            local_pos += static_cast<double>(i) * e;
        }
        const double centroid =
            local_e > 0.0 ? (local_pos / local_e) : (weighted_pos_sum / static_cast<double>(total_et));

        double var_sum = 0.0;
        for (int i = 0; i < 64; ++i) {
            const double d = static_cast<double>(i) - centroid;
            var_sum += static_cast<double>(frame.calorimeter_cells[i]) * d * d;
        }
        out.cluster_centroid = static_cast<float>(centroid);
        out.cluster_width = static_cast<float>(std::sqrt(var_sum / static_cast<double>(total_et)));
        out.isolation_ratio =
            static_cast<float>(max_cluster4) / static_cast<float>(total_et);
    }

    out.total_et_norm = static_cast<float>(total_et) / 4000.0f;

    // 2. Di-muon invariant mass proxy from leading & sub-leading muon hits
    uint16_t pt1 = 0;
    uint16_t pt2 = 0;
    int idx1 = -1;
    int idx2 = -1;
    for (int i = 0; i < 16; ++i) {
        const uint16_t pt = frame.muon_hits[i];
        if (pt > pt1) {
            pt2 = pt1;
            idx2 = idx1;
            pt1 = pt;
            idx1 = i;
        } else if (pt > pt2) {
            pt2 = pt;
            idx2 = i;
        }
    }

    out.peak_muon_norm = static_cast<float>(pt1) / 80.0f;

    if (pt1 > 0 && pt2 > 0 && idx1 >= 0 && idx2 >= 0) {
        constexpr double TWO_PI_OVER_16 = 6.28318530717958647692 / 16.0;
        const int diff = idx1 > idx2 ? (idx1 - idx2) : (idx2 - idx1);
        const double dphi = static_cast<double>(diff) * TWO_PI_OVER_16;
        const double m2 = 2.0 * static_cast<double>(pt1) * static_cast<double>(pt2) *
                          (1.0 - std::cos(dphi));
        out.dimuon_inv_mass = m2 > 0.0 ? static_cast<float>(std::sqrt(m2)) : 0.0f;
    }

    return out;
}

} // namespace triggerdaq
