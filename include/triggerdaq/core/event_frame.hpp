#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace triggerdaq {

enum class TopologyTag : uint8_t {
    Background = 0,
    DiMuon = 1,
    LocalizedJet = 2,
    PileupNoise = 3,
};

struct alignas(64) RawEventFrame {
    uint64_t event_id;
    uint64_t bunch_crossing_id;
    uint64_t ingest_timestamp_ns;
    uint16_t calorimeter_cells[64];
    uint16_t muon_hits[16];
    uint32_t l1_trigger_bits;
    uint16_t hlt_score;
    uint8_t topology_tag;
    uint8_t reserved[45];
};

static_assert(sizeof(RawEventFrame) == 256, "RawEventFrame must be exactly 256 bytes");
static_assert(alignof(RawEventFrame) == 64, "RawEventFrame must be 64-byte aligned");
static_assert(std::is_trivially_copyable_v<RawEventFrame>, "RawEventFrame must be trivially copyable");

struct HltDecisionMetadata {
    uint64_t hlt_timestamp_ns;
    float cluster_centroid;
    float cluster_width;
    float isolation_ratio;
    float dimuon_inv_mass;
    uint16_t hlt_score;
    uint8_t predicted_class;
    uint8_t accepted;
    uint32_t worker_id;
};

static_assert(sizeof(HltDecisionMetadata) == 32, "HltDecisionMetadata must be exactly 32 bytes");
static_assert(std::is_trivially_copyable_v<HltDecisionMetadata>, "HltDecisionMetadata must be trivially copyable");

} // namespace triggerdaq
