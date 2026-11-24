#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace triggerdaq {

inline constexpr size_t CALORIMETER_CELLS = 64;
inline constexpr size_t MUON_CHAMBERS = 16;

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
    uint16_t calorimeter_cells[CALORIMETER_CELLS];
    uint16_t muon_hits[MUON_CHAMBERS];
    uint32_t l1_trigger_bits;
    uint16_t hlt_score;
    uint8_t topology_tag;
    uint8_t reserved[45];
    uint8_t explicit_pad[20];
};

static_assert(sizeof(RawEventFrame) == 256, "RawEventFrame must be exactly 256 bytes");
static_assert(alignof(RawEventFrame) == 64, "RawEventFrame must be 64-byte aligned");
static_assert(std::is_trivially_copyable_v<RawEventFrame>, "RawEventFrame must be trivially copyable");
static_assert(offsetof(RawEventFrame, event_id) == 0);
static_assert(offsetof(RawEventFrame, bunch_crossing_id) == 8);
static_assert(offsetof(RawEventFrame, ingest_timestamp_ns) == 16);
static_assert(offsetof(RawEventFrame, calorimeter_cells) == 24);
static_assert(offsetof(RawEventFrame, muon_hits) == 152);
static_assert(offsetof(RawEventFrame, l1_trigger_bits) == 184);
static_assert(offsetof(RawEventFrame, hlt_score) == 188);
static_assert(offsetof(RawEventFrame, topology_tag) == 190);
static_assert(offsetof(RawEventFrame, reserved) == 191);
static_assert(offsetof(RawEventFrame, explicit_pad) == 236);
static_assert(offsetof(RawEventFrame, explicit_pad) + sizeof(RawEventFrame::explicit_pad) == 256,
              "RawEventFrame must have zero implicit compiler tail padding");

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
static_assert(offsetof(HltDecisionMetadata, hlt_timestamp_ns) == 0);
static_assert(offsetof(HltDecisionMetadata, cluster_centroid) == 8);
static_assert(offsetof(HltDecisionMetadata, cluster_width) == 12);
static_assert(offsetof(HltDecisionMetadata, isolation_ratio) == 16);
static_assert(offsetof(HltDecisionMetadata, dimuon_inv_mass) == 20);
static_assert(offsetof(HltDecisionMetadata, hlt_score) == 24);
static_assert(offsetof(HltDecisionMetadata, predicted_class) == 26);
static_assert(offsetof(HltDecisionMetadata, accepted) == 27);
static_assert(offsetof(HltDecisionMetadata, worker_id) == 28);
static_assert(offsetof(HltDecisionMetadata, worker_id) + sizeof(HltDecisionMetadata::worker_id) == 32,
              "HltDecisionMetadata must have zero implicit compiler padding");

} // namespace triggerdaq
