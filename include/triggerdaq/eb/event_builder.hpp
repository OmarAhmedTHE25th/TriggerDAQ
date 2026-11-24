#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <type_traits>

#include "triggerdaq/core/bounded_channel.hpp"
#include "triggerdaq/core/event_frame.hpp"
#include "triggerdaq/core/slab_arena.hpp"

namespace triggerdaq {

// Independent Calorimeter Read-Out Buffer (ROB) fragment
struct CaloRobFragment {
    uint64_t event_id{0};
    uint64_t bunch_crossing_id{0};
    uint64_t ingest_timestamp_ns{0};
    uint32_t orbit_id{0};
    uint32_t lumi_block_id{1};
    uint16_t bcid{0};
    uint8_t topology_tag{0};
    uint8_t pad{0};
    uint16_t calorimeter_cells[CALORIMETER_CELLS]{};
};

static_assert(std::is_trivially_copyable_v<CaloRobFragment>,
              "CaloRobFragment must be trivially copyable");

// Independent Muon Read-Out Buffer (ROB) fragment
struct MuonRobFragment {
    uint64_t event_id{0};
    uint32_t orbit_id{0};
    uint16_t bcid{0};
    uint16_t pad{0};
    uint16_t muon_hits[MUON_CHAMBERS]{};
};

static_assert(std::is_trivially_copyable_v<MuonRobFragment>,
              "MuonRobFragment must be trivially copyable");

struct alignas(64) EventBuilderTelemetry {
    std::atomic<uint64_t> events_assembled{0};
    std::atomic<uint64_t> partial_events{0};
    std::atomic<uint64_t> timeout_events{0};
    std::atomic<uint64_t> late_fragments_dropped{0};
    std::atomic<uint64_t> calo_fragments_rx{0};
    std::atomic<uint64_t> muon_fragments_rx{0};

    void reset() noexcept {
        events_assembled.store(0, std::memory_order_relaxed);
        partial_events.store(0, std::memory_order_relaxed);
        timeout_events.store(0, std::memory_order_relaxed);
        late_fragments_dropped.store(0, std::memory_order_relaxed);
        calo_fragments_rx.store(0, std::memory_order_relaxed);
        muon_fragments_rx.store(0, std::memory_order_relaxed);
    }
};

// Zero-allocation Event Builder stage assembling independent Calo & Muon ROB
// fragments into unified SlabArena RawEventFrame slots by event_id, with
// configurable fragment timeout flagging partial events via FRAME_STATUS_PARTIAL_EB.
class EventBuilder {
public:
    static constexpr size_t DEFAULT_TABLE_SIZE = 16384;

    EventBuilder() = default;

    void initialize(size_t table_capacity = DEFAULT_TABLE_SIZE, uint64_t timeout_us = 500);

    // Drains both ROB channels until closed & empty, flushes any remaining pending
    // partial frames, and closes l1_queue cleanly without leaking any SlabArena slots.
    void run(SlabArena& arena,
             BoundedChannel<CaloRobFragment>& calo_queue,
             BoundedChannel<MuonRobFragment>& muon_queue,
             BoundedChannel<SlotHandle>& l1_queue,
             EventBuilderTelemetry& telemetry,
             bool pin_thread = false,
             uint32_t core_idx = 1) noexcept;

private:
    struct PendingEntry {
        uint64_t event_id{0};
        uint64_t first_arrival_ns{0};
        SlotHandle slot{INVALID_SLOT};
        bool occupied{false};
        bool has_calo{false};
        bool has_muon{false};
    };

    PendingEntry* find_or_allocate(uint64_t event_id,
                                   uint64_t now_ns,
                                   SlabArena& arena,
                                   BoundedChannel<SlotHandle>& l1_queue,
                                   EventBuilderTelemetry& telemetry) noexcept;

    void emit_entry(PendingEntry& entry,
                    SlabArena& arena,
                    BoundedChannel<SlotHandle>& l1_queue,
                    EventBuilderTelemetry& telemetry,
                    bool is_timeout) noexcept;

    void sweep_timeouts(uint64_t now_ns,
                        SlabArena& arena,
                        BoundedChannel<SlotHandle>& l1_queue,
                        EventBuilderTelemetry& telemetry) noexcept;

    static constexpr size_t TIMED_OUT_CACHE_SIZE = 4096;
    static constexpr size_t TIMED_OUT_CACHE_MASK = TIMED_OUT_CACHE_SIZE - 1;

    [[nodiscard]] bool is_recently_timed_out(uint64_t event_id) const noexcept;
    void record_timed_out(uint64_t event_id) noexcept;

    std::unique_ptr<PendingEntry[]> table_;
    std::unique_ptr<uint64_t[]> timed_out_ids_;
    size_t table_size_{0};
    size_t table_mask_{0};
    uint64_t timeout_ns_{500'000ULL};
};

} // namespace triggerdaq
