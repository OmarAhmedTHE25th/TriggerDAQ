#include "triggerdaq/eb/event_builder.hpp"

#include <chrono>
#include <cstring>

#include "triggerdaq/core/thread_utils.hpp"

namespace triggerdaq {

namespace {

inline uint64_t steady_now_ns() noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

inline size_t hash_event_id(uint64_t id) noexcept {
    id ^= id >> 33;
    id *= 0xff51afd7ed558ccdULL;
    id ^= id >> 33;
    return static_cast<size_t>(id);
}

} // namespace

void EventBuilder::initialize(size_t table_capacity, uint64_t timeout_us) {
    size_t sz = 64;
    while (sz < table_capacity) {
        sz <<= 1;
    }
    table_size_ = sz;
    table_mask_ = sz - 1;
    timeout_ns_ = timeout_us * 1000ULL;
    table_ = std::make_unique<PendingEntry[]>(table_size_);
    timed_out_ids_ = std::make_unique<uint64_t[]>(TIMED_OUT_CACHE_SIZE);
    std::memset(timed_out_ids_.get(), 0, TIMED_OUT_CACHE_SIZE * sizeof(uint64_t));
}

bool EventBuilder::is_recently_timed_out(uint64_t event_id) const noexcept {
    if (event_id == 0 || !timed_out_ids_) {
        return false;
    }
    return timed_out_ids_[hash_event_id(event_id) & TIMED_OUT_CACHE_MASK] == event_id;
}

void EventBuilder::record_timed_out(uint64_t event_id) noexcept {
    if (event_id != 0 && timed_out_ids_) {
        timed_out_ids_[hash_event_id(event_id) & TIMED_OUT_CACHE_MASK] = event_id;
    }
}

void EventBuilder::emit_entry(PendingEntry& entry,
                              SlabArena& arena,
                              BoundedChannel<SlotHandle>& l1_queue,
                              EventBuilderTelemetry& telemetry,
                              bool is_timeout) noexcept {
    if (!entry.occupied) {
        return;
    }
    const SlotHandle slot = entry.slot;
    const bool partial = !(entry.has_calo && entry.has_muon);

    if (slot != INVALID_SLOT) {
        if (partial) {
            RawEventFrame* frame = arena.get_frame(slot);
            frame->status_flags |= FRAME_STATUS_PARTIAL_EB;
            telemetry.partial_events.fetch_add(1, std::memory_order_relaxed);
            if (is_timeout) {
                telemetry.timeout_events.fetch_add(1, std::memory_order_relaxed);
                record_timed_out(entry.event_id);
            }
        }
        telemetry.events_assembled.fetch_add(1, std::memory_order_relaxed);
        l1_queue.push_blocking(slot);
    }

    entry.occupied = false;
    entry.has_calo = false;
    entry.has_muon = false;
    entry.slot = INVALID_SLOT;
    entry.event_id = 0;
}

EventBuilder::PendingEntry* EventBuilder::find_or_allocate(
    uint64_t event_id,
    uint64_t now_ns,
    SlabArena& arena,
    BoundedChannel<SlotHandle>& l1_queue,
    EventBuilderTelemetry& telemetry) noexcept {
    const size_t start = hash_event_id(event_id) & table_mask_;
    PendingEntry* first_free = nullptr;
    PendingEntry* oldest = nullptr;

    constexpr size_t MAX_PROBE = 32;
    for (size_t step = 0; step < MAX_PROBE; ++step) {
        PendingEntry& e = table_[(start + step) & table_mask_];
        if (e.occupied) {
            if (e.event_id == event_id) {
                return &e;
            }
            if (oldest == nullptr || e.first_arrival_ns < oldest->first_arrival_ns) {
                oldest = &e;
            }
        } else if (first_free == nullptr) {
            first_free = &e;
        }
    }

    PendingEntry* target = first_free;
    if (target == nullptr) {
        // Evict oldest partial entry to make room without leaking its SlabArena slot
        emit_entry(*oldest, arena, l1_queue, telemetry, true);
        target = oldest;
    }

    SlotHandle slot = arena.acquire();
    while (slot == INVALID_SLOT) {
        // Reclaim timed-out entries if arena is momentarily exhausted
        sweep_timeouts(steady_now_ns(), arena, l1_queue, telemetry);
        cpu_pause();
        slot = arena.acquire();
    }

    RawEventFrame* frame = arena.get_frame(slot);
    std::memset(frame, 0, sizeof(RawEventFrame));
    frame->event_id = event_id;

    target->occupied = true;
    target->event_id = event_id;
    target->first_arrival_ns = now_ns;
    target->slot = slot;
    target->has_calo = false;
    target->has_muon = false;
    return target;
}

void EventBuilder::sweep_timeouts(uint64_t now_ns,
                                  SlabArena& arena,
                                  BoundedChannel<SlotHandle>& l1_queue,
                                  EventBuilderTelemetry& telemetry) noexcept {
    for (size_t i = 0; i < table_size_; ++i) {
        PendingEntry& e = table_[i];
        if (e.occupied && now_ns >= e.first_arrival_ns &&
            (now_ns - e.first_arrival_ns) >= timeout_ns_) {
            emit_entry(e, arena, l1_queue, telemetry, true);
        }
    }
}

void EventBuilder::run(SlabArena& arena,
                       BoundedChannel<CaloRobFragment>& calo_queue,
                       BoundedChannel<MuonRobFragment>& muon_queue,
                       BoundedChannel<SlotHandle>& l1_queue,
                       EventBuilderTelemetry& telemetry,
                       bool pin_thread,
                       uint32_t core_idx) noexcept {
    if (pin_thread) {
        pin_current_thread(core_idx);
    }

    for (size_t i = 0; i < table_size_; ++i) {
        table_[i] = PendingEntry{};
    }
    if (timed_out_ids_) {
        std::memset(timed_out_ids_.get(), 0, TIMED_OUT_CACHE_SIZE * sizeof(uint64_t));
    }

    uint64_t last_sweep_ns = steady_now_ns();
    uint32_t ops_since_sweep = 0;
    uint32_t idle_iters = 0;

    for (;;) {
        bool did_work = false;
        const uint64_t now_ns = steady_now_ns();

        CaloRobFragment calo{};
        if (calo_queue.try_pop(calo)) {
            did_work = true;
            telemetry.calo_fragments_rx.fetch_add(1, std::memory_order_relaxed);
            if (is_recently_timed_out(calo.event_id)) {
                telemetry.late_fragments_dropped.fetch_add(1, std::memory_order_relaxed);
            } else {
                PendingEntry* e = find_or_allocate(calo.event_id, now_ns, arena, l1_queue, telemetry);
                RawEventFrame* frame = arena.get_frame(e->slot);
                frame->event_id = calo.event_id;
                frame->bunch_crossing_id = calo.bunch_crossing_id;
                frame->ingest_timestamp_ns = calo.ingest_timestamp_ns;
                frame->orbit_id = calo.orbit_id;
                frame->lumi_block_id = calo.lumi_block_id;
                frame->bcid = calo.bcid;
                frame->topology_tag = calo.topology_tag;
                std::memcpy(frame->calorimeter_cells, calo.calorimeter_cells, sizeof(calo.calorimeter_cells));
                e->has_calo = true;
                if (e->has_calo && e->has_muon) {
                    emit_entry(*e, arena, l1_queue, telemetry, false);
                }
            }
        }

        MuonRobFragment muon{};
        if (muon_queue.try_pop(muon)) {
            did_work = true;
            telemetry.muon_fragments_rx.fetch_add(1, std::memory_order_relaxed);
            if (is_recently_timed_out(muon.event_id)) {
                telemetry.late_fragments_dropped.fetch_add(1, std::memory_order_relaxed);
            } else {
                PendingEntry* e = find_or_allocate(muon.event_id, now_ns, arena, l1_queue, telemetry);
                RawEventFrame* frame = arena.get_frame(e->slot);
                frame->event_id = muon.event_id;
                if (!e->has_calo) {
                    frame->orbit_id = muon.orbit_id;
                    frame->bcid = muon.bcid;
                }
                std::memcpy(frame->muon_hits, muon.muon_hits, sizeof(muon.muon_hits));
                e->has_muon = true;
                if (e->has_calo && e->has_muon) {
                    emit_entry(*e, arena, l1_queue, telemetry, false);
                }
            }
        }

        if (++ops_since_sweep >= 128U || (!did_work && (now_ns - last_sweep_ns) >= timeout_ns_ / 2)) {
            ops_since_sweep = 0;
            if (now_ns >= last_sweep_ns && (now_ns - last_sweep_ns) >= timeout_ns_ / 2) {
                sweep_timeouts(now_ns, arena, l1_queue, telemetry);
                last_sweep_ns = now_ns;
            }
        }

        if (did_work) {
            idle_iters = 0;
        } else {
            if (calo_queue.is_closed() && calo_queue.empty() &&
                muon_queue.is_closed() && muon_queue.empty()) {
                break;
            }
            idle_backoff(idle_iters);
        }
    }

    // Flush all remaining partial entries on shutdown so zero SlabArena slots leak
    for (size_t i = 0; i < table_size_; ++i) {
        if (table_[i].occupied) {
            emit_entry(table_[i], arena, l1_queue, telemetry, true);
        }
    }

    l1_queue.close();
}

} // namespace triggerdaq
