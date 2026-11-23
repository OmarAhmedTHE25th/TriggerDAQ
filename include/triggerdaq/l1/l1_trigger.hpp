#pragma once

#include <array>
#include <atomic>
#include <cstdint>

#include "triggerdaq/core/bounded_channel.hpp"
#include "triggerdaq/core/event_frame.hpp"
#include "triggerdaq/core/slab_arena.hpp"
#include "triggerdaq/core/telemetry.hpp"
#include "triggerdaq/l1/prescaler.hpp"

namespace triggerdaq {

// Lock-free double-buffered menu holder for zero-allocation hot-reload.
class AtomicMenuSlot {
public:
    void initialize(const TriggerMenu& initial) noexcept {
        menus_[0] = initial;
        menus_[1] = initial;
        active_idx_.store(0, std::memory_order_release);
    }

    // Writer side (control plane): writes to inactive buffer and flips atomic index.
    void update(const TriggerMenu& updated) noexcept {
        const uint32_t cur = active_idx_.load(std::memory_order_relaxed);
        const uint32_t next = 1U - cur;
        menus_[next] = updated;
        active_idx_.store(next, std::memory_order_release);
    }

    // Reader side (hot path): lock-free read of active menu.
    [[nodiscard]] const TriggerMenu& load() const noexcept {
        return menus_[active_idx_.load(std::memory_order_acquire)];
    }

private:
    TriggerMenu menus_[2]{};
    alignas(64) std::atomic<uint32_t> active_idx_{0};
};

struct alignas(64) L1Telemetry {
    std::atomic<uint64_t> l1_accepted{0};
    std::atomic<uint64_t> l1_rejected{0};
    std::atomic<uint64_t> l1_budget_violations{0};
    std::atomic<uint64_t> dead_time_events{0};
    std::array<std::atomic<uint64_t>, TriggerMenu::MAX_RULES> rule_accept_counts{};
    std::array<std::atomic<float>, TriggerMenu::MAX_RULES> active_prescales{};
    AtomicLatencyHistogram l1_latency_ns;

    void reset() noexcept {
        l1_accepted.store(0, std::memory_order_relaxed);
        l1_rejected.store(0, std::memory_order_relaxed);
        l1_budget_violations.store(0, std::memory_order_relaxed);
        dead_time_events.store(0, std::memory_order_relaxed);
        for (size_t i = 0; i < TriggerMenu::MAX_RULES; ++i) {
            rule_accept_counts[i].store(0, std::memory_order_relaxed);
            active_prescales[i].store(1.0f, std::memory_order_relaxed);
        }
        l1_latency_ns.reset();
    }
};

class L1Trigger {
public:
    L1Trigger() = default;

    // Evaluates a single frame against the menu and prescaler, returning trigger bitmask (0 = reject).
    static uint32_t evaluate_frame(const RawEventFrame& frame,
                                   const TriggerMenu& menu,
                                   Prescaler& prescaler,
                                   L1Telemetry& telemetry) noexcept;

    void run(std::atomic<bool>& stop_flag,
             SlabArena& arena,
             BoundedChannel<SlotHandle>& l1_queue,
             BoundedChannel<SlotHandle>& hlt_queue,
             const AtomicMenuSlot& menu_slot,
             L1Telemetry& telemetry,
             uint64_t l1_budget_ns) noexcept;
};

} // namespace triggerdaq
