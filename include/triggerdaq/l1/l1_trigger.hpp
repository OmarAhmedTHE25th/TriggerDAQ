#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <type_traits>

#include "triggerdaq/core/bounded_channel.hpp"
#include "triggerdaq/core/event_frame.hpp"
#include "triggerdaq/core/slab_arena.hpp"
#include "triggerdaq/core/telemetry.hpp"
#include "triggerdaq/core/thread_utils.hpp"
#include "triggerdaq/l1/prescaler.hpp"

namespace triggerdaq {

static_assert(std::is_trivially_copyable_v<TriggerMenu>,
              "TriggerMenu must be trivially copyable for lock-free atomic seqlock transfer");

// Race-free, TSan-clean seqlock menu slot using atomic word storage.
// Hot-path readers are 100% lock-free and receive a consistent TriggerMenu by value.
class AtomicMenuSlot {
public:
    void initialize(const TriggerMenu& initial) noexcept {
        update(initial);
    }

    // Control-plane writer: serialized via write_mu_, publishes atomically via seqlock.
    void update(const TriggerMenu& updated) noexcept {
        std::lock_guard<std::mutex> lock(write_mu_);
        alignas(TriggerMenu) uint64_t buf[NUM_WORDS]{};
        std::memcpy(buf, &updated, sizeof(TriggerMenu));

        const uint64_t s = seq_.load(std::memory_order_relaxed);
        seq_.store(s + 1, std::memory_order_release); // odd: write in progress
        for (size_t i = 0; i < NUM_WORDS; ++i) {
            words_[i].store(buf[i], std::memory_order_relaxed);
        }
        seq_.store(s + 2, std::memory_order_release); // even: stable
    }

    // Hot-path lock-free reader: returns a consistent TriggerMenu snapshot by value.
    [[nodiscard]] TriggerMenu load() const noexcept {
        alignas(TriggerMenu) uint64_t buf[NUM_WORDS]{};
        for (;;) {
            const uint64_t s1 = seq_.load(std::memory_order_acquire);
            if ((s1 & 1ULL) != 0ULL) {
                cpu_pause();
                continue;
            }
            for (size_t i = 0; i < NUM_WORDS; ++i) {
                buf[i] = words_[i].load(std::memory_order_relaxed);
            }
            const uint64_t s2 = seq_.load(std::memory_order_acquire);
            if (s1 == s2) {
                TriggerMenu out{};
                std::memcpy(&out, buf, sizeof(TriggerMenu));
                return out;
            }
            cpu_pause();
        }
    }

private:
    static constexpr size_t NUM_WORDS = (sizeof(TriggerMenu) + sizeof(uint64_t) - 1) / sizeof(uint64_t);

    std::mutex write_mu_;
    alignas(64) std::atomic<uint64_t> seq_{0};
    std::array<std::atomic<uint64_t>, NUM_WORDS> words_{};
};

struct alignas(64) L1Telemetry {
    std::atomic<uint64_t> l1_accepted{0};
    std::atomic<uint64_t> l1_rejected{0};
    std::atomic<uint64_t> l1_budget_violations{0};
    std::atomic<uint64_t> dead_time_events{0};
    std::atomic<uint32_t> active_menu_num_rules{0};
    std::atomic<uint64_t> active_menu_version{0};
    std::array<std::atomic<uint64_t>, TriggerMenu::MAX_RULES> rule_accept_counts{};
    std::array<std::atomic<float>, TriggerMenu::MAX_RULES> active_prescales{};
    AtomicLatencyHistogram l1_latency_ns;

    void reset() noexcept {
        l1_accepted.store(0, std::memory_order_relaxed);
        l1_rejected.store(0, std::memory_order_relaxed);
        l1_budget_violations.store(0, std::memory_order_relaxed);
        dead_time_events.store(0, std::memory_order_relaxed);
        active_menu_num_rules.store(0, std::memory_order_release);
        active_menu_version.store(0, std::memory_order_release);
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

    // Drains l1_queue until it is closed and empty, then signals hlt_queue closed.
    void run(SlabArena& arena,
             BoundedChannel<SlotHandle>& l1_queue,
             BoundedChannel<SlotHandle>& hlt_queue,
             const AtomicMenuSlot& menu_slot,
             L1Telemetry& telemetry,
             uint64_t l1_budget_ns,
             bool pin_thread = false,
             uint32_t core_idx = 1) noexcept;
};

} // namespace triggerdaq
