#include "triggerdaq/l1/l1_trigger.hpp"

#include <chrono>
#include <thread>

#include "triggerdaq/core/thread_utils.hpp"
#include "triggerdaq/l1/simd_kernels.hpp"

namespace triggerdaq {

namespace {

inline uint64_t steady_now_ns() noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

void publish_menu_metadata(const TriggerMenu& menu,
                           const Prescaler& prescaler,
                           L1Telemetry& telemetry) noexcept {
    for (size_t i = 0; i < TriggerMenu::MAX_RULES; ++i) {
        if (i < menu.num_rules) {
            telemetry.active_prescales[i].store(
                prescaler.current_factor(i), std::memory_order_relaxed);
        } else {
            telemetry.rule_accept_counts[i].store(0, std::memory_order_relaxed);
            telemetry.active_prescales[i].store(1.0f, std::memory_order_relaxed);
        }
    }
    telemetry.active_menu_num_rules.store(menu.num_rules, std::memory_order_release);
    telemetry.active_menu_version.store(menu.version, std::memory_order_release);
}

} // namespace

uint32_t L1Trigger::evaluate_frame(const RawEventFrame& frame,
                                   const TriggerMenu& menu,
                                   Prescaler& prescaler,
                                   L1Telemetry& telemetry) noexcept {
    const L1Features feats = extract_l1_features(frame, menu.muon_pt_threshold);
    uint32_t trigger_bits = 0;

    for (uint32_t i = 0; i < menu.num_rules && i < TriggerMenu::MAX_RULES; ++i) {
        const auto& rule = menu.rules[i];
        if (!rule.enabled) {
            continue;
        }

        bool fired = false;
        switch (rule.condition) {
            case RuleCondition::SingleMuon:
                fired = feats.max_muon_pt >= rule.threshold;
                break;
            case RuleCondition::DiMuon: {
                const uint32_t min_hits = rule.secondary_threshold > 0 ? rule.secondary_threshold : 2U;
                fired = (feats.muon_hit_count >= min_hits) && (feats.max_muon_pt >= rule.threshold);
                break;
            }
            case RuleCondition::JetCluster:
                fired = feats.max_cluster4_et >= rule.threshold;
                break;
            case RuleCondition::TotalET:
                fired = feats.total_et >= rule.threshold;
                break;
        }

        if (fired && prescaler.should_accept(i)) {
            trigger_bits |= (1U << i);
            telemetry.rule_accept_counts[i].fetch_add(1, std::memory_order_relaxed);
        }
    }

    return trigger_bits;
}

void L1Trigger::run(SlabArena& arena,
                    BoundedChannel<SlotHandle>& l1_queue,
                    BoundedChannel<SlotHandle>& hlt_queue,
                    const AtomicMenuSlot& menu_slot,
                    L1Telemetry& telemetry,
                    uint64_t l1_budget_ns,
                    bool pin_thread,
                    uint32_t core_idx) noexcept {
    if (pin_thread) {
        pin_current_thread(core_idx);
    }

    Prescaler prescaler;
    TriggerMenu active_menu = menu_slot.load();
    prescaler.reset(active_menu);
    publish_menu_metadata(active_menu, prescaler, telemetry);

    uint64_t last_pi_ns = steady_now_ns();
    uint32_t iter_since_pi = 0;
    uint32_t idle_iters = 0;

    for (;;) {
        SlotHandle slot = INVALID_SLOT;
        if (!l1_queue.try_pop(slot)) {
            if (l1_queue.is_closed() && l1_queue.empty()) {
                break;
            }
            idle_backoff(idle_iters);
            continue;
        }
        idle_iters = 0;

        const uint64_t t0 = steady_now_ns();
        const TriggerMenu menu = menu_slot.load();

        if (menu.version != active_menu.version || menu.num_rules != active_menu.num_rules) {
            active_menu = menu;
            prescaler.reset(active_menu);
            publish_menu_metadata(active_menu, prescaler, telemetry);
        } else {
            active_menu = menu;
        }

        // Update PI prescaler every 64 events or >= 1 ms
        if (++iter_since_pi >= 64U) {
            iter_since_pi = 0;
            const uint64_t dt_ns = t0 - last_pi_ns;
            if (dt_ns >= 1'000'000ULL) {
                const float dt_sec = static_cast<float>(dt_ns) * 1e-9f;
                const float fill = static_cast<float>(hlt_queue.fill_ratio());
                prescaler.update_control(fill, dt_sec, active_menu);
                for (size_t i = 0; i < active_menu.num_rules && i < TriggerMenu::MAX_RULES; ++i) {
                    telemetry.active_prescales[i].store(
                        prescaler.current_factor(i), std::memory_order_relaxed);
                }
                last_pi_ns = t0;
            }
        }

        RawEventFrame* frame = arena.get_frame(slot);
        const uint32_t bits = evaluate_frame(*frame, active_menu, prescaler, telemetry);
        frame->l1_trigger_bits = bits;

        const uint64_t t1 = steady_now_ns();
        const uint64_t lat_ns = t1 >= t0 ? (t1 - t0) : 0;
        telemetry.l1_latency_ns.record(lat_ns);
        if (lat_ns > l1_budget_ns) {
            telemetry.l1_budget_violations.fetch_add(1, std::memory_order_relaxed);
        }

        if (bits != 0U) {
            telemetry.l1_accepted.fetch_add(1, std::memory_order_relaxed);
            hlt_queue.push_blocking(slot);
        } else {
            telemetry.l1_rejected.fetch_add(1, std::memory_order_relaxed);
            arena.release(slot);
        }
    }

    hlt_queue.close();
}

} // namespace triggerdaq
