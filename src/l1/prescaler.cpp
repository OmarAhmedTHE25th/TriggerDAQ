#include "triggerdaq/l1/prescaler.hpp"

#include <algorithm>
#include <cmath>

namespace triggerdaq {

TriggerMenu TriggerMenu::default_nominal_menu() noexcept {
    TriggerMenu menu{};
    menu.muon_pt_threshold = 12;
    menu.hlt_score_threshold = 500;
    menu.version = 1;

    TriggerRule r0{};
    r0.set_name("L1_DiMu12");
    r0.condition = RuleCondition::DiMuon;
    r0.priority = RulePriority::Critical;
    r0.threshold = 18;
    r0.secondary_threshold = 2;
    r0.base_prescale = 1.0f;
    r0.max_prescale = 1.0f;
    r0.enabled = true;
    menu.add_rule(r0);

    TriggerRule r1{};
    r1.set_name("L1_SingleMu25");
    r1.condition = RuleCondition::SingleMuon;
    r1.priority = RulePriority::High;
    r1.threshold = 25;
    r1.base_prescale = 1.0f;
    r1.max_prescale = 10.0f;
    r1.enabled = true;
    menu.add_rule(r1);

    TriggerRule r2{};
    r2.set_name("L1_JetCluster500");
    r2.condition = RuleCondition::JetCluster;
    r2.priority = RulePriority::Normal;
    r2.threshold = 500;
    r2.base_prescale = 1.0f;
    r2.max_prescale = 25.0f;
    r2.enabled = true;
    menu.add_rule(r2);

    TriggerRule r3{};
    r3.set_name("L1_TotalET1200");
    r3.condition = RuleCondition::TotalET;
    r3.priority = RulePriority::Low;
    r3.threshold = 1200;
    r3.base_prescale = 1.0f;
    r3.max_prescale = 100.0f;
    r3.enabled = true;
    menu.add_rule(r3);

    return menu;
}

Prescaler::Prescaler(float target_fill,
                     float kp,
                     float ki,
                     float ema_alpha,
                     float max_integral,
                     float leak_rate) noexcept
    : target_fill_(target_fill),
      kp_(kp),
      ki_(ki),
      ema_alpha_(std::clamp(ema_alpha, 0.01f, 1.0f)),
      max_integral_(std::max(0.1f, max_integral)),
      leak_rate_(std::max(0.0f, leak_rate)) {
    factors_.fill(1.0f);
    accumulators_.fill(0.0f);
}

void Prescaler::reset(const TriggerMenu& menu) noexcept {
    smoothed_fill_ = 0.0f;
    ema_initialized_ = false;
    integral_error_ = 0.0f;
    for (size_t i = 0; i < TriggerMenu::MAX_RULES; ++i) {
        if (i < menu.num_rules) {
            const auto& r = menu.rules[i];
            factors_[i] = (r.priority == RulePriority::Critical)
                              ? 1.0f
                              : std::max(1.0f, r.base_prescale);
        } else {
            factors_[i] = 1.0f;
        }
        accumulators_[i] = 0.0f;
    }
}

void Prescaler::update_control(float queue_fill_ratio,
                               float dt_sec,
                               const TriggerMenu& menu) noexcept {
    const float clamped_fill = std::clamp(queue_fill_ratio, 0.0f, 1.0f);
    const float safe_dt = std::clamp(dt_sec, 0.0f, 1.0f);

    // EMA smoothing suppresses transient single-tick queue fill spikes
    if (!ema_initialized_) {
        smoothed_fill_ = clamped_fill;
        ema_initialized_ = true;
    } else {
        smoothed_fill_ = ema_alpha_ * clamped_fill + (1.0f - ema_alpha_) * smoothed_fill_;
    }

    // Symmetric leaky-integrator with anti-windup clamp [0.0, max_integral_]
    const float error = smoothed_fill_ - target_fill_;
    const float decayed = integral_error_ * std::exp(-leak_rate_ * safe_dt);
    integral_error_ = std::clamp(decayed + error * safe_dt, 0.0f, max_integral_);

    const float control = std::max(0.0f, kp_ * error + ki_ * integral_error_);

    for (size_t i = 0; i < menu.num_rules && i < TriggerMenu::MAX_RULES; ++i) {
        const auto& r = menu.rules[i];
        if (r.priority == RulePriority::Critical) {
            factors_[i] = 1.0f;
            continue;
        }

        float priority_weight = 1.0f;
        switch (r.priority) {
            case RulePriority::Low:
                priority_weight = 4.0f;
                break;
            case RulePriority::Normal:
                priority_weight = 1.5f;
                break;
            case RulePriority::High:
                priority_weight = 0.4f;
                break;
            case RulePriority::Critical:
                priority_weight = 0.0f;
                break;
        }

        const float base = std::max(1.0f, r.base_prescale);
        const float max_p = std::max(base, r.max_prescale);
        const float target_factor = base + control * priority_weight;
        factors_[i] = std::clamp(target_factor, base, max_p);
    }
}

bool Prescaler::should_accept(size_t rule_idx) noexcept {
    if (rule_idx >= TriggerMenu::MAX_RULES) {
        return false;
    }
    const float factor = factors_[rule_idx];
    if (factor <= 1.0f) {
        return true;
    }
    accumulators_[rule_idx] += 1.0f;
    if (accumulators_[rule_idx] >= factor) {
        accumulators_[rule_idx] -= factor;
        return true;
    }
    return false;
}

} // namespace triggerdaq
