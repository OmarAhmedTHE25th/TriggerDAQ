#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

namespace triggerdaq {

enum class RulePriority : uint8_t {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
};

enum class RuleCondition : uint8_t {
    SingleMuon = 0,
    DiMuon = 1,
    JetCluster = 2,
    TotalET = 3,
};

struct TriggerRule {
    char name[32]{};
    RuleCondition condition{RuleCondition::TotalET};
    RulePriority priority{RulePriority::Normal};
    uint32_t threshold{100};
    uint32_t secondary_threshold{0};
    float base_prescale{1.0f};
    float max_prescale{100.0f};
    bool enabled{true};

    void set_name(std::string_view n) noexcept {
        std::memset(name, 0, sizeof(name));
        const size_t len = n.size() < sizeof(name) - 1 ? n.size() : sizeof(name) - 1;
        if (len > 0) {
            std::memcpy(name, n.data(), len);
        }
    }

    [[nodiscard]] std::string_view get_name() const noexcept {
        return std::string_view(name, strnlen(name, sizeof(name)));
    }
};

struct TriggerMenu {
    static constexpr size_t MAX_RULES = 8;

    uint32_t num_rules{0};
    std::array<TriggerRule, MAX_RULES> rules{};
    uint16_t muon_pt_threshold{10};
    uint16_t hlt_score_threshold{500};
    uint64_t version{1};

    bool add_rule(const TriggerRule& rule) noexcept {
        if (num_rules >= MAX_RULES) {
            return false;
        }
        rules[num_rules++] = rule;
        return true;
    }

    static TriggerMenu default_nominal_menu() noexcept;
};

// Deterministic fractional phase-accumulator prescaler driven by a closed-loop PI controller.
class Prescaler {
public:
    Prescaler(float target_fill = 0.75f, float kp = 25.0f, float ki = 12.0f) noexcept;

    void reset(const TriggerMenu& menu) noexcept;

    // Updates prescale factors based on HLT queue occupancy [0.0, 1.0].
    // Rules with priority == Critical remain locked at 1.0x.
    void update_control(float queue_fill_ratio, float dt_sec, const TriggerMenu& menu) noexcept;

    // Deterministic fractional accumulator: acc += 1.0f; if (acc >= factor) { acc -= factor; accept }
    [[nodiscard]] bool should_accept(size_t rule_idx) noexcept;

    [[nodiscard]] float current_factor(size_t rule_idx) const noexcept {
        return rule_idx < TriggerMenu::MAX_RULES ? factors_[rule_idx] : 1.0f;
    }

    [[nodiscard]] const std::array<float, TriggerMenu::MAX_RULES>& factors() const noexcept {
        return factors_;
    }

private:
    float target_fill_{0.75f};
    float kp_{25.0f};
    float ki_{12.0f};
    float integral_error_{0.0f};
    std::array<float, TriggerMenu::MAX_RULES> factors_{};
    std::array<float, TriggerMenu::MAX_RULES> accumulators_{};
};

} // namespace triggerdaq
