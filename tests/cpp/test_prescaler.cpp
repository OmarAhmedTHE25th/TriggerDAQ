#include <cmath>
#include <cstdlib>
#include <iostream>

#include "triggerdaq/l1/prescaler.hpp"

using namespace triggerdaq;

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::cerr << "CHECK FAILED at " << __FILE__ << ":" << __LINE__ << ": " #cond \
                      << "\n";                                                           \
            std::abort();                                                                \
        }                                                                                \
    } while (0)

int main() {
    TriggerMenu menu = TriggerMenu::default_nominal_menu();
    Prescaler prescaler(0.75f, 25.0f, 12.0f, 0.25f, 4.0f, 0.5f);
    prescaler.reset(menu);

    // 1. Nominal operation below 75% watermark: all rules at base prescale (1.0x)
    for (int i = 0; i < 20; ++i) {
        prescaler.update_control(0.30f, 0.01f, menu);
    }
    for (uint32_t r = 0; r < menu.num_rules; ++r) {
        CHECK(std::fabs(prescaler.current_factor(r) - 1.0f) < 1e-5f);
    }

    // 2. Single-tick transient spike suppression via EMA smoothing
    prescaler.update_control(0.95f, 0.01f, menu);
    // Smoothed fill = 0.25*0.95 + 0.75*0.30 = 0.4625 < 0.75 -> no over-reaction!
    CHECK(prescaler.smoothed_fill() < 0.75f);
    CHECK(std::fabs(prescaler.current_factor(3) - 1.0f) < 1e-5f);

    // 3. Sustained backpressure (95% queue fill > 75% watermark)
    for (int i = 0; i < 200; ++i) {
        prescaler.update_control(0.95f, 0.05f, menu);
    }
    // Anti-windup clamp strictly enforced
    CHECK(prescaler.integral_error() <= 4.0f + 1e-5f);
    // Rule 0 is Critical -> MUST remain locked at 1.0x
    CHECK(std::fabs(prescaler.current_factor(0) - 1.0f) < 1e-5f);
    // Low priority (rule 3) must prescale harder than Normal (rule 2) and High (rule 1)
    CHECK(prescaler.current_factor(3) > prescaler.current_factor(2));
    CHECK(prescaler.current_factor(2) > prescaler.current_factor(1));
    CHECK(prescaler.current_factor(1) > 1.0f);

    // 4. Deterministic fractional accumulator accuracy test
    {
        TriggerMenu frac_menu{};
        TriggerRule r{};
        r.set_name("FracRule");
        r.priority = RulePriority::Normal;
        r.base_prescale = 2.5f;
        r.max_prescale = 10.0f;
        r.enabled = true;
        frac_menu.add_rule(r);

        Prescaler p;
        p.reset(frac_menu);
        int accepted = 0;
        for (int i = 0; i < 1000; ++i) {
            if (p.should_accept(0)) {
                ++accepted;
            }
        }
        CHECK(accepted == 400);
    }

    // 5. Smooth symmetric leaky-integrator recovery when queue fill drops below 75%
    for (int i = 0; i < 200; ++i) {
        prescaler.update_control(0.20f, 0.05f, menu);
    }
    CHECK(prescaler.integral_error() < 1e-4f);
    for (uint32_t r = 0; r < menu.num_rules; ++r) {
        CHECK(std::fabs(prescaler.current_factor(r) - 1.0f) < 1e-4f);
    }

    std::cout << "[PASS] test_prescaler (EMA smoothing, anti-windup, Critical 1.0x lock, symmetric recovery)\n";
    return 0;
}
