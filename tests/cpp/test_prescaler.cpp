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
    Prescaler prescaler(0.75f, 25.0f, 12.0f);
    prescaler.reset(menu);

    // 1. Nominal operation below 75% watermark: all rules at base prescale (1.0x)
    for (int i = 0; i < 20; ++i) {
        prescaler.update_control(0.30f, 0.01f, menu);
    }
    for (uint32_t r = 0; r < menu.num_rules; ++r) {
        CHECK(std::fabs(prescaler.current_factor(r) - 1.0f) < 1e-5f);
    }

    // 2. Sustained backpressure (90% queue fill > 75% watermark)
    for (int i = 0; i < 100; ++i) {
        prescaler.update_control(0.90f, 0.01f, menu);
    }
    // Rule 0 is Critical -> MUST remain locked at 1.0x
    CHECK(std::fabs(prescaler.current_factor(0) - 1.0f) < 1e-5f);
    // Low priority (rule 3) must prescale harder than Normal (rule 2) and High (rule 1)
    CHECK(prescaler.current_factor(3) > prescaler.current_factor(2));
    CHECK(prescaler.current_factor(2) > prescaler.current_factor(1));
    CHECK(prescaler.current_factor(1) > 1.0f);

    // 3. Deterministic fractional accumulator accuracy test
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
        // 1000 / 2.5 = exact 400 accepts
        CHECK(accepted == 400);
    }

    // 4. Recovery when queue fill drops below 75%
    for (int i = 0; i < 400; ++i) {
        prescaler.update_control(0.20f, 0.01f, menu);
    }
    for (uint32_t r = 0; r < menu.num_rules; ++r) {
        CHECK(std::fabs(prescaler.current_factor(r) - 1.0f) < 1e-4f);
    }

    std::cout << "[PASS] test_prescaler (PI backpressure, Critical lock, fractional accumulator)\n";
    return 0;
}
