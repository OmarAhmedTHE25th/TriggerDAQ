#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

#include "triggerdaq/core/slab_arena.hpp"

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
    // 1. Exhaustion and recovery test
    {
        SlabArena arena;
        arena.initialize(128);
        std::vector<SlotHandle> acquired;
        acquired.reserve(128);
        for (int i = 0; i < 128; ++i) {
            SlotHandle h = arena.acquire();
            CHECK(h != INVALID_SLOT);
            acquired.push_back(h);
        }
        CHECK(arena.allocated_count() == 128);
        CHECK(arena.peak_occupancy() == 128);
        CHECK(arena.acquire() == INVALID_SLOT);
        CHECK(arena.exhaustion_drops() == 1);

        for (SlotHandle h : acquired) {
            arena.release(h);
        }
        CHECK(arena.allocated_count() == 0);
        SlotHandle h = arena.acquire();
        CHECK(h != INVALID_SLOT);
        arena.release(h);
        CHECK(arena.allocated_count() == 0);
    }

    // 2. 8-thread stress test: 10,000,000 total acquire/release cycles
    {
        constexpr uint32_t CAPACITY = 4096;
        constexpr int NUM_THREADS = 8;
        constexpr uint64_t CYCLES_PER_THREAD = 1'250'000; // 8 * 1.25M = 10M cycles

        SlabArena arena;
        arena.initialize(CAPACITY);

        std::vector<std::atomic<uint32_t>> in_use(CAPACITY);
        for (auto& flag : in_use) {
            flag.store(0, std::memory_order_relaxed);
        }

        std::atomic<uint64_t> total_ops{0};
        std::vector<std::jthread> threads;
        threads.reserve(NUM_THREADS);

        for (int t = 0; t < NUM_THREADS; ++t) {
            threads.emplace_back([&, t]() {
                for (uint64_t i = 0; i < CYCLES_PER_THREAD; ++i) {
                    SlotHandle h = arena.acquire();
                    CHECK(h != INVALID_SLOT);
                    CHECK(h < CAPACITY);

                    // Verify exclusive ownership (no duplicate slot ever handed out)
                    const uint32_t prev = in_use[h].exchange(1, std::memory_order_acq_rel);
                    CHECK(prev == 0);

                    // Touch the 64-byte aligned frame memory
                    RawEventFrame* f = arena.get_frame(h);
                    CHECK((reinterpret_cast<uintptr_t>(f) & 63U) == 0U);
                    f->event_id = (static_cast<uint64_t>(t) << 32) | i;

                    const uint32_t cleared = in_use[h].exchange(0, std::memory_order_acq_rel);
                    CHECK(cleared == 1);

                    arena.release(h);
                }
                total_ops.fetch_add(CYCLES_PER_THREAD, std::memory_order_relaxed);
            });
        }
        threads.clear(); // Joins all jthreads

        CHECK(total_ops.load() == static_cast<uint64_t>(NUM_THREADS) * CYCLES_PER_THREAD);
        CHECK(arena.allocated_count() == 0);
        CHECK(arena.exhaustion_drops() == 0);
        CHECK(arena.peak_occupancy() <= CAPACITY);
    }

    std::cout << "[PASS] test_slab_arena (10M cycles across 8 threads)\n";
    return 0;
}
