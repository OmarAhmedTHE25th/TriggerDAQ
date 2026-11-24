#pragma once

#include <chrono>
#include <cstdint>
#include <thread>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace triggerdaq {

inline void cpu_pause() noexcept {
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__) || defined(__arm__)
    asm volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

// Progressive idle backoff for empty-queue polling loops:
// short pause spins -> thread yield -> brief microsecond sleep to avoid 100% CPU burn at low rates.
inline void idle_backoff(uint32_t& idle_iters) noexcept {
    if (idle_iters < 64U) {
        cpu_pause();
    } else if (idle_iters < 256U) {
        std::this_thread::yield();
    } else {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    if (idle_iters < 1024U) {
        ++idle_iters;
    }
}

// Checks whether the system has enough logical cores to pin all pipeline threads without wrapping onto Core 0.
inline bool can_pin_without_collision(uint32_t total_threads) noexcept {
    const unsigned hw = std::thread::hardware_concurrency();
    return hw > 0 && total_threads <= hw;
}

inline void pin_current_thread(uint32_t core_idx) noexcept {
#if defined(__linux__)
    const unsigned hw = std::thread::hardware_concurrency();
    if (hw > 0 && core_idx < hw) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_idx, &cpuset);
        (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }
#else
    (void)core_idx;
#endif
}

} // namespace triggerdaq
