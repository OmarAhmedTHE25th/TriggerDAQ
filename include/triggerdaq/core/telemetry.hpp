#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace triggerdaq {

struct LatencyStats {
    double min{0.0};
    double p50{0.0};
    double p95{0.0};
    double p99{0.0};
    double p999{0.0};
    double max{0.0};
};

// Fixed-size, zero-allocation log-linear latency histogram.
// Linear for [0, 64), then 16 sub-buckets per power-of-two decade up to ~2^30.
// Percentiles use within-bucket linear interpolation clamped to [min, max].
class LatencyHistogram {
public:
    static constexpr size_t NUM_LINEAR = 64;
    static constexpr size_t SUB_BITS = 4; // 16 sub-buckets per octave
    static constexpr size_t SUB_COUNT = 1U << SUB_BITS;
    static constexpr size_t MAX_OCTAVES = 24; // covers up to 2^30
    static constexpr size_t NUM_BUCKETS = NUM_LINEAR + MAX_OCTAVES * SUB_COUNT;

    void reset() noexcept;
    void record(uint64_t value) noexcept;
    void merge_from(const LatencyHistogram& other) noexcept;

    [[nodiscard]] LatencyStats compute_stats() const noexcept;
    [[nodiscard]] uint64_t total_count() const noexcept { return total_count_; }
    [[nodiscard]] uint64_t min_value() const noexcept { return total_count_ > 0 ? min_value_ : 0; }
    [[nodiscard]] uint64_t max_value() const noexcept { return max_value_; }

private:
    friend class AtomicLatencyHistogram;
    static size_t value_to_bucket(uint64_t value) noexcept;
    static double bucket_lower_bound(size_t bucket) noexcept;
    static double bucket_upper_bound(size_t bucket) noexcept;

    std::array<uint64_t, NUM_BUCKETS> buckets_{};
    uint64_t total_count_{0};
    uint64_t min_value_{std::numeric_limits<uint64_t>::max()};
    uint64_t max_value_{0};
};

// Single-writer, multi-reader seqlock-protected latency histogram.
// Prevents transient torn percentile reads during concurrent snapshot aggregation.
class AtomicLatencyHistogram {
public:
    void reset() noexcept;
    void record(uint64_t value) noexcept;
    void copy_to(LatencyHistogram& out) const noexcept;

private:
    static size_t value_to_bucket(uint64_t value) noexcept;

    alignas(64) std::atomic<uint64_t> seq_{0};
    std::array<std::atomic<uint64_t>, LatencyHistogram::NUM_BUCKETS> buckets_{};
    std::atomic<uint64_t> total_count_{0};
    std::atomic<uint64_t> min_value_{std::numeric_limits<uint64_t>::max()};
    std::atomic<uint64_t> max_value_{0};
};

struct alignas(64) WorkerTelemetry {
    std::atomic<uint64_t> hlt_accepted{0};
    std::atomic<uint64_t> hlt_rejected{0};
    std::atomic<uint64_t> hlt_true_positives{0};
    std::atomic<uint64_t> hlt_false_positives{0};
    std::atomic<uint64_t> dead_time_events{0};
    AtomicLatencyHistogram hlt_latency_ns;

    void reset() noexcept {
        hlt_accepted.store(0, std::memory_order_relaxed);
        hlt_rejected.store(0, std::memory_order_relaxed);
        hlt_true_positives.store(0, std::memory_order_relaxed);
        hlt_false_positives.store(0, std::memory_order_relaxed);
        dead_time_events.store(0, std::memory_order_relaxed);
        hlt_latency_ns.reset();
    }
};

struct TelemetrySnapshot {
    uint64_t events_generated{0};
    uint64_t l1_accepted{0};
    uint64_t l1_rejected{0};
    uint64_t hlt_accepted{0};
    uint64_t hlt_rejected{0};
    uint64_t dead_time_events{0};
    uint64_t dead_time_ns{0};
    uint64_t l1_budget_violations{0};
    uint64_t bytes_written{0};
    uint64_t events_written{0};
    uint64_t storage_write_errors{0};

    uint64_t arena_allocated{0};
    uint64_t arena_peak{0};
    uint64_t arena_capacity{0};

    double l1_queue_fill{0.0};
    double hlt_queue_fill{0.0};
    double storage_queue_fill{0.0};

    LatencyStats l1_latency_ns{};
    LatencyStats hlt_latency_us{};

    uint32_t num_rules{0};
    std::array<float, 8> prescale_factors{};
    std::array<uint64_t, 8> rule_accept_counts{};

    uint64_t signal_generated{0};
    uint64_t hlt_true_positives{0};
    uint64_t hlt_false_positives{0};

    double elapsed_seconds{0.0};
    double input_rate_hz{0.0};
    double l1_rate_hz{0.0};
    double hlt_rate_hz{0.0};
    double dead_time_fraction{0.0};
    double signal_purity{0.0};
    double signal_efficiency{0.0};
};

static_assert(std::is_trivially_copyable_v<TelemetrySnapshot>,
              "TelemetrySnapshot must be trivially copyable");

} // namespace triggerdaq
