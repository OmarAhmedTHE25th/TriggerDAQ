#include "triggerdaq/core/telemetry.hpp"

#include <algorithm>
#include <cmath>

namespace triggerdaq {

size_t LatencyHistogram::value_to_bucket(uint64_t value) noexcept {
    if (value < NUM_LINEAR) {
        return static_cast<size_t>(value);
    }
    const unsigned bit_width = static_cast<unsigned>(std::bit_width(value));
    // bit_width >= 7 for value >= 64
    const size_t octave = static_cast<size_t>(bit_width - 7);
    if (octave >= MAX_OCTAVES) {
        return NUM_BUCKETS - 1;
    }
    const unsigned shift = bit_width - 1 - static_cast<unsigned>(SUB_BITS);
    const size_t sub = static_cast<size_t>((value >> shift) & (SUB_COUNT - 1));
    return NUM_LINEAR + octave * SUB_COUNT + sub;
}

double LatencyHistogram::bucket_upper_bound(size_t bucket) noexcept {
    if (bucket < NUM_LINEAR) {
        return static_cast<double>(bucket);
    }
    const size_t rel = bucket - NUM_LINEAR;
    const size_t octave = rel / SUB_COUNT;
    const size_t sub = rel % SUB_COUNT;
    const uint64_t base = uint64_t{64} << octave;
    const uint64_t step = base >> SUB_BITS;
    return static_cast<double>(base + (sub + 1) * step);
}

void LatencyHistogram::reset() noexcept {
    buckets_.fill(0);
    total_count_ = 0;
    max_value_ = 0;
}

void LatencyHistogram::record(uint64_t value) noexcept {
    const size_t b = value_to_bucket(value);
    ++buckets_[b];
    ++total_count_;
    if (value > max_value_) {
        max_value_ = value;
    }
}

void LatencyHistogram::merge_from(const LatencyHistogram& other) noexcept {
    for (size_t i = 0; i < NUM_BUCKETS; ++i) {
        buckets_[i] += other.buckets_[i];
    }
    total_count_ += other.total_count_;
    if (other.max_value_ > max_value_) {
        max_value_ = other.max_value_;
    }
}

LatencyStats LatencyHistogram::compute_stats() const noexcept {
    LatencyStats stats{};
    if (total_count_ == 0) {
        return stats;
    }
    stats.max = static_cast<double>(max_value_);

    const auto find_pct = [&](double pct) -> double {
        const uint64_t target = static_cast<uint64_t>(
            std::ceil(pct * static_cast<double>(total_count_)));
        uint64_t acc = 0;
        for (size_t i = 0; i < NUM_BUCKETS; ++i) {
            acc += buckets_[i];
            if (acc >= target) {
                const double upper = bucket_upper_bound(i);
                return upper > stats.max ? stats.max : upper;
            }
        }
        return stats.max;
    };

    stats.p50 = find_pct(0.50);
    stats.p95 = find_pct(0.95);
    stats.p99 = find_pct(0.99);
    stats.p999 = find_pct(0.999);
    return stats;
}

size_t AtomicLatencyHistogram::value_to_bucket(uint64_t value) noexcept {
    if (value < LatencyHistogram::NUM_LINEAR) {
        return static_cast<size_t>(value);
    }
    const unsigned bit_width = static_cast<unsigned>(std::bit_width(value));
    const size_t octave = static_cast<size_t>(bit_width - 7);
    if (octave >= LatencyHistogram::MAX_OCTAVES) {
        return LatencyHistogram::NUM_BUCKETS - 1;
    }
    const unsigned shift = bit_width - 1 - static_cast<unsigned>(LatencyHistogram::SUB_BITS);
    const size_t sub = static_cast<size_t>((value >> shift) & (LatencyHistogram::SUB_COUNT - 1));
    return LatencyHistogram::NUM_LINEAR + octave * LatencyHistogram::SUB_COUNT + sub;
}

void AtomicLatencyHistogram::reset() noexcept {
    for (auto& b : buckets_) {
        b.store(0, std::memory_order_relaxed);
    }
    total_count_.store(0, std::memory_order_relaxed);
    max_value_.store(0, std::memory_order_relaxed);
}

void AtomicLatencyHistogram::record(uint64_t value) noexcept {
    const size_t b = value_to_bucket(value);
    buckets_[b].fetch_add(1, std::memory_order_relaxed);
    total_count_.fetch_add(1, std::memory_order_relaxed);
    uint64_t prev_max = max_value_.load(std::memory_order_relaxed);
    while (value > prev_max &&
           !max_value_.compare_exchange_weak(
               prev_max, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
}

void AtomicLatencyHistogram::copy_to(LatencyHistogram& out) const noexcept {
    out.reset();
    uint64_t sum = 0;
    for (size_t i = 0; i < LatencyHistogram::NUM_BUCKETS; ++i) {
        const uint64_t c = buckets_[i].load(std::memory_order_relaxed);
        out.buckets_[i] = c;
        sum += c;
    }
    out.total_count_ = sum;
    out.max_value_ = max_value_.load(std::memory_order_relaxed);
}

} // namespace triggerdaq
