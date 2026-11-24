#include "triggerdaq/core/telemetry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

namespace triggerdaq {

size_t LatencyHistogram::value_to_bucket(uint64_t value) noexcept {
    if (value < NUM_LINEAR) {
        return static_cast<size_t>(value);
    }
    const unsigned bit_width = static_cast<unsigned>(std::bit_width(value));
    const size_t octave = static_cast<size_t>(bit_width - 7);
    if (octave >= MAX_OCTAVES) {
        return NUM_BUCKETS - 1;
    }
    const unsigned shift = bit_width - 1 - static_cast<unsigned>(SUB_BITS);
    const size_t sub = static_cast<size_t>((value >> shift) & (SUB_COUNT - 1));
    return NUM_LINEAR + octave * SUB_COUNT + sub;
}

double LatencyHistogram::bucket_lower_bound(size_t bucket) noexcept {
    if (bucket < NUM_LINEAR) {
        return static_cast<double>(bucket);
    }
    const size_t rel = bucket - NUM_LINEAR;
    const size_t octave = rel / SUB_COUNT;
    const size_t sub = rel % SUB_COUNT;
    const uint64_t base = uint64_t{64} << octave;
    const uint64_t step = base >> SUB_BITS;
    return static_cast<double>(base + sub * step);
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
    min_value_ = std::numeric_limits<uint64_t>::max();
    max_value_ = 0;
}

void LatencyHistogram::record(uint64_t value) noexcept {
    const size_t b = value_to_bucket(value);
    ++buckets_[b];
    ++total_count_;
    if (value < min_value_) {
        min_value_ = value;
    }
    if (value > max_value_) {
        max_value_ = value;
    }
}

void LatencyHistogram::merge_from(const LatencyHistogram& other) noexcept {
    if (other.total_count_ == 0) {
        return;
    }
    for (size_t i = 0; i < NUM_BUCKETS; ++i) {
        buckets_[i] += other.buckets_[i];
    }
    total_count_ += other.total_count_;
    if (other.min_value_ < min_value_) {
        min_value_ = other.min_value_;
    }
    if (other.max_value_ > max_value_) {
        max_value_ = other.max_value_;
    }
}

LatencyStats LatencyHistogram::compute_stats() const noexcept {
    LatencyStats stats{};
    if (total_count_ == 0) {
        return stats;
    }
    stats.min = static_cast<double>(min_value_);
    stats.max = static_cast<double>(max_value_);

    const auto find_pct = [&](double pct) -> double {
        const double target = pct * static_cast<double>(total_count_);
        uint64_t acc = 0;
        for (size_t i = 0; i < NUM_BUCKETS; ++i) {
            const uint64_t count = buckets_[i];
            if (count == 0) {
                continue;
            }
            const uint64_t prev_acc = acc;
            acc += count;
            if (static_cast<double>(acc) >= target) {
                const double lo = bucket_lower_bound(i);
                const double hi = bucket_upper_bound(i);
                const double frac = std::clamp(
                    (target - static_cast<double>(prev_acc)) / static_cast<double>(count),
                    0.0, 1.0);
                const double interp = lo + frac * (hi - lo);
                return std::clamp(interp, stats.min, stats.max);
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
    return LatencyHistogram::value_to_bucket(value);
}

void AtomicLatencyHistogram::reset() noexcept {
    const uint64_t s = seq_.load(std::memory_order_relaxed);
    seq_.store(s + 1, std::memory_order_release); // odd: write in progress
    for (auto& b : buckets_) {
        b.store(0, std::memory_order_relaxed);
    }
    total_count_.store(0, std::memory_order_relaxed);
    min_value_.store(std::numeric_limits<uint64_t>::max(), std::memory_order_relaxed);
    max_value_.store(0, std::memory_order_relaxed);
    seq_.store(s + 2, std::memory_order_release); // even: stable
}

void AtomicLatencyHistogram::record(uint64_t value) noexcept {
    const size_t b = value_to_bucket(value);
    // Seqlock write protocol (single writer per histogram)
    const uint64_t s = seq_.load(std::memory_order_relaxed);
    seq_.store(s + 1, std::memory_order_release);

    buckets_[b].store(buckets_[b].load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    total_count_.store(total_count_.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);

    const uint64_t cur_min = min_value_.load(std::memory_order_relaxed);
    if (value < cur_min) {
        min_value_.store(value, std::memory_order_relaxed);
    }
    const uint64_t cur_max = max_value_.load(std::memory_order_relaxed);
    if (value > cur_max) {
        max_value_.store(value, std::memory_order_relaxed);
    }

    seq_.store(s + 2, std::memory_order_release);
}

void AtomicLatencyHistogram::copy_to(LatencyHistogram& out) const noexcept {
    constexpr int MAX_RETRIES = 16;
    for (int attempt = 0; attempt < MAX_RETRIES; ++attempt) {
        const uint64_t s1 = seq_.load(std::memory_order_acquire);
        if ((s1 & 1ULL) != 0ULL) {
            std::this_thread::yield();
            continue;
        }

        uint64_t sum = 0;
        for (size_t i = 0; i < LatencyHistogram::NUM_BUCKETS; ++i) {
            const uint64_t c = buckets_[i].load(std::memory_order_relaxed);
            out.buckets_[i] = c;
            sum += c;
        }
        const uint64_t mn = min_value_.load(std::memory_order_relaxed);
        const uint64_t mx = max_value_.load(std::memory_order_relaxed);

        const uint64_t s2 = seq_.load(std::memory_order_acquire);
        if (s1 == s2) {
            out.total_count_ = sum;
            out.min_value_ = sum > 0 ? mn : std::numeric_limits<uint64_t>::max();
            out.max_value_ = sum > 0 ? mx : 0;
            return;
        }
    }

    // Bounded fallback under extreme writer saturation: compute self-consistent sum from copied buckets
    uint64_t sum = 0;
    for (size_t i = 0; i < LatencyHistogram::NUM_BUCKETS; ++i) {
        const uint64_t c = buckets_[i].load(std::memory_order_relaxed);
        out.buckets_[i] = c;
        sum += c;
    }
    out.total_count_ = sum;
    out.min_value_ = sum > 0 ? min_value_.load(std::memory_order_relaxed)
                             : std::numeric_limits<uint64_t>::max();
    out.max_value_ = sum > 0 ? max_value_.load(std::memory_order_relaxed) : 0;
}

} // namespace triggerdaq
