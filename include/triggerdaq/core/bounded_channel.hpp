#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace triggerdaq {

// Lock-free cache-padded bounded MPMC/SPSC ring buffer (Vyukov algorithm).
// Zero heap allocations after initialize().
template <typename T>
class BoundedChannel {
    static_assert(std::is_trivially_copyable_v<T>, "BoundedChannel requires trivially copyable element type");

public:
    BoundedChannel() = default;

    explicit BoundedChannel(size_t requested_capacity) {
        initialize(requested_capacity);
    }

    BoundedChannel(const BoundedChannel&) = delete;
    BoundedChannel& operator=(const BoundedChannel&) = delete;

    void initialize(size_t requested_capacity) {
        if (requested_capacity < 2) {
            requested_capacity = 2;
        }
        size_t cap = 1;
        while (cap < requested_capacity) {
            cap <<= 1;
        }
        capacity_ = cap;
        mask_ = cap - 1;
        buffer_ = std::make_unique<Cell[]>(capacity_);
        for (size_t i = 0; i < capacity_; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
        enqueue_pos_.store(0, std::memory_order_relaxed);
        dequeue_pos_.store(0, std::memory_order_relaxed);
    }

    [[nodiscard]] bool try_push(const T& item) noexcept {
        Cell* cell = nullptr;
        size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
        for (;;) {
            cell = &buffer_[pos & mask_];
            const size_t seq = cell->sequence.load(std::memory_order_acquire);
            const intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
            if (diff == 0) {
                if (enqueue_pos_.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false; // Queue is full
            } else {
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }
        cell->data = item;
        cell->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] bool try_pop(T& item) noexcept {
        Cell* cell = nullptr;
        size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        for (;;) {
            cell = &buffer_[pos & mask_];
            const size_t seq = cell->sequence.load(std::memory_order_acquire);
            const intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
            if (diff == 0) {
                if (dequeue_pos_.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false; // Queue is empty
            } else {
                pos = dequeue_pos_.load(std::memory_order_relaxed);
            }
        }
        item = cell->data;
        cell->sequence.store(pos + mask_ + 1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] size_t size() const noexcept {
        const size_t enq = enqueue_pos_.load(std::memory_order_relaxed);
        const size_t deq = dequeue_pos_.load(std::memory_order_relaxed);
        if (enq <= deq) {
            return 0;
        }
        const size_t diff = enq - deq;
        return diff > capacity_ ? capacity_ : diff;
    }

    [[nodiscard]] size_t capacity() const noexcept {
        return capacity_;
    }

    [[nodiscard]] double fill_ratio() const noexcept {
        if (capacity_ == 0) {
            return 0.0;
        }
        return static_cast<double>(size()) / static_cast<double>(capacity_);
    }

    [[nodiscard]] bool empty() const noexcept {
        return size() == 0;
    }

private:
    struct alignas(64) Cell {
        std::atomic<size_t> sequence{0};
        T data{};
    };

    std::unique_ptr<Cell[]> buffer_;
    size_t capacity_{0};
    size_t mask_{0};

    alignas(64) std::atomic<size_t> enqueue_pos_{0};
    alignas(64) std::atomic<size_t> dequeue_pos_{0};
};

} // namespace triggerdaq
