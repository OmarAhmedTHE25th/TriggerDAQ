#include "triggerdaq/core/slab_arena.hpp"

#include <cstring>
#include <new>
#include <stdexcept>

namespace triggerdaq {

SlabArena::~SlabArena() {
    destroy();
}

void SlabArena::destroy() noexcept {
    if (frames_ != nullptr) {
        ::operator delete[](frames_, std::align_val_t{alignof(RawEventFrame)});
        frames_ = nullptr;
    }
    if (metadata_ != nullptr) {
        ::operator delete[](metadata_, std::align_val_t{alignof(HltDecisionMetadata)});
        metadata_ = nullptr;
    }
    next_indices_.reset();
    capacity_ = 0;
}

void SlabArena::initialize(uint32_t capacity) {
    if (capacity == 0 || capacity == INVALID_SLOT) {
        throw std::invalid_argument("SlabArena capacity must be in [1, UINT32_MAX - 1]");
    }
    destroy();

    capacity_ = capacity;
    frames_ = static_cast<RawEventFrame*>(
        ::operator new[](sizeof(RawEventFrame) * capacity_, std::align_val_t{alignof(RawEventFrame)}));
    std::memset(frames_, 0, sizeof(RawEventFrame) * capacity_);

    metadata_ = static_cast<HltDecisionMetadata*>(
        ::operator new[](sizeof(HltDecisionMetadata) * capacity_,
                         std::align_val_t{alignof(HltDecisionMetadata)}));
    std::memset(metadata_, 0, sizeof(HltDecisionMetadata) * capacity_);

    next_indices_ = std::make_unique<std::atomic<uint32_t>[]>(capacity_);
    for (uint32_t i = 0; i + 1 < capacity_; ++i) {
        next_indices_[i].store(i + 1, std::memory_order_relaxed);
    }
    next_indices_[capacity_ - 1].store(INVALID_SLOT, std::memory_order_relaxed);

    free_head_.store(pack_head(0, 0), std::memory_order_release);
    allocated_count_.store(0, std::memory_order_relaxed);
    peak_occupancy_.store(0, std::memory_order_relaxed);
    exhaustion_drops_.store(0, std::memory_order_relaxed);
}

SlotHandle SlabArena::acquire() noexcept {
    uint64_t head = free_head_.load(std::memory_order_acquire);
    for (;;) {
        const uint32_t idx = unpack_index(head);
        if (idx == INVALID_SLOT) {
            exhaustion_drops_.fetch_add(1, std::memory_order_relaxed);
            return INVALID_SLOT;
        }
        const uint32_t next = next_indices_[idx].load(std::memory_order_relaxed);
        const uint32_t next_tag = unpack_tag(head) + 1U;
        const uint64_t new_head = pack_head(next_tag, next);
        if (free_head_.compare_exchange_weak(
                head, new_head, std::memory_order_acq_rel, std::memory_order_acquire)) {
            const uint64_t cur = allocated_count_.fetch_add(1, std::memory_order_relaxed) + 1;
            uint64_t peak = peak_occupancy_.load(std::memory_order_relaxed);
            while (cur > peak &&
                   !peak_occupancy_.compare_exchange_weak(
                       peak, cur, std::memory_order_relaxed, std::memory_order_relaxed)) {
            }
            return idx;
        }
    }
}

void SlabArena::release(SlotHandle handle) noexcept {
    if (handle >= capacity_) {
        return;
    }
    uint64_t head = free_head_.load(std::memory_order_acquire);
    for (;;) {
        const uint32_t old_idx = unpack_index(head);
        next_indices_[handle].store(old_idx, std::memory_order_relaxed);
        const uint32_t next_tag = unpack_tag(head) + 1U;
        const uint64_t new_head = pack_head(next_tag, handle);
        if (free_head_.compare_exchange_weak(
                head, new_head, std::memory_order_release, std::memory_order_acquire)) {
            allocated_count_.fetch_sub(1, std::memory_order_relaxed);
            return;
        }
    }
}

} // namespace triggerdaq
