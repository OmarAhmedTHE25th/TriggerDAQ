#include "triggerdaq/core/slab_arena.hpp"

#include <cstring>
#include <new>
#include <stdexcept>

namespace triggerdaq {

namespace {

#if TRIGGERDAQ_USE_CAS128
inline unsigned __int128 atomic_load_128(volatile unsigned __int128* ptr) noexcept {
    return __sync_val_compare_and_swap(ptr, static_cast<unsigned __int128>(0),
                                       static_cast<unsigned __int128>(0));
}

inline void atomic_store_128(volatile unsigned __int128* ptr, unsigned __int128 val) noexcept {
    unsigned __int128 cur = atomic_load_128(ptr);
    for (;;) {
        const unsigned __int128 prev = __sync_val_compare_and_swap(ptr, cur, val);
        if (prev == cur) {
            return;
        }
        cur = prev;
    }
}

inline bool atomic_cas_128(volatile unsigned __int128* ptr,
                           unsigned __int128& expected,
                           unsigned __int128 desired) noexcept {
    const unsigned __int128 prev = __sync_val_compare_and_swap(ptr, expected, desired);
    if (prev == expected) {
        return true;
    }
    expected = prev;
    return false;
}
#endif

} // namespace

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

    const HeadWord initial = pack_head(0, 0);
#if TRIGGERDAQ_USE_CAS128
    atomic_store_128(&free_head_, initial);
#else
    free_head_.store(initial, std::memory_order_release);
#endif
    allocated_count_.store(0, std::memory_order_release);
    peak_occupancy_.store(0, std::memory_order_relaxed);
    exhaustion_drops_.store(0, std::memory_order_relaxed);
}

SlotHandle SlabArena::acquire() noexcept {
#if TRIGGERDAQ_USE_CAS128
    HeadWord head = atomic_load_128(&free_head_);
#else
    uint64_t head = free_head_.load(std::memory_order_acquire);
#endif
    for (;;) {
        const uint32_t idx = unpack_index(head);
        if (idx == INVALID_SLOT) {
            exhaustion_drops_.fetch_add(1, std::memory_order_relaxed);
            return INVALID_SLOT;
        }
        const uint32_t next = next_indices_[idx].load(std::memory_order_relaxed);
        const uint64_t next_tag = unpack_tag(head) + 1ULL;
        const HeadWord new_head = pack_head(next_tag, next);
#if TRIGGERDAQ_USE_CAS128
        const bool ok = atomic_cas_128(&free_head_, head, new_head);
#else
        const bool ok = free_head_.compare_exchange_weak(
            head, new_head, std::memory_order_acq_rel, std::memory_order_acquire);
#endif
        if (ok) {
            const uint64_t cur = allocated_count_.fetch_add(1, std::memory_order_acq_rel) + 1;
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

#if TRIGGERDAQ_USE_CAS128
    HeadWord head = atomic_load_128(&free_head_);
#else
    uint64_t head = free_head_.load(std::memory_order_acquire);
#endif
    for (;;) {
        const uint32_t old_idx = unpack_index(head);
        next_indices_[handle].store(old_idx, std::memory_order_relaxed);
        const uint64_t next_tag = unpack_tag(head) + 1ULL;
        const HeadWord new_head = pack_head(next_tag, handle);
#if TRIGGERDAQ_USE_CAS128
        const bool ok = atomic_cas_128(&free_head_, head, new_head);
#else
        const bool ok = free_head_.compare_exchange_weak(
            head, new_head, std::memory_order_release, std::memory_order_acquire);
#endif
        if (ok) {
            allocated_count_.fetch_sub(1, std::memory_order_release);
            return;
        }
    }
}

} // namespace triggerdaq
