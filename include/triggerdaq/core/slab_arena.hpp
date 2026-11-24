#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>

#include "triggerdaq/core/event_frame.hpp"

namespace triggerdaq {

using SlotHandle = uint32_t;
inline constexpr SlotHandle INVALID_SLOT = std::numeric_limits<uint32_t>::max();

// Pre-allocated 64-byte aligned zero-copy memory pool.
// Uses a 128-bit double-word atomic CAS (64-bit monotonic tag + 32-bit index)
// on 64-bit targets to eliminate 32-bit ABA tag wraparound without ring-stall false exhaustion.
class SlabArena {
public:
    SlabArena() = default;
    ~SlabArena();

    SlabArena(const SlabArena&) = delete;
    SlabArena& operator=(const SlabArena&) = delete;
    SlabArena(SlabArena&&) = delete;
    SlabArena& operator=(SlabArena&&) = delete;

    // Pre-allocates contiguous 64-byte aligned RawEventFrame slots and metadata slots.
    // Must be called once before hot-path threads start.
    void initialize(uint32_t capacity);

    // Lock-free, 64-bit-tagged ABA-safe O(1) slot acquisition. Returns INVALID_SLOT on exhaustion.
    [[nodiscard]] SlotHandle acquire() noexcept;

    // Lock-free, 64-bit-tagged ABA-safe O(1) slot release.
    void release(SlotHandle handle) noexcept;

    [[nodiscard]] RawEventFrame* get_frame(SlotHandle handle) noexcept {
        return &frames_[handle];
    }

    [[nodiscard]] const RawEventFrame* get_frame(SlotHandle handle) const noexcept {
        return &frames_[handle];
    }

    [[nodiscard]] HltDecisionMetadata* get_metadata(SlotHandle handle) noexcept {
        return &metadata_[handle];
    }

    [[nodiscard]] const HltDecisionMetadata* get_metadata(SlotHandle handle) const noexcept {
        return &metadata_[handle];
    }

    [[nodiscard]] uint32_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] uint64_t allocated_count() const noexcept {
        return allocated_count_.load(std::memory_order_acquire);
    }
    [[nodiscard]] uint64_t peak_occupancy() const noexcept {
        return peak_occupancy_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] uint64_t exhaustion_drops() const noexcept {
        return exhaustion_drops_.load(std::memory_order_relaxed);
    }

private:
#if defined(__SIZEOF_INT128__) && defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16)
    using HeadWord = unsigned __int128;
    static constexpr HeadWord pack_head(uint64_t tag, uint32_t index) noexcept {
        return (static_cast<HeadWord>(tag) << 64) | static_cast<HeadWord>(index);
    }
    static constexpr uint64_t unpack_tag(HeadWord head) noexcept {
        return static_cast<uint64_t>(head >> 64);
    }
    static constexpr uint32_t unpack_index(HeadWord head) noexcept {
        return static_cast<uint32_t>(head & 0xFFFFFFFFULL);
    }
#else
    using HeadWord = uint64_t;
    static constexpr HeadWord pack_head(uint64_t tag, uint32_t index) noexcept {
        return (static_cast<uint64_t>(static_cast<uint32_t>(tag)) << 32) | static_cast<uint64_t>(index);
    }
    static constexpr uint64_t unpack_tag(HeadWord head) noexcept {
        return static_cast<uint64_t>(head >> 32);
    }
    static constexpr uint32_t unpack_index(HeadWord head) noexcept {
        return static_cast<uint32_t>(head & 0xFFFFFFFFULL);
    }
#endif

    void destroy() noexcept;

    RawEventFrame* frames_{nullptr};
    HltDecisionMetadata* metadata_{nullptr};
    std::unique_ptr<std::atomic<uint32_t>[]> next_indices_;
    uint32_t capacity_{0};

    alignas(64) HeadWord free_head_{pack_head(0, INVALID_SLOT)};
    alignas(64) std::atomic<uint64_t> allocated_count_{0};
    alignas(64) std::atomic<uint64_t> peak_occupancy_{0};
    alignas(64) std::atomic<uint64_t> exhaustion_drops_{0};
};

} // namespace triggerdaq
