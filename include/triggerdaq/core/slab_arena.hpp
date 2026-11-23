#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>

#include "triggerdaq/core/event_frame.hpp"

namespace triggerdaq {

using SlotHandle = uint32_t;
inline constexpr SlotHandle INVALID_SLOT = std::numeric_limits<uint32_t>::max();

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

    // Lock-free, ABA-safe O(1) slot acquisition. Returns INVALID_SLOT on exhaustion.
    [[nodiscard]] SlotHandle acquire() noexcept;

    // Lock-free, ABA-safe O(1) slot release.
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
        return allocated_count_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] uint64_t peak_occupancy() const noexcept {
        return peak_occupancy_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] uint64_t exhaustion_drops() const noexcept {
        return exhaustion_drops_.load(std::memory_order_relaxed);
    }

private:
    static constexpr uint64_t pack_head(uint32_t tag, uint32_t index) noexcept {
        return (static_cast<uint64_t>(tag) << 32) | static_cast<uint64_t>(index);
    }
    static constexpr uint32_t unpack_tag(uint64_t head) noexcept {
        return static_cast<uint32_t>(head >> 32);
    }
    static constexpr uint32_t unpack_index(uint64_t head) noexcept {
        return static_cast<uint32_t>(head & 0xFFFFFFFFULL);
    }

    void destroy() noexcept;

    RawEventFrame* frames_{nullptr};
    HltDecisionMetadata* metadata_{nullptr};
    std::unique_ptr<std::atomic<uint32_t>[]> next_indices_;
    uint32_t capacity_{0};

    alignas(64) std::atomic<uint64_t> free_head_{pack_head(0, INVALID_SLOT)};
    alignas(64) std::atomic<uint64_t> allocated_count_{0};
    alignas(64) std::atomic<uint64_t> peak_occupancy_{0};
    alignas(64) std::atomic<uint64_t> exhaustion_drops_{0};
};

} // namespace triggerdaq
