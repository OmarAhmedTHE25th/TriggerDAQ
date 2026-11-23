#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "triggerdaq/core/bounded_channel.hpp"
#include "triggerdaq/core/slab_arena.hpp"
#include "triggerdaq/core/telemetry.hpp"
#include "triggerdaq/l1/l1_trigger.hpp"

namespace triggerdaq {

class HltFarm {
public:
    HltFarm() = default;
    ~HltFarm();

    HltFarm(const HltFarm&) = delete;
    HltFarm& operator=(const HltFarm&) = delete;

    void initialize(uint32_t num_workers, bool pin_threads);

    void start(std::atomic<bool>& stop_flag,
               SlabArena& arena,
               BoundedChannel<SlotHandle>& hlt_queue,
               BoundedChannel<SlotHandle>& storage_queue,
               const AtomicMenuSlot& menu_slot);

    void stop();

    [[nodiscard]] uint32_t num_workers() const noexcept { return num_workers_; }
    [[nodiscard]] const WorkerTelemetry& worker_telemetry(uint32_t idx) const noexcept {
        return telemetry_[idx];
    }

private:
    void worker_loop(uint32_t worker_id,
                     std::atomic<bool>& stop_flag,
                     SlabArena& arena,
                     BoundedChannel<SlotHandle>& hlt_queue,
                     BoundedChannel<SlotHandle>& storage_queue,
                     const AtomicMenuSlot& menu_slot) noexcept;

    uint32_t num_workers_{0};
    bool pin_threads_{false};
    std::unique_ptr<WorkerTelemetry[]> telemetry_;
    std::vector<std::jthread> workers_;
};

} // namespace triggerdaq
