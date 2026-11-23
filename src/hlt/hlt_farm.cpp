#include "triggerdaq/hlt/hlt_farm.hpp"

#include <chrono>
#include <thread>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#include "triggerdaq/hlt/classifier.hpp"
#include "triggerdaq/hlt/reconstruction.hpp"

namespace triggerdaq {

namespace {

inline uint64_t steady_now_ns() noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

void try_pin_thread(uint32_t worker_id) noexcept {
#if defined(__linux__)
    const unsigned hw = std::thread::hardware_concurrency();
    if (hw > 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        // Offset by 2 cores to leave room for Frontend & L1 threads
        const unsigned core = (worker_id + 2U) % hw;
        CPU_SET(core, &cpuset);
        (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }
#else
    (void)worker_id;
#endif
}

} // namespace

HltFarm::~HltFarm() {
    stop();
}

void HltFarm::initialize(uint32_t num_workers, bool pin_threads) {
    stop();
    num_workers_ = num_workers > 0 ? num_workers : 1U;
    pin_threads_ = pin_threads;
    telemetry_ = std::make_unique<WorkerTelemetry[]>(num_workers_);
    for (uint32_t i = 0; i < num_workers_; ++i) {
        telemetry_[i].reset();
    }
}

void HltFarm::start(std::atomic<bool>& stop_flag,
                    SlabArena& arena,
                    BoundedChannel<SlotHandle>& hlt_queue,
                    BoundedChannel<SlotHandle>& storage_queue,
                    const AtomicMenuSlot& menu_slot) {
    workers_.clear();
    workers_.reserve(num_workers_);
    for (uint32_t i = 0; i < num_workers_; ++i) {
        telemetry_[i].reset();
        workers_.emplace_back([this, i, &stop_flag, &arena, &hlt_queue, &storage_queue, &menu_slot]() {
            worker_loop(i, stop_flag, arena, hlt_queue, storage_queue, menu_slot);
        });
    }
}

void HltFarm::stop() {
    for (auto& w : workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
    workers_.clear();
}

void HltFarm::worker_loop(uint32_t worker_id,
                          std::atomic<bool>& stop_flag,
                          SlabArena& arena,
                          BoundedChannel<SlotHandle>& hlt_queue,
                          BoundedChannel<SlotHandle>& storage_queue,
                          const AtomicMenuSlot& menu_slot) noexcept {
    if (pin_threads_) {
        try_pin_thread(worker_id);
    }

    WorkerTelemetry& wt = telemetry_[worker_id];

    for (;;) {
        SlotHandle slot = INVALID_SLOT;
        if (!hlt_queue.try_pop(slot)) {
            if (stop_flag.load(std::memory_order_acquire) && hlt_queue.empty()) {
                break;
            }
            std::this_thread::yield();
            continue;
        }

        const uint64_t t0 = steady_now_ns();
        RawEventFrame* frame = arena.get_frame(slot);
        HltDecisionMetadata* meta = arena.get_metadata(slot);
        const TriggerMenu& menu = menu_slot.load();

        const HltFeatures feats = reconstruct_event(*frame);
        const ClassifierOutput pred = Int8MlpClassifier::predict(feats);
        const bool accepted = pred.score >= menu.hlt_score_threshold;

        const uint64_t t1 = steady_now_ns();
        frame->hlt_score = pred.score;

        meta->hlt_timestamp_ns = t1;
        meta->cluster_centroid = feats.cluster_centroid;
        meta->cluster_width = feats.cluster_width;
        meta->isolation_ratio = feats.isolation_ratio;
        meta->dimuon_inv_mass = feats.dimuon_inv_mass;
        meta->hlt_score = pred.score;
        meta->predicted_class = pred.predicted_class;
        meta->accepted = accepted ? 1U : 0U;
        meta->worker_id = worker_id;

        wt.hlt_latency_ns.record(t1 >= t0 ? (t1 - t0) : 0);

        if (accepted) {
            wt.hlt_accepted.fetch_add(1, std::memory_order_relaxed);
            const auto tag = static_cast<TopologyTag>(frame->topology_tag);
            if (tag == TopologyTag::DiMuon || tag == TopologyTag::LocalizedJet) {
                wt.hlt_true_positives.fetch_add(1, std::memory_order_relaxed);
            } else {
                wt.hlt_false_positives.fetch_add(1, std::memory_order_relaxed);
            }

            if (!storage_queue.try_push(slot)) {
                arena.release(slot);
                wt.dead_time_events.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            wt.hlt_rejected.fetch_add(1, std::memory_order_relaxed);
            arena.release(slot);
        }
    }
}

} // namespace triggerdaq
