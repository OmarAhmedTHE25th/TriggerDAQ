#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "triggerdaq/engine.hpp"
#include "triggerdaq/l1/simd_kernels.hpp"
#include "triggerdaq/storage/event_writer.hpp"

namespace nb = nanobind;
using namespace triggerdaq;

NB_MODULE(_triggerdaq_ext, m) {
    m.doc() = "TriggerDAQ C++20 high-throughput two-stage detector trigger & DAQ extension";

    m.def("has_avx2_kernel", &has_avx2_kernel, "Returns True if compiled with AVX2 SIMD L1 kernels");

    m.def("get_record_layout", []() {
        nb::dict layout;
        layout["frame_size"] = sizeof(RawEventFrame);
        layout["meta_size"] = sizeof(HltDecisionMetadata);
        layout["record_payload_size"] = static_cast<size_t>(DAQ_RECORD_PAYLOAD_SIZE);

        nb::dict offsets;
        offsets["event_id"] = offsetof(RawEventFrame, event_id);
        offsets["bunch_crossing_id"] = offsetof(RawEventFrame, bunch_crossing_id);
        offsets["ingest_timestamp_ns"] = offsetof(RawEventFrame, ingest_timestamp_ns);
        offsets["calorimeter_cells"] = offsetof(RawEventFrame, calorimeter_cells);
        offsets["muon_hits"] = offsetof(RawEventFrame, muon_hits);
        offsets["l1_trigger_bits"] = offsetof(RawEventFrame, l1_trigger_bits);
        offsets["hlt_score"] = offsetof(RawEventFrame, hlt_score);
        offsets["topology_tag"] = offsetof(RawEventFrame, topology_tag);
        offsets["reserved"] = offsetof(RawEventFrame, reserved);
        offsets["explicit_pad"] = offsetof(RawEventFrame, explicit_pad);

        constexpr size_t M = sizeof(RawEventFrame);
        offsets["hlt_timestamp_ns"] = M + offsetof(HltDecisionMetadata, hlt_timestamp_ns);
        offsets["cluster_centroid"] = M + offsetof(HltDecisionMetadata, cluster_centroid);
        offsets["cluster_width"] = M + offsetof(HltDecisionMetadata, cluster_width);
        offsets["isolation_ratio"] = M + offsetof(HltDecisionMetadata, isolation_ratio);
        offsets["dimuon_inv_mass"] = M + offsetof(HltDecisionMetadata, dimuon_inv_mass);
        offsets["meta_hlt_score"] = M + offsetof(HltDecisionMetadata, hlt_score);
        offsets["predicted_class"] = M + offsetof(HltDecisionMetadata, predicted_class);
        offsets["accepted"] = M + offsetof(HltDecisionMetadata, accepted);
        offsets["worker_id"] = M + offsetof(HltDecisionMetadata, worker_id);

        layout["offsets"] = offsets;
        return layout;
    }, "Return exact C++ sizeof/offsetof binary layout for RawEventFrame + HltDecisionMetadata");

    nb::enum_<EmulatorMode>(m, "EmulatorMode")
        .value("Sustained", EmulatorMode::Sustained)
        .value("PoissonBurst", EmulatorMode::PoissonBurst)
        .value("MaxThroughputStress", EmulatorMode::MaxThroughputStress);

    nb::enum_<RulePriority>(m, "RulePriority")
        .value("Low", RulePriority::Low)
        .value("Normal", RulePriority::Normal)
        .value("High", RulePriority::High)
        .value("Critical", RulePriority::Critical);

    nb::enum_<RuleCondition>(m, "RuleCondition")
        .value("SingleMuon", RuleCondition::SingleMuon)
        .value("DiMuon", RuleCondition::DiMuon)
        .value("JetCluster", RuleCondition::JetCluster)
        .value("TotalET", RuleCondition::TotalET);

    nb::enum_<TopologyTag>(m, "TopologyTag")
        .value("Background", TopologyTag::Background)
        .value("DiMuon", TopologyTag::DiMuon)
        .value("LocalizedJet", TopologyTag::LocalizedJet)
        .value("PileupNoise", TopologyTag::PileupNoise);

    nb::class_<TriggerRule>(m, "TriggerRule")
        .def(nb::init<>())
        .def_prop_rw(
            "name",
            [](const TriggerRule& r) { return std::string(r.get_name()); },
            [](TriggerRule& r, std::string_view n) { r.set_name(n); })
        .def_rw("condition", &TriggerRule::condition)
        .def_rw("priority", &TriggerRule::priority)
        .def_rw("threshold", &TriggerRule::threshold)
        .def_rw("secondary_threshold", &TriggerRule::secondary_threshold)
        .def_rw("base_prescale", &TriggerRule::base_prescale)
        .def_rw("max_prescale", &TriggerRule::max_prescale)
        .def_rw("enabled", &TriggerRule::enabled);

    nb::class_<TriggerMenu>(m, "TriggerMenu")
        .def(nb::init<>())
        .def_ro("num_rules", &TriggerMenu::num_rules)
        .def_rw("muon_pt_threshold", &TriggerMenu::muon_pt_threshold)
        .def_rw("hlt_score_threshold", &TriggerMenu::hlt_score_threshold)
        .def_rw("version", &TriggerMenu::version)
        .def("add_rule", &TriggerMenu::add_rule)
        .def("get_rule", [](const TriggerMenu& menu, uint32_t idx) -> TriggerRule {
            if (idx >= menu.num_rules) {
                throw std::out_of_range("TriggerMenu rule index out of range");
            }
            return menu.rules[idx];
        })
        .def_static("default_nominal_menu", &TriggerMenu::default_nominal_menu);

    nb::class_<EngineConfig>(m, "EngineConfig")
        .def(nb::init<>())
        .def_rw("arena_slots", &EngineConfig::arena_slots)
        .def_rw("l1_queue_capacity", &EngineConfig::l1_queue_capacity)
        .def_rw("hlt_queue_capacity", &EngineConfig::hlt_queue_capacity)
        .def_rw("storage_queue_capacity", &EngineConfig::storage_queue_capacity)
        .def_rw("hlt_workers", &EngineConfig::hlt_workers)
        .def_rw("target_rate_khz", &EngineConfig::target_rate_khz)
        .def_rw("l1_budget_ns", &EngineConfig::l1_budget_ns)
        .def_rw("mode", &EngineConfig::mode)
        .def_rw("pin_threads", &EngineConfig::pin_threads)
        .def_rw("seed", &EngineConfig::seed)
        .def_rw("output_path", &EngineConfig::output_path);

    nb::class_<LatencyStats>(m, "LatencyStats")
        .def(nb::init<>())
        .def_ro("min", &LatencyStats::min)
        .def_ro("p50", &LatencyStats::p50)
        .def_ro("p95", &LatencyStats::p95)
        .def_ro("p99", &LatencyStats::p99)
        .def_ro("p999", &LatencyStats::p999)
        .def_ro("max", &LatencyStats::max)
        .def("to_dict", [](const LatencyStats& s) {
            nb::dict d;
            d["min"] = s.min;
            d["p50"] = s.p50;
            d["p95"] = s.p95;
            d["p99"] = s.p99;
            d["p999"] = s.p999;
            d["max"] = s.max;
            return d;
        });

    nb::class_<TelemetrySnapshot>(m, "TelemetrySnapshot")
        .def(nb::init<>())
        .def_ro("events_generated", &TelemetrySnapshot::events_generated)
        .def_ro("l1_accepted", &TelemetrySnapshot::l1_accepted)
        .def_ro("l1_rejected", &TelemetrySnapshot::l1_rejected)
        .def_ro("hlt_accepted", &TelemetrySnapshot::hlt_accepted)
        .def_ro("hlt_rejected", &TelemetrySnapshot::hlt_rejected)
        .def_ro("dead_time_events", &TelemetrySnapshot::dead_time_events)
        .def_ro("dead_time_ns", &TelemetrySnapshot::dead_time_ns)
        .def_ro("l1_budget_violations", &TelemetrySnapshot::l1_budget_violations)
        .def_ro("bytes_written", &TelemetrySnapshot::bytes_written)
        .def_ro("events_written", &TelemetrySnapshot::events_written)
        .def_ro("storage_write_errors", &TelemetrySnapshot::storage_write_errors)
        .def_ro("arena_allocated", &TelemetrySnapshot::arena_allocated)
        .def_ro("arena_peak", &TelemetrySnapshot::arena_peak)
        .def_ro("arena_capacity", &TelemetrySnapshot::arena_capacity)
        .def_ro("l1_queue_fill", &TelemetrySnapshot::l1_queue_fill)
        .def_ro("hlt_queue_fill", &TelemetrySnapshot::hlt_queue_fill)
        .def_ro("storage_queue_fill", &TelemetrySnapshot::storage_queue_fill)
        .def_ro("l1_latency_ns", &TelemetrySnapshot::l1_latency_ns)
        .def_ro("hlt_latency_us", &TelemetrySnapshot::hlt_latency_us)
        .def_ro("num_rules", &TelemetrySnapshot::num_rules)
        .def_ro("prescale_factors", &TelemetrySnapshot::prescale_factors)
        .def_ro("rule_accept_counts", &TelemetrySnapshot::rule_accept_counts)
        .def_ro("signal_generated", &TelemetrySnapshot::signal_generated)
        .def_ro("hlt_true_positives", &TelemetrySnapshot::hlt_true_positives)
        .def_ro("hlt_false_positives", &TelemetrySnapshot::hlt_false_positives)
        .def_ro("elapsed_seconds", &TelemetrySnapshot::elapsed_seconds)
        .def_ro("input_rate_hz", &TelemetrySnapshot::input_rate_hz)
        .def_ro("l1_rate_hz", &TelemetrySnapshot::l1_rate_hz)
        .def_ro("hlt_rate_hz", &TelemetrySnapshot::hlt_rate_hz)
        .def_ro("dead_time_fraction", &TelemetrySnapshot::dead_time_fraction)
        .def_ro("signal_purity", &TelemetrySnapshot::signal_purity)
        .def_ro("signal_efficiency", &TelemetrySnapshot::signal_efficiency)
        .def("to_dict", [](const TelemetrySnapshot& s) {
            nb::dict d;
            d["events_generated"] = s.events_generated;
            d["l1_accepted"] = s.l1_accepted;
            d["l1_rejected"] = s.l1_rejected;
            d["hlt_accepted"] = s.hlt_accepted;
            d["hlt_rejected"] = s.hlt_rejected;
            d["dead_time_events"] = s.dead_time_events;
            d["dead_time_ns"] = s.dead_time_ns;
            d["l1_budget_violations"] = s.l1_budget_violations;
            d["bytes_written"] = s.bytes_written;
            d["events_written"] = s.events_written;
            d["storage_write_errors"] = s.storage_write_errors;
            d["arena_allocated"] = s.arena_allocated;
            d["arena_peak"] = s.arena_peak;
            d["arena_capacity"] = s.arena_capacity;
            d["l1_queue_fill"] = s.l1_queue_fill;
            d["hlt_queue_fill"] = s.hlt_queue_fill;
            d["storage_queue_fill"] = s.storage_queue_fill;
            nb::dict l1l;
            l1l["min"] = s.l1_latency_ns.min;
            l1l["p50"] = s.l1_latency_ns.p50;
            l1l["p95"] = s.l1_latency_ns.p95;
            l1l["p99"] = s.l1_latency_ns.p99;
            l1l["p999"] = s.l1_latency_ns.p999;
            l1l["max"] = s.l1_latency_ns.max;
            d["l1_latency_ns"] = l1l;
            nb::dict hltl;
            hltl["min"] = s.hlt_latency_us.min;
            hltl["p50"] = s.hlt_latency_us.p50;
            hltl["p95"] = s.hlt_latency_us.p95;
            hltl["p99"] = s.hlt_latency_us.p99;
            hltl["p999"] = s.hlt_latency_us.p999;
            hltl["max"] = s.hlt_latency_us.max;
            d["hlt_latency_us"] = hltl;
            d["num_rules"] = s.num_rules;
            nb::list ps;
            nb::list rc;
            for (uint32_t i = 0; i < s.num_rules && i < TriggerMenu::MAX_RULES; ++i) {
                ps.append(s.prescale_factors[i]);
                rc.append(s.rule_accept_counts[i]);
            }
            d["prescale_factors"] = ps;
            d["rule_accept_counts"] = rc;
            d["signal_generated"] = s.signal_generated;
            d["hlt_true_positives"] = s.hlt_true_positives;
            d["hlt_false_positives"] = s.hlt_false_positives;
            d["elapsed_seconds"] = s.elapsed_seconds;
            d["input_rate_hz"] = s.input_rate_hz;
            d["l1_rate_hz"] = s.l1_rate_hz;
            d["hlt_rate_hz"] = s.hlt_rate_hz;
            d["dead_time_fraction"] = s.dead_time_fraction;
            d["signal_purity"] = s.signal_purity;
            d["signal_efficiency"] = s.signal_efficiency;
            return d;
        });

    nb::class_<TriggerEngine>(m, "TriggerEngine")
        .def(nb::init<>())
        .def("initialize", &TriggerEngine::initialize, nb::arg("config"), nb::arg("menu"))
        .def("start", &TriggerEngine::start, nb::call_guard<nb::gil_scoped_release>())
        .def("stop", &TriggerEngine::stop, nb::call_guard<nb::gil_scoped_release>())
        .def("run_for_seconds", &TriggerEngine::run_for_seconds,
             nb::arg("seconds"), nb::call_guard<nb::gil_scoped_release>())
        .def("update_menu", &TriggerEngine::update_menu, nb::arg("menu"))
        .def("snapshot", &TriggerEngine::snapshot)
        .def("is_running", &TriggerEngine::is_running);

    m.def("verify_and_map_daq", [](const std::string& path) {
        DaqVerificationResult res;
        {
            nb::gil_scoped_release release;
            res = verify_daq_file(path);
        }

        auto* buf = new std::vector<uint8_t>(std::move(res.packed_payloads));
        nb::capsule owner(buf, [](void* p) noexcept {
            delete static_cast<std::vector<uint8_t>*>(p);
        });

        const size_t shape[2] = {
            static_cast<size_t>(res.valid_records),
            static_cast<size_t>(DAQ_RECORD_PAYLOAD_SIZE)
        };
        nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>> view(
            buf->data(), 2, shape, owner);

        nb::dict out;
        out["header_valid"] = res.header_valid;
        out["schema_version"] = res.schema_version;
        out["run_start_timestamp_ns"] = res.run_start_timestamp_ns;
        out["total_records"] = res.total_records;
        out["valid_records"] = res.valid_records;
        out["corrupt_records"] = res.corrupt_records;
        out["raw_buffer"] = view;
        return out;
    }, nb::arg("path"),
       "Verify CRC32C checksums of a .daq file and return a zero-copy NumPy byte buffer of valid records");
}
