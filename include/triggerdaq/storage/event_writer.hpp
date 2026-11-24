#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "triggerdaq/core/bounded_channel.hpp"
#include "triggerdaq/core/event_frame.hpp"
#include "triggerdaq/core/slab_arena.hpp"

namespace triggerdaq {

inline constexpr char DAQ_MAGIC[8] = {'T', 'D', 'A', 'Q', '0', '0', '0', '1'};
inline constexpr uint32_t DAQ_SCHEMA_VERSION = 2;
inline constexpr uint32_t DAQ_RECORD_PAYLOAD_SIZE =
    sizeof(RawEventFrame) + sizeof(HltDecisionMetadata); // 288 bytes

inline constexpr uint64_t DAQ_FLAG_ZERO_SUPPRESSED = 1ULL << 0;

struct DaqFileHeader {
    char magic[8];
    uint32_t schema_version;
    uint32_t record_payload_size;
    uint64_t run_start_timestamp_ns;
    uint64_t flags; // Bitmask (bit 0 = DAQ_FLAG_ZERO_SUPPRESSED)
};

static_assert(sizeof(DaqFileHeader) == 32, "DaqFileHeader must be exactly 32 bytes");
static_assert(std::is_trivially_copyable_v<DaqFileHeader>, "DaqFileHeader must be trivially copyable");

struct DaqRecordHeader {
    uint32_t length;
    uint32_t crc32c;
};

static_assert(sizeof(DaqRecordHeader) == 8, "DaqRecordHeader must be exactly 8 bytes");

inline constexpr size_t DAQ_SERIALIZED_RECORD_SIZE =
    sizeof(DaqRecordHeader) + sizeof(RawEventFrame) + sizeof(HltDecisionMetadata); // 296 bytes

// Zero-Suppressed (ZS) record layout:
// [RawEventFrame prefix (24B)] [RawEventFrame suffix after calo_cells (104B)]
// [HltDecisionMetadata (32B)] [uint16_t num_nonzero_cells] [num_nonzero_cells * (uint8_t idx, uint16_t energy)]
inline constexpr size_t ZS_FRAME_PREFIX_SIZE = offsetof(RawEventFrame, calorimeter_cells); // 24B
inline constexpr size_t ZS_FRAME_SUFFIX_OFFSET = offsetof(RawEventFrame, muon_hits);       // 152B
inline constexpr size_t ZS_FRAME_SUFFIX_SIZE = sizeof(RawEventFrame) - ZS_FRAME_SUFFIX_OFFSET; // 104B
inline constexpr size_t ZS_BASE_PAYLOAD_SIZE =
    ZS_FRAME_PREFIX_SIZE + ZS_FRAME_SUFFIX_SIZE + sizeof(HltDecisionMetadata) + sizeof(uint16_t); // 162B
inline constexpr size_t ZS_CELL_ENTRY_SIZE = 3; // 1B cell_idx + 2B energy
inline constexpr size_t ZS_MAX_PAYLOAD_SIZE =
    ZS_BASE_PAYLOAD_SIZE + CALORIMETER_CELLS * ZS_CELL_ENTRY_SIZE; // 354B worst-case

// Software Castagnoli CRC32C (always available for verification & fallback)
[[nodiscard]] uint32_t crc32c_software(const void* data, size_t len, uint32_t init = 0) noexcept;

// Hardware-accelerated CRC32C when SSE4.2 is enabled, falling back to crc32c_software
[[nodiscard]] uint32_t crc32c_compute(const void* data, size_t len, uint32_t init = 0) noexcept;

struct alignas(64) StorageTelemetry {
    std::atomic<uint64_t> bytes_written{0};
    std::atomic<uint64_t> raw_bytes_uncompressed{0};
    std::atomic<uint64_t> events_written{0};
    std::atomic<uint64_t> flush_count{0};
    std::atomic<uint64_t> storage_write_errors{0};

    void reset() noexcept {
        bytes_written.store(0, std::memory_order_relaxed);
        raw_bytes_uncompressed.store(0, std::memory_order_relaxed);
        events_written.store(0, std::memory_order_relaxed);
        flush_count.store(0, std::memory_order_relaxed);
        storage_write_errors.store(0, std::memory_order_relaxed);
    }
};

struct DaqVerificationResult {
    bool header_valid{false};
    bool zero_suppressed{false};
    uint32_t schema_version{0};
    uint64_t run_start_timestamp_ns{0};
    uint64_t total_records{0};
    uint64_t valid_records{0};
    uint64_t corrupt_records{0};
    std::vector<uint8_t> packed_payloads; // contiguous valid [RawEventFrame(256) + HltDecisionMetadata(32)]
};

// Reads, CRC32C-verifies, and transparently decompresses (if ZS) a .daq binary run file.
[[nodiscard]] DaqVerificationResult verify_daq_file(const std::string& path);

class EventWriter {
public:
    static constexpr size_t STAGING_BUFFER_BYTES = 256 * 1024; // 256 KB per double buffer

    EventWriter() = default;

    void initialize(const std::string& output_path,
                    bool zero_suppression = false,
                    uint16_t zs_pedestal_threshold = 20);

    // Drains storage_queue until it is closed and empty, then flushes remaining buffers.
    void run(SlabArena& arena,
             BoundedChannel<SlotHandle>& storage_queue,
             StorageTelemetry& telemetry,
             bool pin_thread = false,
             uint32_t core_idx = 2) noexcept;

private:
    bool flush_buffer(const uint8_t* data,
                      size_t len,
                      int fd,
                      bool has_file_sink,
                      bool& io_failed,
                      StorageTelemetry& telemetry) noexcept;

    size_t encode_zs_payload(const RawEventFrame& frame,
                             const HltDecisionMetadata& meta,
                             uint8_t* dst) const noexcept;

    std::string output_path_;
    bool zero_suppression_{false};
    uint16_t zs_pedestal_threshold_{20};
    std::unique_ptr<uint8_t[]> buffers_[2];
};

} // namespace triggerdaq
