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
inline constexpr uint32_t DAQ_SCHEMA_VERSION = 1;
inline constexpr uint32_t DAQ_RECORD_PAYLOAD_SIZE =
    sizeof(RawEventFrame) + sizeof(HltDecisionMetadata); // 288 bytes

struct DaqFileHeader {
    char magic[8];
    uint32_t schema_version;
    uint32_t record_payload_size;
    uint64_t run_start_timestamp_ns;
    uint64_t reserved;
};

static_assert(sizeof(DaqFileHeader) == 32, "DaqFileHeader must be exactly 32 bytes");
static_assert(std::is_trivially_copyable_v<DaqFileHeader>, "DaqFileHeader must be trivially copyable");

struct DaqRecordHeader {
    uint32_t length;
    uint32_t crc32c;
};

static_assert(sizeof(DaqRecordHeader) == 8, "DaqRecordHeader must be exactly 8 bytes");

struct DaqDiskRecord {
    DaqRecordHeader header;
    RawEventFrame frame;
    HltDecisionMetadata meta;
};

// Note: RawEventFrame has alignas(64), so we serialize explicitly into packed 296-byte records.
inline constexpr size_t DAQ_SERIALIZED_RECORD_SIZE =
    sizeof(DaqRecordHeader) + sizeof(RawEventFrame) + sizeof(HltDecisionMetadata); // 296 bytes

// Software Castagnoli CRC32C (always available for verification & fallback)
[[nodiscard]] uint32_t crc32c_software(const void* data, size_t len, uint32_t init = 0) noexcept;

// Hardware-accelerated CRC32C when SSE4.2 is enabled, falling back to crc32c_software
[[nodiscard]] uint32_t crc32c_compute(const void* data, size_t len, uint32_t init = 0) noexcept;

struct alignas(64) StorageTelemetry {
    std::atomic<uint64_t> bytes_written{0};
    std::atomic<uint64_t> events_written{0};
    std::atomic<uint64_t> flush_count{0};

    void reset() noexcept {
        bytes_written.store(0, std::memory_order_relaxed);
        events_written.store(0, std::memory_order_relaxed);
        flush_count.store(0, std::memory_order_relaxed);
    }
};

struct DaqVerificationResult {
    bool header_valid{false};
    uint32_t schema_version{0};
    uint64_t run_start_timestamp_ns{0};
    uint64_t total_records{0};
    uint64_t valid_records{0};
    uint64_t corrupt_records{0};
    std::vector<uint8_t> packed_payloads; // contiguous valid [RawEventFrame(256) + HltDecisionMetadata(32)]
};

// Reads and CRC32C-verifies a .daq binary run file.
[[nodiscard]] DaqVerificationResult verify_daq_file(const std::string& path);

class EventWriter {
public:
    static constexpr size_t STAGING_BUFFER_BYTES = 256 * 1024; // 256 KB per double buffer

    EventWriter() = default;

    void initialize(const std::string& output_path);

    void run(std::atomic<bool>& stop_flag,
             SlabArena& arena,
             BoundedChannel<SlotHandle>& storage_queue,
             StorageTelemetry& telemetry) noexcept;

private:
    void flush_buffer(const uint8_t* data, size_t len, int fd, StorageTelemetry& telemetry) noexcept;

    std::string output_path_;
    std::unique_ptr<uint8_t[]> buffers_[2];
};

} // namespace triggerdaq
