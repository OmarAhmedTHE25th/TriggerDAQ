#include "triggerdaq/storage/event_writer.hpp"

#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <thread>

#if defined(__SSE4_2__)
#include <nmmintrin.h>
#endif

#if defined(__linux__) || defined(__unix__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace triggerdaq {

namespace {

constexpr std::array<uint32_t, 256> make_crc32c_table() noexcept {
    constexpr uint32_t POLY = 0x82F63B78U; // Reflected Castagnoli polynomial
    std::array<uint32_t, 256> table{};
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) {
            c = (c & 1U) ? (POLY ^ (c >> 1)) : (c >> 1);
        }
        table[i] = c;
    }
    return table;
}

constexpr auto CRC32C_TABLE = make_crc32c_table();

inline uint64_t system_now_ns() noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}

} // namespace

uint32_t crc32c_software(const void* data, size_t len, uint32_t init) noexcept {
    const auto* bytes = static_cast<const uint8_t*>(data);
    uint32_t crc = ~init;
    for (size_t i = 0; i < len; ++i) {
        crc = CRC32C_TABLE[(crc ^ bytes[i]) & 0xFFU] ^ (crc >> 8);
    }
    return ~crc;
}

uint32_t crc32c_compute(const void* data, size_t len, uint32_t init) noexcept {
#if defined(__SSE4_2__) && defined(__x86_64__)
    const auto* bytes = static_cast<const uint8_t*>(data);
    uint64_t crc = ~static_cast<uint64_t>(init);

    size_t i = 0;
    while (i + 8 <= len) {
        uint64_t chunk = 0;
        std::memcpy(&chunk, bytes + i, sizeof(chunk));
        crc = _mm_crc32_u64(crc, chunk);
        i += 8;
    }
    while (i < len) {
        crc = static_cast<uint64_t>(_mm_crc32_u8(static_cast<uint32_t>(crc), bytes[i]));
        ++i;
    }
    return static_cast<uint32_t>(~crc);
#else
    return crc32c_software(data, len, init);
#endif
}

DaqVerificationResult verify_daq_file(const std::string& path) {
    DaqVerificationResult res{};
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        return res;
    }

    const std::streamsize file_size = ifs.tellg();
    if (file_size < static_cast<std::streamsize>(sizeof(DaqFileHeader))) {
        return res;
    }
    ifs.seekg(0, std::ios::beg);

    DaqFileHeader header{};
    if (!ifs.read(reinterpret_cast<char*>(&header), sizeof(header))) {
        return res;
    }

    if (std::memcmp(header.magic, DAQ_MAGIC, sizeof(DAQ_MAGIC)) != 0 ||
        header.schema_version != DAQ_SCHEMA_VERSION ||
        header.record_payload_size != DAQ_RECORD_PAYLOAD_SIZE) {
        return res;
    }

    res.header_valid = true;
    res.schema_version = header.schema_version;
    res.run_start_timestamp_ns = header.run_start_timestamp_ns;

    const size_t body_bytes = static_cast<size_t>(file_size) - sizeof(DaqFileHeader);
    const size_t est_records = body_bytes / DAQ_SERIALIZED_RECORD_SIZE;
    res.packed_payloads.reserve(est_records * DAQ_RECORD_PAYLOAD_SIZE);

    std::array<uint8_t, DAQ_SERIALIZED_RECORD_SIZE> rec_buf{};
    while (ifs.read(reinterpret_cast<char*>(rec_buf.data()), DAQ_SERIALIZED_RECORD_SIZE)) {
        ++res.total_records;
        DaqRecordHeader rh{};
        std::memcpy(&rh, rec_buf.data(), sizeof(rh));
        const uint8_t* payload = rec_buf.data() + sizeof(rh);
        if (rh.length != DAQ_RECORD_PAYLOAD_SIZE) {
            ++res.corrupt_records;
            continue;
        }
        const uint32_t computed_crc = crc32c_compute(payload, DAQ_RECORD_PAYLOAD_SIZE);
        if (computed_crc != rh.crc32c) {
            ++res.corrupt_records;
            continue;
        }
        ++res.valid_records;
        res.packed_payloads.insert(
            res.packed_payloads.end(), payload, payload + DAQ_RECORD_PAYLOAD_SIZE);
    }

    // Count trailing truncated record as corrupt if any leftover bytes remain
    if (ifs.gcount() > 0) {
        ++res.total_records;
        ++res.corrupt_records;
    }

    return res;
}

void EventWriter::initialize(const std::string& output_path) {
    output_path_ = output_path;
    buffers_[0] = std::make_unique<uint8_t[]>(STAGING_BUFFER_BYTES);
    buffers_[1] = std::make_unique<uint8_t[]>(STAGING_BUFFER_BYTES);
}

void EventWriter::flush_buffer(const uint8_t* data,
                               size_t len,
                               int fd,
                               StorageTelemetry& telemetry) noexcept {
    if (len == 0) {
        return;
    }
#if defined(__linux__) || defined(__unix__)
    if (fd >= 0) {
        size_t written = 0;
        while (written < len) {
            const ssize_t n = ::write(fd, data + written, len - written);
            if (n <= 0) {
                break;
            }
            written += static_cast<size_t>(n);
        }
    }
#else
    (void)data;
    (void)fd;
#endif
    telemetry.bytes_written.fetch_add(len, std::memory_order_relaxed);
    telemetry.flush_count.fetch_add(1, std::memory_order_relaxed);
}

void EventWriter::run(std::atomic<bool>& stop_flag,
                      SlabArena& arena,
                      BoundedChannel<SlotHandle>& storage_queue,
                      StorageTelemetry& telemetry) noexcept {
    int fd = -1;
#if defined(__linux__) || defined(__unix__)
    if (!output_path_.empty()) {
        fd = ::open(output_path_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    }
#endif

    // Write 32-byte file header
    DaqFileHeader file_header{};
    std::memcpy(file_header.magic, DAQ_MAGIC, sizeof(DAQ_MAGIC));
    file_header.schema_version = DAQ_SCHEMA_VERSION;
    file_header.record_payload_size = DAQ_RECORD_PAYLOAD_SIZE;
    file_header.run_start_timestamp_ns = system_now_ns();
    file_header.reserved = 0;
    flush_buffer(reinterpret_cast<const uint8_t*>(&file_header), sizeof(file_header), fd, telemetry);

    int active_buf = 0;
    size_t active_offset = 0;

    for (;;) {
        SlotHandle slot = INVALID_SLOT;
        if (!storage_queue.try_pop(slot)) {
            if (stop_flag.load(std::memory_order_acquire) && storage_queue.empty()) {
                break;
            }
            std::this_thread::yield();
            continue;
        }

        if (active_offset + DAQ_SERIALIZED_RECORD_SIZE > STAGING_BUFFER_BYTES) {
            const int flush_idx = active_buf;
            const size_t flush_len = active_offset;
            active_buf = 1 - active_buf;
            active_offset = 0;
            flush_buffer(buffers_[flush_idx].get(), flush_len, fd, telemetry);
        }

        uint8_t* dst = buffers_[active_buf].get() + active_offset;
        uint8_t* payload_dst = dst + sizeof(DaqRecordHeader);

        const RawEventFrame* frame = arena.get_frame(slot);
        const HltDecisionMetadata* meta = arena.get_metadata(slot);

        std::memcpy(payload_dst, frame, sizeof(RawEventFrame));
        std::memcpy(payload_dst + sizeof(RawEventFrame), meta, sizeof(HltDecisionMetadata));

        // Release slot back to SlabArena immediately after copy into staging buffer
        arena.release(slot);

        DaqRecordHeader rh{};
        rh.length = DAQ_RECORD_PAYLOAD_SIZE;
        rh.crc32c = crc32c_compute(payload_dst, DAQ_RECORD_PAYLOAD_SIZE);
        std::memcpy(dst, &rh, sizeof(rh));

        active_offset += DAQ_SERIALIZED_RECORD_SIZE;
        telemetry.events_written.fetch_add(1, std::memory_order_relaxed);
    }

    if (active_offset > 0) {
        flush_buffer(buffers_[active_buf].get(), active_offset, fd, telemetry);
    }

#if defined(__linux__) || defined(__unix__)
    if (fd >= 0) {
        ::close(fd);
    }
#endif
}

} // namespace triggerdaq
