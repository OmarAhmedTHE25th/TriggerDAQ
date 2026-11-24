#include "triggerdaq/storage/event_writer.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <thread>

#include "triggerdaq/core/thread_utils.hpp"

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
    constexpr uint32_t POLY = 0x82F63B78U;
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

bool decode_zs_payload(const uint8_t* src,
                       size_t len,
                       uint8_t* out_288b) noexcept {
    if (len < ZS_BASE_PAYLOAD_SIZE) {
        return false;
    }
    uint16_t num_cells = 0;
    const size_t count_off = ZS_FRAME_PREFIX_SIZE + ZS_FRAME_SUFFIX_SIZE + sizeof(HltDecisionMetadata);
    std::memcpy(&num_cells, src + count_off, sizeof(num_cells));
    if (num_cells > CALORIMETER_CELLS) {
        return false;
    }
    if (len != ZS_BASE_PAYLOAD_SIZE + static_cast<size_t>(num_cells) * ZS_CELL_ENTRY_SIZE) {
        return false;
    }

    RawEventFrame frame{};
    std::memset(&frame, 0, sizeof(RawEventFrame));
    std::memcpy(&frame, src, ZS_FRAME_PREFIX_SIZE);
    std::memcpy(reinterpret_cast<uint8_t*>(&frame) + ZS_FRAME_SUFFIX_OFFSET,
                src + ZS_FRAME_PREFIX_SIZE, ZS_FRAME_SUFFIX_SIZE);

    HltDecisionMetadata meta{};
    std::memcpy(&meta, src + ZS_FRAME_PREFIX_SIZE + ZS_FRAME_SUFFIX_SIZE, sizeof(HltDecisionMetadata));

    const uint8_t* cell_ptr = src + ZS_BASE_PAYLOAD_SIZE;
    for (uint16_t i = 0; i < num_cells; ++i) {
        const uint8_t idx = cell_ptr[0];
        if (idx >= CALORIMETER_CELLS) {
            return false;
        }
        uint16_t energy = 0;
        std::memcpy(&energy, cell_ptr + 1, sizeof(energy));
        frame.calorimeter_cells[idx] = energy;
        cell_ptr += ZS_CELL_ENTRY_SIZE;
    }

    std::memcpy(out_288b, &frame, sizeof(RawEventFrame));
    std::memcpy(out_288b + sizeof(RawEventFrame), &meta, sizeof(HltDecisionMetadata));
    return true;
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
        (header.schema_version != 1 && header.schema_version != DAQ_SCHEMA_VERSION) ||
        header.record_payload_size != DAQ_RECORD_PAYLOAD_SIZE) {
        return res;
    }

    res.header_valid = true;
    res.schema_version = header.schema_version;
    res.zero_suppressed = (header.flags & DAQ_FLAG_ZERO_SUPPRESSED) != 0ULL;
    res.run_start_timestamp_ns = header.run_start_timestamp_ns;

    const size_t body_bytes = static_cast<size_t>(file_size) - sizeof(DaqFileHeader);
    const size_t est_records = body_bytes / (res.zero_suppressed ? 180U : DAQ_SERIALIZED_RECORD_SIZE);
    res.packed_payloads.reserve(est_records * DAQ_RECORD_PAYLOAD_SIZE);

    if (!res.zero_suppressed) {
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
        if (ifs.gcount() > 0) {
            ++res.total_records;
            ++res.corrupt_records;
        }
    } else {
        std::array<uint8_t, ZS_MAX_PAYLOAD_SIZE> zs_buf{};
        std::array<uint8_t, DAQ_RECORD_PAYLOAD_SIZE> decoded{};
        DaqRecordHeader rh{};
        while (ifs.read(reinterpret_cast<char*>(&rh), sizeof(rh))) {
            ++res.total_records;
            if (rh.length < ZS_BASE_PAYLOAD_SIZE || rh.length > ZS_MAX_PAYLOAD_SIZE) {
                ++res.corrupt_records;
                break;
            }
            if (!ifs.read(reinterpret_cast<char*>(zs_buf.data()), static_cast<std::streamsize>(rh.length))) {
                ++res.corrupt_records;
                break;
            }
            const uint32_t computed_crc = crc32c_compute(zs_buf.data(), rh.length);
            if (computed_crc != rh.crc32c) {
                ++res.corrupt_records;
                continue;
            }
            if (!decode_zs_payload(zs_buf.data(), rh.length, decoded.data())) {
                ++res.corrupt_records;
                continue;
            }
            ++res.valid_records;
            res.packed_payloads.insert(
                res.packed_payloads.end(), decoded.begin(), decoded.end());
        }
        if (ifs.gcount() > 0 && ifs.gcount() < static_cast<std::streamsize>(sizeof(rh))) {
            ++res.total_records;
            ++res.corrupt_records;
        }
    }

    return res;
}

void EventWriter::initialize(const std::string& output_path,
                             bool zero_suppression,
                             uint16_t zs_pedestal_threshold) {
    output_path_ = output_path;
    zero_suppression_ = zero_suppression;
    zs_pedestal_threshold_ = zs_pedestal_threshold;
    buffers_[0] = std::make_unique<uint8_t[]>(STAGING_BUFFER_BYTES);
    buffers_[1] = std::make_unique<uint8_t[]>(STAGING_BUFFER_BYTES);
}

size_t EventWriter::encode_zs_payload(const RawEventFrame& frame,
                                      const HltDecisionMetadata& meta,
                                      uint8_t* dst) const noexcept {
    std::memcpy(dst, &frame, ZS_FRAME_PREFIX_SIZE);
    std::memcpy(dst + ZS_FRAME_PREFIX_SIZE,
                reinterpret_cast<const uint8_t*>(&frame) + ZS_FRAME_SUFFIX_OFFSET,
                ZS_FRAME_SUFFIX_SIZE);
    std::memcpy(dst + ZS_FRAME_PREFIX_SIZE + ZS_FRAME_SUFFIX_SIZE,
                &meta, sizeof(HltDecisionMetadata));

    uint8_t* cell_dst = dst + ZS_BASE_PAYLOAD_SIZE;
    uint16_t count = 0;
    for (size_t i = 0; i < CALORIMETER_CELLS; ++i) {
        const uint16_t e = frame.calorimeter_cells[i];
        if (e > zs_pedestal_threshold_) {
            cell_dst[0] = static_cast<uint8_t>(i);
            std::memcpy(cell_dst + 1, &e, sizeof(e));
            cell_dst += ZS_CELL_ENTRY_SIZE;
            ++count;
        }
    }

    const size_t count_off = ZS_FRAME_PREFIX_SIZE + ZS_FRAME_SUFFIX_SIZE + sizeof(HltDecisionMetadata);
    std::memcpy(dst + count_off, &count, sizeof(count));
    return ZS_BASE_PAYLOAD_SIZE + static_cast<size_t>(count) * ZS_CELL_ENTRY_SIZE;
}

bool EventWriter::flush_buffer(const uint8_t* data,
                               size_t len,
                               int fd,
                               bool has_file_sink,
                               bool& io_failed,
                               StorageTelemetry& telemetry) noexcept {
    if (len == 0) {
        return !io_failed;
    }
    if (!has_file_sink) {
        telemetry.bytes_written.fetch_add(len, std::memory_order_relaxed);
        telemetry.flush_count.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    if (io_failed || fd < 0) {
        telemetry.storage_write_errors.fetch_add(1, std::memory_order_relaxed);
        io_failed = true;
        return false;
    }

#if defined(__linux__) || defined(__unix__)
    size_t written = 0;
    while (written < len) {
        const ssize_t n = ::write(fd, data + written, len - written);
        if (n > 0) {
            written += static_cast<size_t>(n);
        } else if (n < 0 && errno == EINTR) {
            continue;
        } else {
            if (written > 0) {
                telemetry.bytes_written.fetch_add(written, std::memory_order_relaxed);
            }
            telemetry.storage_write_errors.fetch_add(1, std::memory_order_relaxed);
            io_failed = true;
            return false;
        }
    }
    telemetry.bytes_written.fetch_add(written, std::memory_order_relaxed);
    telemetry.flush_count.fetch_add(1, std::memory_order_relaxed);
    return true;
#else
    (void)data;
    telemetry.bytes_written.fetch_add(len, std::memory_order_relaxed);
    telemetry.flush_count.fetch_add(1, std::memory_order_relaxed);
    return true;
#endif
}

void EventWriter::run(SlabArena& arena,
                      BoundedChannel<SlotHandle>& storage_queue,
                      StorageTelemetry& telemetry,
                      bool pin_thread,
                      uint32_t core_idx) noexcept {
    if (pin_thread) {
        pin_current_thread(core_idx);
    }

    const bool has_file_sink = !output_path_.empty();
    bool io_failed = false;
    int fd = -1;
#if defined(__linux__) || defined(__unix__)
    if (has_file_sink) {
        fd = ::open(output_path_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            io_failed = true;
            telemetry.storage_write_errors.fetch_add(1, std::memory_order_relaxed);
        }
    }
#endif

    DaqFileHeader file_header{};
    std::memcpy(file_header.magic, DAQ_MAGIC, sizeof(DAQ_MAGIC));
    file_header.schema_version = DAQ_SCHEMA_VERSION;
    file_header.record_payload_size = DAQ_RECORD_PAYLOAD_SIZE;
    file_header.run_start_timestamp_ns = system_now_ns();
    file_header.flags = zero_suppression_ ? DAQ_FLAG_ZERO_SUPPRESSED : 0ULL;
    (void)flush_buffer(reinterpret_cast<const uint8_t*>(&file_header),
                       sizeof(file_header), fd, has_file_sink, io_failed, telemetry);
    if (!io_failed) {
        telemetry.raw_bytes_uncompressed.fetch_add(sizeof(file_header), std::memory_order_relaxed);
    }

    constexpr size_t MAX_REC_BYTES =
        sizeof(DaqRecordHeader) +
        (ZS_MAX_PAYLOAD_SIZE > DAQ_RECORD_PAYLOAD_SIZE ? ZS_MAX_PAYLOAD_SIZE : DAQ_RECORD_PAYLOAD_SIZE);

    int active_buf = 0;
    size_t active_offset = 0;
    uint32_t idle_iters = 0;

    for (;;) {
        SlotHandle slot = INVALID_SLOT;
        if (!storage_queue.try_pop(slot)) {
            if (storage_queue.is_closed() && storage_queue.empty()) {
                break;
            }
            idle_backoff(idle_iters);
            continue;
        }
        idle_iters = 0;

        if (active_offset + MAX_REC_BYTES > STAGING_BUFFER_BYTES) {
            const int flush_idx = active_buf;
            const size_t flush_len = active_offset;
            active_buf = 1 - active_buf;
            active_offset = 0;
            (void)flush_buffer(buffers_[flush_idx].get(), flush_len, fd,
                               has_file_sink, io_failed, telemetry);
        }

        uint8_t* dst = buffers_[active_buf].get() + active_offset;
        uint8_t* payload_dst = dst + sizeof(DaqRecordHeader);

        RawEventFrame* frame = arena.get_frame(slot);
        const HltDecisionMetadata* meta = arena.get_metadata(slot);

        size_t payload_len = DAQ_RECORD_PAYLOAD_SIZE;
        if (zero_suppression_) {
            frame->status_flags |= FRAME_STATUS_ZERO_SUPPRESSED;
            payload_len = encode_zs_payload(*frame, *meta, payload_dst);
        } else {
            std::memcpy(payload_dst, frame, sizeof(RawEventFrame));
            std::memcpy(payload_dst + sizeof(RawEventFrame), meta, sizeof(HltDecisionMetadata));
        }

        DaqRecordHeader rh{};
        rh.length = static_cast<uint32_t>(payload_len);
        rh.crc32c = crc32c_compute(payload_dst, payload_len);
        std::memcpy(dst, &rh, sizeof(rh));

        // Release slot only after full staging copy + record header + CRC32C completion
        arena.release(slot);

        active_offset += sizeof(DaqRecordHeader) + payload_len;
        if (!io_failed) {
            telemetry.events_written.fetch_add(1, std::memory_order_relaxed);
            telemetry.raw_bytes_uncompressed.fetch_add(
                DAQ_SERIALIZED_RECORD_SIZE, std::memory_order_relaxed);
        }
    }

    if (active_offset > 0) {
        (void)flush_buffer(buffers_[active_buf].get(), active_offset, fd,
                           has_file_sink, io_failed, telemetry);
    }

#if defined(__linux__) || defined(__unix__)
    if (fd >= 0) {
        ::close(fd);
    }
#endif
}

} // namespace triggerdaq
