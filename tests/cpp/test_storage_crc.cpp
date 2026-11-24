#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "triggerdaq/core/bounded_channel.hpp"
#include "triggerdaq/core/slab_arena.hpp"
#include "triggerdaq/frontend/detector_emulator.hpp"
#include "triggerdaq/storage/event_writer.hpp"

using namespace triggerdaq;

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::cerr << "CHECK FAILED at " << __FILE__ << ":" << __LINE__ << ": " #cond \
                      << "\n";                                                           \
            std::abort();                                                                \
        }                                                                                \
    } while (0)

int main() {
    // 1. RFC 3720 CRC32C check value: "123456789" -> 0xE3069283
    {
        const char* vec = "123456789";
        const uint32_t sw = crc32c_software(vec, 9);
        const uint32_t hw = crc32c_compute(vec, 9);
        CHECK(sw == 0xE3069283U);
        CHECK(hw == 0xE3069283U);
    }

    // 2. Random lengths (0..1024) HW vs SW bit-identity
    {
        Xoshiro256StarStar rng(777);
        uint8_t buf[1024];
        for (size_t i = 0; i < sizeof(buf); ++i) {
            buf[i] = static_cast<uint8_t>(rng.next() & 0xFFU);
        }
        for (size_t len = 0; len <= 256; ++len) {
            CHECK(crc32c_software(buf, len) == crc32c_compute(buf, len));
        }
        CHECK(crc32c_software(buf, sizeof(buf)) == crc32c_compute(buf, sizeof(buf)));
    }

    // 3. End-to-end EventWriter -> verify_daq_file -> bit-flip corruption detection
    const std::string path = "/tmp/triggerdaq_test_storage_crc.daq";
    {
        SlabArena arena;
        arena.initialize(1024);
        BoundedChannel<SlotHandle> q(1024);
        StorageTelemetry tel{};
        EventWriter writer;
        writer.initialize(path);

        Xoshiro256StarStar rng(99);
        constexpr int NUM_EVENTS = 250;
        for (int i = 0; i < NUM_EVENTS; ++i) {
            SlotHandle s = arena.acquire();
            CHECK(s != INVALID_SLOT);
            RawEventFrame* f = arena.get_frame(s);
            HltDecisionMetadata* m = arena.get_metadata(s);
            DetectorEmulator::generate_frame(*f, static_cast<uint64_t>(i + 1), static_cast<uint64_t>(i), 1000, rng);
            std::memset(m, 0, sizeof(*m));
            m->hlt_score = 900;
            m->accepted = 1;
            m->worker_id = 0;
            CHECK(q.try_push(s));
        }

        q.close(); // Ordered EOS signal: drain all queued events and exit
        writer.run(arena, q, tel);

        CHECK(arena.allocated_count() == 0);
        CHECK(tel.storage_write_errors.load() == 0);
        CHECK(tel.events_written.load() == NUM_EVENTS);
        CHECK(tel.bytes_written.load() == sizeof(DaqFileHeader) + NUM_EVENTS * DAQ_SERIALIZED_RECORD_SIZE);

        DaqVerificationResult res = verify_daq_file(path);
        CHECK(res.header_valid);
        CHECK(res.total_records == NUM_EVENTS);
        CHECK(res.valid_records == NUM_EVENTS);
        CHECK(res.corrupt_records == 0);
        CHECK(res.packed_payloads.size() == NUM_EVENTS * DAQ_RECORD_PAYLOAD_SIZE);

        // Corrupt a single bit in record #10 payload and re-verify
        {
            std::fstream fs(path, std::ios::binary | std::ios::in | std::ios::out);
            CHECK(fs.is_open());
            const std::streamoff offset =
                static_cast<std::streamoff>(sizeof(DaqFileHeader) + 10 * DAQ_SERIALIZED_RECORD_SIZE + 16);
            fs.seekg(offset);
            char byte = 0;
            fs.read(&byte, 1);
            byte ^= 0x01;
            fs.seekp(offset);
            fs.write(&byte, 1);
            fs.flush();
        }

        DaqVerificationResult corrupt_res = verify_daq_file(path);
        CHECK(corrupt_res.header_valid);
        CHECK(corrupt_res.total_records == NUM_EVENTS);
        CHECK(corrupt_res.valid_records == NUM_EVENTS - 1);
        CHECK(corrupt_res.corrupt_records == 1);
    }

    // 4. Simulated disk write error handling (unwritable directory path)
    {
        SlabArena arena;
        arena.initialize(64);
        BoundedChannel<SlotHandle> q(64);
        StorageTelemetry tel{};
        EventWriter writer;
        writer.initialize("/nonexistent_dir_triggerdaq/out.daq");

        for (int i = 0; i < 16; ++i) {
            SlotHandle s = arena.acquire();
            CHECK(s != INVALID_SLOT);
            CHECK(q.try_push(s));
        }
        q.close();
        writer.run(arena, q, tel);

        CHECK(arena.allocated_count() == 0); // Slots still cleanly released!
        CHECK(tel.storage_write_errors.load() > 0);
        CHECK(tel.bytes_written.load() == 0); // No silent byte-counter inflation
        CHECK(tel.events_written.load() == 0);
    }

    std::remove(path.c_str());
    std::cout << "[PASS] test_storage_crc (RFC3720, HW/SW CRC32C, .daq write/verify, bit-flip & I/O error detection)\n";
    return 0;
}
