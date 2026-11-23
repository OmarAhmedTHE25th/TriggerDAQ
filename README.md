# TriggerDAQ

**Hybrid C++20 / Python Two-Stage Real-Time Detector Trigger & Data Acquisition (DAQ) Engine**

Modeled on CERN's ATLAS/CMS Level-1 (L1) and High-Level Trigger (HLT) architecture, TriggerDAQ separates **lock-free, zero-allocation C++20 hot-path execution** from a **Python 3.12+ control, telemetry, and analysis plane** connected via `nanobind`.

---

## Architecture & Dataflow

```text
+=======================================================================================+
|                       PYTHON 3.12+ CONTROL & TELEMETRY PLANE                          |
|  [Declarative TriggerMenu]    [Live Rich/ANSI Dashboard CLI]    [DaqReader / NumPy]   |
|         |  update_menu()               ^  snapshot() (<1 us)          ^  zero-copy    |
|         |  (atomic swap)               |  (lock-free atomics)         |  CRC32C view  |
+=========|==============================|==============================|===============+
          |          nanobind (_triggerdaq_ext, GIL released)           |
+=========v==============================|==============================|===============+
|                        C++20 ZERO-ALLOCATION HOT-PATH CORE            |               |
|                                                                       |               |
|  +-------------------+   SlotHandle   +-----------------------+       |               |
|  | Detector Emulator | -------------> |    L1 SIMD Trigger    |       |               |
|  | (xoshiro256**)    |  BoundedChan   | AVX2 / Scalar Kernels |       |               |
|  | ~95% Bg, 1.5% DiMu|                | + Closed-Loop PI      |       |               |
|  | 2% Jet, 1.5% PU   |                |   Prescaler (75% WM)  |       |               |
|  +---------+---------+                +-----------+-----------+       |               |
|            | acquire()                            | SlotHandle        |               |
|            v                                      v (BoundedChan)     |               |
|  +-------------------+                +-----------------------+       |               |
|  |    SlabArena      |                |   HLT Worker Farm     |       |               |
|  | 64B-aligned pool  | <------------- | W std::jthreads       |       |               |
|  | ABA-safe freelist |  release()     | Cluster/DiMu Recon +  |       |               |
|  | 256B RawEventFrame|  on reject     | Quantized Int8 MLP    |       |               |
|  +---------^---------+                +-----------+-----------+       |               |
|            |                                      | SlotHandle        |               |
|            | release()                            v (BoundedChan)     |               |
|            |                          +-----------------------+       |               |
|            +------------------------- |  Async EventWriter    | ------+               |
|                                       | Double-Buffered 256KB |  .daq binary file     |
|                                       | Hardware SSE4.2 CRC32C|  [32B hdr][296B recs] |
|                                       +-----------------------+                       |
+=======================================================================================+
```

### Engineering Guarantees
- **Zero Hot-Path Allocations:** After `TriggerEngine::initialize()`, no thread on the Frontend → L1 → HLT → Storage path allocates heap memory or resizes STL containers. Events travel solely as 32-bit `SlotHandle` descriptors referencing pre-allocated, 64-byte-aligned `RawEventFrame` (256 B) and `HltDecisionMetadata` (32 B) slots in `SlabArena`.
- **GIL-Free Execution:** All `nanobind` execution entry points (`start`, `stop`, `run_for_seconds`, `verify_and_map_daq`) release the Python GIL (`nb::gil_scoped_release`). Telemetry polling copies an atomic snapshot in `< 1 µs` without locking worker threads.
- **Bit-Identical SIMD Dispatch:** Compile-time AVX2 L1 calorimeter & muon reduction kernels (`extract_l1_features_avx2`) are verified bit-identical to the portable C++20 scalar implementation across 50,000 full-range frames (`test_l1_simd`).
- **Closed-Loop PI Backpressure:** Monitors HLT queue fill ratio (`> 75%` watermark) and dynamically scales prescale factors on `Low`/`Normal`/`High` priority rules using a deterministic fractional phase accumulator while locking `Critical` discovery rules at `1.0x`.
- **Hardware-Accelerated CRC32C Storage:** Double-buffered staging writer frames every accepted event (`[4B length][4B crc32c][256B RawEventFrame][32B HltDecisionMetadata]`) using SSE4.2 `_mm_crc32_u64` (with Castagnoli software fallback), and `verify_and_map_daq()` exposes verified records directly as a zero-copy NumPy structured array.

---

## Building & Testing

### 1. C++20 Core, Unit Tests & Benchmark (`cmake` + `ctest`)
```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure -V
./build/bench_pipeline
```

### 2. Python Control Plane & `nanobind` Extension
```bash
pip install -e .
pytest -v tests/python/
```

### 3. CLI Runner & Binary `.daq` Inspector
```bash
# Run a 5-second max-throughput stress run and record CRC32C-framed events
triggerdaq --mode stress --duration 5 --menu physics_nominal --output run.daq --report-json report.json

# Verify CRC32C checksums and inspect topology purity
triggerdaq-inspect run.daq
```

---

## Benchmark Results

Measured on Ubuntu 24.04 (WSL2, `g++ 13.3.0`, `-std=c++20 -O3 -mavx2 -msse4.2`, 4 HLT workers, 65,536 `SlabArena` slots, `physics_nominal` menu):

### Pure C++20 Pipeline (`./build/bench_pipeline`)
| Mode | Input Rate | L1 Accept | HLT Accept | Dead-Time | L1 Latency (p50 / p99 / p99.9) | HLT Latency (p50 / p99 / p99.9) | Signal Purity / Eff | Leaked Slots |
| :--- | ---: | ---: | ---: | ---: | :---: | :---: | :---: | ---: |
| **Sustained (200 kHz)** | 175.35 kHz | 8.76 kHz | 6.12 kHz | 0.00% | 200 ns / 512 ns / 3,200 ns | 0.96 µs / 2.94 µs / 73.73 µs | 100.00% / 99.82% | 0 |
| **PoissonBurst (250 kHz)** | 249.16 kHz | 12.56 kHz | 8.76 kHz | 0.00% | 192 ns / 352 ns / 1,472 ns | 0.93 µs / 2.05 µs / 22.53 µs | 100.00% / 99.86% | 0 |
| **MaxThroughputStress** | **1,558.37 kHz** | 77.94 kHz | 54.52 kHz | 0.00% | **184 ns / 304 ns / 832 ns** | **0.83 µs / 1.47 µs / 20.48 µs** | **100.00% / 99.86%** | 0 |

### End-to-End Python CLI Stress Run with Disk `.daq` Storage (`triggerdaq --mode stress --duration 5`)
- **Generated:** `5,879,704` frames (`1,141.98 kHz`)
- **L1 Accepted:** `295,008` (`57.30 kHz`, `p50=200 ns`, `p99=432 ns`, `p99.9=1,472 ns`)
- **HLT Accepted & Written:** `206,703` (`40.15 kHz`, `61.18 MB` CRC32C-framed `.daq`, `p50=0.86 µs`, `p99=1.86 µs`)
- **Dead-Time:** `0.015%` (`866` drops) | **Leaked Arena Slots:** `0`
- **Integrity (`triggerdaq-inspect`):** `206,703 / 206,703` valid records (`0` corrupt), `100.00%` signal purity (`88,481` DiMuon + `118,222` LocalizedJet, `0` Background, `0` PileupNoise).

---

## Verification Summary
- **C++ (`ctest`):** 4/4 passing (`test_slab_arena` 10M cycles across 8 threads, `test_l1_simd` AVX2 vs. scalar 50k frames bit-identical, `test_prescaler` PI backpressure & Critical 1.0x lock, `test_storage_crc` RFC 3720 + bit-flip corruption detection).
- **Python (`pytest`):** 4/4 passing (`test_engine_lifecycle_and_gil_release`, `test_lock_free_telemetry_polling`, `test_runtime_menu_hot_reload`, `test_daq_write_read_and_corruption_detection`).
