"""Tests for binary .daq writing, zero-copy NumPy reading, and CRC32C corruption detection."""

from pathlib import Path

import numpy as np
import triggerdaq as td


def test_daq_write_read_and_corruption_detection(tmp_path: Path):
    daq_path = tmp_path / "run_001.daq"

    cfg = td.EngineConfig()
    cfg.arena_slots = 16384
    cfg.hlt_workers = 2
    cfg.target_rate_khz = 120.0
    cfg.mode = td.EmulatorMode.Sustained
    cfg.seed = 404
    cfg.output_path = str(daq_path)

    engine = td.TriggerEngine()
    engine.initialize(cfg, td.physics_nominal().compile())
    engine.run_for_seconds(0.35)

    snap = engine.snapshot()
    assert snap.events_written > 100
    assert snap.arena_allocated == 0
    assert daq_path.exists()

    # Zero-copy verified read into NumPy structured array
    report = td.DaqReader.read(daq_path)
    assert report.is_clean
    assert report.header_valid
    assert not report.zero_suppressed
    assert report.schema_version == 2
    assert report.total_records == snap.events_written
    assert report.valid_records == snap.events_written
    assert report.corrupt_records == 0
    assert report.records.dtype == td.RECORD_DTYPE
    assert len(report.records) == snap.events_written

    # Verify LHC BCID bounds and abort-gap exclusion
    assert np.all(report.records["bcid"] < 3444)
    assert np.all(report.records["lumi_block_id"] >= 1)

    # Verify structured array fields are sensible
    assert np.all(report.records["accepted"] == 1)
    assert np.all(report.records["hlt_score"] >= 500)
    assert np.all(report.records["hlt_score"] == report.records["meta_hlt_score"])

    topo = report.topology_counts()
    signal = topo["DiMuon"] + topo["LocalizedJet"]
    assert signal / report.valid_records > 0.90

    # Flip one bit in record #5 and verify CRC32C catches exactly 1 corrupt record
    raw_bytes = bytearray(daq_path.read_bytes())
    corrupt_offset = 32 + 5 * 296 + 20  # 32B header + 5 records + inside RawEventFrame
    raw_bytes[corrupt_offset] ^= 0x80
    daq_path.write_bytes(raw_bytes)

    corrupt_report = td.DaqReader.read(daq_path)
    assert corrupt_report.header_valid
    assert not corrupt_report.is_clean
    assert corrupt_report.total_records == snap.events_written
    assert corrupt_report.valid_records == snap.events_written - 1
    assert corrupt_report.corrupt_records == 1
    assert len(corrupt_report.records) == snap.events_written - 1


def test_record_layout_validation_and_io_error_surfacing():
    layout = td.get_record_layout()
    assert layout["frame_size"] == 256
    assert layout["meta_size"] == 32
    assert layout["record_payload_size"] == td.RECORD_DTYPE.itemsize == 288
    assert layout["offsets"]["explicit_pad"] == 236

    cfg = td.EngineConfig()
    cfg.arena_slots = 4096
    cfg.hlt_workers = 2
    cfg.target_rate_khz = 100.0
    cfg.mode = td.EmulatorMode.Sustained
    cfg.output_path = "/nonexistent_triggerdaq_dir/cannot_write.daq"

    engine = td.TriggerEngine()
    engine.initialize(cfg, td.physics_nominal().compile())
    engine.run_for_seconds(0.2)

    snap = engine.snapshot()
    assert snap.arena_allocated == 0
    assert snap.storage_write_errors > 0
    assert snap.bytes_written == 0
    assert snap.to_dict()["storage_write_errors"] == snap.storage_write_errors


def test_zero_suppressed_daq_roundtrip_and_lumiblocks(tmp_path: Path):
    zs_path = tmp_path / "run_zs.daq"

    cfg = td.EngineConfig()
    cfg.arena_slots = 16384
    cfg.hlt_workers = 2
    cfg.target_rate_khz = 120.0
    cfg.mode = td.EmulatorMode.Sustained
    cfg.lumi_block_interval_s = 0.15
    cfg.enable_zero_suppression = True
    cfg.zs_pedestal_threshold = 20
    cfg.seed = 505
    cfg.output_path = str(zs_path)

    engine = td.TriggerEngine()
    engine.configure(cfg, td.physics_nominal().compile())
    engine.run_for_seconds(0.45)

    snap = engine.snapshot()
    assert snap.arena_allocated == 0
    assert snap.events_written > 100
    assert snap.compression_ratio > 1.35

    report = td.DaqReader.read(zs_path)
    assert report.is_clean
    assert report.zero_suppressed
    assert report.valid_records == snap.events_written
    assert np.all(report.records["accepted"] == 1)
    assert np.all(report.records["bcid"] < 3444)
    # Monotonic LumiBlock progression
    lbs = report.records["lumi_block_id"]
    assert lbs.max() >= 2

    # Corrupt a byte in the ZS file and verify CRC32C detection
    raw = bytearray(zs_path.read_bytes())
    raw[32 + 20] ^= 0x10
    zs_path.write_bytes(raw)
    corrupt = td.DaqReader.read(zs_path)
    assert corrupt.header_valid
    assert corrupt.corrupt_records >= 1


