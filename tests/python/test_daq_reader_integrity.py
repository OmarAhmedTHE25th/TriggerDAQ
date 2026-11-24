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
    assert report.schema_version == 1
    assert report.total_records == snap.events_written
    assert report.valid_records == snap.events_written
    assert report.corrupt_records == 0
    assert report.records.dtype == td.RECORD_DTYPE
    assert len(report.records) == snap.events_written

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

