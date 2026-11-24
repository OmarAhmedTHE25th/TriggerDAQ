"""Binary .daq run file inspector & zero-copy NumPy structured array reader."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from ._triggerdaq_ext import get_record_layout, verify_and_map_daq

# Exact packed binary layout matching [RawEventFrame(256B)][HltDecisionMetadata(32B)] = 288B
# Validated directly against C++ sizeof / offsetof at module import time.
RECORD_DTYPE = np.dtype(
    [
        # RawEventFrame (256 bytes, explicit_pad eliminates implicit compiler tail padding)
        ("event_id", "<u8"),
        ("bunch_crossing_id", "<u8"),
        ("ingest_timestamp_ns", "<u8"),
        ("calorimeter_cells", "<u2", (64,)),
        ("muon_hits", "<u2", (16,)),
        ("l1_trigger_bits", "<u4"),
        ("hlt_score", "<u2"),
        ("topology_tag", "u1"),
        ("reserved", "u1", (45,)),
        ("explicit_pad", "u1", (20,)),
        # HltDecisionMetadata (32 bytes)
        ("hlt_timestamp_ns", "<u8"),
        ("cluster_centroid", "<f4"),
        ("cluster_width", "<f4"),
        ("isolation_ratio", "<f4"),
        ("dimuon_inv_mass", "<f4"),
        ("meta_hlt_score", "<u2"),
        ("predicted_class", "u1"),
        ("accepted", "u1"),
        ("worker_id", "<u4"),
    ]
)


def _validate_record_dtype() -> None:
    layout = get_record_layout()
    expected_size = int(layout["record_payload_size"])
    if RECORD_DTYPE.itemsize != expected_size:
        raise RuntimeError(
            f"RECORD_DTYPE itemsize ({RECORD_DTYPE.itemsize}) != C++ record_payload_size ({expected_size})"
        )
    cpp_offsets: Dict[str, int] = layout["offsets"]
    for field_name, expected_offset in cpp_offsets.items():
        if field_name not in RECORD_DTYPE.fields:
            raise RuntimeError(f"Missing field '{field_name}' in RECORD_DTYPE")
        actual_offset = RECORD_DTYPE.fields[field_name][1]
        if actual_offset != expected_offset:
            raise RuntimeError(
                f"Field '{field_name}' offset mismatch: Python={actual_offset} vs C++={expected_offset}"
            )


_validate_record_dtype()

TOPOLOGY_NAMES = {
    0: "Background",
    1: "DiMuon",
    2: "LocalizedJet",
    3: "PileupNoise",
}


@dataclass
class DaqFileReport:
    path: str
    header_valid: bool
    schema_version: int
    run_start_timestamp_ns: int
    total_records: int
    valid_records: int
    corrupt_records: int
    records: np.ndarray

    @property
    def is_clean(self) -> bool:
        return self.header_valid and self.corrupt_records == 0

    def topology_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {name: 0 for name in TOPOLOGY_NAMES.values()}
        if len(self.records) == 0:
            return counts
        tags, freqs = np.unique(self.records["topology_tag"], return_counts=True)
        for tag, freq in zip(tags, freqs):
            counts[TOPOLOGY_NAMES.get(int(tag), f"Unknown({int(tag)})")] = int(freq)
        return counts


class DaqReader:
    """Zero-copy verified reader for TriggerDAQ .daq binary run files."""

    @staticmethod
    def read(path: str | Path) -> DaqFileReport:
        str_path = str(path)
        res = verify_and_map_daq(str_path)
        raw_buf: np.ndarray = res["raw_buffer"]
        if res["valid_records"] > 0:
            structured = np.ascontiguousarray(raw_buf).view(RECORD_DTYPE).reshape(-1)
        else:
            structured = np.empty(0, dtype=RECORD_DTYPE)

        return DaqFileReport(
            path=str_path,
            header_valid=bool(res["header_valid"]),
            schema_version=int(res["schema_version"]),
            run_start_timestamp_ns=int(res["run_start_timestamp_ns"]),
            total_records=int(res["total_records"]),
            valid_records=int(res["valid_records"]),
            corrupt_records=int(res["corrupt_records"]),
            records=structured,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="triggerdaq-inspect",
        description="Verify CRC32C integrity and summarize a TriggerDAQ .daq binary file.",
    )
    parser.add_argument("file", help="Path to .daq file")
    args = parser.parse_args(argv)

    report = DaqReader.read(args.file)
    print(f"TriggerDAQ Inspector: {report.path}")
    print(f"  Header Valid:    {report.header_valid} (schema v{report.schema_version})")
    print(f"  Total Records:   {report.total_records}")
    print(f"  Valid Records:   {report.valid_records}")
    print(f"  Corrupt Records: {report.corrupt_records}")

    if report.valid_records > 0:
        scores = report.records["hlt_score"]
        topo = report.topology_counts()
        signal = topo["DiMuon"] + topo["LocalizedJet"]
        purity = (signal / report.valid_records) * 100.0
        print(f"  HLT Score:       mean={scores.mean():.1f}, min={scores.min()}, max={scores.max()}")
        print(f"  Signal Purity:   {purity:.2f}%")
        print("  Topology Breakdown:")
        for name, cnt in topo.items():
            pct = (cnt / report.valid_records) * 100.0
            print(f"    - {name:14s}: {cnt:8d} ({pct:5.2f}%)")

    return 0 if report.is_clean else 2


if __name__ == "__main__":
    sys.exit(main())
