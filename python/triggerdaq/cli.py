"""CLI runner & live Rich/ANSI telemetry dashboard for TriggerDAQ."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

from ._triggerdaq_ext import (
    EmulatorMode,
    EngineConfig,
    TelemetrySnapshot,
    TriggerEngine,
    has_avx2_kernel,
)
from .menu import MENU_PRESETS, TriggerMenuSpec, get_preset_menu

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


MODE_MAP = {
    "sustained": EmulatorMode.Sustained,
    "burst": EmulatorMode.PoissonBurst,
    "stress": EmulatorMode.MaxThroughputStress,
}


def _build_rich_dashboard(snap: TelemetrySnapshot, menu_spec: TriggerMenuSpec, mode_name: str):
    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row(
        "Input Rate",
        f"{snap.input_rate_hz / 1e3:,.2f} kHz ({snap.events_generated:,})",
        "L1 Latency (ns)",
        f"p50={snap.l1_latency_ns.p50:.0f} | p99={snap.l1_latency_ns.p99:.0f} | p99.9={snap.l1_latency_ns.p999:.0f}",
    )
    table.add_row(
        "L1 Accept Rate",
        f"{snap.l1_rate_hz / 1e3:,.2f} kHz ({snap.l1_accepted:,})",
        "HLT Latency (us)",
        f"p50={snap.hlt_latency_us.p50:.2f} | p99={snap.hlt_latency_us.p99:.2f} | p99.9={snap.hlt_latency_us.p999:.2f}",
    )
    table.add_row(
        "HLT Accept Rate",
        f"{snap.hlt_rate_hz / 1e3:,.2f} kHz ({snap.hlt_accepted:,})",
        "Queue Fill (L1/HLT/IO)",
        f"{snap.l1_queue_fill*100:.1f}% / {snap.hlt_queue_fill*100:.1f}% / {snap.storage_queue_fill*100:.1f}%",
    )
    table.add_row(
        "Dead-Time",
        f"{snap.dead_time_fraction * 100:.3f}% ({snap.dead_time_events:,})",
        "Signal Purity / Eff",
        f"{snap.signal_purity * 100:.2f}% / {snap.signal_efficiency * 100:.2f}%",
    )
    table.add_row(
        "Arena Peak / Cap",
        f"{snap.arena_peak:,} / {snap.arena_capacity:,} (live {snap.arena_allocated:,})",
        "Written (CRC32C)",
        f"{snap.bytes_written / (1024*1024):.2f} MiB ({snap.events_written:,} ev, err={snap.storage_write_errors})",
    )

    prescales: List[str] = []
    for idx, rule in enumerate(menu_spec.rules):
        pf = snap.prescale_factors[idx] if idx < len(snap.prescale_factors) else 1.0
        cnt = snap.rule_accept_counts[idx] if idx < len(snap.rule_accept_counts) else 0
        prescales.append(f"{rule.name}: {pf:.2f}x ({cnt:,})")

    table.add_row("Active Prescales", " | ".join(prescales), "L1 Budget Violations", f"{snap.l1_budget_violations:,}")

    title = (
        f"TriggerDAQ Live Telemetry  |  mode={mode_name}  |  menu={menu_spec.name}  "
        f"|  AVX2={'ON' if has_avx2_kernel() else 'OFF'}  |  t={snap.elapsed_seconds:.2f}s"
    )
    return Panel(table, title=title, border_style="green")


def _format_ansi_line(snap: TelemetrySnapshot, menu_spec: TriggerMenuSpec) -> str:
    ps = ", ".join(
        f"{r.name}={snap.prescale_factors[i]:.1f}x"
        for i, r in enumerate(menu_spec.rules)
    )
    return (
        f"[t={snap.elapsed_seconds:5.2f}s] "
        f"In={snap.input_rate_hz/1e3:7.1f} kHz | "
        f"L1={snap.l1_rate_hz/1e3:6.1f} kHz (p99={snap.l1_latency_ns.p99:5.0f}ns) | "
        f"HLT={snap.hlt_rate_hz/1e3:5.1f} kHz (p99={snap.hlt_latency_us.p99:5.2f}us) | "
        f"DT={snap.dead_time_fraction*100:5.2f}% | "
        f"Purity={snap.signal_purity*100:5.1f}% | "
        f"PS=[{ps}]"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="triggerdaq",
        description="TriggerDAQ — Hybrid C++20 / Python Two-Stage Detector Trigger & DAQ Engine",
    )
    parser.add_argument("--duration", type=float, default=5.0, help="Run duration in seconds")
    parser.add_argument("--rate-khz", type=float, default=150.0, help="Target input rate in kHz")
    parser.add_argument("--hlt-workers", type=int, default=4, help="Number of HLT C++ worker threads")
    parser.add_argument("--arena-slots", type=int, default=65536, help="Pre-allocated SlabArena slots")
    parser.add_argument("--l1-budget-us", type=float, default=10.0, help="L1 latency budget in microseconds")
    parser.add_argument(
        "--mode",
        choices=list(MODE_MAP.keys()),
        default="sustained",
        help="Detector emulator mode",
    )
    parser.add_argument(
        "--menu",
        choices=list(MENU_PRESETS.keys()),
        default="physics_nominal",
        help="Declarative trigger menu preset",
    )
    parser.add_argument("--output", type=str, default="", help="Optional output .daq binary file path")
    parser.add_argument("--report-json", type=str, default="", help="Optional JSON telemetry summary path")
    parser.add_argument("--pin-threads", action="store_true", help="Pin HLT worker threads to CPU cores")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic PRNG seed")
    parser.add_argument("--no-rich", action="store_true", help="Force plain ANSI output")
    args = parser.parse_args(argv)

    menu_spec = get_preset_menu(args.menu)
    cpp_menu = menu_spec.compile()

    cfg = EngineConfig()
    cfg.arena_slots = args.arena_slots
    cfg.hlt_workers = args.hlt_workers
    cfg.target_rate_khz = args.rate_khz
    cfg.l1_budget_ns = int(args.l1_budget_us * 1000.0)
    cfg.mode = MODE_MAP[args.mode]
    cfg.pin_threads = args.pin_threads
    cfg.seed = args.seed
    cfg.output_path = args.output

    engine = TriggerEngine()
    engine.initialize(cfg, cpp_menu)

    use_rich = HAS_RICH and (not args.no_rich) and sys.stdout.isatty()
    engine.start()
    t_end = time.monotonic() + max(0.05, args.duration)

    try:
        if use_rich:
            console = Console()
            with Live(
                _build_rich_dashboard(engine.snapshot(), menu_spec, args.mode),
                console=console,
                refresh_per_second=10,
            ) as live:
                while time.monotonic() < t_end:
                    time.sleep(0.1)
                    live.update(_build_rich_dashboard(engine.snapshot(), menu_spec, args.mode))
        else:
            while time.monotonic() < t_end:
                time.sleep(0.25)
                print(_format_ansi_line(engine.snapshot(), menu_spec), flush=True)
    finally:
        engine.stop()

    final_snap = engine.snapshot()
    print("\n=== TriggerDAQ Final Summary ===")
    print(f"  Duration:       {final_snap.elapsed_seconds:.3f} s  (mode={args.mode}, menu={args.menu})")
    print(f"  Generated:      {final_snap.events_generated:,} ({final_snap.input_rate_hz/1e3:,.2f} kHz)")
    print(f"  L1 Accepted:    {final_snap.l1_accepted:,} ({final_snap.l1_rate_hz/1e3:,.2f} kHz)")
    print(f"  HLT Accepted:   {final_snap.hlt_accepted:,} ({final_snap.hlt_rate_hz/1e3:,.2f} kHz)")
    print(f"  Dead-Time:      {final_snap.dead_time_events:,} ({final_snap.dead_time_fraction*100:.3f}%)")
    print(
        f"  L1 Latency ns:  p50={final_snap.l1_latency_ns.p50:.1f}  "
        f"p99={final_snap.l1_latency_ns.p99:.1f}  "
        f"p99.9={final_snap.l1_latency_ns.p999:.1f}  "
        f"max={final_snap.l1_latency_ns.max:.1f}"
    )
    print(
        f"  HLT Latency us: p50={final_snap.hlt_latency_us.p50:.2f}  "
        f"p99={final_snap.hlt_latency_us.p99:.2f}  "
        f"p99.9={final_snap.hlt_latency_us.p999:.2f}  "
        f"max={final_snap.hlt_latency_us.max:.2f}"
    )
    print(
        f"  Signal Purity:  {final_snap.signal_purity*100:.2f}%  |  "
        f"Efficiency: {final_snap.signal_efficiency*100:.2f}%"
    )
    print(
        f"  Arena Peak:     {final_snap.arena_peak:,} / {final_snap.arena_capacity:,}  "
        f"(leaked: {final_snap.arena_allocated})"
    )
    if args.output:
        print(
            f"  DAQ Output:     {args.output} ({final_snap.events_written:,} events, "
            f"{final_snap.bytes_written:,} bytes, errors={final_snap.storage_write_errors})"
        )
    if final_snap.storage_write_errors > 0:
        print(
            f"  [ERROR] Storage I/O encountered {final_snap.storage_write_errors} write error(s)!",
            file=sys.stderr,
        )

    if args.report_json:
        report = final_snap.to_dict()
        report["config"] = {
            "duration": args.duration,
            "rate_khz": args.rate_khz,
            "hlt_workers": args.hlt_workers,
            "arena_slots": args.arena_slots,
            "l1_budget_us": args.l1_budget_us,
            "mode": args.mode,
            "menu": args.menu,
            "avx2": has_avx2_kernel(),
        }
        Path(args.report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  JSON Report:    {args.report_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
