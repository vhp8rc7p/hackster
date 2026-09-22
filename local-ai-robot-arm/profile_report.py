#!/usr/bin/env python3
"""
Profiler / bottleneck report for the robot pipeline.

Reads the metrics the main script already writes while it runs:
  timing.csv      — one row per phase  (search, approach, down, dwell, lift, ...)
  owl_metrics.csv — one row per OWL inference (ms, cpu, mem, query)

and tells you WHERE THE TIME GOES, worst offender first, plus concrete advice
on which knob to turn.

USAGE
  ./mlx_env/bin/python profile_report.py              # full report
  ./mlx_env/bin/python profile_report.py --last 5     # only the last 5 picks
  ./mlx_env/bin/python profile_report.py --watch      # live, refreshes every 2s

There is also a code-level profiler for finding slow Python (rarely the issue
here — the time is in OWL inference and robot motion, not Python):
  ./mlx_env/bin/python -m cProfile -s cumtime qwen_command.py > prof.txt
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

TIMING_LOG = "timing.csv"
OWL_LOG = "owl_metrics.csv"

# What each phase means and which knob changes it. Keeps the advice concrete
# instead of "make it faster".
PHASE_HELP = {
    "search":   ("OWL hunting for the object",
                 "SEARCH_MAX_SECONDS caps it; each frame ≈ one OWL inference. "
                 "Slow = object hard to see, or OWL cold (check warmup ran)."),
    "approach": ("moving to the hover above the target",
                 "SPEED (higher = faster). Repeated 'attempt 2/3' in the log "
                 "means the target was unreachable — fix reach, not speed."),
    "down":     ("descending onto the object",
                 "SPEED. Short move, so time here is mostly settle/poll."),
    "dwell":    ("waiting with the pump on to seal",
                 "PUMP_DWELL_S — lower it if the cup grabs quickly."),
    "lift":     ("raising the object back to hover",
                 "SPEED."),
    "wait-raise": ("raising after a 'wait' interrupt", "SPEED."),
}


def read_timing(path=TIMING_LOG):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({"t": float(r["timestamp"]),
                             "phase": r["phase"],
                             "s": float(r["seconds"])})
            except (ValueError, KeyError):
                continue
    return rows


def read_owl(path=OWL_LOG):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({"t": float(r["timestamp"]),
                             "site": r.get("site", "?"),
                             "ms": float(r["owl_ms"]),
                             "cpu": float(r.get("cpu_pct") or 0),
                             "mem": float(r.get("mem_gb") or 0),
                             "query": r.get("query", "")})
            except (ValueError, KeyError):
                continue
    return rows


def group_runs(rows, gap_s=20.0):
    """Split phase rows into runs (a gap in time = a new action)."""
    runs, cur, last = [], [], None
    for r in rows:
        if last is not None and r["t"] - last > gap_s:
            if cur:
                runs.append(cur)
            cur = []
        cur.append(r)
        last = r["t"]
    if cur:
        runs.append(cur)
    return runs


def bar(frac, width=28):
    n = max(0, min(width, int(round(frac * width))))
    return "█" * n + "·" * (width - n)


def pct(x, total):
    return (x / total * 100.0) if total else 0.0


def report(last_n=None):
    timing = read_timing()
    owl = read_owl()

    if not timing:
        print(f"No {TIMING_LOG} yet — run qwen_command.py and do a pick first.")
        return

    runs = group_runs(timing)
    if last_n:
        runs = runs[-last_n:]
    rows = [r for run in runs for r in run]

    print("=" * 72)
    print("  PIPELINE PROFILE — where the time goes")
    print("=" * 72)
    print(f"  runs analysed: {len(runs)}   phase samples: {len(rows)}")

    # ── aggregate per phase ──
    agg = defaultdict(list)
    for r in rows:
        agg[r["phase"]].append(r["s"])
    grand = sum(sum(v) for v in agg.values())

    print(f"\n  {'phase':<12} {'total':>7} {'share':>7} {'avg':>7} "
          f"{'min':>6} {'max':>6} {'n':>4}")
    print("  " + "-" * 62)
    ranked = sorted(agg.items(), key=lambda kv: -sum(kv[1]))
    for phase, vals in ranked:
        tot = sum(vals)
        print(f"  {phase:<12} {tot:6.1f}s {pct(tot, grand):6.1f}% "
              f"{tot/len(vals):6.2f}s {min(vals):5.2f}s {max(vals):5.2f}s "
              f"{len(vals):4d}")

    # ── visual ranking ──
    print(f"\n  BOTTLENECKS (share of total time)")
    for phase, vals in ranked:
        tot = sum(vals)
        f = tot / grand if grand else 0
        print(f"    {phase:<12} {bar(f)} {pct(tot, grand):5.1f}%")

    # ── per-run totals ──
    print(f"\n  PER-RUN TOTALS")
    for i, run in enumerate(runs, 1):
        tot = sum(r["s"] for r in run)
        parts = " ".join(f"{r['phase']}:{r['s']:.1f}" for r in run)
        print(f"    run {i:>2}  {tot:5.1f}s   {parts}")
    totals = [sum(r["s"] for r in run) for run in runs]
    if totals:
        print(f"    {'':6} avg {sum(totals)/len(totals):.1f}s   "
              f"best {min(totals):.1f}s   worst {max(totals):.1f}s")

    # ── OWL inference detail ──
    if owl:
        ms = [o["ms"] for o in owl]
        ms_sorted = sorted(ms)
        p50 = ms_sorted[len(ms_sorted)//2]
        p90 = ms_sorted[int(len(ms_sorted)*0.9)] if len(ms_sorted) > 1 else ms_sorted[0]
        print(f"\n  OWL INFERENCE  ({len(ms)} calls)")
        print(f"    median {p50:6.0f}ms   p90 {p90:6.0f}ms   "
              f"min {min(ms):.0f}ms   max {max(ms):.0f}ms")
        if max(ms) > 3 * p50 and max(ms) > 2000:
            print(f"    ⚠ slowest call {max(ms):.0f}ms — that's the cold-start "
                  f"compile. Confirm the startup warmup ran.")
        by_site = defaultdict(list)
        for o in owl:
            by_site[o["site"]].append(o["ms"])
        for site, v in sorted(by_site.items(), key=lambda kv: -sum(kv[1])):
            print(f"    {site:<18} n={len(v):<4} avg {sum(v)/len(v):6.0f}ms")

    # ── advice, worst first ──
    print(f"\n  WHAT TO TUNE (worst offender first)")
    for phase, vals in ranked[:3]:
        what, knob = PHASE_HELP.get(phase, ("—", "—"))
        print(f"    • {phase} ({pct(sum(vals), grand):.0f}% of time) — {what}")
        print(f"        {knob}")
    print("=" * 72)


def watch():
    try:
        while True:
            os.system("clear")
            report(last_n=5)
            print("\n  (live — Ctrl-C to stop)")
            time.sleep(2)
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser(description="Pipeline bottleneck report")
    ap.add_argument("--last", type=int, default=None,
                    help="only analyse the last N runs")
    ap.add_argument("--watch", action="store_true",
                    help="refresh live every 2s")
    args = ap.parse_args()
    if args.watch:
        watch()
    else:
        report(last_n=args.last)


if __name__ == "__main__":
    main()
