#!/usr/bin/env python
"""Aggregate the CPU N=96 balanced-100 run into per-length-bucket docking stats.

Parses exp_runner logs (one line per complex:
  [name] partial=1:1:1 n=96 dt=..s best_direct=X.XXA spread=Y.YYA nearnative=k/96)
and joins them to data/bench_cpu100.csv for length_bucket / ss_class / pep_len.

Reports, per bucket and per (bucket, ss_class):
  n, median and mean best-of-N direct RMSD, <=2A and <=5A counts, and the mean
  fraction of the 96 samples that are near-native -- the last one matters because
  best-of-96 can succeed on a single lucky sample while the distribution is junk,
  which is exactly the "proper poses with proper distribution" question.

Usage: agg_cpu100.py <log> [<log> ...]
"""
from __future__ import annotations

import csv
import os
import re
import statistics as st
import sys
from collections import defaultdict

LINE = re.compile(
    r"^\[(?P<name>[^\]]+)\] partial=\S+ n=(?P<n>\d+) dt=(?P<dt>[\d.]+)s "
    r"best_direct=(?P<best>[\d.]+)A spread=(?P<spread>[\d.]+)A "
    r"nearnative=(?P<nn>\d+)/(?P<tot>\d+)"
)
ORDER = ["short", "medium", "long", "very_long"]


def load_meta(
    path: str = os.environ.get("AGG_META", "data/bench_cpu100.csv"),
) -> dict[str, dict[str, str]]:
    # AGG_META lets a scoped run (e.g. the <=25 aa set) report against its own complex
    # list, instead of counting deliberately-dropped complexes as "not yet run".
    with open(path) as fh:
        return {r["name"]: r for r in csv.DictReader(fh)}


def parse(logs: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for lg in logs:
        if not os.path.exists(lg):
            print(f"  (missing log: {lg})", file=sys.stderr)
            continue
        with open(lg) as fh:
            for line in fh:
                m = LINE.match(line.strip())
                if m:
                    out[m["name"]] = {
                        "best": float(m["best"]),
                        "spread": float(m["spread"]),
                        "nn_frac": int(m["nn"]) / max(int(m["tot"]), 1),
                        "dt": float(m["dt"]),
                        "n": int(m["n"]),
                    }
    return out


def block(title: str, rows: list[tuple[str, dict[str, float]]]) -> None:
    if not rows:
        return
    b = [r[1]["best"] for r in rows]
    nn = [r[1]["nn_frac"] for r in rows]
    le2 = sum(1 for x in b if x <= 2.0)
    le5 = sum(1 for x in b if x <= 5.0)
    print(
        f"  {title:22s} n={len(b):3d}  median={st.median(b):6.2f}A  mean={sum(b)/len(b):6.2f}A"
        f"  <=2A={le2:3d}  <=5A={le5:3d}  mean_nearnative_frac={sum(nn)/len(nn):.3f}"
    )


def main() -> None:
    logs = sys.argv[1:]
    if not logs:
        print(__doc__)
        sys.exit(1)
    meta, res = load_meta(), parse(logs)
    have = [(k, v) for k, v in res.items() if k in meta]
    missing = [k for k in meta if k not in res]
    ns = {v["n"] for _, v in have}
    print(f"parsed {len(have)}/{len(meta)} complexes  (N per complex: {sorted(ns)})")
    if have:
        tot_cpu = sum(v["dt"] for _, v in have)
        print(f"total inference wall time across complexes: {tot_cpu/3600:.2f} h")
    print("\n=== ALL ===")
    block("all", have)
    print("\n=== BY LENGTH BUCKET ===")
    for b in ORDER:
        block(b, [(k, v) for k, v in have if meta[k]["length_bucket"] == b])
    print("\n=== BY BUCKET x SS ===")
    for b in ORDER:
        for s in ("HELIX", "SHEET", "UNUSUAL"):
            block(
                f"{b}/{s}",
                [
                    (k, v)
                    for k, v in have
                    if meta[k]["length_bucket"] == b and meta[k]["ss_class"] == s
                ],
            )
    if missing:
        by = defaultdict(int)
        for k in missing:
            by[meta[k]["length_bucket"]] += 1
        print(f"\nnot yet run ({len(missing)}): {dict(by)}")


if __name__ == "__main__":
    main()
