#!/usr/bin/env python3
"""compare_eval.py — Compare two per_pose_rmsd.csv runs on benchmark_real, per length bucket.

Reports DIRECT docking RMSD (best-of-N) and Hit@2A per length bucket + SS class for two
runs (e.g. pretrained baseline vs long-peptide retrained), with deltas. This is the honest
success test: did long/vlong Hit@2A rise WITHOUT regressing short/medium?

Usage:
    python scripts/compare_eval.py \
        --a /tmp/claude-1000/bench_real_baseline/per_pose_rmsd.csv --a-label pretrained \
        --b /tmp/claude-1000/bench_real_retrained/per_pose_rmsd.csv --b-label retrained \
        --bench data/benchmark_real.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path


def load_best(path: str) -> tuple[dict, dict]:
    """Return {name: best_direct}, {name: best_superposed}."""
    bd, bs = defaultdict(lambda: 1e9), defaultdict(lambda: 1e9)
    for r in csv.DictReader(open(path)):
        n = r["name"]
        bd[n] = min(bd[n], float(r["rmsd_direct"]))
        if "rmsd_superposed" in r and r["rmsd_superposed"]:
            bs[n] = min(bs[n], float(r["rmsd_superposed"]))
    return dict(bd), dict(bs)


def bucket(L: int) -> str:
    return ("short(<=8)" if L <= 8 else "med(9-12)" if L <= 12
            else "long(13-19)" if L <= 19 else "vlong(>=20)")


def agg(names, best, meta, key):
    """Return (mean_best, hit2_n, total_n) for complexes whose bucket/class == key group."""
    vals, hit, tot = [], 0, 0
    for n in names:
        if best.get(n, 1e9) >= 1e9:
            continue
        vals.append(best[n]); tot += 1; hit += 1 if best[n] <= 2.0 else 0
    return (st.mean(vals) if vals else float("nan")), hit, tot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True); ap.add_argument("--a-label", default="A")
    ap.add_argument("--b", required=True); ap.add_argument("--b-label", default="B")
    ap.add_argument("--bench", default="data/benchmark_real.csv")
    args = ap.parse_args()

    meta = {r["name"]: int(r["pep_len"]) for r in csv.DictReader(open(args.bench))}
    ssmeta = {r["name"]: r["ss_class"] for r in csv.DictReader(open(args.bench))}
    a_bd, a_bs = load_best(args.a)
    b_bd, b_bs = load_best(args.b)

    groups = defaultdict(list)
    for n, L in meta.items():
        groups[bucket(L)].append(n)
    for n, ss in ssmeta.items():
        groups["SS:" + ss].append(n)
    groups["ALL"] = list(meta)

    order = ["short(<=8)", "med(9-12)", "long(13-19)", "vlong(>=20)",
             "SS:HELIX", "SS:SHEET", "SS:UNUSUAL", "ALL"]

    print(f"\n{'group':14s} | {args.a_label:^22s} | {args.b_label:^22s} | Δ")
    print(f"{'':14s} | best  Hit@2       | best  Hit@2       | best   Hit@2")
    print("-" * 78)
    for g in order:
        names = groups.get(g, [])
        if not names:
            continue
        am, ah, at = agg(names, a_bd, meta, g)
        bm, bh, bt = agg(names, b_bd, meta, g)
        dbest = bm - am if am == am and bm == bm else float("nan")
        dhit = (bh / bt if bt else 0) - (ah / at if at else 0)
        flag = ""
        if g.startswith(("long", "vlong")) and dhit > 0.001:
            flag = "  ✅ improved"
        elif g.startswith(("short", "med")) and dhit < -0.05:
            flag = "  ⚠ REGRESSED"
        print(f"{g:14s} | {am:4.2f}  {ah:2d}/{at:<2d}={ah/at if at else 0:3.0%} | "
              f"{bm:4.2f}  {bh:2d}/{bt:<2d}={bh/bt if bt else 0:3.0%} | "
              f"{dbest:+5.2f}  {dhit:+4.0%}{flag}")

    # conformation guard: did superposed (conformation) regress?
    if a_bs and b_bs:
        av = st.mean([a_bs[n] for n in meta if n in a_bs])
        bv = st.mean([b_bs[n] for n in meta if n in b_bs])
        print(f"\nCONFORMATION guard (superposed best, mean): "
              f"{args.a_label}={av:.2f}A  {args.b_label}={bv:.2f}A  Δ={bv-av:+.2f}A "
              f"{'⚠ conformation regressed' if bv-av > 0.3 else 'ok'}")


if __name__ == "__main__":
    main()
