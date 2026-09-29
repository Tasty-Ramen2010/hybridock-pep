#!/usr/bin/env python
"""Which ranker should pick our top-1 pose? ADCP currently beats us at it.

THE PROBLEM THIS EXISTS TO FIX. On the length-balanced bench our generation is ahead of RAPiDock
at every pose budget, but at TOP-1 macro-averaged ADCP beats us. ADCP ships a real affinity
ranker; ours is pose_ranker_ml. The gap is a selection problem, not a sampling one -- the oracle
column below is the ceiling a perfect ranker would reach from the exact same 24 poses.

Every candidate here is BLIND: none of them sees the crystal. RMSDs are used only to score the
pick afterwards.

  ml            pose_ranker_ml, what the tool ships today (order from logs/ranking_<arm>.jsonl)
  consensus     mean Ca RMSD to the other 23 poses -- the biggest cluster wins, no receptor,
                no training, no parameters. The classic docking-consensus baseline.
  ml+consensus  per-complex z-scores of both, summed equally. No weight is fitted; fitting one
                on the same 387 would make this a fit, not a comparison.
  random        mean over all 24 poses: what you get with no ranker at all
  oracle        best of the 24: the ceiling, and the number that says how much is left on the table

Usage: top1_bakeoff.py [--arm hybridock_ft] [--thr 5.0]
"""
from __future__ import annotations

import argparse, csv, glob, json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
RUNS = ROOT / "runs/balanced_length"


def ca(p) -> np.ndarray:
    seen, xyz = set(), []
    for l in Path(p).read_text(errors="ignore").splitlines():
        if not l.startswith("ATOM") or l[12:16].strip() != "CA":
            continue
        k = (l[21], l[22:27])
        if k in seen:
            continue
        seen.add(k)
        xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return np.array(xyz)


def consensus_scores(arm: str, name: str) -> np.ndarray | None:
    fs = sorted((RUNS / arm / name).glob("rank*.pdb"), key=lambda p: int(p.stem[4:]))
    if len(fs) < 3:
        return None
    X = [ca(f) for f in fs]
    n = min(len(x) for x in X)
    if n < 3:
        return None
    X = np.stack([x[:n] for x in X])                       # (P, n, 3)
    d = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1).mean(-1))   # (P, P) mean Ca RMSD
    return d.sum(1) / (len(X) - 1)                          # lower = more central


def z(v: np.ndarray) -> np.ndarray:
    s = v.std()
    return (v - v.mean()) / s if s > 1e-9 else np.zeros_like(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="hybridock_ft")
    ap.add_argument("--thr", type=float, default=5.0)
    a = ap.parse_args()

    cache = np.load(ROOT / "logs/balanced_figures_cache.npz", allow_pickle=True)
    R = {k: cache[k].item() for k in cache.files}
    rmsd = R[a.arm]
    bucket = {r["name"]: ("long" if r["length_bucket"] in ("long", "vlong") else r["length_bucket"])
              for r in csv.DictReader(open(ROOT / "data/bench_balanced_length.csv"))}

    ml = {}
    for l in (ROOT / f"logs/ranking_{a.arm}.jsonl").read_text().splitlines():
        if l.strip():
            d = json.loads(l)
            ml[d["name"]] = (d["order"], np.array(d["scores"], dtype=float))

    picks = defaultdict(lambda: defaultdict(list))
    for name, rs in rmsd.items():
        v = np.array(rs, dtype=float)
        if not np.isfinite(v).any() or name not in ml:
            continue
        cons = consensus_scores(a.arm, name)
        if cons is None or len(cons) != len(v):
            continue
        order, mls = ml[name]
        if len(mls) != len(v):
            continue
        b = bucket[name]
        cand = {"ml": v[order[0]],
                "consensus": v[int(np.argmin(cons))],
                "ml+consensus": v[int(np.argmin(z(mls) + z(cons)))],
                "random": float(np.nanmean(v)),
                "oracle": float(np.nanmin(v))}
        for k, val in cand.items():
            picks[k][b].append(val)
            picks[k]["ALL"].append(val)

    n = len(picks["ml"]["ALL"])
    print(f"\ntop-1 ranker bake-off — {a.arm}, {n} complexes, success = ≤{a.thr:g} Å\n")
    cols = ["short", "med", "long", "ALL", "MACRO"]
    print(f"{'ranker':<14}" + "".join(f"{c:>9}" for c in cols) + f"{'median':>9}")
    for k in ("random", "ml", "consensus", "ml+consensus", "oracle"):
        cells = []
        per = []
        for c in cols[:-1]:
            v = np.array(picks[k][c])
            r = 100 * (v <= a.thr).mean() if len(v) else np.nan
            cells.append(f"{r:>8.1f}%")
            if c != "ALL":
                per.append(r)
        cells.insert(4, f"{np.mean(per):>8.1f}%")
        print(f"{k:<14}" + "".join(cells) + f"{np.median(picks[k]['ALL']):>8.2f}Å")
    head = 100 * (np.array(picks["oracle"]["ALL"]) <= a.thr).mean() - \
           100 * (np.array(picks["ml"]["ALL"]) <= a.thr).mean()
    print(f"\n  headroom left for a perfect ranker over pose_ranker_ml: {head:.1f} points")


if __name__ == "__main__":
    main()
