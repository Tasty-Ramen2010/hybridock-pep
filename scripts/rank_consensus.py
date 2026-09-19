#!/usr/bin/env python
"""Parameter-free consensus ranking of RAPiDock's unranked poses, scored against DockQ.

Every number we have published so far is an ORACLE best-of-24, because exp_runner runs with
--scoring_function none --confidence_ckpt null: the 24 poses come out in arbitrary order. That
is not comparable to the paper's ranked top-1. This is the cheapest honest ranker: a diffusion
sampler run 24 times puts most of its density near the mode it believes in, so the pose with
the SMALLEST mean heavy-atom RMSD to the other 23 is the ensemble's own consensus.

It has no parameters and never sees the crystal, so there is nothing to overfit and no
train/test split to get wrong -- the DockQ labels are used only to score the result afterwards.

Reported per arm:
  oracle  best-of-24          the ceiling (what we have been quoting)
  top-1   consensus pick      what a user would actually get
  top-5   best of 5 by rank   the paper also reports a top-5
  random  mean over poses     the no-ranker baseline

Usage: python scripts/rank_consensus.py [arm ...]   (default: all three RecentSet arms)
"""
from __future__ import annotations

import json
import os
import re
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
RUNS = ROOT / "runs/recentset"
DQ = ROOT / "logs/dockq_recentset.jsonl"
LEVELS = [("acceptable", 0.23), ("medium", 0.49), ("high", 0.80)]


def pose_coords(path: Path) -> np.ndarray:
    """Heavy-atom coordinates, in file order (poses share atom order within a complex)."""
    xs = []
    for line in path.read_text().splitlines():
        if line.startswith("ATOM") and line[76:78].strip() != "H":
            xs.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(xs, dtype=np.float64)


def rank_one(d: Path) -> tuple[str, list[int]]:
    """Pose indices (1-based, matching rankN.pdb) ordered most- to least-consensual."""
    files = sorted(d.glob("rank*.pdb"), key=lambda p: int(re.search(r"\d+", p.name).group()))
    idx = [int(re.search(r"\d+", p.name).group()) for p in files]
    mats = [pose_coords(p) for p in files]
    n = len(mats)
    if n < 2 or any(m.shape != mats[0].shape or m.size == 0 for m in mats):
        return d.name, idx  # ragged or empty: leave the arbitrary order alone
    x = np.stack(mats)                                   # (n, atoms, 3)
    # pairwise RMSD without superposition: the poses already share the receptor frame,
    # so the raw difference IS the pose difference (same convention as our direct RMSD).
    diff = x[:, None, :, :] - x[None, :, :, :]
    rms = np.sqrt((diff**2).sum(-1).mean(-1))            # (n, n)
    order = np.argsort(rms.sum(1))                       # smallest total distance first
    return d.name, [idx[i] for i in order]


def main() -> None:
    arms = sys.argv[1:] or ["rapidock_og", "hybridock_ft", "hybridock_ft_shipped"]
    # DockQ per pose, indexed the same way dockq_rs.py globbed them (rank1..rank24 sorted
    # numerically), so position i in the list is rank(i+1).pdb.
    dq: dict[str, dict[str, list[float]]] = {}
    for line in DQ.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            dq.setdefault(r["arm"], {})[r["name"]] = r["dockq"]

    for arm in arms:
        dirs = sorted(p for p in (RUNS / arm).iterdir() if p.is_dir() and p.name in dq.get(arm, {}))
        if not dirs:
            print(f"{arm}: no scored poses yet")
            continue
        workers = int(os.environ.get("RANK_WORKERS", "4"))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            ranked = dict(ex.map(rank_one, dirs, chunksize=8))

        rows: dict[str, list[float]] = {k: [] for k in ("oracle", "top1", "top5", "random")}
        for name, order in ranked.items():
            scores = dq[arm][name]
            ok = [s for s in scores if s == s]
            if not ok:
                continue
            # order holds 1-based rank file numbers; scores is 0-based in the same sequence
            picks = [scores[i - 1] for i in order if i - 1 < len(scores) and scores[i - 1] == scores[i - 1]]
            if not picks:
                continue
            rows["oracle"].append(max(ok))
            rows["top1"].append(picks[0])
            rows["top5"].append(max(picks[:5]))
            rows["random"].append(sum(ok) / len(ok))

        n = len(rows["oracle"])
        print(f"\n{arm}  (n={n})")
        print(f"  {'selection':10s} {'medianDockQ':>11s} " +
              " ".join(f"{lvl[:4]:>7s}" for lvl, _ in LEVELS))
        for key, label in [("oracle", "oracle24"), ("top5", "top-5"),
                           ("top1", "top-1"), ("random", "no ranker")]:
            v = rows[key]
            cells = " ".join(f"{100 * sum(1 for x in v if x >= thr) / len(v):6.1f}%"
                             for _, thr in LEVELS)
            print(f"  {label:10s} {st.median(v):11.3f} {cells}")
        keep = 100 * st.median(rows["top1"]) / st.median(rows["oracle"])
        print(f"  consensus top-1 keeps {keep:.0f}% of the oracle median")


if __name__ == "__main__":
    main()
