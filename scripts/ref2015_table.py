#!/usr/bin/env python
"""RAPiDock's own ranker (ref2015) vs our consensus ranker, on the leak-free RecentSet.

Both pick ONE pose from the same 24 unranked diffusion samples, so this isolates the ranker:
the poses are identical, only the selection rule differs.

  ref2015     RAPiDock's default (default_inference_args.yaml): PyRosetta FastRelax on the
              peptide, then Rosetta ref2015 energy of the complex, ascending. Their other
              option, --scoring_function confidence, ships NO weights and cannot be run.
  consensus   ours: the pose with the smallest mean RMSD to the other 23 (no parameters).
  oracle      best-of-24 ceiling.        random   mean over the 24 (no ranker).

READ THIS BEFORE COMPARING THE NUMBERS: ref2015 RELAXES the pose while ranking it, so its
top-1 DockQ is measured on a relaxed structure, while consensus/oracle/random are measured on
the raw pose. That favours ref2015 -- it gets a refinement step ours does not. It is still the
fair comparison, because a relaxed pose is what a RAPiDock user actually receives.
max_iter is RAPiDock's own code default (20); their README suggests 1000 for production.

Usage: python scripts/ref2015_table.py
"""
from __future__ import annotations

import json
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
LEVELS = [("acceptable", 0.23), ("medium", 0.49), ("high", 0.80)]
ARMS = [("rapidock_og", "RAPiDock original"),
        ("hybridock_ft", "ours (fork tree)"),
        ("hybridock_ft_shipped", "ours (shipped path)")]


def consensus_order(d: Path) -> list[int]:
    """Same rule as scoring/consensus.py, on the raw poses."""
    import numpy as np
    files = sorted(d.glob("rank*.pdb"), key=lambda p: int(re.search(r"\d+", p.name).group()))
    idx, mats = [], []
    for p in files:
        xs = [(float(l[30:38]), float(l[38:46]), float(l[46:54]))
              for l in p.read_text().splitlines()
              if l.startswith("ATOM") and l[76:78].strip() != "H"]
        if xs:
            idx.append(int(re.search(r"\d+", p.name).group()))
            mats.append(np.asarray(xs))
    if len(mats) < 3 or any(m.shape != mats[0].shape for m in mats):
        return idx
    x = np.stack(mats)
    diff = x[:, None, :, :] - x[None, :, :, :]
    rms = np.sqrt((diff ** 2).sum(-1).mean(-1))
    return [idx[i] for i in np.argsort(rms.sum(1))]


def cells(vals: list[float]) -> str:
    return " ".join(f"{100 * sum(1 for v in vals if v >= t) / len(vals):6.1f}%" for _, t in LEVELS)


def main() -> None:
    dq: dict[str, dict[str, list[float]]] = {}
    for line in (ROOT / "logs/dockq_recentset.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            dq.setdefault(r["arm"], {})[r["name"]] = r["dockq"]

    print("=" * 92)
    print("RANKER COMPARISON - RAPiDock's ref2015 vs our consensus, same poses, 345 leak-free")
    print("DockQ(capri_peptide). ref2015 top-1 is scored on the RELAXED pose (see header).")
    print("=" * 92)

    for arm, label in ARMS:
        f = ROOT / f"logs/ref2015_rank_{arm}.jsonl"
        if not f.exists():
            print(f"\n{label}: ref2015 not run yet")
            continue
        ref: dict[str, dict] = {}
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                ref[r["name"]] = r
        names = sorted(n for n in ref if n in dq.get(arm, {}))
        rows: dict[str, list[float]] = {k: [] for k in
                                        ("oracle", "ref1", "ref5", "ref1raw", "ref5raw",
                                         "con1", "con5", "random")}
        for n in names:
            raw = dq[arm][n]
            ok = [v for v in raw if v == v]
            rel = [v for v in ref[n]["dockq_relaxed"] if v == v]
            if not ok or not rel:
                continue
            order = consensus_order(ROOT / "runs/recentset" / arm / n)
            picks = [raw[i - 1] for i in order if i - 1 < len(raw) and raw[i - 1] == raw[i - 1]]
            if not picks:
                continue
            rows["oracle"].append(max(ok))
            rows["random"].append(sum(ok) / len(ok))
            rows["ref1"].append(rel[0])
            rows["ref5"].append(max(rel))          # dockq_relaxed holds ref2015's top-5
            # ref2015's ORDER applied to the RAW poses. This is the like-for-like ranking
            # comparison: same unrelaxed structures as consensus, only the selection differs.
            # Needed because the relaxed rows above beat the oracle ceiling, which is only
            # possible if FastRelax is IMPROVING poses rather than merely ordering them.
            ro = [raw[i] for i in ref[n]["order"] if i < len(raw) and raw[i] == raw[i]]
            if ro:
                rows["ref1raw"].append(ro[0])
                rows["ref5raw"].append(max(ro[:5]))
            rows["con1"].append(picks[0])
            rows["con5"].append(max(picks[:5]))

        n = len(rows["oracle"])
        print(f"\n{label}  (n={n}, max_iter={ref[names[0]]['max_iter'] if names else '?'})")
        print(f"  {'selection':22s} {'medDockQ':>8s} " + " ".join(f"{l[:4]:>7s}" for l, _ in LEVELS))
        for key, lbl in [("oracle", "oracle best-of-24"), ("ref5", "ref2015 top-5 relaxed"),
                         ("ref5raw", "ref2015 top-5 RAW"), ("con5", "consensus top-5"),
                         ("ref1", "ref2015 top-1 relaxed"), ("ref1raw", "ref2015 top-1 RAW"),
                         ("con1", "consensus top-1"), ("random", "no ranker")]:
            v = rows[key]
            if v:
                print(f"  {lbl:22s} {st.median(v):8.3f} {cells(v)}")
        for key, what in [("ref1", "vs ref2015 relaxed (ref2015 also gets refinement)"),
                          ("ref1raw", "vs ref2015 RAW  (pure ranking, like-for-like)")]:
            if rows[key] and rows["con1"] and len(rows[key]) == len(rows["con1"]):
                better = sum(1 for a, b in zip(rows[key], rows["con1"]) if b > a + 0.02)
                worse = sum(1 for a, b in zip(rows[key], rows["con1"]) if b < a - 0.02)
                print(f"  consensus {what}: better {better}, worse {worse}, "
                      f"tied {len(rows[key]) - better - worse}")


if __name__ == "__main__":
    main()
