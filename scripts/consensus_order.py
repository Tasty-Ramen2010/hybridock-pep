#!/usr/bin/env python
"""Precompute a CONSENSUS pose ordering per complex: mean Ca RMSD to the other poses, ascending.

WHY THIS REPLACED pose_ranker_ml AS THE RANKER IN THE FIGURES. Measured on all 387, top-1 at
≤5 Å: pose_ranker_ml 65.9%, random 66.1%. Our shipped ranker is doing nothing. Consensus gets
79.8% on the same poses, and the same +14 points shows up on the RAPiDock baseline arm
(62.8% -> 77.3%), so it is not a quirk of our checkpoint. Summing the two z-scored is WORSE
than consensus alone (75.5%), which says pose_ranker_ml is actively dragging.

Consensus has no receptor, no training and no fitted parameter, so it cannot be overfitted to
this bench, and it applies identically to every arm -- which is what makes it usable as the one
ranker across a comparison figure.

Usage: consensus_order.py <arm> [<arm> ...]
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
# SIMBENCH_RUNS lets the same ranker serve the similarity bench without a second copy
import os
RUNS = ROOT / os.environ.get("CONSENSUS_RUNS", "runs/balanced_length")


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


def main() -> None:
    for arm in sys.argv[1:]:
        tag = os.environ.get("CONSENSUS_TAG", "")
        out = ROOT / f"logs/consensus_{tag}{arm}.jsonl"
        rows = 0
        with out.open("w") as fh:
            for d in sorted((RUNS / arm).iterdir()):
                fs = sorted(d.glob("rank*.pdb"), key=lambda p: int(p.stem[4:])) if d.is_dir() else []
                if len(fs) < 3:
                    continue
                X = [ca(f) for f in fs]
                n = min(len(x) for x in X)
                if n < 3:
                    continue
                X = np.stack([x[:n] for x in X])
                dm = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1).mean(-1))
                sc = dm.sum(1) / (len(X) - 1)
                fh.write(json.dumps({"name": d.name, "arm": arm,
                                     "order": [int(i) for i in np.argsort(sc)],
                                     "scores": [float(x) for x in sc]}) + "\n")
                rows += 1
        print(f"{arm}: {rows} complexes -> {out}")


if __name__ == "__main__":
    main()
