#!/usr/bin/env python
"""Per-pose Ca RMSD cache for the similarity-stratified bench. Same shape as the
balanced-bench cache so the plotting code is shared."""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
ROOT = Path("/home/igem/unknown_software"); RUNS = ROOT / "runs/simbench"

def ca(p):
    seen, xyz = set(), []
    for l in Path(p).read_text(errors="ignore").splitlines():
        if not l.startswith("ATOM") or l[12:16].strip() != "CA":
            continue
        k = (l[21], l[22:27])
        if k in seen:
            continue
        seen.add(k); xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return np.array(xyz)

def rmsd(A, B):
    n = min(len(A), len(B))
    return float(np.sqrt(((A[:n] - B[:n]) ** 2).sum(1).mean())) if n >= 3 else float("nan")

def main() -> None:
    rows = list(csv.DictReader(open(ROOT / "data/bench_similarity.csv")))
    arms = [d.name for d in RUNS.iterdir() if d.is_dir()] if RUNS.is_dir() else []
    out = {}
    for arm in arms:
        d = {}
        for r in rows:
            p = RUNS / arm / r["name"]
            if not p.is_dir():
                continue
            fs = sorted(p.glob("rank*.pdb"), key=lambda x: int(x.stem[4:]))
            if not fs:
                continue
            tru = ca(r["peptide_pdb"])
            if len(tru) < 3:
                continue
            d[r["name"]] = [rmsd(ca(f), tru) for f in fs]
        if d:
            out[arm] = d
            print(f"  {arm}: {len(d)}/{len(rows)}")
    np.savez(ROOT / "logs/simbench_cache.npz",
             **{k: np.array(v, dtype=object) for k, v in out.items()})
    print(f"wrote logs/simbench_cache.npz with {len(out)} arms")

if __name__ == "__main__":
    main()
