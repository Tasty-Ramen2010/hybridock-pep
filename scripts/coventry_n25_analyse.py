#!/usr/bin/env python
"""Did 25 diffusion samples buy anything, and how big is pose noise next to the real signal?

THE BIAS THIS AVOIDS. The minimum of 25 draws is lower than the minimum of 1 draw even when the
draws are pure noise, so "best-of-25 beat best-of-1" proves nothing on its own, and comparing an
escalated cell against a non-escalated one is rigged. Every comparison here is therefore made
between cells that got the SAME number of samples: within each failing row, the cognate and the
three false positives currently beating it all received 25, so asking whether the cognate
overtakes them at k=25 is a fair question with the max-statistic bias applying equally to all
four. k is drawn at random many times rather than taking the first k, so no single lucky
trajectory decides the answer.

THE PRE-REGISTERED PREDICTION, from coventry_disagreement.py:

  if SAMPLING is the deficit  the cognate improves MORE than its false positives with more
                              samples, because a real binding mode exists to be found for the
                              cognate and does not for the others, and some rows flip.
  if SCORING is the deficit   all four improve together, the ordering is unchanged, and no row
                              flips no matter how many samples are drawn.

AND THE NUMBER THAT MATTERS EITHER WAY. Twenty-five independent poses of the same pair, scored
by the same function, give the first direct estimate of POSE NOISE: the spread of interface
energy attributable purely to which structure the generator happened to produce. Set that against
the spread between different pairs after cancellation -- the chemistry signal we are trying to
read -- and the ratio says whether pose precision is the binding constraint. If noise exceeds
signal, no scoring function can rank these cells, and the effort belongs in generation.

Usage: coventry_n25_analyse.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "logs/coventry_boltz_n25.jsonl"
CELLS = ROOT / "data/coventry_disagree_cells.json"
FAIL_CAP = 50.0
DRAWS = 400


def main() -> None:
    rng = np.random.default_rng(0)

    by_cell: dict[str, list[float]] = defaultdict(list)
    meta: dict[str, dict] = {}
    for line in OUT.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        v = r.get("best_iface")
        if v is None:
            continue
        by_cell[r["name"]].append(min(float(v), FAIL_CAP))
        meta[r["name"]] = r
    if not by_cell:
        print("no scored samples yet")
        return

    cells = json.loads(CELLS.read_text())
    rows: dict[str, list[dict]] = defaultdict(list)
    for c in cells:
        rows[c["peptide"]].append(c)

    print(f"{len(by_cell)} cells scored, "
          f"{st.mean(len(v) for v in by_cell.values()):.1f} samples each\n")

    print("POSE NOISE.  Spread of ref2015 interface energy across 25 poses of the SAME pair.\n")
    print(f"  {'cell':<16}{'role':<16}{'n':>4}{'best':>9}{'median':>9}{'sd':>8}{'range':>9}")
    sds, bests = [], {}
    for name in sorted(by_cell):
        v = np.array(by_cell[name])
        bests[name] = float(v.min())
        sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        sds.append(sd)
        print(f"  {name:<16}{meta[name].get('role', ''):<16}{len(v):>4}{v.min():>9.1f}"
              f"{np.median(v):>9.1f}{sd:>8.1f}{v.max() - v.min():>9.1f}")

    print(f"\n  median within-pair pose noise (sd): {np.nanmedian(sds):.1f} REU")
    print(f"  median within-pair range:            "
          f"{np.median([max(v) - min(v) for v in by_cell.values()]):.1f} REU")

    # the signal these cells must be ranked by, on the same scale
    from hybridock_pep.scoring.cancellation import median_polish
    ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
             "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
    cof = {}
    for line in (ROOT / "logs/coventry_boltz_refine.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            v = r.get("best_iface")
            cof[(r["peptide"], r["binder"])] = FAIL_CAP if (v is None or v > FAIL_CAP) else v
    G = np.array([[cof[(p, b)] for b in ORDER] for p in ORDER])
    R = median_polish(G)[0]
    sig = float(R.std())
    print(f"  chemistry signal after cancellation (sd of the interaction term): {sig:.1f} REU")
    print(f"  -> pose noise is {np.nanmedian(sds) / sig:.2f}x the signal it must not drown")

    print("\nTHE PRE-REGISTERED TEST.  Within each failing row, the cognate and the three false")
    print("positives beating it all have 25 samples, so the max-statistic bias is equal across")
    print("all four. Cognate rank among those four, best-of-k, averaged over 400 random draws.\n")
    print(f"  {'row':<8}{'k=1':>8}{'k=5':>8}{'k=25':>8}   {'delta':>7}   competitors")
    flips = 0
    for pep in sorted(rows, key=lambda p: [c["peptide"] for c in cells].index(p)):
        group = [c for c in rows[pep] if f"{c['peptide']}__{c['binder']}" in by_cell]
        cog = [c for c in group if c["role"] == "cognate"]
        if not cog or len(group) < 2:
            continue
        names = [f"{c['peptide']}__{c['binder']}" for c in group]
        ci = names.index(f"{pep}__{pep}")
        got = {}
        for k in (1, 5, 25):
            r_acc = []
            for _ in range(DRAWS if k < 25 else 1):
                vals = [min(rng.choice(by_cell[n], size=min(k, len(by_cell[n])), replace=False))
                        for n in names]
                r_acc.append(sorted(range(len(vals)), key=lambda t: vals[t]).index(ci) + 1)
            got[k] = float(np.mean(r_acc))
        flips += got[25] < got[1] - 0.5
        print(f"  {pep:<8}{got[1]:>8.2f}{got[5]:>8.2f}{got[25]:>8.2f}   "
              f"{got[25] - got[1]:>+7.2f}   "
              f"{', '.join(c['binder'] for c in group if c['role'] != 'cognate')}")

    print(f"\n  rows where the cognate improved by more than half a place: {flips}")

    cg = [n for n in by_cell if meta[n]["cognate"] == 1]
    fp = [n for n in by_cell if meta[n]["cognate"] == 0]

    def gain(names):
        out = []
        for n in names:
            v = np.array(by_cell[n])
            out.append(float(np.mean([v[rng.integers(len(v))] for _ in range(DRAWS)]) - v.min()))
        return out

    gc, gf = gain(cg), gain(fp)
    print(f"\n  best-of-25 improves a COGNATE by        {st.mean(gc):6.1f} REU (n={len(cg)})")
    print(f"  best-of-25 improves a FALSE POSITIVE by {st.mean(gf):6.1f} REU (n={len(fp)})")
    print("  if sampling were the deficit the first number would be the larger one")


if __name__ == "__main__":
    main()
