#!/usr/bin/env python
"""Rebuild the grid using the AF3 binder model wherever our ESMFold model is demonstrably wrong.

WHY.  Every grid cell in a column was docked against one binder model, so a wrong model
poisons all 18 cells of that column.  Four of the 18 columns are wrong: Boltz-2 and AlphaFold 3
-- two independent predictors, both given the IDENTICAL sequence from the paper -- agree with
each other on n3, n7, pc21 and pc26 (1.3, 2.4, 1.6, 0.9 A) and both disagree with our ESMFold
model (16.9, 14.6, 8.7, 5.7 A).  When two independent folders agree against a third, the third
is the outlier.

This is NOT a sequence error.  A wrong sequence would make every predictor fold the same wrong
protein and agree with each other; what we see is the opposite.

The selection criterion is inter-predictor agreement, fixed BEFORE looking at any score, so
swapping these four columns is a structural correction and not cherry-picking on outcome.  All
three grids are reported side by side so the effect of the swap is visible.

Usage: coventry_fix_broken_columns.py
Writes: logs/coventry_refine_fixed.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
#: columns where ESMFold is the outlier (Boltz and AF3 agree against it)
BROKEN = {"n3", "n7", "pc21", "pc26"}


def load(p: Path) -> dict:
    return {json.loads(l)["name"]: json.loads(l)
            for l in p.read_text().splitlines() if l.strip()}


def main() -> None:
    esm = load(ROOT / "logs/coventry_refine.jsonl")
    af3 = load(ROOT / "logs/coventry_refine_hybridock_af3.jsonl")
    out = ROOT / "logs/coventry_refine_fixed.jsonl"
    n_swap = 0
    with out.open("w") as fh:
        for name, row in esm.items():
            binder = name.split("__")[1][:-1]
            if binder in BROKEN and name in af3:
                row = dict(af3[name])
                row["receptor_model"] = "af3"
                n_swap += 1
            else:
                row = dict(row)
                row.setdefault("receptor_model", "esmfold")
            fh.write(json.dumps(row) + "\n")
    print(f"wrote {out}: {n_swap} cells swapped to AF3 models "
          f"({len(BROKEN)} columns), {len(esm) - n_swap} kept on ESMFold")


if __name__ == "__main__":
    main()
