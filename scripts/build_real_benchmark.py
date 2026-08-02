#!/usr/bin/env python3
"""build_real_benchmark.py — Stratified REAL-complex benchmark from RefPepDB-RecentSet.

Replaces the broken benchmark30 (100% peppcf fragments). RecentSet = RAPiDock's
OFFICIAL recent-holdout: pretrained RAPiDock never trained on it, so it is a VALID
held-out set for diagnosing the PRETRAINED model. (It IS leaked for our finetuned
models — those trained on all 523 — so use a Propedia/recent slice for retrain
head-to-heads; this file is for the pretrained baseline + weakness map.)

Stratified across SS class (HELIX/SHEET/UNUSUAL) x length bucket, balanced per cell.
Absolute paths (CLAUDE.md: absolute-only across the RAPiDock boundary). Direct docking
RMSD is computed downstream by nsweep_coverage.py.

Usage:
    python scripts/build_real_benchmark.py --per-cell 5 --dry-run
    python scripts/build_real_benchmark.py --per-cell 5 --out data/benchmark_real.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_benchmark150 import classify_ss, get_seq_from_pdb, length_bucket  # noqa: E402

RECENTSET = ROOT / "datasets" / "RefPepDB-RecentSet"
SS_CLASSES = ["HELIX", "SHEET", "UNUSUAL"]
LEN_BUCKETS = ["short", "medium", "long", "very_long"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-cell", type=int, default=5, help="max complexes per SS x length cell")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "benchmark_real.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    dirs = sorted(d for d in RECENTSET.iterdir() if d.is_dir())
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    scanned = 0
    for d in dirs:
        pep = d / f"{d.name}_peptide.pdb"
        rec = d / f"{d.name}_protein_pocket.pdb"
        if not pep.exists() or not rec.exists():
            continue
        seq = get_seq_from_pdb(pep)
        if not seq or len(seq) < 4:
            continue
        ss, fH, fE, fP = classify_ss(pep, seq)
        lb = length_bucket(len(seq))
        cells[(ss, lb)].append({
            "name": d.name, "receptor": str(rec.resolve()),
            "peptide_pdb": str(pep.resolve()), "seq": seq, "pep_len": len(seq),
            "ss_class": ss, "length_bucket": lb,
            "frac_H": round(fH, 3), "frac_E": round(fE, 3), "frac_P": round(fP, 3),
        })
        scanned += 1

    print(f"[build_real_benchmark] scanned {scanned} real complexes")
    print("available per cell:")
    for ss in SS_CLASSES:
        for lb in LEN_BUCKETS:
            print(f"  {ss:8s} x {lb:10s}: {len(cells[(ss, lb)]):3d}")

    selected: list[dict] = []
    for ss in SS_CLASSES:
        for lb in LEN_BUCKETS:
            pool = cells[(ss, lb)]
            selected.extend(rng.sample(pool, min(args.per_cell, len(pool))))
    rng.shuffle(selected)
    print(f"\n[build_real_benchmark] selected {len(selected)} complexes")
    for ss in SS_CLASSES:
        n = sum(1 for r in selected if r["ss_class"] == ss)
        print(f"  {ss:8s}: {n}")

    if args.dry_run:
        print("[dry-run] not writing")
        return
    fields = ["name", "receptor", "peptide_pdb", "seq", "pep_len", "ss_class",
              "length_bucket", "frac_H", "frac_E", "frac_P"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(selected)
    print(f"[build_real_benchmark] wrote {len(selected)} -> {args.out}")


if __name__ == "__main__":
    main()
