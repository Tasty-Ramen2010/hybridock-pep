#!/usr/bin/env python3
"""inspect_pocket_boxes.py — verify each training/benchmark complex's pocket
receptor actually covers its crystal peptide, matching RAPiDock's own pocket
data-prep convention.

RAPiDock's inference.py has no --site/--box CLI input at all (confirmed: no such
flags in utils/inference_parsing.py) -- the ONLY thing it ever sees is whatever
receptor PDB it's handed, so "giving it the site" happens entirely at DATA-PREP
time: RAPiDock's own pocket_trunction.py builds a pocket receptor via
NeighborSearch -- every protein residue within `threshold` (default 20A) of ANY
peptide atom, a per-atom union, not a sphere cropped around one center point.
For an elongated peptide this produces a naturally asymmetric pocket, so a
naive pocket-centroid-vs-peptide-centroid distance is NOT a meaningful quality
signal (an earlier version of this script measured exactly that and produced a
false alarm about a "training vs benchmark geometry mismatch" -- Sep 2026,
corrected same day). The metric that actually matters, matching
pocket_trunction.py's own definition of a correctly-built pocket, is COVERAGE:
does every peptide atom have SOME pocket atom within the threshold? This script
reports, per complex, the worst (max over peptide atoms) such distance, and
flags any complex where the pocket crop doesn't actually cover its peptide.

Usage:
    python scripts/inspect_pocket_boxes.py --csv data/fullparam_train_pool.csv
    python scripts/inspect_pocket_boxes.py --csv data/benchmark_expanded.csv \\
        --protein-col receptor --peptide-col peptide_pdb --limit 100
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict


def load_coords(pdb_path: str):
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return coords


def worst_coverage(pocket_coords, pep_coords):
    """Max over peptide atoms of (distance to nearest pocket atom). Small = every
    peptide atom has close pocket coverage. Should be <= the crop's own build
    threshold (20A for pocket_trunction.py's default) if the crop is well-formed."""
    if not pocket_coords or not pep_coords:
        return None
    return max(min(math.dist(pa, qa) for qa in pocket_coords) for pa in pep_coords)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--protein-col", default="protein_description")
    ap.add_argument("--peptide-col", default="peptide_description")
    ap.add_argument("--source-col", default="source")
    ap.add_argument("--threshold", type=float, default=20.0,
                     help="pocket_trunction.py's build threshold (A) to flag against")
    ap.add_argument("--sample", type=int, default=300,
                     help="random sample size (0 = every row)")
    ap.add_argument("--limit", type=int, default=None,
                     help="use the first N rows verbatim instead of sampling")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-complex", action="store_true",
                     help="print every complex's coverage, not just the summary")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    if args.limit:
        rows = rows[: args.limit]
    elif args.sample and args.sample < len(rows):
        random.seed(args.seed)
        rows = random.sample(rows, args.sample)

    coverages = []
    by_source = defaultdict(list)
    bad = []
    n_missing = 0
    for r in rows:
        pocket_path = r[args.protein_col]
        pep_path = r[args.peptide_col]
        try:
            pocket = load_coords(pocket_path)
            pep = load_coords(pep_path)
        except Exception:
            n_missing += 1
            continue
        w = worst_coverage(pocket, pep)
        if w is None:
            n_missing += 1
            continue
        coverages.append(w)
        by_source[r.get(args.source_col, "?")].append(w)
        name = r.get("name") or r.get("complex_name") or pocket_path
        if w > args.threshold:
            bad.append((name, w))
        if args.per_complex:
            flag = "" if w <= args.threshold else "  ** POOR COVERAGE **"
            print(f"{name:20s} worst_peptide_atom_to_pocket_coverage={w:6.2f}A{flag}")

    if not coverages:
        print("No valid complexes found (missing files?)")
        return

    n = len(coverages)
    n_ok = sum(1 for w in coverages if w <= args.threshold)
    print(f"\n=== {args.csv} ===")
    print(f"n={n}  (skipped {n_missing} with missing/unreadable files)")
    print(f"worst_peptide_atom_to_pocket_coverage: mean={sum(coverages)/n:6.2f}A  "
          f"min={min(coverages):6.2f}A  max={max(coverages):6.2f}A")
    print(f"within {args.threshold:.0f}A (well-formed pocket): {n_ok}/{n} ({100*n_ok/n:.1f}%)")
    if len(by_source) > 1:
        print("\nby source:")
        for src, ws in sorted(by_source.items()):
            print(f"  {src:20s} n={len(ws):4d}  mean_coverage={sum(ws)/len(ws):6.2f}A")
    if bad:
        print(f"\n{len(bad)} complexes worse than {args.threshold:.0f}A (pocket doesn't fully cover peptide):")
        for name, w in sorted(bad, key=lambda x: -x[1])[:20]:
            print(f"  {name}: {w:.1f}A")


if __name__ == "__main__":
    main()
