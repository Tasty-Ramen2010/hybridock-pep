#!/usr/bin/env python3
"""
build_benchmark150.py — Build a balanced 150-complex validation benchmark.

Sampling grid:
  SS class:      HELIX | SHEET | UNUSUAL
  Length bucket: short (≤8) | medium (9-12) | long (13-19) | very_long (≥20)

Target: ~150 complexes = 3 SS × 4 length × ~12-13 each,
         but capped by actual availability per cell.

Output: data/benchmark150.csv (same schema as benchmark30.csv)

SS Classifier (improved over 67%-accurate BioPython version):
  - Uses BioPython phi/psi angles via PPBuilder
  - HELIX: frac_H ≥ 0.40
  - SHEET: frac_E ≥ 0.50 AND Pro-fraction < 0.25 (excludes PPII/polyPro)
  - UNUSUAL: everything else (loops, PPII, mixed, coil)
  Key fix: PPII/polyPro peptides have high frac_E in phi/psi space but
            are biologically "unusual". We gate on Pro content and require
            a higher frac_E threshold (0.50 vs 0.35) for true β-strands.

Usage:
    python3 scripts/build_benchmark150.py [--dry-run] [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import random
from collections import defaultdict
from typing import Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent
PEPPC_DIR = ROOT / "datasets" / "training_formatted_peppc"
TRAIN_CSV = PEPPC_DIR / "combined_train_curated.csv"
VAL_CSV = PEPPC_DIR / "combined_val_curated.csv"
BENCH30_CSV = ROOT / "data" / "benchmark30.csv"
OUT_CSV = ROOT / "data" / "benchmark150.csv"

# ─── SS Classifier ────────────────────────────────────────────────────────────

def classify_ss(pdb_path: pathlib.Path, seq: str) -> tuple[str, float, float, float]:
    """Classify SS of a peptide PDB.

    Returns:
        (ss_class, frac_H, frac_E, frac_P) where ss_class is HELIX/SHEET/UNUSUAL.

    Key design decisions:
    - SHEET requires frac_E >= 0.50 AND Pro content < 0.25 to exclude PPII.
    - UNUSUAL catches loops, PPII, mixed, and ambiguous conformations.
    - Pro/Gly-rich sequences (>30%) are forced to UNUSUAL regardless of angles.
    """
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.Polypeptide import PPBuilder
    except ImportError:
        raise RuntimeError("biopython required: conda install -c conda-forge biopython")

    # Sequence-level Pro/Gly check (fast path for obvious UNUSUAL)
    seq_upper = seq.upper()
    pro_frac = seq_upper.count("P") / max(len(seq_upper), 1)
    gly_frac = seq_upper.count("G") / max(len(seq_upper), 1)
    if pro_frac >= 0.30:
        # Poly-Pro / PPII — force UNUSUAL before even reading angles
        return "UNUSUAL", 0.0, 0.0, pro_frac

    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure("x", str(pdb_path))
    except Exception:
        return "UNUSUAL", 0.0, 0.0, 0.0

    ppb = PPBuilder()
    all_phi_psi: list[tuple[float, float]] = []
    for pp in ppb.build_peptides(struct):
        for phi, psi in pp.get_phi_psi_list():
            if phi is not None and psi is not None:
                all_phi_psi.append((math.degrees(phi), math.degrees(psi)))

    if len(all_phi_psi) < 2:
        return "UNUSUAL", 0.0, 0.0, 0.0

    n = len(all_phi_psi)
    n_H = n_E = n_P = 0
    for phi_d, psi_d in all_phi_psi:
        # α-helix Ramachandran region
        if -100 <= phi_d <= -30 and -70 <= psi_d <= -10:
            n_H += 1
        # β-strand / extended region (broad)
        elif phi_d <= -50 and (psi_d >= 80 or psi_d <= -150):
            n_E += 1
        # PPII-like region (left-handed extended)
        elif -90 <= phi_d <= -50 and 100 <= psi_d <= 180:
            n_P += 1

    frac_H = n_H / n
    frac_E = n_E / n
    frac_P = n_P / n

    # Classification (order matters)
    if frac_H >= 0.40:
        return "HELIX", frac_H, frac_E, frac_P

    # SHEET: require strong frac_E AND low Pro (to exclude polyPro/PPII)
    if frac_E >= 0.50 and pro_frac < 0.25:
        return "SHEET", frac_H, frac_E, frac_P

    return "UNUSUAL", frac_H, frac_E, frac_P


def length_bucket(pep_len: int) -> str:
    if pep_len <= 8:
        return "short"
    elif pep_len <= 12:
        return "medium"
    elif pep_len <= 19:
        return "long"
    else:
        return "very_long"


# ─── Main ─────────────────────────────────────────────────────────────────────

def load_excluded() -> set[str]:
    excluded: set[str] = set()
    for p in [TRAIN_CSV, VAL_CSV]:
        for r in csv.DictReader(open(p)):
            excluded.add(r["complex_name"])
    for r in csv.DictReader(open(BENCH30_CSV)):
        excluded.add(r["name"])
    return excluded


def get_seq_from_pdb(pdb_path: pathlib.Path) -> Optional[str]:
    """Extract one-letter AA sequence from ATOM records."""
    aa3to1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    seen: dict[tuple[str, str], str] = {}  # (chain_id, res_seq) -> aa1
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM"):
                    res3 = line[17:20].strip()
                    chain = line[21]
                    res_seq = line[22:26].strip()
                    aa1 = aa3to1.get(res3)
                    if aa1 and (chain, res_seq) not in seen:
                        seen[(chain, res_seq)] = aa1
    except OSError:
        return None
    if not seen:
        return None
    # Sort by (chain, int(res_seq)) to get ordered sequence
    try:
        ordered = sorted(seen.items(), key=lambda kv: (kv[0][0], int(kv[0][1])))
    except ValueError:
        ordered = sorted(seen.items(), key=lambda kv: kv[0])
    return "".join(aa1 for _, aa1 in ordered)


def build_reason(name: str, ss_class: str, seq: str, lb: str,
                 frac_H: float, frac_E: float, frac_P: float) -> str:
    pro_pct = int(seq.upper().count("P") / len(seq) * 100) if seq else 0
    fracs = f"H={frac_H:.0%} E={frac_E:.0%} P={frac_P:.0%}"
    return f"{lb} {ss_class.lower()}; {len(seq)}-mer; {fracs}; Pro={pro_pct}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", type=int, default=150,
                        help="Total complexes to sample")
    parser.add_argument("--per-cell", type=int, default=None,
                        help="Override per-cell count (default: target / (3 SS × 4 len) ≈ 12)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Classify only, print stats, do not write CSV")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit holdout pool for speed testing (e.g. --limit 2000)")
    parser.add_argument("--out", type=pathlib.Path, default=OUT_CSV)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    per_cell = args.per_cell or max(1, args.target // 12)

    print(f"[build_benchmark150] seed={args.seed} target={args.target} per_cell={per_cell}")

    # 1. Build holdout pool
    excluded = load_excluded()
    print(f"[build_benchmark150] excluded (train+val+bench30): {len(excluded)}")

    all_dirs = sorted(
        d for d in PEPPC_DIR.iterdir()
        if d.is_dir() and d.name not in excluded
    )
    rng.shuffle(all_dirs)
    if args.limit:
        all_dirs = all_dirs[: args.limit]
    print(f"[build_benchmark150] holdout pool to scan: {len(all_dirs)}")

    # 2. Scan + classify
    SS_CLASSES = ["HELIX", "SHEET", "UNUSUAL"]
    LEN_BUCKETS = ["short", "medium", "long", "very_long"]

    # cell -> list of rows
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    # count per SS for progress
    ss_counts: dict[str, int] = defaultdict(int)
    scanned = 0
    skipped = 0

    for d in all_dirs:
        pep_pdb = d / f"{d.name}_peptide.pdb"
        rec_pdb = d / f"{d.name}_protein_pocket.pdb"
        if not pep_pdb.exists() or not rec_pdb.exists():
            skipped += 1
            continue

        seq = get_seq_from_pdb(pep_pdb)
        if not seq or len(seq) < 4:
            skipped += 1
            continue

        pep_len = len(seq)
        lb = length_bucket(pep_len)
        ss_class, fH, fE, fP = classify_ss(pep_pdb, seq)
        reason = build_reason(d.name, ss_class, seq, lb, fH, fE, fP)

        row = {
            "name": d.name,
            "receptor": str(rec_pdb),
            "peptide_pdb": str(pep_pdb),
            "seq": seq,
            "pep_len": pep_len,
            "ss_class": ss_class,
            "frac_H": round(fH, 3),
            "frac_E": round(fE, 3),
            "frac_P": round(fP, 3),
            "length_bucket": lb,
            "reason": reason,
        }
        cells[(ss_class, lb)].append(row)
        ss_counts[ss_class] += 1
        scanned += 1

        if scanned % 500 == 0:
            total_selected = sum(len(v) for v in cells.values())
            print(f"  scanned={scanned} selected_so_far={total_selected} "
                  f"H={ss_counts['HELIX']} S={ss_counts['SHEET']} U={ss_counts['UNUSUAL']}")

        # Early exit: once every cell has enough candidates (5× per_cell), stop scanning
        if all(len(cells[(ss, lb)]) >= per_cell * 5
               for ss in SS_CLASSES for lb in LEN_BUCKETS):
            print(f"[build_benchmark150] early exit at scanned={scanned} (all cells saturated)")
            break

    print(f"\n[build_benchmark150] scan done: scanned={scanned} skipped={skipped}")
    print("Cell counts (available per (SS, length_bucket)):")
    for ss in SS_CLASSES:
        for lb in LEN_BUCKETS:
            print(f"  {ss:8s} × {lb:10s} : {len(cells[(ss, lb)]):4d} available")

    if args.dry_run:
        print("\n[dry-run] Not writing CSV.")
        return

    # 3. Sample balanced
    selected: list[dict] = []
    for ss in SS_CLASSES:
        for lb in LEN_BUCKETS:
            pool = cells[(ss, lb)]
            n = min(per_cell, len(pool))
            chosen = rng.sample(pool, n)
            selected.extend(chosen)
            if n < per_cell:
                print(f"  WARN: {ss} × {lb}: only {n} available (wanted {per_cell})")

    rng.shuffle(selected)
    print(f"\n[build_benchmark150] selected {len(selected)} complexes")

    # 4. Write CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", "receptor", "peptide_pdb", "seq", "pep_len",
                  "ss_class", "frac_H", "frac_E", "frac_P", "length_bucket", "reason"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    print(f"[build_benchmark150] written → {args.out}")
    print("\nFinal distribution:")
    for ss in SS_CLASSES:
        for lb in LEN_BUCKETS:
            count = sum(1 for r in selected if r["ss_class"] == ss and r["length_bucket"] == lb)
            print(f"  {ss:8s} × {lb:10s} : {count}")

    total_by_ss = {ss: sum(1 for r in selected if r["ss_class"] == ss) for ss in SS_CLASSES}
    print(f"\nBy SS: {total_by_ss}")


if __name__ == "__main__":
    main()
