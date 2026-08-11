#!/usr/bin/env python3
"""
build_benchmark_expanded.py — Grow benchmark_real.csv (52) toward a target size
(e.g. 150) by adding held-out complexes, correctly excluded against EVERY dataset
currently in use (not the stale PepPC-era exclusion list build_benchmark150.py used).

Exclusion set = union of complex_name/name across:
  data/adapter_train.csv, adapter_val.csv (original, cliff v1)
  data/adapter_train_dedup.csv, adapter_val_dedup.csv (cliffv3 training)
  data/adapter_train_mixed.csv, adapter_val_mixed.csv (cliffmixed training)
  data/benchmark_real.csv (kept — these 52 are carried into the output unchanged)

Candidate pool, in priority order:
  1. datasets/RefPepDB-RecentSet/  (same distribution as the original 52 — no domain bias)
  2. datasets/propedia_formatted/  (in-distribution for cliffv3's training — tagged
     'source=propedia' in the output so results can be reported/filtered separately)

Output: data/benchmark_expanded.csv — same schema as benchmark_real.csv plus a
'source' column (refpepdb_recentset | propedia).
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parent.parent
REFPEPDB_DIR = ROOT / "datasets" / "RefPepDB-RecentSet"
PROPEDIA_DIR = ROOT / "datasets" / "propedia_formatted"
BENCH_REAL = ROOT / "data" / "benchmark_real.csv"
EXCLUDE_CSVS = [
    ROOT / "data" / "adapter_train.csv", ROOT / "data" / "adapter_val.csv",
    ROOT / "data" / "adapter_train_dedup.csv", ROOT / "data" / "adapter_val_dedup.csv",
    ROOT / "data" / "adapter_train_mixed.csv", ROOT / "data" / "adapter_val_mixed.csv",
]
OUT_CSV = ROOT / "data" / "benchmark_expanded.csv"

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def get_seq_from_pdb(pdb_path: pathlib.Path) -> str | None:
    seen: dict[tuple[str, str], str] = {}
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM"):
                    res3 = line[17:20].strip()
                    chain = line[21]
                    res_seq = line[22:26].strip()
                    aa1 = AA3TO1.get(res3)
                    if aa1 and (chain, res_seq) not in seen:
                        seen[(chain, res_seq)] = aa1
    except OSError:
        return None
    if not seen:
        return None
    try:
        ordered = sorted(seen.items(), key=lambda kv: (kv[0][0], int(kv[0][1])))
    except ValueError:
        ordered = sorted(seen.items(), key=lambda kv: kv[0])
    return "".join(aa1 for _, aa1 in ordered)


def length_bucket(pep_len: int) -> str:
    if pep_len <= 8:
        return "short"
    elif pep_len <= 12:
        return "medium"
    elif pep_len <= 19:
        return "long"
    return "very_long"


def classify_ss(pdb_path: pathlib.Path, seq: str) -> tuple[str, float, float, float]:
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import PPBuilder

    seq_upper = seq.upper()
    pro_frac = seq_upper.count("P") / max(len(seq_upper), 1)
    if pro_frac >= 0.30:
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
        if -100 <= phi_d <= -30 and -70 <= psi_d <= -10:
            n_H += 1
        elif phi_d <= -50 and (psi_d >= 80 or psi_d <= -150):
            n_E += 1
        elif -90 <= phi_d <= -50 and 100 <= psi_d <= 180:
            n_P += 1

    frac_H, frac_E, frac_P = n_H / n, n_E / n, n_P / n
    if frac_H >= 0.40:
        return "HELIX", frac_H, frac_E, frac_P
    if frac_E >= 0.50 and pro_frac < 0.25:
        return "SHEET", frac_H, frac_E, frac_P
    return "UNUSUAL", frac_H, frac_E, frac_P


def load_excluded() -> set[str]:
    excluded: set[str] = set()
    for p in EXCLUDE_CSVS:
        for r in csv.DictReader(open(p)):
            excluded.add(r["complex_name"])
    return excluded


def scan_dir(pool_dir: pathlib.Path, excluded: set[str], source_tag: str) -> list[dict]:
    rows = []
    if not pool_dir.exists():
        return rows
    for d in sorted(pool_dir.iterdir()):
        if not d.is_dir() or d.name in excluded:
            continue
        pep_pdb = d / f"{d.name}_peptide.pdb"
        rec_pdb = d / f"{d.name}_protein_pocket.pdb"
        if not pep_pdb.exists() or not rec_pdb.exists():
            continue
        seq = get_seq_from_pdb(pep_pdb)
        if not seq or len(seq) < 4:
            continue
        pep_len = len(seq)
        lb = length_bucket(pep_len)
        ss_class, fH, fE, fP = classify_ss(pep_pdb, seq)
        rows.append({
            "name": d.name, "receptor": str(rec_pdb), "peptide_pdb": str(pep_pdb),
            "seq": seq, "pep_len": pep_len, "ss_class": ss_class,
            "length_bucket": lb, "frac_H": round(fH, 3), "frac_E": round(fE, 3),
            "frac_P": round(fP, 3), "source": source_tag,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=pathlib.Path, default=OUT_CSV)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    excluded = load_excluded()
    print(f"[expand] excluded (all current train/val sets): {len(excluded)}")

    existing = list(csv.DictReader(open(BENCH_REAL)))
    for r in existing:
        r["source"] = "refpepdb_recentset"  # the original 52 are all RefPepDB
    existing_names = {r["name"] for r in existing}
    excluded |= existing_names
    print(f"[expand] keeping existing benchmark_real.csv: {len(existing)}")

    need = args.target - len(existing)
    if need <= 0:
        print("[expand] target already met by existing benchmark, nothing to add.")
        rows = existing
    else:
        print(f"[expand] need {need} more complexes")
        refpepdb_pool = scan_dir(REFPEPDB_DIR, excluded, "refpepdb_recentset")
        rng.shuffle(refpepdb_pool)
        print(f"[expand] RefPepDB-RecentSet untouched pool: {len(refpepdb_pool)}")

        take_refpepdb = refpepdb_pool[:need]
        still_need = need - len(take_refpepdb)
        print(f"[expand] taking {len(take_refpepdb)} from RefPepDB-RecentSet; still need {still_need}")

        take_propedia = []
        if still_need > 0:
            propedia_pool = scan_dir(PROPEDIA_DIR, excluded, "propedia")
            rng.shuffle(propedia_pool)
            print(f"[expand] Propedia untouched pool: {len(propedia_pool)}")
            take_propedia = propedia_pool[:still_need]
            print(f"[expand] taking {len(take_propedia)} from Propedia")

        rows = existing + take_refpepdb + take_propedia

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", "receptor", "peptide_pdb", "seq", "pep_len", "ss_class",
                  "length_bucket", "frac_H", "frac_E", "frac_P", "source"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\n[expand] written {len(rows)} complexes -> {args.out}")
    by_source = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    print(f"[expand] by source: {by_source}")
    by_bucket = {}
    for r in rows:
        by_bucket[r["length_bucket"]] = by_bucket.get(r["length_bucket"], 0) + 1
    print(f"[expand] by length bucket: {by_bucket}")


if __name__ == "__main__":
    main()
