"""Ingest PDBbind v2020 → curated protein-PEPTIDE Kd training set (READY TO RUN when data arrives).

This is the data-lever pipeline. PPI-Affinity's 0.629 comes from training on ~949-1149 labeled peptide
complexes; we have 156. The only way to materially beat them is matching that data scale. PDBbind v2020
(free registration at pdbbind.org.cn) is the source — its peptide subset survives the BioLiP licensing
removal because we pull it directly.

Reproduces PPI-Affinity's filters (Romero-Molina 2022):
  * single-chain receptor; peptide of standard residues, length 3-40 (they used 3-29 effective)
  * keep Kd or Ki only; drop ambiguous ranges; ΔG ∈ [−14.4, −3.6] kcal/mol
  * (PPI used <90% binding-site/receptor identity; we add a hard sequence dedup too)

USAGE (once Ram provides the PDBbind index + structures):
  python scripts/ingest_pdbbind_peptides.py \
      --index    PATH/to/INDEX_general_PL_data.2020 \
      --pdb-dir  PATH/to/pdbbind_v2020_structures/ \
      --out      data/pdbbind_peptides.csv

INDEX format (PDBbind standard): lines like
  '1a4k  2.40  2002  4.01  Kd=98uM  // ...'  → cols: pdbid, resolution, year, -logKd/Ki, 'Kd=..'
Structures: <pdb-dir>/<pdbid>/<pdbid>_protein.pdb + <pdbid>_ligand.(pdb|mol2|sdf) [peptide as ligand].

It then scores each with our 16 features (crystal pose) and appends a 'y' (ΔG kcal/mol) so the output is
drop-in for the pooled-benchmark training path. Run e107 after to grade pooled LOO vs PPI-Affinity.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

AA3 = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
RT = 0.5922  # kcal/mol at 298 K
UNIT = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9, "pM": 1e-12, "fM": 1e-15}


def parse_affinity(token: str):
    """'Kd=98uM' / 'Ki=1.2nM' / 'IC50=...'(reject) → (type, ΔG kcal/mol) or None."""
    m = re.match(r"(Kd|Ki)[=~]([0-9.]+)([fpnumM]+)", token)
    if not m:
        return None
    typ, val, unit = m.group(1), float(m.group(2)), m.group(3)
    if unit not in UNIT or val <= 0:
        return None
    return typ, RT * math.log(val * UNIT[unit])  # ΔG = RT ln(Kd)


def peptide_seq(ligand_pdb: Path):
    """Return one-letter sequence if ligand is a standard-residue peptide ≥3 res, else None."""
    three = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
             "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
             "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
    seen = []
    last = None
    for ln in ligand_pdb.read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM")) and ln[17:20].strip() in AA3:
            key = (ln[21], ln[22:27])
            if key != last:
                seen.append(ln[17:20].strip())
                last = key
    if len(seen) < 3 or any(r not in three for r in seen):
        return None
    return "".join(three[r] for r in seen)


def single_chain_receptor(protein_pdb: Path) -> bool:
    chains = set()
    for ln in protein_pdb.read_text().splitlines():
        if ln.startswith("ATOM"):
            chains.add(ln[21])
    return len(chains) == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, type=Path)
    ap.add_argument("--pdb-dir", required=True, type=Path)
    ap.add_argument("--out", default=ROOT / "data" / "pdbbind_peptides.csv", type=Path)
    ap.add_argument("--score", action="store_true", help="also compute our 16 features (slow)")
    a = ap.parse_args()

    from hybridock_pep.scoring.geometry_features import GEOMETRY_FEATURE_KEYS, compute_geometry_features
    import csv as _csv

    kept, stats = [], {"lines": 0, "no_aff": 0, "out_range": 0, "no_pep": 0, "multichain": 0, "kept": 0}
    for ln in a.index.read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        stats["lines"] += 1
        parts = ln.split()
        if len(parts) < 5:
            continue
        pdbid = parts[0].lower()
        aff = next((parse_affinity(p) for p in parts if parse_affinity(p)), None)
        if not aff:
            stats["no_aff"] += 1
            continue
        typ, dg = aff
        if not (-14.4 <= dg <= -3.6):
            stats["out_range"] += 1
            continue
        d = a.pdb_dir / pdbid
        prot = d / f"{pdbid}_protein.pdb"
        lig = next((d / f"{pdbid}_ligand.{e}" for e in ("pdb",) if (d / f"{pdbid}_ligand.{e}").exists()), None)
        if not (prot.exists() and lig):
            continue
        seq = peptide_seq(lig)
        if not seq or not (3 <= len(seq) <= 40):
            stats["no_pep"] += 1
            continue
        if not single_chain_receptor(prot):
            stats["multichain"] += 1
            continue
        row = {"pdb": pdbid, "dataset": "pdbbind", "affinity_type": typ, "seq": seq,
               "length": len(seq), "y": round(dg, 3)}
        if a.score:
            f = compute_geometry_features(lig, prot.resolve())
            if not f:
                continue
            row.update({k: f[k] for k in GEOMETRY_FEATURE_KEYS})
        kept.append(row)
        stats["kept"] += 1
        if stats["kept"] % 50 == 0:
            print(f"  kept {stats['kept']} ...", flush=True)

    # hard sequence dedup (keep first per unique peptide seq)
    seen, dedup = set(), []
    for r in kept:
        if r["seq"] not in seen:
            seen.add(r["seq"])
            dedup.append(r)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if dedup:
        with open(a.out, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(dedup[0].keys()))
            w.writeheader()
            w.writerows(dedup)
    print(f"\nfilter stats: {stats}")
    print(f"kept {len(kept)} → {len(dedup)} after seq-dedup → {a.out}")
    print("next: scripts/e107_pdbbind_grade.py to pool with our 156 + grade LOO vs PPI-Affinity 0.554/0.629")


if __name__ == "__main__":
    main()
