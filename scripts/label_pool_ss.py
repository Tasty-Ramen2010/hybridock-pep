#!/usr/bin/env python3
"""Label every training-pool complex with peptide secondary structure + length bucket.

The training pool carries pep_len but NO secondary-structure column, so it is
impossible to balance the data by SS without computing it first. That matters
because the measured failure modes are exactly SHEET and LONG/VERY-LONG peptides
(Sep-10 bench: 1.81A at <=8mer -> 9.79A at 17+mer), while the pool is 41% 8-mers.

Classification uses the SAME three backbone conformers RAPiDock itself defines
(utils/inference_parsing.py --conformation_type):
    Helical    phi=-57,  psi=-47
    Extended   phi=-139, psi=135     (beta / sheet-like)
    PPII       phi=-78,  psi=149
Each residue is assigned to its nearest conformer by angular distance on the
(phi,psi) torus, and the peptide takes the majority label. Residues that are not
near any of the three are counted as OTHER; a peptide that is mostly OTHER is
labelled IRREGULAR rather than being forced into a class it does not belong to.
"""
from __future__ import annotations

import csv
import math
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

warnings.filterwarnings("ignore")

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder

REFS = {
    "HELIX":    (-57.0, -47.0),
    "SHEET":    (-139.0, 135.0),
    "PPII":     (-78.0, 149.0),
}
TOL = 60.0  # degrees; beyond this from every reference the residue is OTHER


def _ang_delta(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def classify_peptide(pdb_path: str):
    """Return (ss_class, n_res) or (None, 0) if unparseable."""
    try:
        st = PDBParser(QUIET=True).get_structure("p", pdb_path)
    except Exception:
        return None, 0
    counts = {k: 0 for k in REFS}
    counts["OTHER"] = 0
    n = 0
    for pp in PPBuilder().build_peptides(st):
        phipsi = pp.get_phi_psi_list()
        for phi, psi in phipsi:
            if phi is None or psi is None:
                continue
            n += 1
            p, s = math.degrees(phi), math.degrees(psi)
            best, bestd = "OTHER", 1e9
            for name, (rp, rs) in REFS.items():
                d = math.hypot(_ang_delta(p, rp), _ang_delta(s, rs))
                if d < bestd:
                    best, bestd = name, d
            counts[best if bestd <= TOL else "OTHER"] += 1
    if n == 0:
        return None, 0
    struct = {k: v for k, v in counts.items() if k != "OTHER"}
    top = max(struct, key=struct.get)
    if struct[top] == 0 or counts["OTHER"] > n / 2:
        return "IRREGULAR", n
    return top, n


def bucket(L: int) -> str:
    return ("short" if L <= 8 else "med" if L <= 12 else "long" if L <= 16 else "vlong")


def _work(row):
    ss, n = classify_peptide(row["peptide_description"])
    try:
        L = int(row["pep_len"])
    except Exception:
        L = n
    return {
        "complex_name": row["complex_name"],
        "protein_description": row["protein_description"],
        "peptide_description": row["peptide_description"],
        "source": row.get("source", "?"),
        "pep_len": L,
        "ss_class": ss or "UNPARSED",
        "length_bucket": bucket(L),
    }


def main():
    pool_csv, out_csv = sys.argv[1], sys.argv[2]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    rows = list(csv.DictReader(open(pool_csv)))
    print(f"labelling {len(rows)} complexes with {nproc} procs...", flush=True)
    with Pool(nproc) as p:
        out = p.map(_work, rows, chunksize=64)
    fields = ["complex_name", "protein_description", "peptide_description",
              "source", "pep_len", "ss_class", "length_bucket"]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out)

    from collections import Counter
    cells = Counter((r["ss_class"], r["length_bucket"]) for r in out)
    print(f"\nwrote {out_csv}")
    print(f"{'ss_class':11}{'short':>8}{'med':>8}{'long':>8}{'vlong':>8}{'total':>9}")
    for ss in ["HELIX", "SHEET", "PPII", "IRREGULAR", "UNPARSED"]:
        r = [cells.get((ss, b), 0) for b in ("short", "med", "long", "vlong")]
        if sum(r):
            print(f"{ss:11}{r[0]:8d}{r[1]:8d}{r[2]:8d}{r[3]:8d}{sum(r):9d}")


if __name__ == "__main__":
    main()
