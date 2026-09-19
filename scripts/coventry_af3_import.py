#!/usr/bin/env python
"""Import Ram's AlphaFold 3 binder structures and compare them to our ESMFold models.

AlphaFold Server returns five models per job plus per-model confidence JSON. We take the
model with the best ranking_score (AF3's own ranking), convert the mmCIF to PDB, and write
it next to the ESMFold set so the whole grid can be re-run with only a --binders swap.

Also reports, per binder: AF3 pLDDT and pTM, our ESMFold pLDDT, and the CA RMSD between
the two predictions after superposition. A large RMSD means the two folders disagree about
the binder's shape, which matters because every grid cell in that column was docked against
the ESMFold version.

Usage: coventry_af3_import.py [src_dir] [out_dir]
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/claude-1000/af3")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    "/home/igem/unknown_software/datasets/coventry/binders_af3")
ESM = Path("/home/igem/unknown_software/datasets/coventry/binders")
ROOT = Path("/home/igem/unknown_software")
AA3to1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
          "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
          "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
          "TYR": "Y", "VAL": "V", "MSE": "M"}


def read_cif_atoms(path: Path):
    """Minimal mmCIF atom_site reader -- AF3 output is a plain loop_, no multi-line values."""
    cols, rows, in_loop = [], [], False
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("_atom_site."):
            cols.append(s.split(".", 1)[1])
            in_loop = True
            continue
        if in_loop:
            if s.startswith("#") or s.startswith("loop_") or not s:
                if rows:
                    break
                continue
            parts = s.split()
            if len(parts) < len(cols):
                continue
            rows.append(dict(zip(cols, parts)))
    return rows


def to_pdb(rows, out: Path) -> int:
    n = 0
    with out.open("w") as fh:
        for i, r in enumerate(rows, 1):
            if r.get("group_PDB") != "ATOM":
                continue
            name = r["label_atom_id"].strip('"')
            res = r["label_comp_id"]
            ch = r.get("auth_asym_id") or r.get("label_asym_id") or "A"
            seqid = r.get("auth_seq_id") or r.get("label_seq_id")
            x, y, z = float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"])
            b = float(r.get("B_iso_or_equiv") or 0.0)
            el = (r.get("type_symbol") or name[0]).rjust(2)
            an = name if len(name) >= 4 else f" {name:<3}"
            fh.write(f"ATOM  {i:5d} {an:<4}{res:>4} {ch[0]}{int(seqid):4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}          {el}\n")
            n += 1
        fh.write("END\n")
    return n


def ca(path: Path):
    return np.array([(float(l[30:38]), float(l[38:46]), float(l[46:54]))
                     for l in path.read_text().splitlines()
                     if l.startswith("ATOM") and l[12:16].strip() == "CA"])


def kabsch_rmsd(P, Q):
    n = min(len(P), len(Q))
    P, Q = P[:n] - P[:n].mean(0), Q[:n] - Q[:n].mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.sqrt((((R @ P.T).T - Q) ** 2).sum(1).mean()))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    esm_meta = {}
    f = ROOT / "logs/coventry_fold.json"
    if f.exists():
        esm_meta = json.loads(f.read_text())

    jobs = {}
    for cif in SRC.rglob("*_model_*.cif"):
        name = cif.name.split("_model_")[0].replace("fold_", "")
        idx = cif.name.split("_model_")[1].split(".")[0]
        conf = list(cif.parent.glob(f"*summary_confidences_{idx}.json"))
        score, plddt, ptm = -1.0, float("nan"), float("nan")
        if conf:
            try:
                c = json.loads(conf[0].read_text())
                score = float(c.get("ranking_score", -1))
                ptm = float(c.get("ptm", float("nan")))
            except Exception:  # noqa: BLE001
                pass
        jobs.setdefault(name, []).append((score, cif))

    print(f"{'binder':<12}{'AF3 model':>10}{'rank score':>12}{'AF3 pLDDT':>11}"
          f"{'ESM pLDDT':>11}{'CA RMSD':>9}")
    rmsds = []
    for name in sorted(jobs):
        best = sorted(jobs[name], key=lambda t: -t[0])[0]
        rows = read_cif_atoms(best[1])
        out = OUT / f"{name}.pdb"
        natoms = to_pdb(rows, out)
        # AF3 writes per-atom pLDDT into B-factor
        bs = [float(l[60:66]) for l in out.read_text().splitlines()
              if l.startswith("ATOM") and l[12:16].strip() == "CA"]
        af3_plddt = st.mean(bs) if bs else float("nan")
        e = ESM / f"{name}.pdb"
        r = kabsch_rmsd(ca(out), ca(e)) if e.exists() else float("nan")
        if r == r:
            rmsds.append(r)
        print(f"{name:<12}{best[1].name.split('_model_')[1][0]:>10}{best[0]:>12.3f}"
              f"{af3_plddt:>11.1f}{esm_meta.get(name, {}).get('plddt', float('nan')):>11.1f}"
              f"{r:>9.2f}")
    print(f"\nimported {len(jobs)} AF3 binders -> {OUT}")
    if rmsds:
        print(f"AF3 vs ESMFold CA RMSD: median {st.median(rmsds):.2f} A, "
              f"max {max(rmsds):.2f} A")
        print("(large values = the two folders disagree; every grid cell in that column "
              "was docked against the ESMFold model)")


if __name__ == "__main__":
    main()
