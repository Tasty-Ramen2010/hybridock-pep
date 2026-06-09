"""Knowledge-based residue-contact potential — derived + tested on the benchmark.

Tests the one feature class genuinely different from the size axis: contact
*preference* (is an AA-AA interface contact over/under-represented vs random),
not contact *count*. Derives a Miyazawa-Jernigan-style log-odds potential from
our own PepPC peptide interfaces (holding out the 65 benchmark PDBs — no
leakage), then scores the benchmark and reports size-controlled correlation
against the experimental ΔG.

Potential:  e(a,b) = -ln[ p_obs(a,b) / (p(a)·p(b)·mult) ]
  negative e = over-represented contact = favorable.
Complex score = Σ over interface contacts e(a,b)   (lower = better binding).
Also reports the per-contact mean (size-normalized) form.

Usage:  python scripts/knowledge_potential.py [--n-derive 3000]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
PEPPC = ROOT / "datasets" / "training_formatted_peppc"
BENCH = ROOT / "data" / "benchmark_crystal.json"
CONTACT = 4.5  # Å heavy-atom

_AA3to1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q",
           "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
           "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
AAS = "ACDEFGHIKLMNPQRSTVWY"
IDX = {a: i for i, a in enumerate(AAS)}


def residues(pdb: Path) -> list[tuple[str, np.ndarray]]:
    """Return [(aa1, heavy_xyz(n,3))] per residue."""
    res: dict[tuple, list] = {}
    name: dict[tuple, str] = {}
    try:
        lines = pdb.read_text().splitlines()
    except OSError:
        return []
    for ln in lines:
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        an = ln[12:16].strip()
        if not an or an[0] in ("H", "D"):
            continue
        try:
            key = (ln[21], ln[22:27])
            xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        except ValueError:
            continue
        res.setdefault(key, []).append(xyz)
        name[key] = ln[17:20].strip()
    out = []
    for k, xyz in res.items():
        aa = _AA3to1.get(name[k])
        if aa:
            out.append((aa, np.array(xyz)))
    return out


def interface_contacts(pep: list, rec: list) -> list[tuple[str, str]]:
    """Residue-residue contacts (any heavy atom < CONTACT) between peptide & receptor."""
    pairs = []
    c2 = CONTACT * CONTACT
    for aa_p, xp in pep:
        for aa_r, xr in rec:
            d2 = np.sum((xp[:, None, :] - xr[None, :, :]) ** 2, axis=-1)
            if d2.min() <= c2:
                pairs.append((aa_p, aa_r))
    return pairs


def derive_potential(n_derive: int, exclude: set[str]) -> np.ndarray:
    """Build the 20×20 log-odds contact potential from PepPC interfaces."""
    counts = np.zeros((20, 20))
    dirs = sorted(glob.glob(str(PEPPC / "peppc*")))
    used = 0
    for d in dirs:
        if used >= n_derive:
            break
        m = re.match(r"peppc[f]?_([0-9A-Za-z]{4})_", os.path.basename(d))
        if not m or m.group(1).upper() in exclude:
            continue
        pep_f = next(iter(Path(d).glob("*_peptide.pdb")), None)
        rec_f = next(iter(Path(d).glob("*_protein_pocket.pdb")), None)
        if not pep_f or not rec_f:
            continue
        pep, rec = residues(pep_f), residues(rec_f)
        if not pep or not rec:
            continue
        for a, b in interface_contacts(pep, rec):
            i, j = IDX[a], IDX[b]
            counts[i, j] += 1  # directional peptide(a)→receptor(b)
        used += 1
    # symmetrize over the pair-type (contact a-b regardless of side)
    sym = counts + counts.T
    total = sym.sum()
    p_ab = sym / total
    p_a = sym.sum(1) / total
    expected = np.outer(p_a, p_a)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = -np.log(p_ab / expected)
    e[~np.isfinite(e)] = 0.0
    print(f"Derived contact potential from {used} PepPC interfaces ({int(total)} contacts)")
    return e


def score(pep_f: Path, rec_f: Path, e: np.ndarray) -> tuple[float, int]:
    pep, rec = residues(pep_f), residues(rec_f)
    pairs = interface_contacts(pep, rec)
    s = sum(e[IDX[a], IDX[b]] for a, b in pairs)
    return s, len(pairs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-derive", type=int, default=3000)
    args = ap.parse_args()

    bench = json.loads(BENCH.read_text())
    exclude = {r["pdb"].upper() for r in bench}
    e = derive_potential(args.n_derive, exclude)

    y, ssum, smean, L = [], [], [], []
    for r in bench:
        s, n = score(ROOT / r["peptide_pdb"], ROOT / r["pocket_pdb"], e)
        if n == 0:
            continue
        y.append(r["dg_exp"]); ssum.append(s); smean.append(s / n); L.append(n)
    y, ssum, smean, L = map(np.array, (y, ssum, smean, L))

    print(f"\n=== Knowledge-based contact potential on benchmark (n={len(y)}) ===")
    print(f"  (baselines: Vina-docked CV 0.42; size n_contact corr +0.46; honest goal = beat size)")
    print(f"  KB sum        vs ΔG_exp:  r={pearsonr(ssum,y).statistic:+.3f}  rho={spearmanr(ssum,y).statistic:+.3f}")
    print(f"  KB per-contact vs ΔG_exp: r={pearsonr(smean,y).statistic:+.3f}  rho={spearmanr(smean,y).statistic:+.3f}")
    print(f"  KB sum vs size (n_contact): r={pearsonr(ssum,L).statistic:+.3f}  <- is it just size again?")
    print(f"  size (n_contact) vs ΔG:     r={pearsonr(L,y).statistic:+.3f}")
    # partial: KB controlling for size
    def resid(a, b): s = np.polyfit(b, a, 1); return a - (s[0]*b + s[1])
    pr = pearsonr(resid(smean, L), resid(y, L)).statistic
    print(f"  KB per-contact PARTIAL (size removed): r={pr:+.3f}  <- the honest signal")


if __name__ == "__main__":
    main()
