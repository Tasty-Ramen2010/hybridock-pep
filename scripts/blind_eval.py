#!/usr/bin/env python
"""Score a BLIND docking run: full receptor, no site given, direct RMSD to the crystal peptide.

No superposition is applied and none should be. The receptor handed to the sampler is the crystal
receptor itself, so a pose is already in the crystal frame, and superimposing would hide exactly
the placement error this run exists to measure. The peptide could be anywhere on a median
391-residue surface, which is the whole point -- the site-specific bench truncates to ~58
residues around the TRUE peptide, so it is not a blind test at all.

Usage: blind_eval.py --label <tag> [--ref <other tag> to pair against]
"""
from __future__ import annotations
import argparse, csv, os
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("HDP_ROOT", Path(__file__).resolve().parent.parent))

def ca(p: Path) -> np.ndarray:
    out, seen = [], set()
    for l in p.read_text(errors="ignore").splitlines():
        if l.startswith("ATOM") and l[12:16].strip() == "CA":
            k = (l[21], l[22:27])
            if k in seen: continue
            seen.add(k); out.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return np.array(out)

def rmsd(A, B):
    n = min(len(A), len(B))
    return float(np.sqrt(((A[:n] - B[:n]) ** 2).sum(1).mean())) if n >= 3 else float("nan")

BEST_OF = False

def collect(label, rows):
    poses = ROOT / f"runs/blind/{label}"
    d = {}
    for r in rows:
        c = sorted((poses / r["name"]).glob("rank*.pdb")) if (poses / r["name"]).exists() else []
        if not c: continue
        tru = ca(Path(r["peptide_pdb"]))
        v = (min(rmsd(ca(x), tru) for x in c) if BEST_OF
             else rmsd(ca(next(x for x in c if x.name == "rank1.pdb")), tru))
        if np.isfinite(v): d[r["name"]] = v
    return d

def report(name, v):
    v = np.array(v)
    print(f"  {name:<26}{len(v):>5}{np.median(v):>10.2f}{v.mean():>9.2f}"
          f"{int((v <= 5).sum()):>8}{int((v <= 10).sum()):>8}{int((v <= 20).sum()):>8}")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--ref", default=None)
    # The DGX holds the 120-complex bench as blind_bench_120.csv with DGX paths; the old
    # hardcoded data/blind_bench.csv does not exist there, which crashed the Sep-21 scoring
    # pass after all three arms had finished docking.
    ap.add_argument("--bench", default=str(ROOT / "data/blind_bench.csv"))
    ap.add_argument("--best-of", action="store_true",
                    help="score the best of all poses (oracle) instead of rank1. With no "
                         "confidence model rank1 is just the first sample, so both are worth "
                         "reporting: rank1 is what a user gets, best-of is the sampler's ceiling")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.bench)))
    global BEST_OF
    BEST_OF = a.best_of
    mine = collect(a.label, rows)
    print(f"\nBLIND DOCKING — full receptors, no site given\n")
    print(f"  {'arm':<26}{'n':>5}{'median':>10}{'mean':>9}{'<=5A':>8}{'<=10A':>8}{'<=20A':>8}")
    if not mine:
        print("  no poses found"); return
    report(a.label, list(mine.values()))
    if not a.ref: return
    other = collect(a.ref, rows)
    report(a.ref, list(other.values()))
    both = sorted(set(mine) & set(other))
    if len(both) < 10:
        print(f"\n  only {len(both)} complexes in common — cannot pair"); return
    u = np.array([mine[k] for k in both]); f = np.array([other[k] for k in both])
    d = u - f
    from scipy import stats
    print(f"\n  PAIRED on {len(both)} complexes ({a.label} minus {a.ref}):")
    print(f"    median {np.median(d):+.2f} A   mean {d.mean():+.2f} A   sd {d.std(ddof=1):.2f}")
    print(f"    {a.label} better on {(u < f).sum()}/{len(d)}"
          f"   sign p={stats.binomtest(int((u < f).sum()), len(d), 0.5).pvalue:.4f}"
          f"   Wilcoxon p={stats.wilcoxon(u, f).pvalue:.4f}")
    print(f"    minimum detectable effect at this n: "
          f"{(1.96 + 0.84) * d.std(ddof=1) / np.sqrt(len(d)):.2f} A")

if __name__ == "__main__":
    main()
