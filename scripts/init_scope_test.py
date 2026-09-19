#!/usr/bin/env python
"""Does initialisation explain our error on the DESIGNED complexes, or only in blind mode?

WHY THIS EXISTS. I proposed template-seeded initialisation on the strength of the oracle-init
result: naive receptor-centroid init 206.4 A vs true-centroid init 15.8 A, 150/150 improved,
p=2.3e-26. That number is real but it was measured BLIND, on full uncropped receptors where the
peptide could be anywhere in a large protein.

The Coventry setup is not that. We hand the sampler the paper's designed binder -- 170-250
residues -- so the peptide's site is a groove in a small protein, and RAPiDock's native regime is
exactly this: every one of its training and evaluation runs is pocket-given, with the pocket
truncated to ~20 A around the TRUE crystal peptide. The model was never trained to search, which
is why blind fails; but it also means "give it a better start" may already be satisfied here.

THE MEASUREMENT. randomize_position() places the peptide at the receptor centroid plus a draw
from N(0, tr_sigma_max=30). So the naive init's expected offset from truth is the
centroid-to-true-peptide distance, blurred by ~30 A of noise. If that offset is small compared
with the error we actually make, initialisation is NOT the Coventry bottleneck and template
seeding would be solving a problem we do not have.

Reported alongside the receptor's own radius, because an offset of 15 A means something very
different in a 60 A protein than in a 300 A one.

Usage: init_scope_test.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("HDP_ROOT", Path(__file__).resolve().parent.parent))
PAPER = ROOT / "datasets/coventry/binders_paper"
GOOD = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc12", "pc21", "pc26", "pc28",
        "pc34", "pc43", "pc44", "pc46"]
#: our error on this exact bench, paper receptors, from receptor_flexibility_test.py (holo arm)
OURS = {"n1": 14.02, "n2": 5.72, "n3": 17.98, "n4": 24.48, "n7": 27.11, "pc2": 11.18,
        "pc12": 20.95, "pc21": 17.00, "pc26": 12.29, "pc28": 5.14, "pc34": 10.83,
        "pc43": 5.02, "pc44": 10.54, "pc46": 8.30}
TR_SIGMA_MAX = 30.0


def ca(p: Path) -> np.ndarray:
    out, seen = [], set()
    for l in p.read_text().splitlines():
        if l.startswith("ATOM") and l[12:16].strip() == "CA":
            k = (l[21], l[22:27])
            if k in seen:
                continue
            seen.add(k)
            out.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return np.array(out)


def main() -> None:
    print("Is our Coventry error an INITIALISATION error?\n")
    print(f"  {'target':<8}{'rec res':>9}{'rec radius':>12}{'centroid->pep':>15}"
          f"{'our error':>11}")
    off, err, rad = [], [], []
    for p in GOOD:
        rf, pf = PAPER / f"{p}_1b1.pdb", PAPER / f"{p}_peptide.pdb"
        if not (rf.exists() and pf.exists()):
            continue
        R, P = ca(rf), ca(pf)
        if len(R) < 10 or len(P) < 3:
            continue
        c = R.mean(0)
        d = float(np.linalg.norm(P.mean(0) - c))
        r = float(np.linalg.norm(R - c, axis=1).max())
        off.append(d); err.append(OURS[p]); rad.append(r)
        print(f"  {p:<8}{len(R):>9}{r:>12.1f}{d:>15.1f}{OURS[p]:>11.2f}")

    off, err, rad = np.array(off), np.array(err), np.array(rad)
    print(f"\n  {'median':<8}{'':>9}{np.median(rad):>12.1f}{np.median(off):>15.1f}"
          f"{np.median(err):>11.2f}")

    print(f"\n  The naive init draws N(0, {TR_SIGMA_MAX:.0f} A) about the receptor centroid.")
    print(f"  Median centroid-to-peptide offset: {np.median(off):.1f} A")
    print(f"  Median receptor radius:            {np.median(rad):.1f} A")
    print(f"  Median error we actually make:     {np.median(err):.2f} A")

    from scipy import stats
    r = stats.pearsonr(off, err)
    s = stats.spearmanr(off, err)
    print(f"\n  corr(centroid offset, our error)   Pearson {r.statistic:+.3f} (p={r.pvalue:.3f})"
          f"   Spearman {s.statistic:+.3f} (p={s.pvalue:.3f})")
    print("     If initialisation drove the error these would be strongly positive: the targets")
    print("     whose site sits furthest from the centroid would be the ones we get most wrong.")

    # How much of the receptor can a 30 A draw reach? If sigma covers the whole protein, the
    # init is effectively uninformative and a seed would genuinely narrow the search.
    frac = float(np.median(TR_SIGMA_MAX / rad))
    print(f"\n  tr_sigma_max / receptor radius = {frac:.2f}")
    print("     >1 means one draw spans the entire binder, so the starting point carries almost")
    print("     no information about where the groove is.")

    init_bound = np.median(off)
    print("\n  -> " + (
        f"initialisation is plausibly implicated: the typical site sits {init_bound:.1f} A from "
        f"the centroid,\n     comparable to the {np.median(err):.1f} A error we make."
        if init_bound > 0.5 * np.median(err) and r.statistic > 0.3 else
        f"initialisation does NOT explain this error. The site sits only {init_bound:.1f} A from "
        f"the centroid\n     against a {np.median(err):.1f} A error, and the correlation is "
        f"{r.statistic:+.3f}. On a binder this small the\n     sampler already starts near the "
        f"groove -- the peptide is being placed wrongly WITHIN a site it\n     has essentially "
        f"been given. Template seeding would solve a problem we do not have here;\n     its value "
        f"is in BLIND docking, which is a different product mode."))


if __name__ == "__main__":
    main()
