#!/usr/bin/env python
"""Can ANY cheap descriptor tell a forward-threaded pose from a backward one?

Context: on 9CCE our model places the peptide within ~1 A of the right site with roughly
the right shape, but 79-81% of poses thread the groove ANTIPARALLEL to the crystal, and
Rosetta ref2015 interface energy cannot tell the difference (P(forward better) = 0.410,
worse than chance). If some descriptor can, then the failure is recoverable by re-ranking
and does not need a new model.

Descriptors, all direction-sensitive by construction (a reversed peptide in the same
groove gives a different value), all cheap and all computed from coordinates only:

  burial_corr   Spearman-free correlation between the pose's per-residue burial profile
                (receptor neighbours within 8 A, per residue, N->C) and the crystal's.
                A groove that is asymmetric in depth should anti-correlate when reversed.
  nterm_acid    distance from the peptide N-terminal N to the nearest Asp/Glu carboxyl O.
                A free N-terminus is +1 and should sit near acid, not base.
  cterm_base    distance from the C-terminal carboxyl C to the nearest Lys NZ / Arg CZ.
  hbond_n       backbone N/O pairs within 3.5 A across the interface.
  contacts      receptor heavy atoms within 4.5 A of the peptide (size control -- this one
                should NOT discriminate; if it does, we are measuring burial not direction).

Reports AUC for separating true-forward from true-backward poses. 0.5 = useless.

Usage: coventry_direction_probe.py [pose_glob]
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path("/home/igem/unknown_software")
D = ROOT / "datasets/coventry/9cce"
ACID = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
BASE = {("LYS", "NZ"), ("ARG", "CZ"), ("ARG", "NH1"), ("ARG", "NH2")}


def parse(p, want_chain=None):
    out = []
    for l in open(p):
        if l.startswith("ATOM") and l[76:78].strip() != "H":
            if want_chain and l[21] != want_chain:
                continue
            out.append((l[17:20].strip(), l[12:16].strip(), int(l[22:26]),
                        np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])))
    return out


def descriptors(pep, rec, rtree, rxyz, ref_profile):
    resids = sorted({a[2] for a in pep})
    prof = []
    for r in resids:
        xyz = np.array([a[3] for a in pep if a[2] == r])
        prof.append(sum(len(x) for x in rtree.query_ball_point(xyz, 8.0)) / len(xyz))
    prof = np.array(prof, float)
    n = min(len(prof), len(ref_profile))
    a, b = prof[:n] - prof[:n].mean(), ref_profile[:n] - ref_profile[:n].mean()
    corr = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    nterm = next((a[3] for a in pep if a[2] == resids[0] and a[1] == "N"), None)
    cterm = next((a[3] for a in pep if a[2] == resids[-1] and a[1] == "C"), None)
    acid = np.array([a[3] for a in rec if (a[0], a[1]) in ACID]) if rec else np.empty((0, 3))
    base = np.array([a[3] for a in rec if (a[0], a[1]) in BASE]) if rec else np.empty((0, 3))
    d_na = float(np.linalg.norm(acid - nterm, axis=1).min()) if len(acid) and nterm is not None else np.nan
    d_cb = float(np.linalg.norm(base - cterm, axis=1).min()) if len(base) and cterm is not None else np.nan

    pxyz = np.array([a[3] for a in pep])
    pol = np.array([a[3] for a in pep if a[1] in ("N", "O")])
    rpol = np.array([a[3] for a in rec if a[1] in ("N", "O")])
    hb = (sum(len(x) for x in cKDTree(rpol).query_ball_point(pol, 3.5))
          if len(pol) and len(rpol) else 0)
    cont = sum(len(x) for x in rtree.query_ball_point(pxyz, 4.5))
    return {"burial_corr": corr, "nterm_acid": d_na, "cterm_base": d_cb,
            "hbond_n": float(hb), "contacts": float(cont)}


def auc(pos, neg, higher_is_forward=True):
    pos = [x for x in pos if x == x]
    neg = [x for x in neg if x == x]
    if not pos or not neg:
        return float("nan")
    w = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    a = w / (len(pos) * len(neg))
    return a if higher_is_forward else 1 - a


def main() -> None:
    pat = sys.argv[1] if len(sys.argv) > 1 else "runs/coventry/9cce/hybridock_ft__xtal_full_N500"
    rec = parse(D / "receptor_xtal_full.pdb")
    rxyz = np.array([a[3] for a in rec])
    rtree = cKDTree(rxyz)
    refp = parse(D / "peptide_xtal.pdb")
    ref_ca = np.array([a[3] for a in refp if a[1] == "CA"])
    rv = ref_ca[-1] - ref_ca[0]
    rv = rv / np.linalg.norm(rv)
    rres = sorted({a[2] for a in refp})
    ref_profile = np.array([sum(len(x) for x in rtree.query_ball_point(
        np.array([a[3] for a in refp if a[2] == r]), 8.0)) /
        len([a for a in refp if a[2] == r]) for r in rres], float)

    fwd, bwd = [], []
    for p in sorted(glob.glob(f"{pat}/rank*.pdb"),
                    key=lambda q: int(re.search(r"\d+", Path(q).name).group())):
        pep = parse(p)
        ca = np.array([a[3] for a in pep if a[1] == "CA"])
        if len(ca) < 3:
            continue
        v = ca[-1] - ca[0]
        v = v / (np.linalg.norm(v) or 1)
        (fwd if v @ rv > 0 else bwd).append(descriptors(pep, rec, rtree, rxyz, ref_profile))

    print(f"{Path(pat).name}: forward {len(fwd)}, backward {len(bwd)}")
    print(f"\n{'descriptor':<14}{'fwd median':>12}{'bwd median':>12}{'AUC':>8}  interpretation")
    for k, hi in [("burial_corr", True), ("nterm_acid", False), ("cterm_base", False),
                  ("hbond_n", True), ("contacts", True)]:
        f = [d[k] for d in fwd]
        b = [d[k] for d in bwd]
        a = auc(f, b, hi)
        note = ("USEFUL" if a >= 0.70 or a <= 0.30 else
                "weak" if a >= 0.60 or a <= 0.40 else "no signal")
        print(f"{k:<14}{np.nanmedian(f):>12.2f}{np.nanmedian(b):>12.2f}{a:>8.3f}  {note}")
    print("\nAUC is for separating true-forward from true-backward poses; 0.5 = useless.")
    print("contacts is the control: it SHOULD be ~0.5, it measures size not direction.")


if __name__ == "__main__":
    main()
