#!/usr/bin/env python
"""What exactly is wrong with our 9CCE poses? Decompose the error.

"Best RMSD 5.6 A" is a summary, not a diagnosis. A peptide can miss by 5.6 A because it is
in the wrong place, because it is the wrong shape, because it is threaded backwards, or
because it is shifted a residue or two along an otherwise correct groove. Those have
completely different fixes, so measure which one it is:

  direct RMSD    what we report: placement + conformation together, no superposition
                 (receptor frames are shared, so this is the honest docking number)
  Kabsch RMSD    pose superimposed onto the crystal peptide. Removes placement entirely,
                 so what is left is CONFORMATION error -- is the peptide even the right
                 shape? Low Kabsch + high direct = we build it right and put it wrong.
  centroid dist  pure placement error, independent of shape and direction
  direction cos  cosine between the pose's N->C vector and the crystal's. Negative means
                 the chain runs backwards through the groove -- the classic failure for a
                 pseudo-symmetric repeat protein that looks the same from either end.
  best register  slide the pose k residues along the sequence and re-score. If a shift of
                 1-3 residues rescues it, the groove is right and only the phase is wrong.

Usage: coventry_placement_forensics.py [target_dir_glob]
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
REF = ROOT / "datasets/coventry/9cce/peptide_xtal.pdb"


def ca(p):
    return np.array([(float(l[30:38]), float(l[38:46]), float(l[46:54]))
                     for l in open(p) if l.startswith("ATOM") and l[12:16].strip() == "CA"])


def kabsch_rmsd(P, Q):
    """RMSD after optimal superposition -- conformation error only."""
    p, q = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.sqrt((((R @ p.T).T - q) ** 2).sum(1).mean()))


def best_register(P, Q, max_shift=4):
    """Best direct RMSD over sequence shifts and over chain direction."""
    best = (1e9, 0, +1)
    for sign, S in ((+1, P), (-1, P[::-1])):
        for k in range(-max_shift, max_shift + 1):
            a, b = (S[k:], Q[:len(Q) - k]) if k >= 0 else (S[:k], Q[-k:])
            n = min(len(a), len(b))
            if n < 5:
                continue
            r = float(np.sqrt(((a[:n] - b[:n]) ** 2).sum(1).mean()))
            if r < best[0]:
                best = (r, k, sign)
    return best


def main() -> None:
    ref = ca(REF)
    rv = ref[-1] - ref[0]
    rv = rv / np.linalg.norm(rv)
    pat = sys.argv[1] if len(sys.argv) > 1 else "runs/coventry/9cce/*_N500"
    print(f"reference: {REF.name}, {len(ref)} residues, end-to-end {np.linalg.norm(ref[-1] - ref[0]):.1f} A")
    print(f"\n{'arm':<26}{'direct':>8}{'Kabsch':>8}{'centr':>7}{'dir':>7}"
          f"{'reg-fixed':>10}{'shift':>6}{'flipped':>9}")
    print(f"{'':<26}{'(best)':>8}{'(conf)':>8}{'(A)':>7}{'cos':>7}{'RMSD':>10}{'':>6}{'% of N':>9}")
    for d in sorted(glob.glob(pat)):
        poses = sorted(Path(d).glob("rank*.pdb"),
                       key=lambda p: int(re.search(r"\d+", p.name).group()))
        if not poses:
            continue
        rows = []
        for p in poses:
            X = ca(p)
            n = min(len(X), len(ref))
            X, Q = X[:n], ref[:n]
            v = X[-1] - X[0]
            v = v / (np.linalg.norm(v) or 1)
            rows.append((
                float(np.sqrt(((X - Q) ** 2).sum(1).mean())),
                kabsch_rmsd(X, Q),
                float(np.linalg.norm(X.mean(0) - Q.mean(0))),
                float(v @ rv),
                *best_register(X, Q),
            ))
        a = np.array([r[:4] for r in rows])
        reg = np.array([r[4] for r in rows])
        shifts = [r[5] for r in rows]
        signs = np.array([r[6] for r in rows])
        i = int(a[:, 0].argmin())
        flipped = 100 * float((a[:, 3] < 0).mean())
        j = int(reg.argmin())
        print(f"{Path(d).name:<26}{a[:, 0].min():>8.2f}{a[:, 1].min():>8.2f}"
              f"{a[:, 2].min():>7.1f}{a[i, 3]:>7.2f}{reg.min():>10.2f}"
              f"{shifts[j]:>+6d}{flipped:>8.0f}%")
        print(f"{'  (best-direct pose)':<26}{a[i, 0]:>8.2f}{a[i, 1]:>8.2f}{a[i, 2]:>7.1f}"
              f"{a[i, 3]:>7.2f}   register-fixed pose ran "
              f"{'BACKWARDS' if signs[j] < 0 else 'forwards'}")
        print(f"{'  medians':<26}{np.median(a[:, 0]):>8.2f}{np.median(a[:, 1]):>8.2f}"
              f"{np.median(a[:, 2]):>7.1f}{np.median(a[:, 3]):>7.2f}")


if __name__ == "__main__":
    main()
