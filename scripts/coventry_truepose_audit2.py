#!/usr/bin/env python
"""Our pose error against the designed complexes — sequence-aligned, mislabelled columns excluded.

SUPERSEDES coventry_truepose_audit.py, which indexed residues positionally and so was off by the
three-residue MSG tag our models carry and the paper's do not. That error inflated every
superposition and made all eighteen binder models look 5-6 A wrong. Here the correspondence is
found by sequence: the offset that maximises identity is located first, and only matched residues
enter the Kabsch fit.

FOUR COLUMNS ARE EXCLUDED AND SAID SO. pc17 and pc18 are swapped in our set, our pc11 is APE_1b2
(a different target entirely), and pc35 is a different design at 66% identity. Superimposing a
protein on a different protein produces a number, and the number is meaningless, so they are
reported apart from the fourteen where our file really is the paper's binder.

WHAT THE REMAINING FOURTEEN MEASURE. Superimpose our binder model onto the paper's, carry our
docked peptide through the same transform, and compare it to the designed peptide. That is
peptide error in the receptor frame -- the quantity that actually governs whether a scoring
function can see the interface -- and until these structures arrived we had no ground truth for
it on this set at all. The N-to-C axis cosine against the designed pose tests the "79-81% thread
backwards" claim directly, rather than through the geometric proxies that produced it.

Usage: coventry_truepose_audit2.py
"""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
PAPER = ROOT / "datasets/coventry/binders_paper"
ESM = ROOT / "datasets/coventry/binders"
AF3 = ROOT / "datasets/coventry/binders_af3"
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
#: our file is not the paper's protein for these -- see coventry_fix_binders.py
BAD_ID = {"pc11": "is APE_1b2", "pc17": "swapped with pc18",
          "pc18": "swapped with pc17", "pc35": "66% identity, different design"}
AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
       'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
       'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def ca_seq(p: Path, chain: str | None = None):
    xyz, seq, seen = [], [], set()
    for l in p.read_text().splitlines():
        if not l.startswith("ATOM") or l[12:16].strip() != "CA":
            continue
        if chain and l[21] != chain:
            continue
        k = (l[21], l[22:27])
        if k in seen:
            continue
        seen.add(k)
        xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
        seq.append(AA3.get(l[17:20].strip(), 'X'))
    return np.array(xyz), ''.join(seq)


def best_offset(a: str, b: str) -> tuple[int, float]:
    """Offset into `a` that best matches `b`, and the identity there."""
    best = (0, -1.0)
    for off in range(0, 10):
        x = a[off:]
        n = min(len(x), len(b))
        if n < 40:
            continue
        idt = sum(1 for u, v in zip(x[:n], b[:n]) if u == v) / n
        if idt > best[1]:
            best = (off, idt)
    return best


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def rmsd(A, B):
    n = min(len(A), len(B))
    return float(np.sqrt(((A[:n] - B[:n]) ** 2).sum(1).mean()))


def axis(X):
    v = X[-1] - X[0]
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.zeros(3)


def main() -> None:
    print("PART 1 — binder model accuracy, sequence-aligned (MSG tag handled)\n")
    print(f"  {'binder':<8}{'ident':>7}{'ESMFold':>10}{'AF3':>9}   note")
    esm, af3 = {}, {}
    for b in ORDER:
        pb, pseq = ca_seq(PAPER / f"{b}_1b1.pdb")
        res = []
        for d in (ESM, AF3):
            f = d / f"{b}_1b1.pdb"
            if not f.exists():
                res.append((None, 0.0)); continue
            m, mseq = ca_seq(f)
            off, idt = best_offset(mseq, pseq)
            mm = m[off:]
            n = min(len(mm), len(pb))
            if n < 40:
                res.append((None, idt)); continue
            R, t = kabsch(mm[:n], pb[:n])
            res.append((rmsd((R @ mm[:n].T).T + t, pb[:n]), idt))
        esm[b], af3[b] = res[0][0], res[1][0]
        fm = lambda v: f"{v:.2f}" if v is not None else "  -  "
        note = BAD_ID.get(b, "")
        print(f"  {b:<8}{100 * res[0][1]:>6.0f}%{fm(res[0][0]):>10}{fm(res[1][0]):>9}   {note}")

    good = [b for b in ORDER if b not in BAD_ID]
    ge = [esm[b] for b in good if esm[b] is not None]
    ga = [af3[b] for b in good if af3[b] is not None]
    print(f"\n  On the {len(good)} correctly-identified binders:")
    print(f"    ESMFold  median {st.median(ge):.2f} A   <=2A {sum(v <= 2 for v in ge)}/{len(ge)}"
          f"   <=3A {sum(v <= 3 for v in ge)}/{len(ge)}")
    print(f"    AF3      median {st.median(ga):.2f} A   <=2A {sum(v <= 2 for v in ga)}/{len(ga)}"
          f"   <=3A {sum(v <= 3 for v in ga)}/{len(ga)}")
    was_flagged = [b for b in ("n3", "n7", "pc21", "pc26")]
    print(f"    the 4 we had flagged as broken: ESMFold "
          f"{', '.join(f'{b} {esm[b]:.1f}A' for b in was_flagged if esm[b] is not None)}")

    print("\n\nPART 2 — our docked peptide vs the DESIGNED peptide pose "
          f"({len(good)} valid targets)\n")
    print(f"  {'target':<8}{'our dock':>10}{'co-folded':>11}{'cos ours':>10}"
          f"{'cos cofold':>12}   threading")
    orr, cfr, oax, cax = [], [], [], []
    for p in good:
        pb, pseq = ca_seq(PAPER / f"{p}_1b1.pdb")
        tp, _ = ca_seq(PAPER / f"{p}_peptide.pdb")
        src = ESM / f"{p}_1b1.pdb"
        m, mseq = ca_seq(src)
        off, _ = best_offset(mseq, pseq)
        mm = m[off:]
        n = min(len(mm), len(pb))
        R, t = kabsch(mm[:n], pb[:n])

        def go(f: Path):
            if not f.exists():
                return None, None
            q, _ = ca_seq(f)
            if len(q) < 3:
                return None, None
            q = (R @ q.T).T + t
            return rmsd(q, tp), float(np.dot(axis(q), axis(tp)))

        dk = sorted((ROOT / f"runs/coventry/hybridock_ft/{p}__{p}B").glob("rank*.pdb"))
        r1, a1 = go(dk[0]) if dk else (None, None)
        r2, a2 = go(ROOT / f"runs/coventry/boltz/{p}__{p}/peptide.pdb")
        for v, acc in ((r1, orr), (r2, cfr), (a1, oax), (a2, cax)):
            if v is not None:
                acc.append(v)
        f = lambda x: f"{x:.2f}" if x is not None else "  -  "
        th = "" if a1 is None else ("BACKWARDS" if a1 < 0 else "forward")
        print(f"  {p:<8}{f(r1):>10}{f(r2):>11}{f(a1):>10}{f(a2):>12}   {th}")

    for nm, rs in (("our docked peptide", orr), ("co-folded peptide", cfr)):
        if rs:
            print(f"\n  {nm:<20} median {st.median(rs):5.2f} A   "
                  f"<=2A {sum(r <= 2 for r in rs)}/{len(rs)}   "
                  f"<=5A {sum(r <= 5 for r in rs)}/{len(rs)}")
    print("\n  THREADING DIRECTION vs the designed pose (first ground-truth measurement):")
    for nm, ax in (("our docked", oax), ("co-folded ", cax)):
        if ax:
            bk = sum(1 for a in ax if a < 0)
            print(f"    {nm} runs backwards on {bk}/{len(ax)} ({100 * bk / len(ax):.0f}%), "
                  f"mean cosine {st.mean(ax):+.3f}")


if __name__ == "__main__":
    main()
