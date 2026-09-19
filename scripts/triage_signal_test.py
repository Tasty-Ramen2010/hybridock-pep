#!/usr/bin/env python
"""Can we tell our own pose is wrong WITHOUT co-folding it? The gate on the whole triage idea.

WHY THIS COMES FIRST. Co-folding costs 4.3 s/pose at n=25 against our 1.0 s/pose, so triage --
co-folding only the complexes that need it -- is worth roughly 4x on wall clock. But it only
works if the decision can be made BEFORE paying for the co-fold. The signal we already know
predicts our failures is pose disagreement between our arm and the co-folded arm (r = +0.704),
and that is circular: computing it requires the very structure we are deciding whether to build.

So the question is narrower and harder: is there anything in OUR OWN pose pool that says "this
one is wrong"? Five candidates, all free because we already compute them:

  ensemble spread     pairwise CA RMSD among our own 24 poses. If the sampler cannot agree with
                      itself, that is the classical uncertainty signal and costs nothing.
  energy gap          best minus second-best interface energy. A clear winner should mean a
                      converged funnel; a photo-finish should mean the scorer cannot tell.
  best energy         how good the winner looks in absolute terms.
  axis quality        how strand-like the best pose is (end-to-end over contour length).
  contact count       how much interface the pose actually makes.

GROUND TRUTH, which we have only had since yesterday: the 14 designed complexes where our binder
file really is the paper's protein, with the AF2 designed pose as the answer. Our median error
there is 10.88 A and 0/14 land under 5 A, so there is plenty of error to predict -- the question
is only whether it is predictable.

WHAT COUNTS AS PASSING. A usable trigger needs |r| > 0.5 against true RMSD, or it needs to
separate the best and worst halves cleanly. n=14 is small, so a weak correlation here is not
evidence of a weak signal -- but a near-zero one across all five candidates is enough to stop.

Usage: triage_signal_test.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "src"))
PAPER = ROOT / "datasets/coventry/binders_paper"
ESM = ROOT / "datasets/coventry/binders"
POOL = ROOT / "runs/coventry/hybridock_ft"
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
BAD_ID = {"pc11", "pc17", "pc18", "pc35"}
AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
       'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
       'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def ca_seq(p: Path):
    xyz, seq, seen = [], [], set()
    for l in p.read_text().splitlines():
        if not l.startswith("ATOM") or l[12:16].strip() != "CA":
            continue
        k = (l[21], l[22:27])
        if k in seen:
            continue
        seen.add(k)
        xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
        seq.append(AA3.get(l[17:20].strip(), 'X'))
    return np.array(xyz), ''.join(seq)


def best_offset(a, b):
    best = (0, -1.0)
    for off in range(0, 10):
        x = a[off:]
        n = min(len(x), len(b))
        if n < 40:
            continue
        idt = sum(1 for u, v in zip(x[:n], b[:n]) if u == v) / n
        if idt > best[1]:
            best = (off, idt)
    return best[0]


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def rmsd(A, B):
    n = min(len(A), len(B))
    return float(np.sqrt(((A[:n] - B[:n]) ** 2).sum(1).mean()))


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def spear(a, b):
    return corr(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))


def main() -> None:
    energy = {}
    for l in (ROOT / "logs/coventry_full_terms.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            energy[r["name"]] = {x["pose"]: float(x["total"]) for x in r["rows"]}

    rows = []
    for p in ORDER:
        if p in BAD_ID:
            continue
        tb, pseq = ca_seq(PAPER / f"{p}_1b1.pdb")
        tp, _ = ca_seq(PAPER / f"{p}_peptide.pdb")
        m, mseq = ca_seq(ESM / f"{p}_1b1.pdb")
        off = best_offset(mseq, pseq)
        mm = m[off:]
        n = min(len(mm), len(tb))
        R, t = kabsch(mm[:n], tb[:n])

        name = f"{p}__{p}B"
        poses = sorted((POOL / name).glob("rank*.pdb"),
                       key=lambda f: int(''.join(c for c in f.stem if c.isdigit()) or 0))
        if len(poses) < 5:
            continue
        cas = [ca_seq(f)[0] for f in poses]
        e = energy.get(name, {})
        evals = [e.get(f.name) for f in poses]
        evals = [v for v in evals if v is not None]

        # --- the five candidate triggers, all computed from OUR side only ---
        spread = st.mean(rmsd(cas[i], cas[j])
                         for i in range(min(8, len(cas)))
                         for j in range(i + 1, min(8, len(cas))))
        gap = (sorted(evals)[1] - sorted(evals)[0]) if len(evals) > 1 else float("nan")
        best_e = min(evals) if evals else float("nan")
        q = cas[0]
        contour = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
        axisq = float(np.linalg.norm(q[-1] - q[0]) / contour) if contour > 1e-6 else 0.0
        d = np.linalg.norm(q[:, None, :] - mm[None, :, :], axis=2)
        contacts = int((d < 8.0).sum())

        # --- the answer, which triage may not look at ---
        true_rmsd = rmsd((R @ q.T).T + t, tp)
        rows.append(dict(pep=p, spread=spread, gap=gap, best_e=best_e, axisq=axisq,
                         contacts=contacts, true=true_rmsd))

    print(f"{len(rows)} designed complexes with ground truth "
          f"(median error {st.median(r['true'] for r in rows):.2f} A)\n")
    print(f"  {'pep':<7}{'true RMSD':>11}{'spread':>9}{'E gap':>8}{'best E':>9}"
          f"{'axisQ':>8}{'contacts':>10}")
    for r in sorted(rows, key=lambda r: r["true"]):
        print(f"  {r['pep']:<7}{r['true']:>11.2f}{r['spread']:>9.2f}{r['gap']:>8.2f}"
              f"{r['best_e']:>9.1f}{r['axisq']:>8.2f}{r['contacts']:>10}")

    print("\nDOES ANY OF IT PREDICT OUR ERROR?  (we want |r| > 0.5)\n")
    print(f"  {'candidate trigger':<34}{'Pearson':>9}{'Spearman':>10}")
    y = [r["true"] for r in rows]
    best = None
    for lbl, k, sign in (("ensemble spread (self-disagreement)", "spread", 1),
                         ("energy gap, best to 2nd", "gap", -1),
                         ("best interface energy", "best_e", 1),
                         ("axis quality of best pose", "axisq", -1),
                         ("contact count of best pose", "contacts", -1)):
        v = [r[k] for r in rows]
        pr, sp = corr(v, y), spear(v, y)
        print(f"  {lbl:<34}{pr:>9.3f}{sp:>10.3f}")
        if best is None or abs(pr) > abs(best[1]):
            best = (lbl, pr, k, sign)

    lbl, pr, k, sign = best
    print(f"\n  strongest: {lbl} at r = {pr:+.3f}")

    # would it actually route correctly? split at the median of the trigger
    v = np.array([r[k] for r in rows], float)
    y = np.array(y, float)
    hi = v >= np.median(v)
    a, b = y[hi].mean(), y[~hi].mean()
    flagged, other = (a, b) if a > b else (b, a)
    print(f"  routing check — complexes the trigger flags average {flagged:.2f} A, "
          f"the rest {other:.2f} A")
    print(f"  separation: {flagged - other:+.2f} A")

    ok = abs(pr) > 0.5
    print("\n  -> " + ("A TRIGGER EXISTS. Triage is worth implementing."
                       if ok else
                       "NO USABLE TRIGGER at n=14. We cannot tell our own bad poses apart from "
                       "our good ones, so triage cannot be targeted — the choice is co-fold "
                       "everything or nothing."))


if __name__ == "__main__":
    main()
