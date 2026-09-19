#!/usr/bin/env python
"""Peptide by peptide: what each row's error is made of, and whether we inflicted it ourselves.

A mean cognate rank is an average over eighteen very different rows, and averages hide causes.
This takes the grid apart four ways.

1. ROW vs COLUMN. The cognate sits at (i, i), so peptide i and binder i are confounded in every
   row-wise statistic we have reported. They can be separated: a row-wise failure means the other
   binders out-score the true one FOR THIS PEPTIDE; a column-wise failure means this binder
   out-scores its true partner FOR EVERY PEPTIDE -- a sticky column, which is a property of the
   binder model and not of the pair. Per-binder false-positive counts separate the two.

2. THE SYMMETRY WE HAVE BEEN THROWING AWAY. An all-by-all panel constrains the answer twice:
   the cognate should be the best binder for its peptide AND the best peptide for its binder.
   Every number we have reported ranks within rows only, discarding half the design. Ranking
   within columns as well is free, requires no new physics and no fitting, and is the natural
   completion of the cancellation idea in rank space -- median polish removes additive row and
   column effects, but a rank is not additive, so this is genuinely new information rather than
   a re-expression of what the polish already did.

3. PEPTIDE CHEMISTRY. Length, entropy, net charge, hydrophobic and charged fractions, against
   rank in each arm -- with the caveat that n=18 makes any single correlation weak evidence.

4. WHAT WE DID TO OURSELVES. Errors we introduced, separated from errors in the method: which
   binder model each column was scored against, whether the four rebuilt columns still behave
   differently, how many poses each cell actually had, and whether any cell is short.

Usage: coventry_per_peptide.py
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "src"))
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
N = len(ORDER)
BROKEN = {"n3", "n7", "pc21", "pc26"}
KD = set("KRDE")
PHOB = set("AVLIMFWC")


def polish(M):
    from hybridock_pep.scoring.cancellation import median_polish
    return median_polish(M)[0]


def grid_of(path: Path, key: str = "i_total") -> np.ndarray:
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "best" in r:
                rows[(r["peptide"], r["binder"])] = float(r["best"].get(key, 0.0))
    return np.array([[rows.get((p, b), 0.0) for b in ORDER] for p in ORDER])


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def rowrank(S: np.ndarray) -> np.ndarray:
    R = np.zeros((N, N), dtype=int)
    for i in range(N):
        for k, j in enumerate(sorted(range(N), key=lambda j: S[i, j])):
            R[i, j] = k + 1
    return R


def colrank(S: np.ndarray) -> np.ndarray:
    return rowrank(S.T).T


def main() -> None:
    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    seqs = {p: next(r["peptide_seq"] for (pp, _), r in truth.items() if pp == p) for p in ORDER}
    keep = np.zeros((N, N), dtype=bool)
    for i, p in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            keep[i, j] = (i == j) or truth.get((p, b), {}).get("measured") == "0"

    def grade(S, tag=""):
        cog = np.array([S[i, i] for i in range(N)])
        non = np.array([S[i, j] for i in range(N) for j in range(N) if i != j and keep[i, j]])
        a = float(np.mean([(c < non).mean() + 0.5 * (c == non).mean() for c in cog]))
        rr = [sorted(range(N), key=lambda j: S[i, j]).index(i) + 1 for i in range(N)]
        if tag:
            print(f"  {tag:<46}{a:>7.3f}{st.mean(rr):>7.2f}"
                  f"{sum(r <= 3 for r in rr):>5}{sum(r <= 5 for r in rr):>5}")
        return a, st.mean(rr), rr

    OURS = polish(grid_of(ROOT / "logs/sel_coventry_features_hard.jsonl"))
    COF = polish(grid_of(ROOT / "logs/sel_cofold_features.jsonl"))
    aux = {}
    for line in (ROOT / "logs/coventry_boltz_refine.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            aux[(r["peptide"], r["binder"])] = r
    IPT = np.array([[-(aux.get((p, b), {}).get("iptm") or 0.0) for b in ORDER] for p in ORDER])
    _z = lambda A: (A - A.mean()) / (A.std() or 1.0)
    _mc = lambda A: A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean()
    A_our, A_cof = _z(OURS), _z(COF) + _z(_mc(IPT))
    Ro, Rc = rowrank(A_our), rowrank(A_cof)
    COMB = (Ro + Rc).astype(float)
    Rb = rowrank(COMB)

    # ---------------------------------------------------------------- 1. row vs column
    print("PER-BINDER STICKINESS.  How often binder j lands in some OTHER peptide's top five.")
    print("By construction the average is 5.0 of 18. A binder well above that is over-predicted")
    print("for everyone, which is a property of that binder's model, not of any pair.\n")
    print(f"  {'binder':<8}{'model':<9}{'sticky(ours)':>14}{'sticky(boltz)':>15}"
          f"{'its cognate rank':>18}")
    sticky_o, sticky_c = [], []
    for j, b in enumerate(ORDER):
        so = sum(1 for i in range(N) if i != j and Ro[i, j] <= 5)
        scc = sum(1 for i in range(N) if i != j and Rc[i, j] <= 5)
        sticky_o.append(so); sticky_c.append(scc)
        print(f"  {b:<8}{'AF3' if b in BROKEN else 'ESMFold':<9}{so:>14}{scc:>15}"
              f"{Ro[j, j]:>10}{Rc[j, j]:>8}")
    print(f"\n  corr(binder stickiness, that binder's own cognate rank):  "
          f"ours {pearson(sticky_o, [Ro[i, i] for i in range(N)]):+.3f}   "
          f"boltz {pearson(sticky_c, [Rc[i, i] for i in range(N)]):+.3f}")
    print("  a positive value means sticky binders also lose their own row -- a column artefact")

    # ---------------------------------------------------------------- 2. the symmetry
    print("\nUSING BOTH DIRECTIONS OF THE DESIGN.  The cognate should be the best binder for its")
    print("peptide AND the best peptide for its binder. Everything so far used only the first.\n")
    print(f"  {'scoring':<46}{'AUC':>7}{'rank':>7}{'t3':>5}{'t5':>5}")
    for tag, S in (("ours, row-rank only (what we reported)", A_our),
                   ("ours, row + column rank", (rowrank(A_our) + colrank(A_our)).astype(float)),
                   ("boltz, row-rank only (what we reported)", A_cof),
                   ("boltz, row + column rank", (rowrank(A_cof) + colrank(A_cof)).astype(float)),
                   ("combined, row-rank only", COMB),
                   ("combined, row + column rank",
                    (rowrank(A_our) + colrank(A_our) +
                     rowrank(A_cof) + colrank(A_cof)).astype(float))):
        grade(S, tag)

    # ---------------------------------------------------------------- 3. chemistry
    print("\nPER-PEPTIDE.  Rank of the true partner in each arm, with the peptide's chemistry.\n")
    print(f"  {'pep':<7}{'len':>4}{'ent':>6}{'charge':>8}{'phob':>6}{'KRDE':>6}"
          f"{'ours':>6}{'boltz':>6}{'comb':>6}{'colrank':>9}")
    rows = []
    for i, p in enumerate(ORDER):
        s = seqs[p]
        c = Counter(s)
        r = dict(pep=p, length=len(s),
                 ent=-sum((v / len(s)) * math.log2(v / len(s)) for v in c.values()),
                 charge=sum(s.count(x) for x in "KR") - sum(s.count(x) for x in "DE"),
                 phob=sum(s.count(x) for x in PHOB) / len(s),
                 krde=sum(s.count(x) for x in KD) / len(s),
                 our=Ro[i, i], cof=Rc[i, i], comb=Rb[i, i],
                 colr=int(colrank(A_our)[i, i]))
        rows.append(r)
        print(f"  {p:<7}{r['length']:>4}{r['ent']:>6.2f}{r['charge']:>+8d}{r['phob']:>6.2f}"
              f"{r['krde']:>6.2f}{r['our']:>6}{r['cof']:>6}{r['comb']:>6}{r['colr']:>9}")

    print("\n  correlation of rank with peptide chemistry (n=18, so weak evidence either way):\n")
    print(f"  {'property':<28}{'ours':>9}{'boltz':>9}{'combined':>11}")
    for label, k in (("length", "length"), ("sequence entropy", "ent"), ("net charge", "charge"),
                     ("hydrophobic fraction", "phob"), ("charged fraction", "krde")):
        v = [r[k] for r in rows]
        print(f"  {label:<28}{pearson([r['our'] for r in rows], v):>9.3f}"
              f"{pearson([r['cof'] for r in rows], v):>9.3f}"
              f"{pearson([r['comb'] for r in rows], v):>11.3f}")

    # ---------------------------------------------------------------- 4. our own doing
    print("\nWHAT WE DID TO OURSELVES.\n")
    npose = {}
    for line in (ROOT / "logs/sel_coventry_features_hard.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            npose[(r["peptide"], r["binder"])] = r.get("n_poses", 0)
    vals = list(npose.values())
    print(f"  pose pool: {min(vals)}-{max(vals)} per cell, median {st.median(vals)}; "
          f"{sum(1 for v in vals if v < 10)} cells under 10")
    mdl = Counter("AF3" if b in BROKEN else "ESMFold" for b in ORDER)
    print(f"  binder models: {dict(mdl)} -- the four AF3 columns are the ones where our own "
          f"ESMFold model was the proven outlier")
    rb = [i for i, b in enumerate(ORDER) if b in BROKEN]
    ok = [i for i, b in enumerate(ORDER) if b not in BROKEN]
    print(f"  cognate rank, rebuilt columns  {st.mean(Ro[i, i] for i in rb):.2f} (ours) / "
          f"{st.mean(Rc[i, i] for i in rb):.2f} (boltz)   n={len(rb)}")
    print(f"  cognate rank, other columns    {st.mean(Ro[i, i] for i in ok):.2f} (ours) / "
          f"{st.mean(Rc[i, i] for i in ok):.2f} (boltz)   n={len(ok)}")
    print(f"  stickiness, rebuilt columns    {st.mean(sticky_o[i] for i in rb):.1f} (ours) / "
          f"{st.mean(sticky_c[i] for i in rb):.1f} (boltz)")
    print(f"  stickiness, other columns      {st.mean(sticky_o[i] for i in ok):.1f} (ours) / "
          f"{st.mean(sticky_c[i] for i in ok):.1f} (boltz)")

    dupe = Counter(seqs.values())
    print(f"  peptide sequences: {len(set(seqs.values()))} distinct of 18"
          + (f"  DUPLICATES: {[s for s, k in dupe.items() if k > 1]}" if max(dupe.values()) > 1
             else ""))
    lens = [len(s) for s in seqs.values()]
    print(f"  peptide lengths {min(lens)}-{max(lens)} aa, all within our <=25 aa scope")


if __name__ == "__main__":
    main()
