#!/usr/bin/env python
"""Score our 18x18 prediction against the measured grid of Wu et al. 2025 (Fig. 2B / Table S1).

Ground truth is 30 measured cells out of 324: 18 cognate pairs plus 12 cross-reactivities.
The other 294 showed no measurable binding.

WHAT IS AND IS NOT A FAIR TEST HERE.

  Row-wise (fix a peptide, rank the 18 binders) is the clean test.  The peptide is held
  constant down a row, so peptide length and composition cannot bias the comparison.

  Column-wise (fix a binder, rank the 18 peptides) is confounded: any interface energy
  grows with interface size, and these peptides run 8 to 18 residues, so longer peptides
  score better against every binder.  We therefore report column-wise both raw and after
  normalising each peptide's scores to zero mean / unit variance across the row, which
  removes the per-peptide offset that causes the confound.

  A length control is printed either way: if predicted score tracks peptide length as
  strongly as it tracks the truth, the grid is measuring size, not specificity.

Random expectation for top-1 of 18 is 5.6%.

Usage: coventry_grid_table.py [score_field]   (default best_iface; lower = better binding)
"""
from __future__ import annotations

import csv
import json
import os
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
FIELD = sys.argv[1] if len(sys.argv) > 1 else "best_iface"
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]


def spearman(a: list[float], b: list[float]) -> float:
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[s[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def auc(pos: list[float], neg: list[float]) -> float:
    """P(a positive scores better than a negative); lower score = better binding."""
    if not pos or not neg:
        return float("nan")
    w = sum((1.0 if p < n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def main() -> None:
    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    pred: dict[tuple[str, str], dict] = {}
    f = ROOT / os.environ.get("COVENTRY_REFINE", "logs/coventry_refine.jsonl")
    for line in f.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            pep, bl = r["name"].split("__")
            pred[(pep, bl[:-1])] = r

    have = [k for k in truth if k in pred and pred[k].get(FIELD) is not None]
    print("=" * 84)
    print(f"COVENTRY GRID - our prediction vs Wu et al. Science 2025, Fig. 2B / Table S1")
    print(f"score = {FIELD} (Rosetta ref2015 interface energy after repacking; lower = tighter)")
    print(f"cells predicted: {len(have)}/324")
    print("=" * 84)
    if len(have) < 50:
        print("too few cells scored yet for a meaningful table")
        return

    S = {k: pred[k][FIELD] for k in have}
    peps = [p for p in ORDER if sum(1 for b in ORDER if (p, b) in S) == 18]
    print(f"complete rows (all 18 binders scored): {len(peps)}/18")
    if not peps:
        return

    # ---- row-wise: fix peptide, rank binders ---------------------------------
    print(f"\n{'peptide':>8}  {'cognate rank':>12}  {'cognate score':>13}  "
          f"{'best binder':>11}  {'row mean':>9}")
    ranks, top1, top3 = [], 0, 0
    for p in peps:
        row = sorted(ORDER, key=lambda b: S[(p, b)])
        r = row.index(p) + 1
        ranks.append(r)
        top1 += r == 1
        top3 += r <= 3
        vals = [S[(p, b)] for b in ORDER]
        print(f"{p:>8}  {r:>12d}  {S[(p, p)]:>13.1f}  {row[0]:>11}  {st.mean(vals):>9.1f}")
    n = len(peps)
    print(f"\nROW-WISE raw (fix peptide, rank its 18 binders)")
    print(f"  cognate ranked #1 : {top1}/{n} ({100 * top1 / n:.0f}%)   random 5.6%")
    print(f"  cognate in top 3  : {top3}/{n} ({100 * top3 / n:.0f}%)   random 16.7%")
    print(f"  mean cognate rank : {st.mean(ranks):.1f} of 18   random 9.5")

    # ---- normalisation ------------------------------------------------------
    # Every peptide is run against all 18 binders and every binder against all 18 peptides,
    # so the grid is a complete two-way design and we can model it additively:
    #     score(i,j) = grand + peptide_effect(i) + binder_effect(j) + interaction(i,j)
    # Everything we systematically cannot model -- the desolvation offset of a particular
    # peptide, its length and charge, a binder's overall interface size -- is a MAIN EFFECT.
    # It is constant down a row or a column and cancels in the interaction term.  Specificity
    # is by definition the interaction: binder j is good for peptide i beyond what peptide i
    # and binder j are worth on their own.  So double-centring is not a cosmetic normalisation
    # here, it is the estimator that matches what the assay measures.
    mu = {p: st.mean([S[(p, b)] for b in ORDER]) for p in peps}
    sd = {p: (st.pstdev([S[(p, b)] for b in ORDER]) or 1.0) for p in peps}
    Z = {(p, b): (S[(p, b)] - mu[p]) / sd[p] for p in peps for b in ORDER}
    colmu = {b: st.mean([S[(p, b)] for p in peps]) for b in ORDER}
    grand = st.mean([S[(p, b)] for p in peps for b in ORDER])
    # interaction residual: subtract both main effects, add back the grand mean
    W = {(p, b): S[(p, b)] - mu[p] - colmu[b] + grand for p in peps for b in ORDER}
    for label, M in [("raw", S), ("row-normalised", Z), ("two-way interaction", W)]:
        t1 = t3 = tot = 0
        rr = []
        for b in ORDER:
            col = [p for p in peps if (p, b) in M]
            if b not in col:
                continue
            order = sorted(col, key=lambda p: M[(p, b)])
            r = order.index(b) + 1
            rr.append(r)
            t1 += r == 1
            t3 += r <= 3
            tot += 1
        if tot:
            print(f"\nCOLUMN-WISE {label} (fix binder, rank the peptides) n={tot}")
            print(f"  cognate ranked #1 : {t1}/{tot} ({100 * t1 / tot:.0f}%)")
            print(f"  cognate in top 3  : {t3}/{tot} ({100 * t3 / tot:.0f}%)")
            print(f"  mean cognate rank : {st.mean(rr):.1f}")

    # row-wise under the interaction term: within a row, double-centring subtracts a
    # per-BINDER constant, so unlike row-normalisation it genuinely reorders the row --
    # it removes the "some binders are sticky against everything" effect.
    t1 = t3 = 0
    rr = []
    for p in peps:
        row = sorted(ORDER, key=lambda b: W[(p, b)])
        r = row.index(p) + 1
        rr.append(r)
        t1 += r == 1
        t3 += r <= 3
    print(f"\nROW-WISE two-way interaction (removes per-binder stickiness)")
    print(f"  cognate ranked #1 : {t1}/{n} ({100 * t1 / n:.0f}%)   random 5.6%")
    print(f"  cognate in top 3  : {t3}/{n} ({100 * t3 / n:.0f}%)   random 16.7%")
    print(f"  mean cognate rank : {st.mean(rr):.1f} of 18   random 9.5")

    # ---- discrimination over the whole grid ---------------------------------
    cog = [S[k] for k in have if truth[k]["cognate"] == "1"]
    meas = [S[k] for k in have if truth[k]["measured"] == "1"]
    none = [S[k] for k in have if truth[k]["measured"] == "0"]
    zcog = [Z[k] for k in Z if truth[k]["cognate"] == "1"]
    znone = [Z[k] for k in Z if truth[k]["measured"] == "0"]
    wcog = [W[k] for k in W if truth[k]["cognate"] == "1"]
    wnone = [W[k] for k in W if truth[k]["measured"] == "0"]
    wmeas = [W[k] for k in W if truth[k]["measured"] == "1"]
    print(f"\nDISCRIMINATION over all scored cells")
    print(f"  AUC cognate ({len(cog)}) vs non-binding ({len(none)})")
    print(f"     raw {auc(cog, none):.3f}   row-norm {auc(zcog, znone):.3f}   "
          f"two-way {auc(wcog, wnone):.3f}")
    print(f"  AUC any-measured vs non-binding")
    print(f"     raw {auc(meas, none):.3f}   two-way {auc(wmeas, wnone):.3f}")
    print(f"  median score  cognate {st.median(cog):7.1f} | measured {st.median(meas):7.1f} "
          f"| non-binding {st.median(none):7.1f}")

    # ---- affinity correlation on the measured cells -------------------------
    mk = [k for k in have if truth[k]["measured"] == "1" and truth[k]["dG_kcal_mol"]]
    if len(mk) >= 8:
        print(f"\nAFFINITY on the {len(mk)} measured cells")
        print(f"  Spearman(score, dG)  {spearman([S[k] for k in mk], [float(truth[k]['dG_kcal_mol']) for k in mk]):+.3f}")

    # ---- length control -----------------------------------------------------
    L = [len(truth[k]["peptide_seq"]) for k in have]
    print(f"\nCONTROL")
    print(f"  Spearman(score, peptide length) over all cells {spearman([S[k] for k in have], L):+.3f}")
    print(f"     (a large value here means the score tracks interface SIZE, not specificity;")
    print(f"      row-wise numbers above are immune to it, column-wise raw is not)")
    clean = [pred[k]["n_clean"] for k in have]
    print(f"  poses refining clean (>=2.5A) per cell: median {st.median(clean):.0f}/24")


if __name__ == "__main__":
    main()
