#!/usr/bin/env python
"""Where we are underwhelming, and whether it is the same place our poses disagree with Boltz's.

THE QUESTION. Panel D (co-folded poses + cancellation) reaches mean cognate rank 4.22, so eleven
rows are already in the top three and seven are not. The seven are not a random draw. This asks
three things about them, each with a control:

  1  POSE DISAGREEMENT. For every cell we have two independent interface energies for the SAME
     pair -- our docked pose and the co-folded pose -- both scored by the same ref2015 function
     after the same hard repack. After cancellation the two are on the same footing, so their
     difference is a pure pose-disagreement signal with the chemistry divided out. If our bad rows
     are the high-disagreement rows, the deficit is GENERATION. If disagreement is flat across
     good and bad rows, it is SCORING, and more sampling will not help.

  2  ASSAY DIFFICULTY. A row whose cognate binds at 2 nM and whose neighbours do not bind at all
     is an easy row. A row whose cognate binds at 1 uM with three measured cross-reactive
     partners is a hard row, and failing it says less about us. Rank is regressed on cognate
     affinity and on how crowded the row is.

  3  POSE QUALITY COVARIATES, from the co-folding side, which are free: ipTM, the binder
     fold-RMSD at transplant, the direction-prior axis quality, and the closest contact after
     refinement. These say whether a bad row is bad because the structure is bad.

WHY THE DISAGREEMENT IS COMPUTED AFTER CANCELLATION AND NOT BEFORE. Raw, the two scorers sit on
different scales -- co-folded poses are systematically deeper (median -58.5 vs -25.7 REU on the
cognates) -- so a raw difference is dominated by that offset and by each pose's own per-peptide
and per-binder baselines. Median-polishing both first removes exactly those, and z-scoring puts
the two interaction terms in the same units. What is left is disagreement about the PAIR.

Output: the ranked cell list for a Boltz escalation, written to data/coventry_disagree_cells.json

Usage: coventry_disagreement.py
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
FAIL_CAP = 50.0
OUT = ROOT / "data/coventry_disagree_cells.json"


def entropy(seq: str) -> float:
    c = Counter(seq)
    return -sum((v / len(seq)) * math.log2(v / len(seq)) for v in c.values())


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def ranks_of(S: np.ndarray) -> list[int]:
    return [sorted(range(N), key=lambda j: S[i, j]).index(i) + 1 for i in range(N)]


def main() -> None:
    from hybridock_pep.scoring.cancellation import two_way

    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    seqs = {p: next(r["peptide_seq"] for (pp, _), r in truth.items() if pp == p) for p in ORDER}

    ours = {}
    for line in (ROOT / "logs/sel_coventry_features_hard.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "best" in r:
                ours[(r["peptide"], r["binder"])] = float(r["best"].get("i_total", 0.0))
    REF = np.array([[ours[(p, b)] for b in ORDER] for p in ORDER])

    cof, aux = {}, {}
    for line in (ROOT / "logs/coventry_boltz_refine.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            v = r.get("best_iface")
            cof[(r["peptide"], r["binder"])] = FAIL_CAP if (v is None or v > FAIL_CAP) else v
            aux[(r["peptide"], r["binder"])] = r
    COF = np.array([[cof[(p, b)] for b in ORDER] for p in ORDER])
    IPT = np.array([[-(aux[(p, b)].get("iptm") or 0.0) for b in ORDER] for p in ORDER])

    PR = two_way(REF, robust=True)
    PC = two_way(COF, robust=True, scale=True)
    _z = lambda A: (A - A.mean()) / (A.std() or 1.0)
    _mc = lambda A: A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean()
    BEST = _z(PC) + _z(_mc(IPT))

    # disagreement: two independent poses for the same pair, both cancelled, both z-scored
    D = _z(PR) - _z(PC)

    rk_bar, rk_cof, rk_best = ranks_of(PR), ranks_of(PC), ranks_of(BEST)

    print("PER-ROW FAILURE TABLE.  rank is the cognate's position in its own row, 1 = correct.\n")
    print(f"  {'pep':<6}{'len':>4}{'ent':>6}{'meas':>5}{'cogKd':>8}"
          f"{'bar':>5}{'cof':>5}{'best':>6}{'|dis|row':>10}{'dis@cog':>9}"
          f"{'ipTM':>7}{'foldA':>7}{'axisQ':>7}")
    rows = []
    for i, p in enumerate(ORDER):
        meas = sum(1 for b in ORDER if truth.get((p, b), {}).get("measured") == "1")
        kd = truth.get((p, p), {}).get("kd_M")
        pkd = -math.log10(float(kd)) if kd else float("nan")
        a = aux[(p, p)]
        row = dict(pep=p, length=len(seqs[p]), ent=entropy(seqs[p]), meas=meas, pkd=pkd,
                   bar=rk_bar[i], cof=rk_cof[i], best=rk_best[i],
                   dis_row=float(np.abs(D[i]).mean()), dis_cog=float(D[i, i]),
                   iptm=a.get("iptm") or float("nan"), fold=a.get("fold_rmsd") or float("nan"),
                   axisq=a.get("axis_quality") or float("nan"))
        rows.append(row)
        print(f"  {p:<6}{row['length']:>4}{row['ent']:>6.2f}{meas:>5}{pkd:>8.2f}"
              f"{row['bar']:>5}{row['cof']:>5}{row['best']:>6}{row['dis_row']:>10.2f}"
              f"{row['dis_cog']:>9.2f}{row['iptm']:>7.2f}{row['fold']:>7.2f}{row['axisq']:>7.2f}")

    print("\nWHAT PREDICTS A BAD ROW.  Pearson r of cognate rank against each covariate;")
    print("positive r = the covariate goes UP as the rank gets WORSE.\n")
    for label, key in (("mean |disagreement| in row", "dis_row"),
                       ("disagreement at cognate cell", "dis_cog"),
                       ("cognate affinity  -log10 Kd", "pkd"),
                       ("measured partners in row", "meas"),
                       ("peptide length", "length"),
                       ("peptide sequence entropy", "ent"),
                       ("co-fold ipTM", "iptm"),
                       ("binder fold RMSD at transplant", "fold"),
                       ("direction-prior axis quality", "axisq")):
        v = [r[key] for r in rows]
        print(f"  {label:<34}"
              f"bar {pearson([r['bar'] for r in rows], v):+.3f}   "
              f"cofold {pearson([r['cof'] for r in rows], v):+.3f}   "
              f"best {pearson([r['best'] for r in rows], v):+.3f}")

    good = [r for r in rows if r["cof"] <= 3]
    bad = [r for r in rows if r["cof"] > 3]
    print(f"\n  rows the co-folded model gets in its top 3 (n={len(good)}): "
          f"mean |disagreement| {st.mean(r['dis_row'] for r in good):.2f}, "
          f"ipTM {st.mean(r['iptm'] for r in good):.2f}")
    print(f"  rows it does not                 (n={len(bad)}): "
          f"mean |disagreement| {st.mean(r['dis_row'] for r in bad):.2f}, "
          f"ipTM {st.mean(r['iptm'] for r in bad):.2f}")

    # which pose wins, cell by cell, among cells where both are scorable
    ours_better = int(((PR < PC) & (COF < FAIL_CAP)).sum())
    print(f"\n  our cancelled score is lower (better) than the co-folded one in "
          f"{ours_better}/{int((COF < FAIL_CAP).sum())} scorable cells")
    print(f"  corr(our interaction, co-folded interaction) over all 324 cells: "
          f"{pearson(PR.ravel(), PC.ravel()):+.3f}")

    # ---- the escalation list -------------------------------------------------------------
    # A cell is worth more diffusion samples if fixing it could change a row's answer: the
    # cognate of a row we get wrong, and the false positives currently beating it.
    cells = []
    for i, p in enumerate(ORDER):
        if rk_cof[i] <= 3:
            continue
        order = sorted(range(N), key=lambda j: BEST[i, j])
        cells.append({"peptide": p, "binder": p, "role": "cognate",
                      "rank_now": rk_cof[i], "disagree": round(float(D[i, i]), 2)})
        for j in order[:3]:
            if j != i:
                cells.append({"peptide": p, "binder": ORDER[j], "role": "false-positive",
                              "rank_now": order.index(j) + 1,
                              "disagree": round(float(D[i, j]), 2)})
    OUT.write_text(json.dumps(cells, indent=1))
    print(f"\n  {len(cells)} cells written to {OUT.relative_to(ROOT)} "
          f"({sum(c['role'] == 'cognate' for c in cells)} cognates + "
          f"{sum(c['role'] == 'false-positive' for c in cells)} false positives, "
          f"over {len(bad)} failing rows)")


if __name__ == "__main__":
    main()
