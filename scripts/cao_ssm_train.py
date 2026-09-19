#!/usr/bin/env python
"""Train on the SSM data the way Coventry said to: Boolean bind/no-bind, held out BY TARGET.

THREE OF HIS RECOMMENDATIONS, IMPLEMENTED TOGETHER BECAUSE THEY ARE ONE IDEA.

  18:00  "training on the Boolean case is actually really interesting, of bind versus no bind.
          The affinity values I'm going to send you aren't rigorous... but one thing you can
          definitely say is that something binds or something doesn't."
  41:42  "you don't care about mutations up here. You only care about mutations down here."
  (email) "the fairest way to do it is by target. So, leave out one of the targets and train on
          the rest."

So: a classifier for "does this substitution abolish binding", trained only on positions that
demonstrably participate, evaluated leave-one-target-out. Every fold predicts a target whose
designs, scaffold and epitope it has never seen -- the strictest split available and the one he
specified.

WHY THIS IS WORTH BUILDING EVEN THOUGH IT IS SEQUENCE-ONLY. Our single-amino-acid blind spot is
the most stubborn failure in the project: r = 0.18 on congeneric series, and the e438 substitution
head collapsed to a near-constant (r = 0.998 regardless of which residue was substituted). That
failure was never a shortage of physics, it was a shortage of examples -- roughly 1,500 complexes.
Here there are 215,941 labelled substitutions. A model that knows which substitutions break an
interface is exactly the term our scorer is missing, and it needs no structure to learn.

THE WILD-TYPE IS RECOVERABLE, which makes real features possible. A saturation scan tests 19
substitutions per position; the twentieth amino acid -- the one absent from the scan -- is the
native residue. So wt->mut identity, and every physicochemical delta across it, can be
reconstructed from score files that never state the sequence.

CONTROLS. A shuffled-label arm and a majority-class baseline, because class balance varies wildly
between targets (7% to 86% measurable) and accuracy alone would be meaningless.

Usage: cao_ssm_train.py [--hot-only 0.6] [--tier any|somewhat|reasonable|quite]
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
SSM = ROOT / "data/cao_ssm.csv"
POS = ROOT / "data/cao_ssm_positions.csv"
AAS = "ACDEFGHIKLMNPQRSTVWY"

KD = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
      'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
      'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
VOL = {'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5, 'Q': 143.8, 'E': 138.4,
       'G': 60.1, 'H': 153.2, 'I': 166.7, 'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9,
       'P': 112.7, 'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0}
CHG = {'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.1}
HELIX = {'A': 1.42, 'L': 1.21, 'M': 1.45, 'E': 1.51, 'Q': 1.11, 'K': 1.16, 'R': 0.98,
         'H': 1.00, 'F': 1.13, 'I': 1.08, 'W': 1.08, 'V': 1.06, 'D': 1.01, 'T': 0.83,
         'S': 0.77, 'C': 0.70, 'N': 0.67, 'Y': 0.69, 'P': 0.57, 'G': 0.57}
AROM = set("FWY")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hot-only", type=float, default=0.6,
                    help="keep positions whose kill-rate is at least this (0 keeps all)")
    ap.add_argument("--tier", default="any")
    a = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    rows = list(csv.DictReader(open(SSM)))
    hot = {}
    for p in csv.DictReader(open(POS)):
        hot[(p["target"], p["parent"], int(p["pos"]))] = float(p["frac_killed"])

    # recover the native residue: the one amino acid NOT scanned at that position
    seen = defaultdict(set)
    for r in rows:
        if r["mut_aa"] in AAS:
            seen[(r["target"], r["parent"], int(r["pos"]))].add(r["mut_aa"])
    wt = {k: (set(AAS) - v).pop() for k, v in seen.items() if len(set(AAS) - v) == 1}
    print(f"recovered the native residue at {len(wt)} of {len(seen)} scanned positions "
          f"({100 * len(wt) / max(len(seen), 1):.0f}%)")

    TIERS = {"any": {"quite", "reasonable", "somewhat", ""},
             "somewhat": {"quite", "reasonable", "somewhat"},
             "reasonable": {"quite", "reasonable"}, "quite": {"quite"}}
    keep_t = TIERS[a.tier]

    X, y, grp, used_pos = [], [], [], set()
    for r in rows:
        pos = int(r["pos"])
        if pos == 0 or r["mut_aa"] not in AAS or r["tier"] not in keep_t:
            continue
        k = (r["target"], r["parent"], pos)
        if k not in wt or k not in hot:
            continue
        if a.hot_only and hot[k] < a.hot_only:
            continue
        w, m = wt[k], r["mut_aa"]
        if w == m:
            continue
        used_pos.add(k)
        X.append([
            KD[m] - KD[w], VOL[m] - VOL[w], CHG.get(m, 0) - CHG.get(w, 0),
            HELIX[m] - HELIX[w],
            KD[w], VOL[w], CHG.get(w, 0), HELIX[w],
            KD[m], VOL[m], CHG.get(m, 0), HELIX[m],
            float(m in AROM) - float(w in AROM),
            float(m == "P"), float(m == "G"), float(w == "P"), float(w == "G"),
            float(m == "C"), float(w == "C"),
            abs(KD[m] - KD[w]), abs(VOL[m] - VOL[w]),
        ])
        y.append(int(r["unmeasurable"]))          # 1 = this substitution abolished binding
        grp.append(r["target"])
    X, y, grp = np.array(X, float), np.array(y), np.array(grp)
    print(f"training rows {len(y)} over {len(used_pos)} positions and "
          f"{len(set(grp))} targets; {100 * y.mean():.0f}% abolish binding")
    if len(y) < 200:
        print("not enough rows"); return

    print(f"\nLEAVE-ONE-TARGET-OUT (his protocol) — hot-only >= {a.hot_only:g}, "
          f"tier '{a.tier}'\n")
    print(f"  {'held-out target':<24}{'n':>7}{'%kill':>7}{'AUC':>8}{'majority':>10}")
    aucs, shuf, ns = [], [], []
    rng = np.random.default_rng(0)
    for t in sorted(set(grp)):
        te = grp == t
        if te.sum() < 50 or len(set(y[te])) < 2 or len(set(y[~te])) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                                             max_depth=6, random_state=0)
        clf.fit(X[~te], y[~te])
        p = clf.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], p)
        maj = max(y[te].mean(), 1 - y[te].mean())
        c2 = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=6,
                                            random_state=0)
        c2.fit(X[~te], rng.permutation(y[~te]))
        shuf.append(roc_auc_score(y[te], c2.predict_proba(X[te])[:, 1]))
        aucs.append(auc); ns.append(int(te.sum()))
        print(f"  {t:<24}{te.sum():>7}{100 * y[te].mean():>6.0f}%{auc:>8.3f}{maj:>10.2f}")

    w = np.array(ns, float); w /= w.sum()
    print(f"\n  mean AUC {st.mean(aucs):.3f}   weighted {float((np.array(aucs) * w).sum()):.3f}"
          f"   median {st.median(aucs):.3f}   worst {min(aucs):.3f}   best {max(aucs):.3f}")
    print(f"  shuffled-label control: mean AUC {st.mean(shuf):.3f}")
    print(f"  folds above 0.60: {sum(1 for x in aucs if x > 0.60)}/{len(aucs)}")
    print("\n  -> " + ("GENERALISES ACROSS TARGETS. A substitution-tolerance model learned here "
                       "is exactly the single-mutation term our scorer lacks."
                       if st.mean(aucs) > 0.60 and st.mean(aucs) - st.mean(shuf) > 0.08
                       else "does not generalise across targets at this feature set."))


if __name__ == "__main__":
    main()
