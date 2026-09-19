#!/usr/bin/env python
"""Can we pick the right arm per peptide? A leave-one-out test of the obvious tweak.

THE OBSERVATION THAT SUGGESTS IT. The two arms' cognate ranks correlate with peptide chemistry
with OPPOSITE SIGNS: net charge (ours -0.234, boltz +0.322) and hydrophobic fraction (ours -0.205,
boltz +0.381). Our arm does relatively better on charged, hydrophobic peptides; the co-folding arm
does relatively better on the polar, uncharged ones. If that is real rather than eighteen points
of noise, a router could send each peptide to the arm that suits it and beat both.

WHY THIS IS THE EASIEST WAY TO FOOL YOURSELF ON THIS GRID. There are eighteen rows. A rule fitted
on all eighteen and scored on the same eighteen will look excellent no matter what, because with
two arms and a handful of descriptors there is almost always some threshold that happens to sort
them. So the rule is refitted from scratch for every held-out peptide and never sees the row it
is asked about, and the result is compared against two nulls that are constructed the same way:

  ORACLE        take whichever arm actually did better, per row. Not achievable -- it uses the
                answer -- but it bounds what any router could ever deliver.
  COIN FLIP     choose an arm at random per row, averaged over many draws. A router that does not
                beat this is doing nothing.

If honest routing lands between the coin flip and plain averaging, there is no rule to find and
averaging both arms is the right answer -- which is the conclusion worth reaching, because it is
cheaper and cannot overfit.

Usage: coventry_routing_test.py
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


def rowrank(S):
    R = np.zeros((N, N), dtype=int)
    for i in range(N):
        for k, j in enumerate(sorted(range(N), key=lambda j: S[i, j])):
            R[i, j] = k + 1
    return R


def main() -> None:
    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    seqs = {p: next(r["peptide_seq"] for (pp, _), r in truth.items() if pp == p) for p in ORDER}

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
    Ro, Rc = rowrank(_z(OURS)), rowrank(_z(COF) + _z(_mc(IPT)))
    ro = [Ro[i, i] for i in range(N)]
    rc = [Rc[i, i] for i in range(N)]
    rav = [rowrank((Ro + Rc).astype(float))[i, i] for i in range(N)]

    X = []
    for p in ORDER:
        s = seqs[p]
        c = Counter(s)
        X.append([len(s),
                  -sum((v / len(s)) * math.log2(v / len(s)) for v in c.values()),
                  sum(s.count(x) for x in "KR") - sum(s.count(x) for x in "DE"),
                  sum(s.count(x) for x in PHOB) / len(s),
                  sum(s.count(x) for x in "KRDE") / len(s)])
    X = np.array(X)
    names = ["length", "entropy", "net charge", "hydrophobic frac", "charged frac"]

    rng = np.random.default_rng(0)
    coin = st.mean(st.mean(ro[i] if rng.integers(2) else rc[i] for i in range(N))
                   for _ in range(2000))
    oracle = [min(ro[i], rc[i]) for i in range(N)]

    print("MEAN COGNATE RANK, lower is better.  18 rows, so every number here is noisy.\n")
    print(f"  {'strategy':<46}{'rank':>7}{'t5':>5}")
    print(f"  {'our arm always':<46}{st.mean(ro):>7.2f}{sum(r <= 5 for r in ro):>5}")
    print(f"  {'boltz arm always':<46}{st.mean(rc):>7.2f}{sum(r <= 5 for r in rc):>5}")
    print(f"  {'coin flip per row (null)':<46}{coin:>7.2f}{'-':>5}")
    print(f"  {'RANK AVERAGE of both arms':<46}{st.mean(rav):>7.2f}"
          f"{sum(r <= 5 for r in rav):>5}")
    print(f"  {'ORACLE: best arm per row (unachievable)':<46}{st.mean(oracle):>7.2f}"
          f"{sum(r <= 5 for r in oracle):>5}")

    print("\nHONEST ROUTERS, leave-one-out: the rule never sees the row it is asked about.\n")
    print(f"  {'router on':<46}{'rank':>7}{'t5':>5}{'correct calls':>15}")
    best_is_ours = np.array([ro[i] <= rc[i] for i in range(N)])
    for k, nm in enumerate(names):
        picked, correct = [], 0
        for i in range(N):
            tr = [t for t in range(N) if t != i]
            a = [X[t, k] for t in tr if best_is_ours[t]]
            b = [X[t, k] for t in tr if not best_is_ours[t]]
            if not a or not b:
                picked.append(rav[i]); continue
            use_ours = abs(X[i, k] - st.mean(a)) < abs(X[i, k] - st.mean(b))
            correct += use_ours == best_is_ours[i]
            picked.append(ro[i] if use_ours else rc[i])
        print(f"  {nm:<46}{st.mean(picked):>7.2f}{sum(r <= 5 for r in picked):>5}"
              f"{correct:>10}/{N}")

    # the one signal that is not peptide chemistry: each arm's own confidence in that row
    print("\n  a router on each arm's OWN diagnostic rather than on peptide chemistry:")
    dis = [float(_z(OURS)[i, i] - _z(COF)[i, i]) for i in range(N)]
    ipt = [aux.get((ORDER[i], ORDER[i]), {}).get("iptm") or 0.0 for i in range(N)]
    for nm, v, flip in (("pose disagreement at the cognate", dis, False),
                        ("co-folding ipTM", ipt, True)):
        picked, correct = [], 0
        for i in range(N):
            tr = [t for t in range(N) if t != i]
            a = [v[t] for t in tr if best_is_ours[t]]
            b = [v[t] for t in tr if not best_is_ours[t]]
            if not a or not b:
                picked.append(rav[i]); continue
            use_ours = abs(v[i] - st.mean(a)) < abs(v[i] - st.mean(b))
            correct += use_ours == best_is_ours[i]
            picked.append(ro[i] if use_ours else rc[i])
        print(f"  {nm:<46}{st.mean(picked):>7.2f}{sum(r <= 5 for r in picked):>5}"
              f"{correct:>10}/{N}")

    print(f"\n  chance for 'correct calls' is 9/18. Averaging needs no call at all and gets "
          f"{st.mean(rav):.2f}.")


if __name__ == "__main__":
    main()
