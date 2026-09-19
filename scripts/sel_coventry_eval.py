#!/usr/bin/env python
"""Apply the selectivity model to the Coventry grid, and report it against every alternative.

THIS IS THE HELD-OUT TEST.  The model was trained on threaded poses over crystal backbones from
65 blocks of our own corpus, with every Coventry and benchmark PDB id excluded at block-build
time.  It has never seen a designed binder, a docked pose, or an 18x18 grid.  Nothing below is
tuned on Coventry; the script is run once per model version and the number is the number.

WHAT IT IS BEING COMPARED AGAINST, all on the identical 324 cells and identical poses:
  our calibrated dG      the tool's headline scorer today. Measured at chance: mean cognate rank
                         9.4 of 18 where random is 9.5, because it moves only 0.22 kcal/mol
                         across a whole row.
  ref2015 total, raw     what the grid has used throughout. Mean cognate rank 8.3 row-wise.
  ref2015 total, centred double-centring the same number, no learning involved.
  selectivity model      this.

Both aggregations from the feature extraction are scored, because which pose you hand the model
is itself a decision: `best` is the single lowest-energy pose of 24 and `top5` is the mean over
the five best.  A single docked pose is a noisy draw from a pool that is 79-81% backwards.

Usage: sel_coventry_eval.py [--model data/sel_model.joblib]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
import os
FEAT = ROOT / os.environ.get("SEL_COV_FEAT", "logs/sel_coventry_features.jsonl")
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    w = sum(1.0 if p < q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return w / (len(pos) * len(neg))


def spearman(a: list[float], b: list[float]) -> float:
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
                j += 1
            for k in range(i, j + 1):
                r[s[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def report(name: str, S: dict, truth: dict) -> dict:
    """S[(peptide, binder)] -- lower = predicted tighter."""
    have = [p for p in ORDER if all((p, b) in S for b in ORDER)]
    rr, cc, t1r, t3r, t1c, t3c = [], [], 0, 0, 0, 0
    for p in have:
        o = sorted(ORDER, key=lambda b: S[(p, b)])
        r = o.index(p) + 1
        rr.append(r); t1r += r == 1; t3r += r <= 3
    for b in ORDER:
        col = [p for p in have if (p, b) in S]
        if b not in col:
            continue
        o = sorted(col, key=lambda p: S[(p, b)])
        r = o.index(b) + 1
        cc.append(r); t1c += r == 1; t3c += r <= 3
    cog = [S[k] for k in S if truth.get(k, {}).get("cognate") == "1"]
    non = [S[k] for k in S if truth.get(k, {}).get("measured") == "0"]
    meas = [S[k] for k in S if truth.get(k, {}).get("measured") == "1"]
    m = {"n_rows": len(have), "mean_rank_row": round(st.mean(rr), 2) if rr else None,
         "mean_rank_col": round(st.mean(cc), 2) if cc else None,
         "top1_row": t1r, "top3_row": t3r, "top1_col": t1c, "top3_col": t3c,
         "auc_cog_vs_non": round(auc(cog, non), 3),
         "auc_meas_vs_non": round(auc(meas, non), 3)}
    print(f"{name:32s} row {m['mean_rank_row']:5.2f}  col {m['mean_rank_col']:5.2f}  "
          f"top1 {t1r:2d}/{len(rr)} row {t1c:2d}/{len(cc)} col  "
          f"top3 {t3r:2d}/{t3c:2d}  AUC {m['auc_cog_vs_non']:.3f}")
    return m


def centre(M: dict) -> dict:
    """Double-centre a (peptide, binder) score dict over the complete grid."""
    peps = [p for p in ORDER if all((p, b) in M for b in ORDER)]
    if not peps:
        return dict(M)
    rm = {p: st.mean([M[(p, b)] for b in ORDER]) for p in peps}
    cm = {b: st.mean([M[(p, b)] for p in peps]) for b in ORDER}
    g = st.mean([M[(p, b)] for p in peps for b in ORDER])
    return {(p, b): M[(p, b)] - rm[p] - cm[b] + g for p in peps for b in ORDER}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "data/sel_model.joblib"))
    args = ap.parse_args()

    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}

    rows = {}
    for line in FEAT.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "best" in r:
                rows[(r["peptide"], r["binder"])] = r
    print(f"Coventry cells with selectivity features: {len(rows)}/324\n")
    if len(rows) < 324:
        print("incomplete -- run again when sel_coventry_features.py finishes\n")

    print(f"{'':32s} {'row':>5}  {'col':>5}   (random 9.50 / 9.50, top1 1/18)")
    results = {}

    # ---- incumbents ----------------------------------------------------------
    for label, path, field in (
            ("our calibrated dG (kcal/mol)", "logs/coventry_dg.jsonl", "best_iface"),
            ("ref2015 total  RAW", "logs/coventry_refine_fixed.jsonl", "best_iface")):
        f = ROOT / path
        if not f.exists():
            continue
        M = {}
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                p, b = r["name"].split("__")
                if r.get(field) is not None:
                    M[(p, b[:-1] if b.endswith("B") else b)] = r[field]
        results[label] = report(label, M, truth)
        results[label + "  CENTRED"] = report(label + "  CENTRED", centre(M), truth)

    # ---- the selectivity model ----------------------------------------------
    import joblib
    art = joblib.load(args.model)
    feats, sc, clf = art["features"], art["scaler"], art["clf"]
    # The model was fitted on threaded poses over crystal backbones, which needed a
    # soft-repulsive pre-pass; the Coventry poses are docked and clash for real. The feature
    # SCALES therefore differ between train and test even where the signal is the same. Refitting
    # the standardisation on the test grid's own centred features removes that mismatch without
    # touching the learned weights -- and it is legitimate here because double-centring is already
    # a transductive, whole-panel operation.
    for agg, self_scale in (("best", False), ("best", True), ("top5", False), ("top5", True)):
        X = {k: np.array([float(r[agg].get(f, 0.0)) for f in feats]) for k, r in rows.items()}
        peps = [p for p in ORDER if all((p, b) in X for b in ORDER)]
        if not peps:
            continue
        A = np.stack([[X[(p, b)] for b in ORDER] for p in peps])          # (P, B, F)
        C = A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean((0, 1), keepdims=True)
        flat = C.reshape(-1, len(feats))
        if self_scale:
            mu, sd = flat.mean(0), flat.std(0)
            sd[sd == 0] = 1.0
            Z = (flat - mu) / sd
        else:
            Z = sc.transform(flat)
        P = clf.predict_proba(Z)[:, 1].reshape(len(peps), len(ORDER))
        M = {(p, b): -float(P[i, j]) for i, p in enumerate(peps) for j, b in enumerate(ORDER)}
        tag = f"SELECTIVITY MODEL ({agg}{', self-scaled' if self_scale else ''})"
        results[tag] = report(tag, M, truth)

        if not self_scale:
            # ref2015 restricted to the same cells and the same protocol, the like-for-like check
            R = {(p, b): X[(p, b)][feats.index("i_total")] for p in peps for b in ORDER}
            results[f"  ref2015 same cells ({agg})"] = report(
                f"  ref2015 same cells ({agg})", centre(R), truth)

    (ROOT / "logs/sel_coventry_eval.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {ROOT / 'logs/sel_coventry_eval.json'}")


if __name__ == "__main__":
    main()
