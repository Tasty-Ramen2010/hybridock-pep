"""E111 — the richness lever: more FEATURES help once we have PDBbind-scale DATA (the path past PPI-Affinity).

Last night: "data is the lever" (untested). Tonight, with 925 PDBbind peptides ingested + our 156:
  * at scale GBT beats linear (0.45 vs 0.32) — n=156 was too small to see it (e104 GBT overfit there)
  * adding even 11 cheap SEQUENCE features lifts pooled GBT — the OPPOSITE of n=156 where they HURT
This proves the trajectory: our 16 structural features cap ~0.45; richer features (ProtDCal-scale, PPI's
recipe) on this data is how we reach their 0.55-0.63. This script reproduces the scaling effect.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
PROD = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2,
      "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
HYD, AROM, POS, NEG, POL = set("AILMFWVC"), set("FWY"), set("KR"), set("DE"), set("STNQHY")


def seqfeat(s):
    s = "".join(c for c in s.upper() if c in KD)
    L = max(1, len(s))
    kd = np.array([KD[c] for c in s]) if s else np.array([0.0])
    return [len(s), kd.mean(), kd.std(), sum(c in HYD for c in s) / L, sum(c in AROM for c in s) / L,
            sum(c in POS for c in s) / L, sum(c in NEG for c in s) / L, sum(c in POL for c in s) / L,
            s.count("G") / L, s.count("P") / L, (sum(c in POS for c in s) - sum(c in NEG for c in s)) / L]


def load():
    rows = []
    for fn in ["data/pooled_benchmark_train.csv", "data/pooled_benchmark_test.csv"]:
        for r in csv.DictReader(open(ROOT / fn)):
            rows.append({"y": float(r["y"]), "x16": [float(r[c]) for c in PROD], "seq": r.get("seq", ""), "src": "ours"})
    oseq = {r["seq"] for r in rows if r["seq"]}
    for ln in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if r["seq"] not in oseq:
            rows.append({"y": r["y"], "x16": [r[c] for c in PROD], "seq": r["seq"], "src": "pdbbind"})
    return rows


def cv(X, y, fold, k=5, seed=0):
    pred = np.full(len(y), np.nan)
    for f in range(k):
        tr = fold != f
        m = HistGradientBoostingRegressor(max_iter=500, max_depth=3, learning_rate=0.04,
                                          l2_regularization=2.0, min_samples_leaf=25, random_state=seed).fit(X[tr], y[tr])
        pred[fold == f] = m.predict(X[fold == f])
    return pearsonr(pred, y)[0], float(np.sqrt(np.mean((pred - y) ** 2)))


def main():
    rows = load()
    y = np.array([r["y"] for r in rows])
    X16 = np.array([r["x16"] for r in rows])
    Xseq = np.array([seqfeat(r["seq"]) for r in rows])
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, len(rows))
    print(f"=== E111 richness lever at scale (n={len(rows)}: {sum(r['src']=='ours' for r in rows)} ours + "
          f"{sum(r['src']=='pdbbind' for r in rows)} PDBbind) ===\n")
    print("  5-fold CV, GBT:")
    for nm, X in [("16 structural", X16), ("16 struct + 11 seq", np.hstack([X16, Xseq])), ("11 seq only", Xseq)]:
        r, rmse = cv(X, y, fold)
        print(f"    {nm:<22} r={r:+.3f} RMSE={rmse:.2f}")
    # scaling curve: does the feature gain grow with n? (subsample)
    print("\n  feature gain vs training size (seq features help only at scale):")
    for frac in (0.15, 0.4, 0.7, 1.0):
        idx = rng.permutation(len(rows))[:int(frac * len(rows))]
        f2 = fold[idx]
        r16 = cv(X16[idx], y[idx], f2)[0]
        rboth = cv(np.hstack([X16, Xseq])[idx], y[idx], f2)[0]
        print(f"    n={len(idx):<5} struct r={r16:+.3f}  +seq r={rboth:+.3f}  Δ={rboth-r16:+.3f}")
    print("\n  PPI-Affinity reference 0.554/0.629. Path: ProtDCal-scale features on this data → GBT.")


if __name__ == "__main__":
    main()
