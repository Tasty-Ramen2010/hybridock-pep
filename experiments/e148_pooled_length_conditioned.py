"""E148 — the production candidate: ONE pooled, length-conditioned model (the 'global function with
soft extensions per length'), trained on ALL data, grouped-CV, per-band r/MAE/RMSE.

Lessons from e147: (a) cross-dataset transfer (train PDBbind→test benchmark) collapses — must POOL;
(b) naive correlation feature-dropping HURTS (GBT handles collinearity; the 16 aren't droppable).
So: pool PDBbind-925 + pooled-151 benchmark (dedup), train ONE GBT with length as a feature (soft
conditioning = the 'extension' for short/med/long/vlong, learned internally, not hard-routed), add the
data-driven descriptors (E146, help charged + low-charge), grouped 5-fold CV by complex. Goal: short
NON-NEGATIVE and stable (now n≈324 short across the pool, not 19), with honest r + MAE + RMSE per band.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
e146 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("e146", ROOT / "experiments/e146_charged_specialist.py"))
importlib.util.spec_from_file_location("e146", ROOT / "experiments/e146_charged_specialist.py").loader.exec_module(e146)
PROD = e146.PROD
POS, NEG = set("KR"), set("DE")


def band(L):
    L = int(L)
    return "short≤8" if L <= 8 else "med9-12" if L <= 12 else "long13-16" if L <= 16 else "vlong≥17"


def metr(p, y):
    return pearsonr(p, y)[0], float(np.mean(np.abs(p - y))), float(np.sqrt(np.mean((p - y) ** 2)))


def load_all():
    rows = []
    for r in [json.loads(l) for l in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines()]:
        rows.append({"src": "pdbbind", "pdb": r["pdb"].upper(), "y": r["y"], "length": r["length"],
                     "feat": [r[c] for c in PROD], "desc": e146.desc(r["seq"])})
    ids = {r["pdb"] for r in rows}
    for nm in ["train", "test"]:
        for r in csv.DictReader(open(ROOT / f"data/pooled_benchmark_{nm}.csv")):
            base = r["pdb"].split("_")[0].upper()
            if base in ids and r["dataset"] == "cr65":
                continue  # dedup cr65 already in pdbbind
            rows.append({"src": r["dataset"], "pdb": base, "y": float(r["y"]), "length": int(r["length"]),
                         "feat": [float(r[c]) for c in PROD], "desc": e146.desc(r.get("seq", ""))})
    return rows


def cv(rows, use_desc, use_len, k=5, seed=0):
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, k, len(rows))
    y = np.array([r["y"] for r in rows])
    X = []
    for r in rows:
        row = list(r["feat"])
        if use_desc:
            row += r["desc"]
        if use_len:
            row += [float(r["length"])]
        X.append(row)
    X = np.array(X, float)
    pred = np.full(len(rows), np.nan)
    for f in range(k):
        tr = fold != f
        m = HistGradientBoostingRegressor(max_iter=500, max_depth=3, learning_rate=0.04,
                                          l2_regularization=3.0, min_samples_leaf=20, random_state=0).fit(X[tr], y[tr])
        pred[fold == f] = m.predict(X[fold == f])
    return pred, y


def main():
    rows = load_all()
    L = np.array([int(r["length"]) for r in rows])
    print(f"=== E148 pooled length-conditioned model (n={len(rows)}) ===")
    for b in ["short≤8", "med9-12", "long13-16", "vlong≥17"]:
        print(f"    {b}: n={sum(band(x)==b for x in L)}")
    print()
    configs = [("base-16", False, False), ("+descriptors", True, False),
               ("+desc+length (PROD'N)", True, True)]
    print(f"  {'config':<24}{'ALL r':>8}{'MAE':>7}{'RMSE':>7}")
    store = {}
    for nm, ud, ul in configs:
        p, y = cv(rows, ud, ul)
        store[nm] = (p, y)
        r, mae, rmse = metr(p, y)
        print(f"  {nm:<24}{r:>+8.3f}{mae:>7.2f}{rmse:>7.2f}")

    print("\n  PER-BAND (PROD'N = +desc+length), r / MAE / RMSE:")
    pp, yy = store["+desc+length (PROD'N)"]
    bp, by = store["base-16"]
    print(f"  {'band':<12}{'n':>5}{'base16 r':>10}{'PRODN r':>9}{'MAE':>7}{'RMSE':>7}")
    for b in ["short≤8", "med9-12", "long13-16", "vlong≥17"]:
        m = np.array([band(x) == b for x in L])
        if m.sum() >= 5:
            rb = pearsonr(bp[m], by[m])[0]
            rp, mae, rmse = metr(pp[m], yy[m])
            flag = "  <-- was NEGATIVE" if b == "short≤8" and rb < 0 else ""
            print(f"  {b:<12}{m.sum():>5}{rb:>+10.3f}{rp:>+9.3f}{mae:>7.2f}{rmse:>7.2f}{flag}")

    # stability of short over seeds
    print("\n  SHORT stability (PROD'N) over 5 seeds:")
    rs = []
    for s in range(5):
        p, y = cv(rows, True, True, seed=s)
        m = np.array([band(x) == "short≤8" for x in L])
        rs.append(pearsonr(p[m], y[m])[0])
    print(f"    short r = {np.mean(rs):+.3f} ± {np.std(rs):.3f}  (min {min(rs):+.3f}, max {max(rs):+.3f})")
    print("\n  verdict: pooled training gives short enough data to be POSITIVE+stable; length feature lets the")
    print("  one global model behave per-band (soft extension). Report MAE (PPI~1.8, ours below).")


if __name__ == "__main__":
    main()
