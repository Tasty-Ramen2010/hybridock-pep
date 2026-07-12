"""E116 — grade the GPU MD free-state entropy (s_free): does computing the missing physics fix vlong?

The atlas (e114) diagnosed conformational entropy as the missing physics, strongest in vlong (≥17).
e115 computed real MD s_free for 922 peptides. Test: does adding s_free (and s_free×buried) recover the
failure regimes — especially vlong — in a leave-one-complex-out / 5-fold grade? Sign check: floppier
peptide (higher s_free) → larger −TΔS penalty → WEAKER binding (ΔG less negative) → corr(s_free, y) > 0.
Run AFTER e115_md_sfree.py completes (or partial, it grades whatever's done).
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
PROD = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]


def seqhash(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]


def band(L):
    return "short≤8" if L <= 8 else "med9-12" if L <= 12 else "long13-16" if L <= 16 else "vlong≥17"


def load():
    sfree = {}
    for ln in (ROOT / "data/sfree_results.jsonl").read_text().splitlines():
        r = json.loads(ln)
        sfree[r["hash"]] = r
    rows = []
    for fn in ["data/pooled_benchmark_train.csv", "data/pooled_benchmark_test.csv"]:
        for r in csv.DictReader(open(ROOT / fn)):
            s = r.get("seq", "")
            sf = sfree.get(seqhash(s.upper()))
            rows.append({"seq": s, "y": float(r["y"]), "length": int(float(r["length"])),
                         "feat": {c: float(r[c]) for c in PROD}, "sfree": sf})
    oseq = {r["seq"] for r in rows if r["seq"]}
    for ln in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if r["seq"] in oseq:
            continue
        oseq.add(r["seq"])
        sf = sfree.get(seqhash(r["seq"].upper()))
        rows.append({"seq": r["seq"], "y": r["y"], "length": r["length"],
                     "feat": {c: r[c] for c in PROD}, "sfree": sf})
    return [r for r in rows if r["sfree"]]  # only those with computed s_free


def cv(rows, cols, k=5, seed=0):
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, k, len(rows))
    X = np.array([[r["x"][c] for c in cols] for r in rows], float)
    y = np.array([r["y"] for r in rows])
    pred = np.full(len(rows), np.nan)
    for f in range(k):
        tr = fold != f
        m = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                          l2_regularization=2.0, min_samples_leaf=20, random_state=seed).fit(X[tr], y[tr])
        pred[fold == f] = m.predict(X[fold == f])
    return pred, y


def st(p, y, m=None):
    p, y = (p, y) if m is None else (p[m], y[m])
    ok = ~(np.isnan(p) | np.isnan(y))
    return (pearsonr(p[ok], y[ok])[0], float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))), int(ok.sum())) if ok.sum() > 4 else (np.nan, np.nan, int(ok.sum()))


def main():
    rows = load()
    print(f"=== E116 s_free grading (n={len(rows)} with computed MD entropy) ===\n")
    for r in rows:
        bf = 1 - r["feat"]["org_density"]  # proxy for buried/ordered fraction lost on binding
        r["x"] = dict(r["feat"], s_free=r["sfree"]["s_free"], s_free_tot=r["sfree"]["s_free_total"],
                      rmsf=r["sfree"]["rmsf"], s_free_x_burial=r["sfree"]["s_free"] * r["feat"]["mean_burial"])
    y = np.array([r["y"] for r in rows])
    sfv = np.array([r["sfree"]["s_free"] for r in rows])
    print(f"  corr(s_free, ΔG) raw = {pearsonr(sfv, y)[0]:+.3f}  (expect >0: floppier→weaker)")
    L = np.array([r["length"] for r in rows])
    for b in ["short≤8", "med9-12", "long13-16", "vlong≥17"]:
        m = np.array([band(x) == b for x in L])
        if m.sum() >= 8:
            print(f"     {b:<11} corr(s_free,ΔG)={pearsonr(sfv[m],y[m])[0]:+.3f} (n={m.sum()})")

    print("\n  GBT 5-fold, by length band:  base16 → +s_free")
    pb, _ = cv(rows, PROD)
    ps, _ = cv(rows, PROD + ["s_free", "s_free_tot", "rmsf", "s_free_x_burial"])
    for lab, m in [("ALL", np.ones(len(rows), bool))] + [(b, np.array([band(x) == b for x in L])) for b in ["short≤8", "med9-12", "long13-16", "vlong≥17"]]:
        if m.sum() >= 8:
            rb = st(pb, y, m); rs = st(ps, y, m)
            print(f"     {lab:<11} n={m.sum():<4} base={rb[0]:+.3f} → +s_free={rs[0]:+.3f}  Δ={rs[0]-rb[0]:+.3f}")
    print("\n  reading: if +s_free lifts vlong/long r, the computed MD entropy recovers the missing physics.")


if __name__ == "__main__":
    main()
