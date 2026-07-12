"""E126 — kcal/mol × length × RMSE breakdown on ALL pooled PDBs + length-specific long/vlong sub-models.

Ram: (1) where are we failing now (full breakdown), (2) does length-specific scoring help long/vlong like
the short router does (now that PDBbind gave us enough long/vlong to fit them)?

PART 1  GLOBAL model (GBT 5-fold) per length band: n, ΔG range, r, RMSE, mean|err| — the failure map.
PART 2  LENGTH-SPECIFIC sub-models: for each band, compare GLOBAL-on-band vs a sub-model trained ONLY on
        that band (proper grouped CV, no leakage). Does specialization beat the global model per band?
PART 3  ROUTED model: route each complex to its band sub-model; pooled + per-band r/RMSE vs global.
PART 4  optional entropy feature for long/vlong (peptides with computed MD entropy).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROD = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]
BANDS = ["short≤8", "med9-12", "long13-16", "vlong≥17"]


def band(L):
    return "short≤8" if L <= 8 else "med9-12" if L <= 12 else "long13-16" if L <= 16 else "vlong≥17"


def load():
    rows = []
    for fn in ["data/pooled_benchmark_train.csv", "data/pooled_benchmark_test.csv"]:
        for r in csv.DictReader(open(ROOT / fn)):
            rows.append({"seq": r.get("seq", ""), "y": float(r["y"]), "length": int(float(r["length"])),
                         "feat": [float(r[c]) for c in PROD]})
    oseq = {r["seq"] for r in rows if r["seq"]}
    for ln in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if r["seq"] in oseq:
            continue
        oseq.add(r["seq"])
        rows.append({"seq": r["seq"], "y": r["y"], "length": r["length"], "feat": [r[c] for c in PROD]})
    return rows


def gbt(Xtr, ytr, seed=0):
    return HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                         l2_regularization=2.0, min_samples_leaf=20, random_state=seed).fit(Xtr, ytr)


def st(p, y, m=None):
    p, y = (p, y) if m is None else (p[m], y[m])
    ok = ~(np.isnan(p) | np.isnan(y))
    if ok.sum() < 5:
        return (np.nan, np.nan, int(ok.sum()))
    return (pearsonr(p[ok], y[ok])[0], float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))), int(ok.sum()))


def main():
    rows = load()
    X = np.array([r["feat"] for r in rows], float)
    y = np.array([r["y"] for r in rows])
    L = np.array([r["length"] for r in rows])
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, len(rows))
    print(f"=== E126 length × kcal/mol × RMSE breakdown (n={len(rows)}) ===\n")

    # PART 1: global model per band
    glob = np.full(len(rows), np.nan)
    for f in range(5):
        tr = fold != f
        m = gbt(X[tr], y[tr])
        glob[fold == f] = m.predict(X[fold == f])
    print("PART 1 — GLOBAL GBT (5-fold) per length band:")
    print(f"{'band':<12}{'n':>5}{'ΔG range':>16}{'std':>7}{'r':>8}{'RMSE':>7}{'mean|err|':>11}")
    for b in BANDS:
        m = np.array([band(x) == b for x in L])
        if m.sum() < 5:
            print(f"{b:<12}{m.sum():>5}  (too few)")
            continue
        r, rmse, n = st(glob, y, m)
        yb = y[m]
        err = np.abs(glob[m] - yb)
        print(f"{b:<12}{n:>5}  [{yb.min():>5.1f},{yb.max():>5.1f}]{yb.std():>7.2f}{r:>+8.2f}{rmse:>7.2f}{np.nanmean(err):>11.2f}")
    rall, rmseall, _ = st(glob, y)
    print(f"{'ALL':<12}{len(rows):>5}{'':>16}{y.std():>7.2f}{rall:>+8.2f}{rmseall:>7.2f}")

    # PART 2: length-specific sub-models (within-band grouped CV)
    print("\nPART 2 — LENGTH-SPECIFIC sub-models (train only on that band) vs global, per band:")
    print(f"{'band':<12}{'n':>5}{'global r':>10}{'specific r':>12}{'glob RMSE':>11}{'spec RMSE':>11}{'verdict':>14}")
    routed = np.full(len(rows), np.nan)
    for b in BANDS:
        idx = np.array([band(x) == b for x in L])
        if idx.sum() < 25:
            # too few for own model → keep global
            routed[idx] = glob[idx]
            print(f"{b:<12}{idx.sum():>5}  (n<25: route to global)")
            continue
        bi = np.where(idx)[0]
        bfold = fold[bi]
        spec = np.full(len(bi), np.nan)
        for f in range(5):
            tr = bi[bfold != f]
            te = bi[bfold == f]
            if len(tr) < 15:
                continue
            mm = gbt(X[tr], y[tr])
            spec[bfold == f] = mm.predict(X[te])
        routed[bi] = spec
        rg, rmg, _ = st(glob, y, idx)
        rs, rms, _ = st(spec, y[bi])
        verdict = "SPECIFIC WINS" if (rs > rg + 0.02) else "global better" if rg > rs + 0.02 else "≈tie"
        print(f"{b:<12}{idx.sum():>5}{rg:>+10.2f}{rs:>+12.2f}{rmg:>11.2f}{rms:>11.2f}{verdict:>14}")

    # PART 3: routed pooled
    print("\nPART 3 — ROUTED model (each complex → its band sub-model) vs GLOBAL:")
    rr, rmr, _ = st(routed, y)
    print(f"   routed pooled r={rr:+.3f} RMSE={rmr:.2f}  |  global pooled r={rall:+.3f} RMSE={rmseall:.2f}  "
          f"Δr={rr-rall:+.3f}")
    for b in BANDS:
        m = np.array([band(x) == b for x in L])
        if m.sum() >= 5:
            rrb = st(routed, y, m); rgb = st(glob, y, m)
            print(f"     {b:<11} routed r={rrb[0]:+.2f} (RMSE {rrb[1]:.2f})  vs global r={rgb[0]:+.2f} (RMSE {rgb[1]:.2f})")

    # PART 4: entropy feature for long/vlong (peptides with computed MD entropy)
    sf_path = ROOT / "data/sfree_perres.jsonl"
    if sf_path.exists():
        def seqhash(s):
            return hashlib.md5(s.encode()).hexdigest()[:12]
        sfree = {}
        for ln in sf_path.read_text().splitlines():
            r = json.loads(ln)
            v = [e for e in r["per_res_entropy"] if e is not None]
            sfree[r["hash"]] = (float(np.mean(v)) if v else 0.0, float(np.sum(v)) if v else 0.0)
        have = [(i, sfree[seqhash(rows[i]["seq"].upper())]) for i in range(len(rows)) if seqhash(rows[i]["seq"].upper()) in sfree]
        longv = [(i, s) for i, s in have if rows[i]["length"] >= 13]
        print(f"\nPART 4 — entropy feature, long+vlong with computed MD entropy (n={len(longv)}):")
        if len(longv) >= 20:
            ii = [i for i, _ in longv]
            Xe = np.array([rows[i]["feat"] + list(sfree[seqhash(rows[i]["seq"].upper())]) for i in ii], float)
            Xb = np.array([rows[i]["feat"] for i in ii], float)
            ye = y[ii]
            fe = rng.integers(0, 5, len(ii))
            pe = np.full(len(ii), np.nan); pb = np.full(len(ii), np.nan)
            for f in range(5):
                tr = fe != f
                pe[fe == f] = gbt(Xe[tr], ye[tr]).predict(Xe[fe == f])
                pb[fe == f] = gbt(Xb[tr], ye[tr]).predict(Xb[fe == f])
            print(f"   base r={st(pb,ye)[0]:+.3f} → +entropy r={st(pe,ye)[0]:+.3f}  Δ={st(pe,ye)[0]-st(pb,ye)[0]:+.3f}")
        else:
            print(f"   only {len(longv)} long/vlong with entropy — wait for more MD.")


if __name__ == "__main__":
    main()
