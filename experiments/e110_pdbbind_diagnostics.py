"""E110 — WHY did pooling PDBbind hurt? Disambiguate artifact vs real generalization wall.

e109: adding 834 PDBbind peptides dropped pooled CV (0.587→0.29 ridge) and the held-out the98 prediction
went negative. Before concluding "data doesn't help," rule out the confounds:

D0  FEATURE-SCALE SHIFT: are PDBbind features (mol2-converted peptides) on the same scale as ours
    (our pipeline)? If systematically shifted, pooling breaks for a PREP reason, not a science reason.
D1  WITHIN-PDBbind generalization: 5-fold CV on the 834 alone (ridge, GBT). Does the 16-feature signal
    exist at scale within ONE consistent distribution? (If ~0.5 → features fine; failure was cross-dist.)
D2  DOES PDBbind HELP same-distribution? the98 5-fold, train WITH vs WITHOUT PDBbind added. If it helps
    when the98 is also represented in training → data lever real; if not → cross-dataset noise dominates.
D3  WITHIN-PDBbind by charge: does high-charge work at scale (the e107 charged-floor question)?
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
SI = ROOT / "data" / "biolip" / "ppiaffinity_si" / "SI"
PROD = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]
POS, NEG = set("KR"), set("DE")


def absch(s):
    return sum(c in POS | NEG for c in s) / max(1, len(s))


def load_ours():
    out = []
    for fn in ["data/pooled_benchmark_train.csv", "data/pooled_benchmark_test.csv"]:
        for r in csv.DictReader(open(ROOT / fn)):
            out.append({"pdb4": r["pdb"].lower()[:4], "seq": r.get("seq", ""), "y": float(r["y"]),
                        "length": int(float(r["length"])), "dataset": r["dataset"],
                        "feat": {c: float(r[c]) for c in PROD}})
    return out


def load_pb():
    out = []
    for ln in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines():
        r = json.loads(ln)
        out.append({"pdb4": r["pdb"].lower()[:4], "seq": r["seq"], "y": r["y"], "length": r["length"],
                    "dataset": "pdbbind", "feat": {c: r[c] for c in PROD}})
    return out


def cv(rows, kind, k=5, cols=PROD, seed=0):
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, k, len(rows))
    y = np.array([r["y"] for r in rows])
    pred = np.full(len(rows), np.nan)
    for f in range(k):
        tr = [rows[j] for j in range(len(rows)) if fold[j] != f]
        te = [j for j in range(len(rows)) if fold[j] == f]
        Xtr = np.array([[r["feat"][c] for c in cols] for r in tr], float)
        ytr = np.array([r["y"] for r in tr])
        ok = ~np.isnan(Xtr).any(1)
        if kind == "ridge":
            mu, sd = Xtr[ok].mean(0), Xtr[ok].std(0) + 1e-9
            A = np.column_stack([np.ones(ok.sum()), (Xtr[ok] - mu) / sd])
            Rm = np.eye(A.shape[1]); Rm[0, 0] = 0
            w = np.linalg.solve(A.T @ A + Rm, A.T @ ytr[ok])
            for j in te:
                x = np.array([rows[j]["feat"][c] for c in cols])
                pred[j] = np.r_[1.0, (x - mu) / sd] @ w
        else:
            m = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                              l2_regularization=2.0, min_samples_leaf=25,
                                              random_state=seed).fit(Xtr[ok], ytr[ok])
            pred[te] = m.predict(np.array([[rows[j]["feat"][c] for c in cols] for j in te], float))
    m = ~(np.isnan(pred) | np.isnan(y))
    return pearsonr(pred[m], y[m])[0], float(np.sqrt(np.mean((pred[m] - y[m]) ** 2)))


def main():
    ours = load_ours()
    pb = load_pb()
    ours_pdb = {r["pdb4"] for r in ours}
    ours_seq = {r["seq"] for r in ours if r["seq"]}
    pb = [r for r in pb if r["pdb4"] not in ours_pdb and r["seq"] not in ours_seq]
    print(f"=== E110 diagnostics ({len(pb)} PDBbind, {len(ours)} ours) ===\n")

    print("D0. FEATURE-SCALE SHIFT (mean±std by source; |Δmean|/σ_ours > 0.5 = shifted → pooling confound):")
    flag = 0
    for c in PROD:
        a = np.array([r["feat"][c] for r in ours], float)
        b = np.array([r["feat"][c] for r in pb], float)
        d = abs(np.nanmean(a) - np.nanmean(b)) / (np.nanstd(a) + 1e-9)
        tag = "  <== SHIFTED" if d > 0.5 else ""
        if d > 0.5:
            flag += 1
        print(f"   {c:<14} ours {np.nanmean(a):8.2f}±{np.nanstd(a):6.2f}  pdbbind {np.nanmean(b):8.2f}±{np.nanstd(b):6.2f}  Δ/σ={d:.2f}{tag}")
    print(f"   → {flag}/16 features shifted >0.5σ. {'POOLING CONFOUNDED by prep mismatch' if flag >= 4 else 'scales broadly compatible'}\n")

    print("D1. WITHIN-PDBbind 5-fold CV (one consistent distribution — does the 16-feat signal scale?):")
    for kind in ["ridge", "gbt"]:
        r, rmse = cv(pb, kind)
        print(f"   {kind:<6} r={r:+.3f} RMSE={rmse:.2f} (n={len(pb)})")

    print("\nD2. DOES PDBbind HELP the98? (the98 5-fold, train WITH vs WITHOUT PDBbind in training folds)")
    the98 = [r for r in ours if r["dataset"] == "the98"]
    y = np.array([r["y"] for r in the98])
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, len(the98))
    for label, add in [("the98 only", []), ("the98 + PDBbind", pb)]:
        for kind in ["ridge", "gbt"]:
            pred = np.full(len(the98), np.nan)
            for f in range(5):
                tr = [the98[j] for j in range(len(the98)) if fold[j] != f] + add
                te = [j for j in range(len(the98)) if fold[j] == f]
                Xtr = np.array([[r["feat"][c] for c in PROD] for r in tr], float)
                ytr = np.array([r["y"] for r in tr]); ok = ~np.isnan(Xtr).any(1)
                if kind == "ridge":
                    mu, sd = Xtr[ok].mean(0), Xtr[ok].std(0) + 1e-9
                    A = np.column_stack([np.ones(ok.sum()), (Xtr[ok] - mu) / sd]); Rm = np.eye(A.shape[1]); Rm[0, 0] = 0
                    w = np.linalg.solve(A.T @ A + Rm, A.T @ ytr[ok])
                    for j in te:
                        x = np.array([the98[j]["feat"][c] for c in PROD]); pred[j] = np.r_[1.0, (x - mu) / sd] @ w
                else:
                    m = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                                      l2_regularization=2.0, min_samples_leaf=25, random_state=0).fit(Xtr[ok], ytr[ok])
                    pred[te] = m.predict(np.array([[the98[j]["feat"][c] for c in PROD] for j in te], float))
            mk = ~np.isnan(pred)
            print(f"   {label:<18} {kind:<6} the98 r={pearsonr(pred[mk], y[mk])[0]:+.3f}")

    print("\nD3. WITHIN-PDBbind by charge (does high-charge predict at scale — the charged-floor question):")
    ch = np.array([absch(r["seq"]) for r in pb]); med = np.median(ch)
    for lab, m in [("low-charge", ch <= med), ("high-charge", ch > med)]:
        sub = [pb[i] for i in range(len(pb)) if m[i]]
        r, rmse = cv(sub, "gbt")
        print(f"   {lab:<12} n={len(sub):<4} GBT r={r:+.3f}")
    print("\n  VERDICT logic: D1 high → features scale within-dist, e109 failure was cross-dist (fixable by")
    print("  pooling the98 in train: see D2). D2 'the98+PDBbind' > 'the98 only' → DATA LEVER REAL. D3 high")
    print("  high-charge → charged floor IS learnable at scale (e107 confirmed). All low → features are the cap.")


if __name__ == "__main__":
    main()
