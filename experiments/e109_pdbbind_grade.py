"""E109 — does PDBbind-scale data let us BEAT PPI-Affinity? (the payoff test)

872 curated PDBbind peptide complexes (crystal poses, our 16 features) + our 156. The mechanistic
prediction (e107): PPI's entire edge is the charged floor, learnable from data. Test:
  QC   — ΔG/length/charge/degeneracy of the 872 (don't repeat the cr65-vlong flat-label trap).
  A    — pooled LOCO r on combined (linear ridge vs GBT; with ~1000 complexes does GBT finally win?).
  B    — HELD-OUT head-to-head: train on (combined − shared-91), predict the 91 PPI-Affinity test
         complexes, compare to PPI 0.629; SPLIT BY CHARGE (does data close the high-charge gap 0.37→?).
Leakage control: drop any PDBbind entry whose pdb-id or sequence matches our 156 or the 91 test.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
SI = ROOT / "data" / "biolip" / "ppiaffinity_si" / "SI"
PROD = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]
SHORT = ["bsa_hyd", "mj_contact", "strength_bur"]
POS, NEG = set("KR"), set("DE")


def abscharge(seq):
    return sum(c in POS | NEG for c in seq) / max(1, len(seq))


def load_ours():
    out = []
    for fn in ["data/pooled_benchmark_train.csv", "data/pooled_benchmark_test.csv"]:
        for r in csv.DictReader(open(ROOT / fn)):
            out.append({"pdb4": r["pdb"].lower()[:4], "seq": r.get("seq", ""), "y": float(r["y"]),
                        "length": int(float(r["length"])), "src": "ours",
                        "feat": {c: float(r[c]) for c in PROD}})
    return out


def load_pdbbind():
    out = []
    for ln in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines():
        r = json.loads(ln)
        out.append({"pdb4": r["pdb"].lower()[:4], "seq": r["seq"], "y": r["y"],
                    "length": r["length"], "src": "pdbbind", "feat": {c: r[c] for c in PROD}})
    return out


def shared91():
    sh = {}
    for r in csv.DictReader(open(SI / "SI-File-6-protein-peptide-test-set-1.csv")):
        m = re.match(r"([0-9a-zA-Z]{4})", r["PDB_NAME"])
        if m:
            sh[m.group(1).lower()] = float(r["PPI-Affinity"])
    return sh


def ridge(rows, cols, lam=1.0):
    X = np.array([[r["feat"][c] for c in cols] for r in rows], float)
    y = np.array([r["y"] for r in rows])
    ok = ~np.isnan(X).any(1)
    X, y = X[ok], y[ok]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.column_stack([np.ones(len(X)), (X - mu) / sd]); R = np.eye(A.shape[1]) * lam; R[0, 0] = 0
    return mu, sd, np.linalg.solve(A.T @ A + R, A.T @ y)


def rp(feat, cols, p):
    mu, sd, w = p
    return float(np.r_[1.0, (np.array([feat[c] for c in cols]) - mu) / sd] @ w)


def gbt_fit(rows, cols):
    X = np.array([[r["feat"][c] for c in cols] for r in rows], float)
    y = np.array([r["y"] for r in rows])
    ok = ~np.isnan(X).any(1)
    return HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                         l2_regularization=2.0, min_samples_leaf=25,
                                         random_state=0).fit(X[ok], y[ok])


def R(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    return pearsonr(a[m], b[m])[0]


def main():
    ours = load_ours()
    pdbb = load_pdbbind()
    sh = shared91()
    print(f"=== E109 PDBbind payoff ({len(pdbb)} PDBbind + {len(ours)} ours) ===\n")

    # leakage control: drop PDBbind entries overlapping ours (pdb4 or seq) or the 91 test
    ours_pdb = {r["pdb4"] for r in ours}
    ours_seq = {r["seq"] for r in ours if r["seq"]}
    pdbb_clean = [r for r in pdbb if r["pdb4"] not in ours_pdb and r["pdb4"] not in sh
                  and (r["seq"] not in ours_seq)]
    print(f"  leakage filter: {len(pdbb)} → {len(pdbb_clean)} PDBbind after removing overlap w/ ours+test")

    # QC
    y = np.array([r["y"] for r in pdbb_clean])
    L = np.array([r["length"] for r in pdbb_clean])
    ch = np.array([abscharge(r["seq"]) for r in pdbb_clean])
    uq = len(set(np.round(y, 2)))
    print(f"  QC: ΔG [{y.min():.1f},{y.max():.1f}] std={y.std():.2f} UNIQUE={uq} ({uq/len(y):.0%}); "
          f"len {L.min()}-{L.max()} (med {int(np.median(L))}); abs-charge med {np.median(ch):.2f}\n")

    # ===== A. pooled LOCO on combined (subsample LOCO for speed: 5-fold by complex) =====
    comb = ours + pdbb_clean
    yk = np.array([r["y"] for r in comb])
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, len(comb))
    print("A. 5-fold CV on combined (ours+PDBbind) — does bigger data help, and does GBT now win?")
    for nm, kind in [("ridge/16", "ridge"), ("GBT/16", "gbt")]:
        pred = np.full(len(comb), np.nan)
        for k in range(5):
            tr = [comb[j] for j in range(len(comb)) if fold[j] != k]
            te = [j for j in range(len(comb)) if fold[j] == k]
            if kind == "ridge":
                p = ridge(tr, PROD)
                for j in te:
                    pred[j] = rp(comb[j]["feat"], PROD, p)
            else:
                m = gbt_fit(tr, PROD)
                Xte = np.array([[comb[j]["feat"][c] for c in PROD] for j in te], float)
                pred[te] = m.predict(Xte)
        rr = R(pred, yk)
        rmse = float(np.sqrt(np.nanmean((pred - yk) ** 2)))
        print(f"   {nm:<10} pooled r={rr:+.3f} RMSE={rmse:.2f} (n={len(comb)})")

    # ===== B. HELD-OUT head-to-head on shared-91 =====
    print("\nB. HELD-OUT: train on (ours[non-test] + PDBbind), predict the 91 PPI-Affinity test complexes")
    test = [r for r in ours if r["pdb4"] in sh]
    train = [r for r in ours if r["pdb4"] not in sh] + pdbb_clean
    train_small = [r for r in ours if r["pdb4"] not in sh]  # without PDBbind (baseline)
    yt = np.array([r["y"] for r in test])
    ppi = np.array([sh[r["pdb4"]] for r in test])

    def predict_set(trainset, kind):
        if kind == "ridge":
            p = ridge(trainset, PROD)
            return np.array([rp(r["feat"], PROD, p) for r in test])
        m = gbt_fit(trainset, PROD)
        return m.predict(np.array([[r["feat"][c] for c in PROD] for r in test], float))

    chmask = np.array([abscharge(r["seq"]) > np.median([abscharge(x["seq"]) for x in test]) for r in test])
    print(f"   {'model':<28}{'all-91 r':>10}{'low-charge':>12}{'high-charge':>13}")
    print(f"   {'PPI-Affinity (reference)':<28}{R(ppi,yt):>+10.3f}{R(ppi[~chmask],yt[~chmask]):>+12.3f}{R(ppi[chmask],yt[chmask]):>+13.3f}")
    for nm, trs, kind in [("ours 156-only ridge", train_small, "ridge"),
                          ("ours+PDBbind ridge", train, "ridge"),
                          ("ours+PDBbind GBT", train, "gbt")]:
        pr = predict_set(trs, kind)
        flag = "  <== BEATS PPI" if R(pr, yt) > R(ppi, yt) else ""
        print(f"   {nm:<28}{R(pr,yt):>+10.3f}{R(pr[~chmask],yt[~chmask]):>+12.3f}{R(pr[chmask],yt[chmask]):>+13.3f}{flag}")

    print("\n   reading: if ours+PDBbind high-charge r jumps from ~0.37 toward PPI's 0.71, the data learned")
    print("   the charged floor (e107 prediction confirmed). If pooled all-91 > 0.629 → we beat PPI-Affinity.")


if __name__ == "__main__":
    main()
