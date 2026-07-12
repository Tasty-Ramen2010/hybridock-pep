"""E104 — physics+ML combo to BEAT PPI-Affinity (bar = r 0.629 on shared 91; reported 0.554 on T100).

Hypothesis for the gap (ours 0.451 vs PPI 0.629 on the diverse subset): PPI-Affinity uses thousands of
ProtDCal SEQUENCE+structure descriptors via SVM; we use 16 STRUCTURAL features. The missing richness is
likely peptide sequence physicochemistry. Test: add cheap, intensive, transferable sequence features +
nonlinear ML (GBT), validated leave-one-complex-out (never in-sample), permutation-checked.

Models compared (all LOCO):
  M0  ridge / 16 structural          (production baseline)
  M1  GBT   / structural             (linear→nonlinear, same features: is the gap the MODEL?)
  M2  GBT   / structural + sequence  (is the gap missing SEQUENCE features?)
  M3  stack: physics-ridge pred + sequence → GBT  (physics+ML combo)
Eval on: pooled 156 LOCO (vs PPI reported 0.554) AND shared-91 (vs PPI measured 0.629).
Honesty: 5x repeated permutation test on the best model; report if the gain is real or noise.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
SI = ROOT / "data" / "biolip" / "ppiaffinity_si" / "SI"
STRUCT = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
          "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density",
          "cys_frac", "net_dewet", "polar_desolv"]
SHORT = ["bsa_hyd", "mj_contact", "strength_bur"]

KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2,
      "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
      "Y": -1.3, "V": 4.2}
HYDRO = set("AILMFWVC")
AROM = set("FWY")
POS = set("KR")
NEG = set("DE")
POLAR = set("STNQHY")


def seq_features(seq):
    seq = "".join(c for c in seq.upper() if c in KD)
    L = max(1, len(seq))
    kd = np.array([KD[c] for c in seq]) if seq else np.array([0.0])
    return {
        "seq_len": float(len(seq)),
        "kd_mean": float(kd.mean()),
        "kd_std": float(kd.std()),
        "frac_hyd": sum(c in HYDRO for c in seq) / L,
        "frac_arom": sum(c in AROM for c in seq) / L,
        "frac_pos": sum(c in POS for c in seq) / L,
        "frac_neg": sum(c in NEG for c in seq) / L,
        "frac_polar": sum(c in POLAR for c in seq) / L,
        "frac_gly": seq.count("G") / L,
        "frac_pro": seq.count("P") / L,
        "net_charge_L": (sum(c in POS for c in seq) - sum(c in NEG for c in seq)) / L,
    }


SEQF = list(seq_features("ACDEFG").keys())


def load():
    rows = []
    for fn in ["data/pooled_benchmark_train.csv", "data/pooled_benchmark_test.csv"]:
        for r in csv.DictReader(open(ROOT / fn)):
            feat = {c: float(r[c]) for c in STRUCT}
            feat.update(seq_features(r.get("seq", "")))
            rows.append({"pdb4": r["pdb"].lower()[:4], "y": float(r["y"]),
                         "length": int(float(r["length"])), "feat": feat})
    return rows


def shared91():
    f = SI / "SI-File-6-protein-peptide-test-set-1.csv"
    out = {}
    for r in csv.DictReader(open(f)):
        m = re.match(r"([0-9a-zA-Z]{4})", r["PDB_NAME"])
        if m:
            out[m.group(1).lower()] = float(r["Binding_affinity"])
    return out


# ---- linear ridge with router ----
def ridge(rows, cols, lam=1.0):
    X = np.array([[r["feat"][c] for c in cols] for r in rows], float)
    y = np.array([r["y"] for r in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    R = np.eye(A.shape[1]) * lam
    R[0, 0] = 0
    return mu, sd, np.linalg.solve(A.T @ A + R, A.T @ y)


def rpred(feat, cols, p):
    mu, sd, w = p
    x = np.array([feat[c] for c in cols], float)
    return float(np.r_[1.0, (x - mu) / sd] @ w)


def loco(rows, kind, cols, router=False, seed=0):
    """Leave-one-complex-out predictions. kind in {ridge, gbt, stack}."""
    n = len(rows)
    pred = np.full(n, np.nan)
    y = np.array([r["y"] for r in rows])
    for i in range(n):
        tr = [rows[j] for j in range(n) if j != i]
        if kind == "ridge":
            if router and rows[i]["length"] <= 8 and sum(r["length"] <= 8 for r in tr) >= 6:
                c2, base = SHORT, [r for r in tr if r["length"] <= 8]
            else:
                c2, base = cols, tr
            pred[i] = rpred(rows[i]["feat"], c2, ridge(base, c2))
        elif kind == "gbt":
            Xt = np.array([[r["feat"][c] for c in cols] for r in tr], float)
            yt = np.array([r["y"] for r in tr])
            m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                              l2_regularization=1.0, min_samples_leaf=20,
                                              random_state=seed).fit(Xt, yt)
            pred[i] = float(m.predict(np.array([[rows[i]["feat"][c] for c in cols]]))[0])
        elif kind == "stack":
            # physics ridge prediction (inner LOCO-free: fit on tr) as a feature, + sequence feats, → GBT
            pr_params = ridge(tr, STRUCT)
            feats_tr = []
            for r in tr:
                feats_tr.append([rpred(r["feat"], STRUCT, pr_params)] + [r["feat"][c] for c in SEQF])
            yt = np.array([r["y"] for r in tr])
            m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                              l2_regularization=1.0, min_samples_leaf=20,
                                              random_state=seed).fit(np.array(feats_tr), yt)
            xi = [[rpred(rows[i]["feat"], STRUCT, pr_params)] + [rows[i]["feat"][c] for c in SEQF]]
            pred[i] = float(m.predict(np.array(xi))[0])
    return pred


def stat(pred, y, mask=None):
    p, yy = (pred, y) if mask is None else (pred[mask], y[mask])
    m = ~(np.isnan(p) | np.isnan(yy))
    p, yy = p[m], yy[m]
    a, b = np.polyfit(p, yy, 1)
    return pearsonr(p, yy)[0], spearmanr(p, yy).statistic, float(np.sqrt(np.mean((a * p + b - yy) ** 2))), len(yy)


def main():
    rows = load()
    y = np.array([r["y"] for r in rows])
    sh = shared91()
    mask91 = np.array([r["pdb4"] in sh for r in rows])
    print(f"=== E104 physics+ML combo (n={len(rows)}, shared-91 mask={mask91.sum()}) ===")
    print(f"    BAR: PPI-Affinity = 0.554 pooled-T100 / 0.629 on shared-91\n")
    print(f"  {'model':<26}{'pooled r':>10}{'pool RMSE':>11}{'  | shared-91 r':>16}{'sh RMSE':>9}")
    configs = [
        ("M0 ridge/struct+router", "ridge", STRUCT, True),
        ("M1 GBT/struct", "gbt", STRUCT, False),
        ("M2 GBT/struct+seq", "gbt", STRUCT + SEQF, False),
        ("M3 stack physics+seq→GBT", "stack", None, False),
    ]
    best = None
    for nm, kind, cols, router in configs:
        p = loco(rows, kind, cols, router=router)
        rp, rhop, rmsep, _ = stat(p, y)
        r91, rho91, rmse91, n91 = stat(p, y, mask91)
        flag = "  <== BEATS PPI" if r91 > 0.629 else ""
        print(f"  {nm:<26}{rp:>+10.3f}{rmsep:>11.2f}{r91:>+16.3f}{rmse91:>9.2f}{flag}")
        if best is None or r91 > best[1]:
            best = (nm, r91, kind, cols, router, p)

    # permutation honesty check on the best model (shuffle y, redo LOCO, 5x)
    print(f"\n  PERMUTATION CHECK on best ({best[0]}, shared-91 r={best[1]:+.3f}):")
    rng = np.random.default_rng(0)
    null = []
    for s in range(5):
        yp = y.copy(); rng.shuffle(yp)
        rows_p = [dict(r, y=yp[k]) for k, r in enumerate(rows)]
        pp = loco(rows_p, best[2], best[3], router=best[4], seed=s)
        null.append(stat(pp, yp, mask91)[0])
    null = np.array(null)
    print(f"     permuted shared-91 r: mean={null.mean():+.3f} max={null.max():+.3f}  "
          f"→ real {best[1]:+.3f} is {'GENUINE' if best[1] > null.max() + 0.1 else 'SUSPECT'}")
    print(f"\n  honest read: structural physics caps ~0.45 on the diverse subset; whether seq+ML closes")
    print(f"  the gap to 0.629 is shown above. If M2/M3 < 0.55, the gap is FEATURE RICHNESS (ProtDCal-scale),")
    print(f"  not the model — and the real lever is more descriptors or PDBbind-scale data (Drive/registration).")


if __name__ == "__main__":
    main()
