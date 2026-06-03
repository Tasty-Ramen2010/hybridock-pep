"""Deep diagnostic: crystal-vs-production LOO + Vina/AD4 behavioural scan.

Answers four questions:

1. Crystal-pose calibration fair LOO-CV r — for apples-to-apples vs the
   production-pose ridge fit (which we reported as LOO r=+0.755).
2. Why does the multivariate ridge zero out the AD4 weight on production
   poses? Multicollinearity? Pure noise? Already subsumed by N_contact?
3. Within-target rank correlation: does Vina (and AD4) on production
   poses pick the binding-mode-closest pose (Cα RMSD to crystal)?
4. What aggregation choice (top-K, median, mean, min, no-clash filter)
   gives the strongest cross-target calibration?

All scans run on the 6 PepSet complexes. Output: prints to stdout +
writes structured JSON to runs/diagnose_calibration.json for archiving.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut

R_T_LN10 = 1.364  # kcal/mol at 298 K

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "src"))


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #

def load_set(scores_path: Path):
    scores = json.loads(scores_path.read_text())
    rows = list(csv.DictReader((_REPO / "data/training_complexes.csv").open()))
    out = []
    for r in rows:
        pdb = r["pdb_id"]
        if pdb not in scores:
            continue
        s = scores[pdb]
        out.append({
            "pdb": pdb,
            "pkd": float(r["experimental_pkd"]),
            "dG_exp": -R_T_LN10 * float(r["experimental_pkd"]),
            "n_res": len(r["peptide_sequence"]),
            "vina": s["vina_score"],
            "ad4": s["ad4_score"],
            "n_contact": s["n_contact_residues"],
            "all_poses": s.get("all_poses", []),
            "top_k_poses": s.get("top_k_poses", []),
        })
    return out


def matrix(rows, keys):
    return np.array([[r[k] for k in keys] for r in rows])


# --------------------------------------------------------------------------- #
# Ridge fit + LOO helpers                                                      #
# --------------------------------------------------------------------------- #

def ridge_fit(rows, features, *, ridge_alpha=0.1, positive=True):
    X = np.column_stack([
        [r[f] if not f.startswith("-") else -r[f[1:]] for r in rows]
        for f in features
    ]).astype(float)
    y = np.array([r["dG_exp"] for r in rows])
    m = Ridge(alpha=ridge_alpha, positive=positive).fit(X, y)
    pred = m.predict(X)
    r = float(pearsonr(y, pred).statistic) if len(y) > 2 else float("nan")
    rmse = float(np.sqrt(((pred - y) ** 2).mean()))
    return m, X, y, pred, r, rmse


def loo_cv(rows, features, *, ridge_alpha=0.1, positive=True):
    X = np.column_stack([
        [r[f] if not f.startswith("-") else -r[f[1:]] for r in rows]
        for f in features
    ]).astype(float)
    y = np.array([r["dG_exp"] for r in rows])
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        m = Ridge(alpha=ridge_alpha, positive=positive).fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    r = float(pearsonr(y, preds).statistic) if len(y) > 2 else float("nan")
    rmse = float(np.sqrt(((preds - y) ** 2).mean()))
    return preds, r, rmse


# --------------------------------------------------------------------------- #
# Q1: crystal LOO comparison                                                   #
# --------------------------------------------------------------------------- #

def q1_crystal_loo(prod, crys, out):
    print("\n" + "=" * 72)
    print("Q1 — Fair LOO comparison: crystal-pose vs production-pose")
    print("=" * 72)

    print(f"\n{'pose-source':16s} {'feature-set':28s} {'in-sample r':>12s} {'LOO r':>8s} "
          f"{'in-sample RMSE':>16s} {'LOO RMSE':>10s}")
    feature_sets = [
        ("vina only", ["vina"]),
        ("ad4 only", ["ad4"]),
        ("vina+ad4", ["vina", "ad4"]),
        ("vina+ad4+(-n_contact)", ["vina", "ad4", "-n_contact"]),
        ("(-n_contact) only", ["-n_contact"]),
    ]
    rows = []
    for src_name, src in [("crystal", crys), ("production", prod)]:
        for fset_name, fset in feature_sets:
            _, _, _, _, r_in, rmse_in = ridge_fit(src, fset)
            _, r_loo, rmse_loo = loo_cv(src, fset)
            print(f"{src_name:16s} {fset_name:28s} {r_in:>+12.3f} {r_loo:>+8.3f} "
                  f"{rmse_in:>16.2f} {rmse_loo:>10.2f}")
            rows.append({"source": src_name, "features": fset_name,
                         "r_in": r_in, "r_loo": r_loo,
                         "rmse_in": rmse_in, "rmse_loo": rmse_loo})

    out["q1_crystal_vs_prod_loo"] = rows


# --------------------------------------------------------------------------- #
# Q2: why does AD4 drop out?                                                   #
# --------------------------------------------------------------------------- #

def q2_ad4_diagnosis(prod, out):
    print("\n" + "=" * 72)
    print("Q2 — Why does the ridge zero out the AD4 weight?")
    print("=" * 72)

    # 2a — feature correlation matrix
    Xall = matrix(prod, ["vina", "ad4", "n_contact", "n_res"])
    feat = ["vina", "ad4", "n_contact", "n_res"]
    print("\n(2a) Feature correlation matrix on the 6 aggregates:")
    print(f"{'':10s}" + "".join(f"{f:>10s}" for f in feat))
    for i, f in enumerate(feat):
        print(f"{f:10s}" + "".join(
            f"{pearsonr(Xall[:, i], Xall[:, j]).statistic:>+10.3f}" for j in range(len(feat))
        ))

    # 2b — AD4 by itself and conditional on N_contact (partial corr)
    dG = np.array([r["dG_exp"] for r in prod])
    ad4 = np.array([r["ad4"] for r in prod])
    nc = np.array([r["n_contact"] for r in prod])
    r_ad4_dg = pearsonr(ad4, dG).statistic
    r_nc_dg = pearsonr(nc, dG).statistic
    r_ad4_nc = pearsonr(ad4, nc).statistic

    def partial(r_xy, r_xz, r_yz):
        denom = ((1 - r_xz ** 2) * (1 - r_yz ** 2)) ** 0.5
        return (r_xy - r_xz * r_yz) / denom if denom else float("nan")

    print(f"\n(2b) Partial correlations (controls for the size/contact confound):")
    print(f"  r(AD4, ΔGexp)               = {r_ad4_dg:+.3f}")
    print(f"  r(AD4, ΔGexp | -N_contact)  = {partial(r_ad4_dg, r_ad4_nc, r_nc_dg):+.3f}")
    print(f"  r(Vina, ΔGexp | -N_contact) = "
          f"{partial(pearsonr(np.array([r['vina'] for r in prod]), dG).statistic, pearsonr(np.array([r['vina'] for r in prod]), nc).statistic, r_nc_dg):+.3f}")

    # 2c — does AD4 carry signal Vina+contact does not?
    print(f"\n(2c) Incremental r from adding AD4 to a model already containing "
          f"vina+(-n_contact):")
    _, r_a, _ = loo_cv(prod, ["vina", "-n_contact"])
    _, r_b, _ = loo_cv(prod, ["vina", "-n_contact", "ad4"])
    print(f"  LOO r vina+(-n_contact)       = {r_a:+.3f}")
    print(f"  LOO r vina+(-n_contact)+ad4   = {r_b:+.3f}")
    print(f"  Δr                            = {r_b - r_a:+.3f}")

    # 2d — within-target Vina vs AD4 distribution
    print(f"\n(2d) Within-target Vina/AD4 score statistics on ALL scored poses:")
    print(f"{'pdb':6s} {'n':>4s} {'V_med':>7s} {'V_iqr':>7s} {'A_med':>7s} {'A_iqr':>7s} "
          f"{'ρ(V,A)':>7s} {'ρ(V,nC)':>8s} {'ρ(A,nC)':>8s}")
    detail_rows = []
    for r in prod:
        ap = r["all_poses"]
        if not ap:
            continue
        v = np.array([p["vina_score"] for p in ap])
        a = np.array([p["ad4_score"] for p in ap])
        ncp = np.array([p["n_contact_residues"] for p in ap])
        v_iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
        a_iqr = float(np.percentile(a, 75) - np.percentile(a, 25))
        rho_va = float(spearmanr(v, a).statistic) if len(v) > 2 else float("nan")
        rho_vnc = float(spearmanr(v, ncp).statistic) if len(v) > 2 else float("nan")
        rho_anc = float(spearmanr(a, ncp).statistic) if len(a) > 2 else float("nan")
        print(f"{r['pdb']:6s} {len(v):>4d} {float(np.median(v)):>+7.2f} {v_iqr:>7.2f} "
              f"{float(np.median(a)):>+7.2f} {a_iqr:>7.2f} {rho_va:>+7.3f} "
              f"{rho_vnc:>+8.3f} {rho_anc:>+8.3f}")
        detail_rows.append({"pdb": r["pdb"], "n": int(len(v)),
                            "v_med": float(np.median(v)), "v_iqr": v_iqr,
                            "a_med": float(np.median(a)), "a_iqr": a_iqr,
                            "rho_VA": rho_va, "rho_VnC": rho_vnc, "rho_AnC": rho_anc})

    out["q2_ad4_diagnosis"] = {
        "corr_matrix_features": feat,
        "corr_matrix": [
            [float(pearsonr(Xall[:, i], Xall[:, j]).statistic) for j in range(len(feat))]
            for i in range(len(feat))
        ],
        "incremental_loo": {"vina+nc": r_a, "vina+nc+ad4": r_b, "delta": r_b - r_a},
        "within_target": detail_rows,
    }


# --------------------------------------------------------------------------- #
# Q3: within-target rank correlation w/ "binding-mode quality" proxy           #
# --------------------------------------------------------------------------- #

def q3_within_target(prod, out):
    print("\n" + "=" * 72)
    print("Q3 — Within-target: do Vina/AD4 rank poses by N_contact (binding-mode "
          "quality proxy)?")
    print("=" * 72)
    rows = []
    print(f"\n{'pdb':6s} {'n':>4s} {'ρ(V,nC)':>9s} {'ρ(A,nC)':>9s} {'ρ(V,A)':>9s} "
          f"{'frac_clashed':>13s}")
    for r in prod:
        ap = r["all_poses"]
        if not ap:
            continue
        v = np.array([p["vina_score"] for p in ap])
        a = np.array([p["ad4_score"] for p in ap])
        ncp = np.array([p["n_contact_residues"] for p in ap])
        clash_frac = sum(1 for p in ap if p.get("is_clashed")) / len(ap)
        rho_vnc = float(spearmanr(v, ncp).statistic) if len(v) > 2 else float("nan")
        rho_anc = float(spearmanr(a, ncp).statistic) if len(a) > 2 else float("nan")
        rho_va = float(spearmanr(v, a).statistic) if len(v) > 2 else float("nan")
        print(f"{r['pdb']:6s} {len(v):>4d} {rho_vnc:>+9.3f} {rho_anc:>+9.3f} "
              f"{rho_va:>+9.3f} {clash_frac:>13.2%}")
        rows.append({"pdb": r["pdb"], "rho_V_nC": rho_vnc, "rho_A_nC": rho_anc,
                     "rho_V_A": rho_va, "clash_frac": clash_frac})
    out["q3_within_target"] = rows


# --------------------------------------------------------------------------- #
# Q4: aggregation choice scan                                                  #
# --------------------------------------------------------------------------- #

def q4_aggregation_scan(prod, out):
    print("\n" + "=" * 72)
    print("Q4 — Aggregation scan: top-K × {median,mean,min} × {all,no_clashed}")
    print("=" * 72)

    def aggregate(rows, top_k, aggfn, exclude_clashed):
        agg_rows = []
        for r in rows:
            ap = [p for p in r["all_poses"]
                  if not (exclude_clashed and p.get("is_clashed"))]
            if len(ap) == 0:
                return None
            ap_sorted = sorted(ap, key=lambda p: p["vina_score"])[:top_k]
            v = float(aggfn([p["vina_score"] for p in ap_sorted]))
            a = float(aggfn([p["ad4_score"] for p in ap_sorted]))
            nc = int(aggfn([p["n_contact_residues"] for p in ap_sorted]))
            agg_rows.append({"pdb": r["pdb"], "dG_exp": r["dG_exp"],
                             "vina": v, "ad4": a, "n_contact": nc})
        return agg_rows

    print(f"\n{'top-K':>5s} {'aggfn':>6s} {'no_clash':>9s} {'LOO_r_vina':>11s} "
          f"{'LOO_r_v+a+nc':>12s} {'in_r_v+a+nc':>11s} {'RMSE_loo':>9s}")
    rows = []
    for top_k in [1, 5, 10, 20, 50, 100]:
        for aggfn_name, aggfn in [("median", np.median), ("mean", np.mean), ("min", np.min)]:
            for excl in [False, True]:
                ag = aggregate(prod, top_k, aggfn, excl)
                if ag is None:
                    continue
                try:
                    _, r_v, _ = loo_cv(ag, ["vina"])
                    _, r_all, rmse_loo = loo_cv(ag, ["vina", "ad4", "-n_contact"])
                    _, _, _, _, r_in, _ = ridge_fit(ag, ["vina", "ad4", "-n_contact"])
                    print(f"{top_k:>5d} {aggfn_name:>6s} {str(excl):>9s} "
                          f"{r_v:>+11.3f} {r_all:>+12.3f} {r_in:>+11.3f} {rmse_loo:>9.2f}")
                    rows.append({"top_k": top_k, "aggfn": aggfn_name,
                                 "exclude_clashed": excl,
                                 "loo_r_vina": r_v, "loo_r_all": r_all,
                                 "in_r_all": r_in, "rmse_loo": rmse_loo})
                except Exception as exc:
                    print(f"  ERROR k={top_k} {aggfn_name} excl={excl}: {exc}")
    out["q4_aggregation_scan"] = rows


# --------------------------------------------------------------------------- #
# Q5: leave-one-complex-out diagnostic — which one drives the result?          #
# --------------------------------------------------------------------------- #

def q5_drop_complex(prod, out):
    print("\n" + "=" * 72)
    print("Q5 — Drop each complex; how much does the in-sample r change?")
    print("=" * 72)
    feats = ["vina", "ad4", "-n_contact"]
    _, _, _, _, r_all, _ = ridge_fit(prod, feats)
    rows = []
    print(f"\nBaseline (all 6): r = {r_all:+.3f}")
    print(f"\n{'dropped':>8s} {'pdb_pkd':>9s} {'r_5':>7s} {'Δr':>7s} {'w_vina':>8s} "
          f"{'w_ad4':>7s} {'w_nc':>7s}")
    for i in range(len(prod)):
        sub = prod[:i] + prod[i+1:]
        m, _, _, _, r_sub, _ = ridge_fit(sub, feats)
        delta = r_sub - r_all
        print(f"{prod[i]['pdb']:>8s} {prod[i]['pkd']:>9.2f} {r_sub:>+7.3f} {delta:>+7.3f} "
              f"{m.coef_[0]:>+8.3f} {m.coef_[1]:>+7.3f} {m.coef_[2]:>+7.3f}")
        rows.append({"dropped": prod[i]["pdb"], "pkd": prod[i]["pkd"],
                     "r_remaining": r_sub, "delta_r": delta,
                     "w_vina": float(m.coef_[0]), "w_ad4": float(m.coef_[1]),
                     "w_nc": float(m.coef_[2])})
    out["q5_drop_complex"] = rows


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    prod_path = _REPO / "data/training_scores_production.json"
    crys_path = _REPO / "data/training_scores.json"
    prod = load_set(prod_path)
    crys = load_set(crys_path)

    print(f"Loaded prod: {len(prod)} complexes; crys: {len(crys)} complexes")
    print(f"Production data has per-pose 'all_poses': {sum(1 for r in prod if r['all_poses'])}/{len(prod)}")
    print(f"Crystal data has per-pose 'all_poses':    {sum(1 for r in crys if r['all_poses'])}/{len(crys)}")

    out = {"meta": {
        "n_prod": len(prod), "n_crys": len(crys),
        "R_T_LN10": R_T_LN10,
    }}

    q1_crystal_loo(prod, crys, out)
    q2_ad4_diagnosis(prod, out)
    q3_within_target(prod, out)
    q4_aggregation_scan(prod, out)
    q5_drop_complex(prod, out)

    out_path = _REPO / "runs/diagnose_calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
