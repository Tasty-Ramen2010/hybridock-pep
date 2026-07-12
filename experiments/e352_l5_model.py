"""E352 — PRISM Layer 5 model: leakage-free Δ-learning of charged-ΔΔG from structure + mutation features.

Trains a gradient-boosted model on the 1445 ±1 charge-change cases (E351 features) with GroupKFold BY PDB — no
complex appears in both train and test (the leakage discipline from the peptide-clustered-CV work; random CV here
would inflate r by memorising per-complex offsets). Reports out-of-fold r / MAE / RMSE vs baselines (predict-mean,
ridge) and feature importances. This is the at-scale ML backbone of PRISM L5; the physics-engine outputs (GB/QM/
RISM) merge in later as extra columns to make it a true physics+ML Δ-model.

Run: OMP_NUM_THREADS=1 /home/igem/miniconda3/envs/openmm-env/bin/python experiments/e352_l5_model.py
"""
from __future__ import annotations
import sys, json
import numpy as np

FEAT_FILE = "/home/igem/unknown_software/data/e351_l5_features.jsonl"
FEATURES = ["wt_charge", "mut_charge", "dq", "d_volume", "d_hydropathy", "is_alanine", "is_isosteric",
            "buried_frac", "n_contacts", "opp_charge_dist", "same_charge_dist", "n_aromatic",
            "n_polar_neutral", "n_hydrophobic", "metal_near", "complex_atoms"]


def load():
    rows = [json.loads(ln) for ln in open(FEAT_FILE)]
    X = np.array([[r[f] for f in FEATURES] for r in rows], float)
    y = np.array([r["exp"] for r in rows], float)
    groups = np.array([r["tag"].split("_")[0] for r in rows])   # group by PDB → leakage-free
    return X, y, groups, rows


def cv_metrics(model_fn, X, y, groups):
    from sklearn.model_selection import GroupKFold
    from scipy.stats import pearsonr
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        m = model_fn()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    r = pearsonr(oof, y)[0]
    mae = np.mean(np.abs(oof - y))
    rmse = np.sqrt(np.mean((oof - y) ** 2))
    return r, mae, rmse, oof


def main():
    X, y, groups, rows = load()
    print(f"=== E352 PRISM L5 model: n={len(y)} cases, {len(set(groups))} PDBs (GroupKFold by PDB) ===")
    print(f"label: exp ΔΔG  (std={y.std():.2f}, range {y.min():+.1f}..{y.max():+.1f})\n")

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        "predict-mean": lambda: _Mean(),
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "HistGBT": lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                         max_iter=400, l2_regularization=1.0,
                                                         min_samples_leaf=20, random_state=0),
    }
    best = None
    for name, fn in models.items():
        r, mae, rmse, oof = cv_metrics(fn, X, y, groups)
        print(f"{name:14s}  r={r:+.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}")
        if name == "HistGBT":
            best = oof
    # feature importance via permutation on a full-fit HistGBT (directional insight only)
    from sklearn.inspection import permutation_importance
    m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=400,
                                      l2_regularization=1.0, min_samples_leaf=20, random_state=0).fit(X, y)
    imp = permutation_importance(m, X, y, n_repeats=5, random_state=0, n_jobs=1)
    order = np.argsort(imp.importances_mean)[::-1]
    print("\ntop features (permutation importance):")
    for i in order[:8]:
        print(f"  {FEATURES[i]:18s} {imp.importances_mean[i]:+.3f}")
    print("\nNOTE: this is the structure+mutation ML backbone. Next: merge GB/QM/RISM engine outputs as columns "
          "→ true physics+ML Δ-model; expect r to rise where the charge-electrostatic baseline is informative.")


class _Mean:
    def fit(self, X, y): self.m = float(np.mean(y)); return self
    def predict(self, X): return np.full(len(X), self.m)


if __name__ == "__main__":
    main()
