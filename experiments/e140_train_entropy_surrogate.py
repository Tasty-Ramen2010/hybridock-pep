"""E140 — train + SAVE the production per-residue entropy surrogate on the full e120 MD set.

Loads data/sfree_perres.jsonl (922 free-peptide MD runs), trains the context-aware per-residue entropy
model (E123 features), reports grouped-CV r, and saves the fitted model to data/entropy_surrogate.joblib
for deployment (entropy_lost = Σ surrogate(contacting residues)).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("e123", ROOT / "scripts" / "e123_perres_entropy_surrogate.py")
e123 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e123)


def main():
    path = ROOT / "data" / "sfree_perres.jsonl"
    X, y, grp, fl = e123.load(path)
    npep = len(np.unique(grp))
    print(f"=== E140 production entropy surrogate ({len(y)} residues, {npep} peptides) ===")
    pred = e123.grouped_cv(X, y, grp)
    ok = ~np.isnan(pred)
    r = pearsonr(pred[ok], y[ok])[0]
    rmse = float(np.sqrt(np.mean((pred[ok] - y[ok]) ** 2)))
    print(f"  grouped-CV r={r:+.3f}  RMSE={rmse:.3f}  (flex baseline {pearsonr(fl[ok], y[ok])[0]:+.3f})")
    # fit final model on ALL data and save
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        import joblib
        model = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.05,
                                              l2_regularization=2.0, min_samples_leaf=20, random_state=0)
        model.fit(X, y)
        out = ROOT / "data" / "entropy_surrogate.joblib"
        joblib.dump({"model": model, "cv_r": r, "n_pep": npep, "feature_order": getattr(e123, "FEATURE_ORDER", None)}, out)
        print(f"  saved → {out}")
    except Exception as e:  # noqa: BLE001
        print(f"  save skipped: {e}")


if __name__ == "__main__":
    main()
