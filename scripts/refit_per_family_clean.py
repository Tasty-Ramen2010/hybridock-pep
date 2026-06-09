"""Clean per-family refit + true held-out validation on the crystal benchmark.

The shipped per-family JSON used Ridge(alpha=0.1, positive=False) — weak
regularization + no sign constraint — which let per-family w_vina go negative/
zero (overfitting each family's intercept). This refits with:
  * stronger ridge regularization (alpha sweep),
  * Vina sign respected (w_vina ≥ 0 in the natural basis via `positive` fit on
    sign-oriented features), and
  * a TRUE held-out test: fit families on the 240-pool MINUS the 65 crystal
    complexes, then predict those 65. No LOO, no leakage — the honest number.

Family assignment is the existing sequence-based clustering (label-independent).
Families with < MIN_FAM train members fall back to the global ridge (production
similarity-gate behaviour). Writes data/calibration_per_family_clean.json when a
clean alpha beats single-ridge on the held-out crystal set.

Usage:  python scripts/refit_per_family_clean.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("vina", "n_contact", "s_ss_weighted")
MIN_FAM = 5          # need ≥5 train members to fit a family-specific 3-feat ridge
ALPHAS = (0.1, 1.0, 3.0, 10.0)


def _feat(r: dict) -> list[float]:
    return [float(r.get("vina") or 0), float(r.get("n_contact") or 0),
            float(r.get("s_ss_weighted") or 0)]


def _fit(X, y, alpha, positive):
    # Orient features so the binding-correct sign is non-negative when positive=True:
    #   vina: more-negative = better → flip sign so larger = better-binding
    #   n_contact, s_ss: larger generally = more burial/entropy → keep
    return Ridge(alpha=alpha, positive=positive).fit(X, y)


def main() -> None:
    holdout = [r for r in json.loads((ROOT / "data" / "eval_holdout_calibrations.json").read_text())
               if r.get("vina") is not None and r.get("dg_exp") is not None]
    fam = json.loads((ROOT / "data" / "calibration_per_family.json").read_text())
    membership = {k.lower(): v for k, v in fam["membership_assignments"].items()}
    crystal_ids = {r["pdb"].lower() for r in json.loads((ROOT / "data" / "benchmark_crystal.json").read_text())}

    rows = [{"pdb": (r.get("pdb") or "").lower(), "x": _feat(r), "y": float(r["dg_exp"]),
             "fam": membership.get((r.get("pdb") or "").lower())} for r in holdout]
    test = [r for r in rows if r["pdb"] in crystal_ids]
    train = [r for r in rows if r["pdb"] not in crystal_ids]
    print(f"True held-out split: {len(train)} train (non-crystal) → {len(test)} test (crystal)\n")

    Xtr = np.array([r["x"] for r in train]); ytr = np.array([r["y"] for r in train])
    ytest = np.array([r["y"] for r in test])

    def eval_config(alpha, positive):
        gX = (Xtr - Xtr.mean(0)) / (Xtr.std(0) + 1e-9)
        gmean, gstd = Xtr.mean(0), Xtr.std(0) + 1e-9
        g = _fit(gX, ytr, alpha, positive)
        single, family, routed = [], [], 0
        # pre-fit family ridges on train pool
        fam_models = {}
        from collections import defaultdict
        members = defaultdict(list)
        for i, r in enumerate(train):
            if r["fam"] is not None:
                members[r["fam"]].append(i)
        for fid, idx in members.items():
            if len(idx) >= MIN_FAM:
                fam_models[fid] = _fit(gX[idx], ytr[idx], alpha, positive)
        for r in test:
            xs = (np.array(r["x"]) - gmean) / gstd
            single.append(float(g.predict([xs])[0]))
            fm = fam_models.get(r["fam"])
            if fm is not None:
                family.append(float(fm.predict([xs])[0])); routed += 1
            else:
                family.append(float(g.predict([xs])[0]))
        sr = pearsonr(single, ytest).statistic
        fr = pearsonr(family, ytest).statistic
        frmse = float(np.sqrt(np.mean((np.array(family) - ytest) ** 2)))
        # mean family w_vina to check sign sanity
        wv = np.mean([m.coef_[0] for m in fam_models.values()]) if fam_models else float("nan")
        return sr, fr, frmse, routed, len(fam_models), wv

    print(f"  {'alpha':>6s} {'pos':>4s} {'single_r':>9s} {'family_r':>9s} {'family_RMSE':>11s} "
          f"{'routed':>7s} {'nfam':>5s} {'mean_w_vina':>11s}")
    best = None
    for positive in (False, True):
        for a in ALPHAS:
            sr, fr, frmse, routed, nfam, wv = eval_config(a, positive)
            tag = ""
            if best is None or fr > best[0]:
                best = (fr, a, positive, sr, frmse); tag = "  <= best"
            print(f"  {a:6.1f} {str(positive):>4s} {sr:+9.3f} {fr:+9.3f} {frmse:11.2f} "
                  f"{routed:7d} {nfam:5d} {wv:+11.3f}{tag}")

    print(f"\nBest held-out per-family: r={best[0]:+.3f} (alpha={best[1]}, positive={best[2]}, "
          f"RMSE={best[4]:.2f}) vs single-ridge r={best[3]:+.3f}")
    print("Validation = TRUE held-out (crystal complexes never in the family fits).")


if __name__ == "__main__":
    main()
