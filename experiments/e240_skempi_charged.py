"""E240 — Lever-1 baseline + Lever-2 motivation test on SKEMPI charge-changing ΔΔG.
Question: in the RELATIVE (ΔΔG) frame, is the salt-bridge signal learnable, and is DESOLVATION (burial) the
discriminating variable? If burial already lifts charged ΔΔG, a better desolvation measure (3D-RISM, Lever 2)
has headroom. If burial does nothing, desolvation isn't the missing piece and Lever 2 is unlikely.

SKEMPI struct entries: pdb, wt, mutaa, loc, ddg, iface_dist, n5, n8, burial.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHARGED = set("DEKR")


def R(p, y, m=None):
    if m is not None:
        p, y = p[m], y[m]
    ok = ~(np.isnan(p) | np.isnan(y))
    return float(np.corrcoef(p[ok], y[ok])[0, 1]) if ok.sum() > 4 else float("nan")


def _hgb():
    return HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                         l2_regularization=2.0, min_samples_leaf=10, random_state=0)


def main():
    rows = [json.loads(l) for l in open(ROOT / "data/e165_skempi_struct.jsonl")]
    for r in rows:
        r["q"] = (1 if r["wt"] in "KR" else -1 if r["wt"] in "DE" else 0)
        r["charged"] = r["wt"] in CHARGED
        r["burial"] = float(r.get("burial", 0) or 0)
    n_charged = sum(r["charged"] for r in rows)
    print(f"=== SKEMPI ΔΔG: n={len(rows)}  charge-changing(wt∈DEKR)={n_charged} ===")

    def feats(r, withburial):
        f = [float(r["q"]), float(abs(r["q"])), float(r.get("iface_dist", 0) or 0),
             float(r.get("n5", 0) or 0), float(r.get("n8", 0) or 0)]
        return f + ([r["burial"]] if withburial else [])

    def cv(sub, withburial):
        y = np.array([r["ddg"] for r in sub]); grp = np.array([r["pdb"] for r in sub])
        X = np.nan_to_num([feats(r, withburial) for r in sub]); pred = np.full(len(sub), np.nan)
        for tr, te in GroupKFold(min(5, len(set(grp)))).split(X, y, grp):
            pred[te] = _hgb().fit(X[tr], y[tr]).predict(X[te])
        return R(pred, y)

    charged = [r for r in rows if r["charged"]]
    hydro = [r for r in rows if r["wt"] in "AILMFWVY"]
    print("\n=== can we predict ΔΔG, and does BURIAL (desolvation proxy) help? (clustered-CV by pdb) ===")
    for nm, sub in [("charge-changing", charged), ("hydrophobic", hydro), ("ALL", rows)]:
        base = cv(sub, False); bur = cv(sub, True)
        print(f"  {nm:<16} n={len(sub):<5} base r={base:+.3f}   +burial r={bur:+.3f}   Δ={bur-base:+.3f}")

    # direct: how much does burial ALONE correlate with |ddg| for charged vs hydrophobic?
    print("\n=== burial → ΔΔG signal (raw correlations) ===")
    for nm, sub in [("charge-changing", charged), ("hydrophobic", hydro)]:
        b = np.array([r["burial"] for r in sub]); dd = np.array([r["ddg"] for r in sub])
        print(f"  {nm:<16} corr(burial, ddg)={R(b, dd):+.3f}   corr(burial, |ddg|)={R(b, np.abs(dd)):+.3f}")

    # buried vs exposed charged: is the salt-bridge ΔΔG bigger when buried? (the Lever-2 hypothesis)
    bc = np.array([r["burial"] for r in charged]); ddc = np.abs([r["ddg"] for r in charged])
    hi = bc >= np.median(bc)
    print(f"\n  charged BURIED (top 50% burial): mean|ΔΔG|={ddc[hi].mean():.2f}  (n={hi.sum()})")
    print(f"  charged EXPOSED (bot 50% burial): mean|ΔΔG|={ddc[~hi].mean():.2f}  (n={(~hi).sum()})")
    print("  → if buried charged have larger |ΔΔG|, desolvation context IS the discriminator (Lever-2 premise)")


if __name__ == "__main__":
    main()
