"""E190 — STRUCTURE-based within-family selectivity vs the sequence baseline (tau=0.059, E187).

Within each PPIKB family (>=4 peptides, >=2 kcal spread) with structures, rank peptides by predicted ΔG
using STRUCTURE features (ProtDCal-3D contact descriptors on the bound peptide + pocket). Leave-one-family-out.
If within-family tau >> 0.059, structure is the selectivity lever and PPIKB families are OUR benchmark.
Reports overall + charged families (the floor).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "3"
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
e150 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("e150", ROOT / "scripts/e150_protdcal_descriptors.py"))
importlib.util.spec_from_file_location("e150", ROOT / "scripts/e150_protdcal_descriptors.py").loader.exec_module(e150)
SD, POS, NEG = e150.seq_descriptors, e150.POS, e150.NEG


def feats(r, mode):
    pq = sum(c in POS for c in r["seq"]) - sum(c in NEG for c in r["seq"])
    seq = SD(r["seq"]) + [float(pq), float(abs(pq)), float(len(r["seq"]))]
    if mode == "seq":
        return seq
    d3 = list(r["desc3d"]) if r.get("desc3d") else [0.0] * 37
    pk = list(r.get("pocket_pkf", [])) or [0.0] * 22
    if mode == "struct":
        return d3 + pk
    return seq + d3 + pk  # both


def tau_eval(rows, mode):
    fam = defaultdict(list)
    for r in rows:
        fam[r["protein_seq"][:50]].append(r)
    fams = [(k, v) for k, v in fam.items() if len({x["seq"] for x in v}) >= 4
            and (max(x["y"] for x in v) - min(x["y"] for x in v)) >= 2.0]
    X = np.nan_to_num([feats(r, mode) for _, v in fams for r in v])
    y = np.array([r["y"] for _, v in fams for r in v])
    fid = {k: i for i, (k, _) in enumerate(fams)}
    gid = np.array([fid[k] for k, v in fams for _ in v])
    pred = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, gid):
        m = HistGradientBoostingRegressor(max_iter=250, max_depth=3, learning_rate=0.05,
                                          l2_regularization=3.0, min_samples_leaf=6, random_state=0).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    taus, chg_taus = [], []
    for k, v in fams:
        i = fid[k]; mask = gid == i
        if mask.sum() >= 4 and np.std(y[mask]) > 0:
            t = spearmanr(pred[mask], y[mask]).statistic
            if not np.isnan(t):
                taus.append(t)
                if np.mean([abs(x["net_charge"]) for x in v]) >= 2:
                    chg_taus.append(t)
    return np.mean(taus), len(taus), (np.mean(chg_taus) if chg_taus else float("nan")), len(chg_taus)


def main():
    rows = [json.loads(l) for l in open(ROOT / "data/ppikb_features.jsonl")
            if json.loads(l).get("desc3d")]
    print(f"PPIKB entries with structure features: {len(rows)}\n")
    print(f"=== within-family selectivity tau (leave-family-out) ===")
    print(f"  {'features':<16}{'tau':>8}{'n_fam':>7}{'charged_tau':>13}{'n_chg':>7}")
    for mode in ("seq", "struct", "both"):
        t, n, ct, cn = tau_eval(rows, mode)
        print(f"  {mode:<16}{t:>+8.3f}{n:>7}{ct:>+13.3f}{cn:>7}")
    print("\n  seq baseline (all 80 fams, E187) was +0.059 — structure should beat it if it's the lever")


if __name__ == "__main__":
    main()
