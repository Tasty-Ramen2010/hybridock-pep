"""E187 — selectivity baseline from PPIKB families (sequence-only, structure-free first cut).

Within each protein family (same receptor, >=4 distinct peptides, >=2 kcal/mol affinity spread), rank the
peptides by predicted ΔG. Metric = within-family Spearman tau (leave-one-FAMILY-out CV). Sequence-only
features (per-peptide ProtDCal seq descriptors + charge + length). This sets the SEQUENCE CEILING for
selectivity — if it's ~0 (sequence blind, as prior ATLAS/SKEMPI work showed), structure is the only lever
and the PPIKB families become OUR exclusive benchmark.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
e150 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("e150", ROOT / "scripts/e150_protdcal_descriptors.py"))
importlib.util.spec_from_file_location("e150", ROOT / "scripts/e150_protdcal_descriptors.py").loader.exec_module(e150)
SD, SCALES, POS, NEG = e150.seq_descriptors, e150.SCALES, e150.POS, e150.NEG


def feat(seq):
    pq = sum(c in POS for c in seq) - sum(c in NEG for c in seq)
    return SD(seq) + [float(pq), float(abs(pq)), float(len(seq))]


def main():
    rows = [json.loads(l) for l in open(ROOT / "data/ppikb_clean.jsonl")]
    fam = defaultdict(list)
    for r in rows:
        fam[r["protein_seq"][:50]].append(r)
    fams = [(k, v) for k, v in fam.items() if len({x["seq"] for x in v}) >= 4
            and (max(x["y"] for x in v) - min(x["y"] for x in v)) >= 2.0]
    print(f"selectivity families (>=4 peptides, >=2 kcal spread): {len(fams)}")
    allrows = [r for _, v in fams for r in v]
    fam_id = {k: i for i, (k, _) in enumerate(fams)}
    X = np.nan_to_num([feat(r["seq"]) for _, v in fams for r in v])
    y = np.array([r["y"] for _, v in fams for r in v])
    gid = np.array([fam_id[k] for k, v in fams for _ in v])
    print(f"total peptides: {len(allrows)}")

    # leave-one-FAMILY-out: train on all other families, predict this family, within-family tau
    taus = []
    from sklearn.model_selection import LeaveOneGroupOut
    pred = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, gid):
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=3.0, min_samples_leaf=8, random_state=0).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    for fi in np.unique(gid):
        m = gid == fi
        if m.sum() >= 4 and np.std(y[m]) > 0:
            t = spearmanr(pred[m], y[m]).statistic
            if not np.isnan(t):
                taus.append(t)
    print(f"\n=== SEQUENCE-only within-family selectivity (leave-family-out) ===")
    print(f"  mean within-family Spearman tau = {np.mean(taus):+.3f}  (median {np.median(taus):+.3f}, n_fam={len(taus)})")
    print(f"  fraction of families with tau>0: {np.mean([t>0 for t in taus]):.2f}")
    print("  (prior ATLAS/SKEMPI: sequence is BLIND to selectivity tau~0; structure is the only positive lever)")
    # charged families
    chgfam = [fi for fi in np.unique(gid) if np.mean([abs(allrows[j]["net_charge"]) for j in range(len(allrows)) if gid[j] == fi]) >= 2]
    print(f"  (charged families |q|avg>=2: {len(chgfam)})")


if __name__ == "__main__":
    main()
