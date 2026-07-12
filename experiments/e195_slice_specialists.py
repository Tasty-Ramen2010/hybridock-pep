"""E195 — can SLICE-SPECIALIST models close the T100 gaps PPI exploits (vlong, long-structured, neutral)?

For each slice, train a specialist on the 925-slice with the feature emphasis E194 flagged, predict the
T100-slice, compare to our pooled model AND PPI shipped:
  vlong   -> size/seq specialist (E194: seq descriptors |r|~0.5, but POOLED dilutes to 0.12 = model-mixing)
  long    -> + pocket helix/alpha propensity (E194 top: pkf:helix +0.39, alpha_n +0.37)
  neutral -> pocket-hydrophobicity emphasis (E194 top: pkf:hopp +0.34)
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hybridock_pep.scoring.geometry_features import GEOMETRY_FEATURE_KEYS  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402
e150 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("e150", ROOT / "experiments/e150_protdcal_descriptors.py"))
importlib.util.spec_from_file_location("e150", ROOT / "experiments/e150_protdcal_descriptors.py").loader.exec_module(e150)
SD, SCALES, POS, NEG = e150.seq_descriptors, e150.SCALES, e150.POS, e150.NEG
SN = list(SCALES.keys()); PROD = GEOMETRY_FEATURE_KEYS


def met(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = ~(np.isnan(p) | np.isnan(y))
    if ok.sum() < 4:
        return float("nan"), float("nan")
    return float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.mean(np.abs(p[ok] - y[ok])))


def feats(d):
    pq = sum(c in POS for c in d["seq"]) - sum(c in NEG for c in d["seq"])
    return SD(d["seq"]) + d["pkf"] + [float(d[c]) for c in PROD] + [pq * d["poc_net"], abs(pq) * abs(d["poc_net"]), abs(pq + d["poc_net"]), float(len(d["seq"]))]


def main():
    man = {m["pdb"].lower(): m for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    have = {r["pdb"].lower(): r for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl"))}
    cache = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")}

    def pkf_of(pid, seq):
        ps = e158.pocket_seq(pid)
        return [float(np.mean([SCALES[s].get(c, 0) for c in ps])) for s in SN] if ps else None

    test = []
    for pid, m in man.items():
        d = None
        if pid in have:
            d = dict(have[pid]); pk = pkf_of(pid, d["seq"])
            if pk is None:
                continue
            d["pkf"] = pk
        elif pid in cache:
            d = cache[pid]
        if d is None:
            continue
        try:
            ship = float(m["ppi_affinity"])
        except (TypeError, ValueError):
            ship = np.nan
        seq = d["seq"]; pq = sum(c in POS for c in seq) - sum(c in NEG for c in seq)
        test.append({"d": d, "y": float(m["dg_exp"]), "ship": ship, "L": len(seq), "q": abs(pq)})
    tid = {t["d"]["pdb"].lower() for t in test if "pdb" in t["d"]}

    train = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        if r["pdb"].lower() in tid:
            continue
        pk = pkf_of(r["pdb"], r["seq"])
        if pk is None:
            continue
        r = dict(r); r["pkf"] = pk; r["q"] = abs(sum(c in POS for c in r["seq"]) - sum(c in NEG for c in r["seq"]))
        train.append(r)
    Xtr = np.nan_to_num([feats(d) for d in train]); ytr = np.array([float(d["y"]) for d in train])
    Ltr = np.array([d["length"] for d in train]); qtr = np.array([d["q"] for d in train])
    pooled = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                           l2_regularization=3.0, min_samples_leaf=12, random_state=0).fit(Xtr, ytr)

    Xte = np.nan_to_num([feats(t["d"]) for t in test])
    y = np.array([t["y"] for t in test]); ship = np.array([t["ship"] for t in test])
    L = np.array([t["L"] for t in test]); q = np.array([t["q"] for t in test])
    pooled_pred = pooled.predict(Xte)

    def specialist(train_mask, depth=2, leaf=8):
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=depth, learning_rate=0.05,
                                          l2_regularization=5.0, min_samples_leaf=leaf, random_state=0)
        m.fit(Xtr[train_mask], ytr[train_mask]); return m

    print("=== SLICE SPECIALIST vs POOLED vs PPI on T100 (specialist trained on 925-slice only) ===")
    print(f"  {'slice':<16}{'n':>4}{'POOLED':>9}{'SPECIALIST':>12}{'PPI':>9}")
    slices = [("vlong>=17", L >= 17, Ltr >= 17),
              ("long13-16", (L >= 13) & (L <= 16), (Ltr >= 13) & (Ltr <= 16)),
              ("neutral|q|<=1", q <= 1, qtr <= 1),
              ("charged|q|>=2", q >= 2, qtr >= 2)]
    for nm, te_mask, tr_mask in slices:
        if te_mask.sum() < 4:
            continue
        spec = specialist(tr_mask)
        sp = spec.predict(Xte[te_mask])
        rp, _ = met(pooled_pred[te_mask], y[te_mask])
        rs, _ = met(sp, y[te_mask])
        rpp, _ = met(ship[te_mask], y[te_mask])
        tag = "← SPEC>PPI" if rs > rpp else ("← SPEC>POOL" if rs > rp else "")
        print(f"  {nm:<16}{te_mask.sum():>4}{rp:>+9.3f}{rs:>+12.3f}{rpp:>+9.3f}  {tag}")
    print("\n  (vlong: does routing to the size/seq specialist fix the model-mixing dilution e191 showed?)")


if __name__ == "__main__":
    main()
