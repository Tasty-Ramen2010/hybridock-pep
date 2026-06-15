"""E191 — crystal head-to-head PPI vs us on PPI's T100, r + MAE, by LENGTH and CHARGE (Ram: "where do we
fail / are we close?"). Uses our PRODUCTION crystal feature set (seq-ProtDCal + pocket + geometry + charge-
complement + length) exactly as E166 — NOT the experimental ProtDCal-3D fusion (which hurts off-distribution).
PPI = shipped predictions. Full 85 (37 PDBbind-overlap + 48 BioLiP-extracted).
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
e150 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("e150", ROOT / "scripts/e150_protdcal_descriptors.py"))
importlib.util.spec_from_file_location("e150", ROOT / "scripts/e150_protdcal_descriptors.py").loader.exec_module(e150)
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
    compl = [pq * d["poc_net"], abs(pq) * abs(d["poc_net"]), abs(pq + d["poc_net"])]
    return SD(d["seq"]) + d["pkf"] + [float(d[c]) for c in PROD] + compl + [float(len(d["seq"]))]


def main():
    man = {m["pdb"].lower(): m for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    have = {r["pdb"].lower(): r for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl"))}
    cache = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")}
    test = []
    for pid, m in man.items():
        d = None
        if pid in have:
            d = dict(have[pid]); ps = e158.pocket_seq(pid)
            if ps is None:
                continue
            d["pkf"] = [float(np.mean([SCALES[s].get(c, 0) for c in ps])) for s in SN]
        elif pid in cache:
            d = cache[pid]
        if d is None:
            continue
        try:
            ship = float(m["ppi_affinity"])
        except (TypeError, ValueError):
            ship = np.nan
        seq = d["seq"]; pq = sum(c in POS for c in seq) - sum(c in NEG for c in seq)
        test.append({"pid": pid, "d": d, "y": float(m["dg_exp"]), "ship": ship, "L": len(seq), "q": abs(pq)})
    tid = {t["pid"] for t in test}

    # train on PDBbind-925 minus T100
    train = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        if r["pdb"].lower() in tid:
            continue
        ps = e158.pocket_seq(r["pdb"])
        if ps is None:
            continue
        r = dict(r); r["pkf"] = [float(np.mean([SCALES[s].get(c, 0) for c in ps])) for s in SN]
        train.append(r)
    Xtr = np.nan_to_num([feats(d) for d in train]); ytr = np.array([float(d["y"]) for d in train])
    mdl = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                        l2_regularization=3.0, min_samples_leaf=12, random_state=0).fit(Xtr, ytr)
    Xte = np.nan_to_num([feats(t["d"]) for t in test])
    ours = mdl.predict(Xte)
    y = np.array([t["y"] for t in test]); ship = np.array([t["ship"] for t in test])
    L = np.array([t["L"] for t in test]); q = np.array([t["q"] for t in test])

    def row(name, mask):
        ro, mo = met(ours[mask], y[mask]); rp, mp = met(ship[mask], y[mask])
        gap = ro - rp
        tag = "← WE WIN" if (not np.isnan(ro) and ro > rp) else (f"gap {gap:+.2f}" if not np.isnan(ro) else "")
        print(f"  {name:<15} n={mask.sum():<4} | OURS r={ro:+.3f} MAE={mo:.2f} | PPI r={rp:+.3f} MAE={mp:.2f}  {tag}")

    print(f"=== CRYSTAL head-to-head — PPI's T100 (n={len(test)}, ours=production feats, held-out of 925) ===")
    row("OVERALL", np.ones(len(test), bool))
    print("\n  --- by LENGTH ---")
    for nm, mk in [("short<=8", L <= 8), ("med9-12", (L >= 9) & (L <= 12)),
                   ("long13-16", (L >= 13) & (L <= 16)), ("vlong>=17", L >= 17)]:
        if mk.sum():
            row(nm, mk)
    print("\n  --- by CHARGE ---")
    for nm, mk in [("neutral|q|<=1", q <= 1), ("charged|q|>=2", q >= 2), ("v.charged|q|>=3", q >= 3)]:
        if mk.sum():
            row(nm, mk)


if __name__ == "__main__":
    main()
