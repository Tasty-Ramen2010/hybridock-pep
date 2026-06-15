"""E220 — Ram's vlong idea: substitute/augment with EXTREMELY LONG ligands (PPIKB 17-50) + structured-peptide
features, to teach the model vlong behaviour the 53 crystal vlong can't. Need geometry for the XL pool — but
e212 only did up to the long/vlong we parsed (74). Here we test in the seq+pocket+SS feature space (geometry-
free, transferable, no new structure parse needed) whether XL substitution lifts vlong on a held-out vlong test.

Design: held-out test = crystal-925 vlong (53) + e154/e176 real vlong. Train candidates:
  (1) 925 all                      (current)
  (2) 925 + PPIKB vlong 17-25      (close substitution)
  (3) 925 + PPIKB vlong 17-50      (incl XL 26-50 = Ram's "extremely long ligand substitution")
  (4) (3) + structured-peptide SS features (helix/sheet frac) — teach structured-vlong differences
Evaluate vlong-test r/MAE. Seq+pocket(+SS) only (transferable, available for all).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hybridock_pep.scoring.affinity_model import _protdcal_descriptors, _SCALES  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402
import e202_band_routing_build as e202  # noqa: E402
SN = list(_SCALES.keys())
ss = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/ss_features.jsonl")}


def pkf(ps):
    return [float(np.mean([_SCALES[s].get(c, 0) for c in ps])) for s in SN] if ps else [0.0] * len(SN)


def ssv_pep(seq):
    """structured-peptide proxy from sequence helix/sheet propensity means (SS frac unavailable for PPIKB)."""
    return [float(np.mean([_SCALES["helix"].get(c, 0) for c in seq])),
            float(np.mean([_SCALES["sheet"].get(c, 0) for c in seq]))]


def feat(seq, ps, withss):
    f = _protdcal_descriptors(seq) + pkf(ps) + [float(len(seq))]
    return f + (ssv_pep(seq) if withss else [])


def R(p, y, m):
    p, y = p[m], y[m]; ok = ~(np.isnan(p) | np.isnan(y))
    if ok.sum() < 4:
        return (float("nan"), float("nan"))
    return (float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.mean(np.abs(p[ok] - y[ok]))))


def main():
    ours = {json.loads(l)["pdb"].lower() for l in open(ROOT / "data/pdbbind_peptides.jsonl")}
    # crystal-925 (train base + vlong test)
    c925 = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        ps = e158.pocket_seq(r["pdb"])
        if ps is None:
            continue
        c925.append({"seq": r["seq"], "ps": ps, "y": float(r["y"]), "L": r["length"]})
    # PPIKB pools by length
    ppikb = [json.loads(l) for l in open(ROOT / "data/ppikb_features.jsonl") if json.loads(l).get("desc3d")]
    man = {m["pdb"].lower() for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    seqc = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")}
    t100s = {seqc[p]["seq"] for p in man if p in seqc}

    def ppool(lo, hi):
        out = []
        for r in ppikb:
            if lo <= r["length"] <= hi and r["aff_type"] in ("Kd", "KD", "pKd") and r["pdb"].lower() not in ours \
                    and r["pdb"].lower() not in man and r["seq"] not in t100s and -18 < r["y"] < -2 and r.get("pocket_pkf"):
                out.append({"seq": r["seq"], "pocket_pkf": r["pocket_pkf"], "y": r["y"], "L": r["length"]})
        return out
    p1725 = ppool(17, 25); p1750 = ppool(17, 50)
    print(f"PPIKB vlong pools: 17-25={len(p1725)}  17-50={len(p1750)} (incl XL 26-50)")

    # vlong test = crystal-925 vlong (held out), grouped by pocket
    test = [r for r in c925 if r["L"] >= 17]
    train_base = [r for r in c925 if r["L"] < 17]  # non-vlong 925 never leak vlong test
    print(f"vlong TEST (held-out crystal-925 vlong): n={len(test)}\n")

    yte = np.array([r["y"] for r in test])
    grp_test = np.arange(len(test))  # each its own (no pocket dup within vlong assumed; use LOO-like 5fold by index)

    def build(rows, withss):
        X = []
        for r in rows:
            ps = r.get("ps")
            if ps is not None:
                X.append(feat(r["seq"], ps, withss))
            else:  # PPIKB: precomputed pocket_pkf
                f = _protdcal_descriptors(r["seq"]) + list(r["pocket_pkf"]) + [float(r["L"])]
                X.append(f + (ssv_pep(r["seq"]) if withss else []))
        return np.nan_to_num(X), np.array([r["y"] for r in rows])

    # 5-fold over the vlong TEST: train = train_base + 4 folds of vlong + (aug pool) → predict held fold
    from sklearn.model_selection import KFold

    def eval_cfg(aug, withss, label):
        pred = np.full(len(test), np.nan)
        Xaug = build(aug, withss)[0] if aug else None
        yaug = build(aug, withss)[1] if aug else None
        for tr, te in KFold(5, shuffle=True, random_state=0).split(test):
            trainset = train_base + [test[i] for i in tr]
            Xtr, ytr = build(trainset, withss)
            if aug:
                Xtr = np.vstack([Xtr, Xaug]); ytr = np.concatenate([ytr, yaug])
            Xte, _ = build([test[i] for i in te], withss)
            pred[te] = e202._hgb().fit(Xtr, ytr).predict(Xte)
        r, mae = R(pred, yte, np.ones(len(test), bool))
        print(f"  {label:<38} vlong-test r={r:+.3f}  MAE={mae:.2f}")

    eval_cfg(None, False, "(1) 925 only")
    eval_cfg(p1725, False, "(2) 925 + PPIKB vlong 17-25")
    eval_cfg(p1750, False, "(3) 925 + PPIKB vlong 17-50 (XL subst)")
    eval_cfg(p1750, True, "(4) (3) + structured-peptide SS feats")


if __name__ == "__main__":
    main()
