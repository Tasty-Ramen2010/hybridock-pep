"""E196 — Ram's engineered per-slice fixes, as GATED features added to the pooled model (global predictions
for non-target slices stay ~unchanged; only the target slice sees the new signal).

  (1) STRUCTURED gate: ProtDCal-3D contact descriptors × 1[helix+sheet>=0.4]  → only structured peptides
      see the contact network (the feature class PPI wins structured with). 0 elsewhere.
  (2) HYDROPHOBICITY COMPLEMENTARITY: peptide mean-hyd × pocket mean-hyd (hopp/kd/eisen products) — the
      "does the peptide match the pocket" term that drives NEUTRAL binding.
  (3) VLONG DE-DILUTION: peptide size aggregates (vol/bulk/mw/sidechain_vol mean) × 1[L>=17] — gives the
      tree an UNDILUTED copy of the vlong size signal (|r|~0.50 single-feature, but pooled dilutes it).

Evaluate on PPI's T100 (held out of 925), per slice, vs base + vs PPI. Only the gated slice should move.
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
from hybridock_pep.scoring.geometry_features import GEOMETRY_FEATURE_KEYS as PROD  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402
import e179_protdcal_3d as e179  # noqa: E402
e150 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("e150", ROOT / "scripts/e150_protdcal_descriptors.py"))
importlib.util.spec_from_file_location("e150", ROOT / "scripts/e150_protdcal_descriptors.py").loader.exec_module(e150)
SD, SCALES, POS, NEG = e150.seq_descriptors, e150.SCALES, e150.POS, e150.NEG
SN = list(SCALES.keys())
SSK = ["helix", "sheet", "ppii", "turn"]
ss = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/ss_features.jsonl")}


def met(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = ~(np.isnan(p) | np.isnan(y))
    return float(np.corrcoef(p[ok], y[ok])[0, 1]) if ok.sum() > 3 else float("nan")


def pep_mean(seq, scale):
    v = [SCALES[scale].get(c, 0.0) for c in seq]
    return float(np.mean(v)) if v else 0.0


def engineered(d, kinds):
    """d must have: seq, pkf, geometry keys, poc_net, d3 (ProtDCal-3D 37), ssv (4)."""
    seq = d["seq"]
    pq = sum(c in POS for c in seq) - sum(c in NEG for c in seq)
    base = SD(seq) + d["pkf"] + [float(d[c]) for c in PROD] + \
        [pq * d["poc_net"], abs(pq) * abs(d["poc_net"]), abs(pq + d["poc_net"]), float(len(seq))]
    add = []
    structured = 1.0 if (d["ssv"][0] + d["ssv"][1]) >= 0.4 else 0.0
    if "struct" in kinds:
        add += [structured * x for x in d["d3"]]            # gated contact descriptors
        add += [structured]
    if "compl" in kinds:
        pock = {s: d["pkf"][SN.index(s)] for s in ("hopp", "kd", "eisen", "arom")}
        add += [pep_mean(seq, s) * pock[s] for s in ("hopp", "kd", "eisen", "arom")]  # complementarity products
    if "vlong" in kinds:
        g = 1.0 if len(seq) >= 17 else 0.0
        add += [g * pep_mean(seq, s) for s in ("vol", "bulk", "mw", "sidechain_vol")]  # de-diluted vlong size
    return base + add


def main():
    man = {m["pdb"].lower(): m for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    have = {r["pdb"].lower(): r for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl"))}
    cache = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")}
    p3d = {json.loads(l)["pdb"].lower(): json.loads(l)["desc"]
           for l in open(ROOT / "data/e180_protdcal3d.jsonl") if json.loads(l).get("desc")}

    def pkf_of(pid, seq):
        ps = e158.pocket_seq(pid)
        return [float(np.mean([SCALES[s].get(c, 0) for c in ps])) for s in SN] if ps else None

    def ssv(pid):
        s = ss.get(pid, {})
        return [float(s.get(k, 0.0)) for k in SSK]

    def d3_of(pid, seq):
        if pid in p3d:
            return p3d[pid]
        pep = next(iter((ROOT / "runs/t100_extract").glob(f"{pid}_*_pep.pdb")), None)
        res = e179.residue_seq_and_coords(pep) if pep else None
        return e179.descriptors(res, 6.0, 3) if res else [0.0] * 37

    test = []
    for pid, m in man.items():
        d = dict(have[pid]) if pid in have else (cache.get(pid))
        if d is None:
            continue
        pk = pkf_of(pid, d["seq"])
        if pk is None:
            continue
        d = dict(d); d["pkf"] = pk; d["ssv"] = ssv(pid); d["d3"] = d3_of(pid, d["seq"])
        try:
            ship = float(m["ppi_affinity"])
        except (TypeError, ValueError):
            ship = np.nan
        seq = d["seq"]; pq = abs(sum(c in POS for c in seq) - sum(c in NEG for c in seq))
        test.append({"d": d, "y": float(m["dg_exp"]), "ship": ship, "L": len(seq), "q": pq})
    tid = {t["d"]["pdb"].lower() for t in test if "pdb" in t["d"]}

    train = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        pid = r["pdb"].lower()
        if pid in tid:
            continue
        pk = pkf_of(pid, r["seq"])
        if pk is None:
            continue
        r = dict(r); r["pkf"] = pk; r["ssv"] = ssv(pid); r["d3"] = d3_of(pid, r["seq"])
        train.append(r)
    ytr = np.array([float(d["y"]) for d in train])
    y = np.array([t["y"] for t in test]); ship = np.array([t["ship"] for t in test])
    L = np.array([t["L"] for t in test]); q = np.array([t["q"] for t in test])

    def fit_pred(kinds):
        Xtr = np.nan_to_num([engineered(d, kinds) for d in train])
        Xte = np.nan_to_num([engineered(t["d"], kinds) for t in test])
        m = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                          l2_regularization=3.0, min_samples_leaf=12, random_state=0).fit(Xtr, ytr)
        return m.predict(Xte)

    cfgs = {"base": [], "+struct": ["struct"], "+compl": ["compl"], "+vlong": ["vlong"],
            "+ALL": ["struct", "compl", "vlong"]}
    preds = {k: fit_pred(v) for k, v in cfgs.items()}

    slices = [("OVERALL", np.ones(len(test), bool)), ("vlong>=17", L >= 17),
              ("long13-16", (L >= 13) & (L <= 16)), ("neutral|q|<=1", q <= 1),
              ("structured", np.array([(t["d"]["ssv"][0] + t["d"]["ssv"][1]) >= 0.4 for t in test])),
              ("charged|q|>=2", q >= 2)]
    print("=== Ram's engineered GATED features on T100 (held out of 925) — per slice r ===")
    print(f"  {'slice':<14}{'n':>4}" + "".join(f"{k:>9}" for k in cfgs) + f"{'PPI':>9}")
    for nm, mask in slices:
        if mask.sum() < 4:
            continue
        cells = "".join(f"{met(preds[k][mask], y[mask]):>9.3f}" for k in cfgs)
        print(f"  {nm:<14}{int(mask.sum()):>4}{cells}{met(ship[mask], y[mask]):>9.3f}")
    print("\n  (gated = non-target slices should stay ~flat; target slice should move toward PPI)")


if __name__ == "__main__":
    main()
