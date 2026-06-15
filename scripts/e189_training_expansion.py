"""E189 — does ON-DISTRIBUTION data (PPIKB) push us toward beating PPI on crystal (T100)?

Feature set common to all sets via the e179 ProtDCal-3D engine: desc3d(37) + seq-ProtDCal + charge + length.
Train on {925} vs {925 + PPIKB-new}, predict PPI's T100 (held out). r/MAE vs truth, vs PPI's 0.55.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "3"
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e179_protdcal_3d as e179  # noqa: E402
e150 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("e150", ROOT / "scripts/e150_protdcal_descriptors.py"))
importlib.util.spec_from_file_location("e150", ROOT / "scripts/e150_protdcal_descriptors.py").loader.exec_module(e150)
SD, POS, NEG = e150.seq_descriptors, e150.POS, e150.NEG


def fv(seq, desc3d):
    pq = sum(c in POS for c in seq) - sum(c in NEG for c in seq)
    return SD(seq) + list(desc3d) + [float(pq), float(abs(pq)), float(len(seq))]


def met(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = ~(np.isnan(p) | np.isnan(y))
    return float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.mean(np.abs(p[ok] - y[ok])))


def main():
    # T100 test ids (hold out)
    man = {m["pdb"].lower(): m for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    seqc = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")}
    test = []
    for pid, d in seqc.items():
        m = man.get(pid)
        if m is None:
            continue
        pep = next(iter((ROOT / "runs/t100_extract").glob(f"{pid}_*_pep.pdb")), None)
        if pep is None:
            continue
        res = e179.residue_seq_and_coords(pep)
        if res is None:
            continue
        try:
            ship = float(m["ppi_affinity"])
        except (TypeError, ValueError):
            ship = np.nan
        test.append((pid, fv(d["seq"], e179.descriptors(res, 6.0, 3)), float(m["dg_exp"]), ship))
    tid = {t[0] for t in test}

    base = [json.loads(l) for l in open(ROOT / "data/e180_protdcal3d.jsonl") if json.loads(l).get("desc")]
    base = [b for b in base if b["pdb"].lower() not in tid]
    Xb = np.nan_to_num([fv(b["seq"], b["desc"]) for b in base]); yb = np.array([b["y"] for b in base])

    ppikb = [json.loads(l) for l in open(ROOT / "data/ppikb_features.jsonl") if json.loads(l).get("desc3d")]
    base_pdbs = {b["pdb"].lower() for b in base}
    ppikb = [p for p in ppikb if p["pdb"].lower() not in tid and p["pdb"].lower() not in base_pdbs]
    Xp = np.nan_to_num([fv(p["seq"], p["desc3d"]) for p in ppikb]); yp = np.array([p["y"] for p in ppikb])

    Xte = np.nan_to_num([t[1] for t in test]); yte = np.array([t[2] for t in test]); ship = np.array([t[3] for t in test])
    print(f"T100 test={len(test)} | base-925={len(base)} | PPIKB-new={len(ppikb)}")
    print(f"  PPI shipped on T100: r={met(ship,yte)[0]:+.3f} MAE={met(ship,yte)[1]:.2f}  (the target)\n")

    def train_pred(X, y):
        m = HistGradientBoostingRegressor(max_iter=500, max_depth=3, learning_rate=0.03,
                                          l2_regularization=3.0, min_samples_leaf=12, random_state=0).fit(X, y)
        return m.predict(Xte)

    for label, X, y in [("925 only", Xb, yb),
                        ("925 + PPIKB-new", np.vstack([Xb, Xp]), np.concatenate([yb, yp])),
                        ("PPIKB-new only", Xp, yp)]:
        if len(X) == 0:
            continue
        p = train_pred(X, y); r, mae = met(p, yte)
        print(f"  trained on {label:<18} (n={len(X):>4}): T100 r={r:+.3f}  MAE={mae:.2f}")
    print("\n  -> if '925+PPIKB' r climbs toward 0.55, on-distribution DATA is the crystal lever.")


if __name__ == "__main__":
    main()
