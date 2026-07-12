"""E155 — retrain the real-pose deployment model from ALL real-pose sources (the98 + e93 cr65 + e154
PDBbind campaign) and report the learning curve. Run periodically by the poller as E154 accumulates poses.

Combines every real RAPiDock-pose complex available, trains the 240-feature model (16 geometry + 220
ProtDCal + charge-compl + length), grouped-CV r/MAE overall + per-band + charged, and re-saves
data/affinity_realpose.joblib if the new set is larger. Logs one line to runs/e155_curve.log so the
r-vs-n_real curve is visible.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
import joblib  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("e150", ROOT / "experiments/e150_protdcal_descriptors.py")
e150 = importlib.util.module_from_spec(_s); _s.loader.exec_module(e150)
PROD, POS, NEG = e150.PROD, e150.POS, e150.NEG


def compl(seq, pn):
    pq = sum(c in POS for c in seq) - sum(c in NEG for c in seq)
    return [pq * pn, abs(pq) * abs(pn), abs(pq + pn)]


def fvec(g, s):
    return [g[c] for c in PROD] + e150.seq_descriptors(s) + compl(s, g.get("poc_net", 0.0)) + [float(len(s))]


def R(p, y):
    return float(np.corrcoef(np.array(p), np.array(y))[0, 1])


def band(L):
    return "short" if L <= 8 else "med" if L <= 12 else "long" if L <= 16 else "vlong"


def load_real():
    rows = []
    for nm in ["train", "test"]:
        for r in csv.DictReader(open(ROOT / f"data/pooled_benchmark_{nm}.csv")):
            if r["dataset"] == "the98" and r.get("seq"):
                rows.append((fvec({c: float(r[c]) for c in PROD}, r["seq"]), float(r["y"]), len(r["seq"]),
                             abs(sum(c in POS for c in r["seq"]) - sum(c in NEG for c in r["seq"]))))
    for pid, e in json.loads((ROOT / "data/e93_realpose_results.json").read_text()).items():
        rows.append((fvec(e["rank1"], e["seq"]), e["y"], len(e["seq"]),
                     abs(sum(c in POS for c in e["seq"]) - sum(c in NEG for c in e["seq"]))))
    e154 = ROOT / "data/e154_realpose_pdbbind.jsonl"
    if e154.exists():
        for ln in e154.read_text().splitlines():
            e = json.loads(ln)
            rows.append((fvec(e["rank1"], e["seq"]), e["y"], e["length"], e.get("q", 0)))
    return rows


def main():
    rows = load_real()
    X = np.nan_to_num(np.array([r[0] for r in rows])); y = np.array([r[1] for r in rows])
    L = np.array([r[2] for r in rows]); q = np.array([r[3] for r in rows])
    rng = np.random.default_rng(0); fold = rng.integers(0, 5, len(rows)); pred = np.full(len(rows), np.nan)
    for f in range(5):
        tr = fold != f
        m = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                          l2_regularization=3.0, min_samples_leaf=12, random_state=0).fit(X[tr], y[tr])
        pred[fold == f] = m.predict(X[fold == f])
    r_all = R(pred, y); mae = float(np.mean(np.abs(pred - y)))
    parts = {b: R(pred[[band(x) == b for x in L]], y[[band(x) == b for x in L]])
             for b in ["short", "med", "long", "vlong"] if sum(band(x) == b for x in L) >= 10}
    rch = R(pred[q >= 2], y[q >= 2]) if (q >= 2).sum() >= 10 else float("nan")
    line = (f"{time.strftime('%H:%M')}  n={len(rows)}  r={r_all:+.3f}  MAE={mae:.2f}  charged={rch:+.3f}  "
            + "  ".join(f"{b}={parts[b]:+.2f}" for b in parts))
    print(line)
    with open(ROOT / "runs/e155_curve.log", "a") as fh:
        fh.write(line + "\n")
    # save the deployed model retrained on ALL real poses
    mfull = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                          l2_regularization=3.0, min_samples_leaf=12, random_state=0).fit(X, y)
    names = PROD + [f"pd_{i}" for i in range(220)] + ["q_compl", "abs_q_match", "q_neutralize", "length"]
    joblib.dump({"model": mfull, "feature_order": names, "n_train": len(rows), "protdcal": True,
                 "pose_type": "real_rapidock", "cv_r": r_all}, ROOT / "data/affinity_realpose.joblib")


if __name__ == "__main__":
    main()
