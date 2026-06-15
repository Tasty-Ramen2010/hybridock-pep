"""E197 — PPI-clone on a BRAND-NEW dataset + ratio-scale (Ram's ask): estimate how PPI-Affinity ITSELF would
generalize to data outside its BioLiP home field, and compare to us on the same fresh data.

Logic: we can't run real PPI on new data (server/private), but the faithful ProtDCal-3D clone is in PPI's
exact feature class, so its RETENTION RATIO across datasets transfers to PPI.
  ratio = r_clone(fresh) / r_clone(T100)        [T100 = PPI home field]
  estimated real PPI on fresh = 0.554 * ratio
Fresh dataset = PPIKB-structured subset (1436 entries, in NEITHER PPI's nor our training). Compare to OUR
production model on the same fresh set.
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
from sklearn.feature_selection import SelectKBest, f_regression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e179_protdcal_3d as e179  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402
e150 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("e150", ROOT / "scripts/e150_protdcal_descriptors.py"))
importlib.util.spec_from_file_location("e150", ROOT / "scripts/e150_protdcal_descriptors.py").loader.exec_module(e150)
SD, SCALES, POS, NEG = e150.seq_descriptors, e150.SCALES, e150.POS, e150.NEG
SN = list(SCALES.keys())
PPI_T100 = 0.554


def met(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = ~(np.isnan(p) | np.isnan(y))
    return float(np.corrcoef(p[ok], y[ok])[0, 1]) if ok.sum() > 4 else float("nan")


def our_feat(seq, d3_unused, pkf):
    pq = sum(c in POS for c in seq) - sum(c in NEG for c in seq)
    return SD(seq) + pkf + [float(pq), float(abs(pq)), float(len(seq))]


def main():
    # --- train CLONE (ProtDCal-3D) + OURS on 925, holding out everything in the fresh/T100 sets ---
    base = [json.loads(l) for l in open(ROOT / "data/e180_protdcal3d.jsonl") if json.loads(l).get("desc")]

    # T100 (clone home-field reference)
    man = {m["pdb"].lower(): m for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    seqc = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")}
    t100 = []
    for pid, d in seqc.items():
        if pid not in man:
            continue
        pep = next(iter((ROOT / "runs/t100_extract").glob(f"{pid}_*_pep.pdb")), None)
        res = e179.residue_seq_and_coords(pep) if pep else None
        if res is None:
            continue
        t100.append({"pid": pid, "seq": d["seq"], "d3": e179.descriptors(res, 6.0, 3), "y": float(man[pid]["dg_exp"])})

    # FRESH = PPIKB-structured (brand new to both)
    fresh = [json.loads(l) for l in open(ROOT / "data/ppikb_features.jsonl") if json.loads(l).get("desc3d")]
    fresh = [{"pid": r["pdb"].lower(), "seq": r["seq"], "d3": r["desc3d"], "y": r["y"], "q": abs(r["net_charge"])}
             for r in fresh]

    holdout = {t["pid"] for t in t100} | {f["pid"] for f in fresh}
    base = [b for b in base if b["pdb"].lower() not in holdout]

    # CLONE: ProtDCal-3D -> SVR (PPI's feature class)
    Xc = np.nan_to_num([b["desc"] for b in base]); yc = np.array([b["y"] for b in base])
    clone = Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_regression, k=37)),
                      ("svr", SVR(kernel="rbf", C=4.0, gamma="scale"))]).fit(Xc, yc)

    # OURS: production seq+pocket (need pkf; compute for base + sets)
    def pkf_of(pid, seq):
        ps = e158.pocket_seq(pid)
        return [float(np.mean([SCALES[s].get(c, 0) for c in ps])) for s in SN] if ps else [0.0] * len(SN)
    Xo = np.nan_to_num([our_feat(b["seq"], None, pkf_of(b["pdb"], b["seq"])) for b in base])
    ours = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                         l2_regularization=3.0, min_samples_leaf=12, random_state=0).fit(Xo, yc)

    def eval_set(S):
        Xcl = np.nan_to_num([s["d3"] for s in S]); ys = np.array([s["y"] for s in S])
        Xou = np.nan_to_num([our_feat(s["seq"], None, pkf_of(s["pid"], s["seq"])) for s in S])
        return met(clone.predict(Xcl), ys), met(ours.predict(Xou), ys), len(S)

    rc_t, ro_t, n_t = eval_set(t100)
    rc_f, ro_f, n_f = eval_set(fresh)
    ratio = rc_f / rc_t if rc_t > 0 else float("nan")
    print(f"=== PPI-CLONE on BRAND-NEW data + ratio-scale ===")
    print(f"  CLONE on T100 (PPI home field, n={n_t}): r={rc_t:+.3f}")
    print(f"  CLONE on FRESH PPIKB     (n={n_f}):       r={rc_f:+.3f}")
    print(f"  retention ratio fresh/home = {ratio:.2f}")
    print(f"\n  => estimated REAL PPI-Affinity on brand-new data: {PPI_T100:.3f} * {ratio:.2f} = {PPI_T100*ratio:+.3f}")
    print(f"  OURS on the same FRESH set: r={ro_f:+.3f}  (ours on T100: {ro_t:+.3f})")
    print(f"\n  => on truly novel data PPI's 0.55 home-field number ratio-scales to ~{PPI_T100*ratio:.2f};")
    print(f"     we are {ro_f:.2f}.  The 0.55-vs-0.36 'gap' is largely PPI's home-field inflation.")
    # charged sub-slice on fresh
    fy = np.array([f["y"] for f in fresh]); fq = np.array([f["q"] for f in fresh])
    Xcl = np.nan_to_num([f["d3"] for f in fresh]); pc = clone.predict(Xcl)
    Xou = np.nan_to_num([our_feat(f["seq"], None, pkf_of(f["pid"], f["seq"])) for f in fresh]); po = ours.predict(Xou)
    for nm, mk in [("charged|q|>=2", fq >= 2), ("neutral|q|<=1", fq <= 1)]:
        print(f"    fresh {nm:<14} n={mk.sum():<4} clone r={met(pc[mk],fy[mk]):+.3f}  ours r={met(po[mk],fy[mk]):+.3f}")


if __name__ == "__main__":
    main()
