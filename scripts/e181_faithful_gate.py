"""E181 — faithfulness gate for the ProtDCal-3D rebuild: train SMOreg on the 925 PDBbind ProtDCal-3D
descriptors (e180 cache), predict PPI's T100, compare to truth AND to PPI's SHIPPED predictions.
If r_truth ~0.5 and corr_vs_shipped high → faithful PPI clone, usable for the real-pose deployment test.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "3"
import numpy as np  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_regression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e179_protdcal_3d as e179  # noqa: E402


def met(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = ~(np.isnan(p) | np.isnan(y))
    if ok.sum() < 4:
        return float("nan"), float("nan")
    return float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.mean(np.abs(p[ok] - y[ok])))


def main():
    # training: 925 ProtDCal-3D
    tr = [json.loads(l) for l in open(ROOT / "data/e180_protdcal3d.jsonl") if json.loads(l).get("desc")]
    Xtr = np.nan_to_num([d["desc"] for d in tr]); ytr = np.array([d["y"] for d in tr])
    tr_ids = {d["pdb"].lower() for d in tr}
    print(f"train: {len(tr)} PDBbind with ProtDCal-3D descriptors", flush=True)

    # test: T100 (compute descriptors on peptide structures, d=6 t=3)
    man = {m["pdb"].lower(): m for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    seqcache = {json.loads(l)["pdb"].lower(): json.loads(l)
                for l in open(ROOT / "data/t100_extra_features.jsonl")}
    te = []
    for pid, d in seqcache.items():
        m = man.get(pid)
        if m is None or pid in tr_ids:   # hold out any overlap
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
        te.append((pid, e179.descriptors(res, 6.0, 3), float(m["dg_exp"]), ship))
    Xte = np.nan_to_num([t[1] for t in te]); yte = np.array([t[2] for t in te])
    ship = np.array([t[3] for t in te])
    print(f"test: {len(te)} T100 (held out of train)", flush=True)

    mdl = Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_regression, k=37)),
                    ("svr", SVR(kernel="rbf", C=4.0, gamma="scale"))]).fit(Xtr, ytr)
    pred = mdl.predict(Xte)
    r_re, mae_re = met(pred, yte)
    r_sh, mae_sh = met(ship, yte)
    r_corr, _ = met(pred, ship)
    print(f"\n=== FAITHFULNESS GATE (ProtDCal-3D, trained on 925) ===")
    print(f"  rebuilt-PPI-3D vs truth:  r={r_re:+.3f}  MAE={mae_re:.2f}")
    print(f"  shipped-PPI    vs truth:  r={r_sh:+.3f}  MAE={mae_sh:.2f}  (target)")
    print(f"  rebuilt vs SHIPPED preds: r={r_corr:+.3f}  (faithfulness; seq-proxy was ~0)")
    faith = (r_re >= 0.40) and (r_corr >= 0.55)
    print(f"  -> FAITHFUL: {faith}")
    (ROOT / "runs/e181_gate.json").write_text(json.dumps(
        {"r_rebuilt": r_re, "mae_rebuilt": mae_re, "r_shipped": r_sh,
         "r_corr_shipped": r_corr, "faithful": bool(faith), "n_test": len(te), "n_train": len(tr)}, indent=2))


if __name__ == "__main__":
    main()
