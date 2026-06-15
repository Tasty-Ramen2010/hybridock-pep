"""E182 — FULL ProtDCal-3D descriptor space (the "thousands of descriptors" rebuild), then feature-select,
matching PPI's actual methodology (generate large space → select → SMOreg). Uses their 6 properties × 3
contact weightings (Nc/FLC/NLC) × 12 SM-11 groups × 13 invariants = ~2808 descriptors on the peptide
(+ pocket optionally). Trains on 925 PDBbind peptide structures, predicts PPI's T100. Faithfulness =
r vs truth AND correlation with PPI's shipped predictions. Sweeps contact (d,t).

Reuses RCSB structure cache from e180 (all cached now → fast). Run foreground:
    python -u scripts/e182_protdcal_full.py
"""
from __future__ import annotations

import json
import os
import sys
import time
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
sys.path.insert(0, str(ROOT / "third_party/protdcal"))
import e179_protdcal_3d as e179  # noqa: E402
import e180_protdcal_925 as e180  # noqa: E402
from protdcal_spec import GROUPS, PROPS  # noqa: E402

WEIGHTS = ["Nc", "FLC", "NLC"]
INVS = ["N1", "N2", "Ar", "P2", "V", "DE", "RA", "S", "K", "I50", "MI30", "TI30", "SI30"]
GNAMES = list(GROUPS.keys())
# descriptor index = (weight, prop, group, inv)
FULL = [(w, p, g, inv) for w in WEIGHTS for p in PROPS for g in GNAMES for inv in INVS]


def full_vector(res, d_cut, t_cut):
    cache = {p: e179.per_residue_w(res, p, d_cut, t_cut) for p in PROPS}
    aas = np.array([aa for aa, _ in res])
    gmask = {g: np.array([aa in GROUPS[g] for aa in aas]) for g in GNAMES}
    out = []
    for w, p, g, inv in FULL:
        out.append(e179.invariant(cache[p][w][gmask[g]], inv))
    return out


def met(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = ~(np.isnan(p) | np.isnan(y))
    if ok.sum() < 4:
        return float("nan"), float("nan")
    return float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.mean(np.abs(p[ok] - y[ok])))


def main():
    print(f"full descriptor space: {len(FULL)} per (d,t) config", flush=True)
    # training structures: reuse e180's peptide-chain extraction (cached fetch)
    rows = [json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")]
    # test set
    man = {m["pdb"].lower(): m for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    seqc = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")}
    t100_ids = set(seqc) & set(man)

    # gather peptide structures once
    t0 = time.time()
    tr_struct = []
    for r in rows:
        if r["pdb"].lower() in t100_ids:
            continue
        res = e180.peptide_chain(r["pdb"], r["seq"])
        if res is not None:
            tr_struct.append((res, float(r["y"])))
    te_struct = []
    for pid in t100_ids:
        pep = next(iter((ROOT / "runs/t100_extract").glob(f"{pid}_*_pep.pdb")), None)
        if pep is None:
            continue
        res = e179.residue_seq_and_coords(pep)
        if res is None:
            continue
        try:
            ship = float(man[pid]["ppi_affinity"])
        except (TypeError, ValueError):
            ship = np.nan
        te_struct.append((res, float(man[pid]["dg_exp"]), ship))
    print(f"structures: train={len(tr_struct)} test={len(te_struct)}  ({time.time()-t0:.0f}s)", flush=True)

    ytr = np.array([s[1] for s in tr_struct])
    yte = np.array([s[1] for s in te_struct]); ship = np.array([s[2] for s in te_struct])
    print(f"  shipped-PPI vs truth: r={met(ship,yte)[0]:+.3f}\n", flush=True)

    for d_cut, t_cut in [(6.0, 3), (8.0, 3), (10.0, 2)]:
        Xtr = np.nan_to_num([full_vector(s[0], d_cut, t_cut) for s in tr_struct])
        Xte = np.nan_to_num([full_vector(s[0], d_cut, t_cut) for s in te_struct])
        for k in (37, 80, 150):
            mdl = Pipeline([("sc", StandardScaler()),
                            ("sel", SelectKBest(f_regression, k=min(k, Xtr.shape[1]))),
                            ("svr", SVR(kernel="rbf", C=4.0, gamma="scale"))]).fit(Xtr, ytr)
            pred = mdl.predict(Xte)
            rt, mt = met(pred, yte); rc, _ = met(pred, ship)
            print(f"  d={d_cut} t={t_cut} k={k:>3}: r_truth={rt:+.3f} MAE={mt:.2f}  corr_vs_SHIPPED={rc:+.3f}", flush=True)


if __name__ == "__main__":
    main()
