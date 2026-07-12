"""E213 — rebuild PPI-Affinity ONE MORE TIME, new angle: train the ProtDCal-3D clone (PPI's exact feature
class) on a BioLiP-ADJACENT distribution (925 + PPIKB), not PDBbind-925 alone. Every prior rebuild
(E178-182) trained on PDBbind = wrong distribution → ceiling 0.32 vs PPI 0.55. If PPI's edge is DISTRIBUTION,
training the clone on PPIKB (literature-mined, BioLiP-adjacent, closest public proxy to PPI's private T949)
should let it reproduce PPI's T100 number. Tests clone-only, clone+pocket, clone+pocket+seq.
Strict no-leak: T100 pdb+seq excluded from PPIKB training.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_regression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hybridock_pep.scoring.affinity_model import _protdcal_descriptors, _SCALES  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402
import e179_protdcal_3d as e179  # noqa: E402
SN = list(_SCALES.keys())


def rmae(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float); ok = ~(np.isnan(p) | np.isnan(y))
    return (float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.mean(np.abs(p[ok] - y[ok]))))


def pkf(ps):
    return [float(np.mean([_SCALES[s].get(c, 0) for c in ps])) for s in SN] if ps else [0.0] * len(SN)


def clone_pipe():
    return Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_regression, k=37)),
                     ("svr", SVR(kernel="rbf", C=4.0, gamma="scale"))])


def main():
    man = {m["pdb"].lower(): m for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    ours = {json.loads(l)["pdb"].lower() for l in open(ROOT / "data/pdbbind_peptides.jsonl")}
    seqc = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")}
    t100_seqs = {seqc[p]["seq"] for p in man if p in seqc}

    # ---- training pools ----
    # PDBbind-925 (desc3d cached e180)
    pdb925 = [json.loads(l) for l in open(ROOT / "data/e180_protdcal3d.jsonl") if json.loads(l).get("desc")]
    pdb925 = [b for b in pdb925 if b["pdb"].lower() not in man]
    # PPIKB (desc3d + pocket cached e188), no-leak
    ppikb = [json.loads(l) for l in open(ROOT / "data/ppikb_features.jsonl") if json.loads(l).get("desc3d")]
    ppikb = [r for r in ppikb if r["pdb"].lower() not in man and r["pdb"].lower() not in ours
             and r["seq"] not in t100_seqs and -18 < r["y"] < -2]

    def feats925(rows, mode):
        out = []
        for b in rows:
            ps = e158.pocket_seq(b["pdb"]) or ""
            f = list(b["desc"])
            if "pocket" in mode:
                f += pkf(ps)
            if "seq" in mode:
                f += _protdcal_descriptors(b["seq"])
            out.append(f)
        return np.nan_to_num(out), np.array([b["y"] for b in rows])

    def featsppikb(rows, mode):
        out = []
        for r in rows:
            f = list(r["desc3d"])
            if "pocket" in mode:
                f += r["pocket_pkf"]
            if "seq" in mode:
                f += _protdcal_descriptors(r["seq"])
            out.append(f)
        return np.nan_to_num(out), np.array([r["y"] for r in rows])

    # ---- T100 test (with structures for desc3d) ----
    test = []
    for pid, m in man.items():
        d = seqc.get(pid)
        if d is None:
            continue
        pep = next(iter((ROOT / "runs/t100_extract").glob(f"{pid}_*_pep.pdb")), None)
        res = e179.residue_seq_and_coords(pep) if pep else None
        if res is None:
            continue
        try:
            ship = float(m["ppi_affinity"])
        except (TypeError, ValueError):
            continue
        ps = e158.pocket_seq(pid) or ""
        test.append({"d3": e179.descriptors(res, 6.0, 3), "seq": d["seq"], "ps": ps,
                     "y": float(m["dg_exp"]), "ship": ship, "L": len(d["seq"]),
                     "q": abs(sum(c in "KR" for c in d["seq"]) - sum(c in "DE" for c in d["seq"]))})

    def featstest(mode):
        out = []
        for t in test:
            f = list(t["d3"])
            if "pocket" in mode:
                f += pkf(t["ps"])
            if "seq" in mode:
                f += _protdcal_descriptors(t["seq"])
            out.append(f)
        return np.nan_to_num(out)

    y = np.array([t["y"] for t in test]); ship = np.array([t["ship"] for t in test])
    L = np.array([t["L"] for t in test]); q = np.array([t["q"] for t in test])
    nv = L < 17
    rp_full = rmae(ship, y)[0]; rp_fair = rmae(ship[nv], y[nv])[0]
    print(f"PPI shipped on T100: full r={rp_full:+.3f}  fair(no-vlong) r={rp_fair:+.3f}  (the target)")
    print(f"PPIKB training pool (no-leak): {len(ppikb)} | PDBbind-925: {len(pdb925)}\n")

    print(f"  {'clone train set':<26}{'feats':<16}{'T100 r':>8}{'fair r':>8}{'charged':>9}")
    for mode in ["contacts", "contacts+pocket", "contacts+pocket+seq"]:
        for tr_label, builder in [("925", lambda: feats925(pdb925, mode)),
                                  ("925+PPIKB", None), ("PPIKB only", lambda: featsppikb(ppikb, mode))]:
            if tr_label == "925+PPIKB":
                X1, y1 = feats925(pdb925, mode); X2, y2 = featsppikb(ppikb, mode)
                X, yt = np.vstack([X1, X2]), np.concatenate([y1, y2])
            else:
                X, yt = builder()
            m = clone_pipe().fit(X, yt)
            Xte = featstest(mode); p = m.predict(Xte)
            r_full = rmae(p, y)[0]; r_fair = rmae(p[nv], y[nv])[0]; r_ch = rmae(p[q >= 2], y[q >= 2])[0]
            print(f"  {tr_label:<26}{mode:<16}{r_full:>+8.3f}{r_fair:>+8.3f}{r_ch:>+9.3f}")
        print()


if __name__ == "__main__":
    main()
