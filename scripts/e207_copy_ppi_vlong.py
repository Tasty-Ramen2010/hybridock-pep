"""E207 — can we beat PPI on vlong by COPYING their feature class? Add ProtDCal-3D intra-peptide contact
descriptors (e179, literally PPI's wNc/wFLC/wNLC) to our 262-feat model, and test T100 per band — does the
contact network crack vlong/long where PPI wins? Also test GATED (add only for L>=13) to avoid hurting other
bands. Crystal (structures available), held-out T100.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hybridock_pep.scoring.affinity_model import build_feature_vector, GEOMETRY_KEYS  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402
import e179_protdcal_3d as e179  # noqa: E402
import e202_band_routing_build as e202  # noqa: E402


def R(p, y, m):
    p, y = p[m], y[m]; ok = ~(np.isnan(p) | np.isnan(y))
    return float(np.corrcoef(p[ok], y[ok])[0, 1]) if ok.sum() > 3 else float("nan")


def base262(g0, seq, pid):
    g = {k: float(g0.get(k, 0.0)) for k in GEOMETRY_KEYS}
    g["pocket_seq"] = e158.pocket_seq(pid) or ""
    x = build_feature_vector(g, seq)
    return list(x[:262]) if x.shape[0] >= 262 else list(np.pad(x, (0, 262 - x.shape[0])))


def main():
    p3d = {json.loads(l)["pdb"].lower(): json.loads(l)["desc"]
           for l in open(ROOT / "data/e180_protdcal3d.jsonl") if json.loads(l).get("desc")}
    # T100 with structures
    man = {m["pdb"].lower(): m for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    cache = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")}
    have = {r["pdb"].lower(): r for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl"))}
    test = []
    for pid, m in man.items():
        d = have.get(pid) or cache.get(pid)
        if d is None:
            continue
        pep = next(iter((ROOT / "runs/t100_extract").glob(f"{pid}_*_pep.pdb")), None)
        res = e179.residue_seq_and_coords(pep) if pep else None
        d3 = e179.descriptors(res, 6.0, 3) if res else None
        if d3 is None:
            continue
        try:
            ship = float(m["ppi_affinity"])
        except (TypeError, ValueError):
            continue
        seq = d["seq"]
        test.append({"b": base262(d, seq, pid), "d3": d3, "y": float(m["dg_exp"]), "ship": ship,
                     "L": len(seq), "q": abs(sum(c in "KR" for c in seq) - sum(c in "DE" for c in seq))})
    tid = set(man)
    tr = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        pid = r["pdb"].lower()
        if pid in tid or pid not in p3d:
            continue
        if e158.pocket_seq(pid) is None:
            continue
        tr.append({"b": base262(r, r["seq"], pid), "d3": p3d[pid], "y": float(r["y"]), "L": r["length"]})
    print(f"T100 with structure n={len(test)}, train n={len(tr)}")

    def feats(rows, mode):
        out = []
        for r in rows:
            f = list(r["b"])
            if mode == "all":
                f += list(r["d3"])
            elif mode == "gated":  # contact descriptors only for long/vlong (L>=13)
                g = 1.0 if r["L"] >= 13 else 0.0
                f += [g * x for x in r["d3"]]
            out.append(f)
        return np.nan_to_num(out)

    ytr = np.array([r["y"] for r in tr])
    y = np.array([r["y"] for r in test]); L = np.array([r["L"] for r in test]); q = np.array([r["q"] for r in test])
    ship = np.array([r["ship"] for r in test])
    print(f"  {'feature set':<24}{'overall':>9}{'long':>8}{'vlong':>8}{'charged':>9}{'neutral':>9}")
    for mode in ["base", "all", "gated"]:
        Xtr = feats(tr, mode); Xte = feats(test, mode)
        m = e202._hgb().fit(Xtr, ytr); p = m.predict(Xte)
        print(f"  {('262 '+('(base)' if mode=='base' else '+ProtDCal-3D '+mode)):<24}"
              f"{R(p,y,np.ones(len(y),bool)):>+9.3f}{R(p,y,(L>=13)&(L<=16)):>+8.3f}{R(p,y,L>=17):>+8.3f}"
              f"{R(p,y,q>=2):>+9.3f}{R(p,y,q<=1):>+9.3f}")
    print(f"  {'PPI shipped':<24}{R(ship,y,np.ones(len(y),bool)):>+9.3f}{R(ship,y,(L>=13)&(L<=16)):>+8.3f}"
          f"{R(ship,y,L>=17):>+8.3f}{R(ship,y,q>=2):>+9.3f}{R(ship,y,q<=1):>+9.3f}")


if __name__ == "__main__":
    main()
