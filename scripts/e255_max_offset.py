"""E255 — maximal attack on the OFFSET (the whole charged wall, per E254 ceiling). Predict per-pocket mean
charged ΔΔG from the RICHEST receptor-pocket representation: full ProtDCal-220 on the interface sequence +
pocket geometry + composition. LOO over pockets. If offset climbs past 0.25 toward ~0.4, the two-stage
combined lifts materially. Tests several representations + a permutation null to be sure it's real.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
CACHE = ROOT / "data" / "e255_offset.json"


def extract():
    from Bio.PDB import NeighborSearch, PDBParser
    import e180_protdcal_925 as e180
    import e241_rism_skempi as e241
    from hybridock_pep.scoring.affinity_model import _protdcal_descriptors, _SCALES
    SN = list(_SCALES.keys()); T2O = e180.T2O
    _parser = PDBParser(QUIET=True)
    KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2,
          "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}

    def iface_seq(st):
        ns = NeighborSearch([a for a in st[0].get_atoms()]); iface = set()
        for ch in st[0]:
            for r in ch:
                if r.id[0] != " ":
                    continue
                for a in r:
                    if any(nb.get_parent().get_parent().id != ch.id for nb in ns.search(a.coord, 8.0)):
                        iface.add((ch.id, r.id[1], T2O.get(r.resname, ""))); break
        return "".join(s for _, _, s in sorted(iface) if s)

    rows = [json.loads(l) for l in open(ROOT / "data/e165_skempi_struct.jsonl") if json.loads(l)["wt"] in "DEKR"]
    by_pdb = defaultdict(list)
    for r in rows:
        by_pdb[r["pdb"]].append(r)
    out = {}
    for pdb, muts in by_pdb.items():
        f = e180.fetch(pdb)
        if f is None:
            continue
        try:
            st = _parser.get_structure(pdb, str(f))
        except Exception:  # noqa: BLE001
            continue
        seq = iface_seq(st)
        if not seq:
            continue
        pd220 = _protdcal_descriptors(seq)
        scale22 = [float(np.mean([_SCALES[s].get(c, 0) for c in seq])) for s in SN]
        comp = [seq.count(a) / len(seq) for a in "ACDEFGHIKLMNPQRSTVWY"]
        geom = [float(len(seq)), float(np.mean([KD.get(c, 0) for c in seq])),
                (sum(c in "KR" for c in seq) - sum(c in "DE" for c in seq)) / len(seq),
                sum(c in "DEKR" for c in seq) / len(seq), sum(c in "FWY" for c in seq) / len(seq)]
        out[pdb] = {"mean_ddg": float(np.mean([m["ddg"] for m in muts])), "n": len(muts),
                    "pd220": pd220, "scale22": scale22, "comp": comp, "geom": geom}
    json.dump(out, open(CACHE, "w"))
    return out


def R(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); m = ~(np.isnan(a) | np.isnan(b))
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 3 else np.nan


def main():
    d = json.load(open(CACHE)) if CACHE.exists() else extract()
    pks = [p for p in d if d[p]["n"] >= 3]                 # pockets with >=3 muts (stable mean)
    y = np.array([d[p]["mean_ddg"] for p in pks])
    print(f"=== MAX OFFSET: {len(pks)} pockets (>=3 muts), offset std={y.std():.2f} ===")
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(0)

    def loo(X, alpha=5.0, pca=None):
        pred = np.full(len(y), np.nan)
        for i in range(len(y)):
            tr = [j for j in range(len(y)) if j != i]
            Xt, Xi = X[tr], X[i:i + 1]
            sc = StandardScaler().fit(Xt); Xt, Xi = sc.transform(Xt), sc.transform(Xi)
            if pca:
                pc = PCA(n_components=min(pca, len(tr) - 1)).fit(Xt); Xt, Xi = pc.transform(Xt), pc.transform(Xi)
            pred[i] = Ridge(alpha=alpha).fit(Xt, y[tr]).predict(Xi)[0]
        return R(pred, y)

    reps = {
        "scale22+geom (E254 repro)": np.nan_to_num([d[p]["scale22"] + d[p]["geom"] for p in pks]),
        "composition20+geom": np.nan_to_num([d[p]["comp"] + d[p]["geom"] for p in pks]),
        "ProtDCal220": np.nan_to_num([d[p]["pd220"] for p in pks]),
        "ProtDCal220 (PCA-10)": np.nan_to_num([d[p]["pd220"] for p in pks]),
        "ALL (220+22+comp+geom)": np.nan_to_num([d[p]["pd220"] + d[p]["scale22"] + d[p]["comp"] + d[p]["geom"] for p in pks]),
    }
    print(f"\n  {'offset representation':<32}{'LOO r':>8}{'perm-null95':>13}")
    for nm, X in reps.items():
        pca = 10 if "PCA" in nm else None
        alpha = 20.0 if X.shape[1] > 100 else 5.0
        r = loo(X, alpha, pca)
        null = sorted(loo(X[rng.permutation(len(y))] if False else X, alpha, pca) for _ in range(1))  # placeholder
        # proper permutation: shuffle y
        nulls = []
        for _ in range(200):
            ys = rng.permutation(y)
            pred = np.full(len(y), np.nan)
            for i in range(len(y)):
                tr = [j for j in range(len(y)) if j != i]
                sc = StandardScaler().fit(X[tr]); Xt, Xi = sc.transform(X[tr]), sc.transform(X[i:i + 1])
                if pca:
                    from sklearn.decomposition import PCA as _P
                    pc = _P(n_components=min(pca, len(tr) - 1)).fit(Xt); Xt, Xi = pc.transform(Xt), pc.transform(Xi)
                pred[i] = Ridge(alpha=alpha).fit(Xt, ys[tr]).predict(Xi)[0]
            nulls.append(R(pred, ys))
        print(f"  {nm:<32}{r:>+8.3f}{np.quantile(nulls, 0.95):>+13.3f}")


if __name__ == "__main__":
    main()
