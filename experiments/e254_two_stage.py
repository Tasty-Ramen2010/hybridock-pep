"""E254 — the TWO-STAGE (hierarchical) charged model: the right architecture given the variance decomposition.
  STAGE 1 (offset / receptor-baseline): pocket-ProtDCal -> pocket-mean ΔΔG, leave-pocket-out.
  STAGE 2 (within-pocket): charge-charge pairwise + geometry -> (ΔΔG - pocket-mean), leave-pocket-out (the
          held-out pocket's mean is unknown, so we train on centered values and predict the residual).
  COMBINED: pred = offset_pred(pocket) + within_pred(mut).  Compare to the naive one-GBT kitchen-sink (0.132).
This is a mixed-effects model (predicted random pocket intercept + fixed interaction effects). Established in
stats; the test is whether it lifts CHARGED cross-pocket past the wall. Caches features for fast iteration.
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
sys.path.insert(0, str(ROOT / "scripts"))
CACHE = ROOT / "data" / "e254_recs.json"


def extract():
    sys.path.insert(0, str(ROOT / "src"))
    from Bio.PDB import NeighborSearch, PDBParser
    import e180_protdcal_925 as e180
    import e241_rism_skempi as e241
    import e243_longrange_elec as e243
    from hybridock_pep.scoring.affinity_model import _SCALES
    SN = list(_SCALES.keys()); T2O = e180.T2O
    KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2,
          "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
    _parser = PDBParser(QUIET=True)

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

    def pdesc(seq):
        if not seq:
            return [0.0] * (len(SN) + 3)
        return [float(np.mean([_SCALES[s].get(c, 0) for c in seq])) for s in SN] + \
               [float(np.mean([KD.get(c, 0) for c in seq])),
                (sum(c in "KR" for c in seq) - sum(c in "DE" for c in seq)) / max(len(seq), 1),
                sum(c in "DEKR" for c in seq) / max(len(seq), 1)]

    rows = [json.loads(l) for l in open(ROOT / "data/e165_skempi_struct.jsonl") if json.loads(l)["wt"] in "DEKR"]
    by_pdb = defaultdict(list)
    for r in rows:
        by_pdb[r["pdb"]].append(r)
    recs = []
    for pdb, muts in by_pdb.items():
        f = e180.fetch(pdb)
        if f is None:
            continue
        try:
            st = _parser.get_structure(pdb, str(f))
        except Exception:  # noqa: BLE001
            continue
        pkt = pdesc(iface_seq(st)); groups = e243.charged_groups(st)
        for m in muts:
            _, ch, rn, _ = e241.parse_key(m["key"])
            site = e241.sidechain_centroid(st, ch, rn) if rn else None
            if site is None:
                continue
            qw = e243.QSIGN[m["wt"]]
            ds = sorted((float(np.linalg.norm(c - site)), q) for q, c in groups if 1.5 < np.linalg.norm(c - site) < 14)
            pair = [332.0 * sum(qw * q / (d * d) for d, q in ds), sum(qw * q / (d * d) for d, q in ds if d < 6),
                    sum(1 for d, q in ds if q * qw < 0), sum(1 for d, q in ds if q * qw > 0), ds[0][0] if ds else 14.0, float(qw)]
            geom = [float(m.get("burial") or 0), float(m.get("iface_dist") or 0), float(m.get("n5") or 0), float(m.get("n8") or 0)]
            recs.append({"pdb": pdb, "ddg": m["ddg"], "pair": pair, "geom": geom, "pocket": pkt})
    json.dump(recs, open(CACHE, "w"))
    return recs


def R(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); m = ~(np.isnan(a) | np.isnan(b))
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 3 else np.nan


def main():
    recs = json.load(open(CACHE)) if CACHE.exists() else extract()
    print(f"=== TWO-STAGE charged ΔΔG: n={len(recs)} ===", flush=True)
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    y = np.array([r["ddg"] for r in recs]); grp = np.array([r["pdb"] for r in recs])
    pockets = sorted(set(grp)); pidx = {p: i for i, p in enumerate(pockets)}
    pmean = np.array([y[grp == p].mean() for p in pockets])
    P = np.array([recs[[i for i, r in enumerate(recs) if r["pdb"] == p][0]]["pocket"] for p in pockets])
    PAIR = np.nan_to_num([r["pair"] for r in recs]); GEOM = np.nan_to_num([r["geom"] for r in recs])
    WITHIN = np.hstack([PAIR, GEOM])

    # STAGE 1: offset (leave-pocket-out over pockets)
    off_pred = np.full(len(pockets), np.nan)
    for i in range(len(pockets)):
        tr = [j for j in range(len(pockets)) if j != i]
        sc = StandardScaler().fit(P[tr]); off_pred[i] = Ridge(alpha=3.0).fit(sc.transform(P[tr]), pmean[tr]).predict(sc.transform(P[i:i+1]))[0]
    print(f"  STAGE 1 offset (pocket-ProtDCal -> pocket-mean, LOO): r={R(off_pred, pmean):+.3f}")

    # STAGE 2: within-pocket residual (leave-pocket-out over mutations grouped by pocket)
    yc = y - pmean[[pidx[p] for p in grp]]           # centered ddg (training target)
    win_pred = np.full(len(recs), np.nan)
    from sklearn.model_selection import GroupKFold
    for tr, te in GroupKFold(min(6, len(pockets))).split(WITHIN, yc, grp):
        win_pred[te] = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05,
                                                     l2_regularization=2.0, random_state=0).fit(WITHIN[tr], yc[tr]).predict(WITHIN[te])
    print(f"  STAGE 2 within (pairwise+geom -> centered ddg, LOO): r={R(win_pred, yc):+.3f}")

    # COMBINED: offset_pred(pocket) + within_pred(mut)
    combined = off_pred[[pidx[p] for p in grp]] + win_pred
    print(f"\n  COMBINED two-stage  cross-pocket r = {R(combined, y):+.3f}   (naive one-GBT kitchen-sink was +0.132)")
    # ceilings for context
    oracle_off = pmean[[pidx[p] for p in grp]] + win_pred
    print(f"  [ceiling: TRUE offset + predicted within = {R(oracle_off, y):+.3f}]")
    print(f"  [ceiling: predicted offset + TRUE within = {R(off_pred[[pidx[p] for p in grp]] + yc, y):+.3f}]")


if __name__ == "__main__":
    main()
