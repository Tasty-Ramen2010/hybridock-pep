"""E253 — the KITCHEN-SINK charged model (Ram's maximal hypothesis). Train ML on the NET ΔΔG with EVERY
lever: charge-charge pairwise + local geometry + pocket hydrophobicity + RECEPTOR-POCKET ProtDCal (22) +
global baseline. Built up incrementally so we see exactly which group adds cross-pocket signal and which is
dead. Plus the crux test: can rich pocket-ProtDCal predict the OFFSET (receptor-baseline) cross-pocket, beating
the ~0.15 wall? Honest metrics: POOLED, CLUSTERED(leave-pocket-out), within-pocket-centered.
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
from Bio.PDB import NeighborSearch, PDBParser  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import e180_protdcal_925 as e180  # noqa: E402
import e241_rism_skempi as e241  # noqa: E402
import e243_longrange_elec as e243  # noqa: E402
from hybridock_pep.scoring.affinity_model import _SCALES  # noqa: E402
_parser = PDBParser(QUIET=True)
SN = list(_SCALES.keys())
T2O = e180.T2O
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2,
      "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}


def R(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); m = ~(np.isnan(a) | np.isnan(b))
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 3 else np.nan


def interface_seq(st):
    """receptor interface residues: any standard residue with an atom within 8A of a DIFFERENT chain."""
    atoms = [a for a in st[0].get_atoms()]
    ns = NeighborSearch(atoms)
    iface = set()
    for ch in st[0]:
        for r in ch:
            if r.id[0] != " ":
                continue
            for a in r:
                for nb in ns.search(a.coord, 8.0):
                    if nb.get_parent().get_parent().id != ch.id:
                        iface.add((ch.id, r.id[1], T2O.get(r.resname, "")))
                        break
                else:
                    continue
                break
    return "".join(s for _, _, s in sorted(iface) if s)


def pocket_desc(seq):
    if not seq:
        return [0.0] * (len(SN) + 3)
    pd = [float(np.mean([_SCALES[s].get(c, 0) for c in seq])) for s in SN]   # 22 pocket ProtDCal
    hyd = float(np.mean([KD.get(c, 0) for c in seq]))
    netq = (sum(c in "KR" for c in seq) - sum(c in "DE" for c in seq)) / max(len(seq), 1)
    frq = sum(c in "DEKR" for c in seq) / max(len(seq), 1)
    return pd + [hyd, netq, frq]


def main():
    rows = [json.loads(l) for l in open(ROOT / "data/e165_skempi_struct.jsonl") if json.loads(l)["wt"] in "DEKR"]
    by_pdb = defaultdict(list)
    for r in rows:
        by_pdb[r["pdb"]].append(r)
    pdesc = {}
    recs = []
    for pdb, muts in by_pdb.items():
        f = e180.fetch(pdb)
        if f is None:
            continue
        try:
            st = _parser.get_structure(pdb, str(f))
        except Exception:  # noqa: BLE001
            continue
        if pdb not in pdesc:
            pdesc[pdb] = pocket_desc(interface_seq(st))
        groups = e243.charged_groups(st)
        for m in muts:
            _, ch, rn, _ = e241.parse_key(m["key"])
            site = e241.sidechain_centroid(st, ch, rn) if rn else None
            if site is None:
                continue
            qw = e243.QSIGN[m["wt"]]
            ds = sorted((np.linalg.norm(c - site), q) for q, c in groups if 1.5 < np.linalg.norm(c - site) < 14)
            # CHARGE-CHARGE pairwise (not AA): screened sum + shells + partner counts
            pair = [332.0 * sum(qw * q / (d * d) for d, q in ds),
                    sum(qw * q / (d * d) for d, q in ds if d < 6), sum(qw * q / (d * d) for d, q in ds if 6 <= d < 14),
                    sum(1 for d, q in ds if q * qw < 0), sum(1 for d, q in ds if q * qw > 0),
                    ds[0][0] if ds else 14.0, float(qw)]
            geom = [float(m.get("burial") or 0), float(m.get("iface_dist") or 0), float(m.get("n5") or 0), float(m.get("n8") or 0)]
            recs.append({"pdb": pdb, "ddg": m["ddg"], "pair": pair, "geom": geom, "pocket": pdesc[pdb]})
    print(f"=== KITCHEN-SINK charged ΔΔG: n={len(recs)}, {len(by_pdb)} pockets ===", flush=True)
    y = np.array([r["ddg"] for r in recs]); grp = np.array([r["pdb"] for r in recs])
    pockets = sorted(set(grp)); pmean = {p: y[grp == p].mean() for p in pockets}

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold, KFold

    def build(groups):
        out = []
        for r in recs:
            v = []
            for g in groups:
                v += r[g]
            out.append(v)
        return np.nan_to_num(out)

    def cv(groups, splitter, grouped):
        X = build(groups); pred = np.full(len(y), np.nan)
        sp = splitter.split(X, y, grp) if grouped else splitter.split(X, y)
        for tr, te in sp:
            pred[te] = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05,
                                                     l2_regularization=2.0, random_state=0).fit(X[tr], y[tr]).predict(X[te])
        return pred

    print(f"\n  {'feature set':<46}{'POOLED':>8}{'CLUSTERED':>11}{'within-cent':>12}")
    combos = [("charge-charge pairwise", ["pair"]),
              ("+ local geometry", ["pair", "geom"]),
              ("+ pocket hydrophobicity+ProtDCal", ["pair", "geom", "pocket"])]
    for nm, gs in combos:
        pc = cv(gs, KFold(5, shuffle=True, random_state=0), False)
        cc = cv(gs, GroupKFold(min(6, len(pockets))), True)
        ok = ~np.isnan(cc)
        yc = y[ok] - np.array([pmean[p] for p in grp[ok]]); pcen = cc[ok] - np.array([cc[ok][grp[ok] == p].mean() for p in grp[ok]])
        print(f"  {nm:<46}{R(pc, y):>+8.3f}{R(cc, y):>+11.3f}{R(pcen, yc):>+12.3f}")

    # THE CRUX: can rich pocket-ProtDCal predict the OFFSET cross-pocket? (beat the -0.10/0.15 wall)
    print("\n  === offset (receptor-baseline) predictability from RICH pocket features ===")
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    P = np.array([pdesc[p] for p in pockets]); by = np.array([pmean[p] for p in pockets])
    po = np.full(len(pockets), np.nan)
    for i in range(len(pockets)):
        tr = [j for j in range(len(pockets)) if j != i]
        sc = StandardScaler().fit(P[tr]); po[i] = Ridge(alpha=3.0).fit(sc.transform(P[tr]), by[tr]).predict(sc.transform(P[i:i+1]))[0]
    print(f"  pocket-ProtDCal(22)+hyd+charge -> offset (LOO over {len(pockets)} pockets): r={R(po, by):+.3f}")
    print(f"  (crude net-charge was -0.105; proper PB was -0.003; sequence/ESM wall ~0.15)")


if __name__ == "__main__":
    main()
