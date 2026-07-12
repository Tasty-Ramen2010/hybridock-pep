"""E252 — test Ram's EXACT hypothesis: a learned PAIRWISE charge-residue potential, trained on the NET ΔΔG
(not the components), applied cross-pocket. For each SKEMPI charge-changing mutation, featurize it ONLY as the
pairwise charge environment of the mutated residue (the user's reduction): for each interface partner charged
group, (q_wt * q_partner / r) summed + counts of opposite/like partners + nearest-partner distance + burial.
This is exactly "AA-of-peptide charge <-> AA-of-receptor charge -> net kcal/mol", learned on the net.

Decisive tests:
 (1) leave-pocket-out r with pairwise-charge features ONLY (the hypothesis) — does the net transfer?
 (2) cross-pocket VARIANCE of a fixed interaction-signature — is the SAME pair-type's net consistent
     across pockets? (high variance => context-dependent => pairwise net is NOT a transferable quantity)
 (3) does adding the GLOBAL context (which the pairwise model excludes BY DESIGN) explain the residual?
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
from Bio.PDB import PDBParser  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e180_protdcal_925 as e180  # noqa: E402
import e241_rism_skempi as e241  # noqa: E402
import e243_longrange_elec as e243  # noqa: E402
_parser = PDBParser(QUIET=True)


def R(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); m = ~(np.isnan(a) | np.isnan(b))
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 3 else np.nan


def main():
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
        groups = e243.charged_groups(st)   # (q, centroid) for ALL charged groups
        for m in muts:
            _, ch, rn, _ = e241.parse_key(m["key"])
            site = e241.sidechain_centroid(st, ch, rn) if rn else None
            if site is None:
                continue
            qw = e243.QSIGN[m["wt"]]
            # PAIRWISE charge environment (the user's reduction): partners within 12 A
            ds = [(np.linalg.norm(c - site), q) for q, c in groups if 1.5 < np.linalg.norm(c - site) < 12]
            ds.sort()
            sum_pair = sum(qw * q / (d * d) for d, q in ds)            # screened pairwise Coulomb (net-ish)
            n_opp = sum(1 for d, q in ds if q * qw < 0)                # opposite charges (salt-bridge partners)
            n_like = sum(1 for d, q in ds if q * qw > 0)               # like charges (repulsion)
            nearest = ds[0][0] if ds else 12.0
            # interaction signature for the variance test: (charge sign, has-opposite-partner, burial bin)
            bur = float(m.get("burial") or 0)
            sig = (int(qw > 0), int(n_opp > 0), int(bur > 100))
            recs.append({"pdb": pdb, "ddg": m["ddg"], "sum_pair": 332.0 * sum_pair, "n_opp": n_opp,
                         "n_like": n_like, "nearest": nearest, "burial": bur, "sig": sig,
                         "netq_pocket": sum(q for q, _ in groups), "n_charged_pocket": len(groups)})
    print(f"=== n={len(recs)} charge-changing muts with pairwise env, {len(by_pdb)} pockets ===")
    y = np.array([r["ddg"] for r in recs]); grp = np.array([r["pdb"] for r in recs])
    pockets = sorted(set(grp))

    # (1) leave-pocket-out with PAIRWISE-CHARGE features only (the hypothesis)
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold, KFold
    PW = ["sum_pair", "n_opp", "n_like", "nearest", "burial"]
    GL = ["netq_pocket", "n_charged_pocket"]

    def cv(keys, splitter, grouped):
        X = np.nan_to_num([[r[k] for k in keys] for r in recs]); pred = np.full(len(y), np.nan)
        sp = splitter.split(X, y, grp) if grouped else splitter.split(X, y)
        for tr, te in sp:
            pred[te] = HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.05,
                                                     random_state=0).fit(X[tr], y[tr]).predict(X[te])
        return R(pred, y)
    print("\n(1) the hypothesis — pairwise-charge potential learned on the NET:")
    print(f"    POOLED-CV         r={cv(PW, KFold(5, shuffle=True, random_state=0), False):+.3f}")
    print(f"    CLUSTERED (leave-pocket-out) r={cv(PW, GroupKFold(min(6, len(pockets))), True):+.3f}  <- does the net TRANSFER?")
    print(f"    + global pocket context     r={cv(PW + GL, GroupKFold(min(6, len(pockets))), True):+.3f}  <- does excluded context help?")

    # (2) cross-pocket consistency of a fixed interaction signature
    print("\n(2) is a FIXED interaction-signature's net consistent across pockets? (the transferability crux)")
    sigg = defaultdict(lambda: defaultdict(list))
    for r in recs:
        sigg[r["sig"]][r["pdb"]].append(r["ddg"])
    print(f"  {'signature(pos,opp,buried)':<26}{'n_pk':>6}{'mean ddg':>10}{'within-pk std':>14}{'BETWEEN-pk std':>15}")
    for sig, pk in sorted(sigg.items(), key=lambda x: -sum(len(v) for v in x[1].values())):
        pkmeans = [np.mean(v) for v in pk.values() if len(v) >= 2]
        if len(pkmeans) < 4:
            continue
        within = np.mean([np.std(v) for v in pk.values() if len(v) >= 2])
        between = np.std(pkmeans)
        alld = [d for v in pk.values() for d in v]
        print(f"  {str(sig):<26}{len(pkmeans):>6}{np.mean(alld):>+10.2f}{within:>14.2f}{between:>15.2f}")
    print("  → if BETWEEN-pocket std >> within-pocket std, the SAME interaction has different nets per pocket")
    print("    = the pairwise net is NOT transferable (it's context/many-body dependent) = hypothesis fails")


if __name__ == "__main__":
    main()
