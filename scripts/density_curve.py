#!/usr/bin/env python3
"""
density_curve.py — Does pose DENSITY (N per complex) rescue the scatter problem?

Runs on logs/gen_n500 as complexes complete (resume-safe generation). For each
complex with >=N poses, subsamples to N in {50,100,200,350,500} and measures,
averaged over random subsamples:

  coverage(N)       P(a <=2 Å pose exists in the N-subset)     [is the needle there?]
  consensus_top1(N) largest Cα-cluster centroid <=2 Å          [does consensus rank it?]
  n_grooves(N)      mean # Cα clusters at 5 Å                   [does scatter saturate?]

If coverage & consensus_top1 climb with N → density helps, scale to all 57.
If consensus_top1 stays flat → more poses don't rescue ranking; generator quality
(convergence), not quantity, is the lever.

Cheap: needs only ref_rmsds (from benchmark_results.json) + Cα coords. No BSA.

Run (any env w/ numpy+sklearn): python3 scripts/density_curve.py [--gen-dir logs/gen_n500]
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
NS = [50, 100, 200, 350, 500]
REPEATS = 30          # random subsamples per N (averaged)
CLUST_T = 5.0
SEED = 0


def read_ca(pdb: str) -> np.ndarray:
    xyz = [(float(l[30:38]), float(l[38:46]), float(l[46:54]))
           for l in open(pdb) if l.startswith("ATOM") and l[12:16].strip() == "CA"]
    return np.array(xyz, np.float32) if xyz else np.empty((0, 3), np.float32)


def consensus_centroid(cas_sub, rmsd_sub):
    """Largest Cα cluster's centroid pose; return its RMSD."""
    from sklearn.cluster import AgglomerativeClustering
    L = min(len(c) for c in cas_sub)
    A = np.stack([c[:L] for c in cas_sub])
    n = len(A)
    D = np.zeros((n, n), np.float32)
    for i in range(n):
        D[i] = np.sqrt(((A - A[i]) ** 2).sum(-1).mean(-1))
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=CLUST_T,
                                  metric="precomputed", linkage="average").fit_predict(D)
    vals, cnts = np.unique(lab, return_counts=True)
    big = vals[cnts.argmax()]
    members = np.where(lab == big)[0]
    # centroid = member with min mean distance to other members
    sub = D[np.ix_(members, members)]
    centroid = members[sub.mean(1).argmin()]
    return rmsd_sub[centroid], len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default="logs/gen_n500")
    a = ap.parse_args()
    gen = REPO / a.gen_dir
    bjson = json.load(open(gen / "benchmark_results.json"))
    rng = np.random.RandomState(SEED)

    # accumulate per-N metrics over complexes
    cov = {N: [] for N in NS}
    con = {N: [] for N in NS}
    grv = {N: [] for N in NS}
    n_cx = 0

    for cn, mdict in bjson.items():
        entry = mdict.get("pretrained", {})
        rr = entry.get("ref_rmsds", [])
        pdir = Path(entry.get("poses_dir", ""))
        if len(rr) < 50:
            continue
        # load Cα coords for all available poses
        cas, rmsds = [], []
        for pi in range(len(rr)):
            pp = pdir / f"pose_{pi}.pdb"
            if not pp.exists():
                continue
            ca = read_ca(str(pp))
            if len(ca) >= 2:
                cas.append(ca); rmsds.append(rr[pi])
        navail = len(rmsds)
        if navail < 50:
            continue
        rmsds = np.array(rmsds)
        n_cx += 1

        for N in NS:
            if N > navail:
                continue
            cvs, cns, gvs = [], [], []
            for _ in range(REPEATS):
                idx = rng.choice(navail, N, replace=False)
                rs = rmsds[idx]
                cvs.append(rs.min() <= 2.0)
                cr, ng = consensus_centroid([cas[i] for i in idx], rs)
                cns.append(cr <= 2.0); gvs.append(ng)
            cov[N].append(np.mean(cvs))
            con[N].append(np.mean(cns))
            grv[N].append(np.mean(gvs))

    print(f"\n{'='*62}")
    print(f"DENSITY CURVE  ({n_cx} complexes with >=50 poses, {a.gen_dir})")
    print(f"{'='*62}")
    print(f"  {'N':>5} {'coverage':>10} {'consensus_top1':>16} {'n_grooves':>11}")
    print(f"  {'-'*44}")
    for N in NS:
        if not cov[N]:
            continue
        print(f"  {N:>5} {100*np.mean(cov[N]):>8.1f}% {100*np.mean(con[N]):>14.1f}% "
              f"{np.mean(grv[N]):>10.1f}")
    print(f"\n  (baseline N=100 from gen_n100: oracle 49%, consensus weak ~12%)")
    if cov[NS[0]] and cov[NS[-1]]:
        dc = 100*(np.mean(con[NS[-1]]) - np.mean(con[NS[0]]))
        print(f"  Δ consensus_top1 ({NS[0]}→{NS[-1]}) = {dc:+.1f} pts → "
              f"{'DENSITY HELPS, scale up' if dc > 3 else 'flat: quantity is NOT the lever'}")
    (REPO / "logs/training_campaign/density_curve.json").write_text(json.dumps(
        {str(N): {"coverage": float(np.mean(cov[N])) if cov[N] else None,
                  "consensus_top1": float(np.mean(con[N])) if con[N] else None,
                  "n_grooves": float(np.mean(grv[N])) if grv[N] else None,
                  "n_cx": len(cov[N])} for N in NS}, indent=2))
    print(f"\n  Saved → logs/training_campaign/density_curve.json")


if __name__ == "__main__":
    main()
