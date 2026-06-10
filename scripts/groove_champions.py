#!/usr/bin/env python3
"""
groove_champions.py — Ram's keep-every-groove idea.

Two-stage-on-largest-cluster failed because the biggest cluster is the correct
groove only 26% of the time. Fix: cluster the 100 poses by binding location,
take EACH groove's champion (best-fit pose by ref2015 / BSA+clash, which
discriminate well WITHIN a groove where the core is fixed), then rank champions.
The correct groove's near-native survives into the final set even if its groove
is small — which largest-cluster threw away.

KEY METRIC — coverage: does the champion set (one per groove) contain a
near-native MORE often than global top-10 (26%)? That's the ceiling.

Clustering: no-superposition Cα RMSD (poses are in the receptor frame, so this
preserves WHICH pocket). Reuses cached per-pose BSA (feats_gen_n100_bsa.pkl).

Run (rapidock/score-env): python3 scripts/groove_champions.py
"""
from __future__ import annotations

import json, pickle, sys, time
from pathlib import Path
import numpy as np
from scipy import stats as sp

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
BASE = Path("/home/igem/unknown_software/datasets/training_formatted_peppc")
GEN_JSON = REPO / "logs" / "gen_n100" / "benchmark_results.json"
ENC_PKL  = REPO / "logs" / "diagnosis" / "feats_gen_n100.pkl"
PHYS_PKL = REPO / "logs" / "diagnosis" / "feats_gen_n100_physics.pkl"
BSA_CACHE = REPO / "logs" / "diagnosis" / "feats_gen_n100_bsa.pkl"
OUT = REPO / "logs" / "training_campaign" / "groove_champions.json"

THRESHOLDS = [3.0, 5.0, 8.0]   # Cα-RMSD groove definition (Å)


def read_ca(pdb: str) -> np.ndarray:
    xyz = [(float(l[30:38]), float(l[38:46]), float(l[46:54]))
           for l in open(pdb) if l.startswith("ATOM") and l[12:16].strip() == "CA"]
    return np.array(xyz, np.float32) if xyz else np.empty((0, 3), np.float32)


def _z(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / (s if s > 1e-9 else 1.0)


def pairwise_ca(cas):
    L = min(len(c) for c in cas)
    A = np.stack([c[:L] for c in cas])
    n = len(A)
    D = np.zeros((n, n), np.float32)
    for i in range(n):
        D[i] = np.sqrt(((A - A[i]) ** 2).sum(-1).mean(-1))
    return D


def main():
    from sklearn.cluster import AgglomerativeClustering
    bjson = json.load(open(GEN_JSON))
    phys = pickle.load(open(PHYS_PKL, "rb"))
    cache = pickle.load(open(BSA_CACHE, "rb"))
    cxs = sorted(set(k[0] for k in pickle.load(open(ENC_PKL, "rb"))))

    # load all per-complex arrays once
    data = {}
    for cn in cxs:
        entry = bjson.get(cn, {}).get("pretrained", {})
        rr = entry.get("ref_rmsds", [])
        pdir = Path(entry.get("poses_dir", ""))
        if len(rr) < 10:
            continue
        bsas, clashes, refs, rmsds, cas = [], [], [], [], []
        for pi in range(len(rr)):
            c = cache.get((cn, pi)); pv = phys.get((cn, "pretrained", pi))
            ca = read_ca(str(pdir / f"pose_{pi}.pdb"))
            if c is None or pv is None or len(ca) < 2:
                continue
            bsas.append(c[0]); clashes.append(c[1]); refs.append(float(pv[13]))
            rmsds.append(rr[pi]); cas.append(ca)
        if len(rmsds) < 10:
            continue
        data[cn] = dict(bsa=np.array(bsas), clash=np.array(clashes),
                        ref=np.array(refs), rmsd=np.array(rmsds), cas=cas)

    n = len(data)
    print(f"Loaded {n} complexes\n")

    # global baselines (independent of threshold)
    g_ref1 = g_ref3 = g_bsa1 = oracle = 0.0
    cov_top10 = 0.0
    for cn, d in data.items():
        bsa_score = -_z(d["bsa"]) + _z(d["clash"])
        r = d["rmsd"]
        g_ref1 += r[np.argmin(d["ref"])] <= 2.0
        g_ref3 += r[np.argsort(d["ref"])[:3]].min() <= 2.0
        g_bsa1 += r[np.argmin(bsa_score)] <= 2.0
        oracle += r.min() <= 2.0
        cov_top10 += r[np.argsort(d["ref"])[:10]].min() <= 2.0
    print(f"GLOBAL baselines ({n} complexes):")
    print(f"  ref2015 top-1 Hit@2Å      = {100*g_ref1/n:.1f}%")
    print(f"  ref2015 top-3 Hit@2Å      = {100*g_ref3/n:.1f}%")
    print(f"  BSA+clash top-1 Hit@2Å    = {100*g_bsa1/n:.1f}%")
    print(f"  ref2015 top-10 coverage   = {100*cov_top10/n:.1f}%")
    print(f"  oracle (best of 100)      = {100*oracle/n:.1f}%\n")

    results = {}
    for T in THRESHOLDS:
        n_grooves, champ_size = [], []
        cov_champ = 0.0                       # champion set contains near-native
        # champion rankings → top-1/top-3 Hit@2Å
        c1 = {k: 0.0 for k in ["bsa", "ref", "size", "size_bsa"]}
        c3 = {k: 0.0 for k in ["bsa", "ref", "size", "size_bsa"]}
        for cn, d in data.items():
            cas, r = d["cas"], d["rmsd"]
            bsa_score = -_z(d["bsa"]) + _z(d["clash"])
            D = pairwise_ca(cas)
            lab = AgglomerativeClustering(
                n_clusters=None, distance_threshold=T,
                metric="precomputed", linkage="average").fit_predict(D)
            groves = np.unique(lab)
            n_grooves.append(len(groves))

            champs = []  # (pose_idx, bsa_score, ref, size)
            for g in groves:
                idx = np.where(lab == g)[0]
                champ = idx[np.argmin(bsa_score[idx])]    # best fit in groove
                champs.append((champ, bsa_score[champ], d["ref"][champ], len(idx)))
            champ_size.append(len(champs))
            ci = np.array([c[0] for c in champs])
            cb = np.array([c[1] for c in champs])
            cr = np.array([c[2] for c in champs])
            cs = np.array([c[3] for c in champs])
            cov_champ += r[ci].min() <= 2.0

            def hit(order):
                o = ci[order]
                return (r[o[0]] <= 2.0, r[o[:3]].min() <= 2.0)
            for key, order in [
                ("bsa",  np.argsort(cb)),
                ("ref",  np.argsort(cr)),
                ("size", np.argsort(-cs)),
                ("size_bsa", np.lexsort((cb, -cs))),  # big grooves first, BSA tiebreak
            ]:
                h1, h3 = hit(order)
                c1[key] += h1; c3[key] += h3

        print(f"── Threshold {T} Å ──  mean grooves={np.mean(n_grooves):.1f}  "
              f"champ-set={np.mean(champ_size):.1f}")
        print(f"  champion-set coverage      = {100*cov_champ/n:.1f}%  "
              f"(vs global top-10 {100*cov_top10/n:.1f}%, oracle {100*oracle/n:.1f}%)")
        print(f"  champion ranking top-1 / top-3 Hit@2Å:")
        for key in ["bsa", "ref", "size", "size_bsa"]:
            print(f"    by {key:<9} {100*c1[key]/n:>6.1f}% / {100*c3[key]/n:>6.1f}%")
        print()
        results[str(T)] = dict(
            mean_grooves=float(np.mean(n_grooves)),
            coverage=float(cov_champ/n),
            top1={k: float(c1[k]/n) for k in c1},
            top3={k: float(c3[k]/n) for k in c3})

    results["_global"] = dict(ref1=g_ref1/n, ref3=g_ref3/n, bsa1=g_bsa1/n,
                              cov_top10=cov_top10/n, oracle=oracle/n, n=n)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
