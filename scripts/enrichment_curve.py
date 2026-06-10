#!/usr/bin/env python3
"""
enrichment_curve.py — Ranker-as-filter: where does it beat random, and where decay?

Ram's filter strategy is right in principle, but how wide should the cutoff be?
A weak ranker concentrates near-natives at the very top; past some k it dilutes
to random and you just pay more 2nd-stage (MM-GBSA) cost for nothing.

For each top-k, compares near-native (<=2Å) COVERAGE of:
  ref2015 top-k   |  BSA+clash top-k  |  random k (Monte-Carlo)  |  oracle
and the ENRICHMENT = ranker_coverage / random_coverage.

Uses cached gen_n100 (57 cx): physics (ref2015), BSA cache, ref_rmsds. No SASA.
Run: python3 scripts/enrichment_curve.py
"""
from __future__ import annotations

import json, pickle, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
GEN_JSON = REPO / "logs" / "gen_n100" / "benchmark_results.json"
ENC_PKL  = REPO / "logs" / "diagnosis" / "feats_gen_n100.pkl"
PHYS_PKL = REPO / "logs" / "diagnosis" / "feats_gen_n100_physics.pkl"
BSA_CACHE = REPO / "logs" / "diagnosis" / "feats_gen_n100_bsa.pkl"

KS = [1, 2, 3, 5, 8, 10, 15, 20, 25]
RAND_REP = 2000


def _z(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / (s if s > 1e-9 else 1.0)


def main():
    bjson = json.load(open(GEN_JSON))
    phys = pickle.load(open(PHYS_PKL, "rb"))
    cache = pickle.load(open(BSA_CACHE, "rb"))
    cxs = sorted(set(k[0] for k in pickle.load(open(ENC_PKL, "rb"))))
    rng = np.random.RandomState(0)

    ref_cov = {k: [] for k in KS}
    bsa_cov = {k: [] for k in KS}
    rnd_cov = {k: [] for k in KS}
    oracle = []
    near_rate = []

    for cn in cxs:
        entry = bjson.get(cn, {}).get("pretrained", {})
        rr = entry.get("ref_rmsds", [])
        if len(rr) < 25:
            continue
        refs, bsas, clashes, rmsds = [], [], [], []
        for pi in range(len(rr)):
            c = cache.get((cn, pi)); pv = phys.get((cn, "pretrained", pi))
            if c is None or pv is None:
                continue
            refs.append(float(pv[13])); bsas.append(c[0]); clashes.append(c[1])
            rmsds.append(rr[pi])
        if len(rmsds) < 25:
            continue
        refs = np.array(refs); rmsds = np.array(rmsds)
        bsa_score = -_z(np.array(bsas)) + _z(np.array(clashes))
        N = len(rmsds)
        ref_order = np.argsort(refs)            # lower = better
        bsa_order = np.argsort(bsa_score)
        oracle.append(rmsds.min() <= 2.0)
        near_rate.append(np.mean(rmsds <= 2.0))

        for k in KS:
            if k > N:
                continue
            ref_cov[k].append(rmsds[ref_order[:k]].min() <= 2.0)
            bsa_cov[k].append(rmsds[bsa_order[:k]].min() <= 2.0)
            # random k coverage (Monte Carlo)
            hits = 0
            for _ in range(RAND_REP):
                idx = rng.choice(N, k, replace=False)
                hits += rmsds[idx].min() <= 2.0
            rnd_cov[k].append(hits / RAND_REP)

    n = len(oracle)
    print(f"\n{'='*72}")
    print(f"ENRICHMENT CURVE — ranker as filter ({n} complexes, gen_n100 N=100)")
    print(f"{'='*72}")
    print(f"  mean near-native rate = {100*np.mean(near_rate):.1f}% of poses ≤2Å")
    print(f"  oracle (any ≤2Å in 100) = {100*np.mean(oracle):.1f}%\n")
    print(f"  {'k':>4} {'ref2015':>9} {'BSA+cl':>9} {'random':>9} {'ref/rand':>9} {'bsa/rand':>9}")
    print(f"  {'-'*54}")
    for k in KS:
        if not ref_cov[k]:
            continue
        rc = 100*np.mean(ref_cov[k]); bc = 100*np.mean(bsa_cov[k]); rd = 100*np.mean(rnd_cov[k])
        er = rc/rd if rd > 0 else float("nan")
        eb = bc/rd if rd > 0 else float("nan")
        print(f"  {k:>4} {rc:>7.1f}% {bc:>7.1f}% {rd:>7.1f}% {er:>8.2f}x {eb:>8.2f}x")

    # find the k where enrichment drops to ~1 (filter no longer beats random)
    print(f"\n  Interpretation:")
    for k in KS:
        if not ref_cov[k]:
            continue
        rd = np.mean(rnd_cov[k])
        er = np.mean(ref_cov[k]) / rd if rd > 0 else 0
        if er < 1.1:
            print(f"  → ref2015 enrichment falls below 1.1x at k={k}: "
                  f"top-{k} is no better than random. Set cutoff BELOW this.")
            break
    else:
        print(f"  → ref2015 stays >1.1x enriched through k={KS[-1]}.")

    out = {str(k): {"ref": float(np.mean(ref_cov[k])) if ref_cov[k] else None,
                    "bsa": float(np.mean(bsa_cov[k])) if bsa_cov[k] else None,
                    "random": float(np.mean(rnd_cov[k])) if rnd_cov[k] else None}
           for k in KS}
    out["_oracle"] = float(np.mean(oracle))
    (REPO / "logs/training_campaign/enrichment_curve.json").write_text(json.dumps(out, indent=2))
    print(f"\n  Saved → logs/training_campaign/enrichment_curve.json")


if __name__ == "__main__":
    main()
