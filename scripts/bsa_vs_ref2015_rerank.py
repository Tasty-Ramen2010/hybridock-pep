#!/usr/bin/env python
"""Is BSA+clash a drop-in replacement for ref2015 as the stage-1 ranker? The standing check.

THE OPEN QUESTION THIS ANSWERS. ref2015 needs PyRosetta and a repack; BSA+clash needs neither and
is orders of magnitude faster. If the two pick the same pose, stage 1 can drop PyRosetta entirely
and the whole pipeline gets cheaper for free. So this is re-run on every status check rather than
assumed -- it has tied exactly at 18 and 20 complexes before, and a tie that stops being a tie is
exactly the kind of thing that slips past if nobody looks.

Same protocol as mmgbsa_rerank_top5.py so the numbers sit on one scale: stage-1 takes the top 5
poses by ref2015 total_score, and each ranker then picks its own #1 from inside that five. Hit is
top-1 within 2.0 A. The oracle is the best of the five and bounds what any reranker can do here.

BSA+clash is scored as (buried surface area) - LAMBDA * (clash), higher is better: more interface
buried is good, steric overlap is not. LAMBDA is not fitted -- it is the same fixed value used in
the earlier comparisons, because tuning it against this same 57-complex set would turn a control
into a fit.

EQUAL means the two rankers make identical top-1 choices on every complex, which is the strong
claim. Agreeing on the hit RATE while disagreeing on which pose is a weaker result and is
reported separately, because it would mean the tie is a coincidence of this sample.

Usage: bsa_vs_ref2015_rerank.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

REPO = Path("/home/igem/unknown_software")
GEN_JSON = REPO / "logs/gen_n100/benchmark_results.json"
PHYS_PKL = REPO / "logs/diagnosis/feats_gen_n100_physics.pkl"
ENC_PKL = REPO / "logs/diagnosis/feats_gen_n100.pkl"
BSA_PKL = REPO / "logs/diagnosis/feats_gen_n100_bsa.pkl"
DG_CACHE = REPO / "logs/diagnosis/mmgbsa_top5_dg.pkl"
TOPK = 5
LAMBDA = 10.0          # fixed, not fitted -- see docstring
HIT = 2.0


def main() -> None:
    bjson = json.load(open(GEN_JSON))
    phys = pickle.load(open(PHYS_PKL, "rb"))
    bsa = pickle.load(open(BSA_PKL, "rb"))
    dg_cache = pickle.load(open(DG_CACHE, "rb")) if DG_CACHE.exists() else {}
    cxs = sorted({k[0] for k in pickle.load(open(ENC_PKL, "rb"))})

    rows, same_pick, n_bsa_missing = [], 0, 0
    for cn in cxs:
        entry = bjson.get(cn, {}).get("pretrained", {})
        rr = entry.get("ref_rmsds", [])
        if len(rr) < TOPK:
            continue
        refs = np.array([phys.get((cn, "pretrained", pi), [0] * 14)[13]
                         for pi in range(len(rr))], float)
        top5 = [int(i) for i in np.argsort(refs)[:TOPK]]
        rmsd5 = np.array([rr[pi] for pi in top5])

        vals = [bsa.get((cn, pi)) for pi in top5]
        if any(v is None for v in vals):
            n_bsa_missing += 1
            continue
        # higher buried area is better, clash is a penalty -> argmax
        score = np.array([v[0] - LAMBDA * v[1] for v in vals], float)
        bsa_idx = int(np.argmax(score))

        dgs = [dg_cache.get((cn, pi)) for pi in top5]
        mm_idx = int(np.argmin([d if d is not None else 1e9 for d in dgs])) \
            if any(d is not None for d in dgs) else 0

        same_pick += (bsa_idx == 0)          # ref2015's own pick is top5[0]
        rows.append((float(rmsd5[0] <= HIT),
                     float(rmsd5[bsa_idx] <= HIT),
                     float(rmsd5[mm_idx] <= HIT),
                     float(rmsd5.min() <= HIT)))

    if not rows:
        print("no complexes with both caches")
        return
    a = np.array(rows)
    n = len(a)
    ref, bsa_hit, mm, orc = (100 * a[:, i].mean() for i in range(4))

    print(f"BSA+clash vs ref2015 as the stage-1 ranker — {n} gen_n100 complexes, "
          f"top-1 Hit@{HIT:g}A within ref2015's top {TOPK}\n")
    print(f"  ref2015                    {ref:5.1f}%")
    verdict = "EQUAL" if abs(bsa_hit - ref) < 1e-9 else "DIFFERS"
    print(f"  BSA+clash                  {bsa_hit:5.1f}%   <-- {verdict}")
    print(f"  MM-GBSA                    {mm:5.1f}%")
    print(f"  oracle within top-{TOPK}        {orc:5.1f}%   (ceiling of this stage)")
    print(f"\n  identical top-1 pick on {same_pick}/{n} complexes "
          f"({100 * same_pick / n:.0f}%)")
    if verdict == "EQUAL" and same_pick < n:
        print("  NOTE: same hit rate but not the same poses — the tie is a coincidence of "
              "this sample, not equivalence")
    if n_bsa_missing:
        print(f"  {n_bsa_missing} complexes skipped: no BSA cache entry")


if __name__ == "__main__":
    main()
