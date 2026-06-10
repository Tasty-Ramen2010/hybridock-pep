#!/usr/bin/env python3
"""
mmgbsa_rerank_top5.py — Does MM-GBSA rerank the top-5 better than ref2015?

The enrichment curve says ref2015 top-5 CONTAINS a near-native 21% of the time
(vs 10.5% at top-1). The two-stage filter only pays off if a stronger 2nd-stage
scorer can PICK that near-native out of the 5. MM-GBSA (AMBER ff14SB + GBn2,
with minimization) is the candidate — it relieves clashes and models polar
desolvation better than ref2015's implicit terms.

For each of 57 gen_n100 complexes:
  1. stage-1: top-5 poses by ref2015 total_score
  2. run MM-GBSA ΔG_bind on each (CPU OpenMM, won't touch the GPU gen run)
  3. compare top-1 Hit@2Å of: ref2015 (within-5) vs MM-GBSA (within-5) vs
     oracle-within-5 (= top-5 coverage, the ceiling of this stage)

Caveat: single-trajectory MM-GBSA does NOT include configurational entropy
(REF2015 blind spot #1 is only partly addressed). It DOES help with clash
relief and polar desolvation. Resume-safe (caches ΔG per pose).

Run in score-env (OpenMM+pdbfixer): python3 scripts/mmgbsa_rerank_top5.py
"""
from __future__ import annotations

import json, pickle, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

GEN_JSON = REPO / "logs" / "gen_n100" / "benchmark_results.json"
ENC_PKL  = REPO / "logs" / "diagnosis" / "feats_gen_n100.pkl"
PHYS_PKL = REPO / "logs" / "diagnosis" / "feats_gen_n100_physics.pkl"
BASE = Path("/home/igem/unknown_software/datasets/training_formatted_peppc")
DG_CACHE = REPO / "logs" / "diagnosis" / "mmgbsa_top5_dg.pkl"
OUT = REPO / "logs" / "training_campaign" / "mmgbsa_rerank_top5.json"

TOPK = 5


def main():
    from hybridock_pep.scoring.mmgbsa import compute_mmgbsa_single

    bjson = json.load(open(GEN_JSON))
    phys = pickle.load(open(PHYS_PKL, "rb"))
    cxs = sorted(set(k[0] for k in pickle.load(open(ENC_PKL, "rb"))))

    dg_cache = pickle.load(open(DG_CACHE, "rb")) if DG_CACHE.exists() else {}
    n_new = 0
    t0 = time.time()

    per_cx = []  # (cn, ref_top1_hit, mmgbsa_top1_hit, oracle5_hit, n_ok)
    for ci, cn in enumerate(cxs):
        entry = bjson.get(cn, {}).get("pretrained", {})
        rr = entry.get("ref_rmsds", [])
        pdir = Path(entry.get("poses_dir", ""))
        rec_pdb = BASE / cn / f"{cn}_protein_pocket.pdb"
        if not rec_pdb.exists() or len(rr) < 5:
            continue
        # stage-1: top-5 by ref2015 total_score
        refs = np.array([phys.get((cn, "pretrained", pi), [0]*14)[13]
                         for pi in range(len(rr))], float)
        order = np.argsort(refs)
        top5 = [int(i) for i in order[:TOPK]]

        rmsd5 = np.array([rr[pi] for pi in top5])
        dgs = []
        for pi in top5:
            key = (cn, pi)
            if key in dg_cache:
                dgs.append(dg_cache[key]); continue
            pose_pdb = pdir / f"pose_{pi}.pdb"
            try:
                dg = compute_mmgbsa_single(pose_pdb, rec_pdb, force_cpu=True)
            except Exception as exc:
                dg = None
                print(f"    {cn} pose_{pi} MM-GBSA failed: {str(exc)[:80]}", flush=True)
            dg_cache[key] = dg; dgs.append(dg); n_new += 1
            if n_new % 10 == 0:
                pickle.dump(dg_cache, open(DG_CACHE, "wb"), protocol=4)

        dgs = np.array([d if d is not None else np.inf for d in dgs], float)
        n_ok = int(np.isfinite(dgs).sum())
        if n_ok < 2:
            continue

        # ref2015 within-top5 #1 = the global ref2015 top-1 (top5[0])
        ref_hit = float(rmsd5[0] <= 2.0)
        # MM-GBSA #1 of the 5 (lowest ΔG)
        mm_idx = int(np.argmin(dgs))
        mm_hit = float(rmsd5[mm_idx] <= 2.0)
        oracle5 = float(rmsd5.min() <= 2.0)
        per_cx.append((cn, ref_hit, mm_hit, oracle5, n_ok))

        if (ci + 1) % 5 == 0:
            arr = np.array([(r[1], r[2], r[3]) for r in per_cx])
            print(f"  [{ci+1}/{len(cxs)}] {len(per_cx)} cx scored  "
                  f"{(time.time()-t0)/60:.0f}min  "
                  f"ref={100*arr[:,0].mean():.1f}% mmgbsa={100*arr[:,1].mean():.1f}% "
                  f"oracle5={100*arr[:,2].mean():.1f}%", flush=True)

    pickle.dump(dg_cache, open(DG_CACHE, "wb"), protocol=4)
    arr = np.array([(r[1], r[2], r[3]) for r in per_cx])
    n = len(per_cx)
    print(f"\n{'='*60}")
    print(f"MM-GBSA RERANK of ref2015 top-{TOPK}  ({n} complexes)")
    print(f"{'='*60}")
    print(f"  ref2015 top-1 Hit@2Å         = {100*arr[:,0].mean():.1f}%")
    print(f"  MM-GBSA-within-top5 top-1    = {100*arr[:,1].mean():.1f}%")
    print(f"  oracle-within-top5 (ceiling) = {100*arr[:,2].mean():.1f}%")
    delta = 100*(arr[:,1].mean() - arr[:,0].mean())
    print(f"\n  Δ MM-GBSA vs ref2015 = {delta:+.1f} pts")
    if delta > 3:
        print(f"  → MM-GBSA reranks top-5 BETTER. Two-stage filter delivers.")
    elif delta < -3:
        print(f"  → MM-GBSA WORSE than ref2015 even within top-5.")
    else:
        print(f"  → MM-GBSA ≈ ref2015 within top-5 (no rerank benefit).")

    OUT.write_text(json.dumps({
        "n": n, "ref_top1": float(arr[:,0].mean()),
        "mmgbsa_top1": float(arr[:,1].mean()),
        "oracle5": float(arr[:,2].mean()),
        "per_cx": [{"cn": r[0], "ref": r[1], "mmgbsa": r[2], "oracle5": r[3]}
                   for r in per_cx],
    }, indent=2))
    print(f"\n  Saved → {OUT}")


if __name__ == "__main__":
    main()
