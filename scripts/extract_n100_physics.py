#!/usr/bin/env python3
"""
extract_n100_physics.py — Extract ref2015 physics features for gen_n100 poses.

Runs in score-env (PyRosetta). Saves feats_gen_n100_physics.pkl.
Run this first, then retrain_ranker_n100.py --skip-physics in rapidock.
"""
from __future__ import annotations

import json, os, pickle, sys, tempfile, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BASE = Path("/home/igem/unknown_software/datasets/training_formatted_peppc")

GEN_N100_JSON = REPO / "logs" / "gen_n100" / "benchmark_results.json"
GEN_N100_ENC  = REPO / "logs" / "diagnosis" / "feats_gen_n100.pkl"
OUT_PKL       = REPO / "logs" / "diagnosis" / "feats_gen_n100_physics.pkl"

SCORE_TERMS = [
    "fa_atr", "fa_rep", "fa_sol", "fa_intra_rep", "fa_elec",
    "hbond_bb_sc", "hbond_sc", "hbond_lr_bb", "hbond_sr_bb",
    "rama_prepro", "fa_dun", "p_aa_pp",
]


def main():
    import pyrosetta
    pyrosetta.init("-mute all -ex1 -ex2aro", silent=True)
    from pyrosetta.rosetta.core.scoring import get_score_function, ScoreType
    sfxn = get_score_function(True)

    bjson   = json.load(open(GEN_N100_JSON))
    enc_all = pickle.load(open(GEN_N100_ENC, "rb"))
    keys    = sorted(enc_all.keys())
    print(f"Extracting physics for {len(keys)} gen_n100 poses...", flush=True)

    rec_cache: dict = {}
    results: dict   = {}
    n_ok = n_fail   = 0
    t0 = time.time()

    for i, k in enumerate(keys):
        cn, mk, pi = k
        entry    = bjson.get(cn, {}).get(mk, {})
        pose_pdb = str(Path(entry.get("poses_dir", "")) / f"pose_{pi}.pdb")
        rec_pdb  = str(BASE / cn / f"{cn}_protein_pocket.pdb")

        if not Path(pose_pdb).exists() or not Path(rec_pdb).exists():
            results[k] = None; n_fail += 1; continue

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as tmp:
                tmp.write(open(rec_pdb).read().rstrip())
                tmp.write("\nTER\n")
                tmp.write(open(pose_pdb).read())
                tmp_path = tmp.name
            pose = pyrosetta.pose_from_pdb(tmp_path)
            os.unlink(tmp_path)
            total = sfxn(pose)

            if rec_pdb not in rec_cache:
                rp = pyrosetta.pose_from_pdb(rec_pdb)
                rec_cache[rec_pdb] = (sfxn(rp), rp.total_residue())
            e_rec, _ = rec_cache[rec_pdb]

            e = pose.energies().total_energies()
            feats = []
            for term in SCORE_TERMS:
                try:    feats.append(float(e[getattr(ScoreType, term)]))
                except: feats.append(0.0)
            feats.append(float(total - e_rec))  # interface_ddG  [idx 12]
            feats.append(float(total))           # total_score    [idx 13]
            feats.append(0.0); feats.append(0.0) # resp (skip for speed) [idx 14,15]
            results[k] = np.array(feats, dtype=np.float32)
            n_ok += 1
        except Exception as exc:
            results[k] = None; n_fail += 1

        if (i+1) % 200 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(keys)}  ok={n_ok} fail={n_fail}  "
                  f"{el/60:.1f}min  eta={(el/(i+1))*(len(keys)-i-1)/60:.1f}min", flush=True)

    print(f"\nDone. {n_ok} ok  {n_fail} fail  {(time.time()-t0)/60:.1f}min")
    pickle.dump(results, open(OUT_PKL, "wb"), protocol=4)
    print(f"Saved → {OUT_PKL}")


if __name__ == "__main__":
    main()
