#!/usr/bin/env python3
"""
retrain_ranker_n100.py — Retrain RankerV2 encoder head on N=100 homogeneous data.

Problem with bench300 training: 4 models × 5 poses → encoder learned
inter-model discrimination, not pose quality. Collapses to τ≈0.01 on N=100.

Fix: train on gen_n100 data (100 poses, pretrained model only) with physics
features extracted by this script. Then evaluate properly held-out.

Steps:
  1. Extract physics (PyRosetta) for gen_n100 complexes that have encoder feats
  2. Build N=100 pool (encoder + physics, 100 homogeneous poses/complex)
  3. 5-fold held-out CV: train encoder head + burial axis on N=100 style data
  4. Report τ and top-k RMSD honestly

Usage:
    conda run -n score-env python3 scripts/retrain_ranker_n100.py [--skip-physics]
    # --skip-physics to skip step 1 if feats_gen_n100_physics.pkl already exists
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import tempfile
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats as sp

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

D    = REPO / "logs" / "diagnosis"
OUT  = REPO / "logs" / "training_campaign"
BASE = Path("/home/igem/unknown_software/datasets/training_formatted_peppc")

GEN_N100_JSON = REPO / "logs" / "gen_n100" / "benchmark_results.json"
GEN_N100_ENC  = D / "feats_gen_n100.pkl"
GEN_N100_PHYS = D / "feats_gen_n100_physics.pkl"

BURIAL_IDX  = [0, 2, 1]
BURIAL_SIGN = np.array([-1.0, 1.0, 1.0])
ENC_DIM     = 96
SCORE_TERMS = [
    "fa_atr", "fa_rep", "fa_sol", "fa_intra_rep", "fa_elec",
    "hbond_bb_sc", "hbond_sc", "hbond_lr_bb", "hbond_sr_bb",
    "rama_prepro", "fa_dun", "p_aa_pp",
]
EPOCHS = 50
SEEDS  = (0, 1, 2)
FOLDS  = 5


# ── helpers ──────────────────────────────────────────────────────────────────

def _per_cx_z(M): mu, sd = M.mean(0), M.std(0); return (M-mu)/np.where(sd<1e-9,1.,sd)
def _z(x): return (x-x.mean())/(x.std()+1e-9)


# ── Step 1: extract physics for gen_n100 ─────────────────────────────────────

def extract_physics():
    """Run PyRosetta on all gen_n100 poses that have encoder features."""
    import pyrosetta
    pyrosetta.init("-mute all -ex1 -ex2aro", silent=True)
    from pyrosetta.rosetta.core.scoring import get_score_function, ScoreType
    sfxn = get_score_function(True)

    bjson   = json.load(open(GEN_N100_JSON))
    enc_all = pickle.load(open(GEN_N100_ENC, "rb"))
    keys    = sorted(enc_all.keys())
    print(f"Extracting physics for {len(keys)} gen_n100 poses...", flush=True)

    rec_cache: dict[str, int] = {}
    results: dict = {}
    n_ok = n_fail = 0
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
            e_rec, n_rec = rec_cache[rec_pdb]

            e = pose.energies().total_energies()
            feats = []
            for term in SCORE_TERMS:
                try:    feats.append(float(e[getattr(ScoreType, term)]))
                except: feats.append(0.0)
            feats.append(float(total - e_rec))  # interface_ddG
            feats.append(float(total))           # total_score
            feats.append(0.0); feats.append(0.0) # resp_delta_e, resp_ca_disp (skip for speed)
            results[k] = np.array(feats, dtype=np.float32)
            n_ok += 1
        except Exception as exc:
            results[k] = None; n_fail += 1

        if (i+1) % 200 == 0:
            el = time.time()-t0
            print(f"  {i+1}/{len(keys)}  ok={n_ok} fail={n_fail}  "
                  f"{el/60:.1f}min  eta={(el/(i+1))*(len(keys)-i-1)/60:.1f}min", flush=True)

    print(f"Done. {n_ok} ok  {n_fail} fail  {(time.time()-t0)/60:.1f} min")
    pickle.dump(results, open(GEN_N100_PHYS, "wb"), protocol=4)
    print(f"Saved → {GEN_N100_PHYS}")


# ── Step 2: load N=100 pool ───────────────────────────────────────────────────

def load_n100_pool() -> dict:
    bjson   = json.load(open(GEN_N100_JSON))
    enc_all = pickle.load(open(GEN_N100_ENC,  "rb"))
    phys_all= pickle.load(open(GEN_N100_PHYS, "rb"))

    pool: dict = {}
    n_miss = 0
    for k, ev in enc_all.items():
        cn, mk, pi = k
        pv = phys_all.get(k)
        if pv is None: n_miss += 1; continue
        rr = bjson.get(cn, {}).get(mk, {}).get("ref_rmsds", [])
        if pi >= len(rr): n_miss += 1; continue
        pool.setdefault(cn, []).append({
            "phys": np.asarray(pv, np.float64),
            "enc":  np.asarray(ev,  np.float64),
            "rmsd": float(rr[pi]),
            "pi":   pi,
        })
    pool = {c: v for c, v in pool.items() if len(v) >= 10}
    print(f"N=100 pool: {len(pool)} complexes  ({n_miss} entries skipped)")
    return pool


# ── MLP head ─────────────────────────────────────────────────────────────────

def _lazy_torch():
    import torch, torch.nn as nn, torch.nn.functional as F
    return torch, nn, F

class Head:  # defined lazily in _train to avoid import at module load
    pass


def _train(pool, cxs, seed, epochs=EPOCHS):
    import torch, torch.nn as nn, torch.nn.functional as F
    torch.set_num_threads(4)

    class _Head(nn.Module):
        def __init__(self, d=ENC_DIM):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 64), nn.LayerNorm(64), nn.GELU(),
                nn.Dropout(0.2), nn.Linear(64, 1))
        def forward(self, x): return self.net(x)

    # subsample pairs per complex to cap at MAX_PAIRS_PER_CX for speed
    MAX_PAIRS_PER_CX = 1000
    rng_pairs = np.random.RandomState(seed)
    pairs = []
    for c in cxs:
        rows = pool[c]
        E  = _per_cx_z(np.array([r["enc"]  for r in rows]))
        rr = np.array([r["rmsd"] for r in rows])
        all_pairs = [(i, j) for i, j in combinations(range(len(rr)), 2)
                     if abs(rr[i]-rr[j]) > 1e-6]
        if len(all_pairs) > MAX_PAIRS_PER_CX:
            idxs = rng_pairs.choice(len(all_pairs), MAX_PAIRS_PER_CX, replace=False)
            all_pairs = [all_pairs[k] for k in idxs]
        for i, j in all_pairs:
            pairs.append((E[i], E[j], 1.0 if rr[i]<rr[j] else 0.0))
    torch.manual_seed(seed)
    h = _Head()
    opt = torch.optim.Adam(h.parameters(), lr=1e-3, weight_decay=1e-4)
    fi = torch.tensor(np.stack([p[0] for p in pairs]), dtype=torch.float32)
    fj = torch.tensor(np.stack([p[1] for p in pairs]), dtype=torch.float32)
    lb = torch.tensor([p[2] for p in pairs], dtype=torch.float32)
    n  = len(pairs)
    t0 = time.time()
    for ep in range(epochs):
        h.train()
        perm = torch.randperm(n)
        for b in range(0, n, 512):
            idx = perm[b:b+512]
            loss = -F.logsigmoid(
                (h(fi[idx]).squeeze(-1) - h(fj[idx]).squeeze(-1))
                * (lb[idx]*2-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % 10 == 0:
            print(f"    ep{ep+1}/{epochs}  loss={loss.item():.4f}  "
                  f"{(time.time()-t0):.1f}s", flush=True)
    return h.eval()


def _score(h, Farr):
    import torch
    with torch.no_grad():
        return h(torch.tensor(Farr, dtype=torch.float32)).squeeze(-1).numpy()


# ── Step 3: 5-fold CV on N=100 pool ──────────────────────────────────────────

def cv_n100(pool):
    cxs  = sorted(pool)
    rng  = np.random.RandomState(7)
    perm = rng.permutation(len(cxs))
    folds = [[cxs[i] for i in perm[f::FOLDS]] for f in range(FOLDS)]

    cx_results: dict = {}
    fold_taus = {"burial": [], "enc": [], "burial_enc": []}

    for fi in range(FOLDS):
        val_cxs = folds[fi]
        tr_cxs  = [c for c in cxs if c not in set(val_cxs)]
        print(f"\nFold {fi+1}/5  train={len(tr_cxs)}  val={len(val_cxs)}", flush=True)

        cx_seed: dict = {c: {} for c in val_cxs}
        st_ref, st_enc, st_benc = [], [], []

        for sd in SEEDS:
            head = _train(pool, tr_cxs, sd)

            for c in val_cxs:
                rows = pool[c]
                rmsd = np.array([r["rmsd"] for r in rows])
                N    = len(rows)

                phys_arr = np.array([r["phys"] for r in rows])
                enc_arr  = np.array([r["enc"]  for r in rows])
                burial   = _per_cx_z(phys_arr[:, BURIAL_IDX]) @ BURIAL_SIGN
                enc_s    = _score(head, _per_cx_z(enc_arr))
                benc_s   = _z(-burial) + _z(enc_s)

                # RAPiDock top-1
                p0 = next((r["rmsd"] for r in rows if r["pi"]==0), rmsd[0])
                # ref2015 total_score
                total = phys_arr[:, 13]
                ref_top1 = float(rmsd[np.argmin(total)])
                # burial only
                bur_top1 = float(rmsd[np.argmin(burial)])
                # burial+enc
                benc_top1 = float(rmsd[np.argmax(benc_s)])
                # oracle
                oracle = float(rmsd.min())
                # top-k
                sorted_by_benc = np.argsort(-benc_s)
                sorted_by_ref  = np.argsort(total)
                for tk in [1,5,10,25]:
                    cx_seed[c].setdefault(f"benc_top{tk}", []).append(float(rmsd[sorted_by_benc[:tk]].min()))
                    cx_seed[c].setdefault(f"ref_top{tk}",  []).append(float(rmsd[sorted_by_ref[:tk]].min()))
                cx_seed[c].setdefault("rapd",    []).append(float(p0))
                cx_seed[c].setdefault("oracle",  []).append(oracle)

                for arr, key in [(burial, "burial"), (enc_s, "enc"), (benc_s, "benc")]:
                    t, _ = sp.kendalltau(-arr if key != "burial" else arr, rmsd)
                    if not math.isnan(t):
                        (st_ref if key=="burial" else st_enc if key=="enc" else st_benc).append(t)

        for c in val_cxs:
            cx_results[c] = {k: float(np.mean(vs)) for k, vs in cx_seed[c].items()}

        fold_taus["burial"].append(float(np.mean(st_ref)))
        fold_taus["enc"].append(float(np.mean(st_enc)))
        fold_taus["burial_enc"].append(float(np.mean(st_benc)))
        print(f"  burial τ={fold_taus['burial'][-1]:+.4f}  enc τ={fold_taus['enc'][-1]:+.4f}  "
              f"burial+enc τ={fold_taus['burial_enc'][-1]:+.4f}")

    return cx_results, fold_taus


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-physics", action="store_true")
    a = ap.parse_args()

    if not a.skip_physics:
        if GEN_N100_PHYS.exists():
            print(f"Physics pkl already exists ({GEN_N100_PHYS}). Use --skip-physics to reuse.")
        else:
            extract_physics()
    else:
        print("Skipping physics extraction (--skip-physics)")

    print("\nLoading N=100 pool...")
    pool = load_n100_pool()
    if len(pool) < 5:
        print("ERROR: not enough complexes with both physics + encoder features.")
        return

    print("\nRunning 5-fold CV on N=100 pool...")
    cx_results, fold_taus = cv_n100(pool)

    # ── summary ──────────────────────────────────────────────────────────────
    N = len(cx_results)
    print(f"\n{'='*70}")
    print(f"N=100 HELD-OUT EVALUATION  ({N} complexes, 5-fold CV, {len(SEEDS)} seeds)")
    print(f"{'='*70}")

    print(f"\nKendall τ (mean ± std):")
    for s, tv in fold_taus.items():
        print(f"  {s:<15} τ = {np.mean(tv):+.4f} ± {np.std(tv):.4f}")

    print(f"\nTop-k RMSD & Hit@2Å (mean over {N} held-out complexes):")
    print(f"  {'Metric':<20} {'ref2015':>10} {'burial+enc':>12} {'oracle':>10}")
    print(f"  {'-'*56}")

    for tk in [1, 5, 10, 25]:
        ref_vals  = np.array([cx_results[c][f"ref_top{tk}"]  for c in cx_results])
        benc_vals = np.array([cx_results[c][f"benc_top{tk}"] for c in cx_results])
        orc_vals  = np.array([cx_results[c]["oracle"] for c in cx_results])
        rapd_vals = np.array([cx_results[c]["rapd"]   for c in cx_results])

        ref_m  = ref_vals.mean();  ref_h  = 100*np.mean(ref_vals<=2.0)
        benc_m = benc_vals.mean(); benc_h = 100*np.mean(benc_vals<=2.0)
        orc_m  = orc_vals.mean();  orc_h  = 100*np.mean(orc_vals<=2.0)
        rapd_m = rapd_vals.mean(); rapd_h = 100*np.mean(rapd_vals<=2.0)

        if tk == 1:
            print(f"  RAPiDock top-1           {rapd_m:>8.2f}Å           {'—':>10}   {orc_m:>8.2f}Å")
            print(f"    Hit@2Å                 {rapd_h:>8.1f}%           {'—':>10}   {orc_h:>8.1f}%")
        print(f"  Mean RMSD  top-{tk:<2}       {ref_m:>8.2f}Å   {benc_m:>8.2f}Å   {orc_m:>8.2f}Å")
        print(f"  Hit@2Å     top-{tk:<2}       {ref_h:>8.1f}%   {benc_h:>8.1f}%   {orc_h:>8.1f}%")
        print()

    results = {c: cx_results[c] for c in cx_results}
    results["_meta"] = {
        "n_complexes": N, "folds": FOLDS, "seeds": list(SEEDS), "epochs": EPOCHS,
        "fold_taus": {k: list(v) for k, v in fold_taus.items()},
        "note": "N=100 homogeneous retraining — encoder trained on same distribution as eval",
    }
    (OUT / "n100_ranker_cv.json").write_text(json.dumps(results, indent=2))
    print(f"Saved → {OUT}/n100_ranker_cv.json")


if __name__ == "__main__":
    main()
