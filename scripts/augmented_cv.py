#!/usr/bin/env python3
"""
augmented_cv.py — Does adding NEW complexes (gen_n100) improve held-out ranking?

Design (the honest test of the complex-diversity hypothesis):
  - Evaluation anchor = bench300 (115 cx, N=20), the SAME held-out set behind the
    τ=0.212 baseline. Never contaminated.
  - 5-fold over bench complexes. Each fold:
      bench_only : train = 4/5 bench         test = 1/5 bench
      bench+gen  : train = 4/5 bench + ALL gen_n100   test = 1/5 bench
    If bench+gen > bench_only on held-out bench, new complexes generalize.

Pair balancing (critical): gen complexes are N=100 → C(100,2)=4950 pairs vs N=20
bench → 190. Left unchecked, 60 gen complexes contribute 26× the pairs and swamp
bench. We CAP pairs per complex to K (random subsample), so every complex
contributes comparably regardless of pose count — while still drawing those K
pairs from the full N=100 pose set (better RMSD coverage).

Feature config = production winner combined14_CLIP:
  encoder(96) ++ winsorized static ref2015(14), clip bounds fit on TRAIN physics.

Run (rapidock env has torch):
  PYTHONPATH=$(pwd) ~/miniconda3/envs/rapidock/bin/python scripts/augmented_cv.py
"""
from __future__ import annotations

import json
import logging
import pickle
import random
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sp

import scripts.cv_physics_head as cv   # reuse Head, train_fold, bpr, EPOCHS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", force=True)
log = logging.getLogger("augcv")

REPO = Path(__file__).resolve().parent.parent
D = REPO / "logs" / "diagnosis"
BENCH_ENC  = D / "feats_bench300.pkl"
BENCH_PHYS = D / "feats_bench300_physics.pkl"
BENCH_JSON = REPO / "logs" / "analysis_bench300" / "benchmark_results.json"
GEN_ENC    = D / "feats_gen_n100.pkl"
GEN_PHYS   = D / "feats_gen_n100_physics.pkl"
GEN_JSON   = REPO / "logs" / "gen_n100" / "benchmark_results.json"

N_STATIC = 14
N_FOLDS  = 5
K_PAIRS  = 190          # max pairs per complex (= bench N=20 budget)
SEED     = 42


def load_pool(enc_pkl, phys_pkl, json_path):
    """cname -> {enc:(n,96), phys:(n,16), rmsd:(n,)} from a feature pool."""
    enc = pickle.load(open(enc_pkl, "rb"))
    phys = pickle.load(open(phys_pkl, "rb"))
    jd = json.load(open(json_path))
    keys = set(enc) & set(phys)
    tmp: dict = {}
    for k in keys:
        cn, mk, pi = k
        rmsds = jd.get(cn, {}).get(mk, {}).get("ref_rmsds", [])
        if pi >= len(rmsds):
            continue
        tmp.setdefault(cn, []).append((np.asarray(enc[k], np.float32),
                                       np.asarray(phys[k], np.float32), float(rmsds[pi])))
    out = {}
    for c, rows in tmp.items():
        if len(rows) < 2:
            continue
        out[c] = {"enc": np.stack([r[0] for r in rows]),
                  "phys": np.stack([r[1] for r in rows]),
                  "rmsd": np.asarray([r[2] for r in rows], np.float32)}
    return out


def fit_clip(pool, complexes):
    M = np.concatenate([pool[c]["phys"][:, :N_STATIC] for c in complexes], 0)
    return np.percentile(M, 1, 0), np.percentile(M, 99, 0)


def feats(d, lo, hi):
    return np.concatenate([d["enc"], np.clip(d["phys"][:, :N_STATIC], lo, hi)], 1)


def make_pairs(pool, complexes, lo, hi, rng):
    pairs = []
    for c in complexes:
        d = pool[c]; F = feats(d, lo, hi); r = d["rmsd"]
        cps = [(i, j) for i, j in combinations(range(len(r)), 2) if abs(r[i] - r[j]) >= 1e-6]
        if len(cps) > K_PAIRS:
            cps = rng.sample(cps, K_PAIRS)
        for i, j in cps:
            pairs.append((F[i], F[j], 1.0 if r[i] < r[j] else 0.0))
    return pairs


def eval_tau(head, pool, complexes, lo, hi):
    head.eval(); ts = []
    with torch.no_grad():
        for c in complexes:
            d = pool[c]; F = feats(d, lo, hi)
            s = head(torch.tensor(F, dtype=torch.float32)).squeeze(-1).numpy()
            t, _ = sp.kendalltau(-s, d["rmsd"])
            if not np.isnan(t):
                ts.append(t)
    return float(np.mean(ts)) if ts else float("nan")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-enc",  default=str(BENCH_ENC))
    ap.add_argument("--bench-phys", default=str(BENCH_PHYS))
    ap.add_argument("--gen-enc",    default=str(GEN_ENC))
    ap.add_argument("--gen-phys",   default=str(GEN_PHYS))
    ap.add_argument("--gen-json",   default=str(GEN_JSON))
    a = ap.parse_args()
    from pathlib import Path as _P
    benc, bphys = _P(a.bench_enc), _P(a.bench_phys)
    genc, gphys, gjson = _P(a.gen_enc), _P(a.gen_phys), _P(a.gen_json)

    bench = load_pool(benc, bphys, BENCH_JSON)
    has_gen = genc.exists() and gphys.exists() and gjson.exists()
    gen = load_pool(genc, gphys, gjson) if has_gen else {}
    log.info("bench complexes=%d  gen complexes=%d  (gen ready=%s)",
             len(bench), len(gen), has_gen)
    if not has_gen:
        log.warning("gen features not found yet — run after generation+extraction.")
        log.warning("  need: %s, %s, %s", genc, gphys, gjson)
        return

    bcx = sorted(bench)
    rng = np.random.RandomState(SEED); perm = rng.permutation(len(bcx))
    folds = [[bcx[i] for i in perm[f::N_FOLDS]] for f in range(N_FOLDS)]

    for cond in ("bench_only", "bench+gen"):
        fold_taus = []
        for f in range(N_FOLDS):
            val = folds[f]
            bench_train = [c for c in bcx if c not in set(val)]
            # combined pool view
            train_pool = dict(bench)
            train_cx = list(bench_train)
            if cond == "bench+gen":
                train_pool = {**bench, **gen}
                train_cx = bench_train + sorted(gen)
            lo, hi = fit_clip(train_pool, train_cx)
            pr = random.Random(SEED + f)
            tr = make_pairs(train_pool, train_cx, lo, hi, pr)
            in_dim = 96 + N_STATIC
            head = cv.train_fold(tr, in_dim, seed=SEED + f)
            fold_taus.append(eval_tau(head, bench, val, lo, hi))   # eval on held-out BENCH
        m, s = float(np.mean(fold_taus)), float(np.std(fold_taus))
        log.info("  %-12s τ=%.4f ± %.4f  folds=%s  (train_cx≈%d)",
                 cond, m, s, [round(t, 3) for t in fold_taus], len(train_cx))

    log.info("Baseline (no-cap combined14_CLIP, prior run): τ≈0.212")


if __name__ == "__main__":
    main()
