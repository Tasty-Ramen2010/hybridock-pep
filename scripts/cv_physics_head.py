#!/usr/bin/env python3
"""
cv_physics_head.py — Honest k-fold cross-validation of the confidence head over
complexes, plus feature-transform experiments to push τ past the single-split
P3_combined=0.2294 result.

Why this exists:
  train_physics_head.py reports a SINGLE 85/15 split AND selects the best epoch on
  that same val fold (optimistic). This script does proper K-fold CV over complexes
  with NO per-epoch test peeking (fixed epochs, evaluate final model), so condition
  comparisons are fair and the τ numbers are trustworthy.

Conditions:
  encoder_only        96-dim encoder (baseline)
  phys14              14-dim static ref2015 (no response)
  phys14_pcz          14-dim static, per-complex z-scored
  combined14          encoder ++ phys14            ← the candidate
  combined16          encoder ++ phys16 (response) ← confirm response hurts under CV
  combined14_slog     encoder ++ signed-log(phys14)
  combined14_pcz      encoder ++ per-complex-z(phys14)

Feature transforms (physics only; encoder left raw):
  slog : sign(x)*log1p(|x|)          tames the clash-dominated heavy tails
  pcz  : within each complex, (x-mean)/std per dim — pure within-complex contrast,
         which is all a within-complex ranking objective can use.

Run (rapidock env has torch; score-env does not):
  PYTHONPATH=$(pwd) ~/miniconda3/envs/rapidock/bin/python scripts/cv_physics_head.py
"""
from __future__ import annotations

import json
import logging
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sp

torch.set_num_threads(8)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", force=True)
log = logging.getLogger("cv")

REPO       = Path(__file__).resolve().parent.parent
FEAT_BENCH = REPO / "logs" / "diagnosis" / "feats_bench300.pkl"          # 96-dim encoder
PHYS_BENCH = REPO / "logs" / "diagnosis" / "feats_bench300_physics.pkl"  # 16-dim physics
BENCH_JSON = REPO / "logs" / "analysis_bench300" / "benchmark_results.json"
N_STATIC   = 14
N_FOLDS    = 5
EPOCHS     = 120
SEED       = 42


class Head(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        hidden = max(32, in_dim * 2)
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


def bpr(si, sj, label):
    return -F.logsigmoid((si - sj) * (label * 2.0 - 1.0)).mean()


def load_data() -> dict[str, dict]:
    """cname -> {enc:(n,96), phys:(n,16), rmsd:(n,)}."""
    enc = pickle.load(open(FEAT_BENCH, "rb"))
    phys = pickle.load(open(PHYS_BENCH, "rb"))
    bench = json.load(open(BENCH_JSON))
    keys = set(enc) & set(phys)
    tmp: dict[str, list] = {}
    for k in keys:
        cname, mkey, pi = k
        rmsds = bench.get(cname, {}).get(mkey, {}).get("ref_rmsds", [])
        if pi >= len(rmsds):
            continue
        tmp.setdefault(cname, []).append(
            (np.asarray(enc[k], np.float32), np.asarray(phys[k], np.float32), float(rmsds[pi]))
        )
    out = {}
    for c, rows in tmp.items():
        if len(rows) < 2:
            continue
        out[c] = {
            "enc":  np.stack([r[0] for r in rows]),
            "phys": np.stack([r[1] for r in rows]),
            "rmsd": np.asarray([r[2] for r in rows], np.float32),
        }
    return out


def _slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _pcz(x):
    """Per-complex z-score: (x - mean)/std over the complex's poses, per dim."""
    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True) + 1e-6
    return (x - mu) / sd


def build_feats(d: dict, condition: str) -> np.ndarray:
    enc = d["enc"]
    ph = d["phys"]
    ph14 = ph[:, :N_STATIC]
    if condition == "encoder_only":   return enc
    if condition == "phys14":         return ph14
    if condition == "phys14_pcz":     return _pcz(ph14)
    if condition == "combined14":     return np.concatenate([enc, ph14], 1)
    if condition == "combined16":     return np.concatenate([enc, ph], 1)
    if condition == "combined14_slog":return np.concatenate([enc, _slog(ph14)], 1)
    if condition == "combined14_pcz": return np.concatenate([enc, _pcz(ph14)], 1)
    raise ValueError(condition)


def make_pairs(data, complexes, condition):
    pairs = []
    for c in complexes:
        feats = build_feats(data[c], condition)
        rmsd = data[c]["rmsd"]
        for i, j in combinations(range(len(rmsd)), 2):
            if abs(rmsd[i] - rmsd[j]) < 1e-6:
                continue
            pairs.append((feats[i], feats[j], 1.0 if rmsd[i] < rmsd[j] else 0.0))
    return pairs


def eval_tau(head, data, complexes, condition):
    head.eval()
    taus = []
    with torch.no_grad():
        for c in complexes:
            feats = build_feats(data[c], condition)
            rmsd = data[c]["rmsd"]
            if len(rmsd) < 2:
                continue
            s = head(torch.tensor(feats, dtype=torch.float32)).squeeze(-1).numpy()
            tau, _ = sp.kendalltau(-s, rmsd)
            if not np.isnan(tau):
                taus.append(tau)
    return float(np.mean(taus)) if taus else float("nan")


def train_fold(train_pairs, in_dim, seed):
    torch.manual_seed(seed)
    head = Head(in_dim)
    fi = torch.tensor(np.stack([p[0] for p in train_pairs]), dtype=torch.float32)
    fj = torch.tensor(np.stack([p[1] for p in train_pairs]), dtype=torch.float32)
    lbl = torch.tensor([p[2] for p in train_pairs], dtype=torch.float32)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    n = len(train_pairs)
    for _ in range(EPOCHS):
        head.train()
        perm = torch.randperm(n)
        for b in range(0, n, 512):
            idx = perm[b:b + 512]
            loss = bpr(head(fi[idx]).squeeze(-1), head(fj[idx]).squeeze(-1), lbl[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return head


def main():
    data = load_data()
    cx = sorted(data)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(cx))
    folds = [[cx[i] for i in perm[f::N_FOLDS]] for f in range(N_FOLDS)]
    log.info("Loaded %d complexes, %d poses; %d-fold CV",
             len(cx), sum(len(d["rmsd"]) for d in data.values()), N_FOLDS)

    conditions = ["encoder_only", "phys14", "phys14_pcz",
                  "combined14", "combined16", "combined14_slog", "combined14_pcz"]
    results = {}
    for cond in conditions:
        fold_taus = []
        for f in range(N_FOLDS):
            val_c = folds[f]
            train_c = [c for c in cx if c not in set(val_c)]
            tr = make_pairs(data, train_c, cond)
            in_dim = build_feats(data[train_c[0]], cond).shape[1]
            head = train_fold(tr, in_dim, seed=SEED + f)
            fold_taus.append(eval_tau(head, data, val_c, cond))
        mean, std = float(np.mean(fold_taus)), float(np.std(fold_taus))
        results[cond] = (mean, std, in_dim)
        log.info("  %-18s τ=%.4f ± %.4f  (dim=%d)  folds=%s",
                 cond, mean, std, in_dim, [round(t, 3) for t in fold_taus])

    log.info("=" * 60)
    log.info("CROSS-VALIDATED RESULTS (%d-fold, no test-peeking)", N_FOLDS)
    log.info("=" * 60)
    for cond, (m, s, d) in sorted(results.items(), key=lambda x: -x[1][0]):
        log.info("  %-18s  τ=%.4f ± %.4f  dim=%d", cond, m, s, d)
    best = max(results, key=lambda c: results[c][0])
    log.info("Best: %s  τ=%.4f", best, results[best][0])
    enc = results["encoder_only"][0]
    log.info("Δ over encoder_only: %s = %+.4f",
             best, results[best][0] - enc)


if __name__ == "__main__":
    main()
