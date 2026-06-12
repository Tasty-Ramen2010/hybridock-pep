#!/usr/bin/env python3
"""
e0_extended.py — Train E0 head on bench300 + gen_ood combined features.

Tests whether adding 473 gen complexes to the 240 bench complexes
improves the E0 confidence head's ranking τ on the bench300 val set.

Conditions:
  B      bench300 only (238 train cx)        ← control, known τ≈0.28
  B+G50  bench300 + 50% gen (238+236 cx)
  B+G    bench300 + all gen (238+473 cx)

Usage:
  python3 scripts/e0_extended.py  (score-env, CPU-only)
"""
from __future__ import annotations

import copy
import logging
import math
import os
import pickle
import sys
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(8)   # prevent thread explosion on large servers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", force=True)
log = logging.getLogger("e0_extended")

REPO       = Path(__file__).resolve().parent.parent
FEAT_BENCH = REPO / "logs" / "diagnosis" / "feats_bench300.pkl"
FEAT_GEN   = REPO / "logs" / "diagnosis" / "feats_gen_ood.pkl"
BENCH_JSON = REPO / "logs" / "analysis_bench300" / "benchmark_results.json"
GEN_JSON   = REPO / "logs" / "confidence_training_data" / "benchmark_results.json"
OUT        = REPO / "logs" / "training_campaign"


class V2Head(nn.Module):
    def __init__(self, in_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64),     nn.GELU(),           nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x): return self.net(x)


def bpr_loss(si, sj, label):
    return -F.logsigmoid((si - sj) * (label * 2.0 - 1.0)).mean()


def split_complexes(complexes, train_frac=0.85, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(complexes))
    n   = max(1, int(len(complexes) * train_frac))
    return [complexes[i] for i in idx[:n]], [complexes[i] for i in idx[n:]]


def build_dataset(feat_map, json_data):
    ds = {}
    for (cname, mkey, pose_idx), feat in feat_map.items():
        rmsds = json_data.get(cname, {}).get(mkey, {}).get("ref_rmsds", [])
        if pose_idx >= len(rmsds): continue
        ds.setdefault(cname, []).append((feat.astype(np.float32), float(rmsds[pose_idx])))
    return {k: v for k, v in ds.items() if len(v) >= 2}


def build_pairs(ds, complexes):
    pairs = []
    for c in complexes:
        for (fi, ri), (fj, rj) in combinations(ds.get(c, []), 2):
            if abs(ri - rj) < 1e-6: continue
            pairs.append((fi, fj, 1.0 if ri < rj else 0.0))
    return pairs


def train_and_eval(label, train_pairs, val_pairs, val_ds, val_c,
                   epochs=50, lr=1e-3, seed=0):
    from scipy import stats as sp

    torch.manual_seed(seed)
    head = V2Head()
    for m in head.modules():
        if isinstance(m, (nn.Linear, nn.LayerNorm)): m.reset_parameters()

    fi  = torch.tensor(np.stack([p[0] for p in train_pairs]), dtype=torch.float32)
    fj  = torch.tensor(np.stack([p[1] for p in train_pairs]), dtype=torch.float32)
    lbl = torch.tensor([p[2] for p in train_pairs],           dtype=torch.float32)
    vfi = torch.tensor(np.stack([p[0] for p in val_pairs]),   dtype=torch.float32)
    vfj = torch.tensor(np.stack([p[1] for p in val_pairs]),   dtype=torch.float32)

    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    best_acc, best_state = -1.0, None
    n = len(train_pairs)

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n)
        for b in range(0, n, 512):
            idx = perm[b: b+512]
            loss = bpr_loss(head(fi[idx]).squeeze(-1),
                            head(fj[idx]).squeeze(-1), lbl[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            acc = ((head(vfi).squeeze(-1) > head(vfj).squeeze(-1))
                   .float()).mean().item()
        if acc > best_acc:
            best_acc = acc; best_state = copy.deepcopy(head.state_dict())

    head.load_state_dict(best_state); head.eval()

    # Compute Kendall τ on val complexes
    taus, tops = [], []
    with torch.no_grad():
        for c in val_c:
            poses = val_ds.get(c, [])
            if len(poses) < 2: continue
            feats  = torch.tensor(np.array([p[0] for p in poses], dtype=np.float32))
            rmsds  = np.array([p[1] for p in poses])
            scores = head(feats).squeeze(-1).numpy()
            tau, _ = sp.kendalltau(-scores, rmsds)
            if not math.isnan(tau):
                taus.append(tau); tops.append(float(rmsds[np.argmax(scores)]))

    tau_mean = float(np.mean(taus)) if taus else float("nan")
    top1     = float(np.mean(tops)) if tops else float("nan")
    log.info("  %-20s  τ=%.4f  top1=%.3f  train_pairs=%d  val_cx=%d",
             label, tau_mean, top1, len(train_pairs), len(taus))
    return tau_mean, top1, head


def main():
    log.info("Loading features...")
    with open(FEAT_BENCH, "rb") as f: bench_feats = pickle.load(f)
    with open(FEAT_GEN,   "rb") as f: gen_feats   = pickle.load(f)
    bench_json = json.load(open(BENCH_JSON))
    gen_json   = json.load(open(GEN_JSON))

    bench_ds = build_dataset(bench_feats, bench_json)
    gen_ds   = build_dataset(gen_feats,   gen_json)
    log.info("bench300: %d complexes  gen_ood: %d complexes", len(bench_ds), len(gen_ds))

    bench_all = sorted(bench_ds.keys())
    bench_train_c, bench_val_c = split_complexes(bench_all, 0.85, seed=42)

    gen_all = sorted(gen_ds.keys())
    gen_train_c, _ = split_complexes(gen_all, 0.85, seed=42)
    rng = np.random.RandomState(42)
    gen_50 = list(rng.choice(gen_train_c, len(gen_train_c)//2, replace=False))

    # val is always bench300 val (same set for fair comparison)
    val_pairs = build_pairs(bench_ds, bench_val_c)
    log.info("Val: %d complexes, %d pairs", len(bench_val_c), len(val_pairs))

    results = []

    # Condition B: bench300 only
    log.info("\n=== B: bench300 only ===")
    tr_pairs = build_pairs(bench_ds, bench_train_c)
    tau, top1, _ = train_and_eval("B_bench_only", tr_pairs, val_pairs,
                                  bench_ds, bench_val_c)
    results.append({"label": "B_bench_only", "tau": tau, "top1": top1,
                    "n_train_cx": len(bench_train_c), "n_train_pairs": len(tr_pairs)})

    # Condition B+G50: bench + 50% gen
    log.info("\n=== B+G50: bench + 50%% gen ===")
    combined_ds_50 = {**bench_ds, **{c: gen_ds[c] for c in gen_50}}
    tr_pairs_50 = build_pairs(combined_ds_50, bench_train_c + gen_50)
    tau, top1, _ = train_and_eval("B+G50", tr_pairs_50, val_pairs,
                                  bench_ds, bench_val_c)
    results.append({"label": "B+G50", "tau": tau, "top1": top1,
                    "n_train_cx": len(bench_train_c)+len(gen_50),
                    "n_train_pairs": len(tr_pairs_50)})

    # Condition B+G: bench + all gen
    log.info("\n=== B+G: bench + all gen ===")
    combined_ds_all = {**bench_ds, **gen_ds}
    tr_pairs_all = build_pairs(combined_ds_all, bench_train_c + gen_train_c)
    tau, top1, best_head = train_and_eval("B+G_all", tr_pairs_all, val_pairs,
                                          bench_ds, bench_val_c)
    results.append({"label": "B+G_all", "tau": tau, "top1": top1,
                    "n_train_cx": len(bench_train_c)+len(gen_train_c),
                    "n_train_pairs": len(tr_pairs_all)})

    log.info("\n=== Summary ===")
    for r in results:
        log.info("  %-20s  τ=%.4f  top1=%.3f  cx=%d  pairs=%d",
                 r["label"], r["tau"], r["top1"], r["n_train_cx"], r["n_train_pairs"])

    best = max(results, key=lambda x: x["tau"])
    log.info("  Best: %s  τ=%.4f", best["label"], best["tau"])

    # Save best head
    if best["label"] == "B+G_all":
        ckpt_path = REPO / "train_models" / "confidence_model" / "e0_extended_head.pt"
        torch.save(best_head.state_dict(), ckpt_path)
        log.info("  Saved best head → %s", ckpt_path)


if __name__ == "__main__":
    main()
