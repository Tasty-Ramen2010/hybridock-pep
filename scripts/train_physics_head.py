#!/usr/bin/env python3
"""
train_physics_head.py — Train confidence head on physics features.

Feature sources:
  feats_bench300_physics.pkl    14-dim per-term ref2015 + interface ΔΔG
  feats_bench300.pkl            96-dim encoder features (existing)

Conditions tested:
  physics_only   14-dim physics features
  encoder_only   96-dim encoder features (baseline)
  combined       [physics || encoder] = 110-dim concatenated
  physics_linear linear model on 14-dim physics (interpretable baseline)

The best condition informs which features to use in production.

Key finding: FastRelax hurts ranking (τ drops 0.174→0.139). All physics
features here are from score-only on raw diffusion poses.

Usage (score-env or rapidock env):
  python3 scripts/train_physics_head.py
"""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(8)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", force=True)
log = logging.getLogger("physics_head")

REPO       = Path(__file__).resolve().parent.parent
FEAT_BENCH = REPO / "logs" / "diagnosis" / "feats_bench300.pkl"
PHYS_BENCH = REPO / "logs" / "diagnosis" / "feats_bench300_physics.pkl"
BENCH_JSON = REPO / "logs" / "analysis_bench300" / "benchmark_results.json"
OUT        = REPO / "logs" / "training_campaign"
OUT.mkdir(exist_ok=True)


class ConfidenceHead(nn.Module):
    """Small MLP confidence head. in_dim adjusts for physics (14) vs encoder (96) vs combined (110)."""
    def __init__(self, in_dim: int = 14):
        super().__init__()
        hidden = max(32, in_dim * 2)
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),                    # normalise (critical for physics features)
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x): return self.net(x)


class LinearHead(nn.Module):
    """Linear model on physics features — interpretable, shows learned term weights."""
    def __init__(self, in_dim: int = 14):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x): return self.linear(x)


def bpr_loss(si, sj, label):
    return -F.logsigmoid((si - sj) * (label * 2.0 - 1.0)).mean()


def split_complexes(complexes, frac=0.85, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(complexes)); n = int(frac * len(complexes))
    return [complexes[i] for i in idx[:n]], [complexes[i] for i in idx[n:]]


def build_dataset(feat_map, json_data):
    ds = {}
    for (cname, mkey, pi), feat in feat_map.items():
        rmsds = json_data.get(cname, {}).get(mkey, {}).get("ref_rmsds", [])
        if pi >= len(rmsds): continue
        ds.setdefault(cname, []).append((feat.astype(np.float32), float(rmsds[pi])))
    return {k: v for k, v in ds.items() if len(v) >= 2}


def build_pairs(ds, complexes):
    pairs = []
    for c in complexes:
        for (fi, ri), (fj, rj) in combinations(ds.get(c, []), 2):
            if abs(ri - rj) < 1e-6: continue
            pairs.append((fi, fj, 1.0 if ri < rj else 0.0))
    return pairs


def slice_ds(ds, k):
    """Return a copy of ds keeping only the first k feature dims (for ablation)."""
    return {c: [(f[:k].copy(), r) for f, r in v] for c, v in ds.items()}


def train_and_eval(label: str, train_pairs, val_pairs, val_ds, val_c,
                   head: nn.Module, lr: float = 1e-3, epochs: int = 100,
                   seed: int = 0) -> tuple[float, float]:
    from scipy import stats as sp

    torch.manual_seed(seed)
    for m in head.modules():
        if hasattr(m, "reset_parameters"): m.reset_parameters()

    fi  = torch.tensor(np.stack([p[0] for p in train_pairs]), dtype=torch.float32)
    fj  = torch.tensor(np.stack([p[1] for p in train_pairs]), dtype=torch.float32)
    lbl = torch.tensor([p[2] for p in train_pairs], dtype=torch.float32)
    vfi = torch.tensor(np.stack([p[0] for p in val_pairs]), dtype=torch.float32)
    vfj = torch.tensor(np.stack([p[1] for p in val_pairs]), dtype=torch.float32)

    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_tau, best_state = -1.0, None; n = len(train_pairs)

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n)
        for b in range(0, n, 512):
            idx = perm[b: b+512]
            loss = bpr_loss(head(fi[idx]).squeeze(-1), head(fj[idx]).squeeze(-1), lbl[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        head.eval()
        with torch.no_grad():
            acc = (head(vfi).squeeze(-1) > head(vfj).squeeze(-1)).float().mean().item()
        if acc > best_tau: best_tau = acc; best_state = copy.deepcopy(head.state_dict())

    head.load_state_dict(best_state); head.eval()

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

    mean_tau  = float(np.mean(taus)) if taus else float("nan")
    mean_top1 = float(np.mean(tops)) if tops else float("nan")
    log.info("  %-25s τ=%.4f  top1=%.3f  pairs=%d  val_cx=%d",
             label, mean_tau, mean_top1, len(train_pairs), len(taus))
    return mean_tau, mean_top1


def main():
    bench_json = json.load(open(BENCH_JSON))

    # Load encoder features
    with open(FEAT_BENCH, "rb") as f: enc_feats = pickle.load(f)
    enc_ds = build_dataset(enc_feats, bench_json)
    all_cx  = sorted(enc_ds.keys())
    train_c, val_c = split_complexes(all_cx, 0.85, seed=42)
    enc_tr_pairs = build_pairs(enc_ds, train_c)
    enc_va_pairs = build_pairs(enc_ds, val_c)
    log.info("Encoder features: %d cx, %d train pairs, %d val pairs",
             len(all_cx), len(enc_tr_pairs), len(enc_va_pairs))

    results = []

    # ── E0: encoder baseline ─────────────────────────────────────────────────
    log.info("\n=== E0: encoder-only (96-dim, baseline) ===")
    from scripts.e0_extended import V2Head as V2HeadEnc   # reuse definition
    head_enc = V2HeadEnc(in_dim=96)
    t, t1 = train_and_eval("E0_encoder_only", enc_tr_pairs, enc_va_pairs,
                            enc_ds, val_c, head_enc)
    results.append({"label": "E0_encoder", "tau": t, "top1": t1,
                    "feat_dim": 96, "n_pairs": len(enc_tr_pairs)})

    # ── Load physics features ─────────────────────────────────────────────────
    if not PHYS_BENCH.exists():
        log.error("Physics features not found: %s", PHYS_BENCH)
        log.error("Run: python3 scripts/extract_physics_features.py --bench")
        log.info("Showing encoder-only baseline result above. Exiting.")
        return

    with open(PHYS_BENCH, "rb") as f: phys_feats = pickle.load(f)
    phys_ds_full = build_dataset(phys_feats, bench_json)
    phys_dim = next(iter(phys_feats.values())).shape[0]   # 16 (14 static + 2 response)
    n_static = 14
    has_response = phys_dim >= 16

    # Restrict to complexes in both datasets
    common_cx = sorted(set(phys_ds_full.keys()) & set(enc_ds.keys()))
    p_train_c = [c for c in train_c if c in phys_ds_full]
    p_val_c   = [c for c in val_c   if c in phys_ds_full]
    log.info("Physics features: %d cx, dim=%d (static=%d, response=%s)",
             len(phys_ds_full), phys_dim, n_static, "yes" if has_response else "no")

    # ── ABLATION: static-only (14) vs static+response (16) ────────────────────
    # This is the experiment: does the relaxation *response* (resp_delta_e,
    # resp_ca_disp) add ranking signal over raw score-only terms?
    ablation = [("P1_static14", n_static)]
    if has_response:
        ablation.append(("P1_response16", phys_dim))

    heads_by_label: dict[str, nn.Module] = {}
    phys_ds = phys_ds_full   # default for combined section below (full feats)
    for label, k in ablation:
        ds_k = slice_ds(phys_ds_full, k)
        tr = build_pairs(ds_k, p_train_c)
        va = build_pairs(ds_k, p_val_c)
        log.info("\n=== %s: physics MLP (%d-dim) ===", label, k)
        head_k = ConfidenceHead(in_dim=k)
        t, t1 = train_and_eval(label, tr, va, ds_k, p_val_c, head_k)
        heads_by_label[label] = head_k
        results.append({"label": label, "tau": t, "top1": t1,
                        "feat_dim": k, "n_pairs": len(tr)})

    # ── Physics-only (linear, full dims) — interpretable term weights ─────────
    log.info("\n=== P2: physics linear model (%d-dim, interpretable) ===", phys_dim)
    lin_tr = build_pairs(phys_ds_full, p_train_c)
    lin_va = build_pairs(phys_ds_full, p_val_c)
    head_lin = LinearHead(in_dim=phys_dim)
    t, t1 = train_and_eval("P2_physics_linear", lin_tr, lin_va,
                            phys_ds_full, p_val_c, head_lin)
    results.append({"label": "P2_physics_linear", "tau": t, "top1": t1,
                    "feat_dim": phys_dim, "n_pairs": len(lin_tr)})

    # Print learned linear weights (interpretable)
    with torch.no_grad():
        w = head_lin.linear.weight.squeeze().numpy()
        b = head_lin.linear.bias.item()
    feat_names = ["fa_atr","fa_rep","fa_sol","fa_intra_rep","fa_elec",
                  "hbond_bb_sc","hbond_sc","hbond_lr_bb","hbond_sr_bb",
                  "rama_prepro","fa_dun","p_aa_pp","interface_ddG","total_score",
                  "resp_delta_e","resp_ca_disp"][:phys_dim]
    log.info("  Linear weights:")
    for name, weight in sorted(zip(feat_names, w), key=lambda x: -abs(x[1])):
        log.info("    %20s  %.4f", name, weight)

    # ── Combined: physics + encoder ───────────────────────────────────────────
    log.info("\n=== P3: combined physics + encoder (%d-dim) ===", phys_dim + 96)
    # Build combined dataset: only complexes in both
    combined_ds = {}
    for c in common_cx:
        enc_poses   = enc_ds.get(c, [])
        phys_poses  = phys_ds.get(c, [])
        if len(enc_poses) != len(phys_poses): continue
        combined = [(np.concatenate([ep[0], pp[0]]), ep[1])
                    for ep, pp in zip(enc_poses, phys_poses)]
        combined_ds[c] = combined
    c_train = [c for c in p_train_c if c in combined_ds]
    c_val   = [c for c in p_val_c   if c in combined_ds]
    comb_tr = build_pairs(combined_ds, c_train)
    comb_va = build_pairs(combined_ds, c_val)
    head_comb = ConfidenceHead(in_dim=phys_dim + 96)
    t, t1 = train_and_eval("P3_combined", comb_tr, comb_va,
                            combined_ds, c_val, head_comb)
    heads_by_label["P3_combined"] = head_comb
    heads_by_label["P2_physics_linear"] = head_lin
    results.append({"label": "P3_combined", "tau": t, "top1": t1,
                    "feat_dim": phys_dim + 96, "n_pairs": len(comb_tr)})

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("PHYSICS HEAD TRAINING RESULTS")
    log.info("="*60)
    log.info("  %-25s  %8s  %8s  %6s", "Label", "τ", "top1 Å", "dim")
    log.info("  " + "-"*55)
    for r in sorted(results, key=lambda x: -x["tau"]):
        log.info("  %-25s  %8.4f  %8.3f  %6d", r["label"], r["tau"], r["top1"], r["feat_dim"])
    log.info("="*60)

    best = max(results, key=lambda x: x["tau"])
    log.info("Best: %s  τ=%.4f", best["label"], best["tau"])

    # Save best physics head (any trained head we still hold a reference to)
    if best["label"] in heads_by_label:
        ckpt = REPO / "train_models" / "confidence_model" / "physics_head.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        best_head = heads_by_label[best["label"]]
        torch.save({"state_dict": best_head.state_dict(),
                    "label": best["label"],
                    "feat_dim": best["feat_dim"],
                    "tau": best["tau"]}, ckpt)
        log.info("Saved physics head → %s", ckpt)


if __name__ == "__main__":
    main()
