#!/usr/bin/env python3
"""
confidence_training_campaign.py — Overnight training campaign (6 experiments).

Experiments
-----------
A  Data scaling           25/50/75/100% bench complexes, 5 seeds
B  Complex vs pair count  fix pairs/vary complexes  vs  fix complexes/vary pairs, 5 seeds
C  SS specialists         global / short / medium / long / short+med / med+long, 5 seeds
D  Mixture ratios         0–100% bench300 vs gen_ood (6 configs), 5 seeds
E  Frozen vs finetune     frozen head / unfreeze 1 block / unfreeze 2 blocks, GPU, 1 seed
F  Router validation      5 models × 5 folds × 5 seeds

All head-training experiments use cached 96-dim encoder features (no GPU required).
Exp E requires CUDA; falls back gracefully if unavailable.

Usage
-----
  conda run -n rapidock python3 scripts/confidence_training_campaign.py --device cuda
  conda run -n rapidock python3 scripts/confidence_training_campaign.py --device cuda --skip-e
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import multiprocessing
import os
import pickle
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

REPO       = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "RAPiDock"))

FEAT_BENCH = REPO / "logs" / "diagnosis" / "feats_bench300.pkl"
FEAT_GEN   = REPO / "logs" / "diagnosis" / "feats_gen_ood.pkl"
BENCH_JSON = REPO / "logs" / "analysis_bench300" / "benchmark_results.json"
GEN_JSON   = REPO / "logs" / "confidence_training_data" / "benchmark_results.json"
BENCH_CSV  = REPO / "data" / "benchmark300.csv"
PARAMS_YML = REPO / "train_models" / "confidence_model" / "model_parameters.yml"
PRETRAINED = REPO / "third_party" / "RAPiDock" / "train_models" / \
             "CGTensorProductEquivariantModel" / "rapidock_global.pt"
OUT        = REPO / "logs" / "training_campaign"
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S",
                    handlers=[logging.StreamHandler()])
# Force line-buffered stdout/stderr so nohup doesn't hold output in memory
import sys as _sys
for _h in logging.root.handlers:
    if hasattr(_h, "stream"):
        _h.stream.reconfigure(line_buffering=True) if hasattr(_h.stream, "reconfigure") else None
log = logging.getLogger("campaign")

SEEDS = [0, 1, 2, 3, 4]


# ── head architectures ────────────────────────────────────────────────────────

class LinearHead(nn.Module):
    def __init__(self, in_dim: int = 96):
        super().__init__()
        self.w = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)


class V2Head(nn.Module):
    def __init__(self, in_dim: int = 96, h1: int = 128, h2: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.LayerNorm(h1), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── training utilities ────────────────────────────────────────────────────────

def bpr_loss(si: torch.Tensor, sj: torch.Tensor,
             label: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid((si - sj) * (label * 2.0 - 1.0)).mean()


def _pairs_to_tensors(pairs: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fi  = torch.tensor(np.stack([p[0] for p in pairs]), dtype=torch.float32)
    fj  = torch.tensor(np.stack([p[1] for p in pairs]), dtype=torch.float32)
    lbl = torch.tensor([p[2] for p in pairs],           dtype=torch.float32)
    return fi, fj, lbl


def train_head(head: nn.Module, train_pairs: list, val_pairs: list,
               epochs: int = 50, lr: float = 1e-3,
               batch_size: int = 512, seed: int = 0) -> dict:
    """BPR mini-batch training. Checkpoints on best val_acc."""
    if not train_pairs:
        nan = float("nan")
        return {"train_acc": nan, "val_acc": nan, "best_val_acc": nan,
                "best_epoch": -1, "overfit_gap": nan}

    torch.set_num_threads(1)  # tiny tensors — avoid 72-thread BLAS overhead
    torch.manual_seed(seed)
    # Re-init head weights deterministically
    def _reset(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.LayerNorm)):
            m.reset_parameters()
    head.apply(_reset)

    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    tr_fi, tr_fj, tr_lbl = _pairs_to_tensors(train_pairs)
    va_fi, va_fj, va_lbl = (_pairs_to_tensors(val_pairs) if val_pairs
                             else (None, None, None))

    def _acc(fi: torch.Tensor | None, fj: torch.Tensor | None,
             lbl: torch.Tensor | None) -> float:
        if fi is None:
            return float("nan")
        head.eval()
        with torch.no_grad():
            si = head(fi).squeeze(-1)
            sj = head(fj).squeeze(-1)
        return ((si > sj).float() - lbl).abs().lt(0.5).float().mean().item()

    best_val_acc  = -1.0
    best_state    = None
    best_epoch    = -1
    n             = len(train_pairs)

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + ep))
        for b in range(0, n, batch_size):
            idx = perm[b: b + batch_size]
            si  = head(tr_fi[idx]).squeeze(-1)
            sj  = head(tr_fj[idx]).squeeze(-1)
            loss = bpr_loss(si, sj, tr_lbl[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        val_acc = _acc(va_fi, va_fj, va_lbl)
        if not math.isnan(val_acc) and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(head.state_dict())
            best_epoch   = ep

    if best_state is not None:
        head.load_state_dict(best_state)

    trn_acc = _acc(tr_fi[:min(2000, n)], tr_fj[:min(2000, n)], tr_lbl[:min(2000, n)])
    return {
        "train_acc":   trn_acc,
        "val_acc":     _acc(va_fi, va_fj, va_lbl),
        "best_val_acc": best_val_acc,
        "best_epoch":  best_epoch,
        "overfit_gap": (trn_acc - _acc(va_fi, va_fj, va_lbl))
                       if not math.isnan(trn_acc) else float("nan"),
    }


def eval_tau_full(head: nn.Module, ds: dict, complexes: list) -> dict:
    """Returns dict with tau, top1_rmsd, and per-complex details."""
    head.eval()
    taus, tops = [], []
    per_cx     = []
    with torch.no_grad():
        for cname in complexes:
            poses = ds.get(cname, [])
            if len(poses) < 2:
                continue
            feats  = torch.tensor(np.array([p[0] for p in poses], dtype=np.float32))
            rmsds  = np.array([p[1] for p in poses])
            scores = head(feats).squeeze(-1).numpy()
            tau, _ = __import__("scipy").stats.kendalltau(-scores, rmsds)
            if math.isnan(tau):
                continue
            top1 = float(rmsds[np.argmax(scores)])
            taus.append(tau)
            tops.append(top1)
            per_cx.append({"complex": cname, "tau": tau, "top1": top1,
                           "best_rmsd": float(rmsds.min()),
                           "n_poses": len(poses)})
    return {
        "tau":      float(np.mean(taus))  if taus else float("nan"),
        "top1":     float(np.mean(tops))  if tops else float("nan"),
        "per_cx":   per_cx,
    }


# ── dataset utilities ─────────────────────────────────────────────────────────

def build_dataset(feat_map: dict, json_data: dict,
                  variants: list[str] | None = None) -> dict:
    """Build {cname: [(feat_vec, rmsd), ...]} from feature cache."""
    ds: dict[str, list] = {}
    for (cname, mkey, pose_idx), feat in feat_map.items():
        if variants is not None and mkey not in variants:
            continue
        poses_rmsds = json_data.get(cname, {}).get(mkey, {}).get("ref_rmsds", [])
        if pose_idx >= len(poses_rmsds):
            continue
        rmsd = float(poses_rmsds[pose_idx])
        ds.setdefault(cname, []).append((feat.astype(np.float32), rmsd))
    return {k: v for k, v in ds.items() if len(v) >= 2}


def split_complexes(complexes: list, train_frac: float = 0.85,
                    seed: int = 42) -> tuple[list, list]:
    rng   = np.random.RandomState(seed)
    idx   = rng.permutation(len(complexes))
    n_tr  = max(1, int(len(complexes) * train_frac))
    return [complexes[i] for i in idx[:n_tr]], [complexes[i] for i in idx[n_tr:]]


def build_pairs(ds: dict, complexes: list,
                max_pairs_per_complex: int | None = None,
                max_total_pairs: int | None = None,
                seed: int = 0) -> list:
    rng   = np.random.RandomState(seed)
    pairs = []
    for cname in complexes:
        poses = ds.get(cname, [])
        if len(poses) < 2:
            continue
        cands = [(fi.astype(np.float32), fj.astype(np.float32),
                  1.0 if ri < rj else 0.0)
                 for (fi, ri), (fj, rj) in combinations(poses, 2)
                 if abs(ri - rj) > 1e-6]
        if max_pairs_per_complex is not None and len(cands) > max_pairs_per_complex:
            idx   = rng.choice(len(cands), max_pairs_per_complex, replace=False)
            cands = [cands[i] for i in idx]
        pairs.extend(cands)
        if max_total_pairs is not None and len(pairs) >= max_total_pairs:
            pairs = pairs[:max_total_pairs]
            break
    return pairs


def load_meta(bench_csv: Path) -> pd.DataFrame:
    """Load SS class / length bucket labels indexed by complex name."""
    df = pd.read_csv(bench_csv, index_col="name")
    return df[["ss_class", "length_bucket", "pep_len"]]


# ── worker (called in subprocess) ────────────────────────────────────────────

def _worker(spec: dict) -> dict:
    """Run one training job from a flat spec dict. Returns result dict."""
    # Limit PyTorch to 1 thread — 8 workers × 24 torch threads = 192 threads
    # competing for 24 cores causes severe contention and slowdown.
    import torch as _torch
    _torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    # Each subprocess loads caches independently (safe for spawn)
    import pickle as _pickle
    from scipy import stats as _sp

    with open(spec["feat_bench_path"], "rb") as f:
        bench_feats = _pickle.load(f)
    bench_json = json.load(open(spec["bench_json_path"]))

    gen_feats  = None
    gen_json   = {}
    if spec.get("feat_gen_path"):
        with open(spec["feat_gen_path"], "rb") as f:
            gen_feats = _pickle.load(f)
        gen_json = json.load(open(spec["gen_json_path"]))

    bench_ds   = build_dataset(bench_feats, bench_json,
                               variants=spec.get("bench_variants"))
    bench_all  = sorted(bench_ds.keys())
    _, bench_val = split_complexes(bench_all, 0.85, seed=42)  # fixed val

    gen_ds     = (build_dataset(gen_feats, gen_json) if gen_feats else {})
    gen_all    = sorted(gen_ds.keys())
    _, gen_val = split_complexes(gen_all, 0.85, seed=42)

    # ── select train complexes ────────────────────────────────────────────────
    train_c = spec["train_complexes"]  # pre-computed by main
    val_c   = spec["val_complexes"]
    ds_src  = bench_ds if spec["train_source"] == "bench" else gen_ds

    # Mix bench + gen if ratio specified
    if spec.get("bench_frac") is not None:
        bench_frac = spec["bench_frac"]
        rng_s = np.random.RandomState(spec["seed"])
        b_tr, _ = split_complexes(bench_all, 0.85, seed=42)
        g_tr, _ = split_complexes(gen_all,   0.85, seed=42)
        n_b = int(round(bench_frac * len(b_tr)))
        n_g = len(g_tr)
        if n_b > 0:
            b_sel = list(rng_s.choice(b_tr, min(n_b, len(b_tr)), replace=False))
        else:
            b_sel = []
        # Build merged dataset
        ds_src = {}
        for c in b_sel:
            ds_src[c] = bench_ds[c]
        for c in g_tr:
            ds_src[c] = gen_ds[c]
        train_c = b_sel + g_tr
        val_c   = bench_val

    # ── build pairs ───────────────────────────────────────────────────────────
    ppx = spec.get("max_pairs_per_complex")
    ptt = spec.get("max_total_pairs")
    train_pairs = build_pairs(ds_src, train_c,
                              max_pairs_per_complex=ppx,
                              max_total_pairs=ptt,
                              seed=spec["seed"])
    # val_pairs always come from bench_ds so the score is on a consistent distribution
    val_pairs   = build_pairs(bench_ds, bench_val, seed=42)

    # ── train ─────────────────────────────────────────────────────────────────
    head    = V2Head()
    metrics = train_head(head, train_pairs, val_pairs,
                         epochs=spec.get("epochs", 50),
                         seed=spec["seed"])

    # ── eval on bench300 val (always) ─────────────────────────────────────────
    bench_eval_c = spec.get("bench_eval_complexes", bench_val)
    ev = eval_tau_full(head, bench_ds, bench_eval_c)

    result = {
        "exp":         spec["exp"],
        "label":       spec["label"],
        "seed":        spec["seed"],
        "n_train_cx":  len(train_c),
        "n_train_pairs": len(train_pairs),
        "tau":         ev["tau"],
        "top1":        ev["top1"],
        **metrics,
    }

    # Per-subset τ if meta supplied
    if spec.get("meta"):
        meta = pd.DataFrame(spec["meta"]).set_index("name")
        for col in ["ss_class", "length_bucket"]:
            for val2, grp in meta.reindex(bench_eval_c).dropna().groupby(col):
                sub_c = list(grp.index)
                r2    = eval_tau_full(head, bench_ds, sub_c)
                result[f"tau_{col}_{val2}"] = r2["tau"]
                result[f"top1_{col}_{val2}"] = r2["top1"]

    return result


# ── Exp-A: data scaling ───────────────────────────────────────────────────────

def build_exp_a_specs(bench_ds: dict, bench_val: list) -> list[dict]:
    bench_train, _ = split_complexes(sorted(bench_ds.keys()), 0.85, seed=42)
    specs = []
    for frac in [0.25, 0.50, 0.75, 1.00]:
        n = max(1, int(frac * len(bench_train)))
        for seed in SEEDS:
            rng   = np.random.RandomState(seed)
            sel   = list(rng.choice(bench_train, n, replace=False))
            specs.append({
                "exp":              "A_data_scaling",
                "label":            f"frac={frac:.2f}",
                "seed":             seed,
                "train_source":     "bench",
                "train_complexes":  sel,
                "val_complexes":    bench_val,
                "bench_eval_complexes": bench_val,
                "feat_bench_path":  str(FEAT_BENCH),
                "bench_json_path":  str(BENCH_JSON),
                "epochs":           50,
                "frac":             frac,
            })
    return specs


# ── Exp-B: complex count vs pair count ───────────────────────────────────────

def build_exp_b_specs(bench_ds: dict, bench_val: list) -> list[dict]:
    bench_train, _ = split_complexes(sorted(bench_ds.keys()), 0.85, seed=42)
    n_cx_full      = len(bench_train)
    specs          = []

    # B1: vary complex count, fix total pairs to ~2000
    TARGET_PAIRS = 2000
    for n_cx in [30, 60, 120, n_cx_full]:
        ppx = max(1, TARGET_PAIRS // n_cx)
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            sel = list(rng.choice(bench_train, n_cx, replace=False))
            specs.append({
                "exp":                 "B_vary_complexes",
                "label":               f"n_cx={n_cx}_fixed_pairs",
                "seed":                seed,
                "train_source":        "bench",
                "train_complexes":     sel,
                "val_complexes":       bench_val,
                "bench_eval_complexes": bench_val,
                "feat_bench_path":     str(FEAT_BENCH),
                "bench_json_path":     str(BENCH_JSON),
                "max_pairs_per_complex": ppx,
                "epochs":              50,
                "n_cx":                n_cx,
            })

    # B2: fix complex count to 60, vary pairs per complex
    N_CX_FIXED = 60
    rng_fixed   = np.random.RandomState(99)
    fixed_sel   = list(rng_fixed.choice(bench_train, N_CX_FIXED, replace=False))
    for ppx in [1, 5, 20, None]:  # None = all pairs
        for seed in SEEDS:
            specs.append({
                "exp":                 "B_vary_pairs",
                "label":               f"ppx={ppx}_fixed_cx",
                "seed":                seed,
                "train_source":        "bench",
                "train_complexes":     fixed_sel,
                "val_complexes":       bench_val,
                "bench_eval_complexes": bench_val,
                "feat_bench_path":     str(FEAT_BENCH),
                "bench_json_path":     str(BENCH_JSON),
                "max_pairs_per_complex": ppx,
                "epochs":              50,
                "ppx":                 ppx,
            })
    return specs


# ── Exp-C: SS / length specialists ───────────────────────────────────────────

def build_exp_c_specs(bench_ds: dict, bench_val: list,
                      meta: pd.DataFrame) -> list[dict]:
    bench_train, _ = split_complexes(sorted(bench_ds.keys()), 0.85, seed=42)
    meta_train     = meta.reindex(bench_train).dropna()
    specs          = []

    SHORT_LEN = ["short"]
    MED_LEN   = ["medium"]
    LONG_LEN  = ["long", "very_long"]

    subsets = {
        "global":      bench_train,
        "short":       list(meta_train[meta_train.length_bucket.isin(SHORT_LEN)].index),
        "medium":      list(meta_train[meta_train.length_bucket.isin(MED_LEN)].index),
        "long":        list(meta_train[meta_train.length_bucket.isin(LONG_LEN)].index),
        "short_med":   list(meta_train[meta_train.length_bucket.isin(SHORT_LEN + MED_LEN)].index),
        "med_long":    list(meta_train[meta_train.length_bucket.isin(MED_LEN + LONG_LEN)].index),
        "HELIX":       list(meta_train[meta_train.ss_class == "HELIX"].index),
        "SHEET":       list(meta_train[meta_train.ss_class == "SHEET"].index),
        "UNUSUAL":     list(meta_train[meta_train.ss_class == "UNUSUAL"].index),
    }

    meta_records = meta.reset_index().rename(columns={"name": "name"}).to_dict("records")

    for label, train_c in subsets.items():
        if len(train_c) < 5:
            continue
        for seed in SEEDS:
            specs.append({
                "exp":                 "C_specialists",
                "label":               label,
                "seed":                seed,
                "train_source":        "bench",
                "train_complexes":     train_c,
                "val_complexes":       bench_val,
                "bench_eval_complexes": bench_val,
                "feat_bench_path":     str(FEAT_BENCH),
                "bench_json_path":     str(BENCH_JSON),
                "meta":                meta_records,
                "epochs":              50,
            })
    return specs


# ── Exp-D: mixture ratios ─────────────────────────────────────────────────────

def build_exp_d_specs(bench_ds: dict, gen_ds: dict,
                      bench_val: list) -> list[dict]:
    specs = []
    for bench_frac in [0.0, 0.25, 0.50, 0.75, 0.90, 1.00]:
        for seed in SEEDS:
            specs.append({
                "exp":          "D_mixture_ratios",
                "label":        f"B{bench_frac:.0%}G{1-bench_frac:.0%}",
                "seed":         seed,
                "train_source": "bench",   # overridden by bench_frac
                "train_complexes": [],     # computed in worker via bench_frac
                "val_complexes":   bench_val,
                "bench_eval_complexes": bench_val,
                "bench_frac":   bench_frac,
                "feat_bench_path":  str(FEAT_BENCH),
                "bench_json_path":  str(BENCH_JSON),
                "feat_gen_path":    str(FEAT_GEN),
                "gen_json_path":    str(GEN_JSON),
                "epochs":           50,
            })
    return specs


# ── Exp-F: router validation ──────────────────────────────────────────────────

def build_exp_f_specs(bench_ds: dict, gen_ds: dict,
                      meta: pd.DataFrame, n_folds: int = 5) -> list[dict]:
    """5-fold CV × 5 seeds × 5 model configs. Eval routing gain per fold."""
    bench_all = sorted(bench_ds.keys())
    rng_fold  = np.random.RandomState(7)
    shuffled  = list(rng_fold.permutation(len(bench_all)))
    folds     = [shuffled[i::n_folds] for i in range(n_folds)]

    configs = {
        "bench_only":  {"bench_frac": 1.00},
        "gen_only":    {"bench_frac": 0.00},
        "25B_75G":     {"bench_frac": 0.25},
        "50B_50G":     {"bench_frac": 0.50},
        "75B_25G":     {"bench_frac": 0.75},
    }

    specs  = []
    gen_all = sorted(gen_ds.keys())
    _, gen_val = split_complexes(gen_all, 0.85, seed=42)

    meta_records = meta.reset_index().rename(columns={"name": "name"}).to_dict("records")

    for fold_idx, val_idx in enumerate(folds):
        val_c   = [bench_all[i] for i in val_idx]
        train_c = [bench_all[i] for i in shuffled if i not in val_idx]
        for cfg_name, cfg in configs.items():
            for seed in SEEDS:
                specs.append({
                    "exp":                 "F_router_cv",
                    "label":               f"fold={fold_idx}_{cfg_name}",
                    "seed":                seed,
                    "train_source":        "bench",
                    "train_complexes":     train_c,
                    "val_complexes":       val_c,
                    "bench_eval_complexes": val_c,
                    "bench_frac":          cfg.get("bench_frac"),
                    "feat_bench_path":     str(FEAT_BENCH),
                    "bench_json_path":     str(BENCH_JSON),
                    "feat_gen_path":       str(FEAT_GEN),
                    "gen_json_path":       str(GEN_JSON),
                    "meta":                meta_records,
                    "epochs":              50,
                    "fold":                fold_idx,
                    "cfg_name":            cfg_name,
                })
    return specs


# ── Exp-E: frozen vs partial encoder finetune (GPU) ───────────────────────────

def run_exp_e(bench_ds: dict, bench_val: list, device: str) -> list[dict]:
    """
    Three conditions:
      E0: frozen encoder  (use cached features, fast)
      E1: unfreeze cross_convs[-1]   (GPU, 1 seed)
      E2: unfreeze cross_convs[-2:]  (GPU, 1 seed)
    """
    from scipy import stats as _sp
    import yaml

    log.info("=== Exp E: Frozen vs partial encoder finetune ===")
    bench_train, _ = split_complexes(sorted(bench_ds.keys()), 0.85, seed=42)
    results        = []

    # E0: frozen (cached features) — same as best from Exp A at 100%
    log.info("  E0: frozen encoder (cached features, V2Head)")
    head_e0 = V2Head()
    tr_pairs = build_pairs(bench_ds, bench_train)
    va_pairs = build_pairs(bench_ds, bench_val)
    m0 = train_head(head_e0, tr_pairs, va_pairs, epochs=50, seed=0)
    ev0 = eval_tau_full(head_e0, bench_ds, bench_val)
    results.append({
        "exp": "E_finetune", "label": "E0_frozen",
        "seed": 0, "n_train_cx": len(bench_train),
        "n_train_pairs": len(tr_pairs),
        "tau": ev0["tau"], "top1": ev0["top1"], **m0,
        "n_unfreeze_blocks": 0, "trainable_params": sum(
            p.numel() for p in head_e0.parameters()),
    })
    log.info("    E0 τ=%.4f  top1=%.3f", ev0["tau"], ev0["top1"])

    if device == "cpu":
        log.warning("  Skipping E1/E2: no GPU available")
        return results

    # Load encoder for E1 / E2
    log.info("  Loading pretrained encoder for E1/E2...")
    with open(PARAMS_YML) as f:
        import yaml as _yaml
        params = _yaml.safe_load(f)
    params["confidence_mode"] = True

    from models.model import ConfidenceModel
    from argparse import Namespace as _NS
    from utils.diffusion_utils import set_time
    from torch_geometric.data import Batch
    import MDAnalysis

    def _load_pose_positions(pdb: str, exclude_oxt: bool = False):
        try:
            u   = MDAnalysis.Universe(pdb)
            pos = []
            for res in u.residues:
                sel  = "not type H" + (" and not name OXT" if exclude_oxt else "")
                heavy = res.atoms.select_atoms(sel)
                ca    = heavy.select_atoms("name CA")
                if not len(ca) or not len(heavy):
                    continue
                pos.append(heavy.positions.astype(np.float32))
            return torch.tensor(np.concatenate(pos)) if pos else None
        except Exception:
            return None

    def _inject_pose(bg, pos):
        g      = copy.deepcopy(bg)
        center = pos.mean(0)
        g["pep_a"].pos      = pos - center
        if g["pep_a"].x is not None:
            g["pep_a"].x    = g["pep_a"].x.float()
        g["receptor"].pos   = g["receptor"].pos - center
        if hasattr(g["pep_a"], "node_sigma_emb"):
            del g["pep_a"].node_sigma_emb
        return g

    # Build pose graph cache (base graphs already cached in memory from feats build)
    # We need to re-build them here since we can't pass them from main process
    log.info("  Building bench300 pose graphs for E1/E2 (may take ~5 min)...")
    from utils.inference_utils import InferenceDataset
    df_bench = pd.read_csv(BENCH_CSV)
    names, recs, peps = [], [], []
    for _, row in df_bench.iterrows():
        if Path(str(row.get("receptor",""))).exists() and \
           Path(str(row.get("peptide_pdb",""))).exists():
            names.append(row["name"])
            recs.append(str(row["receptor"]))
            peps.append(str(row["peptide_pdb"]))

    tmp_e = "/tmp/exp_e_graphs"
    os.makedirs(tmp_e, exist_ok=True)
    ds_build = InferenceDataset(
        output_dir=tmp_e, complex_name_list=names,
        protein_description_list=recs, peptide_description_list=peps,
        lm_embeddings=True, lm_embeddings_pep=False,
        conformation_type=None, conformation_partial="1:1:1",
    )
    base_graphs = {}
    for i, n in enumerate(names):
        try:
            g = ds_build.get(i)
            if g is not None:
                base_graphs[n] = g
        except Exception:
            pass
    log.info("  Built %d base graphs", len(base_graphs))

    # Assemble per-complex training data: {cname: [(inject_graph, rmsd), ...]}
    bench_json  = json.load(open(BENCH_JSON))
    train_poses: dict[str, list] = {}
    for cname in bench_train:
        if cname not in base_graphs or cname not in bench_json:
            continue
        bg  = base_graphs[cname]
        n_g = bg["pep_a"].pos.shape[0]
        cx_poses = []
        for mkey, res in bench_json[cname].items():
            pdir   = Path(res["poses_dir"])
            rmsds  = res.get("ref_rmsds", [])
            for i, rmsd in enumerate(rmsds):
                pdb = pdir / f"pose_{i}.pdb"
                if not pdb.exists():
                    continue
                pos = _load_pose_positions(str(pdb))
                if pos is None:
                    continue
                if pos.shape[0] != n_g:
                    pos = _load_pose_positions(str(pdb), exclude_oxt=True)
                    if pos is None or pos.shape[0] != n_g:
                        continue
                try:
                    g = _inject_pose(bg, pos)
                    set_time(g, 0.0, 0.0, 0.0, 0.0, 1, device="cpu")
                    cx_poses.append((g, float(rmsd)))
                except Exception:
                    pass
        if len(cx_poses) >= 2:
            train_poses[cname] = cx_poses
    log.info("  %d training complexes with pose graphs", len(train_poses))

    # E1 and E2
    for n_unfreeze, label in [(1, "E1_unfreeze_last1"), (2, "E2_unfreeze_last2")]:
        log.info("  %s: unfreezing %d cross_conv block(s)...", label, n_unfreeze)
        model = ConfidenceModel(_NS(**params))
        ckpt  = torch.load(PRETRAINED, map_location="cpu")
        _ckpt_state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(_ckpt_state, strict=False)
        model.eval()

        # Freeze all params
        for p in model.parameters():
            p.requires_grad_(False)

        # Unfreeze selected cross_conv blocks
        enc = model.encoder
        for blk in list(enc.cross_convs)[-n_unfreeze:]:
            for p in blk.parameters():
                p.requires_grad_(True)

        # Replace confidence_predictor with new V2Head
        new_head = V2Head().to(device)
        for p in new_head.parameters():
            p.requires_grad_(True)
        enc.confidence_predictor = new_head
        model.to(device)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info("    trainable params: %d", trainable)

        # Separate param groups: unfrozen encoder blocks at low LR, head at high LR
        encoder_block_params = [
            p for blk in list(enc.cross_convs)[-n_unfreeze:]
            for p in blk.parameters()
        ]
        opt = torch.optim.Adam([
            {"params": encoder_block_params, "lr": 5e-6, "weight_decay": 1e-4},
            {"params": list(new_head.parameters()), "lr": 1e-3, "weight_decay": 1e-4},
        ])

        N_EPOCHS = 20
        PAIRS_PER_CX = 10  # sample per complex per epoch

        best_tau   = -1.0
        best_state_h = None
        best_state_m = None
        rng_e = np.random.RandomState(0)

        for ep in range(N_EPOCHS):
            model.train()
            # Keep BN frozen in unfrozen blocks
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()

            total_loss = 0.0
            n_pairs    = 0
            cx_order   = list(train_poses.keys())
            rng_e.shuffle(cx_order)

            for cname in cx_order:
                cx_data = train_poses[cname]
                if len(cx_data) < 2:
                    continue
                n_cx = len(cx_data)
                try:
                    batch_g = Batch.from_data_list([g for g, _ in cx_data]).to(device)
                    scores = model(batch_g)  # [n_poses]
                    del batch_g
                except RuntimeError as _oom:
                    if "out of memory" not in str(_oom).lower():
                        continue
                    if device != "cpu":
                        torch.cuda.empty_cache()
                    # Fall back: score one pose at a time
                    try:
                        scores_list = []
                        for g_i, _ in cx_data:
                            g_dev = Batch.from_data_list([g_i]).to(device)
                            scores_list.append(model(g_dev).squeeze(0))
                            del g_dev
                        scores = torch.stack(scores_list)
                    except Exception:
                        continue
                except Exception:
                    continue

                rmsds = torch.tensor([r for _, r in cx_data], device=device)
                pair_idx = list(combinations(range(n_cx), 2))
                if len(pair_idx) > PAIRS_PER_CX:
                    sel = rng_e.choice(len(pair_idx), PAIRS_PER_CX, replace=False)
                    pair_idx = [pair_idx[i] for i in sel]

                loss = torch.tensor(0.0, device=device)
                for i, j in pair_idx:
                    ri, rj = rmsds[i], rmsds[j]
                    if (ri - rj).abs() < 1e-6:
                        continue
                    lbl = torch.tensor(1.0 if ri < rj else 0.0, device=device)
                    loss = loss + bpr_loss(scores[i:i+1], scores[j:j+1], lbl.unsqueeze(0))
                    n_pairs += 1

                if loss.item() > 0:
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        encoder_block_params + list(new_head.parameters()), 1.0)
                    opt.step()
                    total_loss += loss.item()

            # Eval τ on val set using frozen-feature approach (faster)
            model.eval()
            feats_captured: dict[str, list] = defaultdict(list)
            hook_feats: list[torch.Tensor] = []

            def _hook(mod, inp, out):  # noqa: E306
                hook_feats.append(inp[0].detach().cpu())

            handle = enc.confidence_predictor.net[0].register_forward_hook(_hook)
            with torch.no_grad():
                for cname in bench_val:
                    if cname not in base_graphs or cname not in bench_json:
                        continue
                    bg  = base_graphs[cname]
                    n_g = bg["pep_a"].pos.shape[0]
                    val_graphs, val_rmsds = [], []
                    for mkey, res in bench_json[cname].items():
                        for i_p, rmsd in enumerate(res.get("ref_rmsds", [])):
                            pdb = Path(res["poses_dir"]) / f"pose_{i_p}.pdb"
                            if not pdb.exists():
                                continue
                            pos = _load_pose_positions(str(pdb))
                            if pos is None or pos.shape[0] != n_g:
                                continue
                            try:
                                g = _inject_pose(bg, pos)
                                set_time(g, 0.0, 0.0, 0.0, 0.0, 1, device="cpu")
                                val_graphs.append(g)
                                val_rmsds.append(float(rmsd))
                            except Exception:
                                pass
                    if len(val_graphs) < 2:
                        continue
                    hook_feats.clear()
                    batch_v = Batch.from_data_list(val_graphs).to(device)
                    try:
                        model(batch_v)
                    except Exception:
                        continue
                    if hook_feats:
                        feats_captured[cname] = [
                            (hook_feats[0][k].numpy(), val_rmsds[k])
                            for k in range(hook_feats[0].shape[0])
                            if k < len(val_rmsds)
                        ]
            handle.remove()

            # Compute τ on val set
            from scipy import stats as _sp
            taus = []
            for cname, pose_data in feats_captured.items():
                if len(pose_data) < 2:
                    continue
                feats_v = torch.tensor(np.array([p[0] for p in pose_data], dtype=np.float32)).to(device)
                rmsds_v = np.array([p[1] for p in pose_data])
                with torch.no_grad():
                    scores_v = enc.confidence_predictor(feats_v).squeeze(-1).cpu().numpy()
                tau, _ = _sp.kendalltau(-scores_v, rmsds_v)
                if not math.isnan(tau):
                    taus.append(tau)
            ep_tau = float(np.mean(taus)) if taus else float("nan")
            log.info("    %s ep=%d  loss=%.4f  val_τ=%.4f",
                     label, ep, total_loss / max(n_pairs, 1), ep_tau)
            if not math.isnan(ep_tau) and ep_tau > best_tau:
                best_tau     = ep_tau
                best_state_h = copy.deepcopy(enc.confidence_predictor.state_dict())
                best_state_m = {k: v.clone() for k, v in
                                model.state_dict().items()
                                if any(f"cross_convs.{len(enc.cross_convs)-1-i}"
                                       in k for i in range(n_unfreeze))}

        results.append({
            "exp": "E_finetune", "label": label,
            "seed": 0, "n_train_cx": len(train_poses),
            "n_train_pairs": N_EPOCHS * len(train_poses) * PAIRS_PER_CX,
            "tau": best_tau, "top1": float("nan"),
            "best_val_acc": float("nan"), "train_acc": float("nan"),
            "val_acc": float("nan"), "best_epoch": -1, "overfit_gap": float("nan"),
            "n_unfreeze_blocks": n_unfreeze, "trainable_params": trainable,
        })
        log.info("  %s best_τ=%.4f", label, best_tau)

    return results


# ── report generation ─────────────────────────────────────────────────────────

def generate_report(all_results: pd.DataFrame) -> str:
    lines = ["# Confidence Training Campaign — Final Report", ""]

    BASELINE_TAU  = 0.201   # ceiling study 5-seed mean (bench_only, frozen)
    BASELINE_TOP1 = 4.35    # approximate baseline top1 RMSD

    def _fmt_exp(exp_label: str, df: pd.DataFrame, groupby: str = "label") -> str:
        out = [f"\n## {exp_label}\n"]
        grp = df.groupby(groupby)[["tau", "top1"]].agg(["mean", "std"]).reset_index()
        grp.columns = [groupby, "tau_mean", "tau_std", "top1_mean", "top1_std"]
        grp = grp.sort_values("tau_mean", ascending=False)
        out.append(grp.to_csv(index=False))
        return "\n".join(out)

    # Per experiment
    for exp_id in sorted(all_results["exp"].unique()):
        df_e = all_results[all_results["exp"] == exp_id]
        lines.append(_fmt_exp(exp_id, df_e))

    # Summary ranking
    lines.append("\n## Summary — All interventions ranked by τ gain\n")
    summary_rows = []
    for exp_id in sorted(all_results["exp"].unique()):
        df_e = all_results[all_results["exp"] == exp_id]
        for label, grp in df_e.groupby("label"):
            tau_m = grp["tau"].mean()
            tau_s = grp["tau"].std()
            top1_m = grp["top1"].mean()
            summary_rows.append({
                "exp":        exp_id,
                "label":      label,
                "tau_mean":   tau_m,
                "tau_std":    tau_s,
                "tau_gain":   tau_m - BASELINE_TAU,
                "top1_mean":  top1_m,
                "top1_gain":  BASELINE_TOP1 - top1_m,  # lower top1 is better
            })
    sdf = pd.DataFrame(summary_rows).sort_values("tau_gain", ascending=False)
    lines.append(sdf.to_csv(index=False))

    # Final recommendation
    best_row = sdf.iloc[0]
    lines.append(f"\n## Recommendation\n")
    lines.append(f"Best intervention: **{best_row['label']}** (exp={best_row['exp']})")
    lines.append(f"  τ = {best_row['tau_mean']:.4f} ± {best_row['tau_std']:.4f}")
    lines.append(f"  τ gain vs baseline: +{best_row['tau_gain']:+.4f}")
    lines.append(f"  top1 RMSD gain: {best_row['top1_gain']:+.3f} Å")
    lines.append("")
    lines.append("### Answer: If only ONE campaign is possible,")

    top5 = sdf.head(5)
    winner = top5.sort_values("tau_gain", ascending=False).iloc[0]
    exp_winner = winner["exp"]
    if "A_data" in exp_winner:
        verdict = ("DATA SCALING: collect more bench300-style complexes. "
                   f"Expected τ ≈ {winner['tau_mean']:.3f} "
                   f"(+{winner['tau_gain']:+.3f} vs baseline).")
    elif "B_vary" in exp_winner:
        verdict = ("COMPLEX DIVERSITY: the bottleneck is number of distinct complexes, not pair count. "
                   f"Expected τ ≈ {winner['tau_mean']:.3f}.")
    elif "C_spec" in exp_winner:
        verdict = ("SPECIALIST ROUTING: train separate models per peptide length bucket. "
                   f"Expected τ ≈ {winner['tau_mean']:.3f}.")
    elif "D_mix" in exp_winner:
        verdict = (f"MIXTURE TUNING at ratio {winner['label']}: "
                   f"Expected τ ≈ {winner['tau_mean']:.3f}.")
    elif "E_fine" in exp_winner:
        verdict = ("ENCODER FINETUNING: the features are the bottleneck. "
                   "Unfreeze last cross_conv block(s) and train end-to-end. "
                   f"Expected τ ≈ {winner['tau_mean']:.3f}.")
    elif "F_router" in exp_winner:
        verdict = ("ROUTING: deploy multi-model router (5-fold validated). "
                   f"Expected τ ≈ {winner['tau_mean']:.3f}.")
    else:
        verdict = f"run {exp_winner}/{winner['label']}. Expected τ ≈ {winner['tau_mean']:.3f}."

    lines.append(f"  → **{verdict}**")
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",   default="cuda")
    ap.add_argument("--skip-e",   action="store_true",
                    help="Skip encoder finetuning (Exp E)")
    ap.add_argument("--only-e",   action="store_true",
                    help="Skip A-F, load interim CSV, run only Exp E then merge")
    ap.add_argument("--workers",  type=int, default=8,
                    help="Parallel workers for head training")
    ap.add_argument("--epochs",   type=int, default=50)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    log.info("Device: %s | workers: %d", device, args.workers)

    # ── load data ─────────────────────────────────────────────────────────────
    log.info("Loading feature caches...")
    with open(FEAT_BENCH, "rb") as f:
        bench_feats = pickle.load(f)
    with open(FEAT_GEN, "rb") as f:
        gen_feats = pickle.load(f)

    bench_json = json.load(open(BENCH_JSON))
    gen_json   = json.load(open(GEN_JSON))
    meta       = load_meta(BENCH_CSV)

    bench_ds   = build_dataset(bench_feats, bench_json)
    gen_ds     = build_dataset(gen_feats,   gen_json)
    bench_all  = sorted(bench_ds.keys())
    _, bench_val = split_complexes(bench_all, 0.85, seed=42)

    log.info("bench300: %d train+val complexes (%d val)",
             len(bench_all), len(bench_val))
    log.info("gen_ood:  %d complexes", len(gen_ds))

    rows: list[dict] = []

    if args.only_e:
        # Load A-F results from previously saved interim CSV
        interim_path = OUT / "all_results_interim.csv"
        if interim_path.exists():
            rows = pd.read_csv(interim_path).to_dict("records")
            log.info("Loaded %d A-F rows from interim CSV", len(rows))
        else:
            log.warning("--only-e set but no interim CSV found; A-F results will be missing")
    else:
        # ── build all specs ───────────────────────────────────────────────────
        all_specs: list[dict] = []
        all_specs += build_exp_a_specs(bench_ds, bench_val)
        all_specs += build_exp_b_specs(bench_ds, bench_val)
        all_specs += build_exp_c_specs(bench_ds, bench_val, meta)
        all_specs += build_exp_d_specs(bench_ds, gen_ds, bench_val)
        all_specs += build_exp_f_specs(bench_ds, gen_ds, meta)

        log.info("Total head-training jobs: %d", len(all_specs))

        # ── run in parallel ───────────────────────────────────────────────────
        completed = 0
        n_total   = len(all_specs)

        # spawn avoids CUDA-context inheritance that breaks forked subprocesses
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp_ctx) as pool:
            futures = {pool.submit(_worker, spec): spec for spec in all_specs}
            for fut in as_completed(futures):
                try:
                    rows.append(fut.result())
                except Exception as exc:
                    spec = futures[fut]
                    log.error("FAILED %s/%s seed=%s: %s",
                              spec["exp"], spec["label"], spec["seed"], exc)
                completed += 1
                if completed % 20 == 0:
                    log.info("  Progress: %d / %d", completed, n_total)

        log.info("All head-training jobs done. %d results.", len(rows))

        # Intermediate save — protect A–F results before GPU Exp E
        pd.DataFrame(rows).to_csv(OUT / "all_results_interim.csv", index=False)
        log.info("Interim results saved (%d rows)", len(rows))

    # ── Exp E (main process, GPU) ─────────────────────────────────────────────
    if not args.skip_e:
        e_rows = run_exp_e(bench_ds, bench_val, device)
        rows.extend(e_rows)

    # ── save all results ──────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "all_results.csv", index=False)
    log.info("Saved: %s", OUT / "all_results.csv")

    # Per-exp CSVs
    for exp_id in df["exp"].unique():
        df[df["exp"] == exp_id].to_csv(OUT / f"{exp_id}.csv", index=False)

    # ── report ────────────────────────────────────────────────────────────────
    report = generate_report(df)
    (OUT / "campaign_report.md").write_text(report)
    log.info("Report: %s", OUT / "campaign_report.md")

    # Print top-10 results to stdout
    log.info("\n=== TOP 10 INTERVENTIONS BY τ ===")
    df_agg = df.groupby(["exp", "label"])["tau"].agg(["mean", "std"]).reset_index()
    df_agg.columns = ["exp", "label", "tau_mean", "tau_std"]
    for _, row in df_agg.sort_values("tau_mean", ascending=False).head(10).iterrows():
        log.info("  %s / %-30s  τ=%.4f ± %.4f",
                 row["exp"], row["label"], row["tau_mean"], row["tau_std"])


if __name__ == "__main__":
    main()
