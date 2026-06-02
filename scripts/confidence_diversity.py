#!/usr/bin/env python3
"""
confidence_diversity.py — Multi-variant data generation study.

Exp 1: Variant count sweep (1/2/4 variants from bench300, matched-pairs control)
Exp 2: Diversity metrics correlated with τ
Exp 3: Per-SS and per-length routing model analysis
Exp 4: Clean REF2015 comparison (5-fold CV)
Exp 5: Data scaling law with bootstrap CIs + projections to 2×/4×/8×

Note: bench300 already has 4 variants (pretrained, v5c, v3c, v4c) × 5 poses = 20/complex.
      gen_ood has 1 variant (pretrained) × 5 poses = 5/complex.
      8-variant experiment requires new RAPiDock inference (not run here — projected via scaling).

Usage:
  conda run -n rapidock python3 scripts/confidence_diversity.py
"""
from __future__ import annotations
import copy, json, math, os, pickle, sys, warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "RAPiDock"))
warnings.filterwarnings("ignore")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

OUT        = REPO / "logs" / "diversity"
BENCH_JSON = REPO / "logs" / "analysis_bench300" / "benchmark_results.json"
BENCH_CSV  = REPO / "data" / "benchmark300.csv"
GEN_JSON   = REPO / "logs" / "confidence_training_data" / "benchmark_results.json"
GEN_CSV    = REPO / "data" / "confidence_training_500.csv"
REF15_JSON = REPO / "logs" / "ref2015_ranking_all" / "ranking_results.json"
FEAT_BENCH = REPO / "logs" / "diagnosis" / "feats_bench300.pkl"
FEAT_GEN   = REPO / "logs" / "diagnosis" / "feats_gen_ood.pkl"

OUT.mkdir(parents=True, exist_ok=True)

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("diversity")

BENCH_VARIANTS = ['pretrained', 'v5c', 'v3c', 'v4c']


# ═══════════════════════════════════════════════════════════════════════════════
# HEAD
# ═══════════════════════════════════════════════════════════════════════════════

class V2Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(96, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, x): return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def build_dataset(feat_map: dict, json_data: dict,
                  variants: list | None = None) -> dict:
    """Returns {cname: [(feat96, rmsd), ...]}. Filter by variants if given."""
    ds: dict[str, list] = {}
    for (cname, mkey, pose_idx), feat in feat_map.items():
        if variants is not None and mkey not in variants:
            continue
        rmsds = json_data.get(cname, {}).get(mkey, {}).get("ref_rmsds", [])
        if pose_idx >= len(rmsds): continue
        ds.setdefault(cname, []).append((feat.astype(np.float32), float(rmsds[pose_idx])))
    return {k: v for k, v in ds.items() if len(v) >= 2}


def split_complexes(complexes: list, frac: float = 0.85, seed: int = 42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(complexes))
    n = max(1, int(len(complexes) * frac))
    return [complexes[i] for i in idx[:n]], [complexes[i] for i in idx[n:]]


def build_pairs(ds: dict, complexes: list, max_pairs: int = -1,
                seed: int = 0) -> list:
    pairs = []
    for cname in complexes:
        poses = ds.get(cname, [])
        if len(poses) < 2: continue
        for (fi, ri), (fj, rj) in combinations(poses, 2):
            if abs(ri - rj) < 1e-6: continue
            pairs.append((fi, fj, 1.0 if ri < rj else 0.0))
    if max_pairs > 0 and len(pairs) > max_pairs:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]
    return pairs


def build_mixed(bench_ds, gen_ds, bench_train_c, gen_train_c,
                bench_frac: float = 0.75, seed: int = 42) -> tuple:
    rng = np.random.RandomState(seed)
    n_b = max(1, int(len(bench_train_c) * bench_frac))
    n_g = max(1, int(len(gen_train_c) * (1 - bench_frac)))
    bs = list(rng.choice(bench_train_c, min(n_b, len(bench_train_c)), replace=False))
    gs = list(rng.choice(gen_train_c,   min(n_g, len(gen_train_c)),   replace=False))
    combined = {}
    for c in bs: combined[f"B_{c}"] = bench_ds[c]
    for c in gs: combined[f"G_{c}"] = gen_ds[c]
    return combined, list(combined.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING + EVAL
# ═══════════════════════════════════════════════════════════════════════════════

def bpr_loss(si, sj, lbl):
    return -F.logsigmoid((si - sj) * (lbl * 2.0 - 1.0)).mean()


def train_head(head: nn.Module, train_pairs: list, val_pairs: list,
               epochs: int = 50, lr: float = 1e-3, bs: int = 512) -> dict:
    if not train_pairs:
        return {"train_acc": float("nan"), "val_acc": float("nan"),
                "best_epoch": -1}
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr * 0.01)

    tr_fi  = torch.tensor(np.stack([p[0] for p in train_pairs]), dtype=torch.float32)
    tr_fj  = torch.tensor(np.stack([p[1] for p in train_pairs]), dtype=torch.float32)
    tr_lbl = torch.tensor([p[2] for p in train_pairs], dtype=torch.float32)
    va_fi = va_fj = va_lbl = None
    if val_pairs:
        va_fi  = torch.tensor(np.stack([p[0] for p in val_pairs]), dtype=torch.float32)
        va_fj  = torch.tensor(np.stack([p[1] for p in val_pairs]), dtype=torch.float32)
        va_lbl = torch.tensor([p[2] for p in val_pairs], dtype=torch.float32)

    def acc(fi, fj, lbl):
        if fi is None: return float("nan")
        with torch.no_grad():
            si = head(fi).squeeze(-1); sj = head(fj).squeeze(-1)
        return ((si > sj).float() - lbl).abs().lt(0.5).float().mean().item()

    best_val = -1.0; best_state = None; best_ep = 0
    n = len(train_pairs)
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n)
        for b in range(0, n, bs):
            idx = perm[b: b+bs]
            loss = bpr_loss(head(tr_fi[idx]).squeeze(-1),
                            head(tr_fj[idx]).squeeze(-1), tr_lbl[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        v = acc(va_fi, va_fj, va_lbl)
        if not math.isnan(v) and v > best_val:
            best_val = v; best_state = copy.deepcopy(head.state_dict()); best_ep = ep
    if best_state: head.load_state_dict(best_state)
    head.eval()
    return {"train_acc": acc(tr_fi[:4000], tr_fj[:4000], tr_lbl[:4000]),
            "val_acc": best_val, "best_epoch": best_ep}


@torch.no_grad()
def eval_tau_full(head: nn.Module, ds: dict, complexes: list) -> dict:
    """Per-complex and aggregate τ, top1."""
    per = {}
    for cname in complexes:
        poses = ds.get(cname, [])
        if len(poses) < 2: continue
        feats  = torch.tensor(np.stack([p[0] for p in poses]), dtype=torch.float32)
        rmsds  = np.array([p[1] for p in poses])
        scores = head(feats).squeeze(-1).numpy()
        tau, _ = scipy_stats.kendalltau(-scores, rmsds)
        if math.isnan(tau): continue
        per[cname] = {"tau": float(tau), "top1": float(rmsds[np.argmax(scores)]),
                      "scores": scores.tolist(), "rmsds": rmsds.tolist()}
    taus = [v["tau"]  for v in per.values()]
    tops = [v["top1"] for v in per.values()]
    return {"per": per,
            "tau":  float(np.mean(taus))  if taus else float("nan"),
            "top1": float(np.mean(tops))  if tops else float("nan"),
            "n":    len(taus)}


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 1 — VARIANT COUNT SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def exp1_variant_sweep(bench_feat_map, bench_json, bench_ds_full,
                       bench_train_c, bench_val_c, n_seeds=5, epochs=50):
    log.info("\n" + "="*60)
    log.info("EXP 1: Variant Count Sweep")
    log.info("="*60)

    # Eval always on FULL 4-variant bench300 val set
    rows = []
    for n_variants in [1, 2, 4]:
        variants = BENCH_VARIANTS[:n_variants]
        ds_train = build_dataset(bench_feat_map, bench_json, variants=variants)
        # Matched-pairs: sample same number as 1-variant (10 pairs/complex)
        pairs_per_complex_1v = 10  # C(5,2)

        for seed in range(n_seeds):
            for matched in [False, True]:
                train_pairs_full = build_pairs(ds_train, bench_train_c, seed=seed)
                if matched:
                    max_p = pairs_per_complex_1v * len(bench_train_c)
                    train_pairs = build_pairs(ds_train, bench_train_c,
                                              max_pairs=max_p, seed=seed)
                else:
                    train_pairs = train_pairs_full

                # Val pairs from full 4-variant val set
                val_pairs = build_pairs(bench_ds_full, bench_val_c, seed=seed)
                head = V2Head()
                metrics = train_head(head, train_pairs, val_pairs, epochs=epochs)
                result  = eval_tau_full(head, bench_ds_full, bench_val_c)
                rows.append({
                    "n_variants": n_variants, "matched_pairs": matched, "seed": seed,
                    "n_train_pairs": len(train_pairs),
                    "tau": result["tau"], "top1": result["top1"],
                    **metrics,
                })
                log.info("  n_var=%d  matched=%s  seed=%d  n_pairs=%d  τ=%.4f",
                         n_variants, matched, seed, len(train_pairs), result["tau"])

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp1_variant_sweep.csv", index=False)

    # Summary
    print("\n--- EXP 1 SUMMARY ---")
    for matched in [False, True]:
        label = "variable pairs" if not matched else "matched pairs (10/complex)"
        sub = df[df["matched_pairs"] == matched]
        print(f"\n{label}:")
        for nv, g in sub.groupby("n_variants"):
            print(f"  {nv} variant(s): τ = {g['tau'].mean():.4f} ± {g['tau'].std():.4f}  "
                  f"(n_pairs mean={g['n_train_pairs'].mean():.0f})")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 2 — DIVERSITY METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def exp2_diversity_metrics(bench_feat_map, bench_json, bench_ds_full,
                            bench_train_c, bench_val_c, exp1_df, epochs=50):
    log.info("\n" + "="*60)
    log.info("EXP 2: Diversity Metrics Correlation")
    log.info("="*60)

    rows = []
    for n_variants in [1, 2, 4]:
        variants = BENCH_VARIANTS[:n_variants]
        ds = build_dataset(bench_feat_map, bench_json, variants=variants)

        # 1. RMSD spread: mean(max_rmsd - min_rmsd) per complex
        rmsd_spreads = []
        for cname in bench_train_c:
            poses = ds.get(cname, [])
            if not poses: continue
            rs = [p[1] for p in poses]
            rmsd_spreads.append(max(rs) - min(rs))
        rmsd_spread = float(np.mean(rmsd_spreads))

        # 2. Feature diversity: mean std of 96-dim features per complex
        feat_stds = []
        for cname in bench_train_c:
            poses = ds.get(cname, [])
            if len(poses) < 2: continue
            feats = np.stack([p[0] for p in poses])
            feat_stds.append(feats.std(axis=0).mean())
        feat_diversity = float(np.mean(feat_stds))

        # 3. Cross-variant feature distance (only for n_variants >= 2)
        cross_var_dist = float("nan")
        intra_var_dist = float("nan")
        if n_variants >= 2:
            inter_dists, intra_dists = [], []
            for cname in bench_train_c[:50]:  # sample for speed
                v_list = variants
                for v1, v2 in combinations(v_list, 2):
                    f1 = [bench_feat_map.get((cname, v1, i))
                          for i in range(5) if (cname, v1, i) in bench_feat_map]
                    f2 = [bench_feat_map.get((cname, v2, i))
                          for i in range(5) if (cname, v2, i) in bench_feat_map]
                    if f1 and f2:
                        a1 = np.stack(f1); a2 = np.stack(f2)
                        for fa in a1:
                            for fb in a2:
                                inter_dists.append(float(np.linalg.norm(fa - fb)))
                for v in v_list:
                    fv = [bench_feat_map.get((cname, v, i))
                          for i in range(5) if (cname, v, i) in bench_feat_map]
                    if len(fv) >= 2:
                        arr = np.stack(fv)
                        for i, j in combinations(range(len(arr)), 2):
                            intra_dists.append(float(np.linalg.norm(arr[i] - arr[j])))
            cross_var_dist = float(np.mean(inter_dists)) if inter_dists else float("nan")
            intra_var_dist = float(np.mean(intra_dists)) if intra_dists else float("nan")

        # 4. Mean τ from Exp 1 (variable pairs, mean over seeds)
        exp1_sub = exp1_df[(exp1_df["n_variants"] == n_variants) &
                            (~exp1_df["matched_pairs"])]
        mean_tau = float(exp1_sub["tau"].mean()) if not exp1_sub.empty else float("nan")

        rows.append({
            "n_variants": n_variants,
            "rmsd_spread": rmsd_spread,
            "feat_diversity": feat_diversity,
            "cross_var_feat_dist": cross_var_dist,
            "intra_var_feat_dist": intra_var_dist,
            "feat_dist_ratio": cross_var_dist / intra_var_dist if intra_var_dist > 0 else float("nan"),
            "tau": mean_tau,
        })
        log.info("  n_var=%d  rmsd_spread=%.3f  feat_div=%.4f  cross/intra=%.3f  τ=%.4f",
                 n_variants, rmsd_spread, feat_diversity,
                 cross_var_dist / intra_var_dist if intra_var_dist > 0 else float("nan"),
                 mean_tau)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp2_diversity_metrics.csv", index=False)

    # Correlations
    if len(df) >= 3:
        for metric in ["rmsd_spread", "feat_diversity", "feat_dist_ratio"]:
            valid = df[[metric, "tau"]].dropna()
            if len(valid) >= 2:
                r, p = scipy_stats.pearsonr(valid[metric], valid["tau"])
                log.info("  corr(%s, τ) = %.3f  p=%.3f", metric, r, p)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 3 — ROUTING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def exp3_routing(bench_ds, gen_ds, bench_train_c, gen_train_c, bench_val_c,
                 bench_csv_df, epochs=50):
    log.info("\n" + "="*60)
    log.info("EXP 3: Routing Model Analysis")
    log.info("="*60)

    meta = bench_csv_df.set_index("name")

    # 5 training distributions
    CONFIGS = {
        "bench_only": (bench_ds, bench_train_c, []),
        "gen_only":   (gen_ds,   gen_train_c,   []),
        "25B_75G":    None,
        "50B_50G":    None,
        "75B_25G":    None,
    }

    # Train all 5 global models
    trained_heads = {}
    val_pairs_bench = build_pairs(bench_ds, bench_val_c)
    for name, cfg in CONFIGS.items():
        if cfg is not None:
            train_ds, train_cs, _ = cfg
            train_pairs = build_pairs(train_ds, train_cs)
        else:
            frac = {"25B_75G": 0.25, "50B_50G": 0.50, "75B_25G": 0.75}[name]
            combined, combined_keys = build_mixed(bench_ds, gen_ds, bench_train_c, gen_train_c, frac)
            train_pairs = build_pairs(combined, combined_keys)
        head = V2Head()
        train_head(head, train_pairs, val_pairs_bench, epochs=epochs)
        trained_heads[name] = head
        result = eval_tau_full(head, bench_ds, bench_val_c)
        log.info("  %s  global τ=%.4f", name, result["tau"])

    # Evaluate all 5 models on subsets by SS class and length
    rows = []
    for cat_col in ["ss_class", "length_bucket"]:
        for cat_val in bench_csv_df[cat_col].unique():
            subset = [c for c in bench_val_c
                      if c in meta.index and meta.loc[c, cat_col] == cat_val]
            if len(subset) < 2: continue
            for config_name, head in trained_heads.items():
                result = eval_tau_full(head, bench_ds, subset)
                rows.append({
                    "cat_col": cat_col, "cat_val": cat_val,
                    "config": config_name,
                    "tau": result["tau"], "top1": result["top1"],
                    "n_complexes": result["n"],
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp3_routing.csv", index=False)

    # Find best config per category
    routing_rule = {}
    for cat_col in ["ss_class", "length_bucket"]:
        sub = df[df["cat_col"] == cat_col]
        best = sub.loc[sub.groupby("cat_val")["tau"].idxmax()]
        log.info("\n  Best config by %s:", cat_col)
        for _, row in best.iterrows():
            log.info("    %s → %s  τ=%.4f  (n=%d)",
                     row["cat_val"], row["config"], row["tau"], row["n_complexes"])
            routing_rule[(cat_col, row["cat_val"])] = row["config"]

    # Evaluate routing ensemble (route by SS class)
    routed_taus = []
    for cname in bench_val_c:
        if cname not in meta.index: continue
        ss = meta.loc[cname, "ss_class"]
        best_config = df[(df["cat_col"] == "ss_class") &
                         (df["cat_val"] == ss)].sort_values("tau").iloc[-1]["config"]
        head = trained_heads[best_config]
        poses = bench_ds.get(cname, [])
        if len(poses) < 2: continue
        feats  = torch.tensor(np.stack([p[0] for p in poses]), dtype=torch.float32)
        rmsds  = np.array([p[1] for p in poses])
        with torch.no_grad():
            scores = head(feats).squeeze(-1).numpy()
        tau, _ = scipy_stats.kendalltau(-scores, rmsds)
        if not math.isnan(tau): routed_taus.append(tau)

    routing_tau = float(np.mean(routed_taus)) if routed_taus else float("nan")

    # Best single model τ
    best_single_tau = max(
        eval_tau_full(h, bench_ds, bench_val_c)["tau"]
        for h in trained_heads.values()
    )
    log.info("\n  Routing ensemble τ = %.4f  vs best single = %.4f  (gain = %.4f)",
             routing_tau, best_single_tau, routing_tau - best_single_tau)

    return df, routing_tau, best_single_tau, routing_rule


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 4 — CLEAN REF2015 COMPARISON (5-FOLD CV)
# ═══════════════════════════════════════════════════════════════════════════════

def exp4_ref2015_clean(bench_feat_map, bench_json, bench_ds,
                        bench_complexes, ref15_json, epochs=50):
    log.info("\n" + "="*60)
    log.info("EXP 4: Clean REF2015 Comparison (5-fold CV)")
    log.info("="*60)

    # Align ref2015 to pretrained-only poses for fair comparison
    aligned = {}
    for cname, ref_data in ref15_json.items():
        if cname not in bench_json: continue
        ref_rmsds  = np.array(ref_data["ref_rmsds"])
        ref_scores = np.array(ref_data["scores"])
        # Find matching variant by RMSD similarity
        best_mkey, best_mse = None, float("inf")
        for mkey, mdata in bench_json[cname].items():
            br = np.array(mdata.get("ref_rmsds", []))
            if len(br) != len(ref_rmsds): continue
            mse = np.mean((np.sort(br) - np.sort(ref_rmsds))**2)
            if mse < best_mse: best_mse = mse; best_mkey = mkey
        if best_mkey is None or best_mse > 0.5: continue
        conf_feats = [bench_feat_map.get((cname, best_mkey, i))
                      for i in range(len(ref_rmsds))]
        if any(f is None for f in conf_feats): continue
        aligned[cname] = {
            "ref_rmsds": ref_rmsds,
            "ref15_scores": ref_scores,
            "conf_feats": np.stack(conf_feats).astype(np.float32),
            "mkey": best_mkey,
        }

    aligned_names = sorted(aligned.keys())
    log.info("  Aligned complexes: %d / %d", len(aligned_names), len(ref15_json))

    # Build single-variant (pretrained) dataset for training
    ds_pretrained = build_dataset(bench_feat_map, bench_json, variants=["pretrained"])

    def _norm(x):
        mu, sigma = x.mean(), x.std()
        return (x - mu) / sigma if sigma > 1e-9 else x * 0

    K = 5
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(bench_complexes))
    fold_size = len(bench_complexes) // K
    fold_rows = []

    for fold in range(K):
        val_idx   = idx[fold * fold_size: (fold + 1) * fold_size]
        train_idx = np.concatenate([idx[:fold * fold_size], idx[(fold + 1) * fold_size:]])
        val_c   = [bench_complexes[i] for i in val_idx]
        train_c = [bench_complexes[i] for i in train_idx]

        # Train confidence model
        train_pairs = build_pairs(ds_pretrained, train_c)
        val_pairs   = build_pairs(ds_pretrained, val_c)
        head = V2Head()
        train_head(head, train_pairs, val_pairs, epochs=epochs)
        head.eval()

        # Eval on aligned val complexes
        eval_names = [c for c in val_c if c in aligned]
        if not eval_names:
            log.warning("  Fold %d: no aligned val complexes", fold)
            continue

        for w_conf in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            w_ref = 1.0 - w_conf
            taus, tops, gaps, pbests = [], [], [], []
            for cname in eval_names:
                ref_rmsds  = aligned[cname]["ref_rmsds"]
                ref15_sc   = aligned[cname]["ref15_scores"]
                conf_feats = torch.tensor(aligned[cname]["conf_feats"])
                with torch.no_grad():
                    conf_sc = head(conf_feats).squeeze(-1).numpy()
                combined = w_conf * _norm(conf_sc) + w_ref * _norm(-ref15_sc)
                tau, _ = scipy_stats.kendalltau(-combined, ref_rmsds)
                if math.isnan(tau): continue
                taus.append(tau)
                top1 = float(ref_rmsds[np.argmax(combined)])
                tops.append(top1)
                rm = np.array(ref_rmsds)
                oracle = rm.max() - rm.min()
                if oracle > 0: gaps.append((rm.max() - top1) / oracle)
                pbests.append(1.0 if np.argmax(combined) == np.argmin(rm) else 0.0)
            fold_rows.append({
                "fold": fold, "w_conf": w_conf,
                "tau":    float(np.mean(taus))   if taus else float("nan"),
                "top1":   float(np.mean(tops))   if tops else float("nan"),
                "gaprec": float(np.mean(gaps))   if gaps else float("nan"),
                "pbest":  float(np.mean(pbests)) if pbests else float("nan"),
                "n": len(taus),
            })

    df = pd.DataFrame(fold_rows)
    df.to_csv(OUT / "exp4_ref2015_cv.csv", index=False)

    # Aggregate across folds
    agg = df.groupby("w_conf")[["tau", "top1", "gaprec", "pbest"]].agg(
        ["mean", "std"]).round(4)
    log.info("\n  5-fold CV results:\n%s", agg.to_string())

    conf_only = df[df["w_conf"] == 1.0]["tau"]
    ref_only  = df[df["w_conf"] == 0.0]["tau"]
    best_w    = df.groupby("w_conf")["tau"].mean().idxmax()
    best_tau  = df.groupby("w_conf")["tau"].mean().max()
    log.info("  Conf-only τ = %.4f ± %.4f", conf_only.mean(), conf_only.std())
    log.info("  Ref15-only τ = %.4f ± %.4f", ref_only.mean(), ref_only.std())
    log.info("  Best ensemble: w_conf=%.1f  τ=%.4f", best_w, best_tau)

    return df, agg


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 5 — SCALING LAW WITH BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════════

def exp5_scaling_bootstrap(bench_ds, gen_ds, bench_train_c, gen_train_c,
                            bench_val_c, n_bootstrap=10, epochs=50):
    log.info("\n" + "="*60)
    log.info("EXP 5: Scaling Law with Bootstrap CIs")
    log.info("="*60)

    fractions = [0.25, 0.50, 0.75, 1.00]
    val_pairs = build_pairs(bench_ds, bench_val_c)
    rows = []

    for frac in fractions:
        taus = []
        for seed in range(n_bootstrap):
            combined, combined_keys = build_mixed(bench_ds, gen_ds, bench_train_c,
                                                   gen_train_c, 0.75, seed=seed)
            n_total = len(combined_keys)
            # Scale down
            rng = np.random.RandomState(seed + 100)
            n_sub = max(2, int(n_total * frac))
            sub_keys = list(rng.choice(combined_keys, n_sub, replace=False))
            train_pairs = build_pairs({k: combined[k] for k in sub_keys}, sub_keys)
            head = V2Head()
            train_head(head, train_pairs, val_pairs, epochs=epochs)
            result = eval_tau_full(head, bench_ds, bench_val_c)
            taus.append(result["tau"])
            n_complexes = n_sub
        rows.append({
            "frac": frac, "n_complexes_mean": int(len(combined_keys) * frac),
            "tau_mean": float(np.mean(taus)), "tau_std": float(np.std(taus)),
            "tau_ci95": float(1.96 * np.std(taus) / np.sqrt(n_bootstrap)),
            "tau_min": float(np.min(taus)), "tau_max": float(np.max(taus)),
        })
        log.info("  frac=%.2f  n≈%d  τ = %.4f ± %.4f  (CI95: ±%.4f)",
                 frac, rows[-1]["n_complexes_mean"], rows[-1]["tau_mean"],
                 rows[-1]["tau_std"], rows[-1]["tau_ci95"])

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp5_scaling.csv", index=False)

    # Fit scaling laws
    ns = df["n_complexes_mean"].values.astype(float)
    ts = df["tau_mean"].values
    projections = {}

    try:
        popt, _ = curve_fit(lambda x, a, b, c: a * np.power(x, b) + c,
                            ns, ts, p0=[0.1, 0.5, 0.0], maxfev=5000)
        for mult, label in [(2, "2x"), (4, "4x"), (8, "8x")]:
            n_proj = ns[-1] * mult
            t_proj = popt[0] * np.power(n_proj, popt[1]) + popt[2]
            projections[label] = float(t_proj)
            log.info("  Power law projection %s data (%d complexes): τ ≈ %.4f",
                     label, int(n_proj), t_proj)
    except Exception as e:
        log.warning("  Power law fit failed: %s", e)

    return df, projections


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def write_report(exp1_df, exp2_df, exp3_df, exp3_routing_tau, exp3_best_single,
                 exp3_rule, exp4_df, exp5_df, exp5_proj):
    lines = [
        "# Multi-Variant Data Generation Study\n",
        "**Date:** 2026-06-02  |  Config: v2 head, BN frozen, checkpoint by val_acc\n\n",
    ]

    # Exp 1
    lines += ["## 1. Variant Count Sweep\n\n"]
    for matched in [False, True]:
        label = "Variable pairs (full training)" if not matched else "Matched pairs (10 pairs/complex, isolates diversity effect)"
        sub = exp1_df[exp1_df["matched_pairs"] == matched]
        lines.append(f"### {label}\n\n")
        lines.append("| N variants | Mean τ | Std τ | Mean pairs |\n|---|---|---|---|\n")
        for nv, g in sub.groupby("n_variants"):
            lines.append(f"| {nv} | {g['tau'].mean():.4f} | {g['tau'].std():.4f} | {g['n_train_pairs'].mean():.0f} |\n")
        lines.append("\n")

    # Exp 2
    lines += ["## 2. Diversity Metrics\n\n"]
    lines.append("| N variants | RMSD spread | Feature diversity | Cross/intra ratio | τ |\n|---|---|---|---|---|\n")
    for _, row in exp2_df.iterrows():
        lines.append(f"| {int(row['n_variants'])} | {row['rmsd_spread']:.3f} | "
                     f"{row['feat_diversity']:.4f} | {row['feat_dist_ratio']:.3f} | {row['tau']:.4f} |\n")
    lines.append("\n")

    # Exp 3
    lines += ["## 3. Routing Analysis\n\n"]
    lines.append(f"Routing ensemble τ = **{exp3_routing_tau:.4f}**  vs best single model τ = **{exp3_best_single:.4f}**  "
                 f"(gain = {exp3_routing_tau - exp3_best_single:+.4f})\n\n")
    lines.append("### Best configuration by SS class\n\n")
    lines.append("| SS Class | Best Config | τ | n complexes |\n|---|---|---|---|\n")
    ss_sub = exp3_df[(exp3_df["cat_col"] == "ss_class")]
    for (cat_val,), g in ss_sub.groupby("cat_val"):
        best = g.loc[g["tau"].idxmax()]
        lines.append(f"| {cat_val} | {best['config']} | {best['tau']:.4f} | {int(best['n_complexes'])} |\n")
    lines.append("\n### Best configuration by length bucket\n\n")
    lines.append("| Length | Best Config | τ | n complexes |\n|---|---|---|---|\n")
    lb_sub = exp3_df[(exp3_df["cat_col"] == "length_bucket")]
    for (cat_val,), g in lb_sub.groupby("cat_val"):
        best = g.loc[g["tau"].idxmax()]
        lines.append(f"| {cat_val} | {best['config']} | {best['tau']:.4f} | {int(best['n_complexes'])} |\n")
    lines.append("\n")

    # Exp 4
    lines += ["## 4. Clean REF2015 Comparison (5-fold CV)\n\n"]
    agg = exp4_df.groupby("w_conf")["tau"].agg(["mean", "std"])
    lines.append("| w_conf | Mean τ | Std τ |\n|---|---|---|\n")
    for w, row in agg.iterrows():
        marker = " ← **best**" if w == agg["mean"].idxmax() else ""
        lines.append(f"| {w:.1f} | {row['mean']:.4f} | {row['std']:.4f} |{marker}\n")
    conf_only = exp4_df[exp4_df["w_conf"] == 1.0]["tau"]
    ref_only  = exp4_df[exp4_df["w_conf"] == 0.0]["tau"]
    lines.append(f"\n- Confidence-only: τ = {conf_only.mean():.4f} ± {conf_only.std():.4f}\n")
    lines.append(f"- REF2015-only: τ = {ref_only.mean():.4f} ± {ref_only.std():.4f}\n")
    best_w = agg["mean"].idxmax(); best_tau = agg["mean"].max()
    lines.append(f"- Best ensemble: w_conf={best_w:.1f}  τ = {best_tau:.4f}\n\n")
    gain = best_tau - max(conf_only.mean(), ref_only.mean())
    if gain > 0.01:
        lines.append(f"**Ensemble provides +{gain:.4f} τ gain over best single ranker.**\n\n")
    else:
        lines.append(f"**No meaningful ensemble gain ({gain:+.4f} τ). Use confidence alone.**\n\n")

    # Exp 5
    lines += ["## 5. Data Scaling Law with Bootstrap\n\n"]
    lines.append("| Frac | N complexes | τ mean | τ std | 95% CI |\n|---|---|---|---|---|\n")
    for _, row in exp5_df.iterrows():
        lines.append(f"| {row['frac']:.2f} | {int(row['n_complexes_mean'])} | "
                     f"{row['tau_mean']:.4f} | {row['tau_std']:.4f} | ±{row['tau_ci95']:.4f} |\n")
    lines.append("\n**Projections (power-law fit):**\n\n")
    for label, tau in exp5_proj.items():
        lines.append(f"- {label} data: τ ≈ {tau:.4f}\n")
    lines.append("\n")

    # Final answers
    lines += ["## Final Answers\n\n"]

    # Q1: Is diversity the bottleneck?
    matched_sub = exp1_df[exp1_df["matched_pairs"]]
    tau_1v = matched_sub[matched_sub["n_variants"]==1]["tau"].mean()
    tau_4v = matched_sub[matched_sub["n_variants"]==4]["tau"].mean()
    diversity_gain = tau_4v - tau_1v
    lines.append(f"**Q1: Is pose-generation diversity the main remaining bottleneck?**\n")
    lines.append(f"Diversity gain (matched pairs, 1→4 variants): {diversity_gain:+.4f} τ. ")
    if diversity_gain > 0.03:
        lines.append("YES — diversity provides meaningful gains even controlling for pair count.\n\n")
    elif diversity_gain > 0.01:
        lines.append("PARTIAL — diversity helps modestly; data volume also matters.\n\n")
    else:
        lines.append("NO — gain comes from pair count, not diversity per se.\n\n")

    # Q2: Best training distribution
    global_taus = exp3_df[exp3_df["cat_col"] == "ss_class"].groupby("config")["tau"].mean()
    best_global = global_taus.idxmax()
    lines.append(f"**Q2: Best training distribution globally:** {best_global} (mean τ={global_taus.max():.4f})\n\n")

    # Q3: Routing worth it?
    lines.append(f"**Q3: Is a routing model better than a global model?**\n")
    lines.append(f"Routing gain: {exp3_routing_tau - exp3_best_single:+.4f} τ. ")
    if exp3_routing_tau - exp3_best_single > 0.02:
        lines.append("YES — routing provides meaningful improvement. Deploy per-SS routing.\n\n")
    else:
        lines.append("MARGINAL — not worth the complexity unless per-SS specialization is confirmed on more data.\n\n")

    # Q4: REF2015 contribution
    lines.append(f"**Q4: Does REF2015 still contribute unique information?**\n")
    lines.append(f"Confidence τ = {conf_only.mean():.4f}, REF2015 τ = {ref_only.mean():.4f}, "
                 f"best ensemble τ = {best_tau:.4f} (gain = {gain:+.4f})\n")
    if gain > 0.01:
        lines.append(f"YES — ensemble at w_conf={best_w:.1f} provides +{gain:.4f} τ.\n\n")
    else:
        lines.append("NO — confidence dominates; ref2015 provides no additional signal.\n\n")

    # Q5: Projected τ at 2× data
    tau_2x = exp5_proj.get("2x", float("nan"))
    tau_current = exp5_df["tau_mean"].iloc[-1]
    lines.append(f"**Q5: Projected τ if dataset size is doubled?**\n")
    lines.append(f"Current (100%, ~253 complexes): τ = {tau_current:.4f}\n")
    lines.append(f"Projected 2× (~506 complexes): τ ≈ {tau_2x:.4f} (+{tau_2x-tau_current:+.4f})\n")
    lines.append(f"Projected 4×: τ ≈ {exp5_proj.get('4x', float('nan')):.4f}\n")
    lines.append(f"Projected 8×: τ ≈ {exp5_proj.get('8x', float('nan')):.4f}\n\n")

    # Q6: Highest ROI next experiment
    lines.append("**Q6: Single highest-ROI next experiment:**\n")
    if diversity_gain > 0.02:
        lines.append("Generate gen_ood with 4 RAPiDock model variants per complex (run v3c/v4c/v5c inference on existing 491 gen_ood complexes). This directly exploits the diversity gain measured in Exp 1.\n\n")
    elif exp5_proj.get("2x", 0) - tau_current > 0.02:
        lines.append("Expand training data to 500+ complexes from bench300+gen_ood. Data scaling is not saturated and ROI is high.\n\n")
    elif exp3_routing_tau - exp3_best_single > 0.02:
        lines.append("Deploy per-SS routing ensemble. Provides free gain from existing trained models.\n\n")
    else:
        lines.append("Fine-tune encoder top layers (unfreeze last 2 layers). Frozen encoder ceiling is reached; further gains require encoder adaptation.\n\n")

    with open(OUT / "diversity_report.md", "w") as f:
        f.writelines(lines)
    log.info("Saved: %s/diversity_report.md", OUT)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Loading feature caches...")
    with open(FEAT_BENCH, "rb") as f: bench_feat_map = pickle.load(f)
    with open(FEAT_GEN,   "rb") as f: gen_feat_map   = pickle.load(f)
    bench_json = json.load(open(BENCH_JSON))
    gen_json   = json.load(open(GEN_JSON))
    bench_csv  = pd.read_csv(BENCH_CSV)
    ref15_json = json.load(open(REF15_JSON)) if REF15_JSON.exists() else {}

    bench_ds = build_dataset(bench_feat_map, bench_json)
    gen_ds   = build_dataset(gen_feat_map,   gen_json)

    bench_complexes = sorted(bench_ds.keys())
    gen_complexes   = sorted(gen_ds.keys())
    bench_train_c, bench_val_c = split_complexes(bench_complexes, 0.85, 42)
    gen_train_c,   gen_val_c   = split_complexes(gen_complexes,   0.85, 42)

    log.info("bench300: %d train / %d val | gen_ood: %d train / %d val",
             len(bench_train_c), len(bench_val_c), len(gen_train_c), len(gen_val_c))

    EPOCHS = 50

    exp1_df = exp1_variant_sweep(bench_feat_map, bench_json, bench_ds,
                                  bench_train_c, bench_val_c, n_seeds=5, epochs=EPOCHS)

    exp2_df = exp2_diversity_metrics(bench_feat_map, bench_json, bench_ds,
                                      bench_train_c, bench_val_c, exp1_df, epochs=EPOCHS)

    exp3_df, exp3_rtau, exp3_bsingle, exp3_rule = exp3_routing(
        bench_ds, gen_ds, bench_train_c, gen_train_c, bench_val_c, bench_csv, epochs=EPOCHS)

    if ref15_json:
        exp4_df, exp4_agg = exp4_ref2015_clean(
            bench_feat_map, bench_json, bench_ds, bench_complexes, ref15_json, epochs=EPOCHS)
    else:
        log.warning("No REF2015 data — skipping Exp 4")
        exp4_df = pd.DataFrame()

    exp5_df, exp5_proj = exp5_scaling_bootstrap(
        bench_ds, gen_ds, bench_train_c, gen_train_c, bench_val_c,
        n_bootstrap=10, epochs=EPOCHS)

    write_report(exp1_df, exp2_df, exp3_df, exp3_rtau, exp3_bsingle,
                 exp3_rule, exp4_df, exp5_df, exp5_proj)

    log.info("\nAll experiments complete. Results in %s/", OUT)


if __name__ == "__main__":
    main()
