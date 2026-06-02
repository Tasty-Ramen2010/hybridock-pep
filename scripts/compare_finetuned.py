#!/usr/bin/env python3
"""Post-training comparison: pretrained vs finetuned RAPiDock checkpoint.

Runs val_epoch on a held-out set with both checkpoints and reports:
  - Trimmed-mean val loss (checkpoint selection metric)
  - Raw mean, median (instability diagnosis)
  - Max per-sample loss (outlier presence)
  - Translation score norm distribution (instability predictor)
  - Best-of-N inference RMSD (if --run-inference is set, requires GPU)

Usage (in rapidock env):
    # Basic comparison: pretrained vs best Phase 3 checkpoint
    conda run -n rapidock python3 scripts/compare_finetuned.py \
        --pretrained  third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_local.pt \
        --finetuned   third_party/RAPiDock_finetuned/finetune_peppc_phase3/rapidock_finetuned_best.pt \
        --val-csv     datasets/training_formatted_peppc/combined_val_curated.csv \
        --out         logs/comparison_v1.json

    # Compare v1 vs v2 Phase 3
    conda run -n rapidock python3 scripts/compare_finetuned.py \
        --pretrained  .../rapidock_local.pt \
        --finetuned   .../finetune_peppc_v2_phase3/rapidock_finetuned_best.pt \
        --also-compare .../finetune_peppc_phase3/rapidock_finetuned_best.pt \
        --val-csv     datasets/training_formatted_peppc/combined_val_curated.csv \
        --out         logs/comparison_v1_vs_v2.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE / "third_party" / "RAPiDock_finetuned"))

from argparse import Namespace
from utils.utils import get_model, ExponentialMovingAverage
from utils.transform import NoiseTransform
from utils.inference_utils import InferenceDataset
import pandas as pd


# ---------------------------------------------------------------------------
# Re-use training utilities from train_lastlayer
# ---------------------------------------------------------------------------

sys.path.insert(0, str(_HERE / "third_party" / "RAPiDock_finetuned"))
from train_lastlayer import (
    build_dataset,
    compute_loss,
    _inject_batch_for_single_sample,
)


# ---------------------------------------------------------------------------
# Model loading (checkpoint-format aware)
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: str, model_args: Namespace,
                    device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint, applying EMA weights if available."""
    model = get_model(model_args, no_parallel=True)
    raw = torch.load(ckpt_path, map_location="cpu")

    if isinstance(raw, dict) and "model" in raw:
        model.load_state_dict(raw["model"], strict=True)
        if "ema_weights" in raw:
            try:
                ema = ExponentialMovingAverage(model.parameters(),
                                               decay=getattr(model_args, "ema_rate", 0.999))
                ema.load_state_dict(raw["ema_weights"], device=device)
                ema.copy_to(model.parameters())
                print(f"  [EMA weights applied from {Path(ckpt_path).name}]")
            except Exception as exc:
                print(f"  [WARN] EMA apply failed: {exc} — using raw weights")
    else:
        model.load_state_dict(raw, strict=True)

    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Robust val evaluation
# ---------------------------------------------------------------------------

def evaluate(model: torch.nn.Module, val_ds, val_indices: List[int],
             transform: NoiseTransform, device: torch.device) -> dict:
    """Evaluate checkpoint on val set; return dict of robust statistics."""
    model.eval()
    per_sample = []
    tr_norms   = []
    n_fail = 0

    with torch.no_grad():
        for idx in val_indices:
            try:
                data = val_ds.get(idx)
                _norm_out = {"tr": [], "rot": [], "tor_bb": [], "tor_sc": []}
                loss = compute_loss(model, data, transform, device,
                                    _norm_out=_norm_out)
                per_sample.append(loss.item())
                if _norm_out["tr"]:
                    tr_norms.append(_norm_out["tr"][0])
            except Exception as exc:
                n_fail += 1

    if not per_sample:
        return {"error": "all samples failed", "n_fail": n_fail}

    # Trimmed mean (drop top 5%)
    sorted_l = sorted(per_sample)
    n_keep   = max(1, int(len(sorted_l) * 0.95))
    trimmed  = sorted_l[:n_keep]

    return {
        "n_ok":           len(per_sample),
        "n_fail":         n_fail,
        "trimmed_mean":   float(np.mean(trimmed)),
        "raw_mean":       float(np.mean(per_sample)),
        "median":         float(np.median(per_sample)),
        "p95":            float(np.percentile(per_sample, 95)),
        "max":            float(max(per_sample)),
        "n_outliers_1k":  sum(1 for v in per_sample if v > 1000.0),
        "n_outliers_1e6": sum(1 for v in per_sample if v > 1e6),
        "tr_norm_mean":   float(np.mean(tr_norms)) if tr_norms else 0.0,
        "tr_norm_max":    float(max(tr_norms)) if tr_norms else 0.0,
        "tr_norm_p95":    float(np.percentile(tr_norms, 95)) if tr_norms else 0.0,
    }


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_comparison(results: List[tuple]) -> None:
    """Print side-by-side comparison table."""
    print("\n" + "=" * 90)
    print("CHECKPOINT COMPARISON")
    print("=" * 90)
    metrics = [
        ("trimmed_mean",   "Val trimmed-mean (checkpoint metric)"),
        ("raw_mean",       "Val raw mean     (outlier-inflated)"),
        ("median",         "Val median       (most robust)"),
        ("p95",            "Val p95          (near-worst)"),
        ("max",            "Val max          (worst sample)"),
        ("n_outliers_1k",  "Samples > 1000   (instability count)"),
        ("n_outliers_1e6", "Samples > 1e6    (blowup count)"),
        ("tr_norm_mean",   "tr_pred L2 mean  (score-field health)"),
        ("tr_norm_max",    "tr_pred L2 max   (worst score magnitude)"),
        ("tr_norm_p95",    "tr_pred L2 p95"),
    ]
    # Header
    names = [name for name, _ in results]
    print(f"{'Metric':<42}", end="")
    for name in names:
        print(f"  {name[:22]:>22}", end="")
    print()
    print("-" * 90)
    for key, label in metrics:
        print(f"{label:<42}", end="")
        for _, stats in results:
            v = stats.get(key, float("nan"))
            if isinstance(v, float):
                print(f"  {v:>22.4g}", end="")
            else:
                print(f"  {str(v):>22}", end="")
        print()
    print("=" * 90)

    # Verdict: compare each finetuned against pretrained
    if len(results) >= 2:
        baseline_name, baseline = results[0]
        print(f"\nVERDICT (vs {baseline_name}):")
        for name, stats in results[1:]:
            b = baseline.get("trimmed_mean", float("nan"))
            f = stats.get("trimmed_mean", float("nan"))
            if math.isfinite(b) and math.isfinite(f) and b > 0:
                delta_pct = 100.0 * (f - b) / b
                sign = "↓ BETTER" if f < b else "↑ WORSE"
                print(f"  {name}: trimmed_mean {b:.4f} → {f:.4f}  "
                      f"({delta_pct:+.1f}%)  {sign}")
            else:
                print(f"  {name}: could not compare (nan values)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare pretrained vs finetuned RAPiDock checkpoint quality.")
    p.add_argument("--pretrained", required=True,
                   help="Path to original pretrained checkpoint (baseline)")
    p.add_argument("--finetuned", required=True,
                   help="Primary finetuned checkpoint to evaluate")
    p.add_argument("--also-compare", nargs="*", default=[],
                   help="Additional checkpoints to evaluate (e.g. v1 vs v2 comparison)")
    p.add_argument("--val-csv", required=True,
                   help="Validation set CSV (same as used during training)")
    p.add_argument("--model-params",
                   default=str(_HERE / "third_party" / "RAPiDock_finetuned" /
                               "train_models" / "CGTensorProductEquivariantModel" /
                               "model_parameters.yml"),
                   help="Model hyperparameter YAML")
    p.add_argument("--out", default="logs/comparison.json",
                   help="JSON file to write results to")
    p.add_argument("--device", default=None,
                   help="cuda or cpu (default: auto)")
    args = p.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model args
    with open(args.model_params) as fh:
        params = yaml.safe_load(fh)
    model_args = Namespace(**params)
    model_args.esm_embeddings_path_train   = True
    model_args.esm_embeddings_peptide_train = None

    # Build val dataset (ESM cache will be hit from training)
    print(f"\nLoading val set: {args.val_csv}")
    val_ds = build_dataset(
        args.val_csv, model_args,
        output_dir=str(_HERE / "third_party" / "RAPiDock_finetuned" /
                       "finetune_peppc_phase1" / "processed_val"),
        esm_device=device_str,
    )
    val_indices = list(range(len(val_ds)))
    print(f"Val complexes: {len(val_indices)}")
    transform = NoiseTransform(model_args)

    # Build checkpoint list: (name, path)
    checkpoints = [
        ("pretrained",       args.pretrained),
        ("finetuned (main)", args.finetuned),
    ] + [(f"finetuned ({Path(p).parent.name})", p) for p in args.also_compare]

    results = []
    for name, ckpt_path in checkpoints:
        if not Path(ckpt_path).exists():
            print(f"\n[SKIP] {name}: checkpoint not found: {ckpt_path}")
            continue
        print(f"\nEvaluating: {name}  ({ckpt_path})")
        model = load_checkpoint(ckpt_path, model_args, device)
        stats = evaluate(model, val_ds, val_indices, transform, device)
        stats["checkpoint"] = ckpt_path
        print(f"  trimmed_mean={stats.get('trimmed_mean', 'nan'):.4f}  "
              f"median={stats.get('median', 'nan'):.4f}  "
              f"tr_norm_mean={stats.get('tr_norm_mean', 0):.3f}  "
              f"tr_norm_max={stats.get('tr_norm_max', 0):.1f}  "
              f"outliers_1k={stats.get('n_outliers_1k', 0)}")
        results.append((name, stats))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not results:
        print("[FAIL] No checkpoints could be evaluated.")
        sys.exit(1)

    print_comparison(results)

    # Write JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: stats for name, stats in results}
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
