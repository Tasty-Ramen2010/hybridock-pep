#!/usr/bin/env python3
"""Multi-complex inference benchmark: pretrained vs finetuned checkpoints.

Runs RAPiDock inference on a set of protein-peptide complexes and computes
RMSD-based pose quality metrics for each model, producing a cross-complex
comparison table.

Usage (score-env):
    conda run -n score-env python3 scripts/benchmark_inference_multi.py \\
        --benchmark-csv  data/inference_benchmark_set.csv \\
        --pretrained     third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_local.pt \\
        --finetuned      third_party/RAPiDock_finetuned/finetune_peppc_phase3/rapidock_finetuned_best.pt \\
        --also-compare   third_party/RAPiDock_finetuned/finetune_peppc_v2_phase3/rapidock_finetuned_best.pt \\
                         third_party/RAPiDock_finetuned/finetune_peppc_v2b_phase3/rapidock_finetuned_best.pt \\
        --n-samples      20 \\
        --seed           42 \\
        --out-dir        logs/benchmark_inference_multi

Benchmark CSV columns:
    name          — complex identifier
    receptor      — path to receptor pocket PDB
    peptide_pdb   — path to crystal peptide PDB (used as reference)
    seq           — peptide sequence (FASTA, 1-letter)
    pep_len       — peptide length (informational)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = (
    REPO / "third_party" / "RAPiDock_finetuned" /
    "train_models" / "CGTensorProductEquivariantModel"
)
DEFAULT_PRETRAINED = DEFAULT_MODEL_DIR / "rapidock_local.pt"
DEFAULT_RUN_RAPIDOCK_SHIM = REPO / "src" / "hybridock_pep" / "sampling" / "run_rapidock.py"
DEFAULT_DIVERSITY_SCRIPT  = REPO / "scripts" / "eval_pose_diversity.py"


def _find_rapidock_python() -> str:
    env_override = os.environ.get("RAPIDOCK_PYTHON")
    if env_override and Path(env_override).exists():
        return env_override
    for c in [
        "/home/igem/miniconda3/envs/rapidock/bin/python3",
        "/home/igem/miniconda3/envs/rapidock/bin/python",
        shutil.which("python3") or "",
    ]:
        if c and Path(c).exists():
            return c
    sys.exit("Cannot find rapidock python3. Set RAPIDOCK_PYTHON env var.")


def run_inference(
    checkpoint_path: Path,
    receptor: Path,
    peptide_seq: str,
    out_dir: Path,
    n_samples: int,
    seed: Optional[int],
    rapidock_python: str,
    model_dir: Path,
    label: str,
) -> tuple[list[Path], float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    poses_raw_dir = out_dir / "poses_raw"
    poses_dir     = out_dir / "poses"

    # temp model_dir with model_parameters.yml + checkpoint symlink
    tmp_model_dir = out_dir / "_model_dir_tmp"
    tmp_model_dir.mkdir(exist_ok=True)
    model_params_dst = tmp_model_dir / "model_parameters.yml"
    if not model_params_dst.exists():
        shutil.copy2(model_dir / "model_parameters.yml", model_params_dst)

    ckpt_link = tmp_model_dir / checkpoint_path.name
    if ckpt_link.exists() or ckpt_link.is_symlink():
        ckpt_link.unlink()
    try:
        ckpt_link.symlink_to(checkpoint_path.resolve())
    except OSError:
        shutil.copy2(checkpoint_path, ckpt_link)

    cmd = [
        rapidock_python, str(DEFAULT_RUN_RAPIDOCK_SHIM),
        "--peptide",           peptide_seq,
        "--receptor",          str(receptor.resolve()),
        "--output-dir",        str(poses_raw_dir.resolve()),
        "--n-samples",         str(n_samples),
        "--rapidock-dir",      str((REPO / "third_party" / "RAPiDock_finetuned").resolve()),
        "--model-dir",         str(tmp_model_dir.resolve()),
        "--ckpt",              checkpoint_path.name,
        "--scoring-function",  "none",
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]

    print(f"  [{label}] inference ({n_samples} samples)...", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  [{label}] WARN: RAPiDock exited {proc.returncode}")

    poses_dir.mkdir(exist_ok=True)
    raw_inner = poses_raw_dir / "poses_raw"
    if not raw_inner.exists():
        raw_inner = poses_raw_dir

    rank_files = sorted(raw_inner.glob("rank*.pdb"))
    pose_files: list[Path] = []
    for i, rf in enumerate(rank_files):
        dst = poses_dir / f"pose_{i}.pdb"
        shutil.copy2(rf, dst)
        pose_files.append(dst)

    print(f"  [{label}] {len(pose_files)} poses in {elapsed:.0f}s", flush=True)
    return pose_files, elapsed


def eval_diversity(
    poses_dir: Path,
    reference: Optional[Path],
    out_json: Path,
) -> dict:
    cmd = [sys.executable, str(DEFAULT_DIVERSITY_SCRIPT),
           "--poses-dir", str(poses_dir),
           "--out",       str(out_json)]
    if reference:
        cmd += ["--reference", str(reference.resolve())]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  [WARN] eval_pose_diversity exited {proc.returncode}: {proc.stderr[:200]}")
    if out_json.exists():
        with open(out_json) as fh:
            return json.load(fh)
    return {}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-csv", required=True,
                   help="CSV with columns: name, receptor, peptide_pdb, seq, pep_len")
    p.add_argument("--pretrained",  default=str(DEFAULT_PRETRAINED))
    p.add_argument("--finetuned",   required=True,
                   help="Primary finetuned checkpoint (v1 P3 best)")
    p.add_argument("--also-compare", nargs="*", default=[],
                   help="Additional checkpoints to compare")
    p.add_argument("--n-samples",   type=int, default=20)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--out-dir",     default="logs/benchmark_inference_multi")
    p.add_argument("--model-dir",   default=str(DEFAULT_MODEL_DIR))
    args = p.parse_args()

    out_root   = Path(args.out_dir)
    model_dir  = Path(args.model_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Build ordered model list: pretrained first, then finetuned, then extras
    def _label_from_ckpt(ckpt: Path) -> str:
        """Derive a short model label from checkpoint directory name."""
        lbl = ckpt.parent.name  # e.g. finetune_peppc_v5c_phase2
        lbl = lbl.replace("finetune_peppc_", "")
        for suffix in ("_phase1", "_phase2", "_phase3"):
            lbl = lbl.replace(suffix, "")
        return lbl or ckpt.stem

    models: list[tuple[str, Path]] = [
        ("pretrained",  Path(args.pretrained)),
        (_label_from_ckpt(Path(args.finetuned)), Path(args.finetuned)),
    ]
    for extra in args.also_compare:
        ep = Path(extra)
        models.append((_label_from_ckpt(ep), ep))

    rapidock_python = _find_rapidock_python()

    # Load benchmark complexes
    bench = pd.read_csv(args.benchmark_csv)
    print(f"\n{'='*70}")
    print(f"Benchmark: {len(bench)} complexes × {len(models)} models × {args.n_samples} poses")
    print(f"{'='*70}\n")

    # Results store: {complex_name: {model_label: metrics_dict}}
    all_results: dict[str, dict[str, dict]] = {}

    for _, row in bench.iterrows():
        cname    = row["name"]
        receptor = Path(row["receptor"])
        ref_pdb  = Path(row["peptide_pdb"])
        seq      = str(row["seq"])

        print(f"\n{'─'*70}")
        print(f"Complex: {cname}  (len={row['pep_len']}, seq={seq})")
        print(f"{'─'*70}")

        if not receptor.exists():
            print(f"  SKIP: receptor not found: {receptor}")
            continue
        if not ref_pdb.exists():
            print(f"  SKIP: crystal peptide PDB not found: {ref_pdb}")
            continue

        all_results[cname] = {}
        complex_dir = out_root / cname

        for label, ckpt in models:
            if not ckpt.exists():
                print(f"  SKIP model {label}: checkpoint not found: {ckpt}")
                continue

            model_out = complex_dir / label
            diversity_json = model_out / "diversity.json"

            # Skip if already computed (resume support)
            if diversity_json.exists():
                print(f"  [{label}] cached — loading {diversity_json}")
                with open(diversity_json) as fh:
                    metrics = json.load(fh)
                all_results[cname][label] = metrics
                continue

            poses, elapsed = run_inference(
                ckpt, receptor, seq,
                model_out, args.n_samples, args.seed,
                rapidock_python, model_dir, label,
            )

            if not poses:
                print(f"  [{label}] WARN: no poses generated — skipping diversity eval")
                continue

            metrics = eval_diversity(model_out / "poses", ref_pdb, diversity_json)
            metrics["wallclock_s"] = elapsed
            all_results[cname][label] = metrics

            # Print per-model summary
            best  = metrics.get("best_rmsd", float("nan"))
            top1  = metrics.get("top1_rmsd", float("nan"))
            h5    = metrics.get("hit_rate_5A", 0.0)
            div   = metrics.get("diversity_ratio", 0.0)
            print(f"  [{label}]  best={best:.2f}Å  top1={top1:.2f}Å  hit@5Å={h5*100:.0f}%  div={div*100:.0f}%")

    # ── Save full JSON ──────────────────────────────────────────────────────────
    results_json = out_root / "benchmark_results.json"
    with open(results_json, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print(f"\nFull results: {results_json}")

    # ── Summary table ───────────────────────────────────────────────────────────
    model_labels = [m[0] for m in models]
    has_ss_class     = "ss_class"     in bench.columns
    has_len_bucket   = "length_bucket" in bench.columns

    rows_summary = []
    for cname, model_dict in all_results.items():
        bench_row = bench[bench["name"] == cname].iloc[0]
        for label in model_labels:
            m = model_dict.get(label, {})
            row: dict = {
                "complex":      cname,
                "pep_len":      bench_row["pep_len"],
                "model":        label,
                "best_rmsd":    m.get("best_rmsd",      float("nan")),
                "top1_rmsd":    m.get("top1_rmsd",      float("nan")),
                "median_rmsd":  m.get("rmsd_median",    float("nan")),
                "hit_rate_2A":  m.get("hit_rate_2A",    float("nan")),
                "hit_rate_5A":  m.get("hit_rate_5A",    float("nan")),
                "diversity":    m.get("diversity_ratio", float("nan")),
                "n_poses":      m.get("n_poses", 0),
            }
            if has_ss_class:
                row["ss_class"] = bench_row["ss_class"]
            if has_len_bucket:
                row["length_bucket"] = bench_row["length_bucket"]
            rows_summary.append(row)

    summary_df = pd.DataFrame(rows_summary)
    summary_csv = out_root / "benchmark_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # ── Analysis 1: Overall performance ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print("ANALYSIS 1 — Overall Performance")
    print(f"{'='*80}")

    pivot = summary_df.pivot_table(
        index=["complex", "pep_len"], columns="model",
        values="best_rmsd", aggfunc="first"
    )
    ordered_cols = [m for m in model_labels if m in pivot.columns]
    pivot = pivot[ordered_cols]
    print(pivot.to_string(float_format=lambda x: f"{x:6.2f}"))

    print(f"\n{'─'*80}")
    print(f"Per-model averages (N poses={args.n_samples}):")
    grp = summary_df.groupby("model")
    for label in model_labels:
        if label not in grp.groups:
            continue
        g  = grp.get_group(label)
        mr = g["best_rmsd"].mean(); mdr = g["median_rmsd"].mean()
        h2 = g["hit_rate_2A"].mean(); h5 = g["hit_rate_5A"].mean()
        dv = g["diversity"].mean()
        print(f"  {label:30s}  mean_best={mr:.2f}Å  median={mdr:.2f}Å  "
              f"hit@2Å={h2*100:.1f}%  hit@5Å={h5*100:.1f}%  diversity={dv*100:.1f}%")

    # ── Analysis 2: Per-class breakdown (highest priority) ───────────────────────
    if has_ss_class:
        print(f"\n{'='*80}")
        print("ANALYSIS 2 — Per Secondary-Structure Class (HELIX / SHEET / UNUSUAL)")
        print(f"{'='*80}")
        for ss in sorted(summary_df["ss_class"].dropna().unique()):
            sub = summary_df[summary_df["ss_class"] == ss]
            n_complexes = sub["complex"].nunique()
            print(f"\n  [{ss}]  ({n_complexes} complexes)")
            for label in model_labels:
                g = sub[sub["model"] == label]
                if g.empty:
                    continue
                mr = g["best_rmsd"].mean(); h2 = g["hit_rate_2A"].mean()
                h5 = g["hit_rate_5A"].mean(); dv = g["diversity"].mean()
                print(f"    {label:30s}  mean_best={mr:.2f}Å  hit@2Å={h2*100:.1f}%  "
                      f"hit@5Å={h5*100:.1f}%  diversity={dv*100:.1f}%")

    # ── Analysis 3: Length stratification ───────────────────────────────────────
    if has_len_bucket:
        print(f"\n{'='*80}")
        print("ANALYSIS 3 — Length Stratification")
        print(f"{'='*80}")
        for bucket in ["short", "medium", "long", "very_long"]:
            sub = summary_df[summary_df["length_bucket"] == bucket]
            if sub.empty:
                continue
            n_complexes = sub["complex"].nunique()
            print(f"\n  [{bucket}]  ({n_complexes} complexes)")
            for label in model_labels:
                g = sub[sub["model"] == label]
                if g.empty:
                    continue
                mr = g["best_rmsd"].mean(); h2 = g["hit_rate_2A"].mean()
                h5 = g["hit_rate_5A"].mean()
                print(f"    {label:30s}  mean_best={mr:.2f}Å  hit@2Å={h2*100:.1f}%  hit@5Å={h5*100:.1f}%")
    else:
        print(f"\n{'='*80}")
        print("ANALYSIS 3 — Length Stratification")
        print(f"{'='*80}")
        summary_df["_len_bucket"] = pd.cut(
            summary_df["pep_len"], bins=[0, 10, 15, 20, 999],
            labels=["5-10", "11-15", "16-20", "20+"])
        for bucket, sub in summary_df.groupby("_len_bucket", observed=True):
            n_complexes = sub["complex"].nunique()
            print(f"\n  [{bucket}]  ({n_complexes} complexes)")
            for label in model_labels:
                g = sub[sub["model"] == label]
                if g.empty: continue
                mr = g["best_rmsd"].mean(); h5 = g["hit_rate_5A"].mean()
                print(f"    {label:30s}  mean_best={mr:.2f}Å  hit@5Å={h5*100:.1f}%")

    # ── Analysis 4: Diversity preservation ──────────────────────────────────────
    print(f"\n{'='*80}")
    print("ANALYSIS 4 — Diversity Preservation")
    print(f"{'='*80}")
    print(f"  {'model':30s}  {'mean_diversity':>16s}  {'median_diversity':>18s}  "
          f"{'min_diversity':>15s}  {'max_diversity':>15s}")
    for label in model_labels:
        g = summary_df[summary_df["model"] == label]["diversity"].dropna()
        if g.empty: continue
        print(f"  {label:30s}  {g.mean()*100:>14.1f}%  {g.median()*100:>16.1f}%  "
              f"{g.min()*100:>13.1f}%  {g.max()*100:>13.1f}%")

    print(f"\nSummary CSV: {summary_csv}")
    print(f"Full JSON:   {results_json}")


if __name__ == "__main__":
    main()
