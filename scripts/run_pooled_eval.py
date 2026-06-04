#!/usr/bin/env python3
"""Multi-model pooled pose evaluation.

Generates poses from multiple models (default: pretrained + v5c + v6),
pools all poses into one set per complex, then ranks the full pool by
ref2015 and reports metrics. Compare against per-model benchmarks to
measure diversity-of-generation benefit.

Usage:
    conda run -n score-env python3 scripts/run_pooled_eval.py \
        --models pretrained v5c v6 \
        --n-samples 34 \
        --n-per-cell 5 \
        --out-dir logs/analysis_pooled \
        --mmgbsa-topk 5

    # 3 models × 34 poses = 102 total poses per complex (≈ N=100 equivalent)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None  # type: ignore[assignment]

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

BENCH300_CSV  = REPO / "data" / "benchmark300.csv"
BASE_MODEL_DIR = (
    REPO / "third_party" / "RAPiDock_finetuned" /
    "train_models" / "CGTensorProductEquivariantModel"
)
RAPIDOCK_DIR = REPO / "third_party" / "RAPiDock"
RUN_SHIM     = REPO / "src" / "hybridock_pep" / "sampling" / "run_rapidock.py"

FT = REPO / "third_party" / "RAPiDock_finetuned"
MODEL_CKPTS: dict[str, Path] = {
    "pretrained": BASE_MODEL_DIR / "rapidock_local.pt",
    "v3c":        FT / "finetune_peppc_v3c_phase2" / "rapidock_finetuned_best.pt",
    "v4c":        FT / "finetune_peppc_v4c_phase2" / "rapidock_finetuned_best.pt",
    "v5c":        FT / "finetune_peppc_v5c_phase2" / "rapidock_finetuned_best.pt",
    "v6":         REPO / "logs" / "v6_run" / "phase2" / "rapidock_finetuned_best.pt",
}
DEFAULT_POOL = ["pretrained", "v5c", "v6"]

log = logging.getLogger("pooled_eval")


# ── helpers (shared with run_post_pocket_eval.py) ─────────────────────────────

def _ca_coords_from_pdb(pdb: Path) -> np.ndarray:
    coords = []
    for ln in pdb.read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM")) and ln[12:16].strip() == "CA":
            try:
                coords.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
            except ValueError:
                continue
    return np.array(coords) if coords else np.zeros((0, 3))


def _ca_rmsd(pose_pdb: Path, crystal_ca: np.ndarray) -> Optional[float]:
    pose_ca = _ca_coords_from_pdb(pose_pdb)
    if len(pose_ca) == 0 or len(crystal_ca) == 0:
        return None
    n = min(len(pose_ca), len(crystal_ca))
    diff = pose_ca[:n] - crystal_ca[:n]
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


def _merge_pdb(receptor_pdb: Path, pose_pdb: Path, out: Path) -> bool:
    rec = [ln for ln in receptor_pdb.read_text().splitlines()
           if ln.startswith(("ATOM", "HETATM"))]
    pep = []
    for ln in pose_pdb.read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            pep.append(ln[:21] + "P" + ln[22:])
    if not rec or not pep:
        return False
    out.write_text("\n".join(rec + ["TER"] + pep + ["END\n"]))
    return True


_PR_SFXN = None
_PR_MODULE = None


def _init_pyrosetta():
    global _PR_SFXN, _PR_MODULE
    if _PR_SFXN is not None:
        return _PR_MODULE, _PR_SFXN
    try:
        import pyrosetta  # type: ignore[import-untyped]
        pyrosetta.init(
            " ".join(["-mute", "all", "-ignore_unrecognized_res", "-no_fconfig",
                      "-use_terminal_residues", "true"]),
            silent=True,
        )
        _PR_SFXN = pyrosetta.create_score_function("ref2015")
        _PR_MODULE = pyrosetta
    except Exception as exc:
        log.error("PyRosetta init failed: %s", exc)
    return _PR_MODULE, _PR_SFXN


def score_ref2015(receptor_pdb: Path, pose_pdb: Path, tmp_dir: Path) -> Optional[float]:
    pr, sfxn = _init_pyrosetta()
    if pr is None:
        return None
    merged = tmp_dir / f"merged_{os.getpid()}_{time.time_ns()}.pdb"
    try:
        if not _merge_pdb(receptor_pdb, pose_pdb, merged):
            return None
        return float(sfxn(pr.pose_from_pdb(str(merged))))
    except Exception:
        return None
    finally:
        try:
            merged.unlink()
        except OSError:
            pass


def _find_rapidock_python() -> str:
    for base in [Path.home() / "miniconda3", Path.home() / "miniforge3",
                 Path.home() / "anaconda3", Path("/opt/conda")]:
        p = base / "envs" / "rapidock" / "bin" / "python3"
        if p.exists():
            return str(p)
    raise RuntimeError("rapidock conda env not found")


def _build_tmp_model_dir(out_dir: Path, ckpt_path: Path, label: str) -> Path:
    tmp = out_dir / f"_model_dir_tmp_{label}"
    tmp.mkdir(parents=True, exist_ok=True)
    yml_src = ckpt_path.parent / "model_parameters.yml"
    if not yml_src.exists():
        yml_src = (REPO / "third_party" / "RAPiDock_finetuned" /
                   "train_models" / "CGTensorProductEquivariantModel" / "model_parameters.yml")
    shutil.copy2(yml_src, tmp / "model_parameters.yml")
    link = tmp / ckpt_path.name
    if not link.exists():
        link.symlink_to(ckpt_path.resolve())
    return tmp


def run_inference_model(
    label: str,
    ckpt_path: Path,
    receptor_pdb: Path,
    seq: str,
    out_dir: Path,
    n_samples: int,
    seed: int,
    rapidock_python: str,
) -> list[Path]:
    """Generate poses for one model; resume-safe."""
    poses_dir   = out_dir / "poses"
    done_marker = out_dir / "inference_done.json"

    if done_marker.exists():
        existing = sorted(poses_dir.glob("pose_*.pdb"))
        if existing:
            return existing

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "poses_raw"
    tmp_model_dir = _build_tmp_model_dir(out_dir, ckpt_path, label)

    cmd = [
        rapidock_python, str(RUN_SHIM),
        "--peptide", seq,
        "--receptor", str(receptor_pdb.resolve()),
        "--output-dir", str(raw_dir.resolve()),
        "--n-samples", str(n_samples),
        "--seed", str(seed),
        "--rapidock-dir", str(RAPIDOCK_DIR.resolve()),
        "--model-dir", str(tmp_model_dir.resolve()),
        "--ckpt", ckpt_path.name,
        "--scoring-function", "none",
    ]

    log.info("[%s] inference %d poses ...", label, n_samples)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        log.warning("[%s] RAPiDock exit %d in %.0fs", label, proc.returncode, elapsed)

    raw_inner = raw_dir / "poses_raw"
    if not raw_inner.exists():
        raw_inner = raw_dir
    rank_files = sorted(
        raw_inner.glob("rank*.pdb"),
        key=lambda p: int(re.search(r"rank(\d+)", p.stem).group(1)),  # type: ignore[union-attr]
    )
    poses_dir.mkdir(parents=True, exist_ok=True)
    renamed = []
    for i, src in enumerate(rank_files):
        dst = poses_dir / f"pose_{i}.pdb"
        shutil.copy2(src, dst)
        renamed.append(dst)

    log.info("[%s] %d poses in %.0fs", label, len(renamed), elapsed)
    done_marker.write_text(json.dumps({"n_poses": len(renamed), "elapsed_s": elapsed}))
    return renamed


def crop_receptor(receptor_pdb: Path, crystal_pdb: Path, out_path: Path) -> Path:
    """Pocket-crop receptor around crystal peptide centroid."""
    from hybridock_pep.prep.receptor import crop_to_pocket
    ca = _ca_coords_from_pdb(crystal_pdb)
    if len(ca) == 0:
        return receptor_pdb
    site = tuple(ca.mean(axis=0).tolist())
    spans = ca.max(axis=0) - ca.min(axis=0)
    span = float(max(spans))
    n = len(ca)
    if n <= 8:
        box_size = max(span + 6.0, 14.0)
    elif n <= 12:
        box_size = max(span + 8.0, 16.0)
    else:
        box_size = max(span + 10.0, 20.0)
    radius = max(12.0, box_size / 2.0 + 5.0)
    try:
        crop_to_pocket(receptor_pdb, site, radius, out_path)
        return out_path
    except Exception:
        return receptor_pdb


def stratified_sample(df: pd.DataFrame, n_per_cell: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    selected: list[str] = []
    for ss in sorted(df["ss_class"].unique()):
        for lb in sorted(df["length_bucket"].unique()):
            cell = df[(df["ss_class"] == ss) & (df["length_bucket"] == lb)]
            valid = cell[
                cell["receptor"].apply(lambda p: Path(p).exists()) &
                cell["peptide_pdb"].apply(lambda p: Path(p).exists())
            ]
            names = valid["name"].tolist()
            rng.shuffle(names)
            selected.extend(names[:n_per_cell])
    return df[df["name"].isin(selected)].reset_index(drop=True)


# ── per-complex pooled eval ───────────────────────────────────────────────────

def eval_complex_pooled(
    row: pd.Series,
    out_dir: Path,
    pool_models: list[str],
    n_samples_per_model: int,
    seed: int,
    rapidock_python: str,
    tmp_dir: Path,
    mmgbsa_topk: int = 0,
) -> dict:
    """Run inference for each pool model, merge all poses, rank jointly."""
    cname = str(row["name"])
    receptor_pdb_orig = Path(row["receptor"])
    crystal_pdb = Path(row["peptide_pdb"])
    seq = str(row["seq"])

    crystal_ca = _ca_coords_from_pdb(crystal_pdb)

    # Crop receptor once for all models
    cropped_receptor = out_dir / "receptor_cropped.pdb"
    if not cropped_receptor.exists():
        actual = crop_receptor(receptor_pdb_orig, crystal_pdb, cropped_receptor)
        if actual != cropped_receptor:
            shutil.copy2(actual, cropped_receptor)

    # Per-model inference
    all_poses: list[Path] = []
    pose_model_labels: list[str] = []
    for label in pool_models:
        ckpt_path = MODEL_CKPTS.get(label)
        if ckpt_path is None or not ckpt_path.exists():
            log.warning("[%s][%s] checkpoint missing, skipping", cname, label)
            continue
        model_out = out_dir / label
        poses = run_inference_model(
            label=label,
            ckpt_path=ckpt_path,
            receptor_pdb=cropped_receptor,
            seq=seq,
            out_dir=model_out,
            n_samples=n_samples_per_model,
            seed=seed,
            rapidock_python=rapidock_python,
        )
        all_poses.extend(poses)
        pose_model_labels.extend([label] * len(poses))

    if not all_poses:
        return {"error": "no_poses"}

    # RMSDs for the full pool
    rmsds = [_ca_rmsd(p, crystal_ca) for p in all_poses]
    valid_rmsds = [r for r in rmsds if r is not None]
    if not valid_rmsds:
        return {"error": "all_rmsd_failed"}

    best_rmsd = float(min(valid_rmsds))
    log.info("[%s] pool %d poses  best=%.2fÅ  median=%.2fÅ",
             cname, len(valid_rmsds), best_rmsd, float(np.median(valid_rmsds)))

    # ref2015 scoring on full pool
    ref2015_scores: list[Optional[float]] = []
    for pose in all_poses:
        ref2015_scores.append(score_ref2015(cropped_receptor, pose, tmp_dir))
    n_scored = sum(1 for s in ref2015_scores if s is not None)
    log.info("[%s] ref2015: %d/%d poses scored", cname, n_scored, len(all_poses))

    # Rank full pool by ref2015
    ranked_pairs = sorted(
        [(s if s is not None else float("inf"), r, lbl)
         for s, r, lbl in zip(ref2015_scores, rmsds, pose_model_labels) if r is not None],
        key=lambda x: x[0],
    )
    top1_rmsd = float(ranked_pairs[0][1]) if ranked_pairs else float("nan")
    topk_rmsds = [r for _, r, _ in ranked_pairs[:5]]
    hit_at1 = float(top1_rmsd <= 2.0)
    hit_at5 = float(any(r <= 2.0 for r in topk_rmsds))

    # Optional MM-GBSA reranking
    mmgbsa_top1_rmsd = float("nan")
    if mmgbsa_topk > 0:
        try:
            from hybridock_pep.scoring.mmgbsa import compute_mmgbsa_single  # noqa: PLC0415
            topk_indices = [
                i for i, (s, r, _) in enumerate(
                    zip(ref2015_scores, rmsds, pose_model_labels)) if r is not None
            ][:mmgbsa_topk]
            mm_pairs = []
            for i in topk_indices:
                try:
                    dg = compute_mmgbsa_single(all_poses[i], cropped_receptor)
                    mm_pairs.append((dg, rmsds[i]))
                except Exception:
                    pass
            if mm_pairs:
                mm_pairs.sort(key=lambda x: x[0])
                mmgbsa_top1_rmsd = float(mm_pairs[0][1])  # type: ignore[arg-type]
        except ImportError:
            pass

    # Per-model source breakdown in top-1
    top1_model = ranked_pairs[0][2] if ranked_pairs else "unknown"

    # Kendall τ on full pool
    tau = float("nan")
    if scipy_stats is not None:
        paired = [(s, r) for s, r in zip(ref2015_scores, rmsds)
                  if s is not None and r is not None]
        if len(paired) >= 2:
            try:
                tau, _ = scipy_stats.kendalltau([s for s, _ in paired], [r for _, r in paired])
                tau = float(tau)
            except Exception:
                pass

    return {
        "top1_rmsd": top1_rmsd,
        "mmgbsa_top1_rmsd": mmgbsa_top1_rmsd,
        "hit_at1": hit_at1,
        "hit_at5": hit_at5,
        "best_rmsd": best_rmsd,
        "tau": tau,
        "top1_model": top1_model,
        "n_poses_total": len(valid_rmsds),
        "n_poses_per_model": n_samples_per_model,
        "pool_models": pool_models,
        "ss_class": str(row["ss_class"]),
        "length_bucket": str(row["length_bucket"]),
        "pep_len": int(row["pep_len"]),
        "complex": cname,
    }


# ── aggregation + output ──────────────────────────────────────────────────────

def aggregate(results: dict) -> dict:
    rows = list(results.values())
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    df = df[df.get("error", pd.Series([None]*len(df))).isna()] if "error" in df.columns else df

    summary: dict = {}
    for col in ["top1_rmsd", "mmgbsa_top1_rmsd", "hit_at1", "hit_at5", "best_rmsd", "tau"]:
        vals = df[col].dropna().tolist() if col in df.columns else []
        summary[f"mean_{col}"] = float(np.mean(vals)) if vals else float("nan")

    for ss in ["HELIX", "SHEET", "UNUSUAL"]:
        sub = df[df["ss_class"] == ss] if "ss_class" in df.columns else pd.DataFrame()
        if len(sub):
            summary[f"ss_{ss}_top1"] = float(sub["top1_rmsd"].mean())
            summary[f"ss_{ss}_best"] = float(sub["best_rmsd"].mean())
            summary[f"ss_{ss}_hit5"] = float(sub["hit_at5"].mean())

    if "top1_model" in df.columns:
        summary["top1_model_distribution"] = df["top1_model"].value_counts().to_dict()

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models",          nargs="+", default=DEFAULT_POOL,
                    choices=list(MODEL_CKPTS.keys()),
                    help=f"Models to pool (default: {DEFAULT_POOL})")
    ap.add_argument("--n-samples",       type=int, default=34,
                    help="Poses per model (default 34; 3 models → 102 total)")
    ap.add_argument("--n-per-cell",      type=int, default=5)
    ap.add_argument("--seed",            type=int, default=42)
    ap.add_argument("--out-dir",         default="logs/analysis_pooled")
    ap.add_argument("--mmgbsa-topk",     type=int, default=0,
                    help="MM-GBSA reranking on top-K ref2015 poses (0 = off)")
    ap.add_argument("--limit",           type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(out_dir / "run.log"), mode="a"),
        ],
    )

    try:
        rapidock_python = _find_rapidock_python()
    except RuntimeError as exc:
        log.error("%s", exc)
        sys.exit(1)

    bench = pd.read_csv(BENCH300_CSV)
    bench_subset = stratified_sample(bench, args.n_per_cell, args.seed)
    if args.limit:
        bench_subset = bench_subset.head(args.limit)

    n_total = len(bench_subset)
    n_per_model = args.n_samples
    n_pool = len(args.models) * n_per_model
    log.info("Pooled eval: %d complexes, %d models × %d poses = %d total poses/complex",
             n_total, len(args.models), n_per_model, n_pool)
    log.info("Pool: %s", args.models)

    all_results: dict = {}
    t0_global = time.time()

    with tempfile.TemporaryDirectory(prefix="pooled_eval_") as tmpd:
        tmp_dir = Path(tmpd)
        _init_pyrosetta()

        for ci, (_, row) in enumerate(bench_subset.iterrows()):
            cname = str(row["name"])
            complex_out = out_dir / "per_complex" / cname

            log.info("─── [%d/%d] %s  (%s/%s len=%d) ───",
                     ci + 1, n_total, cname, row["ss_class"], row["length_bucket"], row["pep_len"])

            # Cache check
            result_json = complex_out / "pooled_metrics.json"
            if result_json.exists():
                try:
                    with open(result_json) as fh:
                        all_results[cname] = json.load(fh)
                        log.debug("[%s] cached", cname)
                        continue
                except json.JSONDecodeError:
                    pass

            result = eval_complex_pooled(
                row=row,
                out_dir=complex_out,
                pool_models=args.models,
                n_samples_per_model=n_per_model,
                seed=args.seed,
                rapidock_python=rapidock_python,
                tmp_dir=tmp_dir,
                mmgbsa_topk=args.mmgbsa_topk,
            )
            all_results[cname] = result
            result_json.parent.mkdir(parents=True, exist_ok=True)
            result_json.write_text(json.dumps(result, indent=2))

    elapsed = time.time() - t0_global
    log.info("Total wall-clock: %.1f s (%.1f min)", elapsed, elapsed / 60)

    agg = aggregate(all_results)
    out_dir.joinpath("aggregate_stats.json").write_text(
        json.dumps({"summary": agg, "n_complexes": len(all_results)}, indent=2)
    )

    # Print summary table
    print("\n" + "=" * 80)
    print(f"POOLED EVAL — {len(all_results)} complexes  pool={args.models}")
    print("=" * 80)
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:.3f}")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
    print("=" * 80)

    # Save results CSV
    rows = [{"complex": k, **v} for k, v in all_results.items() if isinstance(v, dict)]
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "pooled_results.csv", index=False)
        log.info("Wrote %s", out_dir / "pooled_results.csv")


if __name__ == "__main__":
    main()
