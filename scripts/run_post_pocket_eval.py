#!/usr/bin/env python3
"""Post-pocket-fix benchmark: 60-complex stratified re-evaluation.

Generates N poses per model variant using pocket-cropped receptors (via the
crop_to_pocket() fix in prepare_receptor_pdb()), then ranks with ref2015 and
reports pose-quality + ranking metrics broken down by SS class and length bucket.

Models: pretrained, v1, v2, v3, v3c, v4c, v5c, v6
Metrics: top-1 Cα RMSD (ref2015-ranked), hit@1 (top-1 ≤2Å), hit@5 (any of
         top-5 ≤2Å), oracle best-of-N, Kendall τ, diversity; SS × len breakdown.

Uses RAPiDock-Reloaded (third_party/RAPiDock/) for inference — it has the
valence-5 fix, CUDA 12.8 (RTX 5070 Blackwell) support, and is what the
production driver now uses.

Usage (score-env, PyRosetta symlinked):
    conda run -n score-env python3 scripts/run_post_pocket_eval.py \\
        [--n-samples 25] [--n-per-cell 5] [--seed 42] \\
        [--out-dir logs/analysis_post_pocket_fix] [--skip-ref2015]

Resume-safe: skips (complex, model) pairs that already have a results JSON.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
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

# ── repo & env setup ──────────────────────────────────────────────────────────

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from hybridock_pep.prep.receptor import crop_to_pocket  # noqa: E402

# ── paths ─────────────────────────────────────────────────────────────────────

BENCH300_CSV = REPO / "data" / "benchmark300.csv"
BASE_MODEL_DIR = (
    REPO / "third_party" / "RAPiDock_finetuned" /
    "train_models" / "CGTensorProductEquivariantModel"
)
RAPIDOCK_DIR = REPO / "third_party" / "RAPiDock"   # Reloaded — valence fix + CUDA 12.8
RUN_SHIM     = REPO / "src" / "hybridock_pep" / "sampling" / "run_rapidock.py"
DIVERSITY_SCRIPT = REPO / "scripts" / "eval_pose_diversity.py"

# ── model checkpoint registry ─────────────────────────────────────────────────

FT = REPO / "third_party" / "RAPiDock_finetuned"

MODEL_CKPTS: dict[str, Path] = {
    "pretrained": BASE_MODEL_DIR / "rapidock_local.pt",
    "v1":         FT / "finetune_peppc_phase3"     / "rapidock_finetuned_best.pt",
    "v2":         FT / "finetune_peppc_v2_phase3"  / "rapidock_finetuned_best.pt",
    "v3":         FT / "finetune_peppc_v3_phase2"  / "rapidock_finetuned_best.pt",
    "v3c":        FT / "finetune_peppc_v3c_phase2" / "rapidock_finetuned_best.pt",
    "v4c":        FT / "finetune_peppc_v4c_phase2" / "rapidock_finetuned_best.pt",
    "v5c":        FT / "finetune_peppc_v5c_phase2" / "rapidock_finetuned_best.pt",
    "v6":         REPO / "logs" / "v6_run" / "phase1" / "rapidock_finetuned_best.pt",
}

MODEL_ORDER = ["pretrained", "v1", "v2", "v3", "v3c", "v4c", "v5c", "v6"]

log = logging.getLogger("post_pocket_eval")


# ── rapidock python detection ─────────────────────────────────────────────────

def _find_rapidock_python() -> str:
    override = os.environ.get("RAPIDOCK_PYTHON")
    if override and Path(override).exists():
        return override
    for candidate in [
        "/home/igem/miniconda3/envs/rapidock/bin/python3",
        "/home/igem/miniforge3/envs/rapidock/bin/python3",
        shutil.which("python3") or "",
    ]:
        if candidate and Path(candidate).exists():
            return candidate
    # Try CONDA_EXE-derived path
    conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda_exe:
        p = Path(conda_exe).resolve().parent.parent / "envs" / "rapidock" / "bin" / "python3"
        if p.exists():
            return str(p)
    sys.exit("Cannot find rapidock python3. Set RAPIDOCK_PYTHON env var.")


# ── crystal peptide geometry helpers ─────────────────────────────────────────

def _ca_coords_from_pdb(pdb_path: Path) -> np.ndarray:
    """Return (N, 3) Cα coordinates from PDB, all chains."""
    coords: list[list[float]] = []
    for line in pdb_path.read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if line[12:16].strip() != "CA":
            continue
        try:
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
    return np.array(coords) if coords else np.zeros((0, 3))


def _site_and_box(crystal_peptide_pdb: Path) -> tuple[tuple[float, float, float], float]:
    """Derive (site_coords, box_size) from crystal peptide Cα centroid + extent.

    site_coords = centroid of crystal Cα atoms.
    box_size = max per-axis span of Cα atoms + 10 Å (ensures the whole peptide
               extent is covered, not just the center).  Minimum 20 Å.
    """
    ca = _ca_coords_from_pdb(crystal_peptide_pdb)
    if len(ca) == 0:
        raise ValueError(f"No CA atoms in crystal peptide: {crystal_peptide_pdb}")
    center = ca.mean(axis=0)
    spans = ca.max(axis=0) - ca.min(axis=0)
    box_size = float(max(max(spans) + 10.0, 20.0))
    return (float(center[0]), float(center[1]), float(center[2])), box_size


# ── temporary model directory builder ────────────────────────────────────────

def _build_tmp_model_dir(out_dir: Path, ckpt_path: Path, label: str) -> Path:
    """Create {out_dir}/_model_dir_tmp_{label}/ with model_parameters.yml + ckpt link.

    RAPiDock's inference.py expects model_parameters.yml and the checkpoint
    file to be in the same directory (--model-dir).  We create a temp dir
    per model so all runs can coexist in the same output tree.

    Args:
        out_dir: Base output directory for this complex.
        ckpt_path: Absolute path to the checkpoint file.
        label: Model label (used to name the temp directory).

    Returns:
        Path to the created temp model directory.
    """
    tmp = out_dir / f"_model_dir_tmp_{label}"
    tmp.mkdir(parents=True, exist_ok=True)

    yml_dst = tmp / "model_parameters.yml"
    if not yml_dst.exists():
        shutil.copy2(BASE_MODEL_DIR / "model_parameters.yml", yml_dst)

    ckpt_link = tmp / ckpt_path.name
    if ckpt_link.exists() or ckpt_link.is_symlink():
        ckpt_link.unlink()
    try:
        ckpt_link.symlink_to(ckpt_path.resolve())
    except OSError:
        shutil.copy2(ckpt_path, ckpt_link)

    return tmp


# ── RAPiDock inference ────────────────────────────────────────────────────────

def run_inference(
    label: str,
    ckpt_path: Path,
    receptor_pdb: Path,
    seq: str,
    out_dir: Path,
    n_samples: int,
    seed: int,
    rapidock_python: str,
) -> list[Path]:
    """Run RAPiDock inference; return list of pose_*.pdb paths.

    Skips if poses/ already contains enough .pdb files (resume support).
    """
    poses_dir   = out_dir / "poses"
    raw_dir     = out_dir / "poses_raw"
    done_marker = out_dir / "inference_done.json"

    if done_marker.exists():
        existing = sorted(poses_dir.glob("pose_*.pdb"))
        if len(existing) >= 1:
            log.debug("[%s] cached %d poses", label, len(existing))
            return existing

    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_model_dir = _build_tmp_model_dir(out_dir, ckpt_path, label)
    ckpt_name = ckpt_path.name

    cmd = [
        rapidock_python, str(RUN_SHIM),
        "--peptide",       seq,
        "--receptor",      str(receptor_pdb.resolve()),
        "--output-dir",    str(raw_dir.resolve()),
        "--n-samples",     str(n_samples),
        "--seed",          str(seed),
        "--rapidock-dir",  str(RAPIDOCK_DIR.resolve()),
        "--model-dir",     str(tmp_model_dir.resolve()),
        "--ckpt",          ckpt_name,
        "--scoring-function", "none",
    ]

    log.info("[%s] running inference %d poses …", label, n_samples)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        log.warning("[%s] RAPiDock exited %d in %.0fs\nstderr: %s",
                    label, proc.returncode, elapsed, proc.stderr[-800:])

    # Rename rank*.pdb → pose_*.pdb (RAPiDock-Reloaded nests under complex_name subdir)
    raw_inner = raw_dir / "poses_raw"
    if not raw_inner.exists():
        raw_inner = raw_dir
    rank_files = sorted(
        raw_inner.glob("rank*.pdb"),
        key=lambda p: int(re.search(r"rank(\d+)", p.stem).group(1)),  # type: ignore[union-attr]
    )
    poses_dir.mkdir(parents=True, exist_ok=True)
    renamed: list[Path] = []
    for i, src in enumerate(rank_files):
        dst = poses_dir / f"pose_{i}.pdb"
        shutil.copy2(src, dst)
        renamed.append(dst)

    log.info("[%s] %d poses in %.0fs", label, len(renamed), elapsed)
    done_marker.write_text(json.dumps({"n_poses": len(renamed), "elapsed_s": elapsed}))
    return renamed


# ── RMSD computation ──────────────────────────────────────────────────────────

def _ca_rmsd(pose_pdb: Path, crystal_ca: np.ndarray) -> Optional[float]:
    """Compute Cα RMSD between pose and crystal peptide (matching by count)."""
    pose_ca = _ca_coords_from_pdb(pose_pdb)
    if len(pose_ca) == 0 or len(crystal_ca) == 0:
        return None
    n = min(len(pose_ca), len(crystal_ca))
    diff = pose_ca[:n] - crystal_ca[:n]
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


# ── ref2015 scoring ───────────────────────────────────────────────────────────

_PR_SFXN = None  # lazy init
_PR_MODULE = None


def _init_pyrosetta():
    global _PR_SFXN, _PR_MODULE
    if _PR_SFXN is not None:
        return _PR_MODULE, _PR_SFXN
    try:
        import pyrosetta
        pyrosetta.init(
            " ".join([
                "-mute", "all",
                "-use_input_sc",
                "-ignore_unrecognized_res",
                "-ignore_zero_occupancy", "false",
                "-load_PDB_components", "false",
                "-no_fconfig",
                "-use_terminal_residues", "true",
            ]),
            silent=True,
        )
        _PR_SFXN = pyrosetta.create_score_function("ref2015")
        _PR_MODULE = pyrosetta
        log.info("PyRosetta ready, ref2015 score function loaded")
        return _PR_MODULE, _PR_SFXN
    except Exception as exc:
        log.error("PyRosetta init failed: %s", exc)
        return None, None


def _merge_pdb(receptor_pdb: Path, pose_pdb: Path, out: Path) -> bool:
    """Merge receptor (chain A) and pose (reassigned to chain P) into one PDB."""
    rec_lines = [ln for ln in receptor_pdb.read_text().splitlines()
                 if ln.startswith(("ATOM", "HETATM"))]
    pep_lines = []
    for ln in pose_pdb.read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            ln = ln[:21] + "P" + ln[22:]
            pep_lines.append(ln)
    if not rec_lines or not pep_lines:
        return False
    out.write_text("\n".join(rec_lines + ["TER"] + pep_lines + ["END\n"]))
    return True


def score_ref2015(receptor_pdb: Path, pose_pdb: Path, tmp_dir: Path) -> Optional[float]:
    """Score receptor+pose complex with ref2015 (no relaxation).

    Returns float score (lower = better) or None on failure.
    """
    pr, sfxn = _init_pyrosetta()
    if pr is None:
        return None
    merged = tmp_dir / f"merged_{os.getpid()}_{time.time_ns()}.pdb"
    try:
        if not _merge_pdb(receptor_pdb, pose_pdb, merged):
            return None
        pose = pr.pose_from_pdb(str(merged))
        return float(sfxn(pose))
    except Exception as exc:
        log.debug("ref2015 failed for %s: %s", pose_pdb.name, exc)
        return None
    finally:
        try:
            merged.unlink()
        except OSError:
            pass


# ── metric aggregation ────────────────────────────────────────────────────────

def compute_metrics(
    rmsds: list[Optional[float]],
    ref2015_scores: list[Optional[float]],
    n_topk: int = 5,
    hit_threshold: float = 2.0,
) -> dict:
    """Compute pose-quality and ranking metrics from per-pose RMSD + scores.

    Args:
        rmsds: Per-pose Cα RMSD (None for failed poses).
        ref2015_scores: Per-pose ref2015 score (None for failed; lower = better).
        n_topk: k for hit@k metric.
        hit_threshold: Å threshold for "hit" (default 2.0 Å).

    Returns:
        Dict with keys: top1_rmsd, hit_at1, hit_at_topk, best_rmsd,
        tau, rho, diversity (fraction of distinct poses by 2Å RMSD),
        n_poses_ok.
    """
    valid_idx = [i for i, r in enumerate(rmsds) if r is not None]
    if not valid_idx:
        nan = float("nan")
        return {
            "top1_rmsd": nan, "hit_at1": nan, f"hit_at{n_topk}": nan,
            "best_rmsd": nan, "tau": nan, "rho": nan, "diversity": nan,
            "n_poses_ok": 0, "all_rmsds": [],
        }

    valid_rmsds = [rmsds[i] for i in valid_idx]
    valid_scores = [ref2015_scores[i] if ref2015_scores else None for i in valid_idx]

    best_rmsd = float(min(r for r in valid_rmsds if r is not None))  # type: ignore[arg-type]

    # Rank poses by ref2015 score (lower = better → ascending sort)
    has_scores = any(s is not None for s in valid_scores)
    if has_scores:
        score_pairs = [(s if s is not None else float("inf"), r)
                       for s, r in zip(valid_scores, valid_rmsds)]
        ranked = sorted(score_pairs, key=lambda x: x[0])
        top1_rmsd = float(ranked[0][1])
        topk_rmsds = [r for _, r in ranked[:n_topk]]
    else:
        # Fallback: pose_0 is top-1 (diffusion order)
        top1_rmsd = float(valid_rmsds[0])
        topk_rmsds = valid_rmsds[:n_topk]

    hit_at1 = float(top1_rmsd <= hit_threshold)
    hit_atk = float(any(r <= hit_threshold for r in topk_rmsds))

    # Kendall τ and Spearman ρ (only if we have ref2015 scores)
    tau = float("nan")
    rho = float("nan")
    if has_scores and scipy_stats is not None and len(valid_rmsds) >= 2:
        paired_s = [s for s in valid_scores if s is not None]
        paired_r = [r for s, r in zip(valid_scores, valid_rmsds) if s is not None]
        if len(paired_s) >= 2:
            try:
                tau_val, _ = scipy_stats.kendalltau(paired_s, paired_r)
                rho_val, _ = scipy_stats.spearmanr(paired_s, paired_r)
                tau = float(tau_val)
                rho = float(rho_val)
            except Exception:
                pass

    # Diversity: fraction of pose pairs with Cα RMSD > 2Å
    diversity = float("nan")
    if len(valid_rmsds) >= 2:
        # Use pairwise RMSD proxy: fraction of unique clusters at 2Å
        # (cheap: count distinct clusters by sorting RMSD-from-best)
        diversity = float(min(len(valid_rmsds), len(valid_rmsds)) / max(len(valid_rmsds), 1))
        # More meaningful: count pairs with |rmsd_i - rmsd_j| > epsilon as a proxy
        diffs = []
        for i in range(len(valid_rmsds)):
            for j in range(i + 1, len(valid_rmsds)):
                diffs.append(abs(valid_rmsds[i] - valid_rmsds[j]))  # type: ignore[operator]
        diversity = float(np.mean([d > 2.0 for d in diffs])) if diffs else float("nan")

    return {
        "top1_rmsd": top1_rmsd,
        "hit_at1": hit_at1,
        f"hit_at{n_topk}": hit_atk,
        "best_rmsd": best_rmsd,
        "tau": tau,
        "rho": rho,
        "diversity": diversity,
        "n_poses_ok": len(valid_idx),
        "all_rmsds": [r for r in rmsds if r is not None],
    }


# ── stratified sampling ───────────────────────────────────────────────────────

def stratified_sample(df: pd.DataFrame, n_per_cell: int, seed: int) -> pd.DataFrame:
    """Sample n_per_cell complexes from each (ss_class × length_bucket) cell.

    Filters out rows where the receptor or peptide_pdb files don't exist on disk.

    Args:
        df: benchmark300 DataFrame with columns ss_class, length_bucket.
        n_per_cell: Number of complexes to sample per cell.
        seed: Random seed for reproducibility.

    Returns:
        Filtered DataFrame with ≤ n_per_cell × 12 rows.
    """
    rng = random.Random(seed)
    selected_names: list[str] = []
    for ss in sorted(df["ss_class"].unique()):
        for lb in sorted(df["length_bucket"].unique()):
            cell = df[(df["ss_class"] == ss) & (df["length_bucket"] == lb)]
            # Only include complexes where files exist on disk
            valid = cell[
                cell["receptor"].apply(lambda p: Path(p).exists()) &
                cell["peptide_pdb"].apply(lambda p: Path(p).exists())
            ]
            names = valid["name"].tolist()
            rng.shuffle(names)
            selected_names.extend(names[:n_per_cell])
            if len(names) < n_per_cell:
                log.warning("cell [%s/%s]: only %d valid entries (wanted %d)",
                            ss, lb, len(names), n_per_cell)
    return df[df["name"].isin(selected_names)].reset_index(drop=True)


# ── per-complex driver ────────────────────────────────────────────────────────

def eval_complex(
    row: pd.Series,
    out_dir: Path,
    models: dict[str, Path],
    n_samples: int,
    seed: int,
    rapidock_python: str,
    skip_ref2015: bool,
    tmp_dir: Path,
) -> dict:
    """Run inference + scoring + metrics for one complex × all models.

    Returns a dict:
        {model_label: {top1_rmsd, hit_at1, hit_at5, best_rmsd, tau, rho, ...}}
    """
    cname = row["name"]
    receptor_pdb_orig = Path(row["receptor"])
    crystal_pdb = Path(row["peptide_pdb"])
    seq = str(row["seq"])
    ss  = str(row["ss_class"])
    lb  = str(row["length_bucket"])
    pep_len = int(row["pep_len"])

    # Derive site_coords and box_size from crystal peptide geometry
    try:
        site_coords, box_size = _site_and_box(crystal_pdb)
    except ValueError as exc:
        log.warning("[%s] _site_and_box failed: %s", cname, exc)
        return {}

    # Pocket crop: even though bench300 receptors are pre-cropped pocket PDBs,
    # we explicitly apply crop_to_pocket() using the crystal-derived geometry.
    # This validates the fix end-to-end and ensures consistent crop radius
    # (max(12.0, box_size/2+5.0)) across all complexes.
    crop_radius = max(12.0, box_size / 2.0 + 5.0)
    cropped_receptor = out_dir / "receptor_cropped.pdb"
    if not cropped_receptor.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        n_residues = crop_to_pocket(
            pdb_path=receptor_pdb_orig,
            site_coords=site_coords,
            radius=crop_radius,
            output_path=cropped_receptor,
        )
        log.info("[%s] pocket crop: %d residues within %.1f Å (box=%.1f Å, %s/%s)",
                 cname, n_residues, crop_radius, box_size, ss, lb)
        if n_residues < 10:
            log.warning("[%s] very few pocket residues (%d) — check site_coords",
                        cname, n_residues)

    crystal_ca = _ca_coords_from_pdb(crystal_pdb)
    if len(crystal_ca) == 0:
        log.warning("[%s] no CA atoms in crystal peptide PDB — skip", cname)
        return {}

    results: dict[str, dict] = {}

    for label, ckpt_path in models.items():
        if not ckpt_path.exists():
            log.warning("[%s][%s] checkpoint not found: %s", cname, label, ckpt_path)
            continue

        model_out = out_dir / label
        result_json = model_out / "metrics.json"
        if result_json.exists():
            try:
                with open(result_json) as fh:
                    results[label] = json.load(fh)
                    log.debug("[%s][%s] cached metrics", cname, label)
                    continue
            except json.JSONDecodeError:
                pass  # re-run if corrupt

        # Stage 1: RAPiDock inference
        poses = run_inference(
            label=label,
            ckpt_path=ckpt_path,
            receptor_pdb=cropped_receptor,
            seq=seq,
            out_dir=model_out,
            n_samples=n_samples,
            seed=seed,
            rapidock_python=rapidock_python,
        )

        if not poses:
            log.warning("[%s][%s] no poses generated", cname, label)
            continue

        # Stage 2: Cα RMSD to crystal
        rmsds: list[Optional[float]] = [_ca_rmsd(p, crystal_ca) for p in poses]
        valid_rmsds = [r for r in rmsds if r is not None]
        if not valid_rmsds:
            log.warning("[%s][%s] all RMSD computations failed", cname, label)
            continue

        log.info("[%s][%s] RMSDs: best=%.2fÅ  median=%.2fÅ  n=%d",
                 cname, label,
                 min(r for r in valid_rmsds),  # type: ignore[type-var]
                 float(np.median(valid_rmsds)),
                 len(valid_rmsds))

        # Stage 3: ref2015 scoring (optional)
        ref2015_scores: list[Optional[float]] = []
        if not skip_ref2015:
            for pose in poses:
                score = score_ref2015(cropped_receptor, pose, tmp_dir)
                ref2015_scores.append(score)
            n_scored = sum(1 for s in ref2015_scores if s is not None)
            log.info("[%s][%s] ref2015: %d/%d poses scored", cname, label, n_scored, len(poses))
        else:
            ref2015_scores = [None] * len(poses)

        # Stage 4: aggregate metrics
        metrics = compute_metrics(rmsds, ref2015_scores, n_topk=5)
        metrics.update({
            "complex": cname,
            "model": label,
            "ss_class": ss,
            "length_bucket": lb,
            "pep_len": pep_len,
            "n_samples_requested": n_samples,
            "ref2015_scores": [s for s in ref2015_scores if s is not None],
        })

        model_out.mkdir(parents=True, exist_ok=True)
        result_json.write_text(json.dumps(metrics, indent=2, default=str))
        results[label] = metrics

    return results


# ── aggregate across complexes ────────────────────────────────────────────────

def aggregate(all_results: dict[str, dict[str, dict]], n_topk: int = 5) -> dict:
    """Compute per-model aggregate stats and SS/length breakdowns.

    Args:
        all_results: {complex_name: {model_label: metrics_dict}}
        n_topk: k for hit@k metric (should match compute_metrics call).

    Returns:
        Dict with model_label → {mean_top1_rmsd, hit_at1, hit_atk,
        mean_best_rmsd, mean_tau, ss_breakdown, lb_breakdown}.
    """
    from collections import defaultdict

    model_data: dict[str, list[dict]] = defaultdict(list)
    for cname, model_dict in all_results.items():
        for label, m in model_dict.items():
            if isinstance(m, dict) and "top1_rmsd" in m:
                model_data[label].append(m)

    agg: dict[str, dict] = {}

    for label in MODEL_ORDER:
        rows = model_data.get(label, [])
        if not rows:
            continue

        def _mean_key(key: str) -> float:
            vals = [r[key] for r in rows if isinstance(r.get(key), float) and not math.isnan(r[key])]
            return float(np.mean(vals)) if vals else float("nan")

        hit_atk_key = f"hit_at{n_topk}"

        # SS breakdown
        ss_breakdown: dict[str, dict] = {}
        for ss in ["HELIX", "SHEET", "UNUSUAL"]:
            sub = [r for r in rows if r.get("ss_class") == ss]
            if sub:
                ss_breakdown[ss] = {
                    "n": len(sub),
                    "mean_top1_rmsd": float(np.nanmean([r["top1_rmsd"] for r in sub])),
                    "hit_at1": float(np.nanmean([r["hit_at1"] for r in sub])),
                    hit_atk_key: float(np.nanmean([r.get(hit_atk_key, float("nan")) for r in sub])),
                    "mean_best_rmsd": float(np.nanmean([r["best_rmsd"] for r in sub])),
                    "mean_tau": float(np.nanmean([r["tau"] for r in sub
                                                   if not math.isnan(r.get("tau", float("nan")))])),
                }

        # Length bucket breakdown
        lb_breakdown: dict[str, dict] = {}
        for lb in ["short", "medium", "long", "very_long"]:
            sub = [r for r in rows if r.get("length_bucket") == lb]
            if sub:
                lb_breakdown[lb] = {
                    "n": len(sub),
                    "mean_top1_rmsd": float(np.nanmean([r["top1_rmsd"] for r in sub])),
                    "hit_at1": float(np.nanmean([r["hit_at1"] for r in sub])),
                    hit_atk_key: float(np.nanmean([r.get(hit_atk_key, float("nan")) for r in sub])),
                    "mean_best_rmsd": float(np.nanmean([r["best_rmsd"] for r in sub])),
                }

        agg[label] = {
            "n_complexes": len(rows),
            "mean_top1_rmsd": _mean_key("top1_rmsd"),
            "hit_at1": _mean_key("hit_at1"),
            hit_atk_key: _mean_key(hit_atk_key),
            "mean_best_rmsd": _mean_key("best_rmsd"),
            "mean_tau": _mean_key("tau"),
            "mean_rho": _mean_key("rho"),
            "mean_diversity": _mean_key("diversity"),
            "ss_breakdown": ss_breakdown,
            "lb_breakdown": lb_breakdown,
        }

    return agg


# ── output writers ────────────────────────────────────────────────────────────

def write_outputs(
    all_results: dict[str, dict[str, dict]],
    agg: dict,
    bench_df: pd.DataFrame,
    out_dir: Path,
    n_topk: int,
) -> None:
    """Write benchmark_summary.csv, aggregate_stats.json, and console table."""
    hit_atk_key = f"hit_at{n_topk}"

    # flat CSV: one row per (complex × model)
    rows = []
    for cname, model_dict in all_results.items():
        bench_row = bench_df[bench_df["name"] == cname]
        for label, m in model_dict.items():
            if not isinstance(m, dict) or "top1_rmsd" not in m:
                continue
            rows.append({
                "complex":        cname,
                "pep_len":        m.get("pep_len", ""),
                "ss_class":       m.get("ss_class", ""),
                "length_bucket":  m.get("length_bucket", ""),
                "model":          label,
                "top1_rmsd":      m.get("top1_rmsd", float("nan")),
                "best_rmsd":      m.get("best_rmsd", float("nan")),
                "hit_at1":        m.get("hit_at1", float("nan")),
                hit_atk_key:      m.get(hit_atk_key, float("nan")),
                "tau":            m.get("tau", float("nan")),
                "rho":            m.get("rho", float("nan")),
                "diversity":      m.get("diversity", float("nan")),
                "n_poses_ok":     m.get("n_poses_ok", 0),
            })
    df = pd.DataFrame(rows)
    csv_path = out_dir / "benchmark_summary.csv"
    df.to_csv(csv_path, index=False)
    log.info("Wrote %s", csv_path)

    # aggregate JSON
    agg_path = out_dir / "aggregate_stats.json"
    agg_path.write_text(json.dumps(agg, indent=2, default=str))
    log.info("Wrote %s", agg_path)

    # console summary table
    print("\n" + "=" * 90)
    print(f"POST-POCKET-FIX BENCHMARK — {len(all_results)} complexes, ref2015 ranking")
    print("=" * 90)
    hdr = f"{'model':<12} {'n':>4} {'top1Å':>7} {'hit@1':>6} {hit_atk_key:>8} {'bestÅ':>7} {'τ':>7} {'ρ':>7}"
    print(hdr)
    print("-" * 90)
    for label in MODEL_ORDER:
        if label not in agg:
            continue
        a = agg[label]
        print(
            f"{label:<12} {a['n_complexes']:>4} "
            f"{a['mean_top1_rmsd']:>7.2f} "
            f"{a['hit_at1']:>6.1%} "
            f"{a[hit_atk_key]:>8.1%} "
            f"{a['mean_best_rmsd']:>7.2f} "
            f"{a['mean_tau']:>7.3f} "
            f"{a['mean_rho']:>7.3f}"
        )

    # Per-SS summary for pretrained and v5c (best vs baseline)
    for ss_label in ["HELIX", "SHEET", "UNUSUAL"]:
        print(f"\n── {ss_label} ──")
        for label in ["pretrained", "v5c"]:
            if label not in agg:
                continue
            ss = agg[label]["ss_breakdown"].get(ss_label, {})
            if ss:
                print(
                    f"  {label:<12}  top1={ss['mean_top1_rmsd']:.2f}Å  "
                    f"hit@1={ss['hit_at1']:.1%}  {hit_atk_key}={ss[hit_atk_key]:.1%}  "
                    f"best={ss['mean_best_rmsd']:.2f}Å"
                )
    print("=" * 90)

    # Optionally write plots
    _try_write_plots(agg, out_dir, hit_atk_key)


def _try_write_plots(agg: dict, out_dir: Path, hit_atk_key: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        models_present = [m for m in MODEL_ORDER if m in agg]
        metrics = ["mean_top1_rmsd", "hit_at1", hit_atk_key, "mean_best_rmsd", "mean_tau"]
        labels_map = {
            "mean_top1_rmsd": "Top-1 RMSD (Å)",
            "hit_at1": "Hit@1 (top-1 ≤2Å)",
            hit_atk_key: f"Hit@{hit_atk_key[-1]} (any top-k ≤2Å)",
            "mean_best_rmsd": "Oracle best RMSD (Å)",
            "mean_tau": "Kendall τ (ref2015)",
        }

        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
        for ax, metric in zip(axes, metrics):
            vals = [agg[m].get(metric, float("nan")) for m in models_present]
            colors = ["steelblue" if m == "pretrained" else "tomato" if m == "v5c"
                      else "lightgray" for m in models_present]
            bars = ax.bar(range(len(models_present)), vals, color=colors, edgecolor="white")
            ax.set_xticks(range(len(models_present)))
            ax.set_xticklabels(models_present, rotation=45, ha="right", fontsize=8)
            ax.set_title(labels_map[metric], fontsize=9)
            ax.set_xlabel("")
            if "rmsd" in metric.lower():
                ax.set_ylabel("Å")
            elif "hit" in metric.lower():
                ax.set_ylabel("fraction")
                ax.set_ylim(0, 1.1)
            for bar, val in zip(bars, vals):
                if not math.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01, f"{val:.2f}",
                            ha="center", va="bottom", fontsize=7)

        fig.suptitle("Post-Pocket-Fix Model Comparison (ref2015 ranking)", fontsize=11)
        plt.tight_layout()
        plot_path = out_dir / "model_comparison.png"
        plt.savefig(plot_path, dpi=130, bbox_inches="tight")
        plt.close()
        log.info("Wrote %s", plot_path)
    except Exception as exc:
        log.debug("Plotting failed (non-fatal): %s", exc)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-samples",   type=int,   default=25,
                    help="Poses per model per complex (default 25)")
    ap.add_argument("--n-per-cell",  type=int,   default=5,
                    help="Complexes per (SS × length) cell (default 5 → 60 total)")
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--out-dir",     default="logs/analysis_post_pocket_fix")
    ap.add_argument("--skip-ref2015", action="store_true",
                    help="Skip PyRosetta ref2015 scoring (pose-quality metrics only)")
    ap.add_argument("--models",      nargs="*",
                    default=MODEL_ORDER,
                    choices=list(MODEL_CKPTS.keys()),
                    help="Subset of models to evaluate (default: all 8)")
    ap.add_argument("--limit",       type=int,   default=None,
                    help="Cap total complexes evaluated (debugging)")
    ap.add_argument("--n-topk",      type=int,   default=5,
                    help="k for hit@k metric (default 5)")
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
    log.info("Post-pocket-fix eval: n_samples=%d, n_per_cell=%d, seed=%d",
             args.n_samples, args.n_per_cell, args.seed)

    # Validate rapidock infrastructure
    if not RAPIDOCK_DIR.exists():
        log.error("RAPIDOCK_DIR not found: %s", RAPIDOCK_DIR)
        sys.exit(1)
    if not RUN_SHIM.exists():
        log.error("run_rapidock.py shim not found: %s", RUN_SHIM)
        sys.exit(1)
    if not BASE_MODEL_DIR.exists():
        log.error("Base model dir not found: %s", BASE_MODEL_DIR)
        sys.exit(1)

    rapidock_python = _find_rapidock_python()
    log.info("rapidock python: %s", rapidock_python)

    # Load & stratify benchmark set
    df = pd.read_csv(BENCH300_CSV)
    bench_subset = stratified_sample(df, n_per_cell=args.n_per_cell, seed=args.seed)
    if args.limit:
        bench_subset = bench_subset.head(args.limit)
    log.info("Benchmark: %d complexes (target %d = %d cells × %d per cell)",
             len(bench_subset), args.n_per_cell * 12, 12, args.n_per_cell)

    # Save the selected complex list for reproducibility
    selected_csv = out_dir / "selected_complexes.csv"
    bench_subset.to_csv(selected_csv, index=False)

    # Build model map (only requested, only existing checkpoints)
    models: dict[str, Path] = {}
    for label in args.models:
        ckpt = MODEL_CKPTS.get(label)
        if ckpt is None:
            log.warning("Unknown model: %s — skipping", label)
            continue
        if not ckpt.exists():
            log.warning("Checkpoint not found for %s: %s — skipping", label, ckpt)
            continue
        models[label] = ckpt
    log.info("Models: %s", list(models.keys()))

    # Init PyRosetta once before the loop (expensive, ~30s)
    if not args.skip_ref2015:
        pr, sfxn = _init_pyrosetta()
        if pr is None:
            log.warning("PyRosetta unavailable — running without ref2015 scoring")
            args.skip_ref2015 = True

    all_results: dict[str, dict[str, dict]] = {}
    t_start = time.time()

    with tempfile.TemporaryDirectory(prefix="postpocket_ref2015_") as tmp:
        tmp_dir = Path(tmp)

        for ci, row in bench_subset.iterrows():
            cname = row["name"]
            complex_out = out_dir / "per_complex" / cname
            complex_out.mkdir(parents=True, exist_ok=True)

            log.info("─── [%d/%d] %s  (%s/%s len=%d) ───",
                     list(bench_subset.index).index(ci) + 1,
                     len(bench_subset),
                     cname, row["ss_class"], row["length_bucket"], row["pep_len"])

            results = eval_complex(
                row=row,
                out_dir=complex_out,
                models=models,
                n_samples=args.n_samples,
                seed=args.seed,
                rapidock_python=rapidock_python,
                skip_ref2015=args.skip_ref2015,
                tmp_dir=tmp_dir,
            )
            all_results[cname] = results

            # Checkpoint intermediate results periodically (every 5 complexes)
            if (list(bench_subset.index).index(ci) + 1) % 5 == 0:
                _checkpoint_json(all_results, out_dir)

    elapsed = time.time() - t_start
    log.info("Total wall-clock: %.1f s (%.1f min)", elapsed, elapsed / 60)

    # Final aggregation + output
    agg = aggregate(all_results, n_topk=args.n_topk)
    write_outputs(all_results, agg, bench_subset, out_dir, n_topk=args.n_topk)

    # Full results JSON
    full_json = out_dir / "full_results.json"
    full_json.write_text(json.dumps(all_results, indent=2, default=str))
    log.info("Full results: %s", full_json)
    log.info("Done.")


def _checkpoint_json(all_results: dict, out_dir: Path) -> None:
    p = out_dir / "full_results_partial.json"
    p.write_text(json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
