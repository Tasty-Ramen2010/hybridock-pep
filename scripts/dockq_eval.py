#!/usr/bin/env python3
"""
dockq_eval.py — DockQ/CAPRI evaluation of v3c, v4c, v5c, and pretrained models
on all bench300 complexes.

For each complex × model: run DockQ on every pose, report per-pose scores,
and summarise into CAPRI categories (Incorrect / Acceptable / Medium / High).

DockQ CAPRI thresholds (v2):
  Incorrect:   DockQ < 0.23
  Acceptable:  0.23 ≤ DockQ < 0.49
  Medium:      0.49 ≤ DockQ < 0.80
  High:        DockQ ≥ 0.80

Usage (rapidock env):
  /home/igem/miniconda3/envs/rapidock/bin/python3 -u scripts/dockq_eval.py \
      --out-dir logs/dockq_eval [--workers 8] [--models v3c v4c v5c pretrained]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
BENCH_JSON     = ROOT / "logs/analysis_bench300/benchmark_results.json"
TRAINING_DIR   = ROOT / "datasets/training_formatted_peppc"

CAPRI_THRESHOLDS = [
    ("high",       0.80, 1.01),
    ("medium",     0.49, 0.80),
    ("acceptable", 0.23, 0.49),
    ("incorrect",  0.00, 0.23),
]

def capri_category(dockq: float) -> str:
    for label, lo, hi in CAPRI_THRESHOLDS:
        if lo <= dockq < hi:
            return label
    return "incorrect"


def _build_combined_pdb(rec_path: str, pep_path: str) -> str:
    """Concatenate receptor + peptide PDBs into a temp file; return path."""
    fd, tmp = tempfile.mkstemp(suffix=".pdb")
    with os.fdopen(fd, "w") as f:
        rec_text = Path(rec_path).read_text().rstrip()
        pep_text = Path(pep_path).read_text().rstrip()
        # Strip END records before concatenating so Bio.PDB doesn't stop early
        rec_text = "\n".join(
            l for l in rec_text.splitlines() if not l.startswith("END")
        )
        pep_text = "\n".join(
            l for l in pep_text.splitlines() if not l.startswith("END")
        )
        f.write(rec_text + "\nTER\n" + pep_text + "\nEND\n")
    return tmp


def run_dockq_on_pose(
    pose_pdb: str,
    rec_ref: str,
    pep_ref: str,
) -> dict[str, Any]:
    """Return DockQ metrics for a single pose vs crystal structure."""
    from DockQ.DockQ import run_on_all_native_interfaces, load_PDB

    native_tmp = _build_combined_pdb(rec_ref, pep_ref)
    model_tmp  = _build_combined_pdb(rec_ref, pose_pdb)
    try:
        model_s  = load_PDB(model_tmp)
        native_s = load_PDB(native_tmp)
        all_results, best_dockq = run_on_all_native_interfaces(
            model_s, native_s, capri_peptide=True
        )
        # Find the interface that involves the peptide chain (chain A)
        pep_iface = None
        for iface_key, iface_data in all_results.items():
            if iface_data.get("class1") == "ligand" or iface_data.get("chain1") == "A":
                pep_iface = iface_data
                break
        if pep_iface is None and all_results:
            # Fall back to the interface with highest DockQ
            pep_iface = max(all_results.values(), key=lambda x: x["DockQ"])
        if pep_iface is None:
            return {"dockq": float("nan"), "irmsd": float("nan"),
                    "lrmsd": float("nan"), "fnat": float("nan"),
                    "capri": "incorrect", "error": "no_interface"}
        dq = float(pep_iface["DockQ"])
        return {
            "dockq":  dq,
            "irmsd":  float(pep_iface.get("iRMSD", float("nan"))),
            "lrmsd":  float(pep_iface.get("LRMSD", float("nan"))),
            "fnat":   float(pep_iface.get("fnat",  float("nan"))),
            "capri":  capri_category(dq),
            "error":  None,
        }
    except Exception as exc:
        return {"dockq": float("nan"), "irmsd": float("nan"),
                "lrmsd": float("nan"), "fnat": float("nan"),
                "capri": "incorrect", "error": str(exc)}
    finally:
        os.unlink(native_tmp)
        os.unlink(model_tmp)


def _worker(job: dict) -> dict:
    """Process one (complex, model) pair — compute DockQ for all poses."""
    cx_name   = job["cx_name"]
    model     = job["model"]
    poses_dir = Path(job["poses_dir"])
    rec_ref   = job["rec_ref"]
    pep_ref   = job["pep_ref"]
    ref_rmsds = job["ref_rmsds"]

    pose_files = sorted(poses_dir.glob("pose_*.pdb"),
                        key=lambda p: int(p.stem.split("_")[1]))
    if not pose_files:
        return {"cx_name": cx_name, "model": model, "error": "no_poses",
                **{k: float("nan") for k in
                   ["best_dockq", "top1_dockq", "mean_dockq",
                    "best_irmsd", "top1_irmsd", "best_lrmsd", "top1_lrmsd",
                    "best_fnat", "top1_fnat"]},
                "best_capri": "incorrect", "top1_capri": "incorrect",
                "n_poses": 0,
                "n_acceptable_plus": 0, "n_medium_plus": 0, "n_high": 0}

    pose_results = []
    for i, pf in enumerate(pose_files):
        res = run_dockq_on_pose(str(pf), rec_ref, pep_ref)
        res["pose_idx"] = i
        res["rmsd"]     = float(ref_rmsds[i]) if i < len(ref_rmsds) else float("nan")
        pose_results.append(res)

    dockq_vals = [r["dockq"] for r in pose_results if not np.isnan(r["dockq"])]
    if not dockq_vals:
        best_idx = top1_idx = 0
    else:
        best_idx = int(np.argmax([r["dockq"] for r in pose_results]))
        top1_idx = 0  # pose_0 = top-ranked by RAPiDock

    best = pose_results[best_idx]
    top1 = pose_results[top1_idx]

    capri_counts = {cat: 0 for cat, *_ in CAPRI_THRESHOLDS}
    for r in pose_results:
        if not np.isnan(r["dockq"]):
            capri_counts[r["capri"]] = capri_counts.get(r["capri"], 0) + 1

    return {
        "cx_name":         cx_name,
        "model":           model,
        "n_poses":         len(pose_results),
        "best_dockq":      best["dockq"],
        "best_irmsd":      best["irmsd"],
        "best_lrmsd":      best["lrmsd"],
        "best_fnat":       best["fnat"],
        "best_capri":      best["capri"],
        "best_rmsd":       best["rmsd"],
        "top1_dockq":      top1["dockq"],
        "top1_irmsd":      top1["irmsd"],
        "top1_lrmsd":      top1["lrmsd"],
        "top1_fnat":       top1["fnat"],
        "top1_capri":      top1["capri"],
        "top1_rmsd":       top1["rmsd"],
        "mean_dockq":      float(np.mean(dockq_vals)) if dockq_vals else float("nan"),
        "n_incorrect":     capri_counts.get("incorrect", 0),
        "n_acceptable":    capri_counts.get("acceptable", 0),
        "n_medium":        capri_counts.get("medium", 0),
        "n_high":          capri_counts.get("high", 0),
        "n_acceptable_plus": capri_counts.get("acceptable", 0) + capri_counts.get("medium", 0) + capri_counts.get("high", 0),
        "n_medium_plus":     capri_counts.get("medium", 0) + capri_counts.get("high", 0),
        "error":           None,
    }


def build_jobs(bench_json: dict, models: list[str]) -> list[dict]:
    jobs = []
    for cx_name, model_data in bench_json.items():
        rec_ref = str(TRAINING_DIR / cx_name / f"{cx_name}_protein_pocket.pdb")
        pep_ref = str(TRAINING_DIR / cx_name / f"{cx_name}_peptide.pdb")
        if not (Path(rec_ref).exists() and Path(pep_ref).exists()):
            log.warning("Missing crystal refs for %s — skipping", cx_name)
            continue
        for model in models:
            if model not in model_data:
                continue
            mdata = model_data[model]
            poses_dir = mdata.get("poses_dir", "")
            if not poses_dir or not Path(poses_dir).exists():
                log.warning("Missing poses dir for %s/%s", cx_name, model)
                continue
            jobs.append({
                "cx_name":   cx_name,
                "model":     model,
                "poses_dir": poses_dir,
                "rec_ref":   rec_ref,
                "pep_ref":   pep_ref,
                "ref_rmsds": mdata.get("ref_rmsds", []),
            })
    return jobs


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model aggregate summary."""
    rows = []
    for model, grp in df.groupby("model"):
        n = len(grp)
        # Best-of-N CAPRI rates (oracle)
        best_acc_plus  = (grp["best_capri"].isin(["acceptable","medium","high"])).mean()
        best_med_plus  = (grp["best_capri"].isin(["medium","high"])).mean()
        best_high      = (grp["best_capri"] == "high").mean()
        # Top-1 CAPRI rates (pose_0)
        top1_acc_plus  = (grp["top1_capri"].isin(["acceptable","medium","high"])).mean()
        top1_med_plus  = (grp["top1_capri"].isin(["medium","high"])).mean()
        top1_high      = (grp["top1_capri"] == "high").mean()

        rows.append({
            "model":               model,
            "n_complexes":         n,
            # Oracle (best pose) stats
            "best_mean_dockq":     grp["best_dockq"].mean(),
            "best_acc+_rate":      best_acc_plus,
            "best_med+_rate":      best_med_plus,
            "best_high_rate":      best_high,
            # Top-1 pose stats
            "top1_mean_dockq":     grp["top1_dockq"].mean(),
            "top1_acc+_rate":      top1_acc_plus,
            "top1_med+_rate":      top1_med_plus,
            "top1_high_rate":      top1_high,
            # RMSD reference
            "best_mean_rmsd":      grp["best_rmsd"].mean(),
            "top1_mean_rmsd":      grp["top1_rmsd"].mean(),
            # iRMSD / fnat
            "best_mean_irmsd":     grp["best_irmsd"].mean(),
            "best_mean_fnat":      grp["best_fnat"].mean(),
        })
    return pd.DataFrame(rows).sort_values("top1_mean_dockq", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",  default="logs/dockq_eval")
    ap.add_argument("--workers",  type=int, default=8)
    ap.add_argument("--models",   nargs="+",
                    default=["pretrained", "v3c", "v4c", "v5c"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading bench300 results...")
    with open(BENCH_JSON) as f:
        bench_json = json.load(f)

    jobs = build_jobs(bench_json, args.models)
    log.info("Jobs: %d  (%d complexes × %d models, filtered by availability)",
             len(jobs), len(bench_json), len(args.models))

    rows: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, j): j for j in jobs}
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as exc:
                j = futures[fut]
                log.error("FAILED %s/%s: %s", j["cx_name"], j["model"], exc)
            done += 1
            if done % 50 == 0:
                log.info("  %d / %d done", done, len(jobs))

    df = pd.DataFrame(rows)
    pose_csv = out_dir / "dockq_per_complex.csv"
    df.to_csv(pose_csv, index=False)
    log.info("Per-complex results → %s", pose_csv)

    summary = summarise(df)
    summary_csv = out_dir / "dockq_summary.csv"
    summary.to_csv(summary_csv, index=False)

    log.info("\n=== DockQ/CAPRI Summary ===")
    log.info("Metric        | %s",
             " | ".join(f"{m:>12}" for m in summary["model"]))
    for col in ["top1_mean_dockq", "top1_acc+_rate", "top1_med+_rate",
                "best_mean_dockq", "best_acc+_rate", "best_med+_rate"]:
        vals = " | ".join(f"{v:>12.3f}" for v in summary[col])
        log.info("%-22s| %s", col, vals)
    log.info("Summary → %s", summary_csv)


if __name__ == "__main__":
    main()
