#!/usr/bin/env python3
"""Ranking benchmark: PyRosetta ref2015 scoring on bench300 poses.

Hypothesis: ref2015 Rosetta energy function captures sidechain packing
and electrostatics missed by Vina, potentially giving better pose discrimination.

Protocol per complex:
  1. Combine receptor pocket PDB + pose PDB into a single complex.
  2. Score with ref2015 WITHOUT FastRelax (score-only, same spirit as Vina --score_only).
  3. Rank 5 poses by ref2015 score (lower = better).
  4. Compute ranking metrics vs RMSD labels.

Note: FastRelax scoring (relax_score in pyrosetta_utils.py) is NOT used here
because it modifies the pose, making it a different experiment from plain scoring.
For an apples-to-apples comparison with Vina score_only, we score the original
diffusion pose directly.

Usage (base conda env — PyRosetta is installed there):
    python3 scripts/rank_comparison_ref2015.py \
        --n-per-bucket 15 --out-dir logs/ref2015_ranking --seed 42

Requires: pyrosetta installed (base env), bench300 data at logs/analysis_bench300/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
log = logging.getLogger("ref2015rank")

REPO      = Path(__file__).resolve().parent.parent
BENCH300  = REPO / "logs" / "analysis_bench300" / "benchmark_results.json"
CSV300    = REPO / "data" / "benchmark300.csv"
MODELS_IN_BENCH = ["pretrained"]  # same as other ranking experiments


# ── PyRosetta init ────────────────────────────────────────────────────────────

def init_pyrosetta():
    """Initialise PyRosetta (muted) and return the ref2015 score function."""
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
                "-in:file:silent_struct_type", "binary",
            ]),
            silent=True,
        )
        sfxn = pyrosetta.create_score_function("ref2015")
        log.info("PyRosetta initialised, ref2015 ready")
        return pyrosetta, sfxn
    except Exception as e:
        log.error("PyRosetta init failed: %s", e)
        sys.exit(1)


# ── PDB parsing helpers ───────────────────────────────────────────────────────

def pdb_coords(pdb_path: Path) -> np.ndarray:
    """Return Nx3 heavy-atom coordinates from PDB."""
    pts = []
    for ln in pdb_path.read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        try:
            pts.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
    return np.array(pts) if pts else np.zeros((0, 3))


def merge_receptor_and_pose(receptor_pdb: Path, pose_pdb: Path,
                             out_pdb: Path) -> bool:
    """Merge receptor (chain A) and peptide pose (chain P) into one PDB.

    Reassigns the peptide to chain P to avoid clash with receptor chains.
    Strips any existing ENDMDL/END and writes END at the bottom.
    """
    rec_lines = []
    for ln in receptor_pdb.read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            rec_lines.append(ln)

    pep_lines = []
    n = 0
    for ln in pose_pdb.read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            # reassign chain to P (col 21, 0-indexed)
            ln = ln[:21] + "P" + ln[22:]
            pep_lines.append(ln)
            n += 1

    if not rec_lines or not pep_lines:
        return False

    out_pdb.write_text("\n".join(rec_lines + ["TER"] + pep_lines + ["END\n"]))
    return True


# ── ref2015 scoring ───────────────────────────────────────────────────────────

def score_pose_ref2015(pyrosetta, sfxn, receptor_pdb: Path, pose_pdb: Path,
                        tmp_dir: Path) -> float | None:
    """Score receptor+pose complex with ref2015 (no relaxation).

    Returns the total ref2015 score (lower = better) or None on failure.
    """
    complex_pdb = tmp_dir / f"complex_{os.getpid()}_{time.time_ns()}.pdb"
    try:
        if not merge_receptor_and_pose(receptor_pdb, pose_pdb, complex_pdb):
            return None
        pose = pyrosetta.pose_from_pdb(str(complex_pdb))
        score = sfxn(pose)
        return float(score)
    except Exception as e:
        log.debug("ref2015 scoring failed: %s", e)
        return None
    finally:
        try:
            complex_pdb.unlink()
        except Exception:
            pass


# ── ranking metrics (copied from rank_comparison_vina.py) ────────────────────

def ranking_metrics(scores: list[float | None], rmsds: list[float]):
    """Compute ranking quality metrics given scores (lower=better) and oracle RMSDs."""
    import math
    paired = [(s, r) for s, r in zip(scores, rmsds)
              if s is not None and not math.isnan(s)]
    if len(paired) < 2:
        nan = float("nan")
        return nan, nan, nan, nan, nan, nan, nan, nan

    s_arr = np.array([p[0] for p in paired])
    r_arr = np.array([p[1] for p in paired])

    tau, _   = scipy_stats.kendalltau(s_arr, r_arr)
    rho, _   = scipy_stats.spearmanr(s_arr, r_arr)

    best_idx   = np.argmin(s_arr)   # pose with lowest (best) score
    top1_rmsd  = float(r_arr[best_idx])
    random_mean = float(r_arr.mean())
    best_rmsd  = float(r_arr.min())
    oracle_gap = random_mean - best_rmsd
    achieved   = random_mean - top1_rmsd
    gap_rec    = achieved / oracle_gap if abs(oracle_gap) > 1e-6 else 0.0
    p_best     = float(np.argmin(s_arr) == np.argmin(r_arr))

    return top1_rmsd, random_mean, best_rmsd, oracle_gap, tau, rho, p_best, gap_rec


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-bucket", type=int, default=15,
                    help="Complexes per (length×SS) bucket (default 15)")
    ap.add_argument("--out-dir", default="logs/ref2015_ranking")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="pretrained",
                    choices=["pretrained", "v5c", "v3c", "v4c"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "run.log"),
        ],
    )

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # Load bench300 data
    with open(BENCH300) as f:
        bench300 = json.load(f)
    df300 = pd.read_csv(CSV300)
    receptor_map = dict(zip(df300["name"], df300["receptor"]))

    # Select complexes (same 15-per-bucket strategy as other benchmarks)
    buckets: dict[str, list[str]] = {}
    for _, row in df300.iterrows():
        bk = f"{row['length_bucket']}/{row['ss_class']}"
        buckets.setdefault(bk, []).append(row["name"])

    selected: list[str] = []
    for bk, names in sorted(buckets.items()):
        available = [n for n in names if n in bench300
                     and args.model in bench300[n]
                     and len(bench300[n][args.model].get("ref_rmsds", [])) >= 2]
        rng.shuffle(available)
        selected.extend(available[:args.n_per_bucket])

    log.info("Running ref2015 on %d complexes", len(selected))

    # Init PyRosetta
    pr, sfxn = init_pyrosetta()

    with tempfile.TemporaryDirectory(prefix="ref2015_") as tmp:
        tmp_dir = Path(tmp)
        results = {}
        t_total = time.time()

        for ci, cname in enumerate(selected):
            t0 = time.time()
            row = df300[df300["name"] == cname].iloc[0]
            lb  = row["length_bucket"]
            ss  = row["ss_class"]

            receptor_pdb = Path(receptor_map.get(cname, ""))
            if not receptor_pdb.exists():
                log.warning("[%d/%d] %s — receptor not found: %s",
                            ci + 1, len(selected), cname, receptor_pdb)
                continue

            model_data = bench300[cname].get(args.model, {})
            poses_dir  = Path(model_data.get("poses_dir", ""))
            ref_rmsds  = model_data.get("ref_rmsds", [])
            n_poses    = min(5, len(ref_rmsds))

            scores = []
            for i in range(n_poses):
                pose_pdb = poses_dir / f"pose_{i}.pdb"
                if not pose_pdb.exists():
                    scores.append(None)
                    continue
                sc = score_pose_ref2015(pr, sfxn, receptor_pdb, pose_pdb, tmp_dir)
                scores.append(sc)

            rmsds = ref_rmsds[:n_poses]
            (top1, rand, best, gap, tau, rho, p_best, gap_rec) = ranking_metrics(scores, rmsds)

            n_valid = sum(1 for s in scores if s is not None)
            sc_str  = " ".join(f"{s:.1f}" if s is not None else "NaN" for s in scores)
            log.info(
                "[%d/%d] %s [%s/%s] scores=[%s] τ=%.3f ρ=%.3f P(best)=%d%% rec=%.0f%%  t=%.0fs",
                ci + 1, len(selected), cname, lb, ss, sc_str,
                tau, rho, int(round(p_best * 100)), gap_rec * 100,
                time.time() - t0,
            )

            results[cname] = {
                "top1_rmsd": top1, "random_mean_rmsd": rand, "best_rmsd": best,
                "oracle_gap": gap, "kendall_tau": tau, "spearman_r": rho,
                "p_select_best": p_best, "gap_recovered_frac": gap_rec,
                "lb": lb, "ss": ss,
                "scores": scores, "ref_rmsds": rmsds[:n_poses],
                "n_valid": n_valid,
            }

        # Write results JSON
        results_path = out_dir / "ranking_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=lambda x: None if x != x else x)
        log.info("Saved results → %s", results_path)

        # Aggregate by bucket
        from collections import defaultdict
        buckets_res: dict[str, list] = defaultdict(list)
        for v in results.values():
            if isinstance(v, dict) and "kendall_tau" in v:
                buckets_res[v.get("lb", "?")].append(v)
                buckets_res[v.get("ss", "?")].append(v)
                buckets_res["all"].append(v)

        def agg(vals):
            def mn(k):
                arr = [x[k] for x in vals
                       if isinstance(x.get(k), float) and not np.isnan(x[k])]
                return np.mean(arr) if arr else float("nan"), len(arr)
            return {k: mn(k)[0] for k in
                    ["kendall_tau", "spearman_r", "p_select_best",
                     "top1_rmsd", "random_mean_rmsd", "gap_recovered_frac"]}

        log.info("\n=== ref2015 Ranking Results ===")
        log.info("%-20s %4s %7s %7s %8s %6s %8s",
                 "Bucket", "N", "τ", "ρ", "P(best)", "top1", "gap_rec")
        log.info("-" * 70)
        for bk in ["all", "short", "medium", "long", "very_long",
                   "HELIX", "SHEET", "UNUSUAL"]:
            if bk in buckets_res:
                a = agg(buckets_res[bk])
                n = len(buckets_res[bk])
                log.info("%-20s %4d %7.3f %7.3f %8.1f%% %6.2f %8.1f%%",
                         bk, n, a["kendall_tau"], a["spearman_r"],
                         a["p_select_best"] * 100, a["top1_rmsd"],
                         a["gap_recovered_frac"] * 100)

        wall = time.time() - t_total
        log.info("Total wall time: %.0f min", wall / 60)


if __name__ == "__main__":
    main()
