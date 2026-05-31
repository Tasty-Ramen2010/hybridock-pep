#!/usr/bin/env python3
"""Ranking benchmark: PyRosetta ref2015 + FastRelax on bench300 poses.

Same as rank_comparison_ref2015.py but adds a FastRelax step (peptide chain
only, receptor fixed) before scoring.  FastRelax relieves intra-peptide
clashes without moving the receptor, giving a fairer ref2015 score.

Timing: ~0.6s per pose at max_iter=10 (vs 0.31s no-relax).

Usage (score-env — PyRosetta symlinked):
    conda run -n score-env python3 scripts/rank_comparison_ref2015_relax.py \
        --n-per-bucket 15 --out-dir logs/ref2015_relax_ranking --seed 42
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
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings_imported = False
try:
    import warnings
    warnings.filterwarnings("ignore")
    warnings_imported = True
except Exception:
    pass

log = logging.getLogger("ref2015relax")

REPO      = Path(__file__).resolve().parent.parent
BENCH300  = REPO / "logs" / "analysis_bench300" / "benchmark_results.json"
CSV300    = REPO / "data" / "benchmark300.csv"


# ── PyRosetta init with FastRelax ─────────────────────────────────────────────

def init_pyrosetta(max_iter: int = 10):
    """Init PyRosetta and build a reusable FastRelax mover (peptide chain P only)."""
    import pyrosetta
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
    from pyrosetta.rosetta.core.select import residue_selector as selections
    from pyrosetta.rosetta.core.select.movemap import MoveMapFactory, move_map_action

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

    # Build FastRelax mover (will be reused per-pose with fresh movemap)
    fr = FastRelax()
    fr.set_scorefxn(sfxn)
    fr.max_iter(max_iter)
    fr.constrain_coords(True)  # keep peptide near original position

    # Task: allow repacking of peptide only
    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.RestrictToRepacking())
    # Prevent repacking of all residues first, then allow chain P
    all_sel = selections.TrueResidueSelector()
    pep_sel = selections.ChainSelector("P")
    tf.push_back(operation.OperateOnResidueSubset(operation.PreventRepackingRLT(), all_sel))
    fr.set_task_factory(tf)

    log.info("PyRosetta + FastRelax (max_iter=%d) initialised", max_iter)
    return pyrosetta, sfxn, fr


def score_pose_with_relax(pyrosetta, sfxn, fr, receptor_pdb: Path,
                           pose_pdb: Path) -> float | None:
    """FastRelax peptide chain P then score combined complex with ref2015."""
    try:
        from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
        from pyrosetta.rosetta.core.select import residue_selector as selections
        from pyrosetta.rosetta.core.select.movemap import MoveMapFactory, move_map_action

        protein_pose = pyrosetta.pose_from_pdb(str(receptor_pdb))
        peptide_pose = pyrosetta.pose_from_pdb(str(pose_pdb))
        peptide_pose.pdb_info().set_chains("P")
        protein_pose.append_pose_by_jump(peptide_pose, protein_pose.total_residue())

        # Build per-pose movemap (jump selector needs total_residue, varies per complex)
        jr_sel = pyrosetta.rosetta.core.select.jump_selector.JumpForResidue(
            protein_pose.total_residue()
        )
        all_sel = selections.TrueResidueSelector()
        pep_sel = selections.ChainSelector("P")

        mmf = MoveMapFactory()
        mmf.add_bb_action(move_map_action.mm_disable, all_sel)
        mmf.add_chi_action(move_map_action.mm_disable, all_sel)
        mmf.add_bb_action(move_map_action.mm_enable, pep_sel)
        mmf.add_chi_action(move_map_action.mm_enable, pep_sel)
        mmf.add_jump_action(move_map_action.mm_enable, jr_sel)
        mm = mmf.create_movemap_from_pose(protein_pose)
        fr.set_movemap(mm)

        pose = protein_pose.clone()
        fr.apply(pose)
        return float(sfxn(pose))

    except Exception as e:
        log.debug("FastRelax scoring failed: %s", e)
        return None


# ── PDB / geometry helpers ────────────────────────────────────────────────────

def pdb_coords(pdb_path: Path) -> np.ndarray:
    pts = []
    for ln in pdb_path.read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        try:
            pts.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
    return np.array(pts) if pts else np.zeros((0, 3))


# ── ranking metrics ───────────────────────────────────────────────────────────

def ranking_metrics(scores, rmsds):
    import math
    paired = [(s, r) for s, r in zip(scores, rmsds)
              if s is not None and not math.isnan(s)]
    if len(paired) < 2:
        nan = float("nan")
        return nan, nan, nan, nan, nan, nan, nan, nan
    s_arr = np.array([p[0] for p in paired])
    r_arr = np.array([p[1] for p in paired])
    tau, _ = scipy_stats.kendalltau(s_arr, r_arr)
    rho, _ = scipy_stats.spearmanr(s_arr, r_arr)
    best_idx    = np.argmin(s_arr)
    top1_rmsd   = float(r_arr[best_idx])
    random_mean = float(r_arr.mean())
    best_rmsd   = float(r_arr.min())
    oracle_gap  = random_mean - best_rmsd
    achieved    = random_mean - top1_rmsd
    gap_rec     = achieved / oracle_gap if abs(oracle_gap) > 1e-6 else 0.0
    p_best      = float(np.argmin(s_arr) == np.argmin(r_arr))
    return top1_rmsd, random_mean, best_rmsd, oracle_gap, tau, rho, p_best, gap_rec


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-bucket", type=int, default=15)
    ap.add_argument("--out-dir", default="logs/ref2015_relax_ranking")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-iter", type=int, default=10,
                    help="FastRelax iterations (default 10; original RAPiDock uses 20)")
    ap.add_argument("--model", default="pretrained")
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

    with open(BENCH300) as f:
        bench300 = json.load(f)
    df300 = pd.read_csv(CSV300)
    receptor_map = dict(zip(df300["name"], df300["receptor"]))

    # Same selection as ref2015 no-relax (12 buckets × 15)
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

    log.info("Running ref2015 + FastRelax(iter=%d) on %d complexes",
             args.max_iter, len(selected))

    pr, sfxn, fr = init_pyrosetta(args.max_iter)

    results = {}
    t_total = time.time()

    for ci, cname in enumerate(selected):
        t0 = time.time()
        row = df300[df300["name"] == cname].iloc[0]
        lb  = row["length_bucket"]
        ss  = row["ss_class"]

        receptor_pdb = Path(receptor_map.get(cname, ""))
        if not receptor_pdb.exists():
            log.warning("[%d/%d] %s — receptor not found", ci + 1, len(selected), cname)
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
            sc = score_pose_with_relax(pr, sfxn, fr, receptor_pdb, pose_pdb)
            scores.append(sc)

        rmsds = ref_rmsds[:n_poses]
        (top1, rand, best, gap, tau, rho, p_best, gap_rec) = ranking_metrics(scores, rmsds)

        sc_str = " ".join(f"{s:.1f}" if s is not None else "NaN" for s in scores)
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
            "lb": lb, "ss": ss, "scores": scores, "ref_rmsds": rmsds[:n_poses],
            "n_valid": sum(1 for s in scores if s is not None),
        }

    results_path = out_dir / "ranking_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: None if x != x else x)
    log.info("Saved → %s", results_path)

    # Aggregate
    from collections import defaultdict
    bk_res: dict[str, list] = defaultdict(list)
    for v in results.values():
        if isinstance(v, dict) and "kendall_tau" in v:
            bk_res[v.get("lb", "?")].append(v)
            bk_res[v.get("ss", "?")].append(v)
            bk_res["all"].append(v)

    def agg(vals):
        def mn(k):
            arr = [x[k] for x in vals
                   if isinstance(x.get(k), float) and not np.isnan(x[k])]
            return np.mean(arr) if arr else float("nan"), len(arr)
        return {k: mn(k)[0] for k in ["kendall_tau", "spearman_r", "p_select_best",
                                       "top1_rmsd", "gap_recovered_frac"]}

    log.info("\n=== ref2015 + FastRelax(%d) Ranking Results ===", args.max_iter)
    log.info("%-20s %4s %7s %7s %8s %6s %8s",
             "Bucket", "N", "τ", "ρ", "P(best)", "top1", "gap_rec")
    log.info("-" * 70)
    for bk in ["all", "short", "medium", "long", "very_long",
               "HELIX", "SHEET", "UNUSUAL"]:
        if bk in bk_res:
            a = agg(bk_res[bk])
            n = len(bk_res[bk])
            log.info("%-20s %4d %7.3f %7.3f %8.1f%% %6.2f %8.1f%%",
                     bk, n, a["kendall_tau"], a["spearman_r"],
                     a["p_select_best"] * 100, a["top1_rmsd"],
                     a["gap_recovered_frac"] * 100)

    log.info("Total wall: %.0f min", (time.time() - t_total) / 60)


if __name__ == "__main__":
    main()
