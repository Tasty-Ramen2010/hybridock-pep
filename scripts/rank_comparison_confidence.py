#!/usr/bin/env python3
"""Ranking benchmark: trained confidence model on bench300 poses.

Hypothesis: A confidence model trained to predict pose quality (pairwise
BPR ranking loss) should rank RAPiDock diffusion poses better than Vina.

Protocol per complex:
  1. Build HeteroData graph for each pose (same pipeline as training).
  2. Score each pose with the confidence model (higher = better quality).
  3. Rank 5 poses by confidence score → compute metrics vs RMSD labels.

Usage (rapidock env — same env as training):
    conda run -n rapidock python3 scripts/rank_comparison_confidence.py \
        --ckpt train_models/confidence_model/confidence_model.pt \
        --n-per-bucket 15 --out-dir logs/confidence_ranking --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

REPO      = Path(__file__).resolve().parent.parent
BENCH300  = REPO / "logs" / "analysis_bench300" / "benchmark_results.json"
CSV300    = REPO / "data" / "benchmark300.csv"
MODELS_IN_BENCH = ["pretrained"]

log = logging.getLogger("confrank")

sys.path.insert(0, str(REPO / "third_party" / "RAPiDock"))


# ── model loading ─────────────────────────────────────────────────────────────

def load_confidence_model(ckpt_path: Path, model_dir: Path, device):
    """Load trained confidence model from checkpoint.

    Supports both v1 (stock tiny MLP head) and v2 (128-unit head + dropout).
    Detection is automatic: v2 checkpoints carry a 'config' dict with 'hidden_dim'.
    """
    import math
    import yaml
    import torch.nn as nn
    from argparse import Namespace
    from models.model import ConfidenceModel

    with open(model_dir / "model_parameters.yml") as f:
        args = Namespace(**yaml.full_load(f))
    if hasattr(args, "rmsd_classification_cutoff"):
        delattr(args, "rmsd_classification_cutoff")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model", ckpt)
    config = ckpt.get("config", {})          # v2 checkpoints carry config dict

    model = ConfidenceModel(args)

    if config.get("hidden_dim"):
        # ── v2 checkpoint: replace head with ConfidencePredictorV2 ────────────
        hidden_dim = int(config["hidden_dim"])
        dropout    = float(config.get("dropout", 0.2))
        log.info("Detected v2 checkpoint (hidden_dim=%d, dropout=%.2f)", hidden_dim, dropout)

        # Infer input dim from current (stock) head's first linear layer
        in_dim = 256  # safe default
        try:
            for mod in model.encoder.confidence_predictor.modules():
                if isinstance(mod, nn.Linear):
                    in_dim = mod.in_features
                    break
        except Exception:
            pass

        class _V2Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                )
            def forward(self, x):
                return self.net(x)

        model.encoder.confidence_predictor = _V2Head()

    missing, unexpected = model.load_state_dict(state, strict=False)
    log.info("Loaded checkpoint — missing: %d, unexpected: %d", len(missing), len(unexpected))
    if missing:
        log.debug("First 5 missing: %s", missing[:5])
    if unexpected:
        log.debug("First 5 unexpected: %s", unexpected[:5])
    model.to(device)
    model.eval()
    return model


# ── graph building (reuses training pipeline) ─────────────────────────────────

def build_base_graphs(selected: list[str], bench300_df: pd.DataFrame,
                      tmp_dir: Path) -> dict[str, object]:
    """Build InferenceDataset graphs for selected complexes."""
    from utils.inference_utils import InferenceDataset

    names, receptors, peptides = [], [], []
    df_idx = bench300_df.set_index("name")

    for cname in selected:
        if cname not in df_idx.index:
            continue
        row = df_idx.loc[cname]
        names.append(cname)
        receptors.append(str(row["receptor"]))
        peptides.append(str(row["seq"]))

    log.info("Building InferenceDataset for %d complexes...", len(names))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ds = InferenceDataset(
        output_dir=str(tmp_dir),
        complex_name_list=names,
        protein_description_list=receptors,
        peptide_description_list=peptides,
        lm_embeddings=True,
        lm_embeddings_pep=False,
        conformation_type=None,
        conformation_partial="1:1:1",
    )

    graphs = {}
    for i, cname in enumerate(names):
        try:
            g = ds.get(i)
            graphs[cname] = g
        except Exception as e:
            log.warning("Graph build failed for %s: %s", cname, e)

    log.info("Built %d / %d base graphs", len(graphs), len(names))
    return graphs


def _load_pose_positions(pose_pdb: str) -> torch.Tensor:
    """Load heavy-atom positions from pose PDB via MDAnalysis."""
    import MDAnalysis
    u = MDAnalysis.Universe(pose_pdb)
    positions = []
    for res in u.residues:
        heavy = res.atoms.select_atoms("not type H")
        ca = heavy.select_atoms("name CA")
        if len(ca) == 0 or len(heavy) == 0:
            continue
        positions.append(heavy.positions.astype(np.float32))
    if not positions:
        raise ValueError(f"No heavy atoms found in {pose_pdb}")
    return torch.tensor(np.concatenate(positions, axis=0))


def _inject_and_center(base_graph, pose_positions: torch.Tensor):
    """Deep-copy graph, inject pose positions, center on receptor."""
    import copy
    g = copy.deepcopy(base_graph)
    n_graph = g["pep_a"].pos.shape[0]
    n_pose  = pose_positions.shape[0]
    if n_graph != n_pose:
        raise ValueError(f"Atom count mismatch: graph={n_graph}, pose={n_pose}")
    g["pep_a"].pos      = pose_positions.to(dtype=torch.float)
    g["pep_a"].orig_pos = pose_positions.to(dtype=torch.float)
    center = g["receptor"].pos.mean(dim=0, keepdim=True)
    g["receptor"].pos   -= center
    g["pep_a"].pos      -= center
    g["pep_a"].orig_pos -= center
    return g


def score_poses(model, base_graph, poses_dir: Path, n_poses: int,
                device, set_time_fn) -> list[float | None]:
    """Score n_poses poses for one complex. Returns confidence scores (higher=better)."""
    from torch_geometric.data import Batch

    graphs = []
    for i in range(n_poses):
        pose_pdb = poses_dir / f"pose_{i}.pdb"
        if not pose_pdb.exists():
            graphs.append(None)
            continue
        try:
            pos = _load_pose_positions(str(pose_pdb))
            g   = _inject_and_center(base_graph, pos)
            set_time_fn(g, 0, 0, 0, 0, 1, device="cpu")
            graphs.append(g)
        except Exception as e:
            log.debug("Pose %d build failed: %s", i, e)
            graphs.append(None)

    scores = [None] * n_poses
    valid_idx = [i for i, g in enumerate(graphs) if g is not None]
    if not valid_idx:
        return scores

    valid_graphs = [graphs[i] for i in valid_idx]
    batch = Batch.from_data_list(valid_graphs).to(device)

    with torch.no_grad():
        sc = model(batch).squeeze(-1).cpu().numpy()

    for rank, i in enumerate(valid_idx):
        scores[i] = float(sc[rank])

    return scores


# ── ranking metrics ───────────────────────────────────────────────────────────

def ranking_metrics(scores: list[float | None], rmsds: list[float]):
    """Compute metrics; confidence scores are HIGHER=BETTER so negate for tau/rho."""
    import math
    paired = [(s, r) for s, r in zip(scores, rmsds)
              if s is not None and not math.isnan(s)]
    if len(paired) < 2:
        nan = float("nan")
        return nan, nan, nan, nan, nan, nan, nan, nan

    s_arr = np.array([p[0] for p in paired])   # confidence (higher=better)
    r_arr = np.array([p[1] for p in paired])   # RMSD (lower=better)

    # Negate scores so "lower is better" for tau/rho (like Vina conventions)
    neg_s  = -s_arr
    tau, _ = scipy_stats.kendalltau(neg_s, r_arr)
    rho, _ = scipy_stats.spearmanr(neg_s, r_arr)

    best_idx   = np.argmax(s_arr)              # highest confidence = predicted best
    top1_rmsd  = float(r_arr[best_idx])
    random_mean = float(r_arr.mean())
    best_rmsd  = float(r_arr.min())
    oracle_gap = random_mean - best_rmsd
    achieved   = random_mean - top1_rmsd
    gap_rec    = achieved / oracle_gap if abs(oracle_gap) > 1e-6 else 0.0
    p_best     = float(np.argmax(s_arr) == np.argmin(r_arr))

    return top1_rmsd, random_mean, best_rmsd, oracle_gap, tau, rho, p_best, gap_rec


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="train_models/confidence_model/confidence_model.pt")
    ap.add_argument("--model-dir", default="train_models/confidence_model")
    ap.add_argument("--n-per-bucket", type=int, default=15)
    ap.add_argument("--out-dir", default="logs/confidence_ranking")
    ap.add_argument("--tmp-dir", default="/tmp/conf_train",
                    help="Temp dir for InferenceDataset cache; reuse training cache to skip ESM re-run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Load bench300 data
    with open(BENCH300) as f:
        bench300 = json.load(f)
    df300 = pd.read_csv(CSV300)

    # Select complexes (same strategy as other experiments)
    rng = random.Random(args.seed)
    buckets: dict[str, list[str]] = {}
    for _, row in df300.iterrows():
        bk = row["length_bucket"]
        buckets.setdefault(bk, []).append(row["name"])

    selected: list[str] = []
    for bk, names in sorted(buckets.items()):
        available = [n for n in names if n in bench300
                     and args.model in bench300[n]
                     and len(bench300[n][args.model].get("ref_rmsds", [])) >= 2]
        rng.shuffle(available)
        selected.extend(available[:args.n_per_bucket])

    log.info("Selected %d complexes for confidence benchmark", len(selected))

    # Load model
    model = load_confidence_model(Path(args.ckpt), Path(args.model_dir), device)

    # Build base graphs (runs ESM once for all complexes)
    from utils.diffusion_utils import set_time
    base_graphs = build_base_graphs(selected, df300, Path(args.tmp_dir))

    # Score and evaluate
    results = {}
    t_total = time.time()

    for ci, cname in enumerate(selected):
        t0 = time.time()
        row = df300[df300["name"] == cname].iloc[0]
        lb  = row["length_bucket"]
        ss  = row["ss_class"]

        base_graph = base_graphs.get(cname)
        if base_graph is None:
            log.warning("[%d/%d] %s — no base graph", ci + 1, len(selected), cname)
            continue

        model_data = bench300[cname].get(args.model, {})
        poses_dir  = Path(model_data.get("poses_dir", ""))
        ref_rmsds  = model_data.get("ref_rmsds", [])
        n_poses    = min(5, len(ref_rmsds))

        scores = score_poses(model, base_graph, poses_dir, n_poses, device, set_time)
        rmsds  = ref_rmsds[:n_poses]

        (top1, rand, best, gap, tau, rho, p_best, gap_rec) = ranking_metrics(scores, rmsds)

        n_valid = sum(1 for s in scores if s is not None)
        sc_str  = " ".join(f"{s:.3f}" if s is not None else "NaN" for s in scores)
        log.info(
            "[%d/%d] %s [%s/%s] scores=[%s] τ=%.3f ρ=%.3f P(best)=%d%% rec=%.0f%%  t=%.0fs",
            ci + 1, len(selected), cname, lb, ss, sc_str,
            tau, rho, int(round(p_best * 100)), gap_rec * 100, time.time() - t0,
        )

        results[cname] = {
            "top1_rmsd": top1, "random_mean_rmsd": rand, "best_rmsd": best,
            "oracle_gap": gap, "kendall_tau": tau, "spearman_r": rho,
            "p_select_best": p_best, "gap_recovered_frac": gap_rec,
            "lb": lb, "ss": ss,
            "conf_scores": scores, "ref_rmsds": rmsds[:n_poses], "n_valid": n_valid,
        }

    # Write results
    results_path = out_dir / "ranking_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: None if x != x else x)
    log.info("Saved results → %s", results_path)

    # Aggregate summary
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

    log.info("\n=== Confidence Model Ranking Results ===")
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
