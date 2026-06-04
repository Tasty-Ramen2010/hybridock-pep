#!/usr/bin/env python3
"""Pose diversity and quality metrics for RAPiDock output.

Measures:
  - Pairwise Cα RMSD distribution (spread of generated poses)
  - Agglomerative clustering at multiple RMSD thresholds
  - If a crystal reference is given: hit rates, top-1 RMSD, score-RMSD rank correlation

Usage (score-env):
    # Diversity only (no reference)
    conda run -n score-env python3 scripts/eval_pose_diversity.py \\
        --poses-dir runs/finetuned_1ycr/poses/ \\
        --out logs/diversity_finetuned.json

    # With crystal reference for hit-rate / RMSD metrics
    conda run -n score-env python3 scripts/eval_pose_diversity.py \\
        --poses-dir runs/finetuned_1ycr/poses/ \\
        --reference data/pdbs/1YCR_peptide.pdb \\
        --scores-csv runs/finetuned_1ycr/ranked_poses.csv \\
        --out logs/diversity_finetuned.json

Output JSON keys:
    n_poses          — number of PDBs analysed
    pairwise_rmsd    — {mean, median, p25, p75, max, std}
    clusters_1A      — distinct cluster count at 1 Å cutoff
    clusters_2A      — distinct cluster count at 2 Å cutoff
    clusters_5A      — distinct cluster count at 5 Å cutoff
    diversity_ratio  — clusters_2A / n_poses  (1.0 = every pose unique)
    --- if --reference given ---
    ref_rmsds        — per-pose Cα RMSD to crystal (list, order = pose file sort)
    hit_rate_1A      — fraction of poses within 1 Å of crystal
    hit_rate_2A      — fraction of poses within 2 Å of crystal
    hit_rate_5A      — fraction of poses within 5 Å of crystal
    best_rmsd        — minimum per-pose RMSD (best-of-N)
    top1_rmsd        — RMSD of best-scored pose (rank index 0 in scores CSV, or pose_0)
    spearman_r       — Spearman rank corr between score rank and RMSD rank (if --scores-csv)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from Bio import PDB
    from Bio.PDB.Superimposer import Superimposer
except ImportError:
    sys.exit("BioPython not found — run in score-env: conda activate score-env")

try:
    from sklearn.cluster import AgglomerativeClustering
except ImportError:
    sys.exit("scikit-learn not found — run in score-env: conda activate score-env")


# ---------------------------------------------------------------------------
# PDB loading
# ---------------------------------------------------------------------------

_parser = PDB.PDBParser(QUIET=True)


def _ca_atoms(structure) -> List:
    """Return ordered list of Cα atoms from first model, first chain."""
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == " " and "CA" in residue:
                    atoms.append(residue["CA"])
        break  # first model only
    return atoms


def load_ca_coords(pdb_path: Path) -> np.ndarray:
    """Load Cα coordinates from a PDB file. Shape: (n_residues, 3)."""
    struct = _parser.get_structure("x", str(pdb_path))
    atoms = _ca_atoms(struct)
    if not atoms:
        raise ValueError(f"No Cα atoms found in {pdb_path}")
    return np.array([a.get_vector().get_array() for a in atoms])


def superpose_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Superpose coords_b onto coords_a and return Cα RMSD.

    Uses Kabsch algorithm via BioPython's Superimposer for best-fit RMSD.
    Falls back to non-superposed RMSD if residue counts differ (warn only).
    """
    if coords_a.shape != coords_b.shape:
        # Different residue counts — return raw RMSD after truncation to shorter
        n = min(len(coords_a), len(coords_b))
        diff = coords_a[:n] - coords_b[:n]
        return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

    sup = Superimposer()
    # Superimposer expects Atom objects; build synthetic ones from coords
    # Use numpy Kabsch manually instead
    return _kabsch_rmsd(coords_a, coords_b)


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Kabsch algorithm: minimum RMSD after optimal rotation of Q onto P."""
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)
    H = Q.T @ P
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    Q_rot = Q @ R.T
    diff = P - Q_rot
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


# ---------------------------------------------------------------------------
# Pairwise RMSD matrix
# ---------------------------------------------------------------------------

def pairwise_rmsd_matrix(coords_list: List[np.ndarray]) -> np.ndarray:
    """Compute symmetric pairwise RMSD matrix. Shape: (N, N)."""
    N = len(coords_list)
    mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            r = superpose_rmsd(coords_list[i], coords_list[j])
            mat[i, j] = r
            mat[j, i] = r
    return mat


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_count(dist_matrix: np.ndarray, threshold: float) -> int:
    """Number of clusters via agglomerative clustering (average linkage)."""
    N = dist_matrix.shape[0]
    if N < 2:
        return N
    clust = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=threshold,
    )
    labels = clust.fit_predict(dist_matrix)
    return int(np.unique(labels).shape[0])


# ---------------------------------------------------------------------------
# Score-RMSD correlation
# ---------------------------------------------------------------------------

def spearman_r(scores: np.ndarray, rmsds: np.ndarray) -> float:
    """Spearman rank correlation between score rank and RMSD rank.

    Lower score rank (better score) should correlate with lower RMSD (better pose).
    Returns correlation in [-1, 1]; −1 = perfect (better score → better pose).
    """
    from scipy.stats import spearmanr  # lazy import; scipy in score-env
    corr, _ = spearmanr(scores, rmsds)
    return float(corr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Pose diversity and quality metrics for RAPiDock output.")
    p.add_argument("--poses-dir", required=True,
                   help="Directory containing pose_*.pdb files")
    p.add_argument("--reference", default=None,
                   help="Crystal reference PDB for hit-rate / RMSD metrics")
    p.add_argument("--scores-csv", default=None,
                   help="CSV with 'pose_file' and 'score' columns (for score-RMSD corr)")
    p.add_argument("--n-poses", type=int, default=None,
                   help="Limit analysis to first N poses (default: all)")
    p.add_argument("--out", default=None,
                   help="JSON output path (default: print only)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    poses_dir = Path(args.poses_dir)

    def _pose_idx(p: Path) -> int:
        # NUMERIC sort key. Plain sorted() is lexicographic, which scrambles
        # pose_10 before pose_2 → ref_rmsds[k] no longer matches pose_k for N≥10.
        # Every downstream consumer reads pose_i by integer index, so the order
        # of ref_rmsds MUST be numeric. (This bug silently decoupled labels from
        # structures in all N≥10 generation runs — see gen_n100 forensics.)
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        return int(digits) if digits else 0

    pose_files = sorted(poses_dir.glob("pose_*.pdb"), key=_pose_idx)
    if not pose_files:
        pose_files = sorted(poses_dir.glob("rank*.pdb"), key=_pose_idx)
    if not pose_files:
        sys.exit(f"No pose_*.pdb or rank*.pdb files found in {poses_dir}")

    if args.n_poses:
        pose_files = pose_files[: args.n_poses]

    N = len(pose_files)
    if not args.quiet:
        print(f"Loading {N} poses from {poses_dir} ...")

    # Load Cα coords (skip broken files with warning)
    coords_list: List[np.ndarray] = []
    good_files: List[Path] = []
    for pf in pose_files:
        try:
            coords_list.append(load_ca_coords(pf))
            good_files.append(pf)
        except Exception as e:
            print(f"  [WARN] skipping {pf.name}: {e}", file=sys.stderr)

    N = len(coords_list)
    if N == 0:
        sys.exit("All poses failed to load.")
    if not args.quiet:
        print(f"  {N} poses loaded successfully.")

    results: Dict = {"n_poses": N, "poses_dir": str(poses_dir)}

    # ── Pairwise RMSD ────────────────────────────────────────────────────────
    if not args.quiet:
        print("Computing pairwise Cα RMSD matrix ...")
    mat = pairwise_rmsd_matrix(coords_list)
    upper = mat[np.triu_indices(N, k=1)]

    results["pairwise_rmsd"] = {
        "mean":   float(np.mean(upper)),
        "median": float(np.median(upper)),
        "std":    float(np.std(upper)),
        "p25":    float(np.percentile(upper, 25)),
        "p75":    float(np.percentile(upper, 75)),
        "max":    float(np.max(upper)),
        "min":    float(np.min(upper)),
    }

    # ── Clustering ───────────────────────────────────────────────────────────
    if not args.quiet:
        print("Clustering at 1 Å / 2 Å / 5 Å ...")
    c1 = cluster_count(mat, 1.0)
    c2 = cluster_count(mat, 2.0)
    c5 = cluster_count(mat, 5.0)
    results["clusters_1A"] = c1
    results["clusters_2A"] = c2
    results["clusters_5A"] = c5
    results["diversity_ratio"] = round(c2 / N, 4)

    # ── Reference RMSD ───────────────────────────────────────────────────────
    if args.reference:
        ref_path = Path(args.reference)
        if not args.quiet:
            print(f"Computing RMSD to reference: {ref_path.name} ...")
        try:
            ref_coords = load_ca_coords(ref_path)
        except Exception as e:
            print(f"  [WARN] Could not load reference: {e}", file=sys.stderr)
            ref_coords = None

        if ref_coords is not None:
            ref_rmsds = [superpose_rmsd(ref_coords, c) for c in coords_list]
            results["ref_rmsds"] = [round(r, 4) for r in ref_rmsds]
            results["best_rmsd"]    = round(min(ref_rmsds), 4)
            results["hit_rate_1A"]  = round(sum(r <= 1.0 for r in ref_rmsds) / N, 4)
            results["hit_rate_2A"]  = round(sum(r <= 2.0 for r in ref_rmsds) / N, 4)
            results["hit_rate_5A"]  = round(sum(r <= 5.0 for r in ref_rmsds) / N, 4)
            results["rmsd_p25"]     = round(float(np.percentile(ref_rmsds, 25)), 4)
            results["rmsd_median"]  = round(float(np.median(ref_rmsds)), 4)
            results["rmsd_p75"]     = round(float(np.percentile(ref_rmsds, 75)), 4)

            # Top-1: pose_0.pdb (or rank1.pdb) — assumed to be best-scored if no CSV
            results["top1_rmsd"] = round(ref_rmsds[0], 4)

    # ── Score-RMSD correlation ───────────────────────────────────────────────
    if args.scores_csv and "ref_rmsds" in results:
        try:
            import pandas as pd
            df = pd.read_csv(args.scores_csv)
            # Match file basenames to our loaded order
            name_to_rmsd = {
                pf.name: r for pf, r in zip(good_files, results["ref_rmsds"])
            }
            if "pose_file" in df.columns and "score" in df.columns:
                df["rmsd"] = df["pose_file"].apply(
                    lambda x: name_to_rmsd.get(Path(x).name, float("nan"))
                )
                df = df.dropna(subset=["rmsd"])
                if len(df) >= 5:
                    results["spearman_r"] = spearman_r(
                        df["score"].values, df["rmsd"].values
                    )
        except Exception as e:
            print(f"  [WARN] Score-RMSD correlation failed: {e}", file=sys.stderr)

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"POSE DIVERSITY REPORT  ({N} poses)")
    print("=" * 60)
    pw = results["pairwise_rmsd"]
    print(f"  Pairwise Cα RMSD:  mean={pw['mean']:.2f}  median={pw['median']:.2f}"
          f"  max={pw['max']:.2f}  std={pw['std']:.2f} Å")
    print(f"  Clusters @ 1Å:{c1:4d}  @ 2Å:{c2:4d}  @ 5Å:{c5:4d}"
          f"  (diversity ratio: {results['diversity_ratio']:.2f})")
    if "best_rmsd" in results:
        print(f"  vs Crystal:  best={results['best_rmsd']:.2f}Å"
              f"  top1={results['top1_rmsd']:.2f}Å"
              f"  hit@1Å={results['hit_rate_1A']:.1%}"
              f"  hit@2Å={results['hit_rate_2A']:.1%}"
              f"  hit@5Å={results['hit_rate_5A']:.1%}")
    if "spearman_r" in results:
        print(f"  Score-RMSD Spearman ρ: {results['spearman_r']:+.3f}"
              f"  ({'anti-correlated ✓' if results['spearman_r'] < -0.1 else 'not correlated'})")
    print("=" * 60)

    # ── Write JSON ───────────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
