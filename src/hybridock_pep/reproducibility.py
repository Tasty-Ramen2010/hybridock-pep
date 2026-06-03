"""Multi-seed reproducibility metric for the docking pipeline.

Runs the dock pipeline on the same complex K times with different RAPiDock
seeds, then computes inter-run agreement on the top-scoring cluster centroid:

  * Cα RMSD between best poses across runs (pairwise, lower = better).
  * Pearson r of per-residue Cα coordinates between runs (higher = better).

This mirrors the cross-replica RMSF Pearson metric used in Wahibah-Hasibuan
et al. (Probiotics Antimicrob Proteins, 2026) but at a fraction of the
compute (no 1.2 µs MD). Reports whether the pipeline is deterministic-up-to-
RAPiDock-stochasticity on a given complex.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from hybridock_pep import driver
from hybridock_pep.models import DockConfig, ScoredPose

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReproducibilityResult:
    """Multi-seed reproducibility report.

    Attributes:
        seeds: Seeds used for each run.
        mean_pairwise_rmsd: Mean pairwise Cα RMSD between top-1 poses (Å).
        max_pairwise_rmsd: Worst pairwise RMSD (Å).
        mean_pairwise_pearson: Mean Pearson r between per-residue Cα coords.
        n_runs: Number of independent runs.
        top1_dg_per_run: ΔG_corrected of each run's top-1 pose.
        dg_std: σ of those ΔG values (kcal/mol).
        verdict: Coarse interpretation flag.
    """
    seeds: list[int]
    mean_pairwise_rmsd: float
    max_pairwise_rmsd: float
    mean_pairwise_pearson: float
    n_runs: int
    top1_dg_per_run: list[float]
    dg_std: float
    verdict: str

    def to_json(self) -> dict[str, object]:
        return {
            "seeds": self.seeds,
            "n_runs": self.n_runs,
            "mean_pairwise_rmsd_A": self.mean_pairwise_rmsd,
            "max_pairwise_rmsd_A": self.max_pairwise_rmsd,
            "mean_pairwise_pearson_r": self.mean_pairwise_pearson,
            "top1_dg_per_run_kcal_mol": self.top1_dg_per_run,
            "dg_std_kcal_mol": self.dg_std,
            "verdict": self.verdict,
        }


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Cα RMSD after optimal superposition (Kabsch)."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    sign = np.sign(np.linalg.det(u @ vt))
    s = np.eye(3)
    s[2, 2] = sign
    r = u @ s @ vt
    aligned = a @ r
    return float(np.sqrt(((aligned - b) ** 2).sum(axis=1).mean()))


def _per_residue_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two (N, 3) Cα coordinate sets flattened to 3N vectors."""
    return float(np.corrcoef(a.flatten(), b.flatten())[0, 1])


def _verdict(mean_rmsd: float, pearson_r: float) -> str:
    if mean_rmsd < 1.0 and pearson_r > 0.95:
        return "highly reproducible"
    if mean_rmsd < 2.5 and pearson_r > 0.85:
        return "reproducible"
    if mean_rmsd < 5.0:
        return "moderately reproducible"
    return "low reproducibility (RAPiDock pose diversity dominates)"


def run_reproducibility(
    base_config: DockConfig,
    calibration_path: Path,
    seeds: list[int],
    input_poses_dir: Path | None = None,
) -> ReproducibilityResult:
    """Run the dock pipeline K times with different seeds, report agreement.

    Args:
        base_config: DockConfig template; per-run output dirs are
            ``base_config.output_dir / f"seed_{N}"``.
        calibration_path: Calibration JSON used for all runs.
        seeds: List of integer seeds; len(seeds) defines K.
        input_poses_dir: Forwarded to each run (rarely useful here — multi-seed
            with the same pre-generated poses gives trivial 0 Å RMSD).

    Returns:
        ReproducibilityResult with pairwise RMSD/Pearson statistics + verdict.

    Raises:
        ValueError: If fewer than 2 seeds, or any run produced zero scored poses.
    """
    if len(seeds) < 2:
        raise ValueError("reproducibility needs ≥2 seeds")

    top1_coords: list[np.ndarray] = []
    top1_dg: list[float] = []
    for seed in seeds:
        cfg = base_config.model_copy(update={
            "seed": seed,
            "output_dir": base_config.output_dir / f"seed_{seed}",
        })
        poses, _ = driver.run_dock(
            config=cfg,
            input_poses_dir=input_poses_dir,
            calibration_path=calibration_path,
        )
        scored = [p for p in poses if p.hybrid_score is not None]
        if not scored:
            raise ValueError(f"seed {seed} produced no scored poses")
        scored.sort(key=lambda p: p.hybrid_score)  # type: ignore[arg-type, return-value]
        best = scored[0]
        top1_coords.append(np.asarray(best.ca_coords))
        top1_dg.append(float(best.hybrid_score))  # type: ignore[arg-type]

    # Pairwise stats — only meaningful when lengths match
    lens = {c.shape[0] for c in top1_coords}
    if len(lens) != 1:
        raise ValueError(
            f"top-1 poses have inconsistent Cα counts across seeds: {lens}"
        )
    rmsds: list[float] = []
    pears: list[float] = []
    for a, b in combinations(top1_coords, 2):
        rmsds.append(_kabsch_rmsd(a, b))
        pears.append(_per_residue_pearson(a, b))

    mean_r = float(np.mean(rmsds))
    max_r = float(np.max(rmsds))
    mean_p = float(np.mean(pears))
    return ReproducibilityResult(
        seeds=list(seeds),
        mean_pairwise_rmsd=mean_r,
        max_pairwise_rmsd=max_r,
        mean_pairwise_pearson=mean_p,
        n_runs=len(seeds),
        top1_dg_per_run=top1_dg,
        dg_std=float(np.std(top1_dg, ddof=1)) if len(top1_dg) > 1 else 0.0,
        verdict=_verdict(mean_r, mean_p),
    )
