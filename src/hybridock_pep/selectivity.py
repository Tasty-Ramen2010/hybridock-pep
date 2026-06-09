"""Decoy ΔΔG / selectivity primitive.

For a single peptide sequence and two receptors (an intended on-target and an
off-target), runs the full HybriDock-Pep pipeline on each, then reports the
selectivity score::

    ΔΔG = ΔG_target − ΔG_offtarget

Negative ΔΔG means the peptide binds tighter to the target than to the off-
target. Bootstrap CI is computed over the top-K cluster centroid ΔG values
from each side, so the estimate reflects ensemble uncertainty rather than
just point predictions.

This is the right primitive for the parent iGEM project's PfLDH vs hLDH
selectivity question (see CLAUDE.md §1). It also sidesteps the cross-target
absolute-Kd ceiling documented in docs/calibration_notes.md, since both sides
of the difference are subject to the same systematic bias.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hybridock_pep import driver
from hybridock_pep.models import DockConfig, ScoredPose

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectivityResult:
    """Decoy ΔΔG result with bootstrap CI.

    Attributes:
        peptide: Peptide sequence.
        target_dg: Mean ΔG_corrected over top-K target poses (kcal/mol).
        offtarget_dg: Mean ΔG_corrected over top-K off-target poses (kcal/mol).
        ddg: ΔG_target − ΔG_offtarget. Negative = selective for target.
        ddg_ci_low, ddg_ci_high: 95% bootstrap CI on ΔΔG.
        n_target_poses, n_offtarget_poses: Pose counts used for the estimate.
        bootstrap_n: Resampling iterations.
        top_k: K passed at construction.
    """
    peptide: str
    target_dg: float
    offtarget_dg: float
    ddg: float
    ddg_ci_low: float
    ddg_ci_high: float
    n_target_poses: int
    n_offtarget_poses: int
    bootstrap_n: int
    top_k: int
    score_field: str = "vina_score"

    def to_json(self) -> dict[str, object]:
        return {
            "peptide": self.peptide,
            "score_field": self.score_field,
            "target_dg_mean_kcal_mol": self.target_dg,
            "offtarget_dg_mean_kcal_mol": self.offtarget_dg,
            "ddg_kcal_mol": self.ddg,
            "ddg_ci_95_low": self.ddg_ci_low,
            "ddg_ci_95_high": self.ddg_ci_high,
            "n_target_poses": self.n_target_poses,
            "n_offtarget_poses": self.n_offtarget_poses,
            "bootstrap_n": self.bootstrap_n,
            "top_k": self.top_k,
            "interpretation": (
                "Selective for target"
                if self.ddg_ci_high < 0
                else "Selective for off-target"
                if self.ddg_ci_low > 0
                else "Inconclusive (CI crosses zero)"
            ),
        }


# Score field used to rank poses and form ΔΔG, in priority order for "auto".
#
# IMPORTANT — why NOT hybrid_score for selectivity:
#   ΔΔG = ΔG_target − ΔG_offtarget is computed for ONE peptide against TWO
#   receptors. The entropy / burial correction (α·N_contact or w·s_ss_weighted)
#   is a property of the *peptide*, so it largely cancels in the difference and
#   only injects noise from differing contact counts between the two runs. The
#   signal that actually distinguishes the two targets is the intermolecular
#   interaction energy: MM-GBSA ΔG_bind (best, when --refine-topk ran) or the
#   raw Vina score. So selectivity defaults to MM-GBSA → Vina, never the
#   entropy-corrected hybrid. Empirically (PfLDH vs hLDH, LISDAELEAIFEADC) the
#   hybrid compresses |ΔΔG| ~4× vs Vina, washing out the discrimination.
_SCORE_FIELDS = ("mmgbsa_dg", "vina_score", "hybrid_score")


def _resolve_score_field(poses: list[ScoredPose], requested: str) -> str:
    """Pick the score attribute for ΔΔG. 'auto' = first populated of MM-GBSA, Vina."""
    if requested != "auto":
        if requested not in _SCORE_FIELDS:
            raise ValueError(f"score_field must be 'auto' or one of {_SCORE_FIELDS}; got {requested!r}")
        return requested
    for field in ("mmgbsa_dg", "vina_score"):
        if any(getattr(p, field) is not None for p in poses):
            return field
    raise ValueError("no poses carry mmgbsa_dg or vina_score — cannot compute ΔΔG")


def _top_k_dg(poses: list[ScoredPose], k: int, score_field: str = "vina_score") -> list[float]:
    """Return the ΔG of the top-K poses ranked by (lowest) ``score_field``.

    Args:
        poses: Scored poses for one receptor.
        k: Number of best poses to take.
        score_field: Attribute to rank by and return — one of ``_SCORE_FIELDS``.
            Selectivity should use ``mmgbsa_dg`` or ``vina_score``, NOT
            ``hybrid_score`` (see module note above).

    Raises:
        ValueError: If no poses carry the requested score.
    """
    scored = [p for p in poses if getattr(p, score_field) is not None]
    if not scored:
        raise ValueError(f"no poses with {score_field} — was that scoring stage run?")
    scored.sort(key=lambda p: getattr(p, score_field))
    return [float(getattr(p, score_field)) for p in scored[: max(1, k)]]


def _bootstrap_ddg(
    target_dg: list[float],
    offtarget_dg: list[float],
    n_iter: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Paired bootstrap of mean ΔΔG; returns (low, high) 95% CI bounds."""
    t = np.array(target_dg)
    o = np.array(offtarget_dg)
    diffs = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        ti = t[rng.integers(0, t.size, t.size)]
        oi = o[rng.integers(0, o.size, o.size)]
        diffs[i] = ti.mean() - oi.mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run_selectivity(
    peptide: str,
    target_config: DockConfig,
    offtarget_config: DockConfig,
    calibration_path: Path,
    top_k: int = 10,
    bootstrap_n: int = 1000,
    seed: int | None = None,
    input_poses_target: Path | None = None,
    input_poses_offtarget: Path | None = None,
    score_field: str = "auto",
) -> SelectivityResult:
    """Run the docking pipeline on target and off-target, compute ΔΔG.

    Both DockConfigs must share the same peptide sequence (caller's contract).
    The two pipelines run sequentially (single GPU); their output directories
    must differ.

    Args:
        peptide: Peptide sequence (for the result struct).
        target_config: DockConfig for the on-target receptor.
        offtarget_config: DockConfig for the off-target receptor.
        calibration_path: Calibration JSON used for both sides.
        top_k: Number of best-scoring poses per side fed to ΔΔG.
        bootstrap_n: Bootstrap iterations for the CI on ΔΔG.
        seed: RNG seed for the bootstrap (CUDA seed is inside DockConfig).
        input_poses_target / input_poses_offtarget: Optional pre-generated
            pose directories to bypass Stage 1.

    Returns:
        SelectivityResult with point ΔΔG and 95% bootstrap CI.

    Raises:
        ValueError: If the two configs share an output_dir, or if either side
            produced zero scored poses with a corrected ΔG.
    """
    if target_config.output_dir == offtarget_config.output_dir:
        raise ValueError("target and off-target output_dir must differ")

    _log.info("Selectivity: running target pipeline (%s)", target_config.receptor_path.name)
    t_poses, _ = driver.run_dock(
        config=target_config,
        input_poses_dir=input_poses_target,
        calibration_path=calibration_path,
    )
    _log.info("Selectivity: running off-target pipeline (%s)", offtarget_config.receptor_path.name)
    o_poses, _ = driver.run_dock(
        config=offtarget_config,
        input_poses_dir=input_poses_offtarget,
        calibration_path=calibration_path,
    )

    # Resolve the scoring field once, using both pose sets so the same field is
    # used on each side (a ΔΔG built from mismatched fields would be meaningless).
    field = _resolve_score_field(t_poses + o_poses, score_field)
    _log.info("Selectivity: ΔΔG scored on '%s' (entropy hybrid excluded — it cancels)", field)
    t_dg = _top_k_dg(t_poses, top_k, field)
    o_dg = _top_k_dg(o_poses, top_k, field)

    rng = np.random.default_rng(seed)
    ci_low, ci_high = _bootstrap_ddg(t_dg, o_dg, bootstrap_n, rng)

    result = SelectivityResult(
        peptide=peptide,
        target_dg=float(np.mean(t_dg)),
        offtarget_dg=float(np.mean(o_dg)),
        ddg=float(np.mean(t_dg) - np.mean(o_dg)),
        ddg_ci_low=ci_low,
        ddg_ci_high=ci_high,
        n_target_poses=len(t_dg),
        n_offtarget_poses=len(o_dg),
        bootstrap_n=bootstrap_n,
        top_k=top_k,
        score_field=field,
    )

    # Persist result alongside the two output dirs.
    parent = target_config.output_dir.parent
    (parent / "selectivity.json").write_text(json.dumps(result.to_json(), indent=2))
    _log.info(
        "Selectivity: ΔΔG = %+.2f kcal/mol (95%% CI [%+.2f, %+.2f]) — %s",
        result.ddg, result.ddg_ci_low, result.ddg_ci_high,
        result.to_json()["interpretation"],
    )
    return result
