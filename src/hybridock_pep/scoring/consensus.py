"""Consensus pose ranker — how much the diffusion ensemble agrees on a pose.

RAPiDock samples N times from a stochastic diffusion process, and those samples are returned
in arbitrary order: every accuracy number we quote from a raw run is an ORACLE best-of-N,
which is not what a user gets. This ranker turns the ensemble into a real top-1 pick.

The signal is the ensemble's own agreement: a diffusion model puts most of its probability
mass near the mode it actually believes in, so the pose with the SMALLEST mean heavy-atom
RMSD to every other pose sits in the densest region of the sampled distribution. It is the
structural analogue of cluster-size ranking, computed without clustering.

    consensus_score(i) = mean_j RMSD(pose_i, pose_j)      (lower = better, ascending convention)

RMSD is taken WITHOUT superposition: all poses of one run already share the receptor frame,
so the raw coordinate difference is the pose difference (same convention as our direct RMSD).

Measured on RAPiDock's own leak-free RecentSet (345 complexes, best-of-24, scripts/
rank_consensus.py), against DockQ(capri_peptide):

    selection            median DockQ   acceptable   medium   high
    oracle best-of-24        0.774         99.1%     91.6%   38.6%
    consensus top-1          0.680         96.2%     80.9%   14.8%
    no ranker (mean pose)    0.582         97.4%     70.7%    1.2%

So consensus top-1 keeps ~88% of the oracle median, and lifts high-accuracy poses from
1.2% to 14.8% versus taking an arbitrary pose. It has NO fitted parameters and never sees
the crystal, so there is nothing to overfit and no train/test split to get wrong.

Like the other pose rankers this is STRUCTURAL only: it orders poses and never touches the
affinity / ΔG number.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from hybridock_pep.models import ScoredPose

logger = logging.getLogger(__name__)

MIN_POSES: int = 3  # below this the "ensemble" carries no agreement signal


def _heavy_coords(pdb: Path) -> np.ndarray:
    """Heavy-atom XYZ in file order (poses of one run share atom order)."""
    xs: list[tuple[float, float, float]] = []
    try:
        for line in pdb.read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")) and line[76:78].strip() != "H":
                xs.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    except (OSError, ValueError) as exc:
        logger.debug("Consensus: could not read %s (%s)", pdb, exc)
        return np.empty((0, 3))
    return np.asarray(xs, dtype=np.float64)


def compute_consensus_scores(poses: list[ScoredPose]) -> None:
    """Set ``consensus_score`` on each pose in place (lower = more consensual).

    All poses must come from one run against one receptor. Poses whose atom count differs
    from the majority (a rare malformed write) are left at ``None`` and excluded from every
    other pose's mean, so one bad file cannot distort the ranking.

    Args:
        poses: Scored poses from a single dock run; mutated in place.

    Returns:
        None. ``consensus_score`` is set to a float, or left None where unavailable.
    """
    if len(poses) < MIN_POSES:
        logger.debug("Consensus: %d pose(s) is too few; skipping", len(poses))
        return

    coords = [_heavy_coords(Path(p.pose_path)) for p in poses]
    shapes = [c.shape for c in coords if c.size]
    if not shapes:
        logger.warning("Consensus: no pose had readable heavy atoms; skipping")
        return

    # Majority atom count defines the comparable set.
    ref_shape = max(set(shapes), key=shapes.count)
    usable = [i for i, c in enumerate(coords) if c.shape == ref_shape]
    if len(usable) < MIN_POSES:
        logger.warning("Consensus: only %d pose(s) share an atom count; skipping", len(usable))
        return
    if len(usable) < len(poses):
        logger.info("Consensus: %d of %d poses excluded (atom-count mismatch)",
                    len(poses) - len(usable), len(poses))

    x = np.stack([coords[i] for i in usable])                 # (n, atoms, 3)
    diff = x[:, None, :, :] - x[None, :, :, :]
    rms = np.sqrt((diff ** 2).sum(-1).mean(-1))               # (n, n), self-distance 0
    means = rms.sum(1) / (len(usable) - 1)                    # exclude the self term

    for slot, i in enumerate(usable):
        poses[i].consensus_score = float(means[slot])
