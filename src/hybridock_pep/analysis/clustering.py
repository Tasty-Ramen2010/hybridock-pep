from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hybridock_pep.models import DockConfig, ScoredPose

try:
    from Bio.PDB import PDBParser
    from Bio.Data.IUPACData import protein_letters_3to1 as _prot_3to1

    _STANDARD_AA: frozenset[str] = frozenset(k.upper() for k in _prot_3to1)
except ImportError:
    PDBParser = None  # type: ignore[assignment,misc]
    _STANDARD_AA = frozenset()


def _is_standard_aa(residue: object) -> bool:
    """Return True if residue is one of the 20 standard amino acids.

    Replaces deprecated Bio.PDB.Polypeptide.is_aa (deprecated ≥1.80).
    Uses Bio.Data.IUPACData.protein_letters_3to1 keys, which are stable
    across all Biopython versions that support PDB parsing.
    """
    try:
        return residue.get_resname().strip().upper() in _STANDARD_AA  # type: ignore[union-attr]
    except AttributeError:
        return False

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
except ImportError:
    AgglomerativeClustering = None  # type: ignore[assignment,misc]
    silhouette_score = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Result of clustering a set of scored peptide poses.

    Args:
        k_optimal: Number of clusters selected by silhouette argmax.
        silhouette_score: Silhouette score at k_optimal (-1 to 1; higher is better).
        per_cluster_stats: Per-cluster statistics dicts from statistics.py.
    """

    k_optimal: int
    silhouette_score: float
    per_cluster_stats: list[dict[str, Any]] = field(default_factory=list)


def _load_receptor_ca_coords(receptor_path: Path) -> np.ndarray:
    """Load C-alpha coordinates from a receptor PDB file.

    Uses Biopython to parse standard amino acid residues only.

    Args:
        receptor_path: Absolute path to the receptor PDB file.

    Returns:
        Float64 array of shape (n_residues, 3) with Cα XYZ coordinates.

    Raises:
        RuntimeError: If Biopython is not installed.
        ValueError: If no standard amino acid Cα atoms are found in the PDB.
    """
    if PDBParser is None:
        raise RuntimeError(
            "Biopython is required for receptor Cα extraction. "
            "Install it with: pip install biopython"
        )

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", str(receptor_path))
    model = next(iter(structure))

    coords: list[list[float]] = []
    for chain in model:
        for residue in chain:
            if not _is_standard_aa(residue):
                continue
            if "CA" not in residue:
                continue
            coords.append(list(residue["CA"].get_vector().get_array()))

    if not coords:
        raise ValueError(
            f"No standard amino acid Cα atoms found in receptor PDB: {receptor_path}"
        )

    return np.array(coords, dtype=np.float64)


def _contact_zone_indices(
    pose_ca: np.ndarray,
    receptor_ca: np.ndarray,
    cutoff: float = 6.0,
) -> np.ndarray:
    """Return peptide residue indices whose Cα is within cutoff Å of any receptor Cα.

    Uses numpy broadcasting to compute all pairwise distances in one pass.
    The D-02 fallback (< 3 contacts → full peptide) is applied at the
    RMSD matrix level using per-pair intersection logic.

    Args:
        pose_ca: Shape (n_peptide, 3) float64 array of peptide Cα coordinates.
        receptor_ca: Shape (n_receptor, 3) float64 array of receptor Cα coordinates.
        cutoff: Distance cutoff in Angstrom. Default 6.0.

    Returns:
        1D integer array of peptide residue indices in the contact zone.
        May be empty if no residues are within cutoff distance.
    """
    # Broadcasting: diff[i, j] = pose_ca[i] - receptor_ca[j]
    diff = pose_ca[:, np.newaxis, :] - receptor_ca[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))  # shape [n_pep, n_rec]
    return np.where(np.any(dists < cutoff, axis=1))[0]


def _build_rmsd_matrix(
    ca_arrays: list[np.ndarray],
    contact_indices: list[np.ndarray],
) -> np.ndarray:
    """Build a symmetric pairwise RMSD distance matrix over contact-zone Cα.

    For each pose pair (i, j), RMSD is computed over the intersection of
    their contact-zone indices. If the intersection has fewer than 3 residues,
    falls back to full-peptide indices (D-02).

    Args:
        ca_arrays: List of (n_residues, 3) float64 arrays, one per pose.
        contact_indices: List of 1D integer index arrays, one per pose,
            giving the contact-zone residue indices for that pose.

    Returns:
        Symmetric (n, n) float64 RMSD distance matrix with zero diagonal.
    """
    n = len(ca_arrays)
    dist = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            common = np.intersect1d(contact_indices[i], contact_indices[j])
            if len(common) < 3:
                # D-02 fallback: use full-peptide indices
                max_len = min(len(ca_arrays[i]), len(ca_arrays[j]))
                common = np.arange(max_len)

            coords_i = ca_arrays[i][common]
            coords_j = ca_arrays[j][common]
            rmsd = float(np.sqrt(np.mean(np.sum((coords_i - coords_j) ** 2, axis=1))))
            dist[i, j] = rmsd
            dist[j, i] = rmsd

    return dist


def _select_k_silhouette(
    dist_matrix: np.ndarray,
) -> tuple[int, dict[int, float]]:
    """Select optimal cluster count k by argmax silhouette score.

    Searches k = 2..k_max where k_max = min(15, n // 5). Falls back to
    k=2 without search if k_max < 2 (fewer than 10 poses).

    Args:
        dist_matrix: Symmetric (n, n) precomputed distance matrix.

    Returns:
        Tuple of (k_optimal, sil_scores_dict) where sil_scores_dict maps
        each k tried to its silhouette score. Empty dict if k_max < 2.

    Raises:
        RuntimeError: If scikit-learn is not installed.
    """
    if AgglomerativeClustering is None or silhouette_score is None:
        raise RuntimeError(
            "scikit-learn is required for clustering. "
            "Install it with: pip install scikit-learn"
        )

    n = dist_matrix.shape[0]
    k_max = min(15, n // 5)

    if k_max < 2:
        logger.debug("k_max=%d < 2 (n=%d); using k=2 without silhouette search", k_max, n)
        return 2, {}

    sil_scores: dict[int, float] = {}
    for k in range(2, k_max + 1):
        agg = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average"
        )
        labels = agg.fit_predict(dist_matrix)
        try:
            score = silhouette_score(dist_matrix, labels, metric="precomputed")
            sil_scores[k] = float(score)
            logger.debug("k=%d silhouette=%.4f", k, score)
        except ValueError:
            logger.debug("silhouette_score failed for k=%d, skipping", k)
            continue

    if not sil_scores:
        logger.warning("No valid silhouette scores computed; defaulting to k=2")
        return 2, {}

    k_optimal = max(sil_scores, key=lambda k: sil_scores[k])
    logger.debug("Selected k_optimal=%d (silhouette=%.4f)", k_optimal, sil_scores[k_optimal])
    return k_optimal, sil_scores


def cluster_poses(
    scored_poses: list[ScoredPose],
    config: DockConfig,
) -> ClusterResult:
    """Cluster scored peptide poses by contact-zone Cα RMSD with silhouette k selection.

    Orchestrates the full clustering pipeline:
    1. Load receptor Cα coordinates from config.receptor_path.
    2. Compute per-pose contact-zone indices against the receptor.
    3. Build pairwise RMSD distance matrix using contact-zone Cα (D-01/D-02).
    4. Select k via argmax silhouette score (D-05/D-06).
    5. Fit final AgglomerativeClustering and mutate ScoredPose.cluster_id in-place (D-08).
    6. Delegate CSV and plot output to statistics.py and plotting.py (D-10).

    Args:
        scored_poses: List of ScoredPose objects with populated ca_coords and
            hybrid_score. Mutated in-place to set cluster_id.
        config: DockConfig with receptor_path and output_dir fields.

    Returns:
        ClusterResult with k_optimal, silhouette_score, and per_cluster_stats.

    Raises:
        RuntimeError: If scikit-learn or Biopython are not available.
        ValueError: If scored_poses is empty or receptor has no Cα atoms.
    """
    if not scored_poses:
        raise ValueError("scored_poses must not be empty")

    logger.info("Stage 3: clustering %d poses", len(scored_poses))

    # 1. Load receptor Cα — prefer cleaned receptor (protein-only, no co-crystal peptide)
    cleaned_receptor = config.output_dir / "receptor_for_rapidock.pdb"
    receptor_pdb = cleaned_receptor.resolve() if cleaned_receptor.exists() else config.receptor_path.resolve()
    logger.debug("Contact zone computed vs %s", receptor_pdb)
    receptor_ca = _load_receptor_ca_coords(receptor_pdb)

    # 2. Per-pose contact-zone indices
    ca_arrays = [p.ca_coords for p in scored_poses]
    contact_indices = [
        _contact_zone_indices(p.ca_coords, receptor_ca) for p in scored_poses
    ]

    # 3. RMSD distance matrix
    dist_matrix = _build_rmsd_matrix(ca_arrays, contact_indices)
    logger.debug("Built %dx%d RMSD distance matrix", dist_matrix.shape[0], dist_matrix.shape[1])

    # 4. Silhouette k selection
    k_optimal, sil_scores = _select_k_silhouette(dist_matrix)

    # 5. Final clustering fit
    if AgglomerativeClustering is None:
        raise RuntimeError(
            "scikit-learn is required for clustering. "
            "Install it with: pip install scikit-learn"
        )

    agg = AgglomerativeClustering(
        n_clusters=k_optimal, metric="precomputed", linkage="average"
    )
    labels = agg.fit_predict(dist_matrix)

    # Mutate in-place (D-08, mirrors apply_hybrid_score pattern)
    for i, pose in enumerate(scored_poses):
        pose.cluster_id = int(labels[i])

    logger.info("Assigned %d poses to %d clusters", len(scored_poses), k_optimal)

    # 6. Delegate CSV and plots (lazy imports — defer sklearn dependency)
    from hybridock_pep.analysis.statistics import (  # noqa: PLC0415
        compute_cluster_stats,
        write_cluster_summary_csv,
    )
    from hybridock_pep.analysis.plotting import (  # noqa: PLC0415
        plot_convergence,
        plot_silhouette,
    )

    stats = compute_cluster_stats(scored_poses)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_cluster_summary_csv(stats, config.output_dir / "cluster_summary.csv")
    plot_convergence(scored_poses, config.output_dir / "convergence_plot.png")

    sil_val = sil_scores.get(k_optimal, float("nan"))
    plot_silhouette(
        sil_scores,
        k_optimal=k_optimal,
        output_path=config.output_dir / "silhouette_plot.png",
    )

    return ClusterResult(
        k_optimal=k_optimal,
        silhouette_score=sil_val,
        per_cluster_stats=stats,
    )
