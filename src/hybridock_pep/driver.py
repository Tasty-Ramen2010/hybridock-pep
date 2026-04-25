from __future__ import annotations

import logging
from pathlib import Path

from hybridock_pep.models import DockConfig, PoseRecord, ScoredPose
from hybridock_pep.sampling.rapidock_runner import run_sampling
from hybridock_pep.sampling.pose_io import parse_poses
from hybridock_pep.prep.receptor import prepare_receptor
from hybridock_pep.prep.grids import generate_ad4_maps
from hybridock_pep.prep.ligand import prepare_ligand_batch
from hybridock_pep.scoring.vina import score_vina_batch
from hybridock_pep.scoring.ad4 import score_ad4_batch
from hybridock_pep.scoring.entropy import apply_hybrid_score, load_calibration
from hybridock_pep.output.metadata import write_metadata_skeleton, finalize_metadata

logger = logging.getLogger(__name__)


def run_dock(
    config: DockConfig,
    input_poses_dir: Path | None,
    calibration_path: Path,
) -> list[ScoredPose]:
    """Orchestrate the full two-stage docking pipeline (per D-02).

    Stage 0: Write metadata skeleton before any subprocess is launched.
    Stage 1: Run RAPiDock sampling OR read from input_poses_dir (D-01 bypass).
    Stage 2a: Prepare receptor PDBQT and AD4 grid maps.
    Stage 2b: Prepare ligand PDBQTs in batch.
    Stage 2c: Construct ScoredPose objects from PoseRecord + pdbqt_path pairs.
    Stage 2d: Score with Vina, then AD4, then apply hybrid entropy correction.
    Stage 3 stub: Log handoff message; return scored poses (clustering is Phase 6).

    All paths passed to sub-module calls are resolved to absolute before use.

    Args:
        config: Validated DockConfig from cli._run_dock(). Never re-validated here.
        input_poses_dir: If not None, skip run_sampling() and parse poses from this
            directory instead (--input-poses bypass, D-01). Required on macOS.
        calibration_path: Path to calibration.json for entropy correction coefficients.

    Returns:
        List of ScoredPose objects with vina_score, ad4_score, entropy_correction,
        and hybrid_score populated. May be shorter than config.n_samples if some
        poses failed prep or scoring (failures are logged at WARNING).

    Raises:
        RuntimeError: If all poses fail ligand prep (zero pdbqt_paths produced).
    """
    # Stage 0: Write metadata skeleton BEFORE Stage 1
    metadata_path = config.output_dir.resolve() / "run_metadata.json"
    config.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    write_metadata_skeleton(config, metadata_path)

    # Stage 1: Sampling or bypass (D-01)
    if input_poses_dir is not None:
        poses_dir = input_poses_dir.resolve()
        logger.info("Stage 1 bypassed: reading poses from %s", poses_dir)
    else:
        logger.info(
            "Stage 1: running RAPiDock sampling (%d passes)", config.n_samples
        )
        run_sampling(config)
        poses_dir = (config.output_dir / "poses").resolve()

    records, parse_failures = parse_poses(poses_dir)
    if parse_failures:
        logger.warning(
            "%d poses failed parsing (out of %d files in %s)",
            len(parse_failures),
            len(records) + len(parse_failures),
            poses_dir,
        )
    logger.info("Stage 1 complete: %d poses parsed", len(records))

    # Stage 2a: Receptor prep + AD4 grid maps
    receptor_pdbqt = prepare_receptor(config)
    logger.info("Receptor prepared: %s", receptor_pdbqt)
    maps_dir = generate_ad4_maps(config, receptor_pdbqt)
    logger.info("AD4 maps generated: %s", maps_dir)

    # Stage 2b: Ligand batch prep
    pdb_paths = [record.pdb_path.resolve() for record in records]
    pdbqt_dir = (config.output_dir / "pdbqt").resolve()
    pdbqt_paths, prep_failures = prepare_ligand_batch(pdb_paths, pdbqt_dir)
    if prep_failures:
        logger.warning(
            "%d poses failed ligand prep", len(prep_failures)
        )
    if not pdbqt_paths and records:
        raise RuntimeError(
            "All poses failed ligand prep — cannot continue. "
            "Check meeko installation and pose PDB validity."
        )

    # Stage 2c: Construct ScoredPose objects pairing records with pdbqt paths.
    # pdbqt_by_stem matches pose_N.pdb → pose_N.pdbqt by stem.
    pdbqt_by_stem: dict[str, Path] = {p.stem: p for p in pdbqt_paths}
    scored_poses: list[ScoredPose] = []
    for record in records:
        stem = record.pdb_path.stem
        pdbqt_path = pdbqt_by_stem.get(stem)
        if pdbqt_path is None:
            continue
        scored_poses.append(
            ScoredPose(
                pose_idx=record.pose_idx,
                pdb_path=record.pdb_path,
                sequence=record.sequence,
                ca_coords=record.ca_coords,
                pdbqt_path=pdbqt_path,
            )
        )

    # Stage 2d: Score Vina → AD4 → entropy (order is mandatory)
    receptor_pdbqt_abs = receptor_pdbqt.resolve()
    scored_poses, vina_failures = score_vina_batch(
        scored_poses,
        config,
        receptor_pdbqt_abs,
        verbosity=config.verbosity,
        metadata_path=metadata_path,
    )
    if vina_failures:
        logger.warning("%d poses failed Vina scoring", len(vina_failures))

    maps_dir_abs = maps_dir.resolve()
    scored_poses, ad4_failures = score_ad4_batch(
        scored_poses,
        maps_dir_abs,
        verbosity=config.verbosity,
    )
    if ad4_failures:
        logger.warning("%d poses failed AD4 scoring", len(ad4_failures))

    calibration = load_calibration(calibration_path.resolve())
    alpha: float = calibration["alpha"]
    beta: float = calibration["beta"]
    n_residues = len(config.peptide_sequence)

    for pose in scored_poses:
        apply_hybrid_score(pose, alpha=alpha, beta=beta, n_residues=n_residues)

    logger.info("Stage 2 complete: %d poses scored", len(scored_poses))

    # Stage 3: Clustering and analysis
    if len(scored_poses) >= 2:
        from hybridock_pep.analysis import cluster_poses
        cluster_result = cluster_poses(scored_poses, config)
        logger.info(
            "Stage 3 complete: k=%d clusters, silhouette=%.3f",
            cluster_result.k_optimal,
            cluster_result.silhouette_score,
        )
    else:
        logger.warning("Stage 3 skipped: no scored poses to cluster")

    # Finalize metadata AFTER scoring
    finalize_metadata(metadata_path, poses_generated=len(records))

    return scored_poses
