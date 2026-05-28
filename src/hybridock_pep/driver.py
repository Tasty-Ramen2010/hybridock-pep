from __future__ import annotations

import logging
import shutil
from pathlib import Path

from hybridock_pep.models import DockConfig, PoseRecord, ScoredPose
from hybridock_pep.analysis.clustering import ClusterResult
from hybridock_pep.sampling.rapidock_runner import run_sampling
from hybridock_pep.sampling.pose_io import parse_poses
from hybridock_pep.prep.receptor import prepare_receptor, prepare_receptor_pdb
from hybridock_pep.prep.grids import generate_ad4_maps
from hybridock_pep.prep.ligand import prepare_ligand_batch
from hybridock_pep.scoring.vina import score_vina_batch
from hybridock_pep.scoring.ad4 import score_ad4_batch
from hybridock_pep.scoring.entropy import (
    apply_hybrid_score,
    apply_ensemble_hybrid_scores,
    load_calibration,
    load_receptor_heavy_atom_coords,
    count_contact_residues,
    check_intermolecular_clash,
)
from hybridock_pep.output.metadata import write_metadata_skeleton, finalize_metadata

logger = logging.getLogger(__name__)


def run_dock(
    config: DockConfig,
    input_poses_dir: Path | None,
    calibration_path: Path,
) -> tuple[list[ScoredPose], ClusterResult | None]:
    """Orchestrate the full two-stage docking pipeline.

    Stage 0: Write metadata skeleton before any subprocess is launched.
    Stage 1: Run RAPiDock sampling OR read from input_poses_dir (bypass).
    Stage 2a: Prepare receptor PDBQT and AD4 grid maps.
    Stage 2b: Prepare ligand PDBQTs in batch.
    Stage 2c: Construct ScoredPose objects from PoseRecord + pdbqt_path pairs.
    Stage 2d: Score with Vina, then AD4, then apply hybrid entropy correction.
    Stage 3: Clustering and analysis.
    Stage 4: Write ranked_poses.csv and best_pose.pdb to config.output_dir.

    All paths passed to sub-module calls are resolved to absolute before use.

    Args:
        config: Validated DockConfig from cli._run_dock(). Never re-validated here.
        input_poses_dir: If not None, skip run_sampling() and parse poses from this
            directory instead (--input-poses bypass). Required on macOS.
        calibration_path: Path to calibration.json for entropy correction coefficients.

    Returns:
        Tuple of (scored_poses, cluster_result). scored_poses is a list of
        ScoredPose objects with all score fields populated. cluster_result is
        a ClusterResult if at least 2 poses were scored, otherwise None.

    Raises:
        RuntimeError: If all poses fail ligand prep (zero pdbqt_paths produced).
    """
    # Stage 0: Write metadata skeleton BEFORE Stage 1
    metadata_path = config.output_dir.resolve() / "run_metadata.json"
    config.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    write_metadata_skeleton(config, metadata_path)

    # Stage 1: Sampling or bypass
    if input_poses_dir is not None:
        poses_dir = input_poses_dir.resolve()
        logger.info("Stage 1 bypassed: reading poses from %s", poses_dir)
    else:
        logger.info(
            "Stage 1: running RAPiDock sampling (%d passes)", config.n_samples
        )
        # Clean the receptor with pdbfixer before Stage 1 so MDAnalysis sees
        # the same chain count as BioPython (raw RCSB PDBs can have discontinuous
        # chain segments that MDAnalysis splits into extra segments, causing an
        # IndexError in RAPiDock's ESM embedding lookup).
        cleaned_receptor = prepare_receptor_pdb(config)
        run_sampling(config, receptor_path=cleaned_receptor)
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

    # Stage 1.5: OpenMM energy minimization (RAPiDock poses only, not --input-poses)
    # Relieves intra-pose clashes that cause AD4 to return anomalous positive scores.
    # Skipped when input_poses_dir is set — user-supplied poses are assumed clean.
    if input_poses_dir is None and config.minimize_poses and records:
        from hybridock_pep.scoring.minimization import minimize_poses_batch  # noqa: PLC0415

        min_dir = (config.output_dir / "poses_minimized").resolve()
        raw_paths = [r.pdb_path.resolve() for r in records]
        minimized_paths = minimize_poses_batch(raw_paths, min_dir)
        for record, min_path in zip(records, minimized_paths):
            record.pdb_path = min_path.resolve()
        logger.info("Stage 1.5 complete: %d poses minimized → %s", len(records), min_dir)

    # Stage 1.6: Write poses_scored/ — the exact pose files that will be scored.
    # Each file is either the minimized pose (Stage 1.5 succeeded) or the original
    # RAPiDock pose (minimization failed / displacement check rejected it).
    # benchmark.py uses this directory for the vina-only rescore so both scores
    # come from identical input, making the hybrid-vs-Vina comparison fair.
    if input_poses_dir is None and records:
        scored_dir = (config.output_dir / "poses_scored").resolve()
        scored_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            dest = scored_dir / record.pdb_path.name
            if not dest.exists():
                shutil.copy2(record.pdb_path, dest)
        logger.debug("poses_scored/ written: %d files → %s", len(records), scored_dir)

    # Stage 2a: Receptor prep (always required for Vina scoring)
    receptor_pdbqt = prepare_receptor(config)
    logger.info("Receptor prepared: %s", receptor_pdbqt)

    # AD4 grid maps only when 'ad4' is in the requested scoring backends.
    # Skipping autogrid4 enables --scoring vina on macOS ARM where autogrid4 is absent.
    run_ad4 = "ad4" in config.scoring
    maps_dir: Path | None = None
    if run_ad4:
        maps_dir = generate_ad4_maps(config, receptor_pdbqt)
        logger.info("AD4 maps generated: %s", maps_dir)
    else:
        logger.info("AD4 scoring skipped (not in --scoring backends)")

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
            "Check babel (ADFRsuite) installation and pose PDB validity."
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

    # Stage 2d: Score Vina → (optional AD4) → entropy (order is mandatory)
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

    if run_ad4 and maps_dir is not None:
        maps_dir_abs = maps_dir.resolve()
        scored_poses, ad4_failures = score_ad4_batch(
            scored_poses,
            maps_dir_abs,
            verbosity=config.verbosity,
        )
        if ad4_failures:
            logger.warning("%d poses failed AD4 scoring", len(ad4_failures))

    # Stage 2d-pre: Contact counting and clash detection per pose.
    # Load receptor heavy atoms once; reuse across all poses for efficiency.
    receptor_coords = load_receptor_heavy_atom_coords(config.receptor_path.resolve())
    for pose in scored_poses:
        pose.n_contact_residues = count_contact_residues(
            pose.pdb_path, receptor_coords
        )
        pose.is_clashed = check_intermolecular_clash(
            pose.pdb_path, receptor_coords
        )
        if pose.is_clashed:
            logger.debug(
                "Pose %d: inter-molecular clash detected (heavy atom < 1.5 Å from receptor)",
                pose.pose_idx,
            )
    logger.info(
        "Stage 2d-pre: %d/%d poses have ≥1 contact residue; %d clashed",
        sum(1 for p in scored_poses if p.n_contact_residues),
        len(scored_poses),
        sum(1 for p in scored_poses if p.is_clashed),
    )

    calibration = load_calibration(calibration_path.resolve())
    alpha: float = calibration["alpha"]
    beta: float = calibration["beta"]
    gamma: float = calibration.get("gamma", 0.0)
    ensemble_ad4_weight: float = calibration.get("ensemble_ad4_weight", 0.0)
    n_residues = len(config.peptide_sequence)

    # Use ensemble z-score AD4 blending when beta=0 (calibration degenerate on
    # crystal poses) but AD4 scores are available and ensemble_ad4_weight > 0.
    # This re-integrates AD4's electrostatic signal via within-run normalization
    # instead of absolute-scale blending. Falls back to per-pose when beta > 0
    # (properly calibrated) or ensemble_ad4_weight = 0 (disabled).
    use_ensemble = run_ad4 and ensemble_ad4_weight > 0.0 and beta == 0.0
    if use_ensemble:
        apply_ensemble_hybrid_scores(
            scored_poses,
            alpha=alpha,
            n_residues=n_residues,
            ad4_blend_weight=ensemble_ad4_weight,
            gamma=gamma,
        )
        logger.info(
            "Hybrid scoring: ensemble z-score mode (AD4 weight=%.2f, alpha=%.3f)",
            ensemble_ad4_weight, alpha,
        )
    else:
        for pose in scored_poses:
            apply_hybrid_score(
                pose,
                alpha=alpha,
                beta=beta,
                n_residues=n_residues,
                n_contact_residues=pose.n_contact_residues,
                gamma=gamma,
            )
        logger.info(
            "Hybrid scoring: per-pose mode (beta=%.3f, alpha=%.3f)", beta, alpha,
        )

    logger.info("Stage 2 complete: %d poses scored", len(scored_poses))

    cluster_result: ClusterResult | None = None

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

    # Stage 3.5: MM-GBSA refinement (optional, requires --refine-topk)
    if config.refine_topk is not None and cluster_result is not None:
        from hybridock_pep.scoring.mmgbsa import refine_topk_poses  # noqa: PLC0415
        refine_topk_poses(scored_poses, cluster_result, config)
        n_refined = sum(1 for p in scored_poses if p.mmgbsa_dg is not None)
        logger.info("Stage 3.5 complete: %d poses have MM-GBSA ΔG", n_refined)

    # Finalize metadata AFTER scoring
    finalize_metadata(metadata_path, poses_generated=len(records))

    # Stage 4: Output writing
    from hybridock_pep.output.csv_writer import write_ranked_csv, write_best_pose_pdb
    write_ranked_csv(scored_poses, config)
    if cluster_result is not None:
        write_best_pose_pdb(cluster_result, config, scored_poses)

    return scored_poses, cluster_result
