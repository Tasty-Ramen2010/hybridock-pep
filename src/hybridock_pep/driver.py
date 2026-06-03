from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np

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
    apply_calibration,
    apply_ensemble_hybrid_scores,
    calibration_mode,
    load_calibration,
    load_receptor_heavy_atom_coords,
    count_contact_residues,
    check_intermolecular_clash,
)
from hybridock_pep.output.metadata import write_metadata_skeleton, finalize_metadata

logger = logging.getLogger(__name__)


def _auto_expand_box_for_poses(
    config: DockConfig,
    records: list[PoseRecord],
    safety_margin: float = 4.0,
    log: logging.Logger | None = None,
) -> DockConfig:
    """Return a config with box_size expanded to contain all pose heavy atoms.

    The Vina grid is built around ``config.site_coords`` with edge length
    ``config.box_size``. Any pose atom outside the grid is silently clipped
    during scoring (Vina returns +∞ for that pose). When RAPiDock samples
    extended/groove binding sites it can produce poses with atoms 50+ Å from
    the user-supplied site center; the user's box becomes the bottleneck.

    This function measures the actual max-per-axis distance from
    ``site_coords`` over all pose atoms and, if it exceeds ``box_size / 2``,
    returns a copy of ``config`` with ``box_size`` expanded to fit (plus a
    safety margin). The original ``site_coords`` is preserved — only the
    box edge grows.

    Args:
        config: Frozen DockConfig with user-supplied box_size.
        records: PoseRecord list from Stage 1. May be empty.
        safety_margin: Extra Å added to the computed minimum box (default 4).
        log: Optional logger; defaults to module logger.

    Returns:
        Either the original config (no expansion needed) or a copy with
        ``box_size`` increased.
    """
    log = log or logger
    if not records:
        return config

    site = np.array(config.site_coords)
    half = config.box_size / 2.0

    # Walk pose PDBs once, tracking the maximum per-axis offset from site.
    # ca_coords is N×3 but doesn't cover side-chain atoms — we want heavy atoms
    # that will actually enter the Vina grid. Parse from each pose file.
    max_offset = 0.0
    for record in records:
        try:
            for line in record.pdb_path.read_text().splitlines():
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                atom = line[12:16].strip()
                if atom.startswith("H") or atom == "H":
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                offset = max(abs(x - site[0]), abs(y - site[1]), abs(z - site[2]))
                if offset > max_offset:
                    max_offset = offset
        except OSError:
            continue

    if max_offset <= half:
        log.debug(
            "Auto-box: pose extent %.1f Å fits user box (half-edge %.1f Å); no change",
            max_offset, half,
        )
        return config

    # Need a bigger box; round up to a tenth-of-Å for clean logs.
    new_edge = round((max_offset + safety_margin) * 2 + 0.5, 1)
    log.warning(
        "Auto-box: poses extend %.1f Å from site; user box %.1f Å too small. "
        "Expanding to %.1f Å (+safety %.1f Å). Set --box ≥ %.0f next time to silence.",
        max_offset, config.box_size, new_edge, safety_margin, new_edge,
    )
    return config.model_copy(update={"box_size": float(new_edge)})


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

    # Stage 1.7: Auto-expand box_size if pose spread exceeds user box.
    # The user's --box flag becomes a MINIMUM; we never silently clip poses.
    # Triggered when any pose has a heavy atom > config.box_size / 2 from
    # site_coords on any axis (which would clip during Vina grid scoring).
    config = _auto_expand_box_for_poses(config, records)

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

    # Load calibration once; both the entropy-sum gate below and the
    # hybrid-score stage further down consume the same dict.
    calibration = load_calibration(calibration_path.resolve())

    # Stage 2d-pre-entropy: per-residue + SS-weighted entropy sums per pose.
    # Only computed when the calibration references one of the w_s_* weights
    # (else the work is wasted — legacy and original-v2 calibrations don't
    # need these fields).  Adds ~1 ms per pose for the phi/psi pass.
    _needs_entropy_sums = (
        calibration_mode(calibration) == "ridge"
        and any(
            float(calibration.get(k, 0.0)) != 0.0
            for k in ("w_s_sc", "w_s_bb", "w_s_ss_weighted")
        )
    )
    if _needs_entropy_sums:
        from hybridock_pep.scoring.per_residue_entropy import (  # noqa: PLC0415
            compute_entropy_sums,
        )
        for pose in scored_poses:
            try:
                ent = compute_entropy_sums(
                    pose.pdb_path, config.peptide_sequence,
                    receptor_coords=receptor_coords,
                )
                pose.s_sc_sum = float(ent["s_sc_sum"])
                pose.s_bb_sum = float(ent["s_bb_sum"])
                pose.s_ss_weighted = float(ent["s_ss_weighted"])
                pose.ss_loop_count = int(ent["ss_loop_count"])
                pose.ss_helix_count = int(ent["ss_helix_count"])
                pose.ss_sheet_count = int(ent["ss_sheet_count"])
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Pose %d: entropy sums failed (%s); falling back to 0.0",
                    pose.pose_idx, exc,
                )
                pose.s_sc_sum = 0.0
                pose.s_bb_sum = 0.0
                pose.s_ss_weighted = 0.0
        logger.info(
            "Stage 2d-pre-entropy: per-residue entropy sums computed for %d poses",
            len(scored_poses),
        )

    # `calibration` already loaded earlier (Stage 2d-pre-entropy gate).
    mode = calibration_mode(calibration)
    n_residues = len(config.peptide_sequence)

    if mode == "ridge":
        # Schema v2: multivariate ridge.  Direct per-feature weights — no
        # ensemble z-score blending (the ridge already captures AD4 signal
        # in its w_ad4 weight, and the production-pose calibration gives
        # w_ad4=0 because AD4 carries no marginal signal once N_contact
        # is in the model.  See docs/calibration_notes.md "v2" section.)
        for pose in scored_poses:
            apply_calibration(
                pose,
                calibration,
                n_residues=n_residues,
                n_contact_residues=pose.n_contact_residues,
            )
        logger.info(
            "Hybrid scoring: ridge mode (w_vina=%.3f, w_ad4=%.3f, "
            "w_contact=%.3f, intercept=%.3f)",
            calibration["w_vina"], calibration["w_ad4"],
            calibration["w_contact"], calibration["intercept"],
        )
    else:
        alpha: float = calibration["alpha"]
        beta: float = calibration["beta"]
        gamma: float = calibration.get("gamma", 0.0)
        ensemble_ad4_weight: float = calibration.get("ensemble_ad4_weight", 0.0)

        # Legacy ensemble z-score blending only applies to the single-α
        # schema (where β=0 ⇒ AD4 is unused on absolute scale; this re-
        # introduces it via within-run normalization).
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
