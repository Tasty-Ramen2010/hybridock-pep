"""MM-GBSA free energy refinement for top-K docked poses.

Uses OpenMM AMBER ff14SB + GBn2 implicit solvent. Runs on CUDA GPU by default
with automatic fallback to CPU. Single-trajectory approximation: minimize the
complex once, extract receptor and peptide component energies from the same
minimized geometry (no re-minimization of components).

ΔG_bind = E(complex) − E(receptor_alone) − E(peptide_alone)   [kJ/mol → kcal/mol]
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hybridock_pep.models import DockConfig, ScoredPose
    from hybridock_pep.analysis.clustering import ClusterResult

logger = logging.getLogger(__name__)

_KJ_TO_KCAL: float = 0.239006
_MINIMIZE_TOL: float = 10.0          # kJ/mol/nm — loose; we want structure, not absolute minimum
_MINIMIZE_MAXITER: int = 2000
_TEMPERATURE_K: float = 300.0
_FF_FILES: tuple[str, str] = ("amber14-all.xml", "implicit/gbn2.xml")

# Internal (solute) dielectric for the GB calculation. OpenMM's default is 1.0.
# The protein-peptide MM/PBSA(GBSA) literature finds εin ~1.4-2.0 best on large
# curated sets (JCIM 2018, 10.1021/acs.jcim.8b00248), but our own screen on the
# PepSet complexes (scripts/screen_dielectric.py, 2026-06) did NOT reproduce this:
# εin=2/4 inverted the ΔG correlation sign; only εin=1 kept the physically
# correct sign (+0.58 on n=4). That screen is inconclusive (n=4 after 2 crop
# failures, and the speed-crop confounds the electrostatics εin scales), so we
# keep the OpenMM default 1.0 until a larger structurally-resolved Kd set settles
# it. See docs/scoring_overhaul_plan.md §5 decision log. The value is now a tunable
# parameter (compute_mmgbsa_single(solute_dielectric=...)) so re-screening is cheap.
_SOLUTE_DIELECTRIC: float = 1.0
_SOLVENT_DIELECTRIC: float = 78.5


# ---------------------------------------------------------------------------
# Platform selection
# ---------------------------------------------------------------------------

def _pdbfixer_addH(pdb_path: Path) -> Path:
    """Run pdbfixer on a PDB to add missing terminal atoms + hydrogens.

    Returns a path to a written file (a temp file when fixes were applied,
    or the original path when pdbfixer produced no changes). Handles:

      * Pocket-cropped receptors with truncated backbone at the crop
        boundary (otherwise Modeller.addHydrogens chokes with "No template
        found for residue X (ALA). Chain missing terminal capping group").
      * Peptide poses written as heavy-atoms-only (RAPiDock / minimization
        output) — pdbfixer adds NTermini/CTermini hydrogens that match
        the AMBER ff14SB template set used downstream.
    """
    try:
        from pdbfixer import PDBFixer  # type: ignore[import-untyped]
        import openmm.app as app
    except ImportError:
        logger.warning(
            "pdbfixer not available — MM-GBSA may fail on terminal residues"
        )
        return pdb_path

    fixer = PDBFixer(filename=str(pdb_path))
    # findMissingResidues finds INTERNAL gaps (chain breaks); we don't fill
    # those because they would change connectivity. Clear it so addMissingAtoms
    # only handles partially-resolved residues, not invented sequence.
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.4)

    tmp = Path(tempfile.mkstemp(prefix="mmgbsa_fixed_", suffix=".pdb")[1])
    with tmp.open("w") as fh:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, fh)
    return tmp


def _get_platform(force_cpu: bool):
    """Return (platform, properties) for CUDA → OpenCL → CPU in priority order.

    CUDA platform object is returned even when CUDA is unavailable; the actual
    failure surfaces at Context creation. Callers catch that and retry on CPU.

    Args:
        force_cpu: If True, always return the CPU platform regardless of GPU availability.

    Returns:
        Tuple of (openmm.Platform, dict) where dict holds platform properties
        (e.g. {'Precision': 'mixed'} for CUDA).
    """
    import os

    import openmm

    # CPU platform threads: pin to physical cores (os.cpu_count() is logical).
    n_logical = os.cpu_count() or 1
    cpu_props = {"Threads": str(max(1, n_logical // 2) if n_logical > 2 else n_logical)}

    if force_cpu:
        logger.debug("MM-GBSA: using CPU platform (--mmgbsa-cpu-only)")
        return openmm.Platform.getPlatformByName("CPU"), cpu_props

    # CUDA → NVIDIA; OpenCL → AMD, Intel GPU, and Apple (OpenMM has no Metal
    # backend, so Apple Silicon runs the GPU leg through OpenCL).
    for name, props in [
        ("CUDA", {"DeviceIndex": "0", "Precision": "mixed"}),
        ("OpenCL", {"DeviceIndex": "0", "Precision": "single"}),
    ]:
        try:
            platform = openmm.Platform.getPlatformByName(name)
            logger.debug("MM-GBSA: selected %s platform", name)
            return platform, props
        except Exception:
            continue

    logger.debug("MM-GBSA: GPU unavailable, using CPU platform")
    return openmm.Platform.getPlatformByName("CPU"), cpu_props


def _make_integrator(temperature_k: float = _TEMPERATURE_K):
    """Return a fresh LangevinMiddleIntegrator (required by Context; not stepped)."""
    import openmm
    import openmm.unit as unit
    return openmm.LangevinMiddleIntegrator(
        temperature_k * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds,
    )


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _context_energy_kcal(
    topology,
    positions,
    ff,
    platform,
    platform_props: dict,
    minimize: bool = False,
    solute_dielectric: float = _SOLUTE_DIELECTRIC,
    solvent_dielectric: float = _SOLVENT_DIELECTRIC,
) -> tuple[float, object]:
    """Build an OpenMM Context, optionally minimize, return (energy_kcal, context).

    On CUDA context creation failure, falls back to CPU automatically.

    Args:
        topology: OpenMM Topology object.
        positions: OpenMM Quantity of positions (nm).
        ff: openmm.app.ForceField instance.
        platform: openmm.Platform to use.
        platform_props: Platform-specific properties dict.
        minimize: If True, run LocalEnergyMinimizer before reading energy.
        solute_dielectric: GB internal (solute) dielectric εin. Default
            ``_SOLUTE_DIELECTRIC``. Must match across complex/receptor/peptide
            energies in a single ΔG calculation.
        solvent_dielectric: GB external (solvent) dielectric. Default 78.5.

    Returns:
        Tuple of (potential_energy_kcal, context).
    """
    import openmm
    import openmm.app as app
    import openmm.unit as unit

    system = ff.createSystem(
        topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        soluteDielectric=solute_dielectric,
        solventDielectric=solvent_dielectric,
    )
    integrator = _make_integrator()

    try:
        ctx = openmm.Context(system, integrator, platform, platform_props)
    except Exception as exc:
        if platform.getName() in ("CUDA", "OpenCL"):
            logger.warning(
                "MM-GBSA: %s context failed (%s); falling back to CPU",
                platform.getName(), exc,
            )
            cpu = openmm.Platform.getPlatformByName("CPU")
            integrator = _make_integrator()  # fresh integrator; prior one may be bound to failed ctx
            ctx = openmm.Context(system, integrator, cpu, {})
        else:
            raise

    ctx.setPositions(positions)

    if minimize:
        openmm.LocalEnergyMinimizer.minimize(ctx, _MINIMIZE_TOL, _MINIMIZE_MAXITER)

    e_kj = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    return float(e_kj) * _KJ_TO_KCAL, ctx


# Conformational-entropy penalty coefficient (kcal/mol per unit Cα radius-of-gyration-per-residue).
# Single-snapshot MM-GBSA omits −TΔS_conf, so it over-rates EXTENDED peptides (which pay an ordering
# cost it cannot see). The MM-GBSA residual vs experiment tracks rg_per_L consistently across datasets
# (corr +0.34 crystal-65 / +0.39 the-98, +0.49 pooled; docs E64). Adding α·rg_per_L un-flips MM-GBSA's
# cross-family sign and lifts pooled Pearson 0.05→0.48. α calibrated on pooled crystal-65 + the-98
# (n=156). NOT applied by default (entropy_penalty=False) until re-benchmarked per CLAUDE.md §7.
_CONF_PENALTY_ALPHA = 5.4


def conformational_entropy_penalty(peptide_pdb: Path, alpha: float = _CONF_PENALTY_ALPHA) -> float:
    """Cheap conformational-entropy penalty MM-GBSA omits: +α·(Cα radius of gyration per residue).

    Extended peptides (high rg_per_L) lose more conformational freedom on binding, a −TΔS_conf cost
    a single-snapshot enthalpy+solvation estimate cannot capture. Positive return = weaker binding.

    Args:
        peptide_pdb: PDB of the peptide pose (peptide-only).
        alpha: Penalty coefficient in kcal/mol per unit rg_per_L (default calibrated 5.4).

    Returns:
        Penalty in kcal/mol to ADD to ΔG_bind (0.0 if fewer than two Cα atoms).
    """
    import numpy as np  # noqa: PLC0415
    import openmm.app as app  # noqa: PLC0415

    pdb = app.PDBFile(str(peptide_pdb))
    cas = np.array([list(pos.value_in_unit(pos.unit)) for atom, pos in
                    zip(pdb.topology.atoms(), pdb.positions) if atom.name == "CA"])
    if len(cas) < 2:
        return 0.0
    rg = float(np.sqrt(((cas - cas.mean(0)) ** 2).sum(1).mean()))
    return alpha * (rg / len(cas))


def compute_mmgbsa_single(
    pose_pdb: Path,
    receptor_pdb: Path,
    force_cpu: bool = False,
    solute_dielectric: float = _SOLUTE_DIELECTRIC,
    solvent_dielectric: float = _SOLVENT_DIELECTRIC,
    three_traj: bool = False,
    entropy_penalty: bool = False,
) -> float:
    """Compute MM-GBSA ΔG_bind for a single docked pose.

    Combines the receptor PDB and peptide pose PDB into a single OpenMM
    topology using Modeller, adds hydrogens, minimizes in GBn2 implicit
    solvent, then evaluates receptor-only and peptide-only energies at the
    minimized geometry (single-trajectory approximation).

    ΔG_bind = E(complex) − E(receptor) − E(peptide)

    Args:
        pose_pdb: Path to the peptide pose PDB (peptide-only, from RAPiDock).
        receptor_pdb: Path to the cleaned receptor PDB (protein only, no solvent).
        force_cpu: If True, bypass GPU and use CPU OpenMM platform.

    Returns:
        ΔG_bind in kcal/mol. More negative = stronger binding.

    Raises:
        RuntimeError: If OpenMM or pdbfixer are not installed.
        ValueError: If force field fails to parameterize the complex.
    """
    try:
        import openmm.app as app
        import openmm.unit as unit
    except ImportError as exc:
        raise RuntimeError(
            "OpenMM is required for MM-GBSA refinement. "
            "Install it with: conda install -c conda-forge 'openmm>=8.1'"
        ) from exc

    platform, props = _get_platform(force_cpu)
    ff = app.ForceField(*_FF_FILES)

    # 1. Load receptor and peptide separately, running each through pdbfixer
    #    to add missing heavy atoms + terminal-aware hydrogens. Without this
    #    step, Modeller.addHydrogens silently fails on:
    #      * pocket-cropped receptors (residue at the crop boundary has a
    #        truncated backbone — "missing 1 C atom, chain missing terminal
    #        capping group"), and
    #      * peptide poses (RAPiDock writes heavy-atoms-only PDBs — Modeller
    #        sees the N-terminal residue and can't decide ALA vs NALA).
    #    pdbfixer ships with the terminal residue templates and handles both.
    receptor_pdb_capped = _pdbfixer_addH(receptor_pdb)
    peptide_pdb_capped = _pdbfixer_addH(pose_pdb)
    try:
        receptor_obj = app.PDBFile(str(receptor_pdb_capped))
        peptide_obj = app.PDBFile(str(peptide_pdb_capped))
    finally:
        # Clean up the temp files pdbfixer wrote.
        for tmp in (receptor_pdb_capped, peptide_pdb_capped):
            if tmp != receptor_pdb and tmp != pose_pdb:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
    n_rec_chains = sum(1 for _ in receptor_obj.topology.chains())

    # 2. Combine into one topology via Modeller
    modeller = app.Modeller(receptor_obj.topology, receptor_obj.positions)
    modeller.add(peptide_obj.topology, peptide_obj.positions)

    # 3. Modeller's addHydrogens is now a no-op (pdbfixer already added them)
    #    but call it anyway so any chain-junction residues get standardized.
    try:
        modeller.addHydrogens(ff, pH=7.4)
    except Exception as exc:
        logger.debug("MM-GBSA: post-fixer addHydrogens noop reported %s", exc)

    combined_topology = modeller.topology
    combined_positions = modeller.positions

    # 4. Minimize complex and capture energy + minimized positions
    e_complex, ctx_complex = _context_energy_kcal(
        combined_topology, combined_positions, ff, platform, props, minimize=True,
        solute_dielectric=solute_dielectric, solvent_dielectric=solvent_dielectric,
    )
    min_positions = ctx_complex.getState(getPositions=True).getPositions()

    # 5. Receptor-only energy at minimized geometry
    #    Delete the peptide chains (chains n_rec_chains onwards)
    mod_rec = app.Modeller(combined_topology, min_positions)
    all_chains = list(mod_rec.topology.chains())
    peptide_chains = all_chains[n_rec_chains:]
    if peptide_chains:
        mod_rec.delete(peptide_chains)
    # 1-traj: read receptor energy at the bound geometry (components share the
    # complex's conformation). 3-traj: relax the receptor on its own, capturing
    # the conformational reorganization energy on dissociation.
    e_receptor, _ = _context_energy_kcal(
        mod_rec.topology, mod_rec.positions, ff, platform, props, minimize=three_traj,
        solute_dielectric=solute_dielectric, solvent_dielectric=solvent_dielectric,
    )

    # 6. Peptide-only energy at minimized geometry
    #    Delete the receptor chains (chains 0..n_rec_chains-1)
    mod_pep = app.Modeller(combined_topology, min_positions)
    all_chains_pep = list(mod_pep.topology.chains())
    receptor_chains = all_chains_pep[:n_rec_chains]
    if receptor_chains:
        mod_pep.delete(receptor_chains)
    # 3-traj matters most here: a free linear peptide is disordered, so reading
    # its energy from the bound (ordered) geometry over-stabilizes binding. The
    # extra minimization recovers the peptide's relaxed unbound energy.
    e_peptide, _ = _context_energy_kcal(
        mod_pep.topology, mod_pep.positions, ff, platform, props, minimize=three_traj,
        solute_dielectric=solute_dielectric, solvent_dielectric=solvent_dielectric,
    )

    delta_g = e_complex - e_receptor - e_peptide
    logger.debug(
        "MM-GBSA: E_complex=%.2f E_receptor=%.2f E_peptide=%.2f ΔG=%.2f kcal/mol",
        e_complex, e_receptor, e_peptide, delta_g,
    )
    if entropy_penalty:
        pen = conformational_entropy_penalty(pose_pdb)
        logger.debug("MM-GBSA conformational-entropy penalty +%.2f kcal/mol (rg_per_L)", pen)
        delta_g += pen
    return delta_g


# ---------------------------------------------------------------------------
# Top-K selection and batch refinement
# ---------------------------------------------------------------------------

def _select_topk_representatives(
    scored_poses: list[ScoredPose],
    cluster_result: ClusterResult,
    k: int,
) -> list[ScoredPose]:
    """Select the top-K cluster representatives for MM-GBSA refinement.

    For each cluster: find the pose with the lowest (best) hybrid_score.
    Sort those cluster representatives by cluster mean_hybrid_score ascending.
    Return the top-K of those.

    This guarantees diversity (one pose per binding mode) rather than
    taking the raw top-K which might all come from the same cluster.

    Args:
        scored_poses: All scored poses with cluster_id assigned.
        cluster_result: Completed ClusterResult with per_cluster_stats.
        k: Maximum number of representatives to return.

    Returns:
        List of up to k ScoredPose objects, one per cluster, best clusters first.
    """
    pose_by_idx: dict[int, ScoredPose] = {p.pose_idx: p for p in scored_poses}

    # Build cluster → [poses] map
    cluster_poses: dict[int, list[ScoredPose]] = {}
    for pose in scored_poses:
        if pose.cluster_id is None:
            continue
        cluster_poses.setdefault(pose.cluster_id, []).append(pose)

    # One representative per cluster: best hybrid_score
    representatives: list[tuple[float, ScoredPose]] = []
    for stats in cluster_result.per_cluster_stats:
        cid = stats["cluster_id"]
        poses_in_cluster = cluster_poses.get(cid, [])
        if not poses_in_cluster:
            continue
        best = min(
            poses_in_cluster,
            key=lambda p: p.hybrid_score if p.hybrid_score is not None else float("inf"),
        )
        mean_score = stats.get("mean_hybrid_score", float("inf"))
        representatives.append((mean_score, best))

    # Sort clusters by mean hybrid_score ascending (best cluster first)
    representatives.sort(key=lambda t: t[0])
    return [pose for _, pose in representatives[:k]]


def refine_topk_poses(
    scored_poses: list[ScoredPose],
    cluster_result: ClusterResult,
    config: DockConfig,
) -> None:
    """Run MM-GBSA refinement on top-K cluster representatives, mutating mmgbsa_dg in-place.

    Selects one representative per cluster (best hybrid_score), sorts by cluster
    mean, takes the top config.refine_topk, runs MM-GBSA on each. On per-pose
    failure: logs a warning and leaves mmgbsa_dg=None for that pose.

    Uses the cleaned receptor (receptor_for_rapidock.pdb if present, else
    config.receptor_path) so the MM-GBSA complex matches what was scored.

    Args:
        scored_poses: All scored poses; mmgbsa_dg mutated in-place for top-K.
        cluster_result: Completed ClusterResult with per_cluster_stats.
        config: DockConfig with refine_topk, mmgbsa_cpu_only, output_dir,
            and receptor_path.

    Raises:
        RuntimeError: If OpenMM is not installed (raised by compute_mmgbsa_single).
    """
    k = config.refine_topk
    if k is None or k <= 0:
        return

    cleaned = config.output_dir / "receptor_for_rapidock.pdb"
    receptor_pdb = cleaned.resolve() if cleaned.exists() else config.receptor_path.resolve()
    logger.info(
        "Stage 3.5: MM-GBSA refinement of top %d cluster representatives (receptor: %s)",
        k, receptor_pdb.name,
    )

    representatives = _select_topk_representatives(scored_poses, cluster_result, k)

    # Step-2 pose gating: don't spend MM-GBSA on poses that diffused into the
    # receptor (is_clashed) — their bound geometry is unphysical and the ΔG is
    # garbage. Keep them only if gating would leave nothing to score.
    gated = [p for p in representatives if not p.is_clashed]
    if not gated:
        logger.warning("MM-GBSA: all %d reps are clashed; scoring them anyway", len(representatives))
        gated = representatives
    elif len(gated) < len(representatives):
        logger.info("MM-GBSA gating: dropped %d clashed reps, %d remain",
                    len(representatives) - len(gated), len(gated))

    logger.info(
        "MM-GBSA: %d representatives (mode=%s%s, εin=%.1f)",
        len(gated),
        "3-traj" if config.mmgbsa_3traj else "1-traj",
        "+IE" if config.mmgbsa_include_ie else "",
        config.mmgbsa_solute_dielectric,
    )

    n_ok = 0
    for pose in gated:
        try:
            dg = compute_mmgbsa_single(
                pose_pdb=pose.pdb_path.resolve(),
                receptor_pdb=receptor_pdb,
                force_cpu=config.mmgbsa_cpu_only,
                solute_dielectric=config.mmgbsa_solute_dielectric,
                three_traj=config.mmgbsa_3traj,
            )
            # Step-3: add the signed Interaction-Entropy −TΔS (favours rigid
            # interfaces, penalises floppy ones). Only when requested, since it
            # needs a short trajectory per pose.
            if config.mmgbsa_include_ie:
                from hybridock_pep.scoring.interaction_entropy import (  # noqa: PLC0415
                    interaction_entropy, sample_interaction_energies,
                )
                e_int = sample_interaction_energies(
                    pose_pdb=pose.pdb_path.resolve(),
                    receptor_pdb=receptor_pdb,
                    force_cpu=config.mmgbsa_cpu_only,
                    solute_dielectric=config.mmgbsa_solute_dielectric,
                )
                minus_tds = interaction_entropy(e_int)
                dg = dg + minus_tds
                logger.info("MM-GBSA pose %d: ΔH≈%.2f −TΔS_IE=%.2f → ΔG=%.2f",
                            pose.pose_idx, dg - minus_tds, minus_tds, dg)
            pose.mmgbsa_dg = dg
            n_ok += 1
            logger.info("MM-GBSA pose %d: ΔG = %.2f kcal/mol", pose.pose_idx, dg)
        except Exception as exc:
            logger.warning("MM-GBSA failed for pose %d (%s); skipping", pose.pose_idx, exc)

    # Step-2 two-step workflow: surface the MM-GBSA (affinity) re-ranking of the
    # refined poses. The diffusion/Vina rank chose the binding *mode*; MM-GBSA
    # ΔG re-orders within the refined set. Downstream output ranks by mmgbsa_dg
    # when present, so we just log the re-rank here for traceability.
    refined = sorted((p for p in gated if p.mmgbsa_dg is not None), key=lambda p: p.mmgbsa_dg)
    if refined:
        order = ", ".join(f"#{p.pose_idx}={p.mmgbsa_dg:.1f}" for p in refined[:5])
        logger.info("MM-GBSA affinity re-rank (best first): %s", order)

    logger.info("Stage 3.5 complete: %d/%d MM-GBSA calculations succeeded", n_ok, len(gated))
