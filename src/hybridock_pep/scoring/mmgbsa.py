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


# ---------------------------------------------------------------------------
# Platform selection
# ---------------------------------------------------------------------------

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
    import openmm

    if force_cpu:
        logger.debug("MM-GBSA: using CPU platform (--mmgbsa-cpu-only)")
        return openmm.Platform.getPlatformByName("CPU"), {}

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
    return openmm.Platform.getPlatformByName("CPU"), {}


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


def compute_mmgbsa_single(
    pose_pdb: Path,
    receptor_pdb: Path,
    force_cpu: bool = False,
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

    # 1. Load receptor and peptide separately
    receptor_obj = app.PDBFile(str(receptor_pdb))
    peptide_obj = app.PDBFile(str(pose_pdb))
    n_rec_chains = sum(1 for _ in receptor_obj.topology.chains())

    # 2. Combine into one topology via Modeller
    modeller = app.Modeller(receptor_obj.topology, receptor_obj.positions)
    modeller.add(peptide_obj.topology, peptide_obj.positions)

    # 3. Add hydrogens using the force field's protonation model at pH 7.4
    try:
        modeller.addHydrogens(ff, pH=7.4)
    except Exception as exc:
        logger.warning("MM-GBSA: addHydrogens failed (%s); proceeding without", exc)

    combined_topology = modeller.topology
    combined_positions = modeller.positions

    # 4. Minimize complex and capture energy + minimized positions
    e_complex, ctx_complex = _context_energy_kcal(
        combined_topology, combined_positions, ff, platform, props, minimize=True
    )
    min_positions = ctx_complex.getState(getPositions=True).getPositions()

    # 5. Receptor-only energy at minimized geometry
    #    Delete the peptide chains (chains n_rec_chains onwards)
    mod_rec = app.Modeller(combined_topology, min_positions)
    all_chains = list(mod_rec.topology.chains())
    peptide_chains = all_chains[n_rec_chains:]
    if peptide_chains:
        mod_rec.delete(peptide_chains)
    e_receptor, _ = _context_energy_kcal(
        mod_rec.topology, mod_rec.positions, ff, platform, props, minimize=False
    )

    # 6. Peptide-only energy at minimized geometry
    #    Delete the receptor chains (chains 0..n_rec_chains-1)
    mod_pep = app.Modeller(combined_topology, min_positions)
    all_chains_pep = list(mod_pep.topology.chains())
    receptor_chains = all_chains_pep[:n_rec_chains]
    if receptor_chains:
        mod_pep.delete(receptor_chains)
    e_peptide, _ = _context_energy_kcal(
        mod_pep.topology, mod_pep.positions, ff, platform, props, minimize=False
    )

    delta_g = e_complex - e_receptor - e_peptide
    logger.debug(
        "MM-GBSA: E_complex=%.2f E_receptor=%.2f E_peptide=%.2f ΔG=%.2f kcal/mol",
        e_complex, e_receptor, e_peptide, delta_g,
    )
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
    logger.info("MM-GBSA: selected %d representatives from %d clusters", len(representatives), k)

    n_ok = 0
    for pose in representatives:
        try:
            dg = compute_mmgbsa_single(
                pose_pdb=pose.pdb_path.resolve(),
                receptor_pdb=receptor_pdb,
                force_cpu=config.mmgbsa_cpu_only,
            )
            pose.mmgbsa_dg = dg
            n_ok += 1
            logger.info(
                "MM-GBSA pose %d: ΔG = %.2f kcal/mol", pose.pose_idx, dg
            )
        except Exception as exc:
            logger.warning(
                "MM-GBSA failed for pose %d (%s); skipping", pose.pose_idx, exc
            )

    logger.info("Stage 3.5 complete: %d/%d MM-GBSA calculations succeeded", n_ok, len(representatives))
