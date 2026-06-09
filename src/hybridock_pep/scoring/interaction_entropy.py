"""Interaction Entropy (IE) — a signed, per-complex binding entropy estimate.

Replaces the single-sign ``α·N_contact`` / ``w·s_ss_weighted`` entropy proxies
(which can only ever push the score one direction) with a thermodynamically
grounded entropy term derived from the *fluctuations* of the protein-peptide
interaction energy over a short trajectory.

Method (Duan, Gao & Zhang, J. Am. Chem. Soc. 2016; assessed in JCTC 2021,
10.1021/acs.jctc.1c00374):

    -TΔS_IE = kT · ln⟨ exp(β · ΔE_int) ⟩

where ``ΔE_int = E_int - ⟨E_int⟩`` is the instantaneous deviation of the gas-phase
receptor-peptide interaction energy from its trajectory mean, β = 1/kT, and the
average ⟨·⟩ runs over trajectory frames.

Why this fixes the "entropy is always one sign" problem:
  * ``-TΔS_IE`` is computed from an energy *variance*, so it is large for floppy,
    loosely-packed interfaces (high fluctuation) and small for rigid, well-packed
    ones — it varies per complex rather than scaling monotonically with a residue
    count.
  * It enters ΔG_bind as ``ΔG = ΔE_int_mean + ΔG_solv - TΔS_IE``. Because the
    enthalpic and entropic parts are fit/used independently, the *net* entropy
    contribution to ranking is no longer locked to a single sign.

This module is pure-numpy for the estimator (unit-testable without OpenMM); the
OpenMM trajectory sampling lives in ``sample_interaction_energies`` and is only
imported/exercised when an actual trajectory is requested.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Boltzmann constant × 300 K in kcal/mol (kT). RT and kT are numerically equal
# per-molecule in these units.
_KT_300: float = 0.5961612775922  # kcal/mol at 300 K


def interaction_entropy(
    interaction_energies_kcal: np.ndarray | list[float],
    temperature_k: float = 300.0,
) -> float:
    """Compute -TΔS_IE (kcal/mol) from a series of interaction energies.

    Uses the numerically-stable log-sum-exp form of
    ``-TΔS = kT · ln⟨exp(βΔE)⟩`` with ``ΔE = E - mean(E)``.

    Args:
        interaction_energies_kcal: 1-D series of gas-phase receptor-peptide
            interaction energies (kcal/mol), one per trajectory frame.
        temperature_k: Temperature in Kelvin (default 300).

    Returns:
        ``-TΔS_IE`` in kcal/mol. Positive = entropy penalty (unfavorable),
        which is the physically expected sign on binding. Magnitude grows with
        interaction-energy fluctuation.

    Raises:
        ValueError: If fewer than 2 energies are supplied (variance undefined).
    """
    e = np.asarray(interaction_energies_kcal, dtype=float)
    if e.size < 2:
        raise ValueError(
            f"interaction_entropy needs ≥2 frames to estimate fluctuation; got {e.size}"
        )
    kt = _KT_300 * (temperature_k / 300.0)
    beta = 1.0 / kt
    de = e - e.mean()
    # log-sum-exp: ln(mean(exp(β·ΔE))) = logsumexp(β·ΔE) - ln(N)
    x = beta * de
    x_max = float(x.max())
    log_mean_exp = x_max + math.log(float(np.mean(np.exp(x - x_max))))
    return float(kt * log_mean_exp)


def sample_interaction_energies(
    pose_pdb: Path,
    receptor_pdb: Path,
    n_frames: int = 100,
    steps_between_frames: int = 500,
    temperature_k: float = 300.0,
    force_cpu: bool = True,
    solute_dielectric: float | None = None,
) -> np.ndarray:
    """Run a short OpenMM trajectory and record receptor-peptide interaction energy.

    Builds the same GBn2 complex as ``mmgbsa.compute_mmgbsa_single``, minimizes,
    then runs Langevin dynamics, recording at each sampled frame the gas-phase
    interaction energy ``E_int = E_complex - E_receptor - E_peptide`` evaluated at
    the current geometry. The returned series feeds ``interaction_entropy``.

    Defaults are deliberately short (100 frames × 500 steps = 50 ps at 2 fs) —
    IE converges fast and the literature finds short windows sufficient.

    Args:
        pose_pdb: Peptide pose PDB.
        receptor_pdb: Cleaned receptor PDB.
        n_frames: Number of frames to record.
        steps_between_frames: MD steps between recorded frames.
        temperature_k: Thermostat temperature (K).
        force_cpu: Force CPU platform (default True — never contend with GPU).
        solute_dielectric: GB εin; defaults to the mmgbsa module default.

    Returns:
        (n_frames,) array of interaction energies in kcal/mol.

    Raises:
        RuntimeError: If OpenMM is unavailable.
    """
    # Imported lazily so the estimator stays usable without OpenMM installed.
    from hybridock_pep.scoring import mmgbsa as _m

    try:
        import openmm
        import openmm.app as app
        import openmm.unit as unit
    except ImportError as exc:  # pragma: no cover - exercised only with OpenMM absent
        raise RuntimeError(
            "OpenMM is required for IE trajectory sampling. "
            "Install with: conda install -c conda-forge 'openmm>=8.1'"
        ) from exc

    eps_in = solute_dielectric if solute_dielectric is not None else _m._SOLUTE_DIELECTRIC
    platform, props = _m._get_platform(force_cpu)
    ff = app.ForceField(*_m._FF_FILES)

    receptor_capped = _m._pdbfixer_addH(receptor_pdb)
    peptide_capped = _m._pdbfixer_addH(pose_pdb)
    try:
        receptor_obj = app.PDBFile(str(receptor_capped))
        peptide_obj = app.PDBFile(str(peptide_capped))
    finally:
        for tmp in (receptor_capped, peptide_capped):
            if tmp not in (receptor_pdb, pose_pdb):
                tmp.unlink(missing_ok=True)
    n_rec_chains = sum(1 for _ in receptor_obj.topology.chains())

    modeller = app.Modeller(receptor_obj.topology, receptor_obj.positions)
    modeller.add(peptide_obj.topology, peptide_obj.positions)
    try:
        modeller.addHydrogens(ff, pH=7.4)
    except Exception as exc:  # noqa: BLE001
        logger.debug("IE: post-fixer addHydrogens noop reported %s", exc)
    topo = modeller.topology
    positions = modeller.positions

    system = ff.createSystem(
        topo, nonbondedMethod=app.NoCutoff, constraints=app.HBonds,
        soluteDielectric=eps_in, solventDielectric=_m._SOLVENT_DIELECTRIC,
    )
    integrator = openmm.LangevinMiddleIntegrator(
        temperature_k * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
    )
    sim = app.Simulation(topo, system, integrator, platform, props)
    sim.context.setPositions(positions)
    sim.minimizeEnergy(maxIterations=_m._MINIMIZE_MAXITER)

    # Pre-split chain index lists for component energies.
    all_chains = list(topo.chains())
    pep_chain_idx = list(range(n_rec_chains, len(all_chains)))
    rec_chain_idx = list(range(n_rec_chains))

    def _component_energy(positions_q, keep_chain_idx: list[int]) -> float:
        mod = app.Modeller(topo, positions_q)
        chains = list(mod.topology.chains())
        drop = [c for i, c in enumerate(chains) if i not in keep_chain_idx]
        if drop:
            mod.delete(drop)
        e, _ = _m._context_energy_kcal(
            mod.topology, mod.positions, ff, platform, props, minimize=False,
            solute_dielectric=eps_in,
        )
        return e

    energies = np.empty(n_frames, dtype=float)
    for i in range(n_frames):
        sim.step(steps_between_frames)
        state = sim.context.getState(getEnergy=True, getPositions=True)
        e_complex = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) * _m._KJ_TO_KCAL
        pos = state.getPositions()
        e_rec = _component_energy(pos, rec_chain_idx)
        e_pep = _component_energy(pos, pep_chain_idx)
        energies[i] = e_complex - e_rec - e_pep

    logger.info(
        "IE sampling: %d frames, E_int mean=%.2f std=%.2f kcal/mol",
        n_frames, float(energies.mean()), float(energies.std()),
    )
    return energies
