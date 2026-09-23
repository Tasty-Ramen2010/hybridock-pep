"""Conformational energy landscape of a peptide — the free-state term a bound pose cannot show.

E417 established why this module exists. On the OppA K-X-K series (19 crystal structures, one receptor,
one varying position, 3.00 kcal/mol of affinity range) the *bound* structures are identical to within
coordinate error: pocket Cα RMSD 0.141 Å median, peptide backbone RMSD 0.158 Å, and nothing measurable
in the bound state tracks affinity (best candidate, bridging waters, permutation p = 0.20). The
information distinguishing those peptides is not in the bound state at all, so no scorer reading a bound
pose can resolve them. What is left is the **free** state.

The physics this computes. A peptide in solution explores an ensemble of conformations; on binding it is
confined to (approximately) one. The free-energy cost of that confinement is

    ΔG_conf = −RT ln P_native,    P_native = P(free peptide is already in the bound conformation)

A peptide that is pre-organised — already spending much of its time in the bound shape — pays little;
a floppy one pays a lot. This is a genuinely free-state quantity: two peptides with identical bound
structures can differ by kilocalories here, which is exactly the OppA situation.

What the engine returns is the full landscape, not just a scalar: the potential energy of every sampled
conformation, its RMSD to the reference (bound) pose, and the energy-versus-RMSD profile whose minimum
is the relaxed conformation. Callers wanting a single number use ``ΔG_conf``; callers wanting to look at
the shape of the landscape get the arrays.

Relationship to the existing entropy features. ``scoring/free_entropy.py`` estimates a dihedral-histogram
entropy S_free from the same MD machinery and is validated (0.409 → 0.488 pooled LOO, the one universal
win of the June scoring campaign). This module is the free-energy formulation of the same idea: it
measures confinement against a *specific* reference conformation rather than the spread of the ensemble,
which is what lets it distinguish two peptides that are equally floppy but pre-organised differently.

VALIDATION STATUS: the physics is standard, but whether ΔG_conf predicts affinity on our data is an
empirical question, tested in experiments/e418. Do not wire this into the production scorer before
reading that result.

Cost: one implicit-solvent MD run per peptide (ff14SB + GBn2, ~8 s for 60 ps on GPU, minutes on CPU).
Requires OpenMM; every entry point degrades to None without it rather than raising into a docking run.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: Gas constant in kcal/(mol·K).
R_GAS = 1.987204e-3
#: Default sampling temperature (K).
TEMPERATURE = 300.0
#: RMSD (Å) within which a sampled conformation counts as "the bound conformation".
NATIVE_RMSD_CUT = 1.5
#: Atoms used as the conformational coordinate. Cα alone is DEGENERATE for short peptides — a
#: tripeptide has three Cα, so after optimal superposition almost no freedom remains and every frame
#: scores as native (measured: RMSD range 0.06-0.42 Å, P_native = 1.000). For short peptides the real
#: conformational freedom is in the side chains, so heavy atoms are the default coordinate.
RMSD_ATOMS_DEFAULT = "heavy"
#: Floor on P_native so a completely unvisited basin yields a finite (if large) penalty rather than inf.
MIN_P_NATIVE = 1e-4


@dataclass
class Landscape:
    """A sampled conformational energy landscape.

    Attributes:
        energies: Potential energy of each sampled conformation, kcal/mol.
        rmsd: RMSD of each conformation to the reference pose, Å. Empty if no reference was given.
        e_min: Lowest sampled energy, kcal/mol — the bottom of the curve.
        e_minimized: Energy after explicit local minimisation, kcal/mol.
        e_mean: Boltzmann-sampled mean energy, kcal/mol.
        p_native: Fraction of the ensemble within ``NATIVE_RMSD_CUT`` of the reference.
        dg_conf: −RT ln P_native, kcal/mol — the confinement cost of binding.
        n_frames: Number of sampled conformations.
        temperature: Sampling temperature, K.
    """

    energies: np.ndarray
    rmsd: np.ndarray
    e_min: float
    e_minimized: float
    e_mean: float
    p_native: float
    dg_conf: float
    n_frames: int
    temperature: float = TEMPERATURE
    meta: dict = field(default_factory=dict)

    def curve(self, n_bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """Energy as a function of RMSD to the reference — the landscape profile.

        Args:
            n_bins: Number of RMSD bins.

        Returns:
            (bin_centres, mean_energy_per_bin). Empty bins carry NaN. The minimum of this curve is
            the relaxed conformation; its rise away from the reference is the restoring force that
            holds the peptide in the bound shape.

        Raises:
            ValueError: If the landscape was built without a reference pose.
        """
        if self.rmsd.size == 0:
            raise ValueError("no reference pose was supplied; the curve is undefined")
        edges = np.linspace(float(self.rmsd.min()), float(self.rmsd.max()), n_bins + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        means = np.full(n_bins, np.nan)
        idx = np.clip(np.digitize(self.rmsd, edges) - 1, 0, n_bins - 1)
        for b in range(n_bins):
            sel = idx == b
            if sel.any():
                means[b] = float(self.energies[sel].mean())
        return centres, means


def _ca_rmsd(traj: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """RMSD of every frame to a reference, after optimal superposition (Kabsch).

    Args:
        traj: (n_frames, n_atoms, 3) coordinates.
        ref: (n_atoms, 3) reference coordinates.

    Returns:
        (n_frames,) RMSD in the same length units as the inputs.

    Raises:
        ValueError: If the atom counts disagree.
    """
    if traj.shape[1] != ref.shape[0]:
        raise ValueError(f"{traj.shape[1]} atoms per frame but reference has {ref.shape[0]}")
    ref_c = ref - ref.mean(0)
    out = np.empty(len(traj))
    for i, frame in enumerate(traj):
        f = frame - frame.mean(0)
        V, _, Wt = np.linalg.svd(f.T @ ref_c)
        d = np.sign(np.linalg.det(V @ Wt))
        rot = V @ np.diag([1.0, 1.0, d]) @ Wt
        out[i] = float(np.sqrt(((f @ rot - ref_c) ** 2).sum(1).mean()))
    return out


def sample_landscape(peptide_pdb: str | Path, reference_pdb: str | Path | None = None,
                     prod_ps: int = 200, frame_every_ps: float = 0.5,
                     temperature: float = TEMPERATURE, rmsd_atoms: str = RMSD_ATOMS_DEFAULT,
                     platform: str = "CUDA") -> Landscape | None:
    """Sample a peptide's free-state conformational landscape with implicit-solvent MD.

    The peptide is minimised, equilibrated and then sampled at constant temperature in GBn2 implicit
    solvent with ff14SB. Every saved frame contributes its potential energy and, when a reference is
    supplied, its Cα RMSD to that reference.

    Args:
        peptide_pdb: PDB of the peptide alone. Its conformation only sets the starting point.
        reference_pdb: PDB whose Cα geometry defines "the bound conformation". Usually the crystal
            peptide. If omitted, ``rmsd``/``p_native``/``dg_conf`` are left empty/NaN.
        prod_ps: Production length in picoseconds. 200 ps is adequate for a tri- or tetrapeptide and
            badly under-samples anything long; the returned ``meta`` records it so callers can judge.
        frame_every_ps: Sampling interval.
        temperature: Simulation temperature in kelvin.
        rmsd_atoms: Conformational coordinate — "heavy" (all non-hydrogen atoms, the default and the
            only sensible choice for short peptides) or "ca" (Cα only, degenerate below ~6 residues).
        platform: OpenMM platform name; falls back to CPU if unavailable.

    Returns:
        A :class:`Landscape`, or None if OpenMM is missing or the simulation diverges.
    """
    try:
        import openmm as mm
        from openmm import app, unit
    except ImportError:
        logger.warning("OpenMM not available; conformational landscape disabled")
        return None

    try:
        pdb = app.PDBFile(str(peptide_pdb))
        ff = app.ForceField("amber14/protein.ff14SB.xml", "implicit/gbn2.xml")
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(ff)
        system = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff,
                                 constraints=app.HBonds)
        integrator = mm.LangevinMiddleIntegrator(temperature * unit.kelvin,
                                                 1.0 / unit.picosecond,
                                                 0.002 * unit.picoseconds)
        try:
            plat = mm.Platform.getPlatformByName(platform)
            sim = app.Simulation(modeller.topology, system, integrator, plat)
        except (mm.OpenMMException, Exception):  # noqa: BLE001 - platform lookup raises broadly
            sim = app.Simulation(modeller.topology, system, integrator,
                                 mm.Platform.getPlatformByName("CPU"))

        sim.context.setPositions(modeller.positions)
        sim.minimizeEnergy(maxIterations=2000)
        e_minimized = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilocalorie_per_mole)

        sim.context.setVelocitiesToTemperature(temperature * unit.kelvin)
        sim.step(int(20 / 0.002))  # 20 ps equilibration, discarded

        if rmsd_atoms == "ca":
            sel_idx = [a.index for a in modeller.topology.atoms() if a.name == "CA"]
        else:
            sel_idx = [a.index for a in modeller.topology.atoms() if a.element is not None
                       and a.element.symbol != "H"]
        steps_per_frame = max(1, int(frame_every_ps / 0.002))
        n_frames = max(1, int(prod_ps / frame_every_ps))
        energies, coords = [], []
        for _ in range(n_frames):
            sim.step(steps_per_frame)
            state = sim.context.getState(getEnergy=True, getPositions=True)
            e = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
            if not np.isfinite(e):
                logger.warning("MD diverged for %s", peptide_pdb)
                return None
            energies.append(e)
            pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
            coords.append(pos[sel_idx])
    except (ValueError, KeyError, OSError) as exc:
        logger.warning("landscape sampling failed for %s: %s", peptide_pdb, exc)
        return None
    except Exception as exc:  # noqa: BLE001 - OpenMM raises bare Exception on force-field mismatch
        logger.warning("landscape sampling failed for %s: %s", peptide_pdb, exc)
        return None

    energies = np.array(energies, float)
    traj = np.array(coords, float)

    rmsd = np.array([], float)
    p_native, dg_conf = float("nan"), float("nan")
    if reference_pdb is not None:
        try:
            ref_pdb = app.PDBFile(str(reference_pdb))
            ref_pos = np.array(ref_pdb.positions.value_in_unit(unit.angstrom), float)
            if rmsd_atoms == "ca":
                ref_sel = [a.index for a in ref_pdb.topology.atoms() if a.name == "CA"]
            else:
                ref_sel = [a.index for a in ref_pdb.topology.atoms()
                           if a.element is not None and a.element.symbol != "H"]
            ref_ca = np.array([ref_pos[i] for i in ref_sel], float)
            if len(ref_ca) == traj.shape[1]:
                rmsd = _ca_rmsd(traj, ref_ca)
                p_native = max(float((rmsd < NATIVE_RMSD_CUT).mean()), MIN_P_NATIVE)
                dg_conf = float(-R_GAS * temperature * np.log(p_native))
            else:
                logger.warning("reference has %d selected atoms, trajectory %d; skipping RMSD",
                               len(ref_ca), traj.shape[1])
        except (ValueError, KeyError, OSError) as exc:
            logger.warning("could not use reference %s: %s", reference_pdb, exc)

    return Landscape(energies=energies, rmsd=rmsd, e_min=float(energies.min()),
                     e_minimized=float(e_minimized), e_mean=float(energies.mean()),
                     p_native=p_native, dg_conf=dg_conf, n_frames=len(energies),
                     temperature=temperature,
                     meta={"prod_ps": prod_ps, "n_rmsd_atoms": int(traj.shape[1]),
                           "rmsd_atoms": rmsd_atoms})


def confinement_cost(free: Landscape, bound: Landscape | None = None) -> float:
    """Free-energy cost of confining the peptide to its bound conformation, kcal/mol.

    With only a free-state landscape this is ``free.dg_conf`` = −RT ln P_native. When a landscape
    sampled *in the pocket* is also supplied, the estimate is refined by the width of the bound basin:
    a bound state that itself retains residual motion costs less than a fully frozen one.

    Args:
        free: Landscape of the unbound peptide, sampled against the bound reference.
        bound: Optional landscape of the peptide sampled inside the pocket.

    Returns:
        Confinement cost in kcal/mol (positive = unfavourable), or NaN if unavailable.
    """
    if not np.isfinite(free.dg_conf):
        return float("nan")
    if bound is None or bound.rmsd.size == 0:
        return free.dg_conf
    residual = max(float((bound.rmsd < NATIVE_RMSD_CUT).mean()), MIN_P_NATIVE)
    return float(free.dg_conf + R_GAS * free.temperature * np.log(residual))


def per_residue_entropy(peptide_pdb: str | Path, prod_ps: int = 300,
                        temperature: float = TEMPERATURE,
                        platform: str = "CUDA") -> dict | None:
    """Per-residue free-state conformational entropy, estimated from the shape of the landscape.

    This is the estimator that worked where the thresholded one did not (experiments/e420). Two
    findings drove its design:

    * **Do not threshold.** ``ΔG_conf = −RT ln P_native`` counts how many frames fall inside an RMSD
      cutoff and discards everything about the shape of the distribution. On the OppA series that
      estimator was null (r = −0.041, permutation p = 0.895) while the *width* of the sampled energy
      distribution — the density of states, computed from the very same trajectories — reached
      r = +0.530 (p = 0.023). The information was present; the cutoff threw it away.

    * **Do not average over the peptide.** Per-position spread on that series gave the varying
      position r = +0.602 (p = 0.0086) against +0.145 and −0.022 for the two positions that are the
      same residue in every member. Collapsing those into one number destroys the signal.

    Sign convention: larger values mean a freer residue, which loses more entropy on binding and
    therefore binds more weakly — so these were expected to correlate POSITIVELY with ΔG.

    **THIS DOES NOT PREDICT AFFINITY. Do not wire it into the scorer.** experiments/e421 tested it on
    19 held-out single-mutation pairs over 20 receptors — congeneric comparisons from outside the
    series the hypothesis was formed on — with the predictions fixed in advance, and all three failed:
    the per-residue term at the mutated position gave r = −0.201 (p = 0.413, sign opposite to
    prediction), the whole-peptide curve width gave r = +0.066 (p = 0.787), and the negative control
    (the same quantity at the UNCHANGED positions, r = +0.239) outperformed the test, so nothing here
    is position-specific. On that held-out set the only term that predicts ΔΔG is the change in
    side-chain size (r = −0.499, p = 0.029) — which did NOT predict affinity on OppA (r = +0.168).
    Two congeneric series disagree about which descriptor matters, which is E408's sign-inversion
    (r = −0.337, ~8 sd from its null) observed directly.

    The function is kept because it computes what it claims to compute and the machinery is reusable;
    the negative result is recorded here so it is not rediscovered.

    Args:
        peptide_pdb: PDB of the peptide alone.
        prod_ps: production MD length in picoseconds.
        temperature: simulation temperature in kelvin.
        platform: OpenMM platform name; falls back to CPU.

    Returns:
        Dict with ``backbone_spread`` and ``sidechain_spread`` (one value per residue, in the
        sequence order of the input), ``energy_spread`` (whole-peptide, the curve-width term),
        ``resnames``, and ``n_frames``; or None if OpenMM is missing or the run diverges.
    """
    try:
        import openmm as mm
        from openmm import app, unit
    except ImportError:
        logger.warning("OpenMM not available; per-residue entropy disabled")
        return None
    try:
        pdb = app.PDBFile(str(peptide_pdb))
        ff = app.ForceField("amber14/protein.ff14SB.xml", "implicit/gbn2.xml")
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(ff)
        system = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff,
                                 constraints=app.HBonds)
        integ = mm.LangevinMiddleIntegrator(temperature * unit.kelvin, 1 / unit.picosecond,
                                            0.002 * unit.picoseconds)
        try:
            sim = app.Simulation(modeller.topology, system, integ,
                                 mm.Platform.getPlatformByName(platform))
        except Exception:  # noqa: BLE001 - platform lookup raises broadly
            sim = app.Simulation(modeller.topology, system, integ,
                                 mm.Platform.getPlatformByName("CPU"))
        sim.context.setPositions(modeller.positions)
        sim.minimizeEnergy(maxIterations=2000)
        sim.context.setVelocitiesToTemperature(temperature * unit.kelvin)
        sim.step(int(30 / 0.002))

        residues = list(modeller.topology.residues())
        bb_idx, sc_idx = [], []
        for res in residues:
            names = {a.name: a.index for a in res.atoms()}
            bb_idx.append((names.get("N"), names.get("CA"), names.get("C")))
            sc_idx.append([a.index for a in res.atoms()
                           if a.name not in ("N", "CA", "C", "O", "OXT")
                           and a.element is not None and a.element.symbol != "H"])

        steps = max(1, int(0.5 / 0.002))
        n_frames = max(20, int(prod_ps / 0.5))
        energies = []
        bb_traj: list[list[float]] = [[] for _ in residues]
        sc_traj: list[list[float]] = [[] for _ in residues]
        for _ in range(n_frames):
            sim.step(steps)
            st = sim.context.getState(getEnergy=True, getPositions=True)
            e = st.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
            if not np.isfinite(e):
                logger.warning("MD diverged for %s", peptide_pdb)
                return None
            energies.append(e)
            xyz = np.array(st.getPositions(asNumpy=True).value_in_unit(unit.angstrom), float)
            for k, (n_i, ca_i, c_i) in enumerate(bb_idx):
                if None in (n_i, ca_i, c_i):
                    bb_traj[k].append(np.nan)
                else:
                    v1, v2 = xyz[n_i] - xyz[ca_i], xyz[c_i] - xyz[ca_i]
                    cos = float(np.dot(v1, v2) /
                                (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
                    bb_traj[k].append(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
                # side-chain extension relative to its own CA: a frame-invariant measure of how
                # much that side chain is moving, needing no global superposition
                sc = sc_idx[k]
                if sc and ca_i is not None:
                    sc_traj[k].append(float(np.linalg.norm(xyz[sc] - xyz[ca_i], axis=1).mean()))
                else:
                    sc_traj[k].append(np.nan)
    except Exception as exc:  # noqa: BLE001 - OpenMM raises bare Exception on template mismatch
        logger.warning("per-residue entropy failed for %s: %s", peptide_pdb, exc)
        return None

    return {
        "backbone_spread": [float(np.nanstd(np.array(v, float))) if len(v) else float("nan")
                            for v in bb_traj],
        "sidechain_spread": [float(np.nanstd(np.array(v, float))) if len(v) else float("nan")
                             for v in sc_traj],
        "energy_spread": float(np.std(energies)),
        "resnames": [r.name for r in residues],
        "n_frames": len(energies),
    }
