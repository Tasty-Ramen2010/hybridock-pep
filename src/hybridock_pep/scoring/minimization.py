"""OpenMM clash-relief minimization of raw diffusion-model poses (§2.5).

RAPiDock occasionally places side-chain atoms too close together (intra-pose
clashes).  These contacts cause AD4 (steeper LJ potential) to return large
positive scores while Vina (softer Gaussian repulsion) tolerates them.

Strategy: RESTRAINED clash relief with displacement safety check.
  - AMBER ff14SB + GBn2 implicit solvent.
  - Strong harmonic positional restraints on all heavy atoms
    (k = _RESTRAINT_KJ_PER_NM2 = 50 000 kJ/mol/nm²).
    At k=50 000, moving a heavy atom 1 Å costs ~60 kcal/mol of restraint energy;
    only atoms under extreme local repulsion will shift at all.
  - After minimization, if any heavy atom moved more than _MAX_DISPLACEMENT_ANG
    (default 0.5 Å), the minimization is DISCARDED and the original pose is
    returned unchanged.  This prevents cases where severe aromatic–aromatic
    clashes (PHE/TRP side chains) cause 1 Å movement that destroys the Vina
    score (+10 kcal/mol penalty observed on pose_15).

Tested failure modes:
  - Unrestrained vacuum minimization: peptide refolds, Vina −4.87 → +15.
  - k=500 kJ/mol/nm²: still too loose (1 Å costs 0.6 kcal/mol restraint).
  - k=50 000 + no displacement check: severe aromatic clashes survive by
    moving 0.6–1.0 Å at 60 kcal/mol restraint cost.

Output: heavy-atom-only PDB (same residue numbering and chain IDs as input)
if displacement is acceptable; pdb_path unchanged otherwise.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_MAX_ITER = 200
_TOLERANCE = 5.0              # kJ/mol/nm
_RESTRAINT_KJ_PER_NM2 = 50_000.0  # harmonic spring on heavy atoms
_MAX_DISPLACEMENT_ANG = 0.5   # revert if any heavy atom moves more than this


def minimize_pose(
    pdb_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Restrained clash-relief minimization of a single all-atom pose PDB.

    Adds H via pdbfixer, then runs LocalEnergyMinimizer with strong
    positional restraints on all heavy atoms.  If the minimization moves any
    heavy atom beyond _MAX_DISPLACEMENT_ANG, the original pose is returned
    unchanged to preserve the binding conformation.

    Args:
        pdb_path: Absolute path to the raw pose PDB (all-atom, no H).
        output_path: Destination for the minimized PDB.  Defaults to
            <pdb_path.parent>/<stem>_min.pdb.

    Returns:
        Path to the minimized heavy-atom PDB, or pdb_path if minimization
        failed or the displacement check rejected the result.  Never raises.
    """
    try:
        import openmm
        import openmm.app as app
        import openmm.unit as unit
        from pdbfixer import PDBFixer
    except ImportError as exc:
        logger.warning("OpenMM/pdbfixer not installed (%s) — skipping minimization", exc)
        return pdb_path

    if output_path is None:
        output_path = pdb_path.parent / f"{pdb_path.stem}_min.pdb"

    try:
        # Step 1: pdbfixer — add H only; skip missing-residue insertion.
        # RAPiDock poses are all-atom — all heavy atoms are already present.
        # findMissingResidues() inserts residues based on SEQRES templates that
        # MDAnalysis-written PDBs do not have, corrupting the pose.
        # Setting missingResidues={} satisfies findMissingAtoms()'s precondition.
        fixer = PDBFixer(filename=str(pdb_path))
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        try:
            fixer.addMissingHydrogens(7.4)
        except Exception as exc:
            logger.debug("pdbfixer H addition failed for %s (%s)", pdb_path.name, exc)

        # Step 2: AMBER ff14SB + GBn2 system
        ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
        try:
            system = ff.createSystem(fixer.topology, nonbondedMethod=app.NoCutoff)
        except Exception as exc:
            logger.warning(
                "ForceField.createSystem failed for %s (%s); skipping", pdb_path.name, exc
            )
            return pdb_path

        # Step 3: Harmonic positional restraints on all heavy atoms.
        # k=50 000 kJ/mol/nm² → 1 Å movement costs ~60 kcal/mol of restraint.
        # Only atoms under extreme vdW repulsion will move; the displacement
        # safety check below catches any case where movement is too large.
        restraint = openmm.CustomExternalForce(
            "0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
        )
        restraint.addGlobalParameter(
            "k", _RESTRAINT_KJ_PER_NM2 * unit.kilojoule_per_mole / unit.nanometer**2
        )
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")

        start_positions = fixer.positions
        heavy_atom_indices: list[int] = []

        for atom in fixer.topology.atoms():
            element = atom.element.symbol if atom.element else "H"
            if element not in ("H", "D"):
                pos = start_positions[atom.index]
                restraint.addParticle(atom.index, [pos.x, pos.y, pos.z])
                heavy_atom_indices.append(atom.index)
        system.addForce(restraint)

        # Step 4: Minimise with restraints on the fastest available backend
        # (CUDA → HIP → OpenCL → thread-pinned CPU; centralized in hardware.py).
        # On GPU-context failure, fall back to OpenMM's default platform so a
        # quirky driver never breaks Stage 1.5.
        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        from hybridock_pep.hardware import openmm_platform  # noqa: PLC0415
        try:
            platform, props = openmm_platform()
            ctx = openmm.Context(system, integrator, platform, props)
        except Exception as exc:  # noqa: BLE001 — driver/precision quirk → default platform
            logger.debug("minimize_pose: platform %s failed (%s); using default", exc, exc)
            integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
            ctx = openmm.Context(system, integrator)
        ctx.setPositions(start_positions)
        openmm.LocalEnergyMinimizer.minimize(
            ctx,
            tolerance=_TOLERANCE * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=_MAX_ITER,
        )

        # Step 5: Displacement safety check — if any heavy atom moved more
        # than _MAX_DISPLACEMENT_ANG, the pose conformation has changed enough
        # to invalidate the original binding mode; discard and return original.
        end_positions = ctx.getState(getPositions=True).getPositions(asNumpy=True)
        # getPositions(asNumpy=True) still returns a Quantity in OpenMM ≥ 8.x.
        # Stripping units before numpy arithmetic prevents Quantity.__sub__ from
        # calling .unit on the plain numpy start_np (AttributeError crash).
        end_positions_nm = np.array(
            end_positions.value_in_unit(unit.nanometer), dtype=np.float64
        )
        start_np = np.array(
            [[start_positions[i].x, start_positions[i].y, start_positions[i].z]
             for i in heavy_atom_indices],
            dtype=np.float64,
        )  # nm — Vec3.x/.y/.z are plain floats (no unit wrapper)
        end_np = end_positions_nm[heavy_atom_indices]
        max_disp_ang = float(np.max(np.linalg.norm(end_np - start_np, axis=1))) * 10.0  # Å

        if max_disp_ang > _MAX_DISPLACEMENT_ANG:
            logger.info(
                "Clash-relief displaced heavy atoms up to %.2f Å (> %.1f Å threshold); "
                "reverting to original for %s",
                max_disp_ang,
                _MAX_DISPLACEMENT_ANG,
                pdb_path.name,
            )
            return pdb_path

        # Step 6: Write heavy-atom-only PDB
        with open(output_path, "w") as fh:
            app.PDBFile.writeFile(
                fixer.topology,
                ctx.getState(getPositions=True).getPositions(),
                fh,
                keepIds=True,
            )
        _strip_hydrogens(output_path)

        logger.debug(
            "Clash-relief OK: %s max_disp=%.3f Å → %s",
            pdb_path.name,
            max_disp_ang,
            output_path.name,
        )
        return output_path

    except Exception as exc:
        logger.warning(
            "Minimization failed for %s (%s); using original pose", pdb_path.name, exc
        )
        return pdb_path


def minimize_poses_batch(pdb_paths: list[Path], output_dir: Path) -> list[Path]:
    """Minimize a batch of pose PDBs, writing results to output_dir.

    Args:
        pdb_paths: Absolute paths to raw pose PDBs.
        output_dir: Directory for minimized outputs.

    Returns:
        List of result paths (minimized or original) in input order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Clash-relief minimization: %d poses → %s", len(pdb_paths), output_dir)
    results: list[Path] = []
    for pdb_path in pdb_paths:
        dest = output_dir / pdb_path.name
        results.append(minimize_pose(pdb_path, dest))
    logger.info("Minimization complete: %d poses processed", len(results))
    return results


def _strip_hydrogens(pdb_path: Path) -> None:
    """Remove hydrogen ATOM/HETATM records from a PDB file in-place."""
    heavy: list[str] = []
    for line in pdb_path.read_text().splitlines(keepends=True):
        record = line[:6].strip()
        if record in ("ATOM", "HETATM"):
            element = line[76:78].strip() if len(line) > 76 else ""
            name = line[12:16].strip() if len(line) > 16 else ""
            if element in ("H", "D"):
                continue
            if not element and name and name[0] == "H":
                continue
        heavy.append(line)
    pdb_path.write_text("".join(heavy))
