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

import gc
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_MAX_ITER = 200
_TOLERANCE = 5.0              # kJ/mol/nm
_RESTRAINT_KJ_PER_NM2 = 50_000.0  # harmonic spring on heavy atoms
_MAX_DISPLACEMENT_ANG = 0.5   # revert if any heavy atom moves more than this


class _ClashRelief:
    """Reusable OpenMM machinery for restrained clash relief.

    Every pose in a run is the SAME peptide — same sequence, same atoms, same
    topology — and only the coordinates differ.  Rebuilding the ForceField,
    System and Context per pose is therefore pure waste, and it is not merely
    slow: an OpenMM Context is never fully reclaimed when it is dropped, so a
    fresh one per pose leaks on every backend we tried (~300 MB/pose against a
    broken CUDA platform, ~100 MB/pose on OpenCL).  At `--n-samples 100` that
    reached ~10 GB and the kernel OOM-killer took the process partway through
    Stage 1.5 — which surfaced to the user as a freeze at "40/100 poses
    minimized", with no traceback, because SIGKILL cannot be caught.

    Building once and calling ``setPositions`` per pose removes the leak by
    construction and skips 99 redundant kernel compilations.
    """

    def __init__(self) -> None:
        self._ff = None
        self._ctx = None
        self._system = None
        self._integrator = None
        self._restraint = None
        self._signature = None
        self._heavy: list[int] = []

    @staticmethod
    def _signature_of(topology) -> tuple:
        return (
            topology.getNumAtoms(),
            tuple(a.element.symbol if a.element else "H" for a in topology.atoms()),
        )

    def _ensure(self, openmm, app, unit, topology, positions) -> None:
        """Build the Context, or reuse it when the topology is unchanged."""
        signature = self._signature_of(topology)
        if signature == self._signature and self._ctx is not None:
            return
        self.close()  # topology changed (or first call) — rebuild once

        if self._ff is None:  # ForceField parses two large XMLs; parse them once
            self._ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
        self._system = self._ff.createSystem(topology, nonbondedMethod=app.NoCutoff)

        # k=50 000 kJ/mol/nm2 -> 1 A of movement costs ~60 kcal/mol of restraint.
        restraint = openmm.CustomExternalForce(
            "0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
        )
        restraint.addGlobalParameter(
            "k", _RESTRAINT_KJ_PER_NM2 * unit.kilojoule_per_mole / unit.nanometer**2
        )
        for name in ("x0", "y0", "z0"):
            restraint.addPerParticleParameter(name)

        heavy: list[int] = []
        for atom in topology.atoms():
            element = atom.element.symbol if atom.element else "H"
            if element not in ("H", "D"):
                pos = positions[atom.index]
                restraint.addParticle(atom.index, [pos.x, pos.y, pos.z])
                heavy.append(atom.index)
        self._system.addForce(restraint)
        self._restraint = restraint
        self._heavy = heavy

        from hybridock_pep.hardware import openmm_platform  # noqa: PLC0415

        self._integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        try:
            platform, props = openmm_platform()
            self._ctx = openmm.Context(
                self._system, self._integrator, platform, props
            )
        except Exception as exc:  # noqa: BLE001 — driver quirk -> default platform
            logger.debug(
                "clash relief: preferred platform unusable (%s); using OpenMM default",
                exc,
            )
            self._integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
            self._ctx = openmm.Context(self._system, self._integrator)
        self._signature = signature
        logger.debug(
            "clash relief: built Context on %s for %d atoms",
            self._ctx.getPlatform().getName(), signature[0],
        )

    def _retarget(self, positions) -> None:
        """Point the restraints at THIS pose's coordinates."""
        for slot, atom_index in enumerate(self._heavy):
            pos = positions[atom_index]
            self._restraint.setParticleParameters(
                slot, atom_index, [pos.x, pos.y, pos.z]
            )
        self._restraint.updateParametersInContext(self._ctx)

    def close(self) -> None:
        self._ctx = None
        self._system = None
        self._integrator = None
        self._restraint = None
        self._signature = None
        self._heavy = []
        gc.collect()

    def run(self, pdb_path: Path, output_path: Path) -> Path:
        """Minimize one pose, returning the written path or pdb_path unchanged."""
        try:
            import openmm
            import openmm.app as app
            import openmm.unit as unit
            from pdbfixer import PDBFixer
        except ImportError as exc:
            logger.warning(
                "OpenMM/pdbfixer not installed (%s) — skipping minimization", exc
            )
            return pdb_path

        try:
            # Step 1: pdbfixer — add H only; skip missing-residue insertion.
            # RAPiDock poses are all-atom, and findMissingResidues() would insert
            # residues from SEQRES templates that MDAnalysis-written PDBs lack,
            # corrupting the pose. missingResidues={} satisfies findMissingAtoms().
            fixer = PDBFixer(filename=str(pdb_path))
            fixer.missingResidues = {}
            fixer.findMissingAtoms()
            try:
                fixer.addMissingHydrogens(7.4)
            except Exception as exc:
                logger.debug(
                    "pdbfixer H addition failed for %s (%s)", pdb_path.name, exc
                )

            start_positions = fixer.positions
            self._ensure(openmm, app, unit, fixer.topology, start_positions)
            self._retarget(start_positions)

            self._ctx.setPositions(start_positions)
            openmm.LocalEnergyMinimizer.minimize(
                self._ctx,
                tolerance=_TOLERANCE * unit.kilojoule_per_mole / unit.nanometer,
                maxIterations=_MAX_ITER,
            )

            # Step 2: displacement safety check — a pose that moved far enough to
            # change its binding mode is discarded rather than kept.
            end_positions = self._ctx.getState(getPositions=True).getPositions(
                asNumpy=True
            )
            end_nm = np.array(
                end_positions.value_in_unit(unit.nanometer), dtype=np.float64
            )
            start_np = np.array(
                [
                    [start_positions[i].x, start_positions[i].y, start_positions[i].z]
                    for i in self._heavy
                ],
                dtype=np.float64,
            )  # nm — Vec3.x/.y/.z are plain floats (no unit wrapper)
            end_np = end_nm[self._heavy]
            max_disp_ang = (
                float(np.max(np.linalg.norm(end_np - start_np, axis=1))) * 10.0
            )

            if max_disp_ang > _MAX_DISPLACEMENT_ANG:
                logger.info(
                    "Clash-relief displaced heavy atoms up to %.2f A (> %.1f A "
                    "threshold); reverting to original for %s",
                    max_disp_ang, _MAX_DISPLACEMENT_ANG, pdb_path.name,
                )
                return pdb_path

            # Step 3: write heavy-atom-only PDB
            with open(output_path, "w") as fh:
                app.PDBFile.writeFile(
                    fixer.topology,
                    self._ctx.getState(getPositions=True).getPositions(),
                    fh,
                    keepIds=True,
                )
            _strip_hydrogens(output_path)
            logger.debug(
                "Clash-relief OK: %s max_disp=%.3f A -> %s",
                pdb_path.name, max_disp_ang, output_path.name,
            )
            return output_path

        except Exception as exc:
            logger.warning(
                "Minimization failed for %s (%s); using original pose",
                pdb_path.name, exc,
            )
            return pdb_path


def minimize_pose(pdb_path: Path, output_path: Path | None = None) -> Path:
    """Restrained clash-relief minimization of a single all-atom pose PDB.

    Adds H via pdbfixer, then runs LocalEnergyMinimizer with strong positional
    restraints on all heavy atoms.  If minimization moves any heavy atom beyond
    _MAX_DISPLACEMENT_ANG, the original pose is returned unchanged to preserve
    the binding conformation.

    For more than one pose use :func:`minimize_poses_batch`, which shares one
    OpenMM Context across the batch instead of building (and stranding) one per
    pose.

    Args:
        pdb_path: Absolute path to the raw pose PDB (all-atom, no H).
        output_path: Destination for the minimized PDB. Defaults to
            <pdb_path.parent>/<stem>_min.pdb.

    Returns:
        Path to the minimized heavy-atom PDB, or pdb_path if minimization
        failed or the displacement check rejected the result.  Never raises.
    """
    if output_path is None:
        output_path = pdb_path.parent / f"{pdb_path.stem}_min.pdb"
    session = _ClashRelief()
    try:
        return session.run(pdb_path, output_path)
    finally:
        session.close()


def minimize_poses_batch(pdb_paths: list[Path], output_dir: Path) -> list[Path]:
    """Minimize a batch of pose PDBs, writing results to output_dir.

    All poses share one OpenMM Context.  See :class:`_ClashRelief` for why that
    matters: a Context per pose leaked until the OOM killer intervened.

    Args:
        pdb_paths: Absolute paths to raw pose PDBs.
        output_dir: Directory for minimized outputs.

    Returns:
        List of result paths (minimized or original) in input order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Clash-relief minimization: %d poses -> %s", len(pdb_paths), output_dir)
    from hybridock_pep.output import progress as _progress  # noqa: PLC0415

    results: list[Path] = []
    n_total = len(pdb_paths)
    session = _ClashRelief()
    try:
        for done, pdb_path in enumerate(pdb_paths, 1):
            _progress.tick(done - 1, n_total, "poses minimized")
            results.append(session.run(pdb_path, output_dir / pdb_path.name))
    finally:
        session.close()
    _progress.tick(n_total, n_total, "poses minimized")
    _progress.clear()
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
