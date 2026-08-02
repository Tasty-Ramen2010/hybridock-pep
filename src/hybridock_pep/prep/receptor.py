from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from openmm.app import PDBFile
from pdbfixer import PDBFixer

from hybridock_pep.models import DockConfig
from hybridock_pep.prep.errors import PrepError
from hybridock_pep.prep.pdbqt_convert import convert_pdb_to_pdbqt

logger = logging.getLogger(__name__)


def prepare_receptor(config: DockConfig) -> Path:
    """Clean a receptor PDB with pdbfixer and convert it to PDBQT.

    Always regenerates the PDBQT — no caching, no mtime checks.
    pdbfixer steps run unconditionally:
      1. Strip non-water HETATM and alternate-occupancy atoms (keep alt ' ' or 'A').
      2. Find and add missing residues.
      3. Find and add missing atoms.
      4. Add hydrogens at pH 7.4.

    Backend order: ADFRsuite's prepare_receptor if on PATH, else meeko's
    mk_prepare_receptor.py, else babel/obabel (see prep/pdbqt_convert.py).
    ADFRsuite is NOT required — it ships only an x86_64 tarball, so it is
    absent on Apple Silicon by default. AD4 scoring does not require it
    either: conda-forge autogrid provides autogrid4 natively.

    The backend choice does not move the reported ΔG, which is worth stating
    because it is not obvious. Measured on 1YCR (705 receptor heavy atoms),
    meeko vs obabel:

      * Vina ignores the receptor PDBQT charge column entirely — scores are
        bit-identical with every partial charge zeroed, or sign-flipped and
        tripled. So Gasteiger-assignment differences between backends cannot
        matter on the default --scoring vina path.
      * Vina does read the atom-type column (retyping every heavy atom to C
        moves the score 1.01 kcal/mol), but the types the backends actually
        disagree on are all in Vina's neutral set: N↔NA, A↔C and S↔SA are
        each exactly score-neutral. Only O↔OA moves a Vina score, and the
        backends agree on every oxygen. Net: 22/705 typing disagreements,
        9 of them in the binding interface, produce a 0.000000 kcal/mol
        difference.
      * AD4 is the one backend that does consume charges, and it is off by
        default (the production ridge gives w_ad4=0).

    See tests/test_receptor_prep_fidelity.py, which pins all of the above.

    If receptor PDBQT generation fails, raises PrepError immediately with the
    full stderr captured. No retry.

    Args:
        config: Validated DockConfig. Uses receptor_path and output_dir.

    Returns:
        Path to the written receptor PDBQT (output_dir/receptor.pdbqt).

    Raises:
        PrepError: If PDBQT generation exits non-zero.
        FileNotFoundError: If neither prepare_receptor nor babel/obabel is on PATH.
    """
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pdbqt_path = output_dir / "receptor.pdbqt"

    # --- Step 1: Pre-filter PDB (strip altLoc B/C/... and non-water HETATM) ---
    cleaned_pdb_lines = _filter_pdb_lines(config.receptor_path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
        tmp.writelines(cleaned_pdb_lines)
        cleaned_pdb_path = Path(tmp.name)

    try:
        # --- Step 2: pdbfixer — all three fixes, unconditionally ---
        fixer = PDBFixer(filename=str(cleaned_pdb_path))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        try:
            fixer.addMissingHydrogens(7.4)
        except ValueError as exc:
            # Some HIS residues in raw RCSB downloads have non-standard atom sets
            # that confuse pdbfixer's protonation state detection. Retry with all
            # HIS residues forced to HIE (epsilon-protonated, the most common form).
            logger.warning(
                "addMissingHydrogens failed (%s); retrying with HIS→HIE override", exc
            )
            try:
                # Build a variants list: force every HIS to "HIE"; others get None
                variants = [
                    "HIE"
                    if r.name in ("HIS", "HID", "HIP", "HSE", "HSP", "HSD")
                    else None
                    for r in fixer.topology.residues()
                ]
                fixer.addMissingHydrogens(7.4, variants=variants)
            except Exception as exc2:
                logger.warning(
                    "addMissingHydrogens with HIS override also failed (%s); "
                    "skipping H addition — prepare_receptor will handle it",
                    exc2,
                )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pdb", delete=False
        ) as fixed_tmp:
            PDBFile.writeFile(fixer.topology, fixer.positions, fixed_tmp)
            fixed_pdb_path = Path(fixed_tmp.name)
    finally:
        cleaned_pdb_path.unlink(missing_ok=True)

    # --- Step 3: prepare_receptor (ADFRsuite), or babel/obabel fallback ---
    # -A hydrogens: force H addition even if pdbfixer already added them (idempotent);
    # guards against cases where pdbfixer's addMissingHydrogens fails (e.g. HIS edge cases).
    prepare_receptor_bin = shutil.which("prepare_receptor")
    try:
        if prepare_receptor_bin is not None:
            cmd = [
                "prepare_receptor",
                "-r", str(fixed_pdb_path),
                "-o", str(pdbqt_path),
                "-A", "hydrogens",
            ]
            logger.info("Running: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise PrepError(
                    f"prepare_receptor failed (exit {result.returncode}):\n{result.stderr}"
                )
        elif shutil.which("mk_prepare_receptor.py") is not None:
            # Meeko is the Forli lab's supported successor to prepare_receptor,
            # pip/conda-installable and native on Apple Silicon. It emits proper
            # AutoDock atom types (OA/HD/NA/SA) and Gasteiger charges, which
            # obabel does less faithfully, so it is preferred over the babel
            # fallback below.
            logger.info(
                "prepare_receptor (ADFRsuite) not on PATH — using meeko "
                "mk_prepare_receptor.py instead."
            )
            cmd = [
                "mk_prepare_receptor.py",
                "--read_pdb", str(fixed_pdb_path),
                "-o", str(pdbqt_path.with_suffix("")),
                "-p",
            ]
            logger.info("Running: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)
            # Post-run success check, NOT a skip-if-exists cache (D-02): meeko
            # can exit 0 having written nothing when the input is unusable, so
            # the output is verified after the subprocess, never before it.
            wrote_output = Path(pdbqt_path).is_file()
            if result.returncode != 0 or not wrote_output:
                raise PrepError(
                    f"mk_prepare_receptor.py failed (exit {result.returncode}):\n"
                    f"{result.stderr or result.stdout}"
                )
        else:
            logger.warning(
                "Neither prepare_receptor (ADFRsuite) nor meeko "
                "mk_prepare_receptor.py on PATH — falling back to babel/obabel "
                "for receptor PDBQT prep."
            )
            try:
                result = convert_pdb_to_pdbqt(fixed_pdb_path, pdbqt_path, add_hydrogens=True)
            except FileNotFoundError as exc:
                raise PrepError(
                    "No receptor preparation tool found. Install any one of: "
                    "meeko (`pip install meeko`, provides mk_prepare_receptor.py — "
                    "recommended, no license click-through), openbabel "
                    "(`conda install -c conda-forge openbabel`), or ADFRsuite. "
                    "For AD4 map generation also install autogrid "
                    "(`conda install -c conda-forge autogrid`)."
                ) from exc
            try:
                pdbqt_size = pdbqt_path.stat().st_size
            except FileNotFoundError:
                pdbqt_size = 0
            if result.returncode != 0 or pdbqt_size == 0:
                raise PrepError(
                    f"babel/obabel failed to produce receptor PDBQT "
                    f"(exit {result.returncode}):\n{result.stderr}"
                )
    finally:
        fixed_pdb_path.unlink(missing_ok=True)

    logger.info("Receptor PDBQT written: %s", pdbqt_path)
    return pdbqt_path


def prepare_receptor_pdb(config: DockConfig) -> Path:
    """Clean receptor PDB with pdbfixer and save as PDB (not PDBQT) for RAPiDock.

    Applies the same filter+pdbfixer steps as prepare_receptor but writes a PDB
    file rather than running ADFRsuite. This gives RAPiDock a clean, continuous-
    chain PDB so MDAnalysis and BioPython agree on chain count (avoiding the
    IndexError that occurs when a raw RCSB download has discontinuous chain
    segments that MDAnalysis splits into more segments than BioPython chains).

    Args:
        config: Validated DockConfig. Uses receptor_path and output_dir.

    Returns:
        Path to the written receptor PDB (output_dir/receptor_for_rapidock.pdb).
    """
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdb = output_dir / "receptor_for_rapidock.pdb"

    # Strip ALL HETATM (including water) — RAPiDock only needs protein atoms,
    # and pdbfixer assigns water to extra chains (C, D, …) that confuse its
    # MDAnalysis chain iteration vs. BioPython ESM embedding count.
    protein_only_lines = [
        line for line in _filter_pdb_lines(config.receptor_path)
        if line[:6].strip() not in ("HETATM",)
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
        tmp.writelines(protein_only_lines)
        cleaned_pdb_path = Path(tmp.name)

    try:
        fixer = PDBFixer(filename=str(cleaned_pdb_path))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        # Do NOT add hydrogens for RAPiDock — RAPiDock uses heavy atoms only,
        # and addMissingHydrogens fails on HIS residues with non-standard atom
        # sets that are common in raw RCSB downloads.
        fixed_path = output_dir / "receptor_for_rapidock_full.pdb"
        with open(fixed_path, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
    finally:
        cleaned_pdb_path.unlink(missing_ok=True)

    # Pocket crop: RAPiDock's local-docking model expects a pocket-sized receptor,
    # not the full protein. Without this crop the diffusion sampler ignores
    # site_coords and scatters poses across the whole surface, which then
    # silently blows the Vina grid bounds on extended/groove targets.
    # Radius: half the Vina box edge + 5 Å margin, with a minimum of 12 Å
    # (matches the prior PepSet-6 manual pocket files from May 27).
    radius = max(12.0, config.box_size / 2.0 + 5.0)
    n_residues = crop_to_pocket(
        pdb_path=fixed_path,
        site_coords=config.site_coords,
        radius=radius,
        output_path=output_pdb,
    )
    if n_residues < 10:
        logger.warning(
            "Pocket crop kept only %d residues — site_coords may be off, or "
            "box_size too small. RAPiDock may struggle on this target.",
            n_residues,
        )
    return output_pdb


def crop_to_pocket(
    pdb_path: Path,
    site_coords: tuple[float, float, float],
    radius: float,
    output_path: Path,
) -> int:
    """Write a residue-level pocket crop of a receptor PDB.

    Keeps every residue with at least one heavy atom within ``radius`` Å of
    ``site_coords``. Residues are kept atomically intact (all atoms of a
    qualifying residue are written, including atoms outside the sphere) so
    that residue connectivity stays valid for downstream tools.

    Why this exists: RAPiDock's ``rapidock_local.pt`` checkpoint is a *local*
    docking model — it expects a pocket-sized receptor, not the full protein.
    Passing the full protein causes RAPiDock to sample poses across the
    entire surface rather than at the intended binding site, since the
    diffusion model has no explicit site/box input.

    Args:
        pdb_path: Source receptor PDB (already HETATM/altloc-cleaned).
        site_coords: (x, y, z) binding-site center in Å.
        radius: Inclusion radius. A residue is kept if ANY heavy atom is
            within this distance of site_coords.
        output_path: Destination PDB path.

    Returns:
        Number of residues kept.
    """
    sx, sy, sz = site_coords
    r2 = radius * radius

    # Pass 1: identify qualifying residues by (chain, resseq, icode)
    keep: set[tuple[str, int, str]] = set()
    for line in pdb_path.read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        atom_name = line[12:16].strip()
        if atom_name.startswith("H"):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            res_seq = int(line[22:26].strip())
        except ValueError:
            continue
        d2 = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2
        if d2 <= r2:
            keep.add((line[21], res_seq, line[26]))

    # Pass 2: write all atom lines belonging to qualifying residues
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept_lines: list[str] = ["REMARK   Pocket crop: radius={:.1f} Å around ({:.2f}, {:.2f}, {:.2f})\n".format(
        radius, sx, sy, sz)]
    for line in pdb_path.read_text().splitlines(keepends=True):
        record = line[:6].strip()
        if record not in ("ATOM", "HETATM"):
            if record in ("TER", "END"):
                kept_lines.append(line)
            continue
        try:
            res_seq = int(line[22:26].strip())
        except ValueError:
            continue
        key = (line[21], res_seq, line[26])
        if key in keep:
            kept_lines.append(line)
    if not kept_lines[-1].startswith("END"):
        kept_lines.append("END\n")
    output_path.write_text("".join(kept_lines))
    logger.info(
        "Pocket crop: kept %d residues within %.1f Å of (%.2f, %.2f, %.2f) → %s",
        len(keep), radius, sx, sy, sz, output_path,
    )
    return len(keep)


_PRESERVE_HETATM_RESNAMES: frozenset[str] = frozenset({"TPO", "SEP", "PTR", "HOH", "WAT"})


def _filter_pdb_lines(pdb_path: Path) -> list[str]:
    """Strip alternate-occupancy atoms and non-water HETATM from PDB text.

    Keeps ATOM records, water HETATM (resName HOH or WAT), and phospho-residue
    HETATM (TPO/SEP/PTR — some older PDB entries store these as HETATM rather
    than ATOM). All other HETATM are dropped.

    Args:
        pdb_path: Path to the input PDB file.

    Returns:
        List of filtered PDB lines, each ending with newline.
    """
    kept: list[str] = []
    for line in pdb_path.read_text().splitlines(keepends=True):
        record = line[:6].strip()
        if record == "HETATM":
            res_name = line[17:20].strip()
            if res_name not in _PRESERVE_HETATM_RESNAMES:
                continue  # drop non-water, non-phospho HETATM
        if record in ("ATOM", "HETATM"):
            alt_loc = line[16] if len(line) > 16 else " "
            if alt_loc not in (" ", "A"):
                continue  # drop alternate occupancy B/C/...
        kept.append(line)
    return kept
