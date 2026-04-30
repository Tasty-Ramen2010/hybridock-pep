from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from hybridock_pep.models import DockConfig
from hybridock_pep.prep.errors import PrepError

logger = logging.getLogger(__name__)

# Fallback receptor_types when none can be parsed from the PDBQT.
_RECEPTOR_TYPES = "C A N NA OA SA HD"
# ligand_types controls which map files autogrid4 generates.
# Must cover all atom types babel can assign to peptide PDBQTs:
#   S  = non-H-bonding sulfur (MET side chain)
#   NS = aromatic nitrogen (TRP N1, HIS ND1/NE2)
#   F/Cl/Br/I/P = halogens and phosphorus for non-natural modifications
# Missing types → zero-coverage grid cells → positive (unphysical) AD4 scores.
_LIGAND_TYPES = "C A N NA OA SA HD S NS F Cl Br I P"
_GRID_SPACING = 0.375  # Angstrom — AutoDock standard; spec does not override


def _get_pdbqt_atom_types(pdbqt_path: Path) -> list[str]:
    """Extract unique atom type strings from all ATOM/HETATM records in a PDBQT file.

    PDBQT files place the atom type token at column 78+ (0-indexed: 77+).
    Returns a sorted list of unique types found, or an empty list if the file
    has no typed atoms (caller should fall back to _RECEPTOR_TYPES).

    Args:
        pdbqt_path: Path to a PDBQT file.

    Returns:
        Sorted list of unique atom type strings (e.g. ["A", "C", "HD", "OA"]).
    """
    types: set[str] = set()
    for line in pdbqt_path.read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) > 77:
            parts = line[77:].split()
            if parts:
                types.add(parts[0])
    return sorted(types)


def _find_ad4_parameters_dat() -> str:
    """Return absolute path to AD4_parameters.dat from the ADFRsuite install.

    autogrid4 segfaults when given a relative path for parameter_file — it
    fails to locate the file and crashes on a null pointer. Deriving the path
    from the prepare_receptor binary location guarantees portability across
    ADFRsuite install locations.

    Returns:
        Absolute path string to AD4_parameters.dat.

    Raises:
        FileNotFoundError: If prepare_receptor or AD4_parameters.dat not found.
    """
    prepare_receptor_bin = shutil.which("prepare_receptor")
    if prepare_receptor_bin is None:
        raise FileNotFoundError(
            "prepare_receptor not on PATH — cannot locate AD4_parameters.dat. "
            "Install ADFRsuite and add its bin/ to PATH."
        )
    # prepare_receptor lives in {ADFR_ROOT}/bin/; AD4_parameters.dat is at
    # {ADFR_ROOT}/CCSBpckgs/AutoDockTools/AD4_parameters.dat
    adfr_root = Path(prepare_receptor_bin).resolve().parent.parent
    params_path = adfr_root / "CCSBpckgs" / "AutoDockTools" / "AD4_parameters.dat"
    if not params_path.exists():
        raise FileNotFoundError(
            f"AD4_parameters.dat not found at {params_path}. "
            "Check your ADFRsuite installation."
        )
    return str(params_path)


def generate_ad4_maps(config: DockConfig, receptor_pdbqt: Path) -> Path:
    """Generate AutoDock4 affinity maps via autogrid4 for AD4 scoring.

    Builds a Grid Parameter File (GPF) programmatically from DockConfig fields,
    writes it to output_dir/maps/receptor.gpf, runs autogrid4, then checks that
    receptor.HD.map was produced. If HD.map is absent, raises PrepError immediately
    with a diagnostic message — this is a hard abort (D-05).

    All autogrid4 outputs (.map, .gpf, .glg) go to output_dir/maps/ (D-07).

    Args:
        config: Validated DockConfig. Uses site_coords, box_size, output_dir.
        receptor_pdbqt: Path to the prepared receptor PDBQT
            (written by prep/receptor.py to output_dir/receptor.pdbqt).

    Returns:
        Path to the maps directory (output_dir/maps/) after successful generation.

    Raises:
        PrepError: If autogrid4 exits non-zero or receptor.HD.map is missing.
        FileNotFoundError: If autogrid4 is not on PATH.
    """
    maps_dir = config.output_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    # Copy receptor PDBQT into maps/ so autogrid4 can reference it by filename.
    # autogrid4 runs with cwd=maps_dir, so relative paths inside the GPF work.
    receptor_in_maps = maps_dir / "receptor.pdbqt"
    receptor_in_maps.write_bytes(receptor_pdbqt.read_bytes())

    gpf_content = _build_gpf(config, maps_dir, receptor_pdbqt)
    gpf_path = maps_dir / "receptor.gpf"
    gpf_path.write_text(gpf_content)
    logger.info("GPF written: %s", gpf_path)

    cmd = ["autogrid4", "-p", "receptor.gpf", "-l", "receptor.glg"]
    logger.info("Running: %s (cwd=%s)", " ".join(cmd), maps_dir)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(maps_dir),
        timeout=120,
    )

    if result.returncode != 0:
        raise PrepError(
            f"autogrid4 failed (exit {result.returncode}):\n{result.stderr}"
        )

    # D-05: hard abort if HD map absent — vina --scoring ad4 will silently fail without it
    hd_map = maps_dir / "receptor.HD.map"
    if not hd_map.exists():
        raise PrepError(
            "receptor.HD.map not found after autogrid4 — AD4 scoring will fail. "
            "Check your atom types in the GPF."
        )

    logger.info("AD4 maps generated in: %s", maps_dir)
    return maps_dir


def _build_gpf(config: DockConfig, maps_dir: Path, receptor_pdbqt: Path) -> str:
    """Construct the autogrid4 Grid Parameter File content from DockConfig.

    The GPF references the receptor by filename only (receptor.pdbqt) because
    autogrid4 runs with cwd=maps_dir. Full paths break autogrid4's internal
    file handling.

    Grid points per dimension: npts = int(config.box_size / _GRID_SPACING).
    The box is cubic (same npts for all three dimensions).

    receptor_types is derived dynamically from the actual PDBQT atom type
    column — prevents autogrid4 "Unknown receptor type" errors when the
    receptor contains metals, halogens, or other non-standard atoms.

    Args:
        config: Validated DockConfig. Uses site_coords and box_size.
        maps_dir: The maps output directory (not used in GPF text directly —
            included for signature consistency with generate_ad4_maps caller).
        receptor_pdbqt: Path to receptor.pdbqt — used to extract atom types.

    Returns:
        GPF file content as a string, ready to write to receptor.gpf.
    """
    npts = int(config.box_size / _GRID_SPACING)
    cx, cy, cz = config.site_coords

    receptor_type_list = _get_pdbqt_atom_types(receptor_pdbqt)
    if receptor_type_list:
        receptor_types_str = " ".join(receptor_type_list)
    else:
        logger.warning(
            "No atom types found in %s; falling back to default receptor_types", receptor_pdbqt
        )
        receptor_types_str = _RECEPTOR_TYPES

    map_lines = [f"map receptor.{t}.map" for t in _LIGAND_TYPES.split()]

    lines = [
        f"npts {npts} {npts} {npts}",
        f"parameter_file {_find_ad4_parameters_dat()}",
        "gridfld receptor.maps.fld",
        f"spacing {_GRID_SPACING}",
        f"receptor_types {receptor_types_str}",
        f"ligand_types {_LIGAND_TYPES}",
        "receptor receptor.pdbqt",
        f"gridcenter {cx} {cy} {cz}",
        *map_lines,
        "elecmap receptor.e.map",
        "dsolvmap receptor.d.map",
        "dielectric -0.1465",
    ]
    return "\n".join(lines) + "\n"
