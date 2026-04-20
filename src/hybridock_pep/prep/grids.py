from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from hybridock_pep.models import DockConfig
from hybridock_pep.prep.errors import PrepError

logger = logging.getLogger(__name__)

# AD4 atom types (D-06): covers standard peptide atoms + cysteine sulfur (S) +
# polar hydrogen (HD) — HD type is required for receptor.HD.map to be generated.
_RECEPTOR_TYPES = "C A N O SA S H HD"
_LIGAND_TYPES = "C A N O S H HD"
_GRID_SPACING = 0.375  # Angstrom — AutoDock standard; spec does not override


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

    gpf_content = _build_gpf(config, maps_dir)
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


def _build_gpf(config: DockConfig, maps_dir: Path) -> str:
    """Construct the autogrid4 Grid Parameter File content from DockConfig.

    The GPF references the receptor by filename only (receptor.pdbqt) because
    autogrid4 runs with cwd=maps_dir. Full paths break autogrid4's internal
    file handling.

    Grid points per dimension: npts = int(config.box_size / _GRID_SPACING).
    The box is cubic (same npts for all three dimensions).

    Args:
        config: Validated DockConfig. Uses site_coords and box_size.
        maps_dir: The maps output directory (not used in GPF text directly —
            included for signature consistency with generate_ad4_maps caller).

    Returns:
        GPF file content as a string, ready to write to receptor.gpf.
    """
    npts = int(config.box_size / _GRID_SPACING)
    cx, cy, cz = config.site_coords

    lines = [
        f"npts {npts} {npts} {npts}",
        "parameter_file AD4_parameters.dat",
        "gridfld receptor.maps.fld",
        f"spacing {_GRID_SPACING}",
        f"receptor_types {_RECEPTOR_TYPES}",
        f"ligand_types {_LIGAND_TYPES}",
        "receptor receptor.pdbqt",
        f"gridcenter {cx} {cy} {cz}",
        "nbp_coeffs 12 6 -0.00162 3.86528 -0.00662 3.82836 "
        "-0.1 4.0 -0.25 3.5 -0.0015 3.5 -0.00001 6.0",
        "map receptor.C.map",
        "map receptor.A.map",
        "map receptor.N.map",
        "map receptor.O.map",
        "map receptor.S.map",
        "map receptor.H.map",
        "map receptor.HD.map",
        "elecmap receptor.e.map",
        "dsolvmap receptor.d.map",
        "dielectric -0.1465",
    ]
    return "\n".join(lines) + "\n"
