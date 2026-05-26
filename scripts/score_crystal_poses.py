"""Score PepSet crystal poses with Vina and AD4 for calibration.

For each complex in training_complexes.csv:
  1. Find crystal pose (_pep_ref.pdb) and pocket (_rec_unbound_pocket.pdb)
     from the PepSet directory.
  2. Prepare receptor PDBQT with ADFRsuite prepare_receptor.
  3. Prepare peptide PDBQT with babel -xr (rigid, Gasteiger charges).
  4. Compute grid centre from crystal peptide Cα centroid; box = 25 Å.
  5. Score with Vina (score_only).
  6. Score with AD4 (autogrid4 + vina --scoring ad4).
  7. Count contact residues (receptor residues with ≥1 heavy atom within 4.5 Å
     of any peptide heavy atom).
  8. Write --output JSON (same schema as training_scores_may01.json).

Usage:
    conda run --no-capture-output -n score-env python scripts/score_crystal_poses.py \\
        --training-csv data/training_complexes.csv \\
        --pepset-dir datasets/pepset \\
        --output data/training_scores.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from hybridock_pep.scoring.entropy import CONTACT_DIST_ANG  # Fix A: unified cutoff

_log = logging.getLogger(__name__)

_ADFR_BIN = Path("/home/igem/ADFRsuite_x86_64Linux_1.0/bin")
_BOX_MARGIN = 15.0     # Å — added to each side of peptide bounding box
_BOX_MIN = 20.0        # Å — minimum box size
_CONTACT_CUTOFF = CONTACT_DIST_ANG  # unified with entropy.py — do not set independently
_GRID_SPACING = 0.375  # Å — AutoDock standard

# Atom types for autogrid4 GPF (covers all amino acid atom types)
_RECEPTOR_TYPES = "C A N NA OA SA HD"
_LIGAND_TYPES = "C A N NA OA SA HD S NS F Cl Br I P"


# --------------------------------------------------------------------------- #
# PDB parsing helpers                                                           #
# --------------------------------------------------------------------------- #

def _parse_heavy_atoms(pdb_path: Path) -> list[tuple[str, str, int, float, float, float]]:
    """Return list of (record, atom_name, res_seq, x, y, z) for all heavy atoms."""
    atoms: list[tuple[str, str, int, float, float, float]] = []
    for line in pdb_path.read_text().splitlines():
        rec = line[:6].strip()
        if rec not in ("ATOM", "HETATM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name.startswith("H"):
            continue  # skip hydrogens
        try:
            res_seq = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        atoms.append((rec, atom_name, res_seq, x, y, z))
    return atoms


def _peptide_grid_params(
    pdb_path: Path,
    margin: float = _BOX_MARGIN,
    min_box: float = _BOX_MIN,
) -> tuple[tuple[float, float, float], float]:
    """Compute bounding-box centre and minimum cubic box size from peptide heavy atoms.

    Uses ALL heavy atoms (not just Cα) for the bounding box so the grid fully
    contains the peptide even for extended conformations.

    Args:
        pdb_path: Crystal peptide PDB.
        margin: Extra Å to add to each side of the bounding box.
        min_box: Minimum box size (Å).

    Returns:
        Tuple of ((cx, cy, cz), box_size).
    """
    coords: list[list[float]] = []
    for line in pdb_path.read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        name = line[12:16].strip()
        if name.startswith("H"):
            continue  # skip hydrogens
        try:
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
    if not coords:
        raise ValueError(f"No heavy atoms found in {pdb_path}")
    arr = np.array(coords)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    center = ((mins + maxs) / 2.0)
    box_size = float(max((maxs - mins).max() + margin, min_box))
    return (float(center[0]), float(center[1]), float(center[2])), box_size


def _count_contact_residues(
    rec_pdb: Path,
    pep_pdb: Path,
    cutoff: float = _CONTACT_CUTOFF,
) -> int:
    """Count unique PEPTIDE residues with ≥1 heavy atom within *cutoff* Å of any receptor heavy atom.

    Mirrors entropy.count_contact_residues() semantics exactly — the count returned
    here is passed as n_contact_residues to calibrate_alpha.py, which uses it in
    the same gamma formula as production scoring.
    """
    rec_atoms = _parse_heavy_atoms(rec_pdb)
    pep_atoms = _parse_heavy_atoms(pep_pdb)

    if not rec_atoms or not pep_atoms:
        return 0

    rec_arr = np.array([[a[3], a[4], a[5]] for a in rec_atoms])

    # Group peptide atoms by residue sequence number
    pep_by_res: dict[int, list[list[float]]] = {}
    for _, _, res_seq, px, py, pz in pep_atoms:
        pep_by_res.setdefault(res_seq, []).append([px, py, pz])

    contact_residues: set[int] = set()
    for res_seq, pep_coords in pep_by_res.items():
        pep_arr = np.array(pep_coords)
        # Check if any atom from this peptide residue is within cutoff of any receptor atom
        diffs = pep_arr[:, np.newaxis, :] - rec_arr[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=-1))  # (n_pep_atoms, n_rec_atoms)
        if dists.min() <= cutoff:
            contact_residues.add(res_seq)

    return len(contact_residues)


# --------------------------------------------------------------------------- #
# Preparation helpers                                                            #
# --------------------------------------------------------------------------- #

def _prepare_receptor_pdbqt(rec_pdb: Path, out_dir: Path) -> Path:
    """Convert receptor PDB to PDBQT with prepare_receptor.

    Runs prepare_receptor from ADFRsuite with -A hydrogens to add polar H.
    """
    pdbqt = out_dir / "receptor.pdbqt"
    prepare_receptor = str(_ADFR_BIN / "prepare_receptor")
    cmd = [prepare_receptor, "-r", str(rec_pdb), "-o", str(pdbqt), "-A", "hydrogens"]
    _log.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not pdbqt.exists() or pdbqt.stat().st_size == 0:
        raise RuntimeError(
            f"prepare_receptor failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return pdbqt


def _prepare_ligand_pdbqt(pep_pdb: Path, out_dir: Path) -> Path:
    """Convert peptide PDB to PDBQT with babel (rigid, Gasteiger charges)."""
    pdbqt = out_dir / "peptide.pdbqt"
    babel = str(_ADFR_BIN / "babel")
    cmd = [babel, "-i", "pdb", str(pep_pdb), "-o", "pdbqt", str(pdbqt), "-h", "-xr"]
    _log.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not pdbqt.exists() or pdbqt.stat().st_size == 0:
        raise RuntimeError(
            f"babel failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    # Wrap in ROOT/ENDROOT for Vina 1.2.x Python API (requires torsion tree)
    _wrap_rigid_pdbqt(pdbqt)
    return pdbqt


def _wrap_rigid_pdbqt(pdbqt: Path) -> None:
    """Wrap flat PDBQT atoms in ROOT/ENDROOT/TORSDOF 0 for Vina 1.2.x."""
    lines = pdbqt.read_text().splitlines(keepends=True)
    remarks = [l for l in lines if l.startswith("REMARK")]
    atoms = [l for l in lines if l.startswith("ATOM") or l.startswith("HETATM")]
    if not atoms:
        raise ValueError(f"No ATOM/HETATM records found in {pdbqt}")
    wrapped = "".join(remarks) + "ROOT\n" + "".join(atoms) + "ENDROOT\nTORSDOF 0\n"
    pdbqt.write_text(wrapped)


# --------------------------------------------------------------------------- #
# AD4 grid map generation                                                       #
# --------------------------------------------------------------------------- #

def _get_receptor_atom_types(rec_pdbqt: Path) -> str:
    """Extract unique atom type strings from PDBQT atom type column (col 77+)."""
    types: set[str] = set()
    for line in rec_pdbqt.read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) > 77:
            parts = line[77:].split()
            if parts:
                types.add(parts[0])
    return " ".join(sorted(types)) if types else _RECEPTOR_TYPES


def _find_ad4_parameters_dat() -> str:
    """Return absolute path to AD4_parameters.dat from ADFRsuite."""
    params = _ADFR_BIN.parent / "CCSBpckgs" / "AutoDockTools" / "AD4_parameters.dat"
    if not params.exists():
        raise FileNotFoundError(
            f"AD4_parameters.dat not found at {params}. Check ADFRsuite installation."
        )
    return str(params)


def _generate_ad4_maps(
    rec_pdbqt: Path,
    maps_dir: Path,
    center: tuple[float, float, float],
    box_size: float,
) -> Path:
    """Generate AD4 affinity maps with autogrid4. Returns maps_dir."""
    maps_dir.mkdir(parents=True, exist_ok=True)

    # Copy receptor into maps/ (autogrid4 resolves relative paths from cwd)
    rec_in_maps = maps_dir / "receptor.pdbqt"
    shutil.copy2(rec_pdbqt, rec_in_maps)

    npts = int(box_size / _GRID_SPACING)
    cx, cy, cz = center
    receptor_types_str = _get_receptor_atom_types(rec_pdbqt)

    map_lines = [f"map receptor.{t}.map" for t in _LIGAND_TYPES.split()]
    gpf_lines = [
        f"npts {npts} {npts} {npts}",
        f"parameter_file {_find_ad4_parameters_dat()}",
        "gridfld receptor.maps.fld",
        f"spacing {_GRID_SPACING}",
        f"receptor_types {receptor_types_str}",
        f"ligand_types {_LIGAND_TYPES}",
        "receptor receptor.pdbqt",
        f"gridcenter {cx:.3f} {cy:.3f} {cz:.3f}",
        *map_lines,
        "elecmap receptor.e.map",
        "dsolvmap receptor.d.map",
        "dielectric -0.1465",
    ]
    gpf_path = maps_dir / "receptor.gpf"
    gpf_path.write_text("\n".join(gpf_lines) + "\n")

    autogrid4 = str(_ADFR_BIN / "autogrid4")
    cmd = [autogrid4, "-p", "receptor.gpf", "-l", "receptor.glg"]
    _log.debug("Running autogrid4 (cwd=%s)", maps_dir)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(maps_dir), timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"autogrid4 failed (exit {result.returncode}): {result.stderr[:300]}")

    hd_map = maps_dir / "receptor.HD.map"
    if not hd_map.exists():
        raise RuntimeError("receptor.HD.map missing after autogrid4 — check GPF atom types")

    return maps_dir


# --------------------------------------------------------------------------- #
# Scoring functions                                                              #
# --------------------------------------------------------------------------- #

def _score_vina(
    rec_pdbqt: Path,
    pep_pdbqt: Path,
    center: tuple[float, float, float],
    box_size: float,
) -> float:
    """Score crystal pose with Vina (score_only). Returns kcal/mol."""
    from vina import Vina  # noqa: PLC0415 — import inside fn to allow sys.path from score-env
    v = Vina(sf_name="vina", verbosity=0)
    v.set_receptor(str(rec_pdbqt))
    v.compute_vina_maps(center=list(center), box_size=[box_size] * 3)
    v.set_ligand_from_file(str(pep_pdbqt))
    return float(v.score()[0])


def _score_ad4(pep_pdbqt: Path, maps_dir: Path) -> float:
    """Score crystal pose with AD4 using pre-computed maps. Returns kcal/mol."""
    from vina import Vina  # noqa: PLC0415
    v = Vina(sf_name="ad4", verbosity=0)
    v.load_maps(str(maps_dir / "receptor"))
    v.set_ligand_from_file(str(pep_pdbqt))
    return float(v.score()[0])


# --------------------------------------------------------------------------- #
# Main workflow                                                                  #
# --------------------------------------------------------------------------- #

def score_entry(
    pdb_id: str,
    pepset_dir: Path,
    work_dir: Path,
) -> dict[str, float | int]:
    """Score one complex. Returns {"vina_score", "ad4_score", "n_contact_residues"}.

    Uses the HOLO receptor (_rec_ref.pdb) paired with the crystal peptide
    (_pep_ref.pdb) so both structures are in the same coordinate frame.
    Box size is derived from the peptide bounding box so extended peptides
    (>20 residues) are fully contained.
    """
    entry_dir = pepset_dir / pdb_id
    rec_pdb = entry_dir / f"{pdb_id}_rec_ref.pdb"           # holo receptor
    pep_pdb = entry_dir / f"{pdb_id}_pep_ref.pdb"           # crystal peptide
    rec_pocket = entry_dir / f"{pdb_id}_rec_unbound_pocket.pdb"  # for contact count

    for path in (rec_pdb, pep_pdb):
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    work = work_dir / pdb_id
    work.mkdir(parents=True, exist_ok=True)

    # Step 1: prepare receptor
    _log.info("[%s] Preparing receptor PDBQT...", pdb_id)
    rec_pdbqt = _prepare_receptor_pdbqt(rec_pdb, work)

    # Step 2: prepare peptide
    _log.info("[%s] Preparing peptide PDBQT...", pdb_id)
    pep_pdbqt = _prepare_ligand_pdbqt(pep_pdb, work)

    # Step 3: grid centre (bounding-box centre) and dynamic box size
    center, box_size = _peptide_grid_params(pep_pdb)
    _log.info("[%s] Grid centre: (%.2f, %.2f, %.2f)  box=%.1f Å", pdb_id, *center, box_size)

    # Contact residues: use apo pocket vs crystal peptide if pocket available,
    # else fall back to holo receptor
    contact_ref = rec_pocket if rec_pocket.exists() else rec_pdb
    n_contact = _count_contact_residues(contact_ref, pep_pdb)
    _log.info("[%s] Contact residues: %d", pdb_id, n_contact)

    # Step 4: Vina score_only
    _log.info("[%s] Scoring with Vina...", pdb_id)
    vina_score = _score_vina(rec_pdbqt, pep_pdbqt, center, box_size)
    _log.info("[%s] Vina score: %.3f kcal/mol", pdb_id, vina_score)

    # Step 5: generate AD4 maps
    _log.info("[%s] Generating AD4 maps (autogrid4)...", pdb_id)
    maps_dir = _generate_ad4_maps(rec_pdbqt, work / "maps", center, box_size)

    # Step 6: AD4 score
    _log.info("[%s] Scoring with AD4...", pdb_id)
    ad4_score = _score_ad4(pep_pdbqt, maps_dir)
    _log.info("[%s] AD4 score: %.3f kcal/mol", pdb_id, ad4_score)

    return {
        "vina_score": round(vina_score, 3),
        "ad4_score": round(ad4_score, 3),
        "n_contact_residues": n_contact,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-csv",
        type=Path,
        default=Path("data/training_complexes.csv"),
        help="Training CSV (pdb_id, peptide_sequence, experimental_pkd). Default: data/training_complexes.csv",
    )
    parser.add_argument(
        "--pepset-dir",
        type=Path,
        default=Path("datasets/pepset"),
        help="PepSet directory containing {pdb_id}/ subdirectories. Default: datasets/pepset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/training_scores.json"),
        help="Output scores JSON. Default: data/training_scores.json",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("runs/calibration_scoring"),
        help="Working directory for PDBQT and grid files. Default: runs/calibration_scoring",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args.work_dir.mkdir(parents=True, exist_ok=True)

    # Read training CSV
    with args.training_csv.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    _log.info("Scoring %d complexes from %s", len(rows), args.training_csv)

    scores: dict[str, dict] = {}
    failed: list[str] = []

    for row in rows:
        pdb_id = row["pdb_id"].strip()
        try:
            entry_scores = score_entry(pdb_id, args.pepset_dir.resolve(), args.work_dir.resolve())
            scores[pdb_id] = entry_scores
            _log.info(
                "[%s] Done: vina=%.3f  ad4=%.3f  contacts=%d",
                pdb_id,
                entry_scores["vina_score"],
                entry_scores["ad4_score"],
                entry_scores["n_contact_residues"],
            )
        except Exception as exc:  # noqa: BLE001 — skip failed entries, continue
            _log.error("[%s] FAILED: %s: %s", pdb_id, type(exc).__name__, exc)
            failed.append(pdb_id)

    # Write scores JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2))
    _log.info("Scores written to %s", args.output)
    _log.info("Succeeded: %d  Failed: %d", len(scores), len(failed))
    if failed:
        _log.warning("Failed entries: %s", failed)
    if len(scores) < 4:
        _log.error("Too few successful scores (%d) for calibration — need ≥4", len(scores))


if __name__ == "__main__":
    main()
