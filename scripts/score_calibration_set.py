"""Score all 284 calibration entries from training_complexes_full.csv with Vina + AD4.

This script is the key workhorse for Tier 1.3 production calibration on the Linux RTX machine.
It:
  1. Reads data/training_complexes_full.csv (284 entries with pdb_id, peptide_sequence,
     experimental_pkd, receptor_chain)
  2. Finds each structure on disk (from any dataset directory)
  3. Splits structure into receptor.pdb + peptide.pdb by chain ID
  4. Prepares PDBQT with ADFRsuite (receptor) and babel (peptide)
  5. Scores with AutoDock Vina (score_only) and AD4 (autogrid4 + vina --scoring ad4)
  6. Counts contact residues (heavy atoms within 4.5 Å)
  7. Appends results to --output-csv (checkpoint-safe: skips already-scored entries)
  8. When all done, writes --output-json in training_scores.json format

Designed for crystal-pose calibration (both chains from same deposited structure).
For production-pose calibration (apo receptor + docked pose), use run_production_calibration.sh.

Usage (Linux RTX machine, score-env):
    conda run -n score-env python scripts/score_calibration_set.py \\
        --training-csv data/training_complexes_full.csv \\
        --output-csv runs/calibration_full/scores.csv \\
        --output-json data/training_scores_full.json \\
        --workers 8 \\
        --verbose

    # Resume after interruption (skips already-scored entries):
    conda run -n score-env python scripts/score_calibration_set.py \\
        --training-csv data/training_complexes_full.csv \\
        --output-csv runs/calibration_full/scores.csv \\
        --output-json data/training_scores_full.json

    # Score only Kd/Ki entries (highest quality):
    conda run -n score-env python scripts/score_calibration_set.py \\
        --training-csv data/training_complexes_full.csv \\
        --output-csv runs/calibration_kd_only/scores.csv \\
        --output-json data/training_scores_kd_only.json \\
        --affinity-types Kd Ki

Expected time:
    ~1-3 min per complex on Linux CPU (mostly AutoGrid)
    284 complexes × 2 min avg = ~9.5 hrs single-threaded, ~1.2 hrs at 8 workers
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from hybridock_pep.scoring.entropy import CONTACT_DIST_ANG  # Fix A: single source of truth

_log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent

_ADFR_BIN = Path("/home/igem/ADFRsuite_x86_64Linux_1.0/bin")
_BOX_MARGIN = 15.0
_BOX_MIN = 20.0
_CONTACT_CUTOFF = CONTACT_DIST_ANG  # Fix A: import from entropy.py, not hardcoded
_GRID_SPACING = 0.375
_RECEPTOR_TYPES = "C A N NA OA SA HD"
_LIGAND_TYPES = "C A N NA OA SA HD S NS F Cl Br I P"

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "HSD": "H", "HSE": "H", "HSP": "H", "HIE": "H",
    "HID": "H", "HIP": "H", "CYX": "C", "CYM": "C",
    "TPO": "T", "SEP": "S", "PTR": "Y", "MLY": "K",
}


def _find_structure(pdb_id: str) -> Path | None:
    search_dirs = [
        REPO / "datasets" / ds
        for ds in [
            "raw_pdbs", "pdb_2024_2026/structures", "ppii_enriched/structures",
            "pdb_2019_2023/structures", "pdb_2010_2018/structures", "pdb_pre2010/structures",
            "family_targeted/structures", "ppii_extended/structures",
            "training_expanded_structures",
        ]
    ]
    uid = pdb_id.upper()
    for d in search_dirs:
        if not d.exists():
            continue
        for pattern in [f"{uid}.pdb.gz", f"{uid}.pdb", f"{uid.lower()}.pdb"]:
            p = d / pattern
            if p.exists() and p.stat().st_size > 500:
                return p
    return None


def _read_pdb_text(path: Path) -> str:
    import gzip as _gz
    if str(path).endswith(".gz"):
        with _gz.open(path, "rb") as f:
            return f.read().decode("latin-1")
    return path.read_text("latin-1")


def _iter_clean_atom_lines(pdb_text: str):
    """Yield ATOM/HETATM lines from MODEL 1 only with primary altloc.

    Filters that match analyze_calibration_structures.py and
    build_calibration_from_affinity.py — required to avoid:
      - NMR multi-model stacking (multiple model coordinates superimposed)
      - Alternate conformer duplication (altloc B/C atoms overlapping altloc A)
    Both inflate atom counts and corrupt Vina/AD4 scoring.
    """
    in_model: bool = False
    skip_rest: bool = False
    for line in pdb_text.splitlines():
        tag = line[:6].rstrip()
        if tag == "MODEL":
            if not in_model:
                in_model = True
            else:
                skip_rest = True
            continue
        if tag == "ENDMDL":
            if in_model:
                skip_rest = True
            continue
        if skip_rest:
            continue
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        # ALTLOC filter — only blank or 'A'
        altloc = line[16] if len(line) > 16 else " "
        if altloc not in (" ", "A"):
            continue
        yield line


def _split_chains(pdb_text: str, rec_chain: str, pep_chain: str | None) -> tuple[str, str]:
    """Split PDB text into receptor and peptide chain texts.

    Applies NMR MODEL 1 + ALTLOC filtering via _iter_clean_atom_lines.
    """
    rec_lines, pep_lines = [], []
    for line in _iter_clean_atom_lines(pdb_text):
        if len(line) < 22:
            continue
        chain = line[21]
        if chain == rec_chain:
            rec_lines.append(line)
        elif pep_chain and chain == pep_chain:
            pep_lines.append(line)
    return "\n".join(rec_lines) + "\nEND\n", "\n".join(pep_lines) + "\nEND\n"


def _parse_heavy_atoms(pdb_text: str) -> list[tuple[int, float, float, float]]:
    atoms = []
    for line in _iter_clean_atom_lines(pdb_text):
        atom_name = line[12:16].strip() if len(line) > 16 else ""
        if atom_name.startswith("H"):
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            resnum = int(line[22:26].strip()) if len(line) > 26 else 0
            atoms.append((resnum, x, y, z))
        except (ValueError, IndexError):
            continue
    return atoms


def _count_contact_residues(rec_text: str, pep_text: str) -> int:
    """Number of PEPTIDE residues with ≥1 heavy atom within _CONTACT_CUTOFF of receptor.

    Matches the convention in src/hybridock_pep/scoring/entropy.py:count_contact_residues().
    The hybrid score formula uses ``alpha * n_eff_residues`` where n_eff is a function
    of the peptide-side contact count — so calibration and inference must both count on
    the same side (peptide residues), not opposite sides (receptor residues).

    Peptide side: ATOM records only, amino-acid residues only (AA3 filter). HETATM lines
    (HOH waters, ions) in the peptide chain are excluded — they inflate nc above nres and
    cause the calibration to saturate at alpha=_ALPHA_MAX (2.0) on any dataset with
    co-crystallised water molecules assigned to the peptide chain.
    Receptor side: includes HETATM (metal ions, cofactors) for correct contact geometry.
    """
    rec_atoms = _parse_heavy_atoms(rec_text)
    if not rec_atoms:
        return 0
    rec_arr = np.array([(x, y, z) for _, x, y, z in rec_atoms])

    # Group peptide atoms by residue number — ATOM-only, amino-acid residues only.
    # Using _iter_clean_atom_lines directly (rather than _parse_heavy_atoms) so we
    # can inspect the record type (ATOM vs HETATM) and residue name before including.
    pep_by_res: dict[int, list[tuple[float, float, float]]] = {}
    for line in _iter_clean_atom_lines(pep_text):
        if not line.startswith("ATOM"):          # exclude HETATM in peptide chain
            continue
        resname = line[17:20].strip() if len(line) > 20 else ""
        if resname not in AA3:                   # exclude non-amino-acid records
            continue
        atom_name = line[12:16].strip() if len(line) > 16 else ""
        if atom_name.startswith("H"):            # exclude hydrogens
            continue
        try:
            resnum = int(line[22:26].strip())
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            pep_by_res.setdefault(resnum, []).append((x, y, z))
        except (ValueError, IndexError):
            continue

    if not pep_by_res:
        return 0

    n_contact = 0
    cutoff_sq = _CONTACT_CUTOFF ** 2
    for coords in pep_by_res.values():
        if not coords:
            continue
        arr = np.array(coords)
        # Min squared-distance from any atom in this residue to any receptor atom
        diffs = arr[:, np.newaxis, :] - rec_arr[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=-1)
        if sq_dists.min() <= cutoff_sq:
            n_contact += 1
    return n_contact


def _get_box_center_size(pep_text: str) -> tuple[list[float], list[float]]:
    atoms = _parse_heavy_atoms(pep_text)
    if not atoms:
        raise ValueError("No heavy atoms in peptide")
    coords = np.array([(x, y, z) for _, x, y, z in atoms])
    centre = coords.mean(axis=0).tolist()
    half = (coords.max(axis=0) - coords.min(axis=0)) / 2.0
    size = (2 * half + _BOX_MARGIN).tolist()
    size = [max(s, _BOX_MIN) for s in size]
    return centre, size


def _score_vina(rec_pdbqt: Path, pep_pdbqt: Path, centre: list[float], box: list[float]) -> float:
    """Score with Vina Python API (sf_name='vina').  No CLI dependency."""
    from vina import Vina  # noqa: PLC0415
    v = Vina(sf_name="vina", verbosity=0)
    v.set_receptor(str(rec_pdbqt))
    v.compute_vina_maps(center=centre, box_size=box)
    v.set_ligand_from_file(str(pep_pdbqt))
    return float(v.score()[0])


def _score_ad4(pep_pdbqt: Path, maps_dir: Path) -> float:
    """Score with Vina Python API (sf_name='ad4') using pre-computed autogrid4 maps.

    The maps directory must contain 'receptor.maps.fld' (generated by _run_autogrid).
    Uses the same Python vina API approach as score_crystal_poses.py — no autodock4
    CLI dependency.
    """
    from vina import Vina  # noqa: PLC0415
    v = Vina(sf_name="ad4", verbosity=0)
    v.load_maps(str(maps_dir / "receptor"))
    v.set_ligand_from_file(str(pep_pdbqt))
    return float(v.score()[0])


def _prepare_receptor_pdbqt(rec_pdb: Path, out_pdbqt: Path) -> None:
    cmd = [
        str(_ADFR_BIN / "prepare_receptor"),
        "-r", str(rec_pdb),
        "-o", str(out_pdbqt),
        "-A", "checkhydrogens",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if not out_pdbqt.exists():
        raise RuntimeError(f"prepare_receptor failed: {result.stderr[:300]}")
    # Post-process PDBQT to remove lines that break Vina / autogrid4:
    # 1. Unrecognized AD4 atom types (Na, Ni, K, etc.) → autogrid4 "unknown type"
    # 2. Non-primary alternate conformations (altloc ≠ ' ' and ≠ 'A') → Vina PDBQT
    #    parser misreads coordinate fields when alt-loc char displaces residue name
    # 3. Malformed coordinate fields — prepare_receptor sometimes emits lines where
    #    numeric altloc chars (e.g. '1') appear at column 17 (index 16='A' in atom
    #    name) producing residue-name like '1GL' and unreadable X/Y/Z → Vina crash.
    #    Guard: validate X/Y/Z are parseable floats; drop lines that fail.
    lines = out_pdbqt.read_text().splitlines(keepends=True)
    cleaned = []
    dropped = 0
    for ln in lines:
        if ln.startswith("ATOM") or ln.startswith("HETATM"):
            # Altloc filter: drop non-primary conformers.
            # Also guard against shifted altloc at index 17 (HE2A1GLN pattern):
            # if char at index 16 == 'A' but char at 17 is a digit, the line has a
            # numeric altloc one column to the right — treat as non-primary and drop.
            altloc16 = ln[16] if len(ln) > 16 else " "
            char17 = ln[17] if len(ln) > 17 else " "
            if altloc16 not in (" ", "A"):
                dropped += 1
                continue
            if altloc16 == "A" and char17.isdigit():
                # Shifted numeric altloc (prepare_receptor artefact) — drop
                dropped += 1
                continue
            # Atom-type filter: drop lines with types autogrid4 won't accept.
            # Also handles short lines (len ≤ 77) — treat missing type as unknown
            # for HETATM records (keep for ATOM records which are always protein).
            parts = ln[77:].split() if len(ln) > 77 else []
            if not parts:
                if ln.startswith("HETATM"):
                    dropped += 1
                    continue
            elif parts[0] not in _AD4_KNOWN_TYPES:
                dropped += 1
                continue
            # Coordinate sanity check: drop lines whose X/Y/Z columns don't parse.
            # This catches any remaining malformed lines that would crash Vina.
            if len(ln) >= 54:
                try:
                    float(ln[30:38])
                    float(ln[38:46])
                    float(ln[46:54])
                except ValueError:
                    dropped += 1
                    continue
        cleaned.append(ln)
    if dropped:
        out_pdbqt.write_text("".join(cleaned))
        _log.debug("Stripped %d problematic lines from %s (altloc/unknown-type/coord)", dropped, out_pdbqt.name)


def _wrap_rigid_pdbqt(pdbqt: Path) -> None:
    """Wrap flat babel-generated PDBQT in ROOT/ENDROOT/TORSDOF 0 for Vina 1.2.x.

    The vina Python API set_ligand_from_file() requires the torsion-tree headers.
    babel -xr generates flat ATOM records without ROOT/ENDROOT, so we add them.
    """
    lines = pdbqt.read_text().splitlines(keepends=True)
    remarks = [ln for ln in lines if ln.startswith("REMARK")]
    atoms = [ln for ln in lines if ln.startswith("ATOM") or ln.startswith("HETATM")]
    if not atoms:
        raise ValueError(f"No ATOM/HETATM records found in {pdbqt}")
    wrapped = "".join(remarks) + "ROOT\n" + "".join(atoms) + "ENDROOT\nTORSDOF 0\n"
    pdbqt.write_text(wrapped)


def _prepare_ligand_pdbqt(pep_pdb: Path, out_pdbqt: Path) -> None:
    babel = str(_ADFR_BIN / "babel")
    cmd = [babel, "-i", "pdb", str(pep_pdb), "-o", "pdbqt", str(out_pdbqt), "-h", "-xr"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if not out_pdbqt.exists():
        raise RuntimeError(f"babel failed: {result.stderr[:300]}")
    # Vina 1.2.x Python API requires ROOT/ENDROOT/TORSDOF torsion-tree headers
    _wrap_rigid_pdbqt(out_pdbqt)


_AD4_KNOWN_TYPES = frozenset(
    "C A N NA OA SA HD S NS F Cl Br I P Mg Mn Zn Ca Fe Cu Li "
    "MG MN ZN CA FE CU LI W".split()
)


def _get_receptor_atom_types(rec_pdbqt: Path) -> str:
    """Extract unique AD4 atom types from the PDBQT atom-type column (col 77+).

    Filters to known AD4 types — unknown types (e.g. Na, K, Cs metal ions from
    crystal water/salt) are silently dropped. autogrid4 will error on unknown types.
    """
    types: set[str] = set()
    for line in rec_pdbqt.read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) > 77:
            parts = line[77:].split()
            if parts and parts[0] in _AD4_KNOWN_TYPES:
                types.add(parts[0])
    return " ".join(sorted(types)) if types else _RECEPTOR_TYPES


def _run_autogrid(rec_pdbqt: Path, centre: list[float], box: list[float], maps_dir: Path) -> None:
    """Run autogrid4 to generate AD4 maps.

    GPF uses 'receptor' as the stem name so the output is receptor.maps.fld —
    matching what Vina Python API expects for v.load_maps(maps_dir / 'receptor').
    receptor.pdbqt is copied into maps_dir so autogrid4 can find it when running
    with cwd=maps_dir.
    Receptor atom types are detected dynamically to avoid GPF type-count mismatch.
    """
    import shutil as _shutil
    receptor_copy = maps_dir / "receptor.pdbqt"
    # Always overwrite the maps-dir copy — a stale copy from a previous failed run
    # (generated before the PDBQT post-processing filter was applied) causes autogrid4
    # to fail with "unknown type" errors on Na/Ni/etc. atoms that were later stripped
    # from the parent receptor.pdbqt but survive in the cached copy.
    _shutil.copy2(rec_pdbqt, receptor_copy)

    rec_types = _get_receptor_atom_types(receptor_copy)
    n_pts = [int(s / _GRID_SPACING) | 1 for s in box]  # ensure odd
    ad4_params = str(_ADFR_BIN.parent / "CCSBpckgs" / "AutoDockTools" / "AD4_parameters.dat")
    map_lines = "\n".join(f"map receptor.{t}.map" for t in _LIGAND_TYPES.split())
    gpf_content = (
        f"npts {n_pts[0]} {n_pts[1]} {n_pts[2]}\n"
        f"parameter_file {ad4_params}\n"
        "gridfld receptor.maps.fld\n"
        f"spacing {_GRID_SPACING}\n"
        f"receptor_types {rec_types}\n"
        f"ligand_types {_LIGAND_TYPES}\n"
        "receptor receptor.pdbqt\n"
        f"gridcenter {centre[0]:.3f} {centre[1]:.3f} {centre[2]:.3f}\n"
        "smooth 0.5\n"
        f"{map_lines}\n"
        "elecmap receptor.e.map\n"
        "dsolvmap receptor.d.map\n"
        "dielectric -0.1465\n"
    )
    gpf_path = maps_dir / "receptor.gpf"
    gpf_path.write_text(gpf_content)
    result = subprocess.run(
        [str(_ADFR_BIN / "autogrid4"), "-p", "receptor.gpf", "-l", "receptor.glg"],
        capture_output=True, text=True, timeout=300, cwd=maps_dir,
    )
    if not any(maps_dir.glob("*.e.map")):
        raise RuntimeError(
            f"autogrid4 failed (exit {result.returncode}): {result.stderr[:300] or result.stdout[:300]}"
        )


def _score_row(row_dict: dict, work_dir: Path) -> tuple[str, dict | None, str | None]:
    """Wrap score_one() with row dict input and exception capture.

    Module-level (not a closure) so ProcessPoolExecutor can pickle it for workers.
    """
    pdb_id = row_dict["pdb_id"]
    rec_chain = str(row_dict.get("receptor_chain", "") or "")
    try:
        result = score_one(pdb_id, rec_chain, None, work_dir)
        return pdb_id, result, None
    except Exception as exc:
        return pdb_id, None, str(exc)


def score_one(pdb_id: str, rec_chain: str, pep_chain: str | None, work_dir: Path) -> dict:
    """Score a single complex. Returns dict with vina_score, ad4_score, n_contact_residues."""
    struct_path = _find_structure(pdb_id)
    if not struct_path:
        raise FileNotFoundError(f"No structure found for {pdb_id}")

    pdb_text = _read_pdb_text(struct_path)

    # Determine peptide chain if not given — use shortest non-receptor chain 5-30aa
    if not pep_chain or pep_chain == "nan":
        # Import as flat module — when running `python scripts/score_calibration_set.py`,
        # sys.path[0] is the scripts/ directory, so `from scripts.X import Y` fails.
        import sys
        import os
        _scripts_dir = os.path.dirname(os.path.abspath(__file__))
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        from build_calibration_from_affinity import _extract_chains_from_pdb, _classify_chains
        chains = _extract_chains_from_pdb(struct_path)
        pep_chain_auto, _, _, _ = _classify_chains(chains)
        pep_chain = pep_chain_auto

    rec_text, pep_text = _split_chains(pdb_text, rec_chain, pep_chain)
    if not rec_text.strip() or rec_text == "END\n":
        raise ValueError(f"No atoms for receptor chain {rec_chain}")
    if not pep_text.strip() or pep_text == "END\n":
        raise ValueError(f"No atoms for peptide chain {pep_chain}")

    centre, box = _get_box_center_size(pep_text)
    n_contact = _count_contact_residues(rec_text, pep_text)

    entry_dir = work_dir / pdb_id.upper()
    entry_dir.mkdir(parents=True, exist_ok=True)

    rec_pdb = entry_dir / "receptor.pdb"
    pep_pdb = entry_dir / "peptide.pdb"
    rec_pdb.write_text(rec_text)
    pep_pdb.write_text(pep_text)

    rec_pdbqt = entry_dir / "receptor.pdbqt"
    pep_pdbqt = entry_dir / "peptide.pdbqt"
    _prepare_receptor_pdbqt(rec_pdb, rec_pdbqt)
    _prepare_ligand_pdbqt(pep_pdb, pep_pdbqt)

    # Vina scoring
    vina_score = _score_vina(rec_pdbqt, pep_pdbqt, centre, box)

    # AD4 scoring
    maps_dir = entry_dir / "maps"
    maps_dir.mkdir(exist_ok=True)
    _run_autogrid(rec_pdbqt, centre, box, maps_dir)
    ad4_score = _score_ad4(pep_pdbqt, maps_dir)

    return {
        "vina_score": round(vina_score, 3),
        "ad4_score": round(ad4_score, 3),
        "n_contact_residues": n_contact,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--training-csv", type=Path, default=REPO / "data" / "training_complexes_full.csv",
                        help="Calibration CSV. Default: data/training_complexes_full.csv")
    parser.add_argument("--output-csv", type=Path, default=REPO / "runs" / "calibration_full" / "scores.csv",
                        help="Checkpoint CSV (append-safe). Default: runs/calibration_full/scores.csv")
    parser.add_argument("--output-json", type=Path, default=REPO / "data" / "training_scores_full.json",
                        help="Final JSON output. Default: data/training_scores_full.json")
    parser.add_argument("--work-dir", type=Path, default=REPO / "runs" / "calibration_full" / "work",
                        help="Working directory for per-complex files.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default: 4). Vina+AD4 are CPU-bound.")
    parser.add_argument("--affinity-types", nargs="+", default=None,
                        help="Filter: only score these affinity types (e.g. Kd Ki). Default: all.")
    parser.add_argument("--max-entries", type=int, default=None,
                        help="Score at most N entries (for testing).")
    parser.add_argument("--quality-csv", type=Path,
                        default=None,
                        help="Path to calibration_quality.csv from analyze_calibration_structures.py. "
                             "RED and MISSING entries are automatically excluded. "
                             "Default: datasets/calibration_quality.csv if present.")
    parser.add_argument("--skip-red", action="store_true",
                        help="Skip RED/MISSING entries using quality CSV (auto-enabled if quality CSV exists).")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---------------------------------------------------------------
    # Load training CSV
    # ---------------------------------------------------------------
    import pandas as pd
    df = pd.read_csv(args.training_csv)
    _log.info("Loaded %d entries from %s", len(df), args.training_csv)

    if args.affinity_types:
        df = df[df["affinity_type"].isin(args.affinity_types)]
        _log.info("After affinity filter (%s): %d entries", args.affinity_types, len(df))

    # Exclude RED/MISSING entries from structural quality analysis
    quality_csv = args.quality_csv or (REPO / "datasets" / "calibration_quality.csv")
    if quality_csv.exists() and args.skip_red:
        qdf = pd.read_csv(quality_csv)
        bad_ids = set(qdf[qdf["flag"].isin(["RED", "MISSING"])]["pdb_id"].str.upper())
        before = len(df)
        df = df[~df["pdb_id"].str.upper().isin(bad_ids)]
        _log.info("Excluded %d RED/MISSING entries (quality filter): %d → %d",
                  before - len(df), before, len(df))
    elif quality_csv.exists() and not args.quality_csv:
        # Auto-detect: if quality CSV exists, log a reminder but don't auto-skip
        qdf = pd.read_csv(quality_csv)
        n_bad = (qdf["flag"].isin(["RED", "MISSING"])).sum()
        if n_bad:
            _log.warning(
                "Found %d RED/MISSING entries in %s — add --skip-red to exclude them",
                n_bad, quality_csv
            )

    if args.max_entries:
        df = df.head(args.max_entries)
        _log.info("Capped at %d entries", len(df))

    # ---------------------------------------------------------------
    # Load already-scored entries (checkpoint)
    # ---------------------------------------------------------------
    done: dict[str, dict] = {}
    if args.output_csv.exists():
        done_df = pd.read_csv(args.output_csv)
        for _, row in done_df.iterrows():
            pid = row["pdb_id"].strip().lower()
            done[pid] = {
                "vina_score": float(row["vina_score"]),
                "ad4_score": float(row["ad4_score"]),
                "n_contact_residues": int(row.get("n_contact_residues", 0)),
            }
        _log.info("Loaded %d already-scored entries from checkpoint %s", len(done), args.output_csv)

    # ---------------------------------------------------------------
    # Score missing entries
    # ---------------------------------------------------------------
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    todo = df[~df["pdb_id"].str.lower().isin(done.keys())]
    _log.info("Entries to score: %d / %d", len(todo), len(df))

    if not todo.empty:
        csv_mode = "a" if args.output_csv.exists() else "w"
        with open(args.output_csv, csv_mode, newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["pdb_id", "vina_score", "ad4_score", "n_contact_residues"])
            if csv_mode == "w":
                writer.writeheader()

            succeeded = 0
            failed = []

            todo_dicts = todo.to_dict("records")

            if args.workers > 1:
                from concurrent.futures.process import BrokenProcessPool
                remaining = list(todo_dicts)
                while remaining:
                    batch_done_ids: set[str] = set()
                    try:
                        with ProcessPoolExecutor(max_workers=args.workers) as pool:
                            futures = {
                                pool.submit(_score_row, row, args.work_dir): row["pdb_id"]
                                for row in remaining
                            }
                            for future in as_completed(futures):
                                try:
                                    pdb_id, result, err = future.result()
                                except BrokenProcessPool:
                                    raise  # bubble up to outer except
                                batch_done_ids.add(futures[future].lower())
                                if err:
                                    _log.error("[%s] FAILED: %s", pdb_id, err)
                                    failed.append(pdb_id)
                                else:
                                    done[pdb_id.lower()] = result
                                    writer.writerow({
                                        "pdb_id": pdb_id,
                                        "vina_score": result["vina_score"],
                                        "ad4_score": result["ad4_score"],
                                        "n_contact_residues": result["n_contact_residues"],
                                    })
                                    fh.flush()
                                    succeeded += 1
                                    _log.info("[%s] vina=%.2f ad4=%.2f contacts=%d  (done %d/%d)",
                                              pdb_id, result["vina_score"], result["ad4_score"],
                                              result["n_contact_residues"], succeeded, len(todo))
                        break  # completed successfully
                    except BrokenProcessPool:
                        # A worker crashed (segfault/OOM in native Vina code).
                        # Trim remaining to unseen entries and retry with fewer workers.
                        remaining = [r for r in remaining if r["pdb_id"].lower() not in done
                                     and r["pdb_id"].lower() not in batch_done_ids]
                        new_workers = max(1, args.workers // 2)
                        _log.warning(
                            "ProcessPool crashed — restarting with %d workers, %d entries left",
                            new_workers, len(remaining)
                        )
                        args.workers = new_workers
                        if not remaining:
                            break
            else:
                for row_dict in todo_dicts:
                    pdb_id, result, err = _score_row(row_dict, args.work_dir)
                    if err:
                        _log.error("[%s] FAILED: %s", pdb_id, err)
                        failed.append(pdb_id)
                    else:
                        done[pdb_id.lower()] = result
                        writer.writerow({
                            "pdb_id": pdb_id,
                            "vina_score": result["vina_score"],
                            "ad4_score": result["ad4_score"],
                            "n_contact_residues": result["n_contact_residues"],
                        })
                        fh.flush()
                        succeeded += 1
                        _log.info("[%s] vina=%.2f ad4=%.2f contacts=%d  (done %d/%d)",
                                  pdb_id, result["vina_score"], result["ad4_score"],
                                  result["n_contact_residues"], succeeded, len(todo))

        _log.info("Scoring complete: %d succeeded, %d failed", succeeded, len(failed))
        if failed:
            _log.warning("Failed: %s", failed[:20])

    # ---------------------------------------------------------------
    # Write final JSON
    # ---------------------------------------------------------------
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(done, indent=2))
    _log.info("Written %d entries to %s", len(done), args.output_json)

    print(f"\n=== Calibration Scoring Summary ===")
    print(f"Total entries scored: {len(done)}")
    if done:
        vina_vals = [e["vina_score"] for e in done.values()]
        ad4_vals = [e["ad4_score"] for e in done.values()]
        print(f"Vina range: {min(vina_vals):.2f} – {max(vina_vals):.2f} kcal/mol")
        print(f"AD4 range:  {min(ad4_vals):.2f} – {max(ad4_vals):.2f} kcal/mol")
    print(f"\nNext step:")
    print(f"  python scripts/calibrate_alpha.py \\")
    print(f"    --training-csv data/training_complexes_full.csv \\")
    print(f"    --scores-json {args.output_json} \\")
    print(f"    --output data/calibration_full.json")


if __name__ == "__main__":
    main()
