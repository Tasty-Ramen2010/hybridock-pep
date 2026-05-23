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

_log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent

_ADFR_BIN = Path("/home/igem/ADFRsuite_x86_64Linux_1.0/bin")
_VINA_BIN = "vina"
_BOX_MARGIN = 15.0
_BOX_MIN = 20.0
_CONTACT_CUTOFF = 4.5  # must match CONTACT_DIST_ANG in src/hybridock_pep/scoring/entropy.py
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


def _split_chains(pdb_text: str, rec_chain: str, pep_chain: str | None) -> tuple[str, str]:
    """Split PDB text into receptor and peptide chain texts."""
    rec_lines, pep_lines = [], []
    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
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
    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
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
    rec_atoms = _parse_heavy_atoms(rec_text)
    pep_atoms = _parse_heavy_atoms(pep_text)
    if not rec_atoms or not pep_atoms:
        return 0
    rec_arr = np.array([(x, y, z) for _, x, y, z in rec_atoms])
    pep_arr = np.array([(x, y, z) for _, x, y, z in pep_atoms])
    rec_resnums = np.array([r for r, _, _, _ in rec_atoms])

    contacts: set[int] = set()
    for px, py, pz in pep_arr:
        dists = np.sqrt(np.sum((rec_arr - [px, py, pz]) ** 2, axis=1))
        contacts.update(rec_resnums[dists < _CONTACT_CUTOFF].tolist())
    return len(contacts)


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
    cmd = [
        _VINA_BIN, "--score_only",
        "--receptor", str(rec_pdbqt),
        "--ligand", str(pep_pdbqt),
        f"--center_x={centre[0]:.3f}",
        f"--center_y={centre[1]:.3f}",
        f"--center_z={centre[2]:.3f}",
        f"--size_x={box[0]:.1f}",
        f"--size_y={box[1]:.1f}",
        f"--size_z={box[2]:.1f}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    for line in result.stdout.splitlines():
        if "Affinity" in line or "affinity" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p.lower() in ("affinity:", "affinity"):
                    return float(parts[i + 1])
    raise RuntimeError(f"Vina output unrecognized:\n{result.stdout[:500]}")


def _score_ad4(pep_pdbqt: Path, maps_dir: Path) -> float:
    map_name = next(maps_dir.glob("*.e.map")).stem.rsplit(".", 1)[0]
    dpf_path = maps_dir / "ligand.dpf"
    dpf_content = (
        f"ligand_types {_LIGAND_TYPES}\n"
        f"fld {map_name}.maps.fld\n"
        f"move {pep_pdbqt.resolve()}\n"
        "do_local_only 1\n"
        "ga_run 1\n"
        "analysis\n"
    )
    dpf_path.write_text(dpf_content)
    result = subprocess.run(
        ["autodock4", "-p", str(dpf_path), "-l", str(maps_dir / "autodock.dlg")],
        capture_output=True, text=True, timeout=300, cwd=maps_dir,
    )
    dlg_path = maps_dir / "autodock.dlg"
    if not dlg_path.exists():
        raise RuntimeError("AutoDock DLG not created")
    for line in dlg_path.read_text().splitlines():
        if "DOCKED: USER    Estimated Free Energy" in line:
            return float(line.split("=")[1].split("kcal")[0].strip())
    raise RuntimeError("AD4 energy not found in DLG")


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


def _prepare_ligand_pdbqt(pep_pdb: Path, out_pdbqt: Path) -> None:
    cmd = ["babel", "-ipdb", str(pep_pdb), "-opdbqt", str(out_pdbqt), "-xr", "--partialcharge", "gasteiger"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if not out_pdbqt.exists():
        raise RuntimeError(f"babel failed: {result.stderr[:300]}")


def _run_autogrid(rec_pdbqt: Path, centre: list[float], box: list[float], maps_dir: Path) -> None:
    n_pts = [int(s / _GRID_SPACING) | 1 for s in box]  # ensure odd
    gpf_content = (
        f"npts {n_pts[0]} {n_pts[1]} {n_pts[2]}\n"
        f"gridfld {maps_dir.name}.maps.fld\n"
        f"spacing {_GRID_SPACING}\n"
        f"receptor_types {_RECEPTOR_TYPES}\n"
        f"ligand_types {_LIGAND_TYPES}\n"
        f"receptor {rec_pdbqt.resolve()}\n"
        f"gridcenter {centre[0]:.3f} {centre[1]:.3f} {centre[2]:.3f}\n"
        "smooth 0.5\n"
        "map *.map\n"
        "elecmap *.e.map\n"
        "dsolvmap *.d.map\n"
        "dielectric -0.1465\n"
    )
    gpf_path = maps_dir / f"{maps_dir.name}.gpf"
    gpf_path.write_text(gpf_content)
    result = subprocess.run(
        ["autogrid4", "-p", str(gpf_path), "-l", str(maps_dir / "autogrid.glg")],
        capture_output=True, text=True, timeout=300, cwd=maps_dir,
    )
    if not any(maps_dir.glob("*.e.map")):
        raise RuntimeError(f"autogrid4 failed: {result.stderr[:300]}")


def score_one(pdb_id: str, rec_chain: str, pep_chain: str | None, work_dir: Path) -> dict:
    """Score a single complex. Returns dict with vina_score, ad4_score, n_contact_residues."""
    struct_path = _find_structure(pdb_id)
    if not struct_path:
        raise FileNotFoundError(f"No structure found for {pdb_id}")

    pdb_text = _read_pdb_text(struct_path)

    # Determine peptide chain if not given — use shortest non-receptor chain 5-30aa
    if not pep_chain or pep_chain == "nan":
        from scripts.build_calibration_from_affinity import _extract_chains_from_pdb, _classify_chains
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

            def _score_row(row_dict: dict) -> tuple[str, dict | None, str | None]:
                pdb_id = row_dict["pdb_id"]
                rec_chain = str(row_dict.get("receptor_chain", "") or "")
                try:
                    result = score_one(pdb_id, rec_chain, None, args.work_dir)
                    return pdb_id, result, None
                except Exception as exc:
                    return pdb_id, None, str(exc)

            todo_dicts = todo.to_dict("records")

            if args.workers > 1:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    futures = {pool.submit(_score_row, row): row["pdb_id"] for row in todo_dicts}
                    for future in as_completed(futures):
                        pdb_id, result, err = future.result()
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
            else:
                for row_dict in todo_dicts:
                    pdb_id, result, err = _score_row(row_dict)
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
