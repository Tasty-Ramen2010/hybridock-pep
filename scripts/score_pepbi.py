"""Score PEPBI database protein-peptide complexes with Vina + AD4.

PEPBI (Predicted and Experimental Peptide Binding Information) database
DOI: 10.5061/dryad.wstqjq2wk — ITC-measured Kd for 329 protein-peptide complexes
across 32 binding groups; structures are Rosetta-modelled protein-peptide complexes.

All structures have chain A = protein receptor, chain B = peptide.
All experimental data is ITC-measured Kd (true Kd, not IC50).

Usage:
    python scripts/score_pepbi.py \\
        --pepbi-zip doi_10_5061_dryad_wstqjq2wk__v20250617.zip \\
        --output-csv data/pepbi_scores.csv \\
        --output-training-csv data/training_complexes_pepbi.csv \\
        --output-json data/training_scores_pepbi.json \\
        --workers 4

    # Score only the best-span groups (recommended first pass):
    python scripts/score_pepbi.py --pepbi-zip ... --min-span 1.0 --workers 4

    # Score specific binding groups:
    python scripts/score_pepbi.py --pepbi-zip ... --groups "TtSlyD - S2 Fragment" "SUMO2 - PIASX SIM"
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from hybridock_pep.scoring.entropy import CONTACT_DIST_ANG

# Import scoring helpers from score_calibration_set
_SCRIPT_DIR = Path(__file__).resolve().parent
_sys.path.insert(0, str(_SCRIPT_DIR))
from score_calibration_set import (
    _ADFR_BIN, _CONTACT_CUTOFF, AA3,
    _iter_clean_atom_lines, _parse_heavy_atoms, _split_chains,
    _count_contact_residues, _get_box_center_size,
    _score_vina, _score_ad4, _prepare_receptor_pdbqt,
    _prepare_ligand_pdbqt, _run_autogrid,
)

_log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent


def _load_pepbi_xlsx(xlsx_bytes: bytes) -> pd.DataFrame:
    """Load PEPBI.xlsx and return a cleaned DataFrame."""
    df = pd.read_excel(io.BytesIO(xlsx_bytes), skiprows=1)
    kd_numeric = pd.to_numeric(df["KD (M)"], errors="coerce")
    df["pKd"] = -np.log10(kd_numeric)
    df["dG_calc"] = pd.to_numeric(df["Calculated ΔG (kcal/mol)"], errors="coerce")
    df["T_K"] = pd.to_numeric(df["T (K)"], errors="coerce")
    return df


def _is_clean(change: str) -> bool:
    """Return True if the entry has only peptide-sequence/temperature changes.

    Excludes entries with protein mutations or binding site changes, which
    would alter the receptor structure or scoring baseline.
    """
    c = str(change) if not (isinstance(change, float)) else ""
    return "Protein" not in c and "Binding Site" not in c


def build_clean_scoring_set(
    df: pd.DataFrame,
    min_span: float = 0.0,
    groups: list[str] | None = None,
) -> pd.DataFrame:
    """Filter PEPBI to the cleanest calibration-ready subset.

    Selection criteria:
    - Peptide mutations only (no protein mutations, no binding-site change)
    - Prefer Crystallographic Unit A1B1 (where multiple units exist)
    - Prefer T=298.15 K (where temperature replicates exist)
    - Require structural file to be findable in the zip

    Args:
        df: Full PEPBI DataFrame from _load_pepbi_xlsx.
        min_span: Minimum pKd span required to include a binding group (default 0).
        groups: If given, restrict to these binding group names.

    Returns:
        DataFrame with one row per unique (Binding Group, PEPBI Complex Name),
        columns: Binding Group, PEPBI Complex Name, Crystallographic Unit,
                 peptide_sequence, pKd, dG_calc, T_K, pep_len, struct_zip_subpath.
    """
    df = df.copy()
    df["_clean"] = df["Change from Binding Group Reference Complex"].apply(_is_clean)
    clean = df[df["_clean"]].copy()

    if groups:
        clean = clean[clean["Binding Group"].isin(groups)]

    rows = []
    for (bg, name), grp in clean.groupby(["Binding Group", "PEPBI Complex Name"]):
        # Prefer A1B1 unit
        if grp["Crystallographic Unit"].nunique() > 1:
            sub = grp[grp["Crystallographic Unit"] == "A1B1"]
            if sub.empty:
                sub = grp
        else:
            sub = grp
        # Prefer T≈298.15 K
        temps = pd.to_numeric(sub["T_K"], errors="coerce")
        t298 = sub[abs(temps - 298.15) < 0.5]
        row = t298.iloc[0] if not t298.empty else sub.iloc[0]
        rows.append(row)

    best = pd.DataFrame(rows).reset_index(drop=True)

    # Build zip sub-path for each structure — caller must pass all_zip_paths for validation
    # (placeholder; actual path resolution happens in main() after zip is opened)
    def _zip_subpath(row: pd.Series) -> str | None:
        bg = row["Binding Group"]
        name = row["PEPBI Complex Name"]
        unit = str(row["Crystallographic Unit"]) if not pd.isna(row["Crystallographic Unit"]) else ""
        # Return both candidates; caller resolves against actual zip contents
        if unit:
            return f"Binding Group Structures/{bg}/{unit}/{name}.pdb"
        return f"Binding Group Structures/{bg}/{name}.pdb"

    best["struct_zip_subpath"] = best.apply(_zip_subpath, axis=1)

    # Filter by min_span
    if min_span > 0:
        spans = best.groupby("Binding Group")["pKd"].transform(lambda x: x.max() - x.min())
        best = best[spans >= min_span].reset_index(drop=True)

    return best


def score_pepbi_complex(
    pdb_text: str,
    pepbi_name: str,
    work_dir: Path,
) -> dict:
    """Score a single PEPBI complex (chain A = receptor, chain B = peptide).

    Args:
        pdb_text: Full PDB text of the Rosetta model (protein + peptide).
        pepbi_name: PEPBI complex name (e.g. ttslyd_s2_1), used for work subdir.
        work_dir: Parent directory for per-complex temporary files.

    Returns:
        dict with keys: vina_score, ad4_score, n_contact_residues.

    Raises:
        ValueError: If chain A or B is empty in the PDB.
        RuntimeError: If prepare_receptor/babel/autogrid4/vina fail.
    """
    rec_text, pep_text = _split_chains(pdb_text, rec_chain="A", pep_chain="B")
    if not rec_text.strip() or rec_text.strip() == "END":
        raise ValueError(f"{pepbi_name}: no atoms in chain A (receptor)")
    if not pep_text.strip() or pep_text.strip() == "END":
        raise ValueError(f"{pepbi_name}: no atoms in chain B (peptide)")

    centre, box = _get_box_center_size(pep_text)
    n_contact = _count_contact_residues(rec_text, pep_text)

    entry_dir = work_dir / pepbi_name
    entry_dir.mkdir(parents=True, exist_ok=True)

    rec_pdb = entry_dir / "receptor.pdb"
    pep_pdb = entry_dir / "peptide.pdb"
    rec_pdb.write_text(rec_text)
    pep_pdb.write_text(pep_text)

    rec_pdbqt = entry_dir / "receptor.pdbqt"
    pep_pdbqt = entry_dir / "peptide.pdbqt"
    _prepare_receptor_pdbqt(rec_pdb, rec_pdbqt)
    _prepare_ligand_pdbqt(pep_pdb, pep_pdbqt)

    vina_score = _score_vina(rec_pdbqt, pep_pdbqt, centre, box)

    maps_dir = entry_dir / "maps"
    maps_dir.mkdir(exist_ok=True)
    _run_autogrid(rec_pdbqt, centre, box, maps_dir)
    ad4_score = _score_ad4(pep_pdbqt, maps_dir)

    return {
        "vina_score": round(vina_score, 3),
        "ad4_score": round(ad4_score, 3),
        "n_contact_residues": n_contact,
    }


def _score_extracted_file(
    name: str, pdb_path: Path, work_dir: Path
) -> tuple[str, dict | None, str | None]:
    """Module-level worker for ProcessPoolExecutor (must be picklable)."""
    try:
        pdb_text = pdb_path.read_text("latin-1")
        result = score_pepbi_complex(pdb_text, name, work_dir)
        return name, result, None
    except Exception as exc:
        return name, None, str(exc)


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pepbi-zip",
        type=Path,
        default=REPO / "doi_10_5061_dryad_wstqjq2wk__v20250617.zip",
        help="Path to PEPBI Dryad zip (doi_10_5061_dryad_wstqjq2wk__v20250617.zip).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO / "data" / "pepbi_scores.csv",
        help="Checkpoint CSV (append-safe). Default: data/pepbi_scores.csv",
    )
    parser.add_argument(
        "--output-training-csv",
        type=Path,
        default=REPO / "data" / "training_complexes_pepbi.csv",
        help="Training CSV (pdb_id=PEPBI name, peptide_sequence, experimental_pkd). "
             "Default: data/training_complexes_pepbi.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO / "data" / "training_scores_pepbi.json",
        help="Scores JSON for calibrate_alpha.py. Default: data/training_scores_pepbi.json",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO / "runs" / "pepbi_scoring" / "work",
        help="Per-complex working directory.",
    )
    parser.add_argument(
        "--min-span",
        type=float,
        default=1.0,
        help="Minimum pKd span to include a binding group (default: 1.0). "
             "Groups with span < min-span are too narrow for calibration.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Restrict to these binding group names (space-separated). Default: all.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers (default: 1). Note: Rosetta structures have large "
             "receptors; > 4 workers rarely helps on a single socket.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---------------------------------------------------------------
    # Load PEPBI data
    # ---------------------------------------------------------------
    _log.info("Loading PEPBI zip: %s", args.pepbi_zip)
    outer_zip = zipfile.ZipFile(args.pepbi_zip)
    xlsx_bytes = outer_zip.read("PEPBI.xlsx")
    df = _load_pepbi_xlsx(xlsx_bytes)
    _log.info("Loaded PEPBI: %d rows, %d binding groups", len(df), df["Binding Group"].nunique())

    structures_zip_bytes = outer_zip.read("Binding_Group_Structures.zip")
    structures_zip = zipfile.ZipFile(io.BytesIO(structures_zip_bytes))
    all_zip_paths = set(structures_zip.namelist())

    # ---------------------------------------------------------------
    # Build clean scoring set
    # ---------------------------------------------------------------
    scoring_set = build_clean_scoring_set(df, min_span=args.min_span, groups=args.groups)

    # Resolve actual zip paths — fall back from unit-subdir to flat layout
    def _resolve_zip_path(row: pd.Series) -> str | None:
        bg = row["Binding Group"]
        name = row["PEPBI Complex Name"]
        unit = str(row["Crystallographic Unit"]) if not pd.isna(row["Crystallographic Unit"]) else ""
        # Priority 1: with unit subdir (TtSlyD, CAPERα, etc.)
        if unit:
            p = f"Binding Group Structures/{bg}/{unit}/{name}.pdb"
            if p in all_zip_paths:
                return p
        # Priority 2: flat layout (SUMO, α-adaptin, etc.)
        p = f"Binding Group Structures/{bg}/{name}.pdb"
        if p in all_zip_paths:
            return p
        return None

    scoring_set["struct_zip_subpath"] = scoring_set.apply(_resolve_zip_path, axis=1)
    missing = scoring_set[scoring_set["struct_zip_subpath"].isna()]
    if not missing.empty:
        _log.warning(
            "%d entries have no structure in zip — skipping: %s",
            len(missing),
            missing["PEPBI Complex Name"].tolist(),
        )
    scoring_set = scoring_set[scoring_set["struct_zip_subpath"].notna()].reset_index(drop=True)

    _log.info(
        "Clean scoring set: %d unique structures from %d binding groups",
        len(scoring_set),
        scoring_set["Binding Group"].nunique(),
    )
    for bg, grp in scoring_set.groupby("Binding Group"):
        span = grp["pKd"].max() - grp["pKd"].min()
        _log.info("  %s: n=%d  pKd=%.2f–%.2f  span=%.2f",
                  bg, len(grp), grp["pKd"].min(), grp["pKd"].max(), span)

    # ---------------------------------------------------------------
    # Write training CSV (consumed by calibrate_alpha.py)
    # ---------------------------------------------------------------
    args.output_training_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_training_csv, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["pdb_id", "peptide_sequence", "experimental_pkd", "binding_group"]
        )
        writer.writeheader()
        for _, row in scoring_set.iterrows():
            writer.writerow({
                "pdb_id": row["PEPBI Complex Name"],
                "peptide_sequence": str(row["Peptide Sequence"]),
                "experimental_pkd": round(float(row["pKd"]), 6),
                "binding_group": row["Binding Group"],
            })
    _log.info("Wrote training CSV: %s", args.output_training_csv)

    # ---------------------------------------------------------------
    # Load already-scored entries (checkpoint)
    # ---------------------------------------------------------------
    done: dict[str, dict] = {}
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.output_csv.exists():
        done_df = pd.read_csv(args.output_csv)
        for _, row in done_df.iterrows():
            pid = str(row["pdb_id"]).strip()
            done[pid] = {
                "vina_score": float(row["vina_score"]),
                "ad4_score": float(row["ad4_score"]),
                "n_contact_residues": int(row.get("n_contact_residues", 0)),
            }
        _log.info("Loaded %d checkpoint scores from %s", len(done), args.output_csv)

    todo = scoring_set[~scoring_set["PEPBI Complex Name"].isin(done.keys())]
    _log.info("To score: %d / %d", len(todo), len(scoring_set))

    if todo.empty:
        _log.info("All entries already scored.")
    else:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        csv_mode = "a" if args.output_csv.exists() else "w"
        with open(args.output_csv, csv_mode, newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["pdb_id", "vina_score", "ad4_score", "n_contact_residues"]
            )
            if csv_mode == "w":
                writer.writeheader()

            succeeded = 0
            failed = []

            def _score_one_row(row_series: pd.Series) -> tuple[str, dict | None, str | None]:
                name = row_series["PEPBI Complex Name"]
                zip_path = row_series["struct_zip_subpath"]
                try:
                    pdb_bytes = structures_zip.read(zip_path)
                    pdb_text = pdb_bytes.decode("latin-1")
                    result = score_pepbi_complex(pdb_text, name, args.work_dir)
                    return name, result, None
                except Exception as exc:
                    return name, None, str(exc)

            if args.workers > 1:
                # ProcessPoolExecutor can't pickle zipfile objects — extract to temp dir first
                import shutil
                from concurrent.futures import ProcessPoolExecutor, as_completed

                tmp_pdb_dir = args.work_dir / "_pdbs"
                tmp_pdb_dir.mkdir(exist_ok=True)
                _log.info("Extracting %d structures to %s for parallel scoring...",
                          len(todo), tmp_pdb_dir)

                pdb_paths: dict[str, Path] = {}
                for _, row in todo.iterrows():
                    name = row["PEPBI Complex Name"]
                    pdb_bytes = structures_zip.read(row["struct_zip_subpath"])
                    out = tmp_pdb_dir / f"{name}.pdb"
                    out.write_bytes(pdb_bytes)
                    pdb_paths[name] = out

                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    futures = {
                        pool.submit(_score_extracted_file, name, pdb_paths[name], args.work_dir): name
                        for name in pdb_paths
                    }
                    for future in as_completed(futures):
                        name, result, err = future.result()
                        if err:
                            _log.error("[%s] FAILED: %s", name, err)
                            failed.append(name)
                        else:
                            done[name] = result
                            writer.writerow({"pdb_id": name, **result})
                            fh.flush()
                            succeeded += 1
                            _log.info(
                                "[%s] vina=%.2f ad4=%.2f nc=%d  (done %d/%d)",
                                name, result["vina_score"], result["ad4_score"],
                                result["n_contact_residues"], succeeded, len(todo),
                            )
            else:
                for _, row in todo.iterrows():
                    name, result, err = _score_one_row(row)
                    if err:
                        _log.error("[%s] FAILED: %s", name, err)
                        failed.append(name)
                    else:
                        done[name] = result
                        writer.writerow({"pdb_id": name, **result})
                        fh.flush()
                        succeeded += 1
                        _log.info(
                            "[%s] vina=%.2f ad4=%.2f nc=%d  (done %d/%d)",
                            name, result["vina_score"], result["ad4_score"],
                            result["n_contact_residues"], succeeded, len(todo),
                        )

        _log.info("Scoring complete: %d succeeded, %d failed", succeeded, len(failed))
        if failed:
            _log.warning("Failed entries: %s", failed)

    # ---------------------------------------------------------------
    # Write scores JSON (calibrate_alpha.py format)
    # ---------------------------------------------------------------
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(done, indent=2))
    _log.info("Wrote scores JSON: %s  (%d entries)", args.output_json, len(done))

    # ---------------------------------------------------------------
    # Quick Pearson r preview per binding group
    # ---------------------------------------------------------------
    if done:
        from scipy.stats import pearsonr as _pearsonr
        print("\n=== Quick correlation check (Vina vs ΔG_exp per binding group) ===")
        _RT = 0.001987 * 298.15
        for bg, grp in scoring_set.groupby("Binding Group"):
            names = grp["PEPBI Complex Name"].tolist()
            scored = [(n, done[n]) for n in names if n in done]
            if len(scored) < 3:
                continue
            vinas = [s["vina_score"] for _, s in scored]
            dgs = [-_RT * 2.303 * grp[grp["PEPBI Complex Name"] == n]["pKd"].iloc[0]
                   for n, _ in scored]
            try:
                r, p = _pearsonr(vinas, dgs)
                print(f"  {bg:<42s}  n={len(scored):2d}  r(Vina,ΔG)={r:+.3f}  p={p:.3f}")
            except Exception:
                pass
        print()
        print(f"Next step:")
        print(f"  python scripts/calibrate_alpha.py \\")
        print(f"    --training-csv {args.output_training_csv} \\")
        print(f"    --scores-json {args.output_json} \\")
        print(f"    --output data/calibration_pepbi.json")


if __name__ == "__main__":
    main()
