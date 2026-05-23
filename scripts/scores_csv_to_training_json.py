"""Convert scored CSV(s) to the training-scores JSON expected by calibrate_alpha.py.

Reads one or more CSV files containing docking scores and produces a JSON
mapping pdb_id → {vina_score, ad4_score, n_contact_residues}.

Supported CSV formats:
  1. Flat CSV with columns: pdb_id, vina_score, ad4_score [, n_contact_residues]
  2. Directory of run outputs: each subdir has a ranked_poses.csv with the
     best-scoring pose in the first row

Usage:
    # From a flat scored CSV
    python scripts/scores_csv_to_training_json.py \\
        runs/calibration_expanded/expanded_scores.csv \\
        > data/training_scores_expanded.json

    # From a directory of individual run outputs
    python scripts/scores_csv_to_training_json.py \\
        --runs-dir runs/calibration_expanded/ \\
        --output data/training_scores_expanded.json

    # Combine multiple CSVs (last one wins on conflict)
    python scripts/scores_csv_to_training_json.py \\
        runs/production/scores_batch1.csv \\
        runs/production/scores_batch2.csv \\
        --output data/training_scores_production.json

    # Filter to only entries in a training CSV
    python scripts/scores_csv_to_training_json.py \\
        runs/scores.csv \\
        --filter-csv data/training_complexes_full.csv \\
        --output data/training_scores_expanded.json

    # Merge with existing scores JSON (add/update entries)
    python scripts/scores_csv_to_training_json.py \\
        runs/new_scores.csv \\
        --base-json data/training_scores.json \\
        --output data/training_scores_merged.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent


def _read_flat_csv(csv_path: Path) -> dict[str, dict]:
    """Read a flat CSV with columns pdb_id, vina_score, ad4_score [, n_contact_residues]."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"pdb_id", "vina_score", "ad4_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path} missing required columns: {missing}")

    records: dict[str, dict] = {}
    for _, row in df.iterrows():
        pdb_id = str(row["pdb_id"]).strip().lower()
        if not pdb_id or pdb_id == "nan":
            continue
        entry: dict[str, float | int] = {
            "vina_score": round(float(row["vina_score"]), 3),
            "ad4_score": round(float(row["ad4_score"]), 3),
        }
        if "n_contact_residues" in df.columns and pd.notna(row["n_contact_residues"]):
            try:
                entry["n_contact_residues"] = int(row["n_contact_residues"])
            except (ValueError, TypeError):
                pass
        records[pdb_id] = entry
    return records


def _read_runs_dir(runs_dir: Path) -> dict[str, dict]:
    """Scan a directory tree for ranked_poses.csv files (one per run subdir).

    Expects structure:
        runs_dir/
            <pdb_id>/
                ranked_poses.csv     # first row = best pose
    """
    records: dict[str, dict] = {}
    ranked_files = list(runs_dir.glob("*/ranked_poses.csv"))
    _log.info("Found %d ranked_poses.csv files under %s", len(ranked_files), runs_dir)

    for fpath in ranked_files:
        pdb_id = fpath.parent.name.strip().lower()
        try:
            df = pd.read_csv(fpath)
            if df.empty:
                _log.debug("%s: empty ranked_poses.csv", pdb_id)
                continue
            df.columns = [c.strip().lower() for c in df.columns]
            best = df.iloc[0]

            if "vina_score" not in df.columns or "ad4_score" not in df.columns:
                _log.debug("%s: missing score columns in ranked_poses.csv", pdb_id)
                continue

            entry: dict[str, float | int] = {
                "vina_score": round(float(best["vina_score"]), 3),
                "ad4_score": round(float(best["ad4_score"]), 3),
            }
            if "n_contact_residues" in df.columns and pd.notna(best["n_contact_residues"]):
                try:
                    entry["n_contact_residues"] = int(best["n_contact_residues"])
                except (ValueError, TypeError):
                    pass
            records[pdb_id] = entry
        except Exception as exc:
            _log.warning("Failed to read %s: %s", fpath, exc)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert scored CSV(s) to training-scores JSON for calibrate_alpha.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\nUsage:")[1],
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        type=Path,
        help="Flat CSV files to read (pdb_id, vina_score, ad4_score [, n_contact_residues])",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        help="Directory of per-PDB run outputs, each with ranked_poses.csv",
    )
    parser.add_argument(
        "--base-json",
        type=Path,
        help="Existing training_scores JSON to merge with (new entries override existing)",
    )
    parser.add_argument(
        "--filter-csv",
        type=Path,
        help="Only keep entries whose pdb_id appears in this CSV (e.g. training_complexes_full.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSON path. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--min-vina",
        type=float,
        default=-50.0,
        help="Minimum plausible Vina score (default: -50.0 kcal/mol). Filters noise.",
    )
    parser.add_argument(
        "--max-vina",
        type=float,
        default=5.0,
        help="Maximum plausible Vina score (default: +5.0 kcal/mol).",
    )
    args = parser.parse_args()

    if not args.csv_files and not args.runs_dir:
        parser.error("Provide at least one CSV file or --runs-dir")

    # ---------------------------------------------------------------
    # Load base JSON if provided
    # ---------------------------------------------------------------
    combined: dict[str, dict] = {}
    if args.base_json and args.base_json.exists():
        combined = json.loads(args.base_json.read_text())
        _log.info("Loaded base JSON: %d entries from %s", len(combined), args.base_json)

    # ---------------------------------------------------------------
    # Read CSV files
    # ---------------------------------------------------------------
    for csv_path in (args.csv_files or []):
        if not csv_path.exists():
            _log.warning("CSV not found: %s — skipping", csv_path)
            continue
        recs = _read_flat_csv(csv_path)
        _log.info("Read %d entries from %s", len(recs), csv_path)
        combined.update(recs)

    # ---------------------------------------------------------------
    # Read runs directory
    # ---------------------------------------------------------------
    if args.runs_dir:
        if not args.runs_dir.exists():
            _log.error("Runs directory not found: %s", args.runs_dir)
            sys.exit(1)
        recs = _read_runs_dir(args.runs_dir)
        _log.info("Read %d entries from runs dir %s", len(recs), args.runs_dir)
        combined.update(recs)

    # ---------------------------------------------------------------
    # Filter by plausibility
    # ---------------------------------------------------------------
    before = len(combined)
    bad_vina = {
        pid for pid, e in combined.items()
        if not (args.min_vina <= e["vina_score"] <= args.max_vina)
    }
    if bad_vina:
        _log.warning("Removing %d entries with implausible Vina scores: %s",
                     len(bad_vina), sorted(bad_vina)[:10])
        for pid in bad_vina:
            del combined[pid]

    # ---------------------------------------------------------------
    # Filter by training CSV
    # ---------------------------------------------------------------
    if args.filter_csv and args.filter_csv.exists():
        train_df = pd.read_csv(args.filter_csv)
        allowed = set(train_df["pdb_id"].str.lower().tolist())
        before_filter = len(combined)
        combined = {pid: e for pid, e in combined.items() if pid in allowed}
        _log.info("Filter to training CSV: %d → %d entries", before_filter, len(combined))

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    _log.info("Final: %d entries (removed %d implausible Vina)", len(combined), len(bad_vina))
    has_contact = sum(1 for e in combined.values() if "n_contact_residues" in e)
    _log.info("  With n_contact_residues: %d / %d", has_contact, len(combined))
    if combined:
        vina_vals = [e["vina_score"] for e in combined.values()]
        ad4_vals = [e["ad4_score"] for e in combined.values()]
        _log.info("  Vina: %.2f – %.2f  AD4: %.2f – %.2f",
                  min(vina_vals), max(vina_vals), min(ad4_vals), max(ad4_vals))

    if not combined:
        _log.error("No valid entries produced — check input files")
        sys.exit(1)

    # ---------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------
    out_text = json.dumps(combined, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out_text)
        _log.info("Written to %s", args.output)
    else:
        print(out_text)


if __name__ == "__main__":
    main()
