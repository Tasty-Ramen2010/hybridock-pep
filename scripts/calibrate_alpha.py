"""Calibration CLI: fit alpha and beta from pre-computed scores and experimental pKd.

This script is a thin wrapper around ``hybridock_pep.scoring.entropy.fit_calibration()``.
All optimization logic lives in entropy.py. This script handles I/O only:

1. Read training CSV (calibration schema: pdb_id, peptide_sequence, experimental_pkd).
2. Read --scores-json (mapping pdb_id → {vina_score, ad4_score}).
3. Derive n_residues from len(peptide_sequence) in the CSV.
4. Call fit_calibration() with the assembled arrays.
5. Write calibration.json via write_calibration() (calibration schema).
6. Self-validate with load_calibration() — aborts if bounds exceeded.

Usage (Phase 3, pre-computed scores):
    python scripts/calibrate_alpha.py \\
        --training-csv data/training_complexes.csv \\
        --scores-json path/to/scores.json \\
        --output data/calibration.json

Note: Live Vina/AD4 scoring is wired in Phase 5 via hybridock-pep calibrate.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

_log = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for calibrate_alpha.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed argparse.Namespace with training_csv, scores_json, output, verbose.
    """
    parser = argparse.ArgumentParser(
        prog="calibrate_alpha",
        description=(
            "Fit entropy calibration parameters (alpha, beta) from pre-computed "
            "Vina/AD4 scores and experimental pKd values. Thin wrapper around "
            "entropy.fit_calibration()."
        ),
    )
    parser.add_argument(
        "--training-csv",
        dest="training_csv",
        type=Path,
        default=Path("data/training_complexes.csv"),
        metavar="PATH",
        help=(
            "Path to 3-column  training CSV "
            "(pdb_id, peptide_sequence, experimental_pkd). "
            "Default: data/training_complexes.csv"
        ),
    )
    parser.add_argument(
        "--scores-json",
        dest="scores_json",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to JSON mapping pdb_id → {vina_score, ad4_score}. "
            "REQUIRED for Phase 3 calibration. "
            "If not provided, the script aborts with a clear error. "
            "Live scoring is wired in Phase 5 via hybridock-pep calibrate."
        ),
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=Path,
        default=Path("data/calibration.json"),
        metavar="PATH",
        help="Path to write calibration.json (calibration schema). Default: data/calibration.json",
    )
    parser.add_argument(
        "--gamma",
        dest="gamma",
        type=float,
        default=0.2,
        metavar="FLOAT",
        help=(
            "Non-contact residue entropy fraction [0.0, 1.0]. "
            "Non-contact residues pay gamma * alpha per-residue entropy. "
            "Requires n_contact_residues in scores JSON to have effect. "
            "Default: 0.2"
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def main(args: argparse.Namespace | None = None) -> None:
    """Run the calibration workflow.

    Reads training CSV and scores JSON, calls fit_calibration(), writes
    calibration.json, and self-validates with load_calibration().

    Args:
        args: Parsed argparse.Namespace. If None, parse_args() is called.

    Raises:
        ValueError: If --scores-json is not provided, or if a pdb_id in the
            training CSV is missing from the scores JSON, or if converted float
            values are malformed, or if the fitted alpha/beta fail load_calibration()
            post-write validation.
        FileNotFoundError: If training_csv or scores_json path does not exist.
    """
    if args is None:
        args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate --scores-json is provided (required in Phase 3)
    if args.scores_json is None:
        raise ValueError(
            "--scores-json is required for Phase 3 calibration; "
            "run hybridock-pep calibrate in Phase 5 for live scoring"
        )

    _log.debug("Loading scores JSON from %s", args.scores_json)
    scores: dict[str, dict[str, float]] = json.loads(Path(args.scores_json).read_text())

    # Read training CSV (calibration schema)
    _log.debug("Reading training CSV from %s", args.training_csv)
    with Path(args.training_csv).open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    vina_scores: list[float] = []
    ad4_scores: list[float] = []
    n_residues_list: list[int] = []
    n_contact_list: list[int] = []
    pkd_list: list[float] = []

    for row in rows:
        pdb_id = row["pdb_id"]
        peptide_sequence = row["peptide_sequence"]
        experimental_pkd_str = row["experimental_pkd"]

        if pdb_id not in scores:
            raise ValueError(
                f"pdb_id '{pdb_id}' from training CSV not found in scores JSON. "
                f"Available pdb_ids: {sorted(scores.keys())}"
            )

        entry = scores[pdb_id]
        try:
            vina_score = float(entry["vina_score"])
        except (ValueError, KeyError) as exc:
            raise ValueError(
                f"Invalid vina_score for pdb_id '{pdb_id}' in scores JSON: {exc}"
            ) from exc
        try:
            ad4_score = float(entry["ad4_score"])
        except (ValueError, KeyError) as exc:
            raise ValueError(
                f"Invalid ad4_score for pdb_id '{pdb_id}' in scores JSON: {exc}"
            ) from exc
        try:
            pkd = float(experimental_pkd_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid experimental_pkd for pdb_id '{pdb_id}' in training CSV: {exc}"
            ) from exc

        vina_scores.append(vina_score)
        ad4_scores.append(ad4_score)
        n_residues_list.append(len(peptide_sequence))  # derived from CSV, not scores JSON
        pkd_list.append(pkd)
        if "n_contact_residues" in entry:
            n_contact_list.append(int(entry["n_contact_residues"]))

    n_complexes = len(pkd_list)
    has_contact_data = len(n_contact_list) == n_complexes
    if has_contact_data:
        _log.info(
            "Calibrating on %d complexes (contact-based, γ=%.2f) from %s",
            n_complexes, args.gamma, args.training_csv,
        )
    else:
        _log.info(
            "Calibrating on %d complexes (residue-based, n_contact_residues absent) from %s",
            n_complexes, args.training_csv,
        )

    # Import scoring functions lazily to ensure package is installed in score-env
    from hybridock_pep.scoring.entropy import fit_calibration, load_calibration, write_calibration

    result = fit_calibration(
        vina_scores,
        ad4_scores,
        n_residues_list,
        pkd_list,
        n_contact_residues_list=n_contact_list if has_contact_data else None,
        gamma=args.gamma,
    )

    extra_meta: dict = {}
    if has_contact_data:
        extra_meta["n_contact_residues_training"] = n_contact_list

    write_calibration(
        Path(args.output),
        training_csv=str(args.training_csv),
        n_complexes=n_complexes,
        **extra_meta,
        **result,
    )

    _log.info(
        "Calibrated: alpha=%.3f beta=%.3f r=%.3f RMSE=%.3f -> %s",
        result["alpha"],
        result["beta"],
        result["pearson_r"],
        result["rmse_kcal_mol"],
        args.output,
    )

    # Post-write self-check: load_calibration validates the just-written file.
    # If alpha/beta converged outside bounds (should not happen with L-BFGS-B),
    # this aborts with an informative ValueError rather than silently writing
    # a corrupt calibration.
    load_calibration(Path(args.output))
    _log.debug("Post-write self-check passed: %s", args.output)


if __name__ == "__main__":
    main()
