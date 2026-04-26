from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from hybridock_pep.models import DockConfig

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the hybridock-pep argparse parser with fully defined subcommands.

    Returns:
        Configured ArgumentParser with dock/calibrate/benchmark/prep subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="hybridock-pep",
        description="Hybrid peptide docking: RAPiDock sampling + physics-based rescoring.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v INFO, -vv DEBUG).",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = False

    # dock subparser
    p_dock = sub.add_parser("dock", help="Run end-to-end docking pipeline.")
    p_dock.add_argument(
        "--peptide",
        required=True,
        metavar="SEQ",
        help="Peptide amino acid sequence (single-letter codes, e.g. LISDAELEAIFEADC).",
    )
    p_dock.add_argument(
        "--receptor",
        required=True,
        metavar="PDB",
        help="Path to receptor PDB file.",
    )
    p_dock.add_argument(
        "--site",
        required=True,
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Grid box center coordinates in Angstroms (x y z).",
    )
    p_dock.add_argument(
        "--box",
        required=True,
        type=float,
        metavar="ANGSTROMS",
        help="Grid box edge length in Angstroms.",
    )
    p_dock.add_argument(
        "--n-samples",
        type=int,
        default=None,
        metavar="N",
        help="Number of RAPiDock inference passes (default: 100). Mutually exclusive with --input-poses.",
    )
    p_dock.add_argument(
        "--scoring",
        default="vina,ad4",
        metavar="BACKENDS",
        help="Comma-separated scoring backends: vina, ad4 (default: vina,ad4).",
    )
    p_dock.add_argument(
        "--refine-topk",
        type=int,
        default=None,
        metavar="K",
        help="Top-K poses for MM-GBSA refinement (v2; validated but not dispatched in v1).",
    )
    p_dock.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory for run outputs (created if absent).",
    )
    p_dock.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Random seed for deterministic sampling (modulo CUDA nondeterminism).",
    )
    p_dock.add_argument(
        "--input-poses",
        default=None,
        metavar="DIR",
        help="Directory of pre-generated pose PDBs (skips RAPiDock Stage 1). Required on macOS.",
    )
    p_dock.add_argument(
        "--calibration",
        default="data/calibration.json",
        metavar="JSON",
        help="Path to calibration.json for entropy correction (default: data/calibration.json).",
    )

    # calibrate subparser
    p_cal = sub.add_parser("calibrate", help="Calibrate entropy correction coefficient alpha.")
    p_cal.add_argument(
        "--training-csv",
        default="data/training_complexes.csv",
        metavar="CSV",
        help="Training CSV with columns: pdb_id, peptide_sequence, experimental_pkd.",
    )
    p_cal.add_argument(
        "--scores-json",
        required=True,
        metavar="JSON",
        help="JSON mapping pdb_id to {vina_score, ad4_score} for calibration.",
    )
    p_cal.add_argument(
        "--output",
        default="data/calibration.json",
        metavar="JSON",
        help="Output calibration.json path (default: data/calibration.json).",
    )

    # prep subparser
    p_prep = sub.add_parser("prep", help="Prepare receptor PDBQT for docking.")
    p_prep.add_argument(
        "--receptor",
        required=True,
        metavar="PDB",
        help="Receptor PDB file to prepare.",
    )
    p_prep.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory to write receptor.pdbqt.",
    )

    # benchmark subparser
    p_bench = sub.add_parser(
        "benchmark",
        help="Run benchmark suite against reference complexes (Phase 8).",
    )
    p_bench.add_argument(
        "--test-csv",
        required=True,
        metavar="CSV",
        help="Test complexes CSV (columns: pdb_id, peptide_sequence, experimental_pkd).",
    )
    p_bench.add_argument(
        "--baselines",
        default=None,
        metavar="SCORERS",
        help="Comma-separated baseline scorers to compare (e.g., vina,adcp,rapidock).",
    )
    p_bench.add_argument(
        "--report",
        default=None,
        metavar="MD",
        help="Path to write benchmark report in Markdown format.",
    )

    return parser


def _run_dock(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate inputs and orchestrate the full docking pipeline.

    Args:
        args: Parsed CLI arguments from the dock subcommand.
        parser: Root ArgumentParser for calling parser.error() (exits with code 2).
    """
    from pydantic import ValidationError

    # Mutual exclusion: --input-poses and --n-samples may not both be specified
    n_samples_explicit = args.n_samples is not None
    if args.input_poses is not None and n_samples_explicit:
        parser.error(
            "--input-poses and --n-samples are mutually exclusive. "
            "Use --input-poses to skip Stage 1 (required on macOS), "
            "or omit it to run RAPiDock."
        )
    n_samples = args.n_samples if args.n_samples is not None else 100

    # DockConfig is the single validation gate — raises ValidationError on bad input
    try:
        config = DockConfig(
            peptide_sequence=args.peptide,
            receptor_path=Path(args.receptor).resolve(),
            site_coords=(args.site[0], args.site[1], args.site[2]),
            box_size=args.box,
            n_samples=n_samples,
            seed=args.seed,
            scoring=set(args.scoring.split(",")),
            output_dir=Path(args.output_dir).resolve(),
            verbosity=args.verbose,
        )
    except ValidationError as exc:
        parser.error(str(exc))
        return  # unreachable — parser.error() exits; satisfies type checker

    input_poses_dir: Path | None = (
        Path(args.input_poses).resolve() if args.input_poses else None
    )
    calibration_path = Path(args.calibration).resolve()

    if args.refine_topk is not None:
        logger.info(
            "--refine-topk %d noted; MM-GBSA refinement is v2 scope and will not run.",
            args.refine_topk,
        )

    from hybridock_pep import driver  # imported late — driver.py is Wave 2 scope
    scored_poses, _cluster_result = driver.run_dock(
        config=config,
        input_poses_dir=input_poses_dir,
        calibration_path=calibration_path,
    )
    logger.info("Docking complete. %d poses scored.", len(scored_poses))


def _run_calibrate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Dispatch to calibrate_alpha.main() as a module call.

    Args:
        args: Parsed CLI arguments from the calibrate subcommand.
        parser: Root ArgumentParser (unused; present for dispatch signature consistency).
    """
    import sys as _sys
    from pathlib import Path as _Path

    scripts_dir = str(_Path(__file__).resolve().parents[2] / "scripts")
    if scripts_dir not in _sys.path:
        _sys.path.insert(0, scripts_dir)
    import calibrate_alpha  # type: ignore[import]

    ns = argparse.Namespace(
        training_csv=Path(args.training_csv),
        scores_json=Path(args.scores_json),
        output=Path(args.output),
        verbose=args.verbose > 0,
    )
    calibrate_alpha.main(ns)


def _run_prep(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Prepare receptor PDBQT using a minimal DockConfig.

    Args:
        args: Parsed CLI arguments from the prep subcommand.
        parser: Root ArgumentParser for calling parser.error() on validation failure.
    """
    from pydantic import ValidationError
    from hybridock_pep.prep.receptor import prepare_receptor

    try:
        config = DockConfig(
            peptide_sequence="A",
            receptor_path=Path(args.receptor).resolve(),
            site_coords=(0.0, 0.0, 0.0),
            box_size=20.0,
            output_dir=Path(args.output_dir).resolve(),
        )
    except ValidationError as exc:
        parser.error(str(exc))
        return

    pdbqt_path = prepare_receptor(config)
    logger.info("Receptor prepared: %s", pdbqt_path)


def _run_benchmark(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Benchmark subcommand stub — Phase 8 scope.

    Args:
        args: Parsed CLI arguments from the benchmark subcommand.
        parser: Root ArgumentParser (unused).
    """
    raise NotImplementedError("benchmark: Phase 8 scope")


def main() -> None:
    """hybridock-pep CLI entry point.

    Parses arguments, configures logging, then dispatches to the appropriate
    subcommand handler. Input validation (peptide sequence, receptor path,
    mutual exclusions) is performed before any subprocess is spawned.
    """
    parser = _build_parser()
    args = parser.parse_args()

    if args.verbose == 0:
        log_level = logging.INFO
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")

    dispatch = {
        "dock": _run_dock,
        "calibrate": _run_calibrate,
        "prep": _run_prep,
        "benchmark": _run_benchmark,
    }
    if args.command is None:
        parser.print_help()
        return
    dispatch[args.command](args, parser)
