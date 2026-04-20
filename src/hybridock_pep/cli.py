from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the hybridock-pep argparse parser with subcommand stubs.

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

    sub.add_parser("dock", help="Run end-to-end docking pipeline (Phase 5).")
    sub.add_parser("calibrate", help="Calibrate entropy correction coefficient alpha (Phase 3).")
    sub.add_parser("benchmark", help="Run benchmark suite against reference complexes (Phase 8).")
    sub.add_parser("prep", help="Prepare receptor/ligand files (Phase 2).")

    return parser


def main() -> None:
    """hybridock-pep CLI entry point.

    Phase 1 wires the subcommand names and entry point. Subcommand bodies
    are implemented in later phases and currently raise via parser.error().
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

    if args.command is None:
        parser.print_help()
        return

    parser.error(f"Subcommand '{args.command}' is not yet implemented (Phase 1 stub).")
