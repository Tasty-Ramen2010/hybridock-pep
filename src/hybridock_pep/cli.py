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
        default="vina",
        metavar="BACKENDS",
        help=(
            "Force-field backends to run (default: vina). The headline ΔG comes from "
            "the AI-pose affinity model, NOT from Vina/AD4 — Vina is retained only for "
            "clash relief on RAPiDock poses, and its score is raw telemetry. AD4 is "
            "off by default (production ridge gives w_ad4=0). Pass --scoring vina,ad4 "
            "to additionally run autogrid4 + AD4 for research/telemetry."
        ),
    )
    p_dock.add_argument(
        "--refine-topk",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Run MM-GBSA (AMBER ff14SB + GBn2) on the top-K cluster representative poses "
            "after hybrid scoring. Re-ranks those K poses by ΔG_bind. Requires OpenMM. "
            "Runs on GPU (CUDA) by default; use --mmgbsa-cpu-only to force CPU."
        ),
    )
    p_dock.add_argument(
        "--ultra",
        type=int,
        nargs="?",
        const=32,
        default=0,
        metavar="K",
        help=(
            "Ultra ranking mode: compute rank_score as the mean of K feature-jittered evaluations "
            "(randomized smoothing, E314). Reduces within-target ranking variance (~+2 pts pairwise) at "
            "~K× the scoring cost. Bare --ultra uses K=32. Does NOT improve absolute-ΔG accuracy — it "
            "refines the rank_score ordering only."
        ),
    )
    p_dock.add_argument(
        "--mmgbsa-cpu-only",
        action="store_true",
        default=False,
        help=(
            "Force MM-GBSA refinement to use OpenMM CPU platform instead of CUDA/OpenCL. "
            "Slower (~30–60 s/pose) but avoids GPU driver issues. "
            "Has no effect unless --refine-topk is set."
        ),
    )
    p_dock.add_argument(
        "--mmgbsa-ie", action="store_true", default=False,
        help="Add the signed Interaction-Entropy −TΔS term to MM-GBSA ΔG (short "
             "trajectory per pose). Targets the conformational entropy that "
             "dominates flexible-peptide binding. Requires --refine-topk.",
    )
    p_dock.add_argument(
        "--mmgbsa-3traj", action="store_true", default=False,
        help="Three-trajectory MM-GBSA: relax the unbound peptide/receptor "
             "separately instead of reading them from the bound geometry "
             "(removes the disorder bias for floppy peptides). Requires --refine-topk.",
    )
    p_dock.add_argument(
        "--mmgbsa-dielectric", type=float, default=1.0, metavar="EPS",
        help="GB internal dielectric εin for MM-GBSA (default 1.0; the PepSet "
             "screen did not support raising it — see docs/scoring_overhaul_plan.md).",
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
        default="data/calibration_v1_2_production_entropy.json",
        metavar="JSON",
        help="Path to calibration JSON for entropy correction "
             "(default: data/calibration_v1_2_production_entropy.json; "
             "v1.4 was reverted — LOO r=0.30 vs v1.2's 0.72, see "
             "docs/scoring_overhaul_plan.md).",
    )
    p_dock.add_argument(
        "--no-minimize",
        action="store_true",
        default=False,
        help=(
            "Disable OpenMM energy minimization of RAPiDock poses before scoring. "
            "Minimization relieves intra-pose clashes that cause AD4 anomalous scores. "
            "Has no effect when --input-poses is set (pre-generated poses are not minimized)."
        ),
    )
    p_dock.add_argument(
        "--ensemble",
        action="store_true",
        default=False,
        help=(
            "Compute the geometry+Vina ensemble ΔG (kcal/mol) per pose: pocket+interface+MJ "
            "per-contact-energy linear model z-blended with Vina (scoring/ensemble.py). "
            "Validated to beat Vina-alone on real RAPiDock poses (docs E22/E24). "
            "Writes the ensemble_dg column to ranked_poses.csv."
        ),
    )
    p_dock.add_argument(
        "--ensemble-calibration",
        type=Path,
        default=None,
        metavar="JSON",
        help="Override the ensemble calibration JSON (default: data/ensemble_calibration.json).",
    )
    p_dock.add_argument(
        "--free-entropy",
        action="store_true",
        default=False,
        help=(
            "Add the free-state conformational entropy feature to the ensemble (scoring/"
            "free_entropy.py): ~8 s/pose GPU free-peptide MD measuring how much entropy the "
            "peptide loses on binding. Validated to lift cross-target r (docs E40). Requires "
            "--ensemble and a calibration that includes s_free_bur."
        ),
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
        default="data/calibration_v1_4_balanced.json",
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
    p_bench.add_argument(
        "--output-dir",
        dest="output_dir",
        default="runs/benchmark",
        metavar="DIR",
        help="Directory for per-complex run output (default: runs/benchmark/).",
    )
    p_bench.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="Random seed for dock runs (default: 42).",
    )
    p_bench.add_argument(
        "--box-size",
        dest="box_size",
        type=float,
        default=40.0,
        metavar="ANG",
        help="Grid box edge length in Angstroms (default: 40.0). Larger than dock default to cover RAPiDock prediction variance.",
    )
    p_bench.add_argument(
        "--n-samples",
        dest="n_samples",
        type=int,
        default=100,
        metavar="N",
        help="RAPiDock sampling passes per complex (default: 100).",
    )
    p_bench.add_argument(
        "--calibration",
        default="data/calibration_v1_4_balanced.json",
        metavar="JSON",
        help="Path to calibration.json (default: data/calibration.json).",
    )

    # reproducibility subparser — multi-seed top-1 pose agreement
    p_rep = sub.add_parser(
        "reproducibility",
        help="Run the pipeline K times with different seeds; report pose agreement.",
    )
    p_rep.add_argument("--peptide", required=True, metavar="SEQ")
    p_rep.add_argument("--receptor", required=True, metavar="PDB")
    p_rep.add_argument("--site", required=True, nargs=3, type=float,
                       metavar=("X", "Y", "Z"))
    p_rep.add_argument("--box", required=True, type=float, metavar="ANG")
    p_rep.add_argument("--seeds", required=True, type=int, nargs="+",
                       metavar="N",
                       help="Two or more integer seeds (e.g. --seeds 1 2 3).")
    p_rep.add_argument("--n-samples", type=int, default=100, metavar="N")
    p_rep.add_argument("--scoring", default="vina", metavar="BACKENDS")
    p_rep.add_argument("--calibration",
                       default="data/calibration_v1_2_production_entropy.json",
                       metavar="JSON")
    p_rep.add_argument("--output-dir", required=True, metavar="DIR",
                       help="Parent dir; seed_N/ subdirs are created per seed.")

    # selectivity subparser — decoy ΔΔG between two receptors for one peptide
    p_sel = sub.add_parser(
        "selectivity",
        help="Run pipeline on two receptors and report ΔΔG with bootstrap CI.",
    )
    p_sel.add_argument("--peptide", required=True, metavar="SEQ",
                       help="Peptide sequence (single-letter).")
    p_sel.add_argument("--target-receptor", required=True, metavar="PDB",
                       help="On-target receptor PDB.")
    p_sel.add_argument("--target-site", required=True, nargs=3, type=float,
                       metavar=("X", "Y", "Z"),
                       help="Target grid box center coords (Å).")
    p_sel.add_argument("--target-box", required=True, type=float, metavar="ANG",
                       help="Target grid box edge length (Å).")
    p_sel.add_argument("--offtarget-receptor", required=True, metavar="PDB",
                       help="Off-target receptor PDB.")
    p_sel.add_argument("--offtarget-site", required=True, nargs=3, type=float,
                       metavar=("X", "Y", "Z"),
                       help="Off-target grid box center coords (Å).")
    p_sel.add_argument("--offtarget-box", required=True, type=float, metavar="ANG",
                       help="Off-target grid box edge length (Å).")
    p_sel.add_argument("--n-samples", type=int, default=100, metavar="N",
                       help="RAPiDock samples per receptor (default: 100).")
    p_sel.add_argument("--top-k", type=int, default=10, metavar="K",
                       help="Top-K poses per side fed to ΔΔG (default: 10).")
    p_sel.add_argument("--bootstrap", type=int, default=1000, metavar="N",
                       help="Bootstrap iterations for ΔΔG 95%% CI (default: 1000).")
    p_sel.add_argument("--seed", type=int, default=None, metavar="N",
                       help="RNG seed (CUDA + bootstrap).")
    p_sel.add_argument("--scoring", default="vina", metavar="BACKENDS",
                       help="Scoring backends (default: vina). Same flag as dock.")
    p_sel.add_argument("--calibration",
                       default="data/calibration_v1_2_production_entropy.json", metavar="JSON",
                       help="Calibration JSON used on both sides (default: v1.2; v1.4 reverted).")
    p_sel.add_argument("--output-dir", required=True, metavar="DIR",
                       help="Parent dir; target/ and offtarget/ subdirs are created.")
    p_sel.add_argument("--input-poses-target", default=None, metavar="DIR",
                       help="Optional pre-generated target poses (skip Stage 1).")
    p_sel.add_argument("--input-poses-offtarget", default=None, metavar="DIR",
                       help="Optional pre-generated off-target poses (skip Stage 1).")
    p_sel.add_argument("--no-minimize", action="store_true", default=False,
                       help="Disable OpenMM minimization on both sides.")
    p_sel.add_argument("--score-field", default="auto",
                       choices=["auto", "mmgbsa_dg", "vina_score", "hybrid_score"],
                       help="Score used for ΔΔG. 'auto' = MM-GBSA if --refine-topk ran, "
                            "else Vina. The entropy-corrected hybrid is a poor selectivity "
                            "signal (it cancels in ΔΔG) and is only allowed if forced.")

    # ----- crystal-score: score an EXISTING crystal complex (no docking) -----
    p_cry = sub.add_parser(
        "crystal-score",
        help="Score a crystal-quality pose with the crystal-tuned model (geometry + interaction map).",
    )
    p_cry.add_argument("--receptor", required=True, metavar="PDB",
                       help="Receptor PDB (protein only).")
    p_cry.add_argument("--peptide-pdb", required=True, metavar="PDB",
                       help="Peptide pose PDB (the bound peptide; a crystal/native pose).")
    p_cry.add_argument("--peptide", required=True, metavar="SEQ",
                       help="Peptide sequence (single-letter codes).")
    p_cry.add_argument("--artifact", default=None, metavar="JOBLIB",
                       help="Override the crystal-IFP model artifact "
                            "(default: data/affinity_crystal_ifp.joblib).")
    p_cry.add_argument("--allow-clashes", action="store_true",
                       help="Score even if the pose sterically clashes with the receptor. By default a "
                            "physically invalid (overlapping) pose is refused, because the featurizer "
                            "scores atom overlaps as strong contacts and would return a confident but "
                            "meaningless ΔG.")

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
            minimize_poses=not args.no_minimize,
            refine_topk=args.refine_topk,
            ultra=args.ultra,
            mmgbsa_cpu_only=args.mmgbsa_cpu_only,
            mmgbsa_include_ie=args.mmgbsa_ie,
            mmgbsa_3traj=args.mmgbsa_3traj,
            mmgbsa_solute_dielectric=args.mmgbsa_dielectric,
            compute_ensemble=args.ensemble,
            ensemble_calibration=(
                Path(args.ensemble_calibration).resolve()
                if args.ensemble_calibration else None
            ),
            compute_free_entropy=args.free_entropy,
        )
    except ValidationError as exc:
        parser.error(str(exc))
        return  # unreachable — parser.error() exits; satisfies type checker

    input_poses_dir: Path | None = (
        Path(args.input_poses).resolve() if args.input_poses else None
    )
    calibration_path = Path(args.calibration).resolve()

    from hybridock_pep import driver
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


def _run_reproducibility(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Run dock pipeline K times with different seeds, report top-1 pose agreement."""
    import json
    from pydantic import ValidationError
    from hybridock_pep.reproducibility import run_reproducibility

    if len(args.seeds) < 2:
        parser.error("--seeds requires ≥2 integers")
    out_root = Path(args.output_dir).resolve()

    try:
        cfg = DockConfig(
            peptide_sequence=args.peptide,
            receptor_path=Path(args.receptor).resolve(),
            site_coords=(args.site[0], args.site[1], args.site[2]),
            box_size=args.box,
            n_samples=args.n_samples,
            scoring=set(args.scoring.split(",")),
            output_dir=out_root,
            verbosity=args.verbose,
        )
    except ValidationError as exc:
        parser.error(str(exc))
        return

    result = run_reproducibility(
        base_config=cfg,
        calibration_path=Path(args.calibration).resolve(),
        seeds=args.seeds,
    )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "reproducibility.json").write_text(json.dumps(result.to_json(), indent=2))
    logger.info(
        "Reproducibility: mean RMSD=%.2fÅ  pearson=%.3f  ΔG σ=%.2f kcal/mol  → %s",
        result.mean_pairwise_rmsd, result.mean_pairwise_pearson,
        result.dg_std, result.verdict,
    )


def _run_selectivity(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Run the dock pipeline on two receptors and report ΔΔG with bootstrap CI."""
    from pydantic import ValidationError
    from hybridock_pep.selectivity import run_selectivity

    out_root = Path(args.output_dir).resolve()
    target_out = out_root / "target"
    offtarget_out = out_root / "offtarget"

    try:
        target_cfg = DockConfig(
            peptide_sequence=args.peptide,
            receptor_path=Path(args.target_receptor).resolve(),
            site_coords=(args.target_site[0], args.target_site[1], args.target_site[2]),
            box_size=args.target_box,
            n_samples=args.n_samples,
            seed=args.seed,
            scoring=set(args.scoring.split(",")),
            output_dir=target_out,
            verbosity=args.verbose,
            minimize_poses=not args.no_minimize,
        )
        offtarget_cfg = DockConfig(
            peptide_sequence=args.peptide,
            receptor_path=Path(args.offtarget_receptor).resolve(),
            site_coords=(args.offtarget_site[0], args.offtarget_site[1], args.offtarget_site[2]),
            box_size=args.offtarget_box,
            n_samples=args.n_samples,
            seed=args.seed,
            scoring=set(args.scoring.split(",")),
            output_dir=offtarget_out,
            verbosity=args.verbose,
            minimize_poses=not args.no_minimize,
        )
    except ValidationError as exc:
        parser.error(str(exc))
        return

    result = run_selectivity(
        peptide=args.peptide,
        target_config=target_cfg,
        offtarget_config=offtarget_cfg,
        calibration_path=Path(args.calibration).resolve(),
        top_k=args.top_k,
        bootstrap_n=args.bootstrap,
        seed=args.seed,
        input_poses_target=Path(args.input_poses_target).resolve() if args.input_poses_target else None,
        input_poses_offtarget=Path(args.input_poses_offtarget).resolve() if args.input_poses_offtarget else None,
        score_field=args.score_field,
    )
    logger.info(
        "Selectivity complete: ΔΔG=%+.2f kcal/mol  CI95=[%+.2f, %+.2f]  →  %s",
        result.ddg, result.ddg_ci_low, result.ddg_ci_high,
        result.to_json()["interpretation"],
    )


def _run_benchmark(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Dispatch benchmark subcommand to scripts/benchmark.py:main().

    Dynamically injects scripts/ onto sys.path and calls benchmark.main()
    with an argparse.Namespace forwarding all relevant flags. Mirrors the
    _run_calibrate() pattern.

    Args:
        args: Parsed CLI arguments from the benchmark subcommand.
        parser: Root ArgumentParser (unused; present for dispatch signature consistency).
    """
    import sys as _sys
    from pathlib import Path as _Path

    scripts_dir = str(_Path(__file__).resolve().parents[2] / "scripts")
    if scripts_dir not in _sys.path:
        _sys.path.insert(0, scripts_dir)
    import benchmark  # type: ignore[import]

    ns = argparse.Namespace(
        test_csv=_Path(args.test_csv),
        meta_csv=_Path("data/test_complexes_meta.csv"),
        output_dir=_Path(args.output_dir),
        seed=args.seed,
        box_size=args.box_size,
        n_samples=args.n_samples,
        calibration=_Path(args.calibration),
        verbose=args.verbose > 0,
    )
    benchmark.main(ns)


def _run_crystal_score(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Score a single crystal-quality complex with the crystal-tuned IFP model (no docking).

    This is the standalone crystal scorer: the sibling of the AI-pose model, calibrated on
    crystal/native poses. Use it when you already have a high-quality bound pose (a crystal
    structure or an equivalently accurate model) and want the crystal-grade ΔG directly.
    """
    from pathlib import Path

    from hybridock_pep.scoring.interaction_map import score_crystal_complex

    receptor = Path(args.receptor)
    peptide_pdb = Path(args.peptide_pdb)
    for label, path in (("receptor", receptor), ("peptide-pdb", peptide_pdb)):
        if not path.is_file():
            parser.error(f"--{label} not found: {path}")
    seq = args.peptide.strip().upper()
    if not seq or not seq.isalpha():
        parser.error("--peptide must be a non-empty single-letter amino-acid sequence")

    # Guard the common footgun: a PDBQT-derived pose (e.g. dock's best_pose.pdb, written from
    # the Vina-optimized PDBQT) labels residues UNK, which the geometry/IFP featurizer can't read.
    n_unk = sum(1 for ln in peptide_pdb.open() if ln.startswith(("ATOM", "HETATM")) and ln[17:20].strip() == "UNK")
    if n_unk:
        parser.error(
            f"--peptide-pdb has {n_unk} UNK-labelled atoms ({peptide_pdb.name}). crystal-score "
            "needs standard residue names. If this is dock's best_pose.pdb, re-score the original "
            "RAPiDock pose instead (see the 'pose_filename' column in ranked_poses.csv, "
            "e.g. <output-dir>/poses/pose_N.pdb)."
        )

    kwargs = {"artifact": args.artifact} if args.artifact else {}
    try:
        dg = score_crystal_complex(
            str(receptor), str(peptide_pdb), seq, allow_clashes=args.allow_clashes, **kwargs
        )
    except ValueError as exc:
        parser.error(str(exc))
    if dg is None:
        parser.error(
            f"Crystal scoring failed for {peptide_pdb.name}. Either the model artifact is "
            "missing/unloadable, or geometry features couldn't be computed (check the pose has "
            "standard residue names and a real receptor interface). Re-run with -v for details."
        )
    print(f"Crystal ΔG = {dg:.2f} kcal/mol  ({receptor.name} + {peptide_pdb.name}, {len(seq)}-mer)")


def main() -> None:
    """hybridock-pep CLI entry point.

    Parses arguments, configures logging, then dispatches to the appropriate
    subcommand handler. Input validation (peptide sequence, receptor path,
    mutual exclusions) is performed before any subprocess is spawned.
    """
    parser = _build_parser()
    args = parser.parse_args()

    if args.verbose >= 2:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")

    dispatch = {
        "dock": _run_dock,
        "calibrate": _run_calibrate,
        "prep": _run_prep,
        "benchmark": _run_benchmark,
        "selectivity": _run_selectivity,
        "reproducibility": _run_reproducibility,
        "crystal-score": _run_crystal_score,
    }
    if args.command is None:
        parser.print_help()
        return
    dispatch[args.command](args, parser)
