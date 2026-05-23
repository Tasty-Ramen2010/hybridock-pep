"""Benchmark harness: evaluate HybriDock-Pep against 10 held-out peptide-protein complexes.

Runs the full pipeline (Stage 1+2 via CLI) for each complex, then re-runs Vina-only
scoring from the same pose set (--input-poses) for a controlled baseline comparison.
Computes Pearson r for hybrid vs experimental pKd and Vina-only vs experimental pKd.

Usage:
    python scripts/benchmark.py \\
        --test-csv data/test_complexes.csv \\
        --output-dir runs/benchmark/ \\
        --seed 42

Outputs:
    benchmark_report.md  — Markdown table + Pearson r summary + PASS/FAIL
    benchmark_results.csv — raw numbers per complex

Note: Requires score-env (ADFRsuite on PATH, hybridock-pep installed). Not run in CI.
See CLAUDE.md §8 for accuracy targets (Pearson r >= 0.55, +0.10 over Vina-alone).
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

_log = logging.getLogger(__name__)

VALID_STATUSES = {"ok", "skipped_download", "skipped_prep", "skipped_scoring"}
_PDB_ID_RE = re.compile(r"^[0-9][A-Z0-9]{3}$")
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def validate_pdb_id(pdb_id: str) -> bool:
    """Validate a PDB ID matches the RCSB 4-character format.

    Args:
        pdb_id: String to validate.

    Returns:
        True if the ID matches ^[0-9][A-Z0-9]{3}$, False otherwise.
    """
    return bool(_PDB_ID_RE.match(pdb_id))


def get_peptide_center(
    pdb_path: Path, peptide_chain: str
) -> Optional[tuple[float, float, float]]:
    """Compute the geometric centre of Cα atoms from the specified chain.

    Args:
        pdb_path: Path to the PDB file.
        peptide_chain: Chain ID of the peptide in the structure.

    Returns:
        (x, y, z) centroid as floats, or None if no CA atoms found.
    """
    from Bio.PDB import PDBParser  # lazy — biopython in score-env

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("pep", str(pdb_path))
    cas = [
        atom.get_vector().get_array()
        for chain in struct.get_chains()
        if chain.id == peptide_chain
        for res in chain
        for atom in res
        if atom.get_name() == "CA"
    ]
    if not cas:
        return None
    centre = np.mean(cas, axis=0)
    return (float(centre[0]), float(centre[1]), float(centre[2]))


def extract_receptor_chain(pdb_path: Path, receptor_chain: str, dest: Path) -> bool:
    """Write a PDB containing only the specified receptor chain.

    Strips the co-crystallized peptide and any other non-receptor chains so
    that hybridock-pep receives a clean single-chain receptor. Without this,
    the co-crystal peptide occupies the binding site and causes RAPiDock to
    generate clashing or off-site poses.

    Args:
        pdb_path: Path to the full multi-chain RCSB PDB.
        receptor_chain: Chain ID to keep (e.g. "A").
        dest: Destination path for the single-chain PDB.

    Returns:
        True on success, False if no ATOM lines for the chain were found.
    """
    kept: list[str] = []
    for line in pdb_path.read_text().splitlines(keepends=True):
        record = line[:6].strip()
        if record in ("ATOM", "HETATM"):
            chain = line[21] if len(line) > 21 else " "
            if chain != receptor_chain:
                continue
        kept.append(line)
    atom_lines = [l for l in kept if l[:4] == "ATOM"]
    if not atom_lines:
        return False
    dest.write_text("".join(kept))
    return True


def download_pdb(pdb_id: str, dest: Path) -> bool:
    """Download a PDB file from RCSB.

    Args:
        pdb_id: 4-character PDB ID (already validated).
        dest: Destination file path.

    Returns:
        True on success, False on network/HTTP failure.
    """
    url = RCSB_URL.format(pdb_id=pdb_id)
    try:
        urllib.request.urlretrieve(url, str(dest))
        return True
    except Exception as exc:
        _log.warning("Download failed for %s (%s): %s", pdb_id, url, exc)
        return False


def extract_best_score(ranked_csv: Path, column: str) -> Optional[float]:
    """Extract the score of the top-ranked pose from ranked_poses.csv.

    The top-ranked pose is the first row (lowest hybrid_score after sorting).

    Args:
        ranked_csv: Path to ranked_poses.csv written by the dock run.
        column: Column name to extract ("hybrid_score" or "vina_score").

    Returns:
        Float score of the best pose, or None if file missing/empty.
    """
    if not ranked_csv.exists():
        return None
    with ranked_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        return None
    try:
        return float(rows[0][column])
    except (KeyError, ValueError):
        return None


def run_complex(
    row: dict,
    meta: dict,
    args: argparse.Namespace,
    work_dir: Path,
) -> dict:
    """Run the full benchmark pipeline for one complex.

    Implements the two-invocation pattern:
    1. Download PDB from RCSB.
    2. Run hybridock-pep dock --scoring vina,ad4 (full hybrid run).
    3. Run hybridock-pep dock --scoring vina --input-poses <same dir> (vina-only rescore).
    4. Extract hybrid_score from Step 2 output; vina_score from Step 3 output.

    Args:
        row: Dict from test_complexes.csv (pdb_id, peptide_sequence, experimental_pkd).
        meta: Dict from test_complexes_meta.csv (receptor_chain, peptide_chain).
        args: Parsed argparse.Namespace with output_dir, seed, box_size, n_samples.
        work_dir: Absolute path to complex-specific working directory.

    Returns:
        Dict matching benchmark_results.csv schema.
    """
    pdb_id = row["pdb_id"]
    peptide = row["peptide_sequence"]
    exp_pkd = float(row["experimental_pkd"])
    peptide_chain = meta.get("peptide_chain", "B")
    receptor_chain = meta.get("receptor_chain", "A")
    result: dict = {
        "pdb_id": pdb_id,
        "peptide_sequence": peptide,
        "experimental_pkd": exp_pkd,
        "hybrid_score": float("nan"),
        "vina_score": float("nan"),
        "delta_improvement": float("nan"),
        "n_poses": 0,
        "runtime_hybrid_s": 0.0,
        "runtime_vina_s": 0.0,
        "status": "ok",
    }

    # Step 1: Download PDB
    pdb_path = work_dir / f"{pdb_id}.pdb"
    if not download_pdb(pdb_id, pdb_path):
        result["status"] = "skipped_download"
        return result

    # Extract receptor chain only — the full RCSB PDB includes the co-crystallized
    # peptide (chain B) which would occupy the binding site and cause RAPiDock to
    # generate clashing/off-site poses. Pass only chain A to hybridock-pep.
    receptor_pdb = work_dir / f"{pdb_id}_receptor.pdb"
    if not extract_receptor_chain(pdb_path, receptor_chain, receptor_pdb):
        _log.warning("%s: no ATOM lines for chain %s; skipping", pdb_id, receptor_chain)
        result["status"] = "skipped_prep"
        return result

    # Compute binding site from peptide Cα centroid (still use full PDB — has both chains)
    site = get_peptide_center(pdb_path, peptide_chain)
    if site is None:
        _log.warning("%s: no CA atoms found in chain %s; skipping", pdb_id, peptide_chain)
        result["status"] = "skipped_prep"
        return result

    # Step 2: Full hybrid run (vina + ad4)
    hybrid_out = (work_dir / "hybrid").resolve()
    hybrid_out.mkdir(parents=True, exist_ok=True)
    cmd_hybrid = [
        "hybridock-pep", "dock",
        "--peptide", peptide,
        "--receptor", str(receptor_pdb.resolve()),
        "--site", str(site[0]), str(site[1]), str(site[2]),
        "--box", str(args.box_size),
        "--n-samples", str(args.n_samples),
        "--scoring", "vina,ad4",
        "--seed", str(args.seed),
        "--output-dir", str(hybrid_out),
        "--calibration", str(Path(args.calibration).resolve()),
    ]
    _log.info("%s: running hybrid dock", pdb_id)
    t0 = time.monotonic()
    proc = subprocess.run(cmd_hybrid, capture_output=True, text=True)
    result["runtime_hybrid_s"] = round(time.monotonic() - t0, 1)
    if proc.returncode != 0:
        _log.warning(
            "%s: hybrid dock failed (exit %d):\n%s",
            pdb_id, proc.returncode, proc.stderr[-500:],
        )
        result["status"] = "skipped_scoring"
        return result

    # poses_scored/ contains the exact files scored by the hybrid run (minimized where
    # minimization succeeded, original otherwise). Using these for the vina-only
    # rescore ensures the hybrid and vina-only comparisons use identical input poses,
    # eliminating the confound introduced by scoring minimized vs non-minimized poses.
    poses_scored_dir = hybrid_out / "poses_scored"
    poses_dir = poses_scored_dir if poses_scored_dir.exists() else hybrid_out / "poses"
    ranked_hybrid = hybrid_out / "ranked_poses.csv"
    hybrid_score = extract_best_score(ranked_hybrid, "hybrid_score")
    if hybrid_score is None:
        result["status"] = "skipped_scoring"
        return result
    result["hybrid_score"] = hybrid_score

    # Step 3: Vina-only rescore from SAME poses (avoids nondeterminism confound)
    vina_out = (work_dir / "vina_only").resolve()
    vina_out.mkdir(parents=True, exist_ok=True)
    cmd_vina = [
        "hybridock-pep", "dock",
        "--peptide", peptide,
        "--receptor", str(receptor_pdb.resolve()),
        "--site", str(site[0]), str(site[1]), str(site[2]),
        "--box", str(args.box_size),
        "--scoring", "vina",
        "--seed", str(args.seed),
        "--input-poses", str(poses_dir.resolve()),
        "--output-dir", str(vina_out),
        "--calibration", str(Path(args.calibration).resolve()),
    ]
    _log.info("%s: running vina-only rescore from %s", pdb_id, poses_dir.name)
    t1 = time.monotonic()
    proc_v = subprocess.run(cmd_vina, capture_output=True, text=True)
    result["runtime_vina_s"] = round(time.monotonic() - t1, 1)
    if proc_v.returncode != 0:
        _log.warning("%s: vina-only run failed (exit %d)", pdb_id, proc_v.returncode)
        # Keep hybrid score; vina_score stays NaN
        return result

    vina_score = extract_best_score(vina_out / "ranked_poses.csv", "vina_score")
    result["vina_score"] = vina_score if vina_score is not None else float("nan")

    # Count poses (use the same directory that was scored)
    if poses_dir.exists():
        result["n_poses"] = len(list(poses_dir.glob("pose_*.pdb")))

    return result


def write_results_csv(results: list[dict], path: Path) -> None:
    """Write benchmark_results.csv with calibration schema.

    Args:
        results: List of result dicts from run_complex().
        path: Output CSV file path.
    """
    cols = [
        "pdb_id", "peptide_sequence", "experimental_pkd",
        "hybrid_score", "vina_score", "delta_improvement",
        "n_poses", "runtime_hybrid_s", "runtime_vina_s", "status",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    _log.info("Results CSV: %s", path)


def write_report_md(
    results: list[dict],
    r_hybrid: float,
    r_vina: float,
    path: Path,
    args: argparse.Namespace,
) -> None:
    """Write benchmark_report.md with Markdown table and PASS/FAIL summary.

    Args:
        results: Per-complex result dicts.
        r_hybrid: Pearson r for hybrid scores vs experimental pKd.
        r_vina: Pearson r for Vina-only scores vs experimental pKd.
        path: Output Markdown file path.
        args: Parsed args (for run metadata: seed, n_samples, calibration).
    """
    import datetime

    delta = r_hybrid - r_vina
    pass_r = r_hybrid >= 0.55
    pass_delta = delta >= 0.10
    overall = "PASS" if (pass_r and pass_delta) else "FAIL"

    lines = [
        "# HybriDock-Pep Benchmark Report",
        "",
        f"**Date:** {datetime.date.today().isoformat()}",
        f"**Seed:** {args.seed}  **n-samples:** {args.n_samples}"
        f"  **Calibration:** {args.calibration}",
        "",
        "## Per-Complex Results",
        "",
        "| pdb_id | Peptide | Exp. pKd | Hybrid ΔG | Vina-only ΔG | Status |",
        "|--------|---------|----------|-----------|--------------|--------|",
    ]
    for r in results:
        h = r["hybrid_score"]
        v = r["vina_score"]
        hybrid = f"{h:.2f}" if h == h else "N/A"  # NaN check
        vina = f"{v:.2f}" if v == v else "N/A"
        lines.append(
            f"| {r['pdb_id']} | {r['peptide_sequence']} | {r['experimental_pkd']:.2f}"
            f" | {hybrid} | {vina} | {r['status']} |"
        )
    lines += [
        "",
        "## Accuracy Summary",
        "",
        "| Metric | Value | Target | Result |",
        "|--------|-------|--------|--------|",
        f"| Pearson r (hybrid vs exp. pKd) | {r_hybrid:.3f} | ≥ 0.55 | {'PASS' if pass_r else 'FAIL'} |",
        f"| Pearson r (Vina-only vs exp. pKd) | {r_vina:.3f} | — | — |",
        f"| Δ improvement (hybrid − Vina) | {delta:.3f} | ≥ 0.10 | {'PASS' if pass_delta else 'FAIL'} |",
        f"| **Overall** | | | **{overall}** |",
        "",
        "> Scores are negated before correlating with pKd (more negative = stronger binding = higher pKd).",
    ]
    path.write_text("\n".join(lines) + "\n")
    _log.info("Report: %s (%s)", path, overall)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for benchmark.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description=(
            "Benchmark HybriDock-Pep against 10 held-out peptide-protein complexes. "
            "Requires score-env with ADFRsuite on PATH and hybridock-pep installed."
        ),
    )
    parser.add_argument(
        "--test-csv",
        dest="test_csv",
        type=Path,
        required=True,
        metavar="CSV",
        help="Test complexes CSV (columns: pdb_id, peptide_sequence, experimental_pkd).",
    )
    parser.add_argument(
        "--meta-csv",
        dest="meta_csv",
        type=Path,
        default=Path("data/test_complexes_meta.csv"),
        metavar="CSV",
        help="Receptor/peptide chain mapping CSV (default: data/test_complexes_meta.csv).",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path("runs/benchmark"),
        metavar="DIR",
        help="Directory for per-complex run output (default: runs/benchmark/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="Random seed for reproducible dock runs (default: 42).",
    )
    parser.add_argument(
        "--box-size",
        dest="box_size",
        type=float,
        default=40.0,
        metavar="ANG",
        help="Grid box edge length in Angstroms for all complexes (default: 40.0).",
    )
    parser.add_argument(
        "--n-samples",
        dest="n_samples",
        type=int,
        default=500,
        metavar="N",
        help="Number of RAPiDock sampling passes per complex (default: 500).",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("data/calibration.json"),
        metavar="JSON",
        help="Path to calibration.json (default: data/calibration.json).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def main(args: argparse.Namespace | None = None) -> None:
    """Run the full benchmark workflow.

    Args:
        args: Parsed argparse.Namespace. If None, parse_args() is called.

    Raises:
        RuntimeError: If hybridock-pep or prepare_receptor not found on PATH.
        FileNotFoundError: If test_csv does not exist.
    """
    if args is None:
        args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Pre-flight checks — fail fast before iterating complexes
    if not shutil.which("hybridock-pep"):
        raise RuntimeError(
            "hybridock-pep not found on PATH. Install score-env: "
            "conda activate score-env && pip install -e ."
        )
    if not shutil.which("prepare_receptor"):
        raise RuntimeError(
            "prepare_receptor not found on PATH. Install ADFRsuite and add to PATH. "
            "See INSTALL.md Step 3."
        )
    if not args.test_csv.exists():
        raise FileNotFoundError(f"test_csv not found: {args.test_csv}")

    # Load test complexes
    with args.test_csv.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    # Load chain metadata (optional — fallback to chain B for peptide)
    meta_map: dict[str, dict] = {}
    if args.meta_csv.exists():
        with args.meta_csv.open(newline="") as fh:
            for m in csv.DictReader(fh):
                meta_map[m["pdb_id"]] = m
    else:
        _log.warning(
            "meta_csv not found (%s); assuming peptide_chain=B for all", args.meta_csv
        )

    # Validate all PDB IDs before starting any downloads
    for row in rows:
        if not validate_pdb_id(row["pdb_id"]):
            raise ValueError(
                f"Invalid PDB ID in test_csv: {row['pdb_id']!r}. "
                "Expected format: ^[0-9][A-Z0-9]{3}$"
            )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _log.info("Starting benchmark on %d complexes -> %s", len(rows), output_dir)

    # Run each complex
    results = []
    for i, row in enumerate(rows, 1):
        pdb_id = row["pdb_id"]
        _log.info("Complex %s [%d/%d]", pdb_id, i, len(rows))
        work_dir = (output_dir / pdb_id).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        meta = meta_map.get(pdb_id, {"peptide_chain": "B", "receptor_chain": "A"})
        result = run_complex(row, meta, args, work_dir)
        results.append(result)
        _log.info(
            "Complex %s: hybrid=%.2f vina=%.2f status=%s",
            pdb_id,
            result["hybrid_score"],
            result["vina_score"],
            result["status"],
        )

    # Compute Pearson r on successful results only
    from scipy.stats import pearsonr  # lazy — score-env

    ok_results = [r for r in results if r["status"] == "ok"]
    if len(ok_results) >= 2:
        hybrid_scores = [-r["hybrid_score"] for r in ok_results]
        exp_pkds = [r["experimental_pkd"] for r in ok_results]
        r_hybrid, _ = pearsonr(hybrid_scores, exp_pkds)
        vina_ok = [r for r in ok_results if r["vina_score"] == r["vina_score"]]
        if len(vina_ok) >= 2:
            r_vina, _ = pearsonr(
                [-r["vina_score"] for r in vina_ok],
                [r["experimental_pkd"] for r in vina_ok],
            )
        else:
            r_vina = float("nan")
    else:
        _log.warning("Fewer than 2 successful complexes; Pearson r not computed")
        r_hybrid = float("nan")
        r_vina = float("nan")

    # Compute delta_improvement per result
    for r in results:
        h, v = r["hybrid_score"], r["vina_score"]
        if h == h and v == v:  # both non-NaN
            r["delta_improvement"] = v - h  # more negative hybrid = better

    # Write outputs
    results_csv = output_dir / "benchmark_results.csv"
    report_md = output_dir / "benchmark_report.md"
    write_results_csv(results, results_csv)
    write_report_md(results, r_hybrid, r_vina, report_md, args)

    _log.info(
        "Benchmark complete: r_hybrid=%.3f r_vina=%.3f delta=%.3f -> %s",
        r_hybrid,
        r_vina,
        r_hybrid - r_vina,
        "PASS" if (r_hybrid >= 0.55 and (r_hybrid - r_vina) >= 0.10) else "FAIL",
    )


if __name__ == "__main__":
    main()
