"""
Benchmark: HybriDock-Pep vs DiffPepDock
========================================
Runs both tools on a set of benchmark protein–peptide complexes and
compares top-1 Cα RMSD (vs crystal pose) and wallclock time.

Usage:
    # From score-env, with diffpepdock env on PATH:
    python scripts/benchmark_vs_diffpepdock.py \
        --out-dir runs/benchmark_diffpepdock \
        [--n-samples 50]

Benchmark complexes (all redocking, crystal receptor + crystal peptide seq):
    1YCR  MDM2 / p53   Chain A (receptor), Chain B (SQETFSDLWKLPEN, 15-mer)
    3EQS  MDM2-like     Chain A (receptor), Chain B (TSFAEYWNLLSP,   12-mer)

Metrics computed per complex per tool:
    top1_rmsd    — Cα RMSD of best-scored pose vs crystal peptide
    topN_best    — best Cα RMSD among all generated poses
    wallclock_s  — total inference + scoring time (seconds)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio import PDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")

# ─── repo-relative paths ──────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
DIFFPEPDOCK_DIR = REPO / "third_party" / "DiffPepDock"
CHECKPOINT = DIFFPEPDOCK_DIR / "experiments" / "checkpoints" / "diffpepdock_v1.pth"
DIFFPEPDOCK_PYTHON = Path(
    os.environ.get(
        "DIFFPEPDOCK_PYTHON",
        "/home/igem/miniconda3/envs/diffpepdock/bin/python",
    )
)
HYBRIDOCK_PYTHON = Path(
    os.environ.get(
        "HYBRIDOCK_PYTHON",
        "/home/igem/miniconda3/envs/score-env/bin/python",
    )
)

# ─── benchmark registry ───────────────────────────────────────────────────────
BENCHMARKS = [
    {
        "id": "1YCR",
        "pdb": str(REPO / "data" / "benchmark_pdbs" / "1YCR.pdb"),
        "receptor_chain": "A",
        "ligand_chain": "B",
        "peptide_seq": "SQETFSDLWKLPEN",
        "experimental_pkd": 6.22,   # K_d ≈ 0.6 µM
        "site_note": "MDM2/p53 helix binding groove",
    },
    {
        "id": "3EQS",
        "pdb": str(REPO / "data" / "benchmark_pdbs" / "3EQS.pdb"),
        "receptor_chain": "A",
        "ligand_chain": "B",
        "peptide_seq": "TSFAEYWNLLSP",
        "experimental_pkd": 8.48,
        "site_note": "MDM2 phage-display peptide",
    },
]

# ─── PDB utilities ────────────────────────────────────────────────────────────

def extract_chain(pdb_path: str, chain_id: str, out_path: str) -> None:
    """Write a single chain from a PDB to a new file."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_path)
    io = PDB.PDBIO()
    io.set_structure(structure)

    class ChainSelect(PDB.Select):
        def accept_chain(self, chain):
            return chain.get_id() == chain_id

    io.save(out_path, ChainSelect())


def get_ca_coords(pdb_path: str, chain_id: str | None = None) -> np.ndarray:
    """Return Cα coordinates (N, 3) for a chain (or all chains if None)."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_path)
    coords = []
    for model in structure:
        for chain in model:
            if chain_id and chain.get_id() != chain_id:
                continue
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].get_vector().get_array())
    return np.array(coords)


def binding_site_center(pdb_path: str, ligand_chain: str) -> np.ndarray:
    """Compute centroid of the crystal ligand Cα atoms."""
    ca = get_ca_coords(pdb_path, ligand_chain)
    if len(ca) == 0:
        raise ValueError(f"No Cα atoms found in chain {ligand_chain} of {pdb_path}")
    return ca.mean(axis=0)


def ca_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Compute Cα RMSD between two conformations (must have same length)."""
    if len(coords_a) != len(coords_b):
        raise ValueError(
            f"RMSD length mismatch: {len(coords_a)} vs {len(coords_b)}"
        )
    diff = coords_a - coords_b
    return float(np.sqrt((diff**2).sum(axis=1).mean()))


def parse_ranked_poses(ranked_csv: str, poses_dir: str) -> list[tuple[float, str]]:
    """
    Read HybriDock-Pep ranked_poses.csv → [(score, pdb_path), ...] sorted by score.
    """
    df = pd.read_csv(ranked_csv)
    results = []
    for _, row in df.iterrows():
        pdb_file = row.get("pose_file") or row.get("pdb_path") or ""
        if not os.path.isabs(pdb_file):
            pdb_file = str(Path(poses_dir) / pdb_file)
        score = float(row.get("score_corrected", row.get("score_vina", 0.0)))
        results.append((score, pdb_file))
    results.sort(key=lambda x: x[0])
    return results

# ─── DiffPepDock runner ───────────────────────────────────────────────────────

def run_diffpepdock(
    bench: dict,
    work_dir: Path,
    n_samples: int = 32,
    denoising_steps: int = 200,
) -> dict[str, Any]:
    """
    Run DiffPepDock on a single benchmark complex.
    Returns metrics dict with top1_rmsd, topN_best, wallclock_s.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    pdb_id = bench["id"]
    pdb_src = bench["pdb"]

    # ── prepare receptor + peptide inputs ──
    # DiffPepDock wants the complex PDB (with lig chain) + peptide FASTA
    shutil.copy(pdb_src, work_dir / f"{pdb_id}.pdb")

    fasta_path = work_dir / "peptide.fasta"
    fasta_path.write_text(f">{pdb_id}_pep\n{bench['peptide_seq']}\n")

    receptor_info = {
        pdb_id: {
            "lig_chain": bench["ligand_chain"],
        }
    }
    info_path = work_dir / "docking_cases.json"
    info_path.write_text(json.dumps(receptor_info, indent=2))

    processed_dir = work_dir / "processed"
    processed_dir.mkdir(exist_ok=True)

    # ── preprocess ──
    log.info("[DiffPepDock] %s: running process_batch_dock.py", pdb_id)
    preproc_cmd = [
        str(DIFFPEPDOCK_PYTHON),
        str(DIFFPEPDOCK_DIR / "experiments" / "process_batch_dock.py"),
        "--pdb_dir", str(work_dir),
        "--write_dir", str(processed_dir),
        "--receptor_info_path", str(info_path),
        "--peptide_seq_path", str(fasta_path),
    ]
    t0 = time.perf_counter()
    result = subprocess.run(
        preproc_cmd,
        cwd=str(DIFFPEPDOCK_DIR),
        capture_output=True,
        text=True,
        env={**os.environ, "BASE_PATH": str(DIFFPEPDOCK_DIR)},
    )
    if result.returncode != 0:
        log.error("[DiffPepDock] preprocess failed:\n%s", result.stderr[-2000:])
        raise RuntimeError(f"DiffPepDock preprocessing failed for {pdb_id}")

    metadata_csv = processed_dir / "metadata_test.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Expected {metadata_csv} after preprocessing")

    # ── inference ──
    log.info("[DiffPepDock] %s: running inference (%d samples, %d steps)",
             pdb_id, n_samples, denoising_steps)

    run_dir = work_dir / "runs"
    run_dir.mkdir(exist_ok=True)

    infer_cmd = [
        str(DIFFPEPDOCK_PYTHON),
        str(DIFFPEPDOCK_DIR / "experiments" / "run_docking.py"),
        "--config-name", "docking_single_gpu",
        f"data.val_csv_path={metadata_csv}",
        f"data.num_repeat_per_eval_sample={n_samples}",
        f"data.num_t={denoising_steps}",
        f"experiment.eval_ckpt_path={CHECKPOINT}",
        f"experiment.eval_dir={run_dir}",
    ]

    result = subprocess.run(
        infer_cmd,
        cwd=str(DIFFPEPDOCK_DIR),
        capture_output=True,
        text=True,
        env={**os.environ, "BASE_PATH": str(DIFFPEPDOCK_DIR)},
    )
    wallclock = time.perf_counter() - t0

    if result.returncode != 0:
        log.error("[DiffPepDock] inference failed:\n%s", result.stderr[-2000:])
        log.error("[DiffPepDock] stdout:\n%s", result.stdout[-1000:])
        raise RuntimeError(f"DiffPepDock inference failed for {pdb_id}")

    # ── parse output poses ──
    # DiffPepDock writes: {eval_dir}/docking/<date_stamp>/<pdb_id>/<pep_id>/<pdb_id>_<pep_id>_sample_N.pdb
    # Each output PDB contains Chain A = peptide, Chain B = receptor pocket.
    pose_pdbs = sorted(run_dir.glob("**/*.pdb"))
    log.info("[DiffPepDock] %s: found %d pose PDBs", pdb_id, len(pose_pdbs))

    crystal_ca = get_ca_coords(pdb_src, bench["ligand_chain"])
    n_crystal = len(crystal_ca)
    log.info("[DiffPepDock] %s: crystal peptide has %d residues", pdb_id, n_crystal)

    rmsds = []
    for pose_path in pose_pdbs:
        try:
            # DiffPepDock output: Chain A = peptide, Chain B = receptor
            pose_ca = get_ca_coords(str(pose_path), chain_id="A")
            if len(pose_ca) == 0:
                # Fall back to all chains if single-chain output
                pose_ca = get_ca_coords(str(pose_path))
            mn = min(len(pose_ca), n_crystal)
            rmsds.append(ca_rmsd(pose_ca[:mn], crystal_ca[:mn]))
        except Exception as exc:
            log.warning("[DiffPepDock] could not parse %s: %s", pose_path, exc)

    if not rmsds:
        log.error("[DiffPepDock] %s: no valid RMSD values computed", pdb_id)
        return {"tool": "DiffPepDock", "pdb_id": pdb_id,
                "top1_rmsd": float("nan"), "topN_best": float("nan"),
                "n_poses": 0, "wallclock_s": wallclock}

    # DiffPepDock doesn't score poses — treat first sample as top-1
    # (samples are generated in random order; best-of-N is the fair metric)
    top1_rmsd = rmsds[0]

    return {
        "tool": "DiffPepDock",
        "pdb_id": pdb_id,
        "top1_rmsd": top1_rmsd,
        "topN_best": min(rmsds),
        "n_poses": len(rmsds),
        "wallclock_s": wallclock,
    }


# ─── HybriDock-Pep runner ─────────────────────────────────────────────────────

def run_hybridock(
    bench: dict,
    work_dir: Path,
    n_samples: int = 100,
) -> dict[str, Any]:
    """
    Run HybriDock-Pep on a single benchmark complex.
    Returns metrics dict.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    pdb_id = bench["id"]
    pdb_src = bench["pdb"]

    # Extract receptor chain (no ligand) → used as input PDB
    receptor_pdb = work_dir / f"{pdb_id}_receptor.pdb"
    extract_chain(pdb_src, bench["receptor_chain"], str(receptor_pdb))

    # Binding site center from crystal ligand
    center = binding_site_center(pdb_src, bench["ligand_chain"])
    cx, cy, cz = center

    run_out = work_dir / "hdp_run"

    hybridock_cmd = [
        str(HYBRIDOCK_PYTHON), "-m", "hybridock_pep",
        "dock",
        "--peptide", bench["peptide_seq"],
        "--receptor", str(receptor_pdb),
        "--site", f"{cx:.3f}", f"{cy:.3f}", f"{cz:.3f}",
        "--box", "25",
        "--n-samples", str(n_samples),
        "--scoring", "vina,ad4",
        "--output-dir", str(run_out),
        "--seed", "42",
    ]

    log.info("[HybriDock-Pep] %s: running pipeline", pdb_id)
    t0 = time.perf_counter()
    result = subprocess.run(
        hybridock_cmd,
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    wallclock = time.perf_counter() - t0

    if result.returncode != 0:
        log.error("[HybriDock-Pep] pipeline failed:\n%s", result.stderr[-2000:])
        log.error("[HybriDock-Pep] stdout:\n%s", result.stdout[-1000:])
        raise RuntimeError(f"HybriDock-Pep pipeline failed for {pdb_id}")

    ranked_csv = run_out / "ranked_poses.csv"
    if not ranked_csv.exists():
        raise FileNotFoundError(f"ranked_poses.csv not found at {ranked_csv}")

    crystal_ca = get_ca_coords(pdb_src, bench["ligand_chain"])
    n_crystal = len(crystal_ca)

    # Parse best_pose.pdb for top-1 RMSD
    best_pose_pdb = run_out / "best_pose.pdb"
    rmsds = []

    if best_pose_pdb.exists():
        try:
            best_ca = get_ca_coords(str(best_pose_pdb))
            mn = min(len(best_ca), n_crystal)
            rmsds.append(ca_rmsd(best_ca[:mn], crystal_ca[:mn]))
        except Exception as exc:
            log.warning("[HybriDock-Pep] could not parse best_pose.pdb: %s", exc)

    # Compute RMSD for all poses
    poses_dir = run_out / "poses"
    for pose_pdb in sorted(poses_dir.glob("pose_*.pdb")) if poses_dir.exists() else []:
        try:
            pose_ca = get_ca_coords(str(pose_pdb))
            mn = min(len(pose_ca), n_crystal)
            rmsds.append(ca_rmsd(pose_ca[:mn], crystal_ca[:mn]))
        except Exception:
            pass

    top1_rmsd = rmsds[0] if rmsds else float("nan")
    topN_best = min(rmsds) if rmsds else float("nan")

    return {
        "tool": "HybriDock-Pep",
        "pdb_id": pdb_id,
        "top1_rmsd": top1_rmsd,
        "topN_best": topN_best,
        "n_poses": len(rmsds),
        "wallclock_s": wallclock,
    }


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark HybriDock-Pep vs DiffPepDock"
    )
    parser.add_argument(
        "--out-dir", default="runs/benchmark_diffpepdock",
        help="Root directory for all benchmark outputs",
    )
    parser.add_argument(
        "--n-samples", type=int, default=50,
        help="Number of poses to generate per tool (default 50; use 100 for production)",
    )
    parser.add_argument(
        "--tools", nargs="+",
        choices=["hybridock", "diffpepdock", "both"],
        default=["both"],
        help="Which tools to run",
    )
    parser.add_argument(
        "--complexes", nargs="+",
        choices=["1YCR", "3EQS", "all"],
        default=["all"],
        help="Which benchmark complexes to run",
    )
    parser.add_argument(
        "--skip-if-done", action="store_true",
        help="Skip a complex+tool combination if results already exist",
    )
    args = parser.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run_both = "both" in args.tools
    run_hybridock = run_both or "hybridock" in args.tools
    run_diffpepdock_flag = run_both or "diffpepdock" in args.tools
    run_all = "all" in args.complexes

    benchmarks = [b for b in BENCHMARKS if run_all or b["id"] in args.complexes]

    if run_diffpepdock_flag:
        if not CHECKPOINT.exists():
            log.error(
                "DiffPepDock weights not found at %s. "
                "Run: wget https://zenodo.org/records/15398020/files/diffpepdock_v1.pth "
                "-O %s",
                CHECKPOINT, CHECKPOINT,
            )
            sys.exit(1)
        if not DIFFPEPDOCK_PYTHON.exists():
            log.error(
                "diffpepdock Python not found at %s. "
                "Create the env: conda env create -f envs/diffpepdock-env.yml",
                DIFFPEPDOCK_PYTHON,
            )
            sys.exit(1)

    results = []

    for bench in benchmarks:
        pdb_id = bench["id"]
        log.info("=== Complex %s ===", pdb_id)

        if run_hybridock:
            hdp_dir = out_dir / pdb_id / "hybridock"
            done_flag = hdp_dir / ".done"
            if args.skip_if_done and done_flag.exists():
                log.info("[HybriDock-Pep] %s: skipping (already done)", pdb_id)
                cached = json.loads((hdp_dir / "metrics.json").read_text())
                results.append(cached)
            else:
                try:
                    metrics = run_hybridock(bench, hdp_dir, n_samples=args.n_samples)
                    (hdp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
                    done_flag.touch()
                    results.append(metrics)
                    log.info("[HybriDock-Pep] %s done — top1 RMSD=%.2f Å  topN=%.2f Å  %.0fs",
                             pdb_id, metrics["top1_rmsd"], metrics["topN_best"],
                             metrics["wallclock_s"])
                except Exception as exc:
                    log.error("[HybriDock-Pep] %s FAILED: %s", pdb_id, exc)
                    results.append({
                        "tool": "HybriDock-Pep", "pdb_id": pdb_id,
                        "error": str(exc),
                        "top1_rmsd": float("nan"), "topN_best": float("nan"),
                        "n_poses": 0, "wallclock_s": float("nan"),
                    })

        if run_diffpepdock_flag:
            dpd_dir = out_dir / pdb_id / "diffpepdock"
            done_flag = dpd_dir / ".done"
            if args.skip_if_done and done_flag.exists():
                log.info("[DiffPepDock] %s: skipping (already done)", pdb_id)
                cached = json.loads((dpd_dir / "metrics.json").read_text())
                results.append(cached)
            else:
                try:
                    metrics = run_diffpepdock(
                        bench, dpd_dir,
                        n_samples=args.n_samples,
                        denoising_steps=200,
                    )
                    (dpd_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
                    done_flag.touch()
                    results.append(metrics)
                    log.info("[DiffPepDock] %s done — top1 RMSD=%.2f Å  topN=%.2f Å  %.0fs",
                             pdb_id, metrics["top1_rmsd"], metrics["topN_best"],
                             metrics["wallclock_s"])
                except Exception as exc:
                    log.error("[DiffPepDock] %s FAILED: %s", pdb_id, exc)
                    results.append({
                        "tool": "DiffPepDock", "pdb_id": pdb_id,
                        "error": str(exc),
                        "top1_rmsd": float("nan"), "topN_best": float("nan"),
                        "n_poses": 0, "wallclock_s": float("nan"),
                    })

    # ── summary table ──
    df = pd.DataFrame(results)
    summary_path = out_dir / "comparison_summary.csv"
    df.to_csv(summary_path, index=False)
    log.info("\n\nRESULTS:\n%s", df.to_string(index=False))

    # pretty Markdown table
    md_lines = ["# DiffPepDock vs HybriDock-Pep Benchmark", "",
                "| Complex | Tool | Top-1 RMSD (Å) | Top-N Best (Å) | Poses | Time (s) |",
                "|---------|------|---------------|----------------|-------|----------|"]
    for _, row in df.iterrows():
        md_lines.append(
            f"| {row['pdb_id']} | {row['tool']} "
            f"| {row['top1_rmsd']:.2f} | {row['topN_best']:.2f} "
            f"| {int(row['n_poses'])} | {row['wallclock_s']:.0f} |"
        )
    md_text = "\n".join(md_lines)
    report_path = out_dir / "comparison_report.md"
    report_path.write_text(md_text)
    log.info("Report written to %s", report_path)


if __name__ == "__main__":
    main()
