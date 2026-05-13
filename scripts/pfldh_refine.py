"""Local Vina refinement for top-N PfLDH poses.

Takes pre-scored PDBQTs from a pfldh run directory, runs Vina minimize()
(local gradient descent from each RAPiDock starting geometry), and reports:
  - score_only (original)
  - post-minimize score
  - hybrid score under old MDM2 calibration vs new PfLDH calibration
  - AD4 score on the minimized pose

Usage:
    conda run -n score-env python3 scripts/pfldh_refine.py [--run-dir DIR] [--top-n N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from vina import Vina
except ImportError:
    print("ERROR: vina not importable — activate score-env", file=sys.stderr)
    sys.exit(1)

from hybridock_pep.scoring.entropy import (
    count_contact_residues,
    load_calibration,
    load_receptor_heavy_atom_coords,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        default=str(ROOT / "runs" / "pfldh_local_v2"),
        help="Path to a completed hybridock-pep run directory (default: runs/pfldh_local_v2)",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top-ranked poses to refine (default: 5)",
    )
    p.add_argument(
        "--calibration",
        default=str(ROOT / "data" / "pfldh_calibration.json"),
        help="PfLDH calibration JSON (default: data/pfldh_calibration.json)",
    )
    p.add_argument(
        "--old-calibration",
        default=str(ROOT / "data" / "calibration.json"),
        help="MDM2 calibration JSON for comparison (default: data/calibration.json)",
    )
    p.add_argument(
        "--exhaustiveness",
        type=int,
        default=16,
        help="Vina minimize max_evals (gradient steps); analogous to exhaustiveness (default: 16)",
    )
    return p.parse_args()


def score_and_minimize(
    v: Vina,
    pdbqt_path: Path,
    max_evals: int,
    out_dir: Path,
    pose_name: str,
) -> tuple[float, float, Path]:
    """Score, minimize, return (score_before, score_after, minimized_pdbqt_path)."""
    v.set_ligand_from_file(str(pdbqt_path))
    score_before = float(v.score()[0])
    v.optimize(max_steps=max_evals)
    score_after = float(v.score()[0])
    min_path = out_dir / f"{pose_name}_minimized.pdbqt"
    v.write_pose(str(min_path), overwrite=True)
    return score_before, score_after, min_path


def ad4_score(maps_prefix: str, pose_pdbqt: Path) -> float | None:
    """Return AD4 score_only for a single pose using precomputed maps, or None on failure.

    AD4 requires precomputed autogrid maps loaded via load_maps(); set_receptor()
    + compute_vina_maps() is a Vina-only path and crashes with the AD4 SF.
    """
    try:
        v4 = Vina(sf_name="ad4", verbosity=0)
        v4.load_maps(maps_prefix)
        v4.set_ligand_from_file(str(pose_pdbqt))
        return float(v4.score()[0])
    except Exception as e:
        print(f"  AD4 scoring failed: {e}", file=sys.stderr)
        return None


def apply_hybrid(
    vina_score: float,
    ad4_score: float | None,
    n_contact: int,
    n_residues: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> tuple[float, float]:
    """Return (entropy_correction, hybrid_score)."""
    n_non_contact = max(0, n_residues - n_contact)
    n_eff = n_contact + gamma * n_non_contact
    ec = alpha * n_eff
    is_anomaly = ad4_score is not None and ad4_score > 0
    eff_beta = 0.0 if is_anomaly or ad4_score is None else beta
    if ad4_score is not None:
        hybrid = vina_score + eff_beta * (ad4_score - vina_score) + ec
    else:
        hybrid = vina_score + ec
    return ec, hybrid


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)

    ranked_csv = run_dir / "ranked_poses.csv"
    receptor_pdbqt = run_dir / "receptor.pdbqt"
    receptor_pdb = run_dir / "receptor_for_rapidock.pdb"
    pdbqt_dir = run_dir / "pdbqt"
    poses_dir = run_dir / "poses"

    for p in (ranked_csv, receptor_pdbqt, pdbqt_dir):
        if not p.exists():
            print(f"ERROR: expected path missing: {p}", file=sys.stderr)
            sys.exit(1)

    meta = json.loads((run_dir / "run_metadata.json").read_text())
    site = meta["cli_args"]["site_coords"]
    box = float(meta["cli_args"]["box_size"])
    n_residues = len(meta["cli_args"]["peptide_sequence"])

    # Load calibrations
    pfldh_cal = load_calibration(Path(args.calibration))
    old_cal = load_calibration(Path(args.old_calibration))

    # Read ranked_poses.csv to get top-N pose filenames + original scores
    import csv
    with ranked_csv.open() as fh:
        rows = list(csv.DictReader(fh))
    top_rows = rows[: args.top_n]

    # Receptor coords for contact counting
    rec_coords = load_receptor_heavy_atom_coords(receptor_pdb)

    # Set up Vina with precomputed maps
    v = Vina(sf_name="vina", verbosity=0)
    v.set_receptor(str(receptor_pdbqt))
    v.compute_vina_maps(center=site, box_size=[box] * 3)

    out_dir = run_dir / "refined"
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'='*90}")
    print(f"PfLDH local Vina refinement  |  run: {run_dir.name}  |  top {args.top_n} poses")
    print(f"Site: {site}  Box: {box} Å  Peptide: {meta['cli_args']['peptide_sequence']} (n={n_residues})")
    print(f"PfLDH calibration: alpha={pfldh_cal['alpha']:.3f}  beta={pfldh_cal['beta']:.3f}  gamma={pfldh_cal['gamma']:.3f}")
    print(f"MDM2 calibration:  alpha={old_cal['alpha']:.3f}  beta={old_cal['beta']:.3f}  gamma={old_cal.get('gamma', 0.0):.3f}")
    print(f"{'='*90}\n")

    header = (
        f"{'Rank':<5} {'Pose':<12} {'Vina(orig)':<12} {'Vina(min)':<12} "
        f"{'ΔVina':<8} {'AD4(min)':<11} {'n_c':<5} "
        f"{'Hybrid(pfldh)':<16} {'Hybrid(mdm2)':<14}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for row in top_rows:
        rank = int(row["rank"])
        pose_fname = row["pose_filename"]  # e.g. pose_169.pdb
        pose_name = Path(pose_fname).stem  # pose_169
        pdbqt_path = pdbqt_dir / f"{pose_name}.pdbqt"

        if not pdbqt_path.exists():
            print(f"  Rank {rank}: PDBQT not found at {pdbqt_path}")
            continue

        orig_vina = float(row["vina_score"])
        orig_n_contact = int(row["n_contact_residues"])

        # Verify original score matches CSV (sanity check)
        v.set_ligand_from_file(str(pdbqt_path))
        verify_score = float(v.score()[0])
        if abs(verify_score - orig_vina) > 0.05:
            print(f"  WARNING rank {rank}: CSV vina={orig_vina:.4f} vs recomputed={verify_score:.4f}")

        # Local minimization
        score_before, score_after, min_pdbqt = score_and_minimize(
            v, pdbqt_path, max_evals=args.exhaustiveness * 100, out_dir=out_dir, pose_name=pose_name
        )
        delta_vina = score_after - score_before

        # Contact count on minimized pose (pose PDB in poses/ dir)
        min_pose_pdb = out_dir / f"{pose_name}_minimized.pdb"
        # Convert PDBQT → PDB for contact counting
        try:
            pdbqt_text = min_pdbqt.read_text()
            pdb_lines = [
                line for line in pdbqt_text.splitlines()
                if line.startswith("ATOM") or line.startswith("HETATM") or line.startswith("END")
            ]
            min_pose_pdb.write_text("\n".join(pdb_lines) + "\n")
            n_contact = count_contact_residues(min_pose_pdb, rec_coords, cutoff=5.0)
        except Exception:
            n_contact = orig_n_contact  # fallback to original

        # AD4 score on minimized pose
        maps_prefix = str(run_dir / "maps" / "receptor")
        ad4 = ad4_score(maps_prefix, min_pdbqt)

        # Hybrid scores
        ec_pfldh, hybrid_pfldh = apply_hybrid(
            score_after, ad4, n_contact, n_residues,
            pfldh_cal["alpha"], pfldh_cal["beta"], pfldh_cal["gamma"],
        )
        ec_mdm2, hybrid_mdm2 = apply_hybrid(
            score_after, ad4, n_contact, n_residues,
            old_cal["alpha"], old_cal["beta"], old_cal.get("gamma", 0.0),
        )

        ad4_str = f"{ad4:.3f}" if ad4 is not None else "N/A"
        print(
            f"{rank:<5} {pose_name:<12} {score_before:<12.3f} {score_after:<12.3f} "
            f"{delta_vina:<8.3f} {ad4_str:<11} {n_contact:<5} "
            f"{hybrid_pfldh:<16.3f} {hybrid_mdm2:<14.3f}"
        )
        results.append({
            "rank": rank, "pose": pose_name,
            "vina_orig": orig_vina, "vina_min": score_after, "delta_vina": delta_vina,
            "ad4_min": ad4, "n_contact": n_contact,
            "ec_pfldh": ec_pfldh, "hybrid_pfldh": hybrid_pfldh,
            "ec_mdm2": ec_mdm2, "hybrid_mdm2": hybrid_mdm2,
        })

    print("-" * len(header))
    print(f"\nMinimized PDBQTs written to: {out_dir}/")
    print(f"\nADCP reference (finalbinder.pdbqt, full Vina dock, exhaustiveness=16):")
    print(f"  Mode 1: -11.667  Mode 2: -11.261  Mode 3: -11.210  (all 9 modes: -10.64 to -11.67)")

    # Summary
    valid = [r for r in results if r["vina_min"] is not None]
    if valid:
        best = min(valid, key=lambda r: r["hybrid_pfldh"])
        print(f"\nBest pose (pfldh hybrid):  {best['pose']}  hybrid={best['hybrid_pfldh']:.3f}  vina_min={best['vina_min']:.3f}")

    # Write JSON summary
    summary_path = out_dir / "refinement_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
