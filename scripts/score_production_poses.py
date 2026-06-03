"""Score PepSet *production-equivalent* poses with auto-box Vina + AD4.

Production-pose counterpart to ``score_crystal_poses.py``. Generates poses with
RAPiDock-Reloaded against the APO receptor, applies OpenMM clash relief,
auto-computes the Vina/AD4 grid box from the actual pose distribution, scores
with Vina (with optimize() clash relief) + AD4, and aggregates top-K by Vina.

Two distinct failure modes diagnosed on the first pass (2026-06-02) are
addressed here:

1. **Auto-box from poses.** The previous version computed the grid box from
   the crystal peptide. RAPiDock-Reloaded does its own binding-site discovery
   and can place poses several Å away — on 2hwn the actual pose distribution
   needed a 52 Å box where the crystal-derived box was only 37 Å, dropping
   70% of poses to is_clipped. Fix: after RAPiDock + minimization, compute
   the bounding box of all surviving pose heavy atoms, plus ``--box-margin``
   margin (default 4 Å), cubic. Auto-box always covers what's actually there.

2. **Binding-site filter.** Some complexes (1ddv: 18.5 Å offset) get
   poses placed entirely in the wrong pocket. ``--site-filter-radius R``
   (default 15 Å) drops poses whose centroid is more than R Å from the
   intended binding site (crystal peptide centroid). Calibration runs at
   the intended site, not wherever RAPiDock guessed. Disable with ``-R 0``
   to score everything RAPiDock generated.

Output JSON consumable by ``calibrate_alpha.py --scores-json``.

Usage:
    /path/to/score-env/python scripts/score_production_poses.py \\
        --training-csv data/training_complexes.csv \\
        --pepset-dir datasets/pepset \\
        --n-samples 100 \\
        --top-k 10 \\
        --box-margin 4.0 \\
        --site-filter-radius 15.0 \\
        --output data/training_scores_production.json \\
        --work-dir runs/calibration_production
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from hybridock_pep.models import DockConfig, ScoredPose  # noqa: E402
from hybridock_pep.sampling.rapidock_runner import run_sampling  # noqa: E402
from hybridock_pep.sampling.pose_io import parse_poses  # noqa: E402
from hybridock_pep.scoring.minimization import minimize_poses_batch  # noqa: E402
from hybridock_pep.prep.receptor import prepare_receptor, prepare_receptor_pdb  # noqa: E402
from hybridock_pep.prep.grids import generate_ad4_maps  # noqa: E402
from hybridock_pep.prep.ligand import prepare_ligand_batch  # noqa: E402
from hybridock_pep.scoring.vina import score_vina_batch  # noqa: E402
from hybridock_pep.scoring.ad4 import score_ad4_batch  # noqa: E402
from hybridock_pep.scoring.entropy import (  # noqa: E402
    load_receptor_heavy_atom_coords,
    count_contact_residues,
    check_intermolecular_clash,
)

_log = logging.getLogger(__name__)

_DEFAULT_BOX_MARGIN = 4.0
_DEFAULT_MIN_BOX = 22.5
_DEFAULT_SITE_FILTER_R = 15.0  # Å; 0 disables


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _read_peptide_sequence(entry_dir: Path, pdb_id: str) -> str:
    """Read peptide sequence from {pdb_id}_peptide_sequence file."""
    seq_file = entry_dir / f"{pdb_id}_peptide_sequence"
    if not seq_file.exists():
        raise FileNotFoundError(f"Peptide sequence file missing: {seq_file}")
    seq = seq_file.read_text().strip()
    if not seq:
        raise ValueError(f"Empty peptide sequence in {seq_file}")
    return seq


def _parse_heavy_atom_coords(pdb_path: Path) -> np.ndarray:
    """Return (N,3) array of heavy-atom XYZ from a PDB. Empty if none."""
    coords: list[list[float]] = []
    for line in pdb_path.read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        name = line[12:16].strip()
        if name.startswith("H"):
            continue
        try:
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
    return np.array(coords) if coords else np.zeros((0, 3))


def _pose_centroid(pdb_path: Path) -> np.ndarray | None:
    """Centroid of heavy atoms; None if no atoms parsed."""
    atoms = _parse_heavy_atom_coords(pdb_path)
    return atoms.mean(axis=0) if atoms.size else None


def _filter_poses_by_site(
    pose_paths: list[Path],
    site_center: np.ndarray,
    radius: float,
) -> tuple[list[Path], list[float]]:
    """Keep poses whose centroid is within ``radius`` of ``site_center``.

    Returns:
        (kept_paths, dropped_distances) — kept is a subset of pose_paths;
        dropped_distances is the distance for each pose that was dropped
        (for logging / reporting).
    """
    if radius <= 0:
        return list(pose_paths), []
    kept: list[Path] = []
    dropped_d: list[float] = []
    for p in pose_paths:
        c = _pose_centroid(p)
        if c is None:
            continue
        d = float(np.linalg.norm(c - site_center))
        if d <= radius:
            kept.append(p)
        else:
            dropped_d.append(d)
    return kept, dropped_d


def _auto_box_from_poses(
    pose_paths: list[Path],
    margin: float,
    min_box: float,
) -> tuple[tuple[float, float, float], float]:
    """Compute a cubic Vina grid box covering all heavy atoms in all poses.

    Centre is the midpoint of the per-axis (min, max) ranges. Box size is
    the max axis extent plus ``margin``, clamped to ``min_box``. Cubic
    because Vina map generation requires a single box edge for all axes.

    Args:
        pose_paths: PDB files to read atoms from. Must be non-empty.
        margin: Extra Å added to each axis beyond the bounding box.
        min_box: Minimum edge length.

    Returns:
        ((cx, cy, cz), box_size) — float coordinates and float edge length.

    Raises:
        ValueError: If no heavy atoms can be read across all pose files.
    """
    all_atoms: list[np.ndarray] = []
    for p in pose_paths:
        a = _parse_heavy_atom_coords(p)
        if a.size:
            all_atoms.append(a)
    if not all_atoms:
        raise ValueError("No heavy atoms found across pose files — cannot derive box")
    stacked = np.vstack(all_atoms)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = maxs - mins
    box_edge = float(max(float(extent.max()) + margin, min_box))
    return (float(center[0]), float(center[1]), float(center[2])), box_edge


def _aggregate_top_k(
    per_pose: list[dict],
    top_k: int,
) -> tuple[dict[str, float | int], list[dict]]:
    """Pick top-K poses by Vina score; return (aggregate, top_k_poses).

    Aggregation uses MEDIAN — robust to one outlier in the top-K bucket.
    """
    if not per_pose:
        raise ValueError("No successful poses to aggregate")
    sorted_poses = sorted(per_pose, key=lambda r: r["vina_score"])
    k_actual = min(top_k, len(sorted_poses))
    top_poses = sorted_poses[:k_actual]
    return ({
        "vina_score": round(float(statistics.median(p["vina_score"] for p in top_poses)), 3),
        "ad4_score":  round(float(statistics.median(p["ad4_score"]  for p in top_poses)), 3),
        "n_contact_residues": int(statistics.median(p["n_contact_residues"] for p in top_poses)),
    }, top_poses)


# --------------------------------------------------------------------------- #
# Per-complex workflow                                                          #
# --------------------------------------------------------------------------- #

def score_entry(
    pdb_id: str,
    pepset_dir: Path,
    work_dir: Path,
    n_samples: int,
    top_k: int,
    seed: int,
    box_margin: float,
    min_box: float,
    site_filter_radius: float,
) -> dict:
    """Run the production-pose scoring workflow for one complex with auto-box.

    Pipeline order (mirrors driver.run_dock, with auto-box inserted between
    minimization and scoring):

        1. run_sampling           — RAPiDock-Reloaded on apo receptor
        2. parse_poses            — read pose_*.pdb
        3. minimize_poses_batch   — OpenMM restrained clash relief
        4. site-radius filter     — drop poses far from intended site
        5. auto-box               — derive cubic box from kept pose atoms
        6. prepare_receptor       — receptor PDBQT
        7. generate_ad4_maps      — AD4 maps at the auto-box
        8. prepare_ligand_batch   — per-pose PDBQT
        9. score_vina_batch       — Vina with optimize() clash relief
       10. score_ad4_batch        — AD4 against optimized PDBQTs
       11. contact counting       — n_contact_residues per pose

    Returns a JSON entry suitable for calibrate_alpha.py.
    """
    entry_dir = pepset_dir / pdb_id
    # CRITICAL: use the POCKET-TRUNCATED PDB as RAPiDock input. The
    # `rapidock_local.pt` checkpoint is the local-docking model — it expects
    # a pocket file (binding-site residues only) so the binding site is
    # implicitly specified by which residues are present. Feeding it the
    # full apo receptor causes it to pick its own pocket (sometimes the
    # wrong one — observed 18 Å offset on 1ddv, 12 Å on 2hwn).
    pocket_pdb = entry_dir / f"{pdb_id}_rec_unbound_pocket.pdb"
    full_apo_pdb = entry_dir / f"{pdb_id}_rec_unbound.pdb"
    crystal_pep_pdb = entry_dir / f"{pdb_id}_pep_ref.pdb"
    for path in (pocket_pdb, full_apo_pdb, crystal_pep_pdb):
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    peptide_seq = _read_peptide_sequence(entry_dir, pdb_id)
    _log.info("[%s] Peptide: %s (n_res=%d)", pdb_id, peptide_seq, len(peptide_seq))
    _log.info("[%s] RAPiDock input = pocket file (%s)", pdb_id, pocket_pdb.name)

    work_complex = (work_dir / pdb_id).resolve()
    work_complex.mkdir(parents=True, exist_ok=True)

    # Site reference: the crystal peptide centroid is the intended binding
    # site for this calibration entry. Used by the site-radius filter.
    crystal_pep_atoms = _parse_heavy_atom_coords(crystal_pep_pdb)
    site_center = crystal_pep_atoms.mean(axis=0)
    _log.info("[%s] Crystal peptide centroid (intended site): (%.2f, %.2f, %.2f)",
              pdb_id, *site_center)

    # ---- Step 1: RAPiDock ------------------------------------------------- #
    # Initial DockConfig with a placeholder box. RAPiDock-Reloaded does not
    # take a binding site — the box will be replaced with the auto-derived
    # one after minimization.
    placeholder_box = max(30.0, min_box)
    # RAPiDock input: the pocket PDB. Vina/AD4 scoring will use this same
    # pocket as the receptor — the pocket file contains all binding-site
    # residues so contact counting and Vina maps are correct against it.
    # (Using the full apo for scoring would inflate AD4 map size with no
    # gain — contacts and Vina interactions only happen at the pocket.)
    cfg_sampling = DockConfig(
        peptide_sequence=peptide_seq,
        receptor_path=pocket_pdb.resolve(),
        site_coords=(float(site_center[0]), float(site_center[1]), float(site_center[2])),
        box_size=placeholder_box,
        n_samples=n_samples,
        seed=seed,
        scoring={"vina", "ad4"},
        output_dir=work_complex,
        run_id=f"calibration_prod_{pdb_id}",
        minimize_poses=True,
    )

    _log.info("[%s] Step 1: RAPiDock sampling (N=%d, seed=%d)...",
              pdb_id, n_samples, seed)
    t0 = time.perf_counter()
    cleaned_receptor = prepare_receptor_pdb(cfg_sampling)
    run_sampling(cfg_sampling, receptor_path=cleaned_receptor)
    poses_dir = (cfg_sampling.output_dir / "poses").resolve()
    _log.info("[%s] RAPiDock done in %.1fs → %s", pdb_id,
              time.perf_counter() - t0, poses_dir)

    # ---- Step 2: parse poses --------------------------------------------- #
    records, parse_failures = parse_poses(poses_dir)
    if parse_failures:
        _log.warning("[%s] %d poses failed parsing", pdb_id, len(parse_failures))
    _log.info("[%s] Parsed %d poses", pdb_id, len(records))
    if not records:
        raise RuntimeError(f"[{pdb_id}] No poses parsed from {poses_dir}")

    # ---- Step 3: OpenMM clash relief ------------------------------------- #
    _log.info("[%s] Step 3: OpenMM clash relief...", pdb_id)
    min_dir = (cfg_sampling.output_dir / "poses_minimized").resolve()
    raw_paths = [r.pdb_path.resolve() for r in records]
    minimized_paths = minimize_poses_batch(raw_paths, min_dir)
    for rec, mp in zip(records, minimized_paths):
        rec.pdb_path = mp.resolve()
    _log.info("[%s] Minimized %d poses", pdb_id, len(records))

    # ---- Step 4: site-radius filter -------------------------------------- #
    if site_filter_radius > 0:
        kept_paths, dropped_d = _filter_poses_by_site(
            [r.pdb_path for r in records], site_center, site_filter_radius,
        )
        kept_set = set(kept_paths)
        records = [r for r in records if r.pdb_path in kept_set]
        n_dropped = len(dropped_d)
        if n_dropped:
            d_arr = np.array(dropped_d)
            _log.warning(
                "[%s] Site filter (R=%.1f Å vs crystal centroid): dropped %d/%d poses "
                "(distances min=%.1f median=%.1f max=%.1f Å)",
                pdb_id, site_filter_radius, n_dropped,
                n_dropped + len(records),
                float(d_arr.min()), float(np.median(d_arr)), float(d_arr.max()),
            )
        else:
            _log.info("[%s] Site filter passed all %d poses", pdb_id, len(records))
        if not records:
            raise RuntimeError(
                f"[{pdb_id}] All poses dropped by site filter (radius {site_filter_radius} Å) "
                f"— RAPiDock placed poses in the wrong pocket. "
                f"Re-run with --site-filter-radius 0 to score everything, "
                f"or use the holo receptor as RAPiDock input."
            )

    # ---- Step 5: auto-box from kept poses -------------------------------- #
    auto_center, auto_box = _auto_box_from_poses(
        [r.pdb_path for r in records], margin=box_margin, min_box=min_box,
    )
    site_offset = float(np.linalg.norm(np.array(auto_center) - site_center))
    _log.info(
        "[%s] Step 5: auto-box centre=(%.2f, %.2f, %.2f) edge=%.1f Å  (offset from "
        "crystal centroid = %.2f Å, margin=%.1f Å, min=%.1f Å)",
        pdb_id, *auto_center, auto_box, site_offset, box_margin, min_box,
    )

    # Rebuild config with the auto box for the scoring stages
    cfg_score = cfg_sampling.model_copy(update={
        "site_coords": auto_center,
        "box_size": auto_box,
    })

    # ---- Step 6: receptor PDBQT ------------------------------------------ #
    _log.info("[%s] Step 6: prepare receptor PDBQT...", pdb_id)
    receptor_pdbqt = prepare_receptor(cfg_score)

    # ---- Step 7: AD4 grid maps ------------------------------------------- #
    _log.info("[%s] Step 7: generate AD4 maps...", pdb_id)
    maps_dir = generate_ad4_maps(cfg_score, receptor_pdbqt)

    # ---- Step 8: ligand batch prep --------------------------------------- #
    _log.info("[%s] Step 8: ligand batch prep (%d poses)...", pdb_id, len(records))
    pdbqt_dir = (cfg_score.output_dir / "pdbqt").resolve()
    pdbqt_paths, prep_failures = prepare_ligand_batch(
        [r.pdb_path.resolve() for r in records], pdbqt_dir,
    )
    if prep_failures:
        _log.warning("[%s] %d poses failed ligand prep", pdb_id, len(prep_failures))
    pdbqt_by_stem = {p.stem: p for p in pdbqt_paths}
    scored_poses: list[ScoredPose] = []
    for rec in records:
        pdbqt_path = pdbqt_by_stem.get(rec.pdb_path.stem)
        if pdbqt_path is None:
            continue
        scored_poses.append(ScoredPose(
            pose_idx=rec.pose_idx,
            pdb_path=rec.pdb_path,
            sequence=rec.sequence,
            ca_coords=rec.ca_coords,
            pdbqt_path=pdbqt_path,
        ))
    _log.info("[%s] %d poses ready for scoring", pdb_id, len(scored_poses))

    # ---- Step 9: Vina batch with optimize() clash relief ----------------- #
    metadata_path = cfg_score.output_dir / "run_metadata.json"
    _log.info("[%s] Step 9: Vina batch...", pdb_id)
    scored_poses, vina_failures = score_vina_batch(
        scored_poses, cfg_score, receptor_pdbqt.resolve(),
        metadata_path=metadata_path,
    )
    if vina_failures:
        _log.warning("[%s] %d poses failed Vina", pdb_id, len(vina_failures))

    # ---- Step 10: AD4 batch ---------------------------------------------- #
    _log.info("[%s] Step 10: AD4 batch...", pdb_id)
    scored_poses, ad4_failures = score_ad4_batch(scored_poses, maps_dir.resolve())
    if ad4_failures:
        _log.warning("[%s] %d poses failed AD4", pdb_id, len(ad4_failures))

    # ---- Step 11: contact counts ----------------------------------------- #
    receptor_coords = load_receptor_heavy_atom_coords(pocket_pdb.resolve())
    for pose in scored_poses:
        pose.n_contact_residues = count_contact_residues(pose.pdb_path, receptor_coords)
        pose.is_clashed = check_intermolecular_clash(pose.pdb_path, receptor_coords)

    # ---- Step 12: filter scored poses + aggregate ------------------------ #
    per_pose: list[dict] = []
    for pose in scored_poses:
        if pose.vina_score is None or pose.ad4_score is None:
            continue
        if pose.is_clipped:
            continue
        per_pose.append({
            "pose": pose.pdb_path.name,
            "pose_idx": pose.pose_idx,
            "vina_score": round(float(pose.vina_score), 3),
            "ad4_score": round(float(pose.ad4_score), 3),
            "n_contact_residues": int(pose.n_contact_residues or 0),
            "is_clashed": bool(pose.is_clashed),
        })

    if not per_pose:
        raise RuntimeError(
            f"[{pdb_id}] no usable scored poses (all clipped or score=None) "
            f"even after auto-box — investigate manually."
        )

    aggregate, top_poses = _aggregate_top_k(per_pose, top_k)
    _log.info(
        "[%s] Top-%d aggregate: vina=%.3f  ad4=%.3f  contacts=%d  (n_scored=%d)",
        pdb_id, top_k, aggregate["vina_score"], aggregate["ad4_score"],
        aggregate["n_contact_residues"], len(per_pose),
    )

    return {
        **aggregate,
        "top_k": top_k,
        "n_samples_requested": n_samples,
        "n_poses_parsed": len(records) + sum(1 for _ in []),  # post-filter records
        "n_poses_scored": len(per_pose),
        "seed": seed,
        "box_center_auto": list(auto_center),
        "box_size_auto": auto_box,
        "box_margin": box_margin,
        "site_filter_radius": site_filter_radius,
        "site_offset_from_crystal": round(site_offset, 2),
        "top_k_poses": top_poses,
        "all_poses": per_pose,
    }


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--training-csv", type=Path,
                        default=Path("data/training_complexes.csv"))
    parser.add_argument("--pepset-dir", type=Path, default=Path("datasets/pepset"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/training_scores_production.json"))
    parser.add_argument("--work-dir", type=Path,
                        default=Path("runs/calibration_production"))
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--box-margin", type=float, default=_DEFAULT_BOX_MARGIN,
        help=("Å added to each side of the pose bounding box. "
              f"Default {_DEFAULT_BOX_MARGIN}."),
    )
    parser.add_argument(
        "--min-box", type=float, default=_DEFAULT_MIN_BOX,
        help=f"Minimum auto-box edge length (Å). Default {_DEFAULT_MIN_BOX}.",
    )
    parser.add_argument(
        "--site-filter-radius", type=float, default=_DEFAULT_SITE_FILTER_R,
        help=("Drop poses whose centroid is further than R Å from the crystal "
              "peptide centroid before auto-box / scoring. Use 0 to disable. "
              f"Default {_DEFAULT_SITE_FILTER_R}."),
    )
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated pdb_ids to score (default: all)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.training_csv.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    if args.only:
        only_set = {s.strip().lower() for s in args.only.split(",") if s.strip()}
        rows = [r for r in rows if r["pdb_id"].strip().lower() in only_set]
        _log.info("Filtered to %d rows via --only", len(rows))

    _log.info("Scoring %d complexes  N=%d  top-K=%d  margin=%.1f Å  min-box=%.1f Å  "
              "site-filter=%.1f Å",
              len(rows), args.n_samples, args.top_k, args.box_margin,
              args.min_box, args.site_filter_radius)

    scores: dict[str, dict] = {}
    if args.output.exists():
        try:
            existing = json.loads(args.output.read_text())
            if isinstance(existing, dict):
                scores = existing
                _log.info("Resume: loaded %d existing entries from %s",
                          len(scores), args.output)
        except json.JSONDecodeError:
            _log.warning("Existing %s not JSON; starting fresh", args.output)

    failed: list[str] = []
    for row in rows:
        pdb_id = row["pdb_id"].strip()
        # Skip if already scored (any case)
        if pdb_id in scores or pdb_id.lower() in scores or pdb_id.upper() in scores:
            _log.info("[%s] already scored — skipping (delete entry to re-run)", pdb_id)
            continue
        try:
            entry = score_entry(
                pdb_id=pdb_id,
                pepset_dir=args.pepset_dir.resolve(),
                work_dir=args.work_dir.resolve(),
                n_samples=args.n_samples,
                top_k=args.top_k,
                seed=args.seed,
                box_margin=args.box_margin,
                min_box=args.min_box,
                site_filter_radius=args.site_filter_radius,
            )
            scores[pdb_id] = entry
            args.output.write_text(json.dumps(scores, indent=2))
            _log.info("[%s] Persisted to %s", pdb_id, args.output)
        except Exception as exc:  # noqa: BLE001
            _log.error("[%s] FAILED: %s: %s", pdb_id, type(exc).__name__, exc)
            failed.append(pdb_id)

    _log.info("Done. Succeeded: %d  Failed: %d", len(scores), len(failed))
    if failed:
        _log.warning("Failed entries: %s", failed)


if __name__ == "__main__":
    main()
