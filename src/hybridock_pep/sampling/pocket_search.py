"""Blind pocket-search-then-refine Stage 1 — score-env, Python 3.11.

Used when the user gives ``--blind`` with no ``--site`` at all (true "I don't know
where it binds" blind docking, as opposed to ``--blind --site X Y Z`` which just
disables the off-pocket filter around a known hint). Three stages:

1. **Search**: run RAPiDock once with ``rapidock_global.pt`` (the checkpoint
   trained for whole-receptor / no-pocket-hint sampling — see README "Blind
   docking") across the full receptor at N=``config.n_pocket_search``.
2. **Cluster**: group the search poses' Cα centroids spatially (agglomerative
   clustering on the 3D centroid coordinates — spatial *grouping*, not the
   Kabsch superposition used elsewhere in this codebase for RMSD between two
   *already-corresponding* point sets; there's no correspondence between poses
   in different candidate pockets, so Kabsch alignment isn't the right
   primitive here) into up to ``config.n_pockets`` candidate binding sites,
   ranked by population (most poses landing there = most consensus).
3. **Refine**: for each candidate site, crop the receptor to a neighborhood of
   its centroid (mirrors RAPiDock's own pocket-given training convention —
   PDBs pre-cropped to the true pocket, not the literal ``--site`` args this
   pipeline invented for Vina) and re-run RAPiDock at N=``config.n_per_pocket``
   with the length-routed local checkpoint (``rapidock_local.pt`` or
   ``longer_local.pt`` — see ``select_checkpoint()``).

The merged refinement poses (not the exploratory search poses, which come from
the less-specialized global checkpoint) become Stage 1's output and flow into
the existing Stage 2 scoring pipeline exactly as a single-site run would.
"""
from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path

import numpy as np

from hybridock_pep.models import DockConfig
from hybridock_pep.sampling.pose_io import parse_poses
from hybridock_pep.sampling.rapidock_runner import run_sampling

logger = logging.getLogger(__name__)

#: Radius (Å) of receptor kept around a candidate pocket centroid for the
#: refinement stage. Generous relative to a typical Vina box (20–30 Å edge)
#: so the cropped receptor still gives the local/longer_local checkpoint a
#: full pocket neighborhood, not just the handful of contact residues.
_POCKET_CROP_RADIUS_AA: float = 25.0

#: Minimum number of search poses required before clustering is attempted.
#: Below this, every pose gets its own singleton "cluster" and pocket-finding
#: degenerates to no-op; failing loudly here is more useful than silently
#: refining N indistinguishable "pockets" that are really just noise.
_MIN_POSES_FOR_CLUSTERING = 4


def _crop_receptor_to_site(
    receptor_path: Path, center: np.ndarray, radius: float, dest: Path,
) -> Path:
    """Write a copy of ``receptor_path`` containing only residues with ≥1 atom
    within ``radius`` Å of ``center``.

    Whole-residue inclusion (not per-atom truncation) — a partially-cropped
    residue would have a nonsense truncated side chain, worse than either
    fully keeping or fully dropping it.

    Args:
        receptor_path: Full (already pdbfixer-cleaned) receptor PDB.
        center: (x, y, z) pocket centroid in Angstroms.
        radius: Crop radius in Angstroms.
        dest: Output path for the cropped PDB.

    Returns:
        ``dest``, for chaining.
    """
    kept_lines: list[str] = []
    current_key: tuple[str, str, str] | None = None
    current_block: list[str] = []
    current_hit = False

    def _flush() -> None:
        if current_hit:
            kept_lines.extend(current_block)

    for line in receptor_path.read_text().splitlines(keepends=True):
        if not line.startswith(("ATOM", "HETATM")):
            if not line.startswith(("ANISOU", "TER")):
                kept_lines.append(line)
            continue
        key = (line[21], line[22:26], line[26])  # chain, resseq, icode
        if key != current_key:
            _flush()
            current_key = key
            current_block = []
            current_hit = False
        current_block.append(line)
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            continue
        if np.linalg.norm(np.array([x, y, z]) - center) <= radius:
            current_hit = True
    _flush()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(kept_lines))
    return dest


def _cluster_pocket_centroids(
    centroids: np.ndarray, n_pockets: int,
) -> list[tuple[np.ndarray, int]]:
    """Cluster pose centroids spatially and return per-cluster (mean center,
    population), ordered largest-population first.

    Args:
        centroids: (n_poses, 3) array of per-pose Cα centroids.
        n_pockets: Requested number of candidate pockets.

    Returns:
        List of (center, population) tuples, length ≤ n_pockets, sorted by
        population descending.
    """
    from sklearn.cluster import AgglomerativeClustering

    n_clusters = min(n_pockets, len(centroids))
    if n_clusters <= 1:
        return [(centroids.mean(axis=0), len(centroids))]
    labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="average").fit_predict(
        centroids
    )
    sites: list[tuple[np.ndarray, int]] = []
    for label in sorted(set(labels)):
        mask = labels == label
        sites.append((centroids[mask].mean(axis=0), int(mask.sum())))
    sites.sort(key=lambda t: t[1], reverse=True)
    return sites


def run_pocket_search_and_refine(
    config: DockConfig, receptor_path: Path,
) -> list[Path]:
    """Run the search → cluster → refine blind-docking pipeline.

    Args:
        config: Validated DockConfig with pocket_search=True.
        receptor_path: pdbfixer-cleaned receptor PDB (full, uncropped).

    Returns:
        List of absolute Paths to the merged refinement pose_*.pdb files
        under config.output_dir/poses/ (the exploratory search poses are
        kept in poses_search/ for inspection but not scored).

    Raises:
        RuntimeError: If the exploratory search produces too few poses to
            cluster, or every pocket's refinement stage fails.
    """
    receptor_path = receptor_path.resolve()
    search_dir = config.output_dir / "search"
    logger.info(
        "Stage 1 (blind): pocket search — N=%d with rapidock_global.pt, whole receptor",
        config.n_pocket_search,
    )
    search_poses = run_sampling(
        config,
        receptor_path=receptor_path,
        n_samples=config.n_pocket_search,
        ckpt_name="rapidock_global.pt",
        output_dir=search_dir,
    )
    records, _failures = parse_poses(search_dir / "poses")
    records = [r for r in records if r.ca_coords is not None and len(r.ca_coords) > 0]
    if len(records) < _MIN_POSES_FOR_CLUSTERING:
        raise RuntimeError(
            f"Pocket search produced only {len(records)} usable poses "
            f"(need ≥{_MIN_POSES_FOR_CLUSTERING} to cluster into candidate pockets). "
            "Try a larger --n-pocket-search or check the receptor PDB."
        )

    centroids = np.array([r.ca_coords.mean(axis=0) for r in records])
    sites_with_pop = _cluster_pocket_centroids(centroids, config.n_pockets)
    sites = [center for center, _pop in sites_with_pop]
    logger.info(
        "Stage 1 (blind): %d candidate pocket(s) found from %d search poses "
        "(populations: %s)",
        len(sites), len(records), [pop for _center, pop in sites_with_pop],
    )

    poses_dir = (config.output_dir / "poses").resolve()
    poses_dir.mkdir(parents=True, exist_ok=True)
    assignment_rows: list[dict[str, object]] = []
    merged: list[Path] = []
    n_pocket_failures = 0

    for pocket_idx, center in enumerate(sites):
        pocket_dir = config.output_dir / f"pocket_{pocket_idx}"
        cropped_receptor = _crop_receptor_to_site(
            receptor_path, center, _POCKET_CROP_RADIUS_AA,
            pocket_dir / "receptor_cropped.pdb",
        )
        logger.info(
            "Stage 1 (blind): refining pocket %d/%d at %s — N=%d, cropped receptor (%.0f Å radius)",
            pocket_idx + 1, len(sites),
            tuple(round(float(c), 1) for c in center),
            config.n_per_pocket, _POCKET_CROP_RADIUS_AA,
        )
        try:
            pocket_poses = run_sampling(
                config,
                receptor_path=cropped_receptor,
                n_samples=config.n_per_pocket,
                ckpt_name=None,  # length-routed: rapidock_local.pt or longer_local.pt
                output_dir=pocket_dir,
            )
        except RuntimeError as exc:
            n_pocket_failures += 1
            logger.warning("Pocket %d refinement failed, skipping (%s)", pocket_idx, exc)
            continue
        for src in pocket_poses:
            dst = poses_dir / f"pose_{len(merged)}.pdb"
            shutil.copy2(src, dst)
            merged.append(dst)
            assignment_rows.append({
                "pose_filename": dst.name,
                "pocket_id": pocket_idx,
                "pocket_center_x": round(float(center[0]), 2),
                "pocket_center_y": round(float(center[1]), 2),
                "pocket_center_z": round(float(center[2]), 2),
            })

    if n_pocket_failures == len(sites):
        raise RuntimeError(
            f"All {len(sites)} candidate pocket refinements failed — see log above "
            "for per-pocket errors."
        )

    assignment_path = config.output_dir / "pocket_assignment.csv"
    with assignment_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(assignment_rows[0].keys()))
        writer.writeheader()
        writer.writerows(assignment_rows)
    logger.info(
        "Stage 1 (blind) complete: %d poses across %d/%d pocket(s) → %s (assignment: %s)",
        len(merged), len(sites) - n_pocket_failures, len(sites), poses_dir, assignment_path,
    )
    del search_poses  # kept on disk under search_dir for inspection; not part of merged output
    return merged
