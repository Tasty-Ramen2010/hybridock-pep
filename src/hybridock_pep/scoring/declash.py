"""Move a docked peptide out of the protein it was sampled into.

RAPiDock draws poses by diffusion, and a share of them end up partly inside the receptor (heavy atoms a fraction of an
ångström from protein atoms). Those poses look wrong in a viewer and give Vina nothing sensible to score. This step slides
each pose, as one rigid body, to the nearest place where no peptide heavy atom is inside the protein. The peptide's own
shape is untouched; only its position and orientation change, and only when it overlaps.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

_log = logging.getLogger(__name__)

#: repulsion shell: pairs closer than this push apart (a van der Waals contact between heavy atoms is ~3.4 Å)
SHELL = 3.1
#: a pose counts as clear of the protein when its closest heavy-atom pair is at least this far apart (a tight hydrogen bond is ~2.6 Å)
CLEAR = 2.7
_STEPS = 400


def _heavy_mask(lines: list[str]) -> tuple[list[int], np.ndarray]:
    idx: list[int] = []
    xyz: list[tuple[float, float, float]] = []
    for i, ln in enumerate(lines):
        if not ln.startswith(("ATOM  ", "HETATM")):
            continue
        name = ln[12:16].strip()
        if not name or name[0] in ("H", "D"):
            continue
        try:
            xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
        except ValueError:
            continue
        idx.append(i)
    return idx, np.asarray(xyz, dtype=float).reshape(-1, 3)


def _energy_and_forces(pep: np.ndarray, tree: cKDTree, rec: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Overlap energy, the force on each peptide atom, and the closest heavy-atom distance."""
    hits = tree.query_ball_point(pep, SHELL)
    force = np.zeros_like(pep)
    energy = 0.0
    for i, js in enumerate(hits):
        if not js:
            continue
        delta = pep[i] - rec[js]
        dist = np.maximum(np.linalg.norm(delta, axis=1), 1e-3)
        overlap = SHELL - dist
        energy += float(np.sum(overlap ** 2))
        force[i] = np.sum((2.0 * overlap / dist)[:, None] * delta, axis=0)
    nearest = float(tree.query(pep)[0].min())
    return energy, force, nearest


def _rotate(points: np.ndarray, centre: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return points
    k = rotvec / angle
    p = points - centre
    cos, sin = np.cos(angle), np.sin(angle)
    return centre + p * cos + np.cross(k, p) * sin + np.outer(p @ k, k) * (1.0 - cos)


def _descend(pep: np.ndarray, tree: cKDTree, rec: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Rigid-body steepest descent on the overlap energy. Returns the moved atoms, the energy and the closest distance."""
    energy, force, nearest = _energy_and_forces(pep, tree, rec)
    step = 0.02
    for _ in range(_STEPS):
        if energy < 1e-4:
            break
        centre = pep.mean(axis=0)
        push = force.sum(axis=0)
        torque = np.cross(pep - centre, force).sum(axis=0)
        while step > 1e-6:
            trial = _rotate(pep, centre, torque * step * 0.05) + push * step
            e2, f2, n2 = _energy_and_forces(trial, tree, rec)
            if e2 < energy:
                pep, energy, force, nearest = trial, e2, f2, n2
                step *= 1.3
                break
            step *= 0.5
        else:
            break
    return pep, energy, nearest


def declash_pose(pose_pdb: Path, receptor_xyz: np.ndarray, out_pdb: Path) -> dict:
    """Write ``out_pdb``: the pose moved clear of the receptor. Returns ``{"moved": Å, "clearance": Å, "clear": bool}``.

    A pose that is already clear is copied unchanged. A pose that cannot be freed (a peptide threaded through a thin part of
    the protein) is written at the best position found, and ``clear`` is False so the caller can rank it last.
    """
    lines = pose_pdb.read_text(encoding="utf-8", errors="ignore").splitlines()
    heavy_idx, pep0 = _heavy_mask(lines)
    if not len(pep0) or not len(receptor_xyz):
        out_pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"moved": 0.0, "clearance": float("inf"), "clear": True}
    tree = cKDTree(receptor_xyz)
    start = float(tree.query(pep0)[0].min())
    if start >= CLEAR:
        out_pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"moved": 0.0, "clearance": start, "clear": True}

    centre0 = pep0.mean(axis=0)
    away = centre0 - receptor_xyz.mean(axis=0)
    away = away / (np.linalg.norm(away) or 1.0)
    rng = np.random.default_rng(0)
    best: tuple[float, float, np.ndarray] | None = None  # (energy, movement, atoms)
    # the pose as it is, then the same pose lifted outward by growing amounts, then a few tilts: the first that is clear wins
    starts = [pep0] + [pep0 + away * d for d in (3.0, 6.0, 10.0)]
    starts += [_rotate(pep0 + away * 4.0, centre0 + away * 4.0, rng.normal(size=3) * 0.5) for _ in range(6)]
    for cand in starts:
        moved, energy, nearest = _descend(cand.copy(), tree, receptor_xyz)
        shift = float(np.linalg.norm(moved.mean(axis=0) - centre0))
        if nearest >= CLEAR:
            key = (0.0, shift, moved)
        else:
            key = (energy, shift, moved)
        if best is None or (key[0], key[1]) < (best[0], best[1]):
            best = key
        if nearest >= CLEAR and shift < 4.0:
            break
    assert best is not None
    final = best[2]
    # the same rigid motion applied to every atom of the pose, hydrogens included
    all_idx = [i for i, ln in enumerate(lines) if ln.startswith(("ATOM  ", "HETATM"))]
    xyz = np.asarray([(float(lines[i][30:38]), float(lines[i][38:46]), float(lines[i][46:54])) for i in all_idx])
    # recover the motion from the heavy atoms (Kabsch fit of the original onto the moved heavy atoms)
    c_a, c_b = pep0.mean(axis=0), final.mean(axis=0)
    u, _, vt = np.linalg.svd((pep0 - c_a).T @ (final - c_b))
    sign = np.sign(np.linalg.det(u @ vt))
    rot = (u @ np.diag([1.0, 1.0, sign]) @ vt).T
    new_xyz = (xyz - c_a) @ rot.T + c_b
    for row, i in zip(new_xyz, all_idx):
        ln = lines[i]
        lines[i] = f"{ln[:30]}{row[0]:8.3f}{row[1]:8.3f}{row[2]:8.3f}{ln[54:]}"
    out_pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")
    clearance = float(tree.query(_heavy_mask(lines)[1])[0].min())
    return {"moved": float(np.linalg.norm(c_b - c_a)), "clearance": clearance, "clear": clearance >= CLEAR}


def declash_poses_batch(pose_paths: list[Path], receptor_xyz: np.ndarray, out_dir: Path) -> list[Path]:
    """Declash every pose into ``out_dir``. A pose that errors is passed through unchanged (it is never lost)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    freed = 0
    for src in pose_paths:
        dest = out_dir / src.name
        try:
            info = declash_pose(src, receptor_xyz, dest)
            freed += int(info["moved"] > 0 and info["clear"])
            out.append(dest)
        except Exception as exc:  # noqa: BLE001 - never lose a pose because of this step
            _log.warning("declash: %s left as it was (%s)", src.name, exc)
            out.append(src)
    _log.info("declash: %d of %d poses moved out of the protein", freed, len(pose_paths))
    return out
