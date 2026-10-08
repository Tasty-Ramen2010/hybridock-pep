"""Vina score_only batch scorer (SCORE-01).

Implements per-pose grid boundary validation and single-instance Vina Python API
scoring. Returns (list[ScoredPose], list[PoseFailure]) — never raises on
per-pose failures.

Key design decisions (from RESEARCH.md / STATE.md):
- One Vina instance per batch; set_ligand_from_file() called per pose.
- compute_vina_maps() called ONCE before the pose loop (all 22 atom types).
- float(v.score()[0]) used throughout — never raw numpy array comparisons.
- Vina SWIG bindings have no documented thread safety, so parallelism is by *process*:
  each worker owns a private Vina instance (see ``_score_vina_batch_impl``).
- optimize_clashing=True (default): when initial score > 0 (receptor-peptide clash),
  call v.optimize() up to max_clash_relief_rounds times, stopping early when the
  score drops below 0 or BFGS has converged (< 0.5 kcal/mol improvement per round).
  RAPiDock-Reloaded generates side-chain torsions that can overlap receptor atoms;
  Vina local optimization resolves these clashes in the same scoring function used
  for ranking, keeping the comparison self-consistent.
  Multi-round relief helps for marginal clashes; severe overlaps (> 50 kcal/mol)
  typically need no_final_step_noise=True at the RAPiDock level to prevent the
  clash at source.
"""

from __future__ import annotations

import json
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from hybridock_pep.models import DockConfig, PoseFailure, ScoredPose

try:
    from vina import Vina
except ImportError:  # score-env not active (e.g. during unit tests with mocks)
    Vina = None  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)


def check_grid_boundary(
    pdbqt_path: Path,
    site_coords: tuple[float, float, float],
    box_size: float,
) -> bool:
    """Return True if any heavy ATOM/HETATM atom in the PDBQT falls outside the grid box.

    Hydrogen atoms (element H or HD in cols 76-78; or name starting with 'H'
    when the element column is absent) are excluded from the check.  babel adds
    polar H to PDBQT for Gasteiger charges; these can lie marginally outside the
    grid even when all heavy atoms are within bounds — falsely flagging otherwise
    good poses.  Vina and AD4 grid-based scoring are not materially affected by H
    positions, so only heavy-atom placement determines whether a pose is clipped.

    Parses fixed-column PDB/PDBQT coordinate fields (cols 30-38 x, 38-46 y,
    46-54 z). Boundary is inclusive: an atom exactly on the edge is NOT clipped.
    Malformed coordinate lines are silently skipped.

    Args:
        pdbqt_path: Path to the prepared PDBQT file for one pose.
        site_coords: (cx, cy, cz) grid box center in Angstrom.
        box_size: Grid box edge length in Angstrom.

    Returns:
        True if any heavy atom lies strictly outside site_coords ± box_size/2 on
        any axis; False otherwise (including the case of no parseable atoms).
    """
    cx, cy, cz = site_coords
    half = box_size / 2.0

    for line in pdbqt_path.read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue

        # Skip hydrogens — babel-added H can lie marginally outside the grid
        # without affecting scoring accuracy (Fix I, 2026-04-30).
        element = line[76:78].strip() if len(line) > 76 else ""
        name = line[12:16].strip() if len(line) > 16 else ""
        if element in ("H", "HD") or (not element and name.startswith("H")):
            continue

        if (
            x < cx - half
            or x > cx + half
            or y < cy - half
            or y > cy + half
            or z < cz - half
            or z > cz + half
        ):
            return True

    return False


def _append_clipped_pose(path: Path, pose_idx: int, pdbqt_path: Path | None) -> None:
    """Append a clipped-pose entry to the run_metadata.json file.

    Reads an existing JSON file if present, appends to the "clipped_poses"
    list, and writes back atomically. Malformed JSON is silently overwritten.
    Parent directories are created if they do not exist.

    Args:
        path: Absolute path to the metadata JSON file.
        pose_idx: Index of the clipped pose.
        pdbqt_path: Path to the clipped pose's PDBQT file (may be None).
    """
    if not path.exists():
        data: dict = {}
    else:
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            data = {}

    data.setdefault("clipped_poses", []).append(
        {"pose_idx": pose_idx, "pdbqt_path": str(pdbqt_path)}
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)  # atomic on POSIX; overwrites destination


def score_vina_batch(
    poses: list[ScoredPose],
    config: DockConfig,
    receptor_pdbqt: Path,
    *,
    verbosity: int = 0,
    metadata_path: Path | None = None,
    optimize_clashing: bool = True,
    max_clash_relief_rounds: int = 5,
) -> tuple[list[ScoredPose], list[PoseFailure]]:
    """Thin wrapper around _score_vina_batch_impl that silences Vina's native
    stdout ("Computing Vina grid ... done.", "Performing local search ...
    done.") when verbosity == 0 — these are unconditional C++-layer prints
    that ignore the Vina(verbosity=...) constructor arg, so a Python log
    level can't reach them. See output/progress.suppress_native_stdout.
    """
    from hybridock_pep.output.progress import suppress_native_stdout  # noqa: PLC0415

    with suppress_native_stdout(enabled=(verbosity == 0)):
        return _score_vina_batch_impl(
            poses,
            config,
            receptor_pdbqt,
            verbosity=verbosity,
            metadata_path=metadata_path,
            optimize_clashing=optimize_clashing,
            max_clash_relief_rounds=max_clash_relief_rounds,
        )


#: Memory model of one Vina instance, measured on Vina 1.2.7 with a 164-atom rigid peptide:
#:   * grid maps: ~400 B per grid point (161^3 points = 1.6 GB RSS at a 60 A box);
#:   * ``set_ligand_from_file`` builds a per-atom-pair interaction table, so its transient
#:     peak grows with the SQUARE of the ligand atom count: 8.2 GB first call, 12.8 GB
#:     steady state at 164 atoms (~0.48 MB x n_atoms^2), independent of receptor and box.
_VINA_BYTES_PER_POINT = 400
_VINA_BYTES_PER_ATOM_SQ = 480_000
_VINA_FIXED_BYTES = 400 * 1024**2
_VINA_GRID_SPACING = 0.375  # Vina's default map granularity, Angstrom
#: Below this many poses the worker start-up + map build outweighs any speedup.
_MIN_POSES_FOR_POOL = 4
#: glibc malloc settings applied to pool workers only. Each ``set_ligand_from_file`` call
#: allocates then frees GBs; by default glibc mmap()s that and hands it back to the OS, so
#: every call pays page-fault cost again (slow on WSL2). Keeping it in the heap cut warm
#: calls from ~4.5 s to ~2.8 s. Trade-off: the worker's RSS stays at its peak, which the
#: memory model above already budgets for.
_WORKER_MALLOC_ENV = {
    "MALLOC_MMAP_MAX_": "0",
    "MALLOC_TRIM_THRESHOLD_": str(1 << 40),
    "MALLOC_TOP_PAD_": str(256 * 1024**2),
}

# (level, message) pairs recorded inside a worker and replayed by the parent so that
# log output is identical for sequential and parallel runs.
_Outcome = tuple[str, float | None, bool, str | None, list[tuple[int, str]]]


def _count_pdbqt_atoms(path: Path | None) -> int:
    """Number of ATOM/HETATM records in a PDBQT (0 if unreadable)."""
    if path is None:
        return 0
    try:
        with open(path) as fh:
            return sum(1 for ln in fh if ln.startswith(("ATOM", "HETATM")))
    except OSError:
        return 0


def _vina_worker_count(n_poses: int, box_size: float, n_ligand_atoms: int) -> int:
    """How many Vina worker processes the memory budget allows (0 = score in-process).

    A worker holds its own grid maps AND the quadratic ``set_ligand`` peak (see the
    memory model above), so memory — not cores — is the binding limit: ~15 GB per worker
    for a 17-mer, i.e. one worker under a 24 GB cap. The budget is cgroup-aware
    (``memory_budget_bytes``), 10% is held back, and when not even one worker fits the
    caller keeps the original in-process loop. ``HYBRIDOCK_VINA_WORKERS`` overrides the
    count (0/1 still honour the batch-size floor); it cannot exceed ``n_poses``.

    Args:
        n_poses: Number of poses to score.
        box_size: Grid box edge length in Angstrom.
        n_ligand_atoms: Atom count of a representative pose PDBQT.

    Returns:
        Worker count >= 0.
    """
    from hybridock_pep.hardware import cpu_threads, memory_budget_bytes  # noqa: PLC0415

    if n_poses < _MIN_POSES_FOR_POOL:
        return 0
    env = os.environ.get("HYBRIDOCK_VINA_WORKERS")
    if env and env.isdigit():
        return min(int(env), n_poses)
    points = (box_size / _VINA_GRID_SPACING + 1) ** 3
    per_worker = (points * _VINA_BYTES_PER_POINT + _VINA_FIXED_BYTES
                  + _VINA_BYTES_PER_ATOM_SQ * n_ligand_atoms ** 2)
    budget = memory_budget_bytes()
    by_memory = int((budget * 0.9) // per_worker) if budget is not None else 1
    return max(0, min(cpu_threads(), n_poses // 2 or 1, by_memory))


def _make_vina(receptor_pdbqt: str, site: tuple[float, float, float], box: float,
               cpu: int, verbosity: int) -> Any:
    """Build a Vina instance with the receptor loaded and Vina maps computed."""
    v = Vina(sf_name="vina", cpu=cpu, verbosity=verbosity)
    v.set_receptor(str(receptor_pdbqt))
    v.compute_vina_maps(center=list(site), box_size=[box] * 3)
    return v


def _score_one_pose(
    v: Any,
    pose_idx: int,
    pdbqt_path: Path | None,
    site: tuple[float, float, float],
    box: float,
    optimize_clashing: bool,
    max_clash_relief_rounds: int,
) -> _Outcome:
    """Score (and clash-relieve) one pose; never raises.

    Returns:
        ``(status, vina_score, is_clipped, error_msg, logs)`` where status is
        ``"scored"``, ``"clipped"`` or ``"failed"``.
    """
    logs: list[tuple[int, str]] = []
    is_clipped = False
    try:
        if pdbqt_path is None:
            raise ValueError(
                f"Pose {pose_idx} has pdbqt_path=None; "
                "was prep/ligand.py run before scoring?"
            )
        is_clipped = check_grid_boundary(pdbqt_path, site, box)
        if is_clipped:
            logs.append((logging.WARNING, (
                f"Pose {pose_idx}: atoms outside grid bounds (is_clipped=True) — "
                "skipping Vina scoring to avoid ~12s RuntimeError")))
            return ("clipped", None, True, "is_clipped: heavy atom outside grid bounds", logs)

        v.set_ligand_from_file(str(pdbqt_path))
        raw_score = float(v.score()[0])

        # Clash relief: locally optimize poses with positive Vina score (receptor overlap).
        # v.optimize() finds the nearest energy minimum in the Vina scoring function —
        # it moves the ligand just enough to relieve VDW clashes without global resampling.
        # Multi-round: repeat until score < 0, BFGS converged (< 0.5 kcal/mol improvement),
        # or max_clash_relief_rounds reached. The optimized PDBQT is written back so
        # AD4 scoring uses the same clash-free geometry.
        if optimize_clashing and raw_score > 0:
            prev_score = raw_score
            final_score = raw_score
            rounds_done = 0
            for round_num in range(1, max_clash_relief_rounds + 1):
                v.optimize()
                round_score = float(v.score()[0])
                improvement = prev_score - round_score
                logs.append((logging.DEBUG, (
                    f"Pose {pose_idx}: clash relief round {round_num}/"
                    f"{max_clash_relief_rounds}: {prev_score:.2f} → {round_score:.2f}"
                    f" kcal/mol (Δ={improvement:.2f})")))
                final_score = round_score
                rounds_done = round_num
                if round_score <= 0:
                    break  # clash fully resolved — stop early
                if improvement < 0.5:
                    break  # BFGS has converged; further rounds won't help
                prev_score = round_score

            logs.append((logging.INFO, (
                f"Pose {pose_idx}: clash relief {rounds_done} round(s): "
                f"{raw_score:.2f} → {final_score:.2f} kcal/mol")))

            if final_score > 0:
                # All rounds failed to relieve clash — ligand still inside receptor.
                # Recording a positive Vina score would corrupt ensemble z-score
                # statistics (the huge positive outliers inflate mean and std,
                # making well-scored poses appear as implausibly extreme outliers).
                # Treat as a scoring failure instead.
                logs.append((logging.WARNING, (
                    f"Pose {pose_idx}: clash relief failed after {rounds_done} "
                    f"round(s) ({raw_score:.2f} → {final_score:.2f} kcal/mol still "
                    "positive); excluding from scored set")))
                err = (
                    f"clash_relief_failed: Vina optimize ({rounds_done} rounds) "
                    f"reduced {raw_score:.2f} → {final_score:.2f} kcal/mol "
                    "but score remains positive (receptor overlap unresolved)"
                )
                return ("failed", None, False, err, logs)
            # Overwrite PDBQT with clash-free geometry (AD4 scoring reads this file)
            v.write_pose(str(pdbqt_path), overwrite=True)
            return ("scored", final_score, False, None, logs)
        return ("scored", raw_score, False, None, logs)

    except Exception as e:  # noqa: BLE001 — per-pose isolation required
        logs.append((logging.WARNING, f"Pose {pose_idx} scoring failed: {type(e).__name__}: {e}"))
        return ("failed", None, is_clipped, f"{type(e).__name__}: {e}", logs)


# --- worker-process state (one private Vina instance per worker) -------------------
_W_VINA: Any = None


def _worker_init(receptor_pdbqt: str, site: tuple[float, float, float], box: float,
                 verbosity: int) -> None:
    """ProcessPool initializer: build this worker's private Vina instance (1 thread)."""
    global _W_VINA
    _W_VINA = _make_vina(receptor_pdbqt, site, box, 1, verbosity)


def _worker_score(args: tuple[int, Path | None, tuple[float, float, float], float, bool, int]
                  ) -> _Outcome:
    """ProcessPool task: score one pose with this worker's Vina instance."""
    pose_idx, pdbqt_path, site, box, optimize_clashing, rounds = args
    return _score_one_pose(_W_VINA, pose_idx, pdbqt_path, site, box, optimize_clashing, rounds)


def _score_vina_batch_impl(
    poses: list[ScoredPose],
    config: DockConfig,
    receptor_pdbqt: Path,
    *,
    verbosity: int = 0,
    metadata_path: Path | None = None,
    optimize_clashing: bool = True,
    max_clash_relief_rounds: int = 5,
) -> tuple[list[ScoredPose], list[PoseFailure]]:
    """Score a batch of poses with Vina --score_only, in parallel across processes.

    Each pose is independent (``set_ligand_from_file`` -> ``score`` -> optional
    ``optimize``) and Vina's SWIG bindings are not thread-safe, so parallelism is by
    process: every worker builds its own Vina instance and maps once, then takes
    poses off a shared queue. Scoring is deterministic per pose, and results are
    reassembled in input order, so output equals the in-process path. Batches below
    ``_MIN_POSES_FOR_POOL`` poses, a memory budget too small for even one worker (e.g. a
    100 A blind-mode box under a 24 GB cap), or a worker-pool failure all use the
    original in-process loop.

    Clipped poses (atoms outside grid bounds) are flagged with is_clipped=True,
    a WARNING is logged, and the pose entry is appended to run_metadata.json
    (SCORE-01).

    When optimize_clashing=True (default): poses with initial score > 0 (positive =
    receptor-peptide clash or severe steric overlap) are locally optimized via up to
    max_clash_relief_rounds rounds of v.optimize(). Each round is a BFGS minimization
    from the current ligand position. Rounds stop early when:
      (a) the score drops below 0 (clash resolved), or
      (b) a round improves the score by less than 0.5 kcal/mol (BFGS converged).
    Multi-round relief recovers marginal clashes that need > 1 BFGS pass;
    deeply-embedded poses (> 50 kcal/mol initial) should instead be prevented at
    source via RAPiDock's no_final_step_noise=True flag.
    The optimized PDBQT is written back to pdbqt_path so that subsequent AD4
    scoring uses the same clash-free geometry.

    Args:
        poses: List of ScoredPose objects; each must have pdbqt_path set.
        config: Validated DockConfig supplying site_coords and box_size.
        receptor_pdbqt: Path to the prepared receptor PDBQT file.
        verbosity: Vina verbosity level (0=silent). Default 0.
        metadata_path: If provided, clipped pose entries are appended to this
            JSON file. Parent directories are created if absent.
        optimize_clashing: When True (default), locally optimize poses whose
            initial Vina score > 0 (indicating receptor clash) and update the
            PDBQT file with the clash-free geometry before final scoring.
        max_clash_relief_rounds: Maximum number of BFGS optimization rounds to
            attempt when the initial score is positive. Default 5.

    Returns:
        A tuple (scored, failures) where scored contains successfully scored
        ScoredPose objects and failures contains PoseFailure records for poses
        that raised an exception, both in input order.

    Raises:
        RuntimeError: Vina bindings not importable.
        Exception: Exceptions during Vina instance creation or receptor loading
            propagate to the caller (not silently swallowed); only per-pose
            scoring exceptions are caught.
    """
    if Vina is None:
        raise RuntimeError(
            "AutoDock Vina Python bindings are not importable in the active environment. "
            "The hybridock-pep CLI must be run from the 'score-env' conda environment "
            "(which provides `vina`), not the base environment. Activate it with "
            "`conda activate score-env` (or invoke "
            "`/path/to/envs/score-env/bin/hybridock-pep`) and re-run."
        )

    from hybridock_pep.output import progress as _progress  # noqa: PLC0415

    site = tuple(config.site_coords)
    box = float(config.box_size)
    n_total = len(poses)
    n_atoms = next((n for n in (_count_pdbqt_atoms(p.pdbqt_path) for p in poses[:3]) if n), 0)
    n_workers = _vina_worker_count(n_total, box, n_atoms)
    outcomes: list[_Outcome | None] = [None] * n_total

    logger.info(
        "Vina scorer: %d poses, receptor=%s, optimize_clashing=%s, max_clash_relief_rounds=%d, "
        "workers=%d (0 = in-process)",
        n_total, receptor_pdbqt, optimize_clashing, max_clash_relief_rounds, n_workers,
    )

    if n_workers >= 1:
        tasks = [(p.pose_idx, p.pdbqt_path, site, box, optimize_clashing,
                  max_clash_relief_rounds) for p in poses]
        saved_env = {k: os.environ.get(k) for k in _WORKER_MALLOC_ENV}
        os.environ.update(_WORKER_MALLOC_ENV)  # inherited by the spawned workers only
        try:
            # 'spawn', not fork: the parent may already hold CUDA/OpenMM contexts and
            # threads, neither of which survive a fork. Started inside the caller's
            # native-stdout suppression, so workers inherit the silenced fds.
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=get_context("spawn"),
                initializer=_worker_init,
                initargs=(str(receptor_pdbqt), site, box, verbosity),
            ) as pool:
                chunk = max(1, math.ceil(n_total / (n_workers * 8)))
                for i, out in enumerate(pool.map(_worker_score, tasks, chunksize=chunk)):
                    outcomes[i] = out
                    _progress.tick(i, n_total, "poses scored")
        except (BrokenProcessPool, OSError, MemoryError) as exc:
            done = sum(o is not None for o in outcomes)
            logger.warning(
                "Vina worker pool failed (%s: %s) after %d/%d poses — finishing "
                "the remainder sequentially.", type(exc).__name__, exc, done, n_total,
            )
        finally:
            for k, old in saved_env.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old

    if any(o is None for o in outcomes):
        v = _make_vina(str(receptor_pdbqt), site, box, _cpu_threads(), verbosity)
        for i, pose in enumerate(poses):
            if outcomes[i] is None:
                _progress.tick(i, n_total, "poses scored")
                outcomes[i] = _score_one_pose(v, pose.pose_idx, pose.pdbqt_path, site, box,
                                              optimize_clashing, max_clash_relief_rounds)

    scored: list[ScoredPose] = []
    failures: list[PoseFailure] = []
    for pose, out in zip(poses, outcomes):
        assert out is not None
        status, vina_score, is_clipped, err, logs = out
        for level, msg in logs:
            logger.log(level, msg)
        pose.is_clipped = is_clipped
        if status == "scored":
            pose.vina_score = vina_score
            scored.append(pose)
            continue
        if status == "clipped" and metadata_path is not None:
            _append_clipped_pose(metadata_path, pose.pose_idx, pose.pdbqt_path)
        failures.append(PoseFailure(pose_idx=pose.pose_idx, stage="scoring",
                                    error_msg=err or "unknown"))

    _progress.tick(n_total, n_total, "poses scored")
    _progress.clear()
    return scored, failures


def _cpu_threads() -> int:
    """Thread count for the single in-process Vina instance (map build parallelism)."""
    from hybridock_pep.hardware import cpu_threads  # noqa: PLC0415

    return cpu_threads()
