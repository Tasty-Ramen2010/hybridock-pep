"""RAPiDock subprocess orchestrator — score-env, Python 3.11.

Orchestrates Stage 1 of the HybriDock-Pep pipeline: N stochastic RAPiDock
inference passes executed directly via the rapidock env's Python 3 interpreter.
Streams stdout/stderr in real time so GPU OOM errors surface immediately. Renames
rank*.pdb output files to pose_0.pdb...pose_{N-1}.pdb for downstream stages.

Architecture:
- subprocess.Popen (not subprocess.run/communicate) for real-time streaming (D-01, D-02)
- stderr drained on a daemon thread to prevent pipe deadlock (D-01)
- All paths are resolved to absolute (D-07)
- Seed forwarded as --seed N only when DockConfig.seed is not None (D-08)
- RAPIDOCK_PYTHON env var overrides the rapidock Python 3 path; auto-detected otherwise
- RAPIDOCK_DIR and RAPIDOCK_MODEL_DIR/RAPIDOCK_CKPT env vars configure
  RAPiDock install location (Phase 5 will wire via DockConfig)

NOTE: We call the rapidock env's python3 directly rather than `conda run -n rapidock
python3`, because `conda run` does not reliably activate the target environment when
called from a Python subprocess that was itself launched with a non-default PATH.
Direct invocation avoids PATH-based python3 resolution and works correctly in all
contexts (terminal, pytest, CI, nested subprocess from the benchmark).
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

from hybridock_pep.models import DockConfig

logger = logging.getLogger(__name__)


def _stream_stderr(stderr_pipe) -> None:
    """Drain RAPiDock stderr line-by-line on a daemon thread; emit to logger.

    Must run on a daemon thread — the main thread reads stdout. Running both
    readline loops on the same thread would deadlock when the pipe buffers fill.

    Args:
        stderr_pipe: Opened stderr binary pipe from subprocess.Popen.
    """
    for raw_line in iter(stderr_pipe.readline, b""):
        line = raw_line.decode("utf-8", errors="replace").rstrip()
        if line:
            logger.debug("[rapidock stderr] %s", line)


_RAPIDOCK_SEARCH_PATHS = [
    Path.home() / "RAPiDock",
    Path("/opt/RAPiDock"),
]
_CKPT_DEFAULT = "rapidock_local.pt"
_CONDA_ENV_NAME = "rapidock"

_CONDA_BASE_SEARCH_PATHS = [
    Path.home() / "miniconda3",
    Path.home() / "miniforge3",
    Path.home() / "anaconda3",
    Path("/opt/conda"),
    Path("/opt/miniconda3"),
    Path("/opt/miniforge3"),
]


def _find_rapidock_python() -> str:
    """Resolve the Python 3 binary in the rapidock conda environment.

    RAPIDOCK_PYTHON env var takes priority. Otherwise searches standard conda
    base paths for envs/{_CONDA_ENV_NAME}/bin/python3.

    Calling the python3 binary directly (rather than `conda run -n rapidock
    python3`) avoids a conda run PATH-resolution bug where the activated
    environment's python3 is shadowed by the calling process's PATH.

    Returns:
        Absolute path string to the rapidock env's python3 binary.

    Raises:
        RuntimeError: If the binary cannot be located.
    """
    override = os.environ.get("RAPIDOCK_PYTHON")
    if override:
        return override

    # CONDA_EXE gives us the conda binary; its grandparent is the base prefix
    conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda_exe:
        conda_base = Path(conda_exe).resolve().parent.parent
        candidate = conda_base / "envs" / _CONDA_ENV_NAME / "bin" / "python3"
        if candidate.exists():
            logger.debug("Located rapidock python3 via CONDA_EXE: %s", candidate)
            return str(candidate)

    for base in _CONDA_BASE_SEARCH_PATHS:
        candidate = base / "envs" / _CONDA_ENV_NAME / "bin" / "python3"
        if candidate.exists():
            logger.debug("Located rapidock python3 at: %s", candidate)
            return str(candidate)

    raise RuntimeError(
        f"Cannot locate Python 3 in conda env '{_CONDA_ENV_NAME}'. "
        "Set RAPIDOCK_PYTHON to the full absolute path of the python3 binary "
        f"(e.g. ~/miniconda3/envs/{_CONDA_ENV_NAME}/bin/python3), "
        "or ensure the environment exists."
    )


def _find_rapidock_dir() -> Path:
    """Resolve the RAPiDock source directory.

    Checks RAPIDOCK_DIR env var first, then common install locations
    (~~/RAPiDock, /opt/RAPiDock). Raises RuntimeError if none found.

    Returns:
        Absolute Path to the RAPiDock source directory (contains inference.py).

    Raises:
        RuntimeError: If RAPiDock cannot be located.
    """
    rapidock_dir = os.environ.get("RAPIDOCK_DIR")
    if rapidock_dir:
        return Path(rapidock_dir).resolve()
    for candidate in _RAPIDOCK_SEARCH_PATHS:
        if (candidate / "inference.py").exists():
            logger.debug("Auto-detected RAPiDock at %s", candidate)
            return candidate.resolve()
    raise RuntimeError(
        "Cannot locate RAPiDock. Set RAPIDOCK_DIR to the RAPiDock source directory "
        "(the one containing inference.py), or install it at ~/RAPiDock."
    )


def _find_model_dir() -> Path:
    """Resolve the RAPiDock model directory (train_models/CGTensorProductEquivariantModel/).

    Checks RAPIDOCK_MODEL_DIR env var first, then derives the standard path
    from the RAPiDock source directory located by _find_rapidock_dir().

    Returns:
        Absolute Path to train_models/CGTensorProductEquivariantModel/.

    Raises:
        RuntimeError: If the model directory cannot be found.
    """
    model_dir = os.environ.get("RAPIDOCK_MODEL_DIR")
    if model_dir:
        return Path(model_dir).resolve()
    rapidock_dir = _find_rapidock_dir()
    derived = rapidock_dir / "train_models" / "CGTensorProductEquivariantModel"
    if derived.exists():
        logger.debug("Auto-detected model dir at %s", derived)
        return derived.resolve()
    raise RuntimeError(
        f"Model directory not found at {derived}. "
        "Set RAPIDOCK_MODEL_DIR to train_models/CGTensorProductEquivariantModel/ "
        "inside your RAPiDock install."
    )


def _find_ckpt_name() -> str:
    """Resolve the RAPiDock checkpoint filename.

    Checks RAPIDOCK_CKPT env var first, then defaults to rapidock_local.pt.

    Returns:
        Checkpoint filename string (e.g., 'rapidock_local.pt').
    """
    return os.environ.get("RAPIDOCK_CKPT", _CKPT_DEFAULT)


def run_sampling(config: DockConfig, receptor_path: Path | None = None) -> list[Path]:
    """Run RAPiDock N=config.n_samples inference passes via direct python3 subprocess.

    Calls the rapidock conda env's python3 binary directly (bypassing conda run)
    to avoid a PATH-resolution bug where conda run picks the caller's python3
    instead of the target environment's. See module docstring for details.

    All paths are resolved to absolute (D-07).

    Streaming: stdout read in main thread readline() loop; stderr on daemon
    thread to prevent pipe deadlock (D-01). Both use iter(pipe.readline, b"")
    sentinel pattern.

    Renaming: RAPiDock writes rank*.pdb to {output_dir}/poses_raw/poses_raw/.
    These are renamed to pose_0.pdb...pose_{N-1}.pdb under {output_dir}/poses/
    (D-09, D-10, D-11).

    Args:
        config: Validated DockConfig. Uses peptide_sequence, receptor_path,
                output_dir, n_samples, seed.
        receptor_path: Optional override for the receptor PDB path. If None,
                uses config.receptor_path. Pass a pdbfixer-cleaned PDB here
                to avoid MDAnalysis chain-splitting issues on raw RCSB downloads.

    Returns:
        List of absolute Paths to renamed pose_*.pdb files under
        config.output_dir/poses/.

    Raises:
        RuntimeError: If RAPiDock subprocess exits non-zero (D-03).
        RuntimeError: If zero poses are produced after subprocess exits (D-11).
    """
    rapidock_python = _find_rapidock_python()
    shim_path = str((Path(__file__).resolve().parent / "run_rapidock.py"))
    effective_receptor = receptor_path if receptor_path is not None else config.receptor_path
    receptor_abs = str(effective_receptor.resolve())
    raw_output_abs = str((config.output_dir / "poses_raw").resolve())
    rapidock_dir_abs = str(_find_rapidock_dir())
    model_dir_abs = str(_find_model_dir())
    ckpt_name = _find_ckpt_name()

    cmd = [
        rapidock_python, shim_path,
        "--peptide", config.peptide_sequence,
        "--receptor", receptor_abs,
        "--output-dir", raw_output_abs,
        "--n-samples", str(config.n_samples),
        "--rapidock-dir", rapidock_dir_abs,
        "--model-dir", model_dir_abs,
        "--ckpt", ckpt_name,
        "--scoring-function", "none",
    ]
    if config.seed is not None:
        cmd += ["--seed", str(config.seed)]

    logger.info("Running: %s", " ".join(str(c) for c in cmd))

    # Popen with pipes — bytes mode (no text=True); both pipes needed for streaming (D-01)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Drain stderr on daemon thread — prevents pipe buffer deadlock when stderr fills (D-01)
    t = threading.Thread(target=_stream_stderr, args=(proc.stderr,), daemon=True)
    t.start()

    # Drain stdout on main thread — readline sentinel loop (D-02)
    for raw_line in iter(proc.stdout.readline, b""):
        line = raw_line.decode("utf-8", errors="replace").rstrip()
        if line:
            logger.debug("[rapidock stdout] %s", line)

    proc.wait()
    t.join()

    # Non-zero exit is always a fatal error — no retry (D-03)
    if proc.returncode != 0:
        raise RuntimeError(
            f"RAPiDock subprocess exited with code {proc.returncode}"
        )

    # Rename rank*.pdb → pose_N.pdb (D-09, D-10, D-11)
    # RAPiDock writes to {output_dir}/{complex_name}/ where complex_name="poses_raw"
    # so raw files are at: {output_dir}/poses_raw/poses_raw/rank*.pdb
    raw_dir = config.output_dir / "poses_raw" / "poses_raw"
    poses_dir = config.output_dir / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)

    rank_files = sorted(
        raw_dir.glob("rank*.pdb"),
        key=lambda p: int(re.search(r"rank(\d+)", p.stem).group(1)),  # type: ignore[union-attr]
    )

    renamed: list[Path] = []
    for i, src in enumerate(rank_files):
        dst = poses_dir / f"pose_{i}.pdb"
        src.rename(dst)
        renamed.append(dst)

    logger.info("Renamed %d rank*.pdb → pose_*.pdb in %s", len(renamed), poses_dir)

    # Zero poses = hard failure (D-11)
    if len(renamed) == 0:
        raise RuntimeError(
            f"RAPiDock produced 0 poses in {raw_dir}. Check stderr logs above."
        )

    # Shortfall = warning only (D-09); caller decides what to do
    if len(renamed) < config.n_samples:
        logger.warning(
            "RAPiDock pose shortfall: requested %d, generated %d",
            config.n_samples,
            len(renamed),
        )

    return renamed
