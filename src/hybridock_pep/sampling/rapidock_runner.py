"""RAPiDock subprocess orchestrator — score-env, Python 3.11.

Orchestrates Stage 1 of the HybriDock-Pep pipeline: N stochastic RAPiDock
inference passes executed inside `rapidock-env` via `conda run`. Streams
stdout/stderr in real time so GPU OOM errors surface immediately. Renames
rank*.pdb output files to pose_0.pdb...pose_{N-1}.pdb for downstream stages.

Architecture:
- subprocess.Popen (not subprocess.run/communicate) for real-time streaming (D-01, D-02)
- stderr drained on a daemon thread to prevent pipe deadlock (D-01)
- All paths crossing the conda boundary are resolved to absolute (D-07)
- Seed forwarded as --seed N only when DockConfig.seed is not None (D-08)
- RAPIDOCK_DIR and RAPIDOCK_MODEL_DIR/RAPIDOCK_CKPT env vars configure
  RAPiDock install location (Phase 5 will wire via DockConfig)
"""
from __future__ import annotations

import logging
import os
import re
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


_RAPIDOCK_DIR_UNSET = "/tmp/rapidock_not_configured"
_MODEL_DIR_UNSET = "/tmp/rapidock_model_not_configured"
_CKPT_UNSET = "rapidock_not_configured.pt"


def _find_rapidock_dir() -> Path:
    """Resolve the RAPiDock source directory from the RAPIDOCK_DIR env var.

    Phase 5 will wire this through DockConfig. When RAPIDOCK_DIR is not set,
    returns a placeholder path and logs a warning — the subprocess will fail
    if actually invoked without this, but command construction succeeds.

    Returns:
        Absolute Path to the RAPiDock source directory (contains inference.py).
    """
    rapidock_dir = os.environ.get("RAPIDOCK_DIR")
    if not rapidock_dir:
        logger.warning(
            "RAPIDOCK_DIR env var not set; using placeholder. "
            "Set RAPIDOCK_DIR to the RAPiDock install directory before running."
        )
        return Path(_RAPIDOCK_DIR_UNSET)
    return Path(rapidock_dir).resolve()


def _find_model_dir() -> Path:
    """Resolve the RAPiDock model directory from RAPIDOCK_MODEL_DIR env var.

    When RAPIDOCK_MODEL_DIR is not set, returns a placeholder path and logs
    a warning.

    Returns:
        Absolute Path to train_models/ directory inside RAPiDock install.
    """
    model_dir = os.environ.get("RAPIDOCK_MODEL_DIR")
    if not model_dir:
        logger.warning(
            "RAPIDOCK_MODEL_DIR env var not set; using placeholder. "
            "Set RAPIDOCK_MODEL_DIR before running."
        )
        return Path(_MODEL_DIR_UNSET)
    return Path(model_dir).resolve()


def _find_ckpt_name() -> str:
    """Resolve the RAPiDock checkpoint filename from RAPIDOCK_CKPT env var.

    When RAPIDOCK_CKPT is not set, returns a placeholder string and logs
    a warning.

    Returns:
        Checkpoint filename string (e.g., 'rapidock_local.pt').
    """
    ckpt = os.environ.get("RAPIDOCK_CKPT")
    if not ckpt:
        logger.warning(
            "RAPIDOCK_CKPT env var not set; using placeholder. "
            "Set RAPIDOCK_CKPT to the checkpoint filename before running."
        )
        return _CKPT_UNSET
    return ckpt


def run_sampling(config: DockConfig) -> list[Path]:
    """Run RAPiDock N=config.n_samples inference passes via conda run subprocess.

    Executes: conda run --no-capture-output -n rapidock-env python {run_rapidock.py} [args]

    All file paths crossing the conda boundary are resolved to absolute (D-07).

    Streaming: stdout read in main thread readline() loop; stderr on daemon
    thread to prevent pipe deadlock (D-01). Both use iter(pipe.readline, b"")
    sentinel pattern.

    Renaming: RAPiDock writes rank*.pdb to {output_dir}/poses_raw/poses_raw/.
    These are renamed to pose_0.pdb...pose_{N-1}.pdb under {output_dir}/poses/
    (D-09, D-10, D-11).

    Args:
        config: Validated DockConfig. Uses peptide_sequence, receptor_path,
                output_dir, n_samples, seed.

    Returns:
        List of absolute Paths to renamed pose_*.pdb files under
        config.output_dir/poses/.

    Raises:
        RuntimeError: If RAPiDock subprocess exits non-zero (D-03).
        RuntimeError: If zero poses are produced after subprocess exits (D-11).
    """
    # Resolve all paths to absolute before crossing the conda boundary (D-07)
    shim_path = str((Path(__file__).resolve().parent / "run_rapidock.py"))
    receptor_abs = str(config.receptor_path.resolve())
    raw_output_abs = str((config.output_dir / "poses_raw").resolve())
    rapidock_dir_abs = str(_find_rapidock_dir())
    model_dir_abs = str(_find_model_dir())
    ckpt_name = _find_ckpt_name()

    cmd = [
        "conda", "run", "--no-capture-output", "-n", "rapidock-env",
        "python", shim_path,
        "--peptide", config.peptide_sequence,
        "--receptor", receptor_abs,
        "--output-dir", raw_output_abs,
        "--n-samples", str(config.n_samples),
        "--rapidock-dir", rapidock_dir_abs,
        "--model-dir", model_dir_abs,
        "--ckpt", ckpt_name,
        "--scoring-function", "confidence",
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
