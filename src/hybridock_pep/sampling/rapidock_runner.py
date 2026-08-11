"""RAPiDock subprocess orchestrator — score-env, Python 3.11.

Orchestrates Stage 1 of the HybriDock-Pep pipeline: N stochastic RAPiDock
inference passes executed directly via the rapidock env's Python 3 interpreter.
Streams stdout/stderr in real time so GPU OOM errors surface immediately. Renames
rank*.pdb output files to pose_0.pdb...pose_{N-1}.pdb for downstream stages.

Architecture:
- subprocess.Popen (not subprocess.run/communicate) for real-time streaming
- stderr drained on a daemon thread to prevent pipe deadlock
- All paths are resolved to absolute
- Seed forwarded as --seed N only when DockConfig.seed is not None
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
import platform
import shutil
import subprocess
from collections import deque
import sys
import threading
from pathlib import Path

from hybridock_pep.models import DockConfig

logger = logging.getLogger(__name__)


def _detect_device_platform() -> str:
    """Return a human-readable string describing the expected inference device.

    Used for log messages and user-facing warnings.  Detection priority:
      1. NVIDIA CUDA  — nvidia-smi present
      2. AMD ROCm     — rocminfo or rocm-smi present, or /dev/kfd exists
      3. Intel XPU    — sycl-ls present (Intel Level-Zero / oneAPI driver)
      4. Apple MPS    — macOS arm64
      5. CPU fallback

    ROCm note: PyTorch ROCm uses the CUDA API (torch.cuda.is_available() returns
    True on ROCm).  We detect it here via CLI tools rather than torch to avoid
    importing torch in the score-env process.

    Returns:
        Human-readable device label, e.g. "CUDA — NVIDIA (Linux/WSL2)",
        "ROCm — AMD GPU (Linux)", "XPU — Intel GPU (Linux)",
        "MPS (macOS Apple Silicon)", or "CPU".
    """
    if sys.platform == "darwin":
        machine = platform.machine()  # "arm64" on Apple Silicon, "x86_64" on Intel
        if machine == "arm64":
            return "MPS (macOS Apple Silicon)"
        return "CPU (macOS Intel — MPS not available on x86_64)"

    # Linux / WSL2 — check GPU backends in priority order.
    # WSL2 quirk: the WSL kernel exposes the host GPU via /usr/lib/wsl/lib/
    # (libcuda.so.1, nvidia-smi) but does NOT add that dir to PATH and does
    # NOT create /dev/nvidia* or /proc/driver/nvidia/. So shutil.which() and
    # the usual procfs probes both miss the GPU even when PyTorch can see it.
    # We check for the WSL libcuda stub and the WSL-mounted nvidia-smi binary
    # in addition to PATH.
    import os as _os
    _WSL_NVIDIA_PATHS = (
        "/usr/lib/wsl/lib/nvidia-smi",
        "/usr/lib/wsl/lib/libcuda.so.1",
    )
    # WSL2 fast path: explicit libcuda or nvidia-smi in /usr/lib/wsl/lib/
    # means the WSL kernel exposes the host GPU; no need to spawn a probe.
    if any(_os.path.exists(p) for p in _WSL_NVIDIA_PATHS):
        return "CUDA — NVIDIA GPU (Linux/WSL2)"
    # Native Linux: nvidia-smi on PATH.
    if shutil.which("nvidia-smi") is not None:
        try:
            import subprocess as _sp
            _sp.run(["nvidia-smi"], capture_output=True, timeout=5, check=True)
            return "CUDA — NVIDIA GPU (Linux/WSL2)"
        except Exception:
            pass  # nvidia-smi present but failed; continue

    # AMD ROCm: rocminfo (preferred) → rocm-smi → /dev/kfd kernel device
    if shutil.which("rocminfo") is not None:
        return "ROCm — AMD GPU (Linux)"
    if shutil.which("rocm-smi") is not None:
        return "ROCm — AMD GPU (Linux)"
    import pathlib as _pl
    if _pl.Path("/dev/kfd").exists():
        return "ROCm — AMD GPU (Linux, /dev/kfd)"

    # Intel XPU: sycl-ls from Intel oneAPI Base Toolkit
    if shutil.which("sycl-ls") is not None:
        return "XPU — Intel GPU (Linux)"

    return "CPU (Linux — no GPU detected)"


#: Trailing RAPiDock stderr kept in memory so a failure can quote it. Sized to
#: hold a Python traceback plus surrounding context without unbounded growth
#: (RAPiDock prints a progress bar to stderr on every step).
_STDERR_TAIL_LINES = 40


def _stream_stderr(stderr_pipe, sink: deque | None = None) -> None:
    """Drain RAPiDock stderr line-by-line on a daemon thread; emit to logger.

    Must run on a daemon thread — the main thread reads stdout. Running both
    readline loops on the same thread would deadlock when the pipe buffers fill.

    Lines also go into `sink`, a bounded deque, because they are logged at
    DEBUG: at the default log level "check stderr logs above" pointed the user
    at output that was never printed. Keeping the tail lets the failure quote
    the actual cause. (A GB10 host failed here with an NVRTC arch error that
    was invisible without -vv.)

    Args:
        stderr_pipe: Opened stderr binary pipe from subprocess.Popen.
        sink: Optional bounded deque collecting the trailing lines.
    """
    for raw_line in iter(stderr_pipe.readline, b""):
        line = raw_line.decode("utf-8", errors="replace").rstrip()
        if line:
            logger.debug("[rapidock stderr] %s", line)
            if sink is not None:
                sink.append(line)


# Repo-bundled submodule: src/hybridock_pep/sampling/ → src/hybridock_pep/ → src/ → repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_RAPIDOCK_SEARCH_PATHS = [
    _REPO_ROOT / "third_party" / "RAPiDock",  # bundled submodule (preferred)
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


#: Filename of the long/very_long-specialized checkpoint (V6 3-phase cross_conv
#: retrain, see logs/v6_run/ and docs/architecture.md "Long-peptide checkpoint").
#: Optional — a fresh install only ships rapidock_local.pt/rapidock_global.pt;
#: this file is dropped in by the user (or a future release asset) when available.
LONGER_CKPT_NAME = "longer_local.pt"


def select_checkpoint(peptide_len: int, threshold: int, model_dir: Path) -> str:
    """Route to the long-peptide checkpoint when the peptide is long enough and
    the checkpoint is actually present; otherwise use the standard local checkpoint.

    RAPIDOCK_CKPT always wins when set — this is an explicit power-user override
    (used throughout scripts/nsweep_coverage.py-style tooling to point at arbitrary
    fine-tuned checkpoints) and must not be silently second-guessed by length
    routing.

    Args:
        peptide_len: Peptide length in residues.
        threshold: Length at/above which longer_local.pt is preferred
            (DockConfig.long_checkpoint_threshold).
        model_dir: Directory expected to contain both checkpoint files.

    Returns:
        Checkpoint filename to pass as RAPiDock's --ckpt.
    """
    override = os.environ.get("RAPIDOCK_CKPT")
    if override:
        return override
    if peptide_len < threshold:
        return _CKPT_DEFAULT
    longer_path = model_dir / LONGER_CKPT_NAME
    if longer_path.exists():
        logger.info(
            "Peptide length %d ≥ threshold %d — routing to %s",
            peptide_len, threshold, LONGER_CKPT_NAME,
        )
        return LONGER_CKPT_NAME
    logger.warning(
        "Peptide length %d ≥ threshold %d would route to %s, but it is not "
        "present in %s — falling back to %s. See INSTALL.md for how to add it.",
        peptide_len, threshold, LONGER_CKPT_NAME, model_dir, _CKPT_DEFAULT,
    )
    return _CKPT_DEFAULT


def run_sampling(
    config: DockConfig,
    receptor_path: Path | None = None,
    *,
    n_samples: int | None = None,
    ckpt_name: str | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Run RAPiDock N inference passes via direct python3 subprocess.

    Calls the rapidock conda env's python3 binary directly (bypassing conda run)
    to avoid a PATH-resolution bug where conda run picks the caller's python3
    instead of the target environment's. See module docstring for details.

    All paths are resolved to absolute.

    Streaming: stdout read in main thread readline() loop; stderr on daemon
    thread to prevent pipe deadlock. Both use iter(pipe.readline, b"")
    sentinel pattern.

    Renaming: RAPiDock writes rank*.pdb to {output_dir}/poses_raw/poses_raw/.
    These are renamed to pose_0.pdb...pose_{N-1}.pdb under {output_dir}/poses/.

    Args:
        config: Validated DockConfig. Uses peptide_sequence, receptor_path,
                output_dir, n_samples, seed (n_samples/output_dir are overridable
                below for the pocket-search multi-stage flow — see
                sampling/pocket_search.py).
        receptor_path: Optional override for the receptor PDB path. If None,
                uses config.receptor_path. Pass a pdbfixer-cleaned PDB here
                to avoid MDAnalysis chain-splitting issues on raw RCSB downloads,
                or a pocket-cropped PDB for a pocket-search refinement stage.
        n_samples: Optional override for the number of poses (defaults to
                config.n_samples). Used by pocket_search.py to run the N=300
                exploratory pass and the N=150-per-pocket refinement passes,
                which both differ from the run's headline --n-samples.
        ckpt_name: Optional override for the RAPiDock checkpoint filename
                (defaults to length-routed selection via select_checkpoint(),
                which itself still honors RAPIDOCK_CKPT). Used by
                pocket_search.py to force rapidock_global.pt for the
                exploratory pass.
        output_dir: Optional override for where poses_raw/ and poses/ are
                written (defaults to config.output_dir). Used by
                pocket_search.py so each stage's poses land in their own
                subdirectory instead of overwriting one another.

    Returns:
        List of absolute Paths to renamed pose_*.pdb files under
        {output_dir or config.output_dir}/poses/.

    Raises:
        RuntimeError: If RAPiDock subprocess exits non-zero.
        RuntimeError: If zero poses are produced after subprocess exits.
    """
    rapidock_python = _find_rapidock_python()
    shim_path = str((Path(__file__).resolve().parent / "run_rapidock.py"))
    effective_receptor = receptor_path if receptor_path is not None else config.receptor_path
    receptor_abs = str(effective_receptor.resolve())
    effective_output_dir = output_dir if output_dir is not None else config.output_dir
    raw_output_abs = str((effective_output_dir / "poses_raw").resolve())
    rapidock_dir_abs = str(_find_rapidock_dir())
    model_dir_abs = str(_find_model_dir())
    effective_n_samples = n_samples if n_samples is not None else config.n_samples
    if ckpt_name is not None:
        effective_ckpt = ckpt_name
    else:
        effective_ckpt = select_checkpoint(
            len(config.peptide_sequence),
            config.long_checkpoint_threshold,
            Path(model_dir_abs),
        )
    ckpt_name = effective_ckpt

    device_label = _detect_device_platform()
    logger.info("Stage 1: RAPiDock-Reloaded sampling — device: %s", device_label)
    if sys.platform == "darwin" and platform.machine() == "arm64":
        logger.info(
            "macOS Apple Silicon detected — using MPS backend. "
            "PYTORCH_ENABLE_MPS_FALLBACK=1 is set by inference.py for ops not yet on Metal. "
            "For full 100-pose runs, a CUDA machine is faster (~10× vs MPS). "
            "Use --n-samples 20 for quick exploratory runs on MPS."
        )

    cmd = [
        rapidock_python, shim_path,
        "--peptide", config.peptide_sequence,
        "--receptor", receptor_abs,
        "--output-dir", raw_output_abs,
        "--n-samples", str(effective_n_samples),
        "--rapidock-dir", rapidock_dir_abs,
        "--model-dir", model_dir_abs,
        "--ckpt", ckpt_name,
        "--scoring-function", "none",
    ]
    if config.seed is not None:
        cmd += ["--seed", str(config.seed)]

    logger.info("Running: %s", " ".join(str(c) for c in cmd))

    # Popen with pipes — bytes mode (no text=True); both pipes needed for streaming
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Drain stderr on daemon thread — prevents pipe buffer deadlock when stderr fills
    stderr_tail: deque = deque(maxlen=_STDERR_TAIL_LINES)
    t = threading.Thread(
        target=_stream_stderr, args=(proc.stderr, stderr_tail), daemon=True
    )
    t.start()

    # Drain stdout on main thread — readline sentinel loop.
    # The shim emits "[[HDP-PROGRESS]] <done> <total>" once per diffusion
    # step/batch; render those as a live bar so the longest silent stage of the
    # run visibly moves. Anything else stays DEBUG chatter.
    from hybridock_pep.output import progress as _progress  # noqa: PLC0415
    from hybridock_pep.sampling.run_rapidock import PROGRESS_PREFIX  # noqa: PLC0415

    saw_progress = False
    for raw_line in iter(proc.stdout.readline, b""):
        line = raw_line.decode("utf-8", errors="replace").rstrip()
        if not line:
            continue
        if line.startswith(PROGRESS_PREFIX):
            try:
                done_s, total_s = line[len(PROGRESS_PREFIX):].split()
                _progress.tick(int(done_s), int(total_s), "denoising steps")
                saw_progress = True
            except ValueError:
                logger.debug("[rapidock stdout] malformed progress line: %s", line)
            continue
        logger.debug("[rapidock stdout] %s", line)

    if saw_progress:
        _progress.clear()

    proc.wait()
    t.join()

    # Non-zero exit is always a fatal error — no retry
    if proc.returncode != 0:
        device_hint = ""
        if sys.platform == "darwin":
            device_hint = (
                " On macOS, verify: (1) the rapidock env has PyTorch installed "
                "(conda run -n rapidock python -c 'import torch; print(torch.__version__)'), "
                "(2) torch-scatter/torch-sparse are installed. "
                "See envs/rapidock-env-macos.yml for the install sequence."
            )
        raise RuntimeError(
            f"RAPiDock subprocess exited with code {proc.returncode}.{device_hint}"
        )

    # Rename rank*.pdb → pose_N.pdb (
    # RAPiDock writes to {output_dir}/{complex_name}/ where complex_name="poses_raw"
    # so raw files are at: {output_dir}/poses_raw/poses_raw/rank*.pdb
    raw_dir = effective_output_dir / "poses_raw" / "poses_raw"
    poses_dir = effective_output_dir / "poses"
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

    # Zero poses = hard failure
    if len(renamed) == 0:
        tail = "\n".join(f"    {ln}" for ln in stderr_tail)
        detail = (
            f"\n\nLast {len(stderr_tail)} lines of RAPiDock stderr:\n{tail}"
            if stderr_tail
            else "\n\nRAPiDock printed nothing to stderr. Re-run with -vv for full logs."
        )
        raise RuntimeError(
            f"RAPiDock produced 0 poses in {raw_dir}.{detail}"
        )

    # Shortfall = warning only; caller decides what to do
    if len(renamed) < effective_n_samples:
        logger.warning(
            "RAPiDock pose shortfall: requested %d, generated %d",
            effective_n_samples,
            len(renamed),
        )

    return renamed
