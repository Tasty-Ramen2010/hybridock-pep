"""Provenance metadata writer for HybriDock-Pep runs (SAMP-02).

Implements the two-write pattern (D-15):
1. write_metadata_skeleton() — written BEFORE conda run launches. Contains
   status="running" so a crash leaves a diagnosable partial record.
2. finalize_metadata() — written AFTER pose_io parsing completes. Uses
   read-modify-write (same atomic pattern as scoring/vina.py _append_clipped_pose)
   to preserve any clipped_poses entries written during scoring (Pitfall 6).

Required fields (D-16): git_sha, rapidock_commit_sha, cli_args, seed,
vina_version, openmm_version, cuda_version, receptor_sha256,
peptide_sequence_hash, timestamp_start, timestamp_end,
poses_requested, poses_generated, status.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from hybridock_pep.models import DockConfig

logger = logging.getLogger(__name__)


def write_metadata_skeleton(config: DockConfig, metadata_path: Path) -> None:
    """Write initial run_metadata.json before sampling starts (D-15).

    Contains status="running" and all fields knowable before inference:
    git_sha, rapidock_commit_sha, receptor_sha256, peptide_sequence_hash,
    vina_version, openmm_version, cli_args, seed, poses_requested,
    timestamp_start, cuda_version (None until rapidock-env reports it).

    Args:
        config: Validated DockConfig for this run.
        metadata_path: Absolute path to write the JSON file.
    """
    data = {
        "status": "running",
        "timestamp_start": datetime.now(tz=timezone.utc).isoformat(),
        "poses_requested": config.n_samples,
        "seed": config.seed,
        "cli_args": config.model_dump(mode="json"),
        "git_sha": _get_git_sha(),
        "rapidock_commit_sha": get_rapidock_commit_sha(),
        "receptor_sha256": _sha256_file(config.receptor_path),
        "peptide_sequence_hash": hashlib.sha256(
            config.peptide_sequence.encode()
        ).hexdigest(),
        "vina_version": _get_vina_version(),
        "openmm_version": _get_openmm_version(),
        "cuda_version": _detect_cuda_driver_version(),
    }
    _write_json_atomic(metadata_path, data)
    logger.info("Metadata skeleton written to %s", metadata_path)


def finalize_metadata(
    metadata_path: Path,
    poses_generated: int,
    cuda_version: Optional[str] = None,
    status: str = "complete",
) -> None:
    """Overwrite run_metadata.json with final counts and status (D-15).

    Uses read-modify-write to preserve any clipped_poses entries that
    scoring/vina.py _append_clipped_pose() may have written (Pitfall 6).

    Args:
        metadata_path: Path to the existing skeleton JSON (may not exist if
                       crash occurred before skeleton write).
        poses_generated: Actual count of successfully parsed PoseRecord objects.
        cuda_version: CUDA version string from rapidock-env (may be None).
        status: Final status string; "complete" or "failed".
    """
    if metadata_path.exists():
        try:
            data = json.loads(metadata_path.read_text())
        except (json.JSONDecodeError, OSError):
            data = {}
    else:
        data = {}

    data["status"] = status
    data["timestamp_end"] = datetime.now(tz=timezone.utc).isoformat()
    data["poses_generated"] = poses_generated
    if cuda_version is not None:
        data["cuda_version"] = cuda_version

    _write_json_atomic(metadata_path, data)
    logger.info(
        "Metadata finalized: %d/%s poses, status=%s",
        poses_generated,
        data.get("poses_requested", "?"),
        status,
    )


def get_rapidock_commit_sha() -> str:
    """Return the git commit SHA of the installed RAPiDock package (PEP 610).

    Reads from direct_url.json in the .dist-info directory, which pip writes
    when installing from a git URL (pip install git+https://...).

    Returns:
        Commit SHA string from direct_url.json vcs_info, or "unknown" if
        the package is not installed from a git URL or not installed at all.
    """
    try:
        dist = importlib.metadata.distribution("rapidock")
        for f in dist.files or []:
            if f.name == "direct_url.json":
                raw = f.read_text(encoding="utf-8")
                data = json.loads(raw)
                sha = data.get("vcs_info", {}).get("commit_id", "unknown")
                return str(sha) if sha else "unknown"
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not determine RAPiDock commit SHA: %s", e)
    return "unknown"


def _write_json_atomic(path: Path, data: dict) -> None:
    """Atomically write data as JSON to path using a .tmp intermediate file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    """Return hex SHA256 digest of the file at path."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _get_git_sha() -> Optional[str]:
    """Return current repo HEAD SHA, or None if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except FileNotFoundError:
        return None


def _get_vina_version() -> Optional[str]:
    """Return Vina version string from the installed Python package, or None."""
    try:
        return importlib.metadata.version("vina")
    except importlib.metadata.PackageNotFoundError:
        return None


def _detect_cuda_driver_version() -> Optional[str]:
    """Query CUDA driver version from nvidia-smi, or None on non-NVIDIA hosts."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return f"driver/{result.stdout.strip().splitlines()[0].strip()}"
    except FileNotFoundError:
        pass
    return None


def _get_openmm_version() -> Optional[str]:
    """Return OpenMM version string, or None if not installed."""
    try:
        import openmm  # type: ignore[import]
        return str(openmm.__version__)
    except ImportError:
        return None
