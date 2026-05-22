from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


_VALID_AA: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")


class DockConfig(BaseModel):
    """Validated configuration for a single docking run.

    All fields are validated at construction time so that bad inputs surface
    BEFORE any subprocess is spawned. This is the first line of defence for
    CLI-02 (pre-subprocess validation).

    Args:
        peptide_sequence: Single-letter amino acid sequence; must contain only
            the 20 standard AAs (ACDEFGHIKLMNPQRSTVWY). Coerced to uppercase.
        receptor_path: Path to the receptor PDB; must exist on disk.
        site_coords: (x, y, z) grid box center in Angstrom.
        box_size: Grid box edge length in Angstrom.
        n_samples: Number of RAPiDock inference passes. Default 100.
        seed: Optional seed for deterministic sampling. Default None.
        scoring: Set of scoring backends to run in parallel. Default
            {"vina", "ad4"}.
        output_dir: Directory where run outputs are written.
        run_id: Per-run identifier. Auto-generated (timestamp + seed hash)
            if omitted.
        verbosity: argparse -v count. Default 0.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=False)

    peptide_sequence: str
    receptor_path: Path
    site_coords: tuple[float, float, float]
    box_size: float
    n_samples: int = 100
    seed: int | None = None
    scoring: set[Literal["vina", "ad4"]] = {"vina", "ad4"}
    output_dir: Path
    run_id: str = ""
    verbosity: int = 0
    minimize_poses: bool = True
    refine_topk: int | None = None
    mmgbsa_cpu_only: bool = False

    @field_validator("peptide_sequence")
    @classmethod
    def _validate_peptide(cls, v: str) -> str:
        up = v.upper()
        bad = set(up) - _VALID_AA
        if bad:
            raise ValueError(f"Non-standard amino acid characters: {sorted(bad)}")
        if not up:
            raise ValueError("peptide_sequence must not be empty")
        return up

    @field_validator("receptor_path", mode="after")
    @classmethod
    def _receptor_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Receptor path does not exist: {v}")
        return v

    @field_validator("box_size")
    @classmethod
    def _box_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"box_size must be positive, got {v}")
        return v

    @field_validator("n_samples")
    @classmethod
    def _nsamples_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"n_samples must be positive, got {v}")
        return v

    @model_validator(mode="before")
    @classmethod
    def _generate_run_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("run_id"):
            seed_str = str(data.get("seed"))
            seed_part = hashlib.sha1(seed_str.encode()).hexdigest()[:8]
            data["run_id"] = f"{int(time.time())}_{seed_part}"
        return data


@dataclass
class PoseRecord:
    """Parsed peptide pose with pre-extracted C-alpha coordinates.

    Args:
        pose_idx: Zero-based index of this pose within the sampling run.
        pdb_path: Absolute path to the raw PDB file produced by RAPiDock.
        sequence: Single-letter amino acid sequence parsed from the PDB.
        ca_coords: Shape (n_residues, 3) float64 array of C-alpha XYZ
            coordinates. Populated at parse time; never re-read from disk.
    """

    pose_idx: int
    pdb_path: Path
    sequence: str
    ca_coords: np.ndarray


@dataclass
class ScoredPose(PoseRecord):
    """PoseRecord extended with scoring results.

    All score fields default to None; they are filled in sequentially by the
    scoring pipeline. hybrid_score is set last (Vina + AD4 + entropy).

    Args:
        vina_score: Vina --score_only output in kcal/mol.
        ad4_score: AutoDock4 scoring output in kcal/mol.
        entropy_correction: Calibrated backbone entropy term in kcal/mol.
        hybrid_score: Final combined score in kcal/mol.
        cluster_id: Cluster assignment from Phase 6 analysis; None until
            clustering runs.
        pdbqt_path: Path to the prepared PDBQT produced in Phase 2.
        is_ad4_anomaly: True when ad4_score > 0 (flagged per SCORE-02).
        is_clipped: True when any atoms fell outside grid bounds (logged
            per SCORE-01; never silently dropped).
    """

    vina_score: float | None = None
    ad4_score: float | None = None
    entropy_correction: float | None = None
    hybrid_score: float | None = None
    mmgbsa_dg: float | None = None
    cluster_id: int | None = None
    pdbqt_path: Path | None = None
    is_ad4_anomaly: bool = False
    is_clipped: bool = False
    n_contact_residues: int | None = None
    is_clashed: bool = False


@dataclass
class PoseFailure:
    """Record of a pose that failed at some pipeline stage.

    Args:
        pose_idx: Index of the failed pose.
        stage: Pipeline stage where failure occurred.
        error_msg: Human-readable error message. No traceback stored here;
            full tracebacks go to the run log.
    """

    pose_idx: int
    stage: Literal["parsing", "prep", "scoring", "clustering"]
    error_msg: str
