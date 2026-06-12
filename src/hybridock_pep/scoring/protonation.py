"""Charge-aware protonation assignment (the cheap entry to charged-floor accuracy).

The charged-binding floor (r≈0.07–0.16, see docs E47) exists partly because we score at DEFAULT
protonation: His neutral, all Asp/Glu deprotonated, all Lys/Arg protonated, regardless of local
environment. A buried Asp or a His in a salt bridge can flip state, making or breaking the very
interactions that dominate charged binding. The cheapest correction — no MD — is to assign
pH-dependent protonation states with PROPKA before MM-GBSA / electrostatics.

This wraps ``pdb2pqr30 --titration-state-method=propka``. It is OPTIONAL and OFF by default; callers
opt in. If pdb2pqr is not installed the functions degrade gracefully (return the input unchanged and
log a warning) so the pipeline never hard-fails on a missing optional dependency.

PROPKA: Olsson et al., JCTC 2011, 10.1021/ct100578z. pdb2pqr: Dolinsky et al., NAR 2007.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def pdb2pqr_available() -> bool:
    """True if the pdb2pqr30 binary is on PATH."""
    return shutil.which("pdb2pqr30") is not None


def assign_protonation(
    pdb_path: Path,
    out_pqr: Path | None = None,
    ph: float = 7.0,
    forcefield: str = "AMBER",
    timeout_s: int = 300,
) -> Path | None:
    """Assign pH-dependent protonation states with PROPKA and write a PQR.

    Args:
        pdb_path: Input structure (receptor or complex).
        out_pqr: Output PQR path; defaults to ``pdb_path`` with a ``.pqr`` suffix.
        ph: Target pH for titration-state assignment (default physiological 7.0).
        forcefield: pdb2pqr force field for partial charges (default AMBER, matches ff14SB MM-GBSA).
        timeout_s: Subprocess timeout.

    Returns:
        Path to the written PQR, or None if pdb2pqr is unavailable or failed (caller should then
        fall back to default protonation).
    """
    if not pdb2pqr_available():
        logger.warning("pdb2pqr30 not on PATH; skipping PROPKA protonation (using default states). "
                       "Install with: pip install pdb2pqr propka")
        return None
    pdb_path = Path(pdb_path)
    out_pqr = Path(out_pqr) if out_pqr else pdb_path.with_suffix(".pqr")
    cmd = ["pdb2pqr30", "--ff", forcefield, "--titration-state-method", "propka",
           "--with-ph", str(ph), str(pdb_path), str(out_pqr)]
    logger.info("PROPKA protonation @ pH %.1f: %s", ph, " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout_s)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("pdb2pqr failed (%s); falling back to default protonation", type(exc).__name__)
        return None
    if not out_pqr.exists():
        return None
    return out_pqr


def titratable_state_summary(pqr_path: Path) -> dict[str, int]:
    """Count protonation-state-assigned titratable residues in a PQR for logging/metadata.

    Returns a dict of residue-name → count for the titratable set (ASP/GLU/HIS/LYS/ARG/CYS variants),
    including PROPKA's protonated-variant names (ASH/GLH/HID/HIE/HIP/LYN/CYM) when present.
    """
    titratable = {"ASP", "ASH", "GLU", "GLH", "HIS", "HID", "HIE", "HIP",
                  "LYS", "LYN", "ARG", "CYS", "CYM"}
    counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for line in Path(pqr_path).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        resname = line[17:20].strip()
        chain = line[21:22]
        resseq = line[22:26].strip()
        if resname in titratable and (chain, resseq) not in seen:
            seen.add((chain, resseq))
            counts[resname] = counts.get(resname, 0) + 1
    return counts
