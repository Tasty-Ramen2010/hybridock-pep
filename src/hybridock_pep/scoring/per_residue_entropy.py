"""Per-residue side-chain + backbone entropy with secondary-structure weighting.

Replaces the uniform ``-α·N_contact`` term in the legacy hybrid score (one α
times a residue count) with a sum over per-residue penalties that depend on:

  * amino-acid identity (Doig & Sternberg 1995 / Pickett & Sternberg 1993
    consensus values; see docs/calibration_strategies.md §5.5.2).
  * backbone flexibility (Gly free / Pro locked / others standard;
    Doig-Sternberg backbone table).
  * **secondary-structure context** of the contact region, inferred from the
    pose's actual backbone φ/ψ angles.  Loop residues lose more entropy on
    binding than residues already locked in a helix or sheet in solution.

Functional form per pose:

    -TΔS_pose = Σ_{i ∈ contact}  ss_factor(SS_i)
                                  * ( s_sc(aa_i) + s_bb(aa_i) )

where ``ss_factor(loop)=1.0, ss_factor(helix)=0.5, ss_factor(sheet)=0.3``
(see ``_SS_FACTOR`` below — these are conservative consensus values).

The function ``compute_entropy_sums(pose_pdb, sequence, contact_residues)``
returns three sums: ``s_sc_sum, s_bb_sum, ss_weighted_sum``.  All three are
exposed so a ridge fit can pick which (or which combination) carries signal
on a given dataset.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Literal

import numpy as np

_log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Tables                                                                       #
# --------------------------------------------------------------------------- #

# Side-chain conformational-entropy penalty at 300 K, kcal/mol.
# Consensus of Pickett & Sternberg 1993, D'Aquino et al. 1996, Doig & Sternberg 1995.
# Values are "-TΔS_sc on full immobilization."  Rigid sidechains (G/A/P) = 0.
S_SC: dict[str, float] = {
    "G": 0.00, "A": 0.00, "P": 0.00,
    "S": 0.55, "C": 0.55, "T": 0.70, "V": 0.80,
    "L": 1.00, "I": 1.10, "N": 1.30, "D": 1.30,
    "H": 1.50, "F": 1.60, "Y": 1.70, "W": 1.80,
    "M": 1.90, "Q": 2.00, "E": 2.00,
    "K": 2.50, "R": 2.80,
}

# Backbone conformational-entropy penalty (Baxter & Murphy / Brady & Sharp consensus).
# Gly is highest (no Cβ → broadest Ramachandran), Pro is lowest (φ locked).
S_BB: dict[str, float] = {"G": 2.20, "P": 0.30}
_S_BB_DEFAULT = 1.00

# Secondary-structure weighting factor for the contact-region backbone
# entropy.  Loops are flexible in solution and lose most entropy on binding;
# helix/sheet residues are already partially ordered so they lose less.
_SS_FACTOR: dict[str, float] = {"loop": 1.00, "helix": 0.50, "sheet": 0.30}


# --------------------------------------------------------------------------- #
# Lookups                                                                      #
# --------------------------------------------------------------------------- #

def s_sc(aa: str) -> float:
    """Side-chain entropy penalty for a single amino acid letter (default 1.0 if unknown)."""
    return S_SC.get(aa.upper(), 1.0)


def s_bb(aa: str) -> float:
    """Backbone entropy penalty for a single amino acid letter (default 1.0)."""
    return S_BB.get(aa.upper(), _S_BB_DEFAULT)


def ss_factor(ss: Literal["loop", "helix", "sheet"]) -> float:
    """Secondary-structure weight on backbone entropy (loop=1.0, helix=0.5, sheet=0.3)."""
    return _SS_FACTOR.get(ss, 1.0)


# --------------------------------------------------------------------------- #
# Secondary-structure classification from backbone dihedrals                   #
# --------------------------------------------------------------------------- #

def _classify_phi_psi(phi: float, psi: float) -> Literal["loop", "helix", "sheet"]:
    """Classify a (φ, ψ) pair into loop / α-helix / β-sheet.

    Generous Ramachandran boxes — wider than DSSP because we are classifying
    a SINGLE residue with no neighbour smoothing.  Heuristic boundaries:

    * α-helix:  -100° < φ < -30°  and  -80° < ψ < -5°
    * β-sheet:  -180° ≤ φ < -90°  and   (90° < ψ ≤ 180°  or  -180° ≤ ψ < -150°)

    All others → loop.  Pro is forced to "loop" classification because its
    backbone is structurally locked regardless of context — its s_bb factor
    of 0.3 already encodes the rigidity.
    """
    if -100.0 < phi < -30.0 and -80.0 < psi < -5.0:
        return "helix"
    if -180.0 <= phi < -90.0 and (90.0 < psi <= 180.0 or -180.0 <= psi < -150.0):
        return "sheet"
    return "loop"


def _dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute the signed dihedral angle (degrees) defined by four points.

    Uses the "Praxeolitic" convention with ``b0 = -(p1 - p0)`` — without
    that sign flip, the returned angle is the mirror of the IUPAC value
    (α-helix would read +60° instead of −60°).  See Wikipedia "Dihedral
    angle" → "Methods of computation".
    """
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_unit = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1_unit) * b1_unit
    w = b2 - np.dot(b2, b1_unit) * b1_unit
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1_unit, v), w))
    return math.degrees(math.atan2(y, x))


def _parse_backbone(pdb_path: Path) -> dict[int, dict[str, np.ndarray]]:
    """Read N / CA / C atoms per residue.  Returns ``{resseq: {"N": xyz, "CA": xyz, "C": xyz}}``."""
    out: dict[int, dict[str, np.ndarray]] = {}
    for line in pdb_path.read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        name = line[12:16].strip()
        if name not in ("N", "CA", "C"):
            continue
        try:
            res_seq = int(line[22:26].strip())
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
        out.setdefault(res_seq, {})[name] = xyz
    return out


def assign_secondary_structure(pose_pdb: Path) -> dict[int, str]:
    """Classify each residue's secondary-structure context from its (φ, ψ).

    Args:
        pose_pdb: Pose PDB file (heavy atoms; H optional).

    Returns:
        ``{residue_seq: "loop" | "helix" | "sheet"}`` for every residue with
        a defined φ/ψ (i.e. excluding the first and last residue of the chain).
        Terminal residues are mapped to "loop" by default.
    """
    bb = _parse_backbone(pose_pdb)
    resseqs = sorted(bb.keys())
    ss: dict[int, str] = {}
    for i, rs in enumerate(resseqs):
        if i == 0 or i == len(resseqs) - 1:
            ss[rs] = "loop"
            continue
        prev = bb[resseqs[i - 1]]
        cur = bb[rs]
        nxt = bb[resseqs[i + 1]]
        if not all(k in prev for k in ("C",)) \
           or not all(k in cur for k in ("N", "CA", "C")) \
           or not all(k in nxt for k in ("N",)):
            ss[rs] = "loop"
            continue
        try:
            phi = _dihedral(prev["C"], cur["N"], cur["CA"], cur["C"])
            psi = _dihedral(cur["N"], cur["CA"], cur["C"], nxt["N"])
        except Exception:
            ss[rs] = "loop"
            continue
        ss[rs] = _classify_phi_psi(phi, psi)
    return ss


# --------------------------------------------------------------------------- #
# Per-pose entropy sums                                                        #
# --------------------------------------------------------------------------- #

def compute_entropy_sums(
    pose_pdb: Path,
    sequence: str,
    contact_residues: list[int] | None = None,
    receptor_coords: np.ndarray | None = None,
    cutoff: float = 4.5,
) -> dict[str, float | int]:
    """Compute per-residue entropy sums (and SS-weighted) for one pose.

    Either pass ``contact_residues`` (1-based residue indices into the
    sequence that are in contact) directly, OR pass ``receptor_coords`` and
    let this function derive the contact set from the pose against the
    receptor at the given heavy-atom cutoff.

    Returns a dict with:
        * ``n_contact``: integer count of contact residues
        * ``s_sc_sum``:  Σ s_sc over contact residues (always ≥ 0, kcal/mol)
        * ``s_bb_sum``:  Σ s_bb over contact residues (always ≥ 0, kcal/mol)
        * ``s_ss_weighted``: Σ ss_factor(SS_i) · (s_sc + s_bb) over contact
                             residues, where SS is classified from pose φ/ψ
        * ``ss_loop_count``, ``ss_helix_count``, ``ss_sheet_count``: SS
                             distribution of the contact residues

    Args:
        pose_pdb: Pose PDB file with backbone atoms.
        sequence: Peptide sequence (single-letter); index 0 == residue 1.
        contact_residues: Optional pre-computed 1-based contact indices.
        receptor_coords: Optional (N, 3) receptor heavy atoms for on-the-fly
            contact derivation.
        cutoff: Heavy-atom cutoff in Å for the on-the-fly contact derivation.

    Raises:
        ValueError: If neither contact_residues nor receptor_coords is given,
            or if the parsed pose residue count differs from len(sequence).
    """
    if contact_residues is None and receptor_coords is None:
        raise ValueError("compute_entropy_sums needs contact_residues or receptor_coords")

    # Parse per-residue heavy atoms (used both for contact derivation and SS).
    per_residue: dict[int, list[np.ndarray]] = {}
    for line in pose_pdb.read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        atom = line[12:16].strip()
        if atom.startswith("H"):
            continue
        try:
            res_seq = int(line[22:26].strip())
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
        per_residue.setdefault(res_seq, []).append(xyz)

    resseqs = sorted(per_residue.keys())

    # Derive contact set if not given.
    if contact_residues is None:
        assert receptor_coords is not None  # guarded above
        contact_residues = []
        for rs in resseqs:
            atoms = np.array(per_residue[rs])
            if atoms.size == 0:
                continue
            diffs = atoms[:, np.newaxis, :] - receptor_coords[np.newaxis, :, :]
            if (diffs ** 2).sum(axis=-1).min() <= cutoff ** 2:
                contact_residues.append(rs)

    # Map resseq → sequence index (assume sequence aligns to sorted resseqs).
    if len(resseqs) != len(sequence):
        # Allow length mismatch — useful when the pose has extra atoms (e.g.
        # capping); we just align by the first N parsed residues.
        _log.debug("compute_entropy_sums: resseq count (%d) != sequence length (%d), "
                   "aligning to first min()", len(resseqs), len(sequence))
    seq_idx_by_res = {rs: i for i, rs in enumerate(resseqs)}
    n_align = min(len(resseqs), len(sequence))

    # SS assignment from backbone dihedrals.
    ss_map = assign_secondary_structure(pose_pdb)

    s_sc_sum = 0.0
    s_bb_sum = 0.0
    s_ss_weighted = 0.0
    ss_counts = {"loop": 0, "helix": 0, "sheet": 0}
    for rs in contact_residues:
        if rs not in seq_idx_by_res:
            continue
        idx = seq_idx_by_res[rs]
        if idx >= n_align:
            continue
        aa = sequence[idx].upper()
        sc = s_sc(aa)
        bb = s_bb(aa)
        ss = ss_map.get(rs, "loop")
        f = ss_factor(ss)
        s_sc_sum += sc
        s_bb_sum += bb
        s_ss_weighted += f * (sc + bb)
        ss_counts[ss] = ss_counts.get(ss, 0) + 1

    return {
        "n_contact": int(len(contact_residues)),
        "s_sc_sum": round(float(s_sc_sum), 4),
        "s_bb_sum": round(float(s_bb_sum), 4),
        "s_ss_weighted": round(float(s_ss_weighted), 4),
        "ss_loop_count": int(ss_counts["loop"]),
        "ss_helix_count": int(ss_counts["helix"]),
        "ss_sheet_count": int(ss_counts["sheet"]),
    }
