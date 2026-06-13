"""Anchor / specific-interaction features — rescue short peptides (and help med/vlong).

Short peptides (≤8) bind via a dominant ANCHOR residue plugged into a specificity pocket, not via a large
distributed interface. Our 16 structural features are interface-SIZE sums whose dynamic range collapses on
small interfaces, so they can't tell a deep-anchored strong binder from a shallow weak one. These features
capture the anchor / specific-interaction physics directly. Validated (E128) on 925 PDBbind crystal poses:
+anchor lifts short 0.43→0.51, med 0.27→0.38, vlong 0.28→0.35 (long already saturated, +0.03).

Computed from the bound complex (peptide pose PDB + receptor PDB) — pipeline-compatible with
``geometry_features.compute_geometry_features``. All geometry-only; no OpenMM.

Keys (ANCHOR_FEATURE_KEYS):
    max_burial            deepest single peptide residue (# receptor heavy atoms within 8 Å of its centroid)
    burial_concentration  max_burial / Σ burial — is binding concentrated in one anchor?
    best_salt_bridge      # peptide charged residues forming a salt bridge (<4.5 Å opposite charge)
    charged_anchor        burial of the most-buried salt-bridged charge (buried ion pair = strong)
    buried_inert          max_burial · 1/(1+hbond_per_contact) — buried-but-inert (polyproline over-pred signature)
    pro_run               longest proline run / length — polyproline rigidity/low-affinity penalty
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ANCHOR_FEATURE_KEYS = [
    "max_burial", "burial_concentration", "best_salt_bridge", "charged_anchor", "buried_inert", "pro_run",
]
# Cross-dataset sign-stability gate (E129, PDBbind→the98): only these 3 hold their sign on a second
# dataset and are safe to use in calibration. The salt-bridge anchors (best_salt_bridge, charged_anchor)
# FLIP sign across datasets — the charged floor again (single-pose electrostatics don't transfer); and
# burial_concentration goes weak. Validated lift with the STABLE set: short +0.043, med +0.063, vlong
# +0.054, long ~0 (already saturated). USE ANCHOR_STABLE_KEYS for production calibration.
ANCHOR_STABLE_KEYS = ["max_burial", "buried_inert", "pro_run"]
_BURIAL_R = 8.0      # Å shell around a residue centroid for burial count
_SB_R = 4.5          # Å salt-bridge cutoff between charge centers
_THREE1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
           "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
           "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def _parse_residues(pdb: Path) -> list[dict]:
    """Group heavy ATOM records into residues (ordered), keeping atom-name → xyz and the residue centroid."""
    res: list[dict] = []
    cur_key = None
    cur: dict | None = None
    for ln in Path(pdb).read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        name = ln[12:16].strip()
        if not name or name[0] in ("H", "D"):
            continue
        try:
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
        key = (ln[21], ln[22:27])
        if key != cur_key:
            cur = {"rn": ln[17:20].strip(), "xyz": [], "at": {}}
            res.append(cur)
            cur_key = key
        cur["xyz"].append(xyz)
        cur["at"][name] = xyz
    return res


def _charge_center(rn: str, at: dict) -> tuple[int, np.ndarray] | None:
    if rn in ("LYS",) and "NZ" in at:
        return (+1, at["NZ"])
    if rn == "ARG" and "CZ" in at:
        return (+1, at["CZ"])
    if rn == "ASP":
        o = [at[k] for k in ("OD1", "OD2", "CG") if k in at]
        return (-1, np.mean(o, 0)) if o else None
    if rn == "GLU":
        o = [at[k] for k in ("OE1", "OE2", "CD") if k in at]
        return (-1, np.mean(o, 0)) if o else None
    return None


def _receptor(pdb: Path) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    res = _parse_residues(pdb)
    heavy = np.array([xyz for r in res for xyz in r["xyz"]]) if res else np.zeros((1, 3))
    charged = [c for r in res if (c := _charge_center(r["rn"], r["at"])) is not None]
    return heavy, charged


def compute_anchor_features(peptide_pose_pdb: Path, receptor_pdb: Path, hb_count: float = 0.0) -> dict | None:
    """Anchor / specific-interaction features for one bound complex.

    Args:
        peptide_pose_pdb: peptide pose PDB (one chain, standard residues).
        receptor_pdb: receptor PDB.
        hb_count: interface H-bond count (from geometry_features) for the buried-inert term; 0 if unknown.

    Returns:
        Dict with ANCHOR_FEATURE_KEYS, or None if the peptide has no parseable residues.
    """
    pep = _parse_residues(peptide_pose_pdb)
    if not pep:
        return None
    rec_xyz, rec_charged = _receptor(receptor_pdb)
    burial: list[float] = []
    n_contact = 0
    salt = 0
    charged_anchor = 0.0
    pro_run = cur_run = 0
    for r in pep:
        aa = _THREE1.get(r["rn"], "X")
        rx = np.asarray(r["xyz"])
        nb = int((np.linalg.norm(rec_xyz - rx.mean(0), axis=1) < _BURIAL_R).sum()) if rx.size else 0
        burial.append(nb)
        if rx.size and np.linalg.norm(rec_xyz[:, None, :] - rx[None, :, :], axis=2).min() < _SB_R:
            n_contact += 1
        cc = _charge_center(r["rn"], r["at"])
        if cc is not None and rec_charged and any(
            cc[0] * sr < 0 and np.linalg.norm(cc[1] - xr) < _SB_R for sr, xr in rec_charged
        ):
            salt += 1
            charged_anchor = max(charged_anchor, float(nb))
        cur_run = cur_run + 1 if aa == "P" else 0
        pro_run = max(pro_run, cur_run)
    burial_a = np.asarray(burial, float)
    hpc = hb_count / max(1, n_contact)
    return {
        "max_burial": float(burial_a.max()),
        "burial_concentration": float(burial_a.max() / (burial_a.sum() + 1e-9)),
        "best_salt_bridge": float(salt),
        "charged_anchor": charged_anchor,
        "buried_inert": float(burial_a.max()) * (1.0 / (1.0 + hpc)),
        "pro_run": pro_run / max(1, len(pep)),
    }
