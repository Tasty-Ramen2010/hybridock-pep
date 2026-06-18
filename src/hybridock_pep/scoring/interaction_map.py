"""Typed protein–peptide interaction fingerprint (IFP) — crystal-pose scoring enhancement.

Represents a complex by *where and how the peptide touches the receptor* (typed, distance-weighted
per-contact bonds) rather than by residue identity alone. On crystal-quality poses this orthogonal physics
adds **+0.10 Pearson r** to absolute ΔG scoring (PDBbind, leave-receptor-out: 0.383→0.485, charged
0.346→0.448 — the first charged improvement of the campaign) and **7× within-receptor ranking** (E295–E297).

IMPORTANT — crystal-pose only. The IFP is sensitive to exact contact geometry; a RAPiDock *docked* rank-1
pose reproduces only ~70% of the crystal map, so the gain largely reverts on docked poses (E299). This
module is therefore wired for the case where the user supplies (or has) a crystal-quality complex. For
docked-pose deployment use the standard pose-robust affinity model; do not feed docked poses here expecting
the crystal gain.

Public API:
  * ``compute_ifp(receptor_pdb, peptide_structure)`` -> dict of the 19 IFP features (auto-detects PDB/mol2).
  * ``IFP_FEATURE_ORDER`` -> the canonical feature order (matches the trained crystal-IFP artifact).
  * ``score_crystal_complex(...)`` -> calibrated ΔG using geometry + IFP and the crystal-IFP model.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# --- atom classification ----------------------------------------------------------------------------
_POS_RES = {"LYS", "ARG"}
_NEG_RES = {"ASP", "GLU"}
_AROM_RES = {"PHE", "TYR", "TRP", "HIS"}
_HYD_RES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
_POL_RES = {"SER", "THR", "ASN", "GLN", "TYR", "HIS", "CYS"}
_POS_ATOMS = {"NZ", "NH1", "NH2", "NE"}      # Lys/Arg cationic tips
_NEG_ATOMS = {"OD1", "OD2", "OE1", "OE2"}    # Asp/Glu carboxylate O

#: Canonical IFP feature order — MUST match the order the crystal-IFP artifact was trained on.
IFP_FEATURE_ORDER: tuple[str, ...] = (
    "sb_fav", "sb_fav_str", "sb_unfav", "sb_d2", "sb_d3", "sb_d4", "hbond", "hbond_str",
    "hb_to_chg", "hb_to_pol", "hb_to_hyd", "hb_to_aro", "hydrophobic", "hyd_str", "aromatic",
    "contact_chg", "contact_pol", "contact_hyd", "contact_aro",
)


def _restype(res: str) -> str:
    if res in _POS_RES | _NEG_RES:
        return "chg"
    if res in _POL_RES:
        return "pol"
    if res in _AROM_RES:
        return "aro"
    return "hyd"


def _atom_class(res: str, atom: str) -> str | None:
    """Map a (residue, atom) to an interaction class, or None if not interaction-relevant."""
    el = atom[0] if atom else ""
    if res in _POS_RES and atom in _POS_ATOMS:
        return "pos"
    if res in _NEG_RES and atom in _NEG_ATOMS:
        return "neg"
    if el == "N":
        return "don"
    if el == "O":
        return "acc"
    if el == "C" and res in _AROM_RES:
        return "aro"
    if el == "C" and res in _HYD_RES:
        return "hyd"
    return None


def receptor_atoms(pdb_path: str | Path) -> list[tuple[str, str, np.ndarray]]:
    """Parse typed receptor atoms ``(interaction_class, residue_type, xyz)`` from a PDB.

    Args:
        pdb_path: Path to the receptor PDB (or full complex; peptide atoms are harmless extra contacts).

    Returns:
        List of ``(cls, restype, xyz)`` for interaction-relevant heavy atoms; empty if none parse.
    """
    out: list[tuple[str, str, np.ndarray]] = []
    for ln in Path(pdb_path).read_text().splitlines():
        if not ln.startswith("ATOM"):
            continue
        res, atom = ln[17:20].strip(), ln[12:16].strip()
        cls = _atom_class(res, atom)
        if cls is None:
            continue
        try:
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
        out.append((cls, _restype(res), xyz))
    return out


def _peptide_atoms_pdb(pdb_path: str | Path) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    for ln in Path(pdb_path).read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        res, atom = ln[17:20].strip(), ln[12:16].strip()
        el = atom[0] if atom else ""
        if res in _POS_RES and atom in _POS_ATOMS:
            cls = "pos"
        elif res in _NEG_RES and atom in _NEG_ATOMS:
            cls = "neg"
        elif el == "N":
            cls = "don"
        elif el == "O":
            cls = "acc"
        elif el == "C" and res in _AROM_RES:
            cls = "aro"
        elif el == "C":
            cls = "hyd"
        else:
            continue
        try:
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
        out.append((cls, xyz))
    return out


def _peptide_atoms_mol2(mol2_path: str | Path) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    in_atom = False
    for ln in Path(mol2_path).read_text().splitlines():
        if ln.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            continue
        if ln.startswith("@<TRIPOS>") and in_atom:
            break
        if not in_atom:
            continue
        p = ln.split()
        if len(p) < 6:
            continue
        try:
            xyz = np.array([float(p[2]), float(p[3]), float(p[4])])
        except ValueError:
            continue
        t = p[5]
        cls = ("pos" if t == "N.4" else "neg" if t == "O.co2" else "don" if t.startswith("N")
               else "acc" if t.startswith("O") else "aro" if t == "C.ar"
               else "hyd" if t.startswith("C") else None)
        if cls:
            out.append((cls, xyz))
    return out


def interaction_fingerprint(
    rec: list[tuple[str, str, np.ndarray]],
    pep: list[tuple[str, np.ndarray]],
) -> dict[str, float]:
    """Compute the typed, distance-weighted interaction fingerprint between receptor and peptide atoms.

    Args:
        rec: receptor atoms from :func:`receptor_atoms`.
        pep: peptide atoms ``(cls, xyz)``.

    Returns:
        Dict keyed by :data:`IFP_FEATURE_ORDER`: favorable/unfavorable salt-bridge counts + distance bins,
        H-bonds (typed by receptor residue class) + strength, hydrophobic, aromatic, and per-residue-type
        total contact strength. Strengths are ``1/d`` summed over contacts ≤ 6 Å.
    """
    f: dict[str, float] = {k: 0.0 for k in IFP_FEATURE_ORDER}
    for kp, xp in pep:
        for kr, rt, xr in rec:
            d = float(np.linalg.norm(xp - xr))
            if d < 1.5 or d > 6.0:
                continue
            w = 1.0 / d
            if {kp, kr} <= {"pos", "neg"} and kp != kr and d < 4.5:
                f["sb_fav"] += 1.0
                f["sb_fav_str"] += w
                f[f"sb_d{min(int(d), 4)}"] = f.get(f"sb_d{min(int(d), 4)}", 0.0) + 1.0
            elif kp == kr and kp in ("pos", "neg") and d < 4.5:
                f["sb_unfav"] += 1.0
            elif kp in ("don", "acc", "pos", "neg") and kr in ("don", "acc", "pos", "neg") and d < 3.6:
                f["hbond"] += 1.0
                f["hbond_str"] += w
                f[f"hb_to_{rt}"] = f.get(f"hb_to_{rt}", 0.0) + 1.0
            elif kp == "hyd" and kr == "hyd" and d < 4.8:
                f["hydrophobic"] += 1.0
                f["hyd_str"] += w
            elif kp == "aro" and kr == "aro" and d < 5.5:
                f["aromatic"] += 1.0
            f[f"contact_{rt}"] = f.get(f"contact_{rt}", 0.0) + w
    return {k: float(f.get(k, 0.0)) for k in IFP_FEATURE_ORDER}


def compute_ifp(receptor_pdb: str | Path, peptide_structure: str | Path) -> dict[str, float]:
    """Compute the interaction fingerprint for a crystal complex.

    Args:
        receptor_pdb: receptor (or full complex) PDB.
        peptide_structure: peptide structure — ``.pdb`` or ``.mol2`` (format auto-detected by extension).

    Returns:
        Dict of the 19 IFP features in :data:`IFP_FEATURE_ORDER`.

    Raises:
        ValueError: if the peptide structure extension is neither ``.pdb`` nor ``.mol2``.
    """
    rec = receptor_atoms(receptor_pdb)
    ext = Path(peptide_structure).suffix.lower()
    if ext == ".mol2":
        pep = _peptide_atoms_mol2(peptide_structure)
    elif ext in (".pdb", ".ent"):
        pep = _peptide_atoms_pdb(peptide_structure)
    else:
        raise ValueError(f"peptide structure must be .pdb or .mol2, got {ext!r}")
    if not rec or not pep:
        logger.warning("IFP: empty receptor (%d) or peptide (%d) atom set", len(rec), len(pep))
    return interaction_fingerprint(rec, pep)


def ifp_vector(ifp: dict[str, float]) -> np.ndarray:
    """Return the IFP dict as a fixed-order float vector matching the trained artifact."""
    return np.array([ifp[k] for k in IFP_FEATURE_ORDER], dtype=float)


#: Geometry feature order the crystal-IFP artifact was trained on (16 GEOMETRY_KEYS + length).
_CRYSTAL_GEOM_ORDER: tuple[str, ...] = (
    "arom_cc", "bsa_hyd", "cys_frac", "hb_count", "length", "mean_burial", "mj_contact",
    "org_density", "poc_eis", "poc_f_arom", "poc_f_hyd", "poc_n", "poc_net", "rg_per_L",
    "sasa_hb", "sasa_sb", "strength_bur",
)
_DEFAULT_ARTIFACT = "data/affinity_crystal_ifp.joblib"
_CRYSTAL_MODEL_CACHE: dict[str, object] = {}


def _geom17(geometry: dict[str, float], seq: str) -> list[float]:
    return [float(len(seq)) if k == "length" else float(geometry.get(k, 0.0))
            for k in _CRYSTAL_GEOM_ORDER]


def score_crystal_complex(
    receptor_pdb: str | Path,
    peptide_pdb: str | Path,
    seq: str,
    *,
    artifact: str | Path = _DEFAULT_ARTIFACT,
) -> float | None:
    """Predict ΔG (kcal/mol) for a CRYSTAL complex using geometry + interaction map.

    Combines the 16 pose geometry descriptors (+ length) with the 19 typed interaction-fingerprint
    features. On crystal-quality poses this is +0.10 Pearson r over the geometry-only model (E296). Do NOT
    use on docked poses (the IFP gain reverts ~70%, E299) — use the standard ``predict_affinity`` there.

    Args:
        receptor_pdb: receptor PDB (or full complex).
        peptide_pdb: crystal peptide structure (``.pdb`` or ``.mol2``).
        seq: peptide one-letter sequence.
        artifact: trained crystal-IFP joblib bundle. Defaults to the shipped artifact.

    Returns:
        Predicted ΔG in kcal/mol, or ``None`` if the artifact or geometry features are unavailable.
    """
    import joblib  # local import: keep module import light

    from hybridock_pep.scoring.geometry_features import compute_geometry_features

    path = Path(artifact)
    if not path.exists():
        logger.warning("crystal-IFP artifact not found: %s", path)
        return None
    bundle = _CRYSTAL_MODEL_CACHE.get(str(path))
    if bundle is None:
        bundle = joblib.load(path)
        _CRYSTAL_MODEL_CACHE[str(path)] = bundle

    geometry = compute_geometry_features(Path(peptide_pdb), Path(receptor_pdb))
    if geometry is None:
        logger.warning("crystal-IFP: geometry features unavailable for %s", peptide_pdb)
        return None
    ifp = compute_ifp(receptor_pdb, peptide_pdb)
    vec = np.array(_geom17(geometry, seq) + list(ifp_vector(ifp)), dtype=float).reshape(1, -1)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return float(bundle["model"].predict(vec)[0])
