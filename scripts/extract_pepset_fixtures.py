"""Extract PepSet fixtures (receptor pocket + crystal poses) for e2e tests.

For each target complex this script:
  1. Reads the holo receptor (``_rec_ref.pdb``) and crystal peptide pose
     (``_pep_ref.pdb``) from ``datasets/pepset/{pdb_id}/``.
  2. Identifies receptor residues with any heavy atom within POCKET_CUTOFF Å
     of any peptide heavy atom (pocket extraction).
  3. Strips hydrogen atoms from both receptor pocket and peptide pose.
  4. Writes:
       tests/fixtures/{tag}/receptor_pocket.pdb  — pocket heavy atoms
       tests/fixtures/{tag}/pose_000.pdb … pose_004.pdb  — 5 copies of crystal pose
  5. Prints the Cα centroid and box size for copy-pasting into test_e2e.py.

Usage:
    python scripts/extract_pepset_fixtures.py

Requires only stdlib + numpy (score-env).
"""
from __future__ import annotations

import shutil
from itertools import combinations
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PEPSET_DIR = REPO_ROOT / "datasets" / "pepset"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
POCKET_CUTOFF = 10.0          # Å — residues with any heavy atom within this distance
MIN_BOX = 25.0                # Å — minimum box size
BOX_PADDING = 12.0            # Å — added to max Cα–Cα span
N_POSE_COPIES = 5             # copies of crystal pose (pose_000 … pose_N-1)

# (tag, pdb_id, family_label)
TARGETS: list[tuple[str, str, str]] = [
    ("sh3_1a0n",  "1a0n", "SH3 domain / PPXP proline-rich"),
    ("ww_1ywi",   "1ywi", "WW domain / proline-rich"),
    ("bcl2_2vzg", "2vzg", "BCL-2 family / BH3 helix"),
    ("kin_2khh",  "2khh", "Kinase substrate"),
    ("helix_1yfn","1yfn", "Amphipathic helix binder"),
    ("arm_2cny",  "2cny", "ARM / HEAT repeat"),
    ("mdm2_1pmx", "1pmx", "MDM2 / MDMX"),
]


# ---------------------------------------------------------------------------
# PDB helpers
# ---------------------------------------------------------------------------

def _is_heavy(atom_name: str) -> bool:
    name = atom_name.strip()
    return bool(name) and name[0] not in ("H", "D")


def _parse_atoms(pdb_path: Path) -> list[dict]:
    """Parse ATOM/HETATM records, returning list of field dicts (no H)."""
    atoms: list[dict] = []
    with pdb_path.open() as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16]
            if not _is_heavy(atom_name):
                continue
            try:
                atoms.append({
                    "record": line[:6],
                    "serial": int(line[6:11]),
                    "name": atom_name,
                    "alt": line[16],
                    "resname": line[17:20],
                    "chain": line[21],
                    "resseq": int(line[22:26]),
                    "icode": line[26],
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                    "rest": line[54:],
                    "raw": line,
                })
            except ValueError:
                continue
    return atoms


def _coords(atoms: list[dict]) -> np.ndarray:
    return np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=np.float64)


def _write_atoms(atoms: list[dict], path: Path) -> None:
    """Write atoms to PDB, renumbering serials from 1, with TER/END."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for i, a in enumerate(atoms, start=1):
            fh.write(
                f"{a['record']}{i:5d} {a['name']}{a['alt']}{a['resname']} "
                f"{a['chain']}{a['resseq']:4d}{a['icode']}   "
                f"{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}{a['rest']}"
            )
        fh.write("TER\nEND\n")


# ---------------------------------------------------------------------------
# Pocket extraction
# ---------------------------------------------------------------------------

def extract_pocket(
    rec_atoms: list[dict],
    pep_coords: np.ndarray,
    cutoff: float,
) -> list[dict]:
    """Return receptor atoms belonging to residues within cutoff of peptide."""
    # Group receptor atoms by (chain, resseq, icode)
    res_map: dict[tuple, list[dict]] = {}
    for a in rec_atoms:
        key = (a["chain"], a["resseq"], a["icode"])
        res_map.setdefault(key, []).append(a)

    selected: list[dict] = []
    cutoff2 = cutoff ** 2
    for key, res_atoms in res_map.items():
        rec_xyz = np.array([[a["x"], a["y"], a["z"]] for a in res_atoms])
        # (n_rec, 1, 3) - (1, n_pep, 3) → (n_rec, n_pep, 3)
        diffs = rec_xyz[:, np.newaxis, :] - pep_coords[np.newaxis, :, :]
        sq_dists = (diffs ** 2).sum(axis=-1)
        if sq_dists.min() <= cutoff2:
            selected.extend(res_atoms)
    return selected


def ca_centroid_and_box(pep_atoms: list[dict]) -> tuple[tuple[float, float, float], float]:
    """Return (cx, cy, cz) Cα centroid and box edge for the peptide."""
    cas = [a for a in pep_atoms if a["name"].strip() == "CA"]
    if not cas:
        cas = pep_atoms  # fallback: all heavy atoms
    xyz = np.array([[a["x"], a["y"], a["z"]] for a in cas])
    cx, cy, cz = xyz.mean(axis=0)
    if len(xyz) > 1:
        dists = [
            float(np.linalg.norm(xyz[i] - xyz[j]))
            for i, j in combinations(range(len(xyz)), 2)
        ]
        box = max(max(dists) + BOX_PADDING, MIN_BOX)
    else:
        box = MIN_BOX
    return (float(cx), float(cy), float(cz)), float(box)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_target(tag: str, pdb_id: str, family: str) -> None:
    pepset_dir = PEPSET_DIR / pdb_id
    rec_ref = pepset_dir / f"{pdb_id}_rec_ref.pdb"
    pep_ref = pepset_dir / f"{pdb_id}_pep_ref.pdb"

    if not rec_ref.exists() or not pep_ref.exists():
        print(f"  [SKIP] {tag}: missing {rec_ref} or {pep_ref}")
        return

    rec_atoms = _parse_atoms(rec_ref)
    pep_atoms = _parse_atoms(pep_ref)
    if not rec_atoms or not pep_atoms:
        print(f"  [SKIP] {tag}: no atoms parsed")
        return

    pep_coords = _coords(pep_atoms)
    pocket_atoms = extract_pocket(rec_atoms, pep_coords, POCKET_CUTOFF)
    if not pocket_atoms:
        print(f"  [WARN] {tag}: no pocket residues found within {POCKET_CUTOFF} Å")
        return

    out_dir = FIXTURES_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write receptor pocket
    _write_atoms(pocket_atoms, out_dir / "receptor_pocket.pdb")

    # Write N copies of crystal pose
    for i in range(N_POSE_COPIES):
        _write_atoms(pep_atoms, out_dir / f"pose_{i:03d}.pdb")

    # Compute binding site parameters
    (cx, cy, cz), box = ca_centroid_and_box(pep_atoms)

    seq_file = pepset_dir / f"{pdb_id}_peptide_sequence"
    seq = seq_file.read_text().strip() if seq_file.exists() else "?"

    print(
        f"  [OK] {tag} ({family})\n"
        f"       seq={seq}  n_pep={len(pep_atoms)} atoms  pocket_res≈{len(set((a['resseq'],a['chain']) for a in pocket_atoms))}\n"
        f"       site=({cx:.2f}, {cy:.2f}, {cz:.2f})  box={box:.1f}"
    )


def main() -> None:
    print(f"PepSet dir: {PEPSET_DIR}")
    print(f"Fixtures dir: {FIXTURES_DIR}\n")
    for tag, pdb_id, family in TARGETS:
        process_target(tag, pdb_id, family)
    print("\nDone. Copy site/box values into test_e2e.py _PEPSET_CASES.")


if __name__ == "__main__":
    main()
