#!/usr/bin/env python3
"""Generate 25 MDM2/p53 fixture PDB files for test_e2e.py (TEST-02).

Peptide: ETFSDLWKLLPE (12 residues, PDB 2OY2 p53 transactivation domain fragment)
Geometry: Idealized extended backbone centered on MDM2 binding groove (~26, 3.5, -5.5 Å)
Strategy: 25 copies with ±0.3 Å random xyz perturbation per heavy atom position.
All files are valid Biopython PDBParser input and contain N/CA/C/O backbone atoms.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

_THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}

SEQUENCE = "ETFSDLWKLLPE"

BASE_CA_COORDS = [
    (26.0,  3.0, -5.0),   # E1
    (28.5,  3.5, -5.5),   # T2
    (31.0,  4.0, -6.0),   # F3
    (33.5,  3.5, -5.0),   # S4
    (36.0,  4.0, -4.5),   # D5
    (38.5,  4.5, -5.0),   # L6
    (41.0,  4.0, -5.5),   # W7
    (43.5,  3.5, -5.0),   # K8
    (46.0,  4.0, -4.5),   # L9
    (48.5,  4.5, -5.0),   # L10
    (51.0,  4.0, -5.5),   # P11
    (53.5,  3.5, -5.0),   # E12
]

BACKBONE_OFFSETS = {
    "N":  (-1.2, -0.5, 0.0),
    "CA": (0.0,   0.0, 0.0),
    "C":  (1.2,   0.5, 0.0),
    "O":  (1.5,   1.5, 0.0),
}


def _format_pdb_line(serial: int, name: str, resname: str, chain: str,
                     resseq: int, x: float, y: float, z: float, element: str) -> str:
    if len(name) < 4:
        atom_col = f" {name:<3}" if name[0].isalpha() and len(element) == 1 else f"{name:<4}"
    else:
        atom_col = name[:4]
    return (
        f"ATOM  {serial:5d} {atom_col} {resname:3s} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element:>2s}\n"
    )


def write_pose(path: Path, seed_offset: int) -> None:
    rng = random.Random(seed_offset)
    lines: list[str] = []
    serial = 1

    for res_idx, (one_letter, ca) in enumerate(zip(SEQUENCE, BASE_CA_COORDS)):
        resname = _THREE[one_letter]
        resseq = res_idx + 1

        for atom_name, offset in BACKBONE_OFFSETS.items():
            element = atom_name[0]
            x = ca[0] + offset[0] + rng.uniform(-0.3, 0.3)
            y = ca[1] + offset[1] + rng.uniform(-0.3, 0.3)
            z = ca[2] + offset[2] + rng.uniform(-0.3, 0.3)
            lines.append(_format_pdb_line(serial, atom_name, resname, "A", resseq, x, y, z, element))
            serial += 1

    lines.append("END\n")
    path.write_text("".join(lines))


def main() -> None:
    repo_root = Path(__file__).parent.parent
    out_dir = repo_root / "tests" / "fixtures" / "mdm2_p53"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(25):
        pose_path = out_dir / f"pose_{i:03d}.pdb"
        write_pose(pose_path, seed_offset=i * 100)
        print(f"Written: {pose_path}")

    cal_path = repo_root / "tests" / "fixtures" / "mdm2_calibration.json"
    cal_path.write_text(json.dumps({"alpha": 0.2, "beta": 0.0}, indent=2))
    print(f"Written: {cal_path}")
    print("Done — 25 fixture PDBs + 1 calibration JSON")


if __name__ == "__main__":
    main()
