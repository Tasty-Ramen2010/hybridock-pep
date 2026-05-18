"""Combine pfldh_tetramer.pdb + best ADCP pose into a GROMACS-ready complex PDB.

Peptide is assigned chain E, residues renumbered 1-15.
Writes data/pdbs/pfldh_lisdaeleaifeadc_complex.pdb

Usage: python scripts/make_gromacs_complex.py [--pose PATH]
POSE defaults to adcp_32/poses_extracted/pose_0.pdb relative to the repo root.
That directory is local-only (gitignored); supply --pose to override.
"""
from __future__ import annotations

import argparse
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pose", default=None, help="Path to peptide pose PDB (overrides default)")
_args = parser.parse_args()

RECEPTOR = _REPO_ROOT / "data" / "pdbs" / "pfldh_tetramer.pdb"
POSE = Path(_args.pose) if _args.pose else _REPO_ROOT / "adcp_32" / "poses_extracted" / "pose_0.pdb"
OUT = _REPO_ROOT / "data" / "pdbs" / "pfldh_lisdaeleaifeadc_complex.pdb"

# --- receptor ---
rec_lines = []
current_chain = None
for line in RECEPTOR.read_text().splitlines():
    if line.startswith("ATOM") or line.startswith("HETATM"):
        chain = line[21]
        if current_chain is not None and chain != current_chain:
            rec_lines.append(f"TER")
        current_chain = chain
        rec_lines.append(line)
    elif line.startswith("TER"):
        rec_lines.append(line)
        current_chain = None
if current_chain is not None:
    rec_lines.append("TER")

# --- peptide (chain E, renumbered 1-15) ---
pose_atoms = [l for l in POSE.read_text().splitlines()
              if l.startswith("ATOM") or l.startswith("HETATM")]

# Map old resseq → new (sequential 1-N)
resmap: dict[str, int] = {}
counter = 0
for line in pose_atoms:
    old = line[22:26].strip()
    if old not in resmap:
        counter += 1
        resmap[old] = counter

pep_lines = []
atom_serial = 99001
for line in pose_atoms:
    old_res = line[22:26].strip()
    new_res = resmap[old_res]
    # Rebuild line with chain=E, new resseq, new serial
    new_line = (
        line[:6]
        + f"{atom_serial:5d}"
        + line[11:21]
        + "E"
        + f"{new_res:4d}"
        + line[26:]
    )
    pep_lines.append(new_line)
    atom_serial += 1
pep_lines.append("TER")

all_lines = rec_lines + pep_lines + ["END"]
OUT.write_text("\n".join(all_lines) + "\n")

# Summary
n_rec = sum(1 for l in rec_lines if l.startswith("ATOM") or l.startswith("HETATM"))
n_pep = len(pose_atoms)
print(f"Receptor atoms : {n_rec}")
print(f"Peptide atoms  : {n_pep}  (chain E, res 1-{counter})")
print(f"Written        : {OUT}")
print(f"Peptide sequence residues: {counter} (LISDAELEAIFEADC = 15 expected)")
