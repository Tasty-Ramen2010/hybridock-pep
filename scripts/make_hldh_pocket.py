"""Extract hLDH pocket residues within a given cutoff of the ADCP binding site."""
from __future__ import annotations
import numpy as np
from pathlib import Path

pdb = Path("/home/igem/unknown_software/data/pdbs/hldh.pdb")
site = np.array([49.832, 15.634, 2.544])
cutoff = 20.0

residues_to_keep: set[tuple[str, str, str]] = set()
lines = pdb.read_text().splitlines()

for line in lines:
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        continue
    try:
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        chain = line[21]
        resseq = line[22:26].strip()
        icode = line[26].strip()
        if float(np.linalg.norm(np.array([x, y, z]) - site)) <= cutoff:
            residues_to_keep.add((chain, resseq, icode))
    except (ValueError, IndexError):
        continue

print(f"Residues within {cutoff}Å: {len(residues_to_keep)}")
chains = sorted(set(r[0] for r in residues_to_keep))
print(f"Chains contributing: {chains}")

kept_lines = []
for line in lines:
    if line.startswith("ATOM") or line.startswith("HETATM"):
        chain = line[21]
        resseq = line[22:26].strip()
        icode = line[26].strip()
        if (chain, resseq, icode) in residues_to_keep:
            kept_lines.append(line)

kept_lines.append("END")
out = Path("/home/igem/unknown_software/data/pdbs/hldh_pocket.pdb")
out.write_text("\n".join(kept_lines) + "\n")
print(f"Atoms in pocket: {len(kept_lines) - 1}")
print(f"Written: {out}")
