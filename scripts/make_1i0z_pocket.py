"""Extract hLDH (1I0Z) chain-A pocket within 20Å of the OXM binding site."""
from __future__ import annotations
import numpy as np
from pathlib import Path

pdb = Path("/home/igem/unknown_software/data/pdbs/1I0Z.pdb")
# OXM (oxamate = substrate analog) center in chain A
site = np.array([23.991, 47.586, 56.913])
cutoff = 20.0

lines = pdb.read_text().splitlines()

# Use chain A ATOM records only
chain_a_lines = [l for l in lines if (l.startswith("ATOM") and l[21] == "A")]

residues_to_keep: set[tuple[str, str]] = set()
for line in chain_a_lines:
    try:
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        resseq = line[22:26].strip()
        icode = line[26].strip()
        if float(np.linalg.norm(np.array([x, y, z]) - site)) <= cutoff:
            residues_to_keep.add((resseq, icode))
    except (ValueError, IndexError):
        continue

print(f"Residues within {cutoff}Å of OXM site: {len(residues_to_keep)}")

kept = []
for line in chain_a_lines:
    resseq = line[22:26].strip()
    icode = line[26].strip()
    if (resseq, icode) in residues_to_keep:
        kept.append(line)
kept.append("END")

out = Path("/home/igem/unknown_software/data/pdbs/hldh_1i0z_pocket.pdb")
out.write_text("\n".join(kept) + "\n")
print(f"Atoms: {len(kept) - 1}")
print(f"Site coords for docking: {site[0]:.3f} {site[1]:.3f} {site[2]:.3f}")
print(f"Written: {out}")
