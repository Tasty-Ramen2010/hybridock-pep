import numpy as np
from pathlib import Path

pdb = Path("/home/igem/unknown_software/data/pdbs/hldh.pdb")
site = np.array([49.832, 15.634, 2.544])

for cutoff in [18.0, 20.0, 22.0]:
    residues = set()
    atoms = 0
    for line in pdb.read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            chain, resseq, icode = line[21], line[22:26].strip(), line[26].strip()
            if np.linalg.norm(np.array([x,y,z]) - site) <= cutoff:
                residues.add((chain, resseq, icode))
                atoms += 1
        except (ValueError, IndexError):
            continue
    chains = sorted(set(r[0] for r in residues))
    print(f"{cutoff}A: {len(residues)} residues, {atoms} atoms, chains {chains}")
