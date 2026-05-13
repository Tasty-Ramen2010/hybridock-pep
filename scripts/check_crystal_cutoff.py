import numpy as np
from pathlib import Path
lines = Path("/home/igem/unknown_software/raw/hldh_crystal.pdb").read_text().splitlines()
site = np.array([39.5, 23.75, 128.35])
for cutoff in [20, 25, 30, 35]:
    res = set()
    atoms = 0
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        try:
            x,y,z = float(line[30:38]),float(line[38:46]),float(line[46:54])
            if float(np.linalg.norm(np.array([x,y,z])-site)) <= cutoff:
                res.add((line[21],line[22:26].strip()))
                atoms += 1
        except Exception:
            pass
    chains = sorted(set(r[0] for r in res))
    print(f"{cutoff}A: {len(res)} residues, {atoms} atoms, chains {chains}")
