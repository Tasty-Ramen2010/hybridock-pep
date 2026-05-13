"""Extract hLDH crystal tetramer pocket at C-D interface within 35Å of site."""
import numpy as np
from pathlib import Path

pdb = Path("/home/igem/unknown_software/raw/hldh_crystal.pdb")
lines = pdb.read_text().splitlines()

# C5 Cα=(42.6,26.1,132.6), D176 Cα=(36.5,21.4,124.1) → midpoint
site = np.array([39.5, 23.75, 128.35])
cutoff = 35.0

residues_to_keep: set[tuple[str, str, str]] = set()
for line in lines:
    if not line.startswith("ATOM"):
        continue
    try:
        x,y,z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        chain, resseq, icode = line[21], line[22:26].strip(), line[26].strip()
        if float(np.linalg.norm(np.array([x,y,z]) - site)) <= cutoff:
            residues_to_keep.add((chain, resseq, icode))
    except Exception:
        continue

print(f"Residues within {cutoff}Å: {len(residues_to_keep)}")
chains = sorted(set(r[0] for r in residues_to_keep))
print(f"Chains: {chains}")

kept = [l for l in lines if l.startswith("ATOM") and
        (l[21], l[22:26].strip(), l[26].strip()) in residues_to_keep]
kept.append("END")
out = Path("/home/igem/unknown_software/data/pdbs/hldh_crystal_pocket.pdb")
out.write_text("\n".join(kept) + "\n")
print(f"Atoms: {len(kept)-1}")
print(f"Site: {site[0]:.3f} {site[1]:.3f} {site[2]:.3f}")
print(f"Written: {out}")
