import numpy as np
from pathlib import Path

site = np.array([49.832, 15.634, 2.544])
lines = Path("/home/igem/unknown_software/data/pdbs/hldh.pdb").read_text().splitlines()

near = []
for line in lines:
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        continue
    try:
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        dist = float(np.linalg.norm(np.array([x-site[0], y-site[1], z-site[2]])))
        if dist < 8.0:
            near.append((dist, line[17:20].strip(), line[21], line[22:26].strip(), line[12:16].strip()))
    except Exception:
        pass

near.sort()
print(f"Residues within 8A of ADCP site {site} in hldh.pdb:")
seen = set()
for dist, resname, chain, resseq, atom in near[:30]:
    key = (chain, resseq)
    if key not in seen:
        print(f"  {resname} {chain}{resseq}  ({dist:.1f}A)")
        seen.add(key)

# Also check 1I0Z for comparison
print()
site2 = np.array([23.991, 47.586, 56.913])
lines2 = Path("/home/igem/unknown_software/data/pdbs/1I0Z.pdb").read_text().splitlines()
near2 = []
for line in lines2:
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        continue
    try:
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        dist = float(np.linalg.norm(np.array([x-site2[0], y-site2[1], z-site2[2]])))
        if dist < 6.0:
            near2.append((dist, line[17:20].strip(), line[21], line[22:26].strip(), line[12:16].strip()))
    except Exception:
        pass

near2.sort()
print(f"Residues within 6A of OXM site {site2} in 1I0Z:")
seen2 = set()
for dist, resname, chain, resseq, atom in near2[:20]:
    key = (chain, resseq)
    if key not in seen2:
        print(f"  {resname} {chain}{resseq}  ({dist:.1f}A)")
        seen2.add(key)
