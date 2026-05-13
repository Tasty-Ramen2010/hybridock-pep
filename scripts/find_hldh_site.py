"""Find binding site in hldh_crystal.pdb by checking ADCP site coords and C-D interface."""
import numpy as np
from pathlib import Path

pdb = Path("/home/igem/unknown_software/raw/hldh_crystal.pdb")
lines = pdb.read_text().splitlines()

adcp_site = np.array([49.832, 15.634, 2.544])
print(f"=== Near ADCP site {adcp_site} ===")
near = []
for line in lines:
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        continue
    try:
        x,y,z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        dist = float(np.linalg.norm(np.array([x,y,z]) - adcp_site))
        if dist < 8.0:
            near.append((dist, line[17:20].strip(), line[21], line[22:26].strip()))
    except Exception:
        pass
near.sort()
seen: set = set()
for d,rn,ch,rs in near[:15]:
    if (ch,rs) not in seen:
        print(f"  {rn} {ch}{rs} ({d:.1f}A)")
        seen.add((ch,rs))

# Find C-D interface centroid
print("\n=== C-D interface: chain C residues 1-10 and D residues 170-185 ===")
cd_coords = []
for line in lines:
    if not line.startswith("ATOM"):
        continue
    chain = line[21]
    resseq = int(line[22:26].strip())
    atom = line[12:16].strip()
    if atom != "CA":
        continue
    if (chain == "C" and resseq <= 15) or (chain == "D" and 170 <= resseq <= 190):
        try:
            x,y,z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            cd_coords.append([x,y,z])
            print(f"  {chain}{resseq}: {x:.1f} {y:.1f} {z:.1f}")
        except Exception:
            pass
if cd_coords:
    c = np.mean(cd_coords, axis=0)
    print(f"  Interface centroid: {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}")

# Also compute global center of mass of all 4 chains
print("\n=== Chain centroids (CA atoms) ===")
for ch in "ABCD":
    coords = []
    for line in lines:
        if line.startswith("ATOM") and line[21]==ch and line[12:16].strip()=="CA":
            try:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except Exception:
                pass
    if coords:
        c = np.mean(coords, axis=0)
        print(f"  Chain {ch} centroid: {c[0]:.2f} {c[1]:.2f} {c[2]:.2f}")
