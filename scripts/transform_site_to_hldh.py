"""Superimpose hLDH chain A onto PfLDH chain A (Cα), apply the same rigid
transform to the PfLDH binding-site coords to find the equivalent site on hLDH.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

PFLDH = Path("/home/igem/unknown_software/pfldh.pdb")
HLDH = Path("/home/igem/unknown_software/data/pdbs/hldh.pdb")
PFLDH_SITE = np.array([45.149, 32.445, 49.428])


def parse_ca(pdb: Path, chain_id: str) -> tuple[list[int], np.ndarray]:
    """Return (resseqs, Cα xyz array) for the given chain."""
    by_res = {}
    for line in pdb.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[21] != chain_id:
            continue
        if line[12:16].strip() != "CA":
            continue
        try:
            r = int(line[22:26].strip())
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        by_res[r] = (x, y, z)
    res = sorted(by_res.keys())
    coords = np.array([by_res[r] for r in res])
    return res, coords


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, t) such that R @ P.T + t.T fits Q (P, Q are N×3)."""
    cP = P.mean(0); cQ = Q.mean(0)
    Pc = P - cP; Qc = Q - cQ
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.eye(3); D[2, 2] = d
    R = Vt.T @ D @ U.T
    t = cQ - R @ cP
    return R, t


def main() -> None:
    # Parse Cα of both chain A's
    pf_res, pf_ca = parse_ca(PFLDH, "A")
    hl_res, hl_ca = parse_ca(HLDH, "A")
    print(f"PfLDH chain A: {len(pf_res)} Cα ({pf_res[0]}..{pf_res[-1]})")
    print(f"hLDH chain A:  {len(hl_res)} Cα ({hl_res[0]}..{hl_res[-1]})")

    # Match by residue number (LDHs are homologs with conserved numbering)
    common = sorted(set(pf_res) & set(hl_res))
    if len(common) < 50:
        print(f"WARN: only {len(common)} shared residues — alignment may be off")
    pf_idx = {r: i for i, r in enumerate(pf_res)}
    hl_idx = {r: i for i, r in enumerate(hl_res)}
    P = np.array([pf_ca[pf_idx[r]] for r in common])  # PfLDH coords
    Q = np.array([hl_ca[hl_idx[r]] for r in common])  # hLDH coords
    print(f"Aligning on {len(common)} shared residue numbers")

    # Kabsch: rotate PfLDH → hLDH frame
    R, t = kabsch(P, Q)
    P_aligned = (R @ P.T).T + t
    rmsd = float(np.sqrt(((P_aligned - Q) ** 2).sum(-1).mean()))
    print(f"Kabsch RMSD on Cα: {rmsd:.2f} Å")

    # Transform the PfLDH site coords
    hldh_site = R @ PFLDH_SITE + t
    print(f"\nPfLDH site:  ({PFLDH_SITE[0]:.3f}, {PFLDH_SITE[1]:.3f}, {PFLDH_SITE[2]:.3f})")
    print(f"hLDH site:   ({hldh_site[0]:.3f}, {hldh_site[1]:.3f}, {hldh_site[2]:.3f})")

    # Sanity: how many hLDH residues fall within 20 Å of the new site?
    all_atoms = []
    for line in HLDH.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            all_atoms.append([x, y, z])
        except ValueError:
            continue
    A = np.array(all_atoms)
    d = np.linalg.norm(A - hldh_site, axis=1)
    print(f"hLDH atoms within 20 Å of new site: {(d < 20).sum()} / {len(A)}")
    print(f"\nUse for hLDH dock:")
    print(f"  --site {hldh_site[0]:.3f} {hldh_site[1]:.3f} {hldh_site[2]:.3f}")


if __name__ == "__main__":
    main()
