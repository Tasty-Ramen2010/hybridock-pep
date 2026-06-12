"""Proper sequence-aligned Kabsch superposition: PfLDH chain A → hLDH chain A.

The v1 script naively matched residues by number which fails for homologs
with different numbering (PfLDH 18..329, hLDH 2..333, only 10% identity at
shared numbers → wrong alignment, 5.79 Å Kabsch RMSD, wrong site coords).

This version:
  1. Extracts each chain's full sequence.
  2. Runs Needleman-Wunsch global alignment (BLOSUM62, affine gaps).
  3. Builds the matched-Cα pair set from the alignment.
  4. Kabsch on those pairs only — expects ~2-3 Å RMSD on real LDH homologs.
  5. Transforms the PfLDH site coords through the same R, t.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from Bio import PDB  # noqa: F401
    from Bio import Align
    from Bio.Align import substitution_matrices
except ImportError as exc:
    raise SystemExit(f"BioPython required: {exc}")

PFLDH = Path("/home/igem/unknown_software/pfldh.pdb")
HLDH = Path("/home/igem/unknown_software/data/pdbs/hldh.pdb")
PFLDH_SITE = np.array([45.149, 32.445, 49.428])


AA3to1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q",
          "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
          "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}


def chain_seq_and_ca(pdb: Path, chain_id: str):
    """Return (resseqs ordered, sequence string, Cα xyz array)."""
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
        if r not in by_res:
            by_res[r] = (AA3to1.get(line[17:20].strip(), "X"), (x, y, z))
    res = sorted(by_res.keys())
    seq = "".join(by_res[r][0] for r in res)
    ca = np.array([by_res[r][1] for r in res])
    return res, seq, ca


def kabsch(P: np.ndarray, Q: np.ndarray):
    cP = P.mean(0); cQ = Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.eye(3); D[2, 2] = d
    R = Vt.T @ D @ U.T
    t = cQ - R @ cP
    return R, t


def main() -> None:
    pf_res, pf_seq, pf_ca = chain_seq_and_ca(PFLDH, "A")
    hl_res, hl_seq, hl_ca = chain_seq_and_ca(HLDH, "A")
    print(f"PfLDH chain A: {len(pf_seq)} aa  ({pf_res[0]}..{pf_res[-1]})")
    print(f"hLDH  chain A: {len(hl_seq)} aa  ({hl_res[0]}..{hl_res[-1]})")

    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.mode = "global"
    alignments = aligner.align(pf_seq, hl_seq)
    aln = alignments[0]
    print(f"\nGlobal alignment score: {aln.score:.0f}")

    # Build matched Cα pairs by walking the aligned columns
    pf_i = hl_i = 0
    matched_pf, matched_hl = [], []
    matches = mismatches = 0
    for a_pf, a_hl in zip(str(aln[0]), str(aln[1])):
        if a_pf == "-":
            hl_i += 1; continue
        if a_hl == "-":
            pf_i += 1; continue
        # both aligned residues — record Cα pair
        matched_pf.append(pf_ca[pf_i])
        matched_hl.append(hl_ca[hl_i])
        if a_pf == a_hl:
            matches += 1
        else:
            mismatches += 1
        pf_i += 1; hl_i += 1

    n = len(matched_pf)
    print(f"Aligned pairs: {n}  ({matches} identical, {mismatches} differ)")
    print(f"Sequence identity in alignment: {100*matches/n:.1f}%")
    P = np.array(matched_pf); Q = np.array(matched_hl)

    R, t = kabsch(P, Q)
    P_aln = (R @ P.T).T + t
    rmsd = float(np.sqrt(((P_aln - Q) ** 2).sum(-1).mean()))
    print(f"Kabsch Cα RMSD on aligned pairs: {rmsd:.2f} Å")

    # Iteratively prune outliers (>5 Å), re-Kabsch, until stable — gives the
    # core-domain alignment, ignoring flexible loops.
    for _ in range(3):
        d = np.linalg.norm((R @ P.T).T + t - Q, axis=1)
        keep = d < 5.0
        if keep.sum() < 50 or keep.sum() == len(P):
            break
        P = P[keep]; Q = Q[keep]
        R, t = kabsch(P, Q)
        rmsd = float(np.sqrt(((R @ P.T).T + t - Q) ** 2).sum(-1).mean() ** 0.5)
        print(f"  After pruning to {len(P)} core residues: Kabsch RMSD = {rmsd:.2f} Å")

    # Transform PfLDH site → hLDH frame
    hldh_site = R @ PFLDH_SITE + t
    print(f"\nPfLDH site:    ({PFLDH_SITE[0]:8.3f}, {PFLDH_SITE[1]:8.3f}, {PFLDH_SITE[2]:8.3f})")
    print(f"hLDH site:     ({hldh_site[0]:8.3f}, {hldh_site[1]:8.3f}, {hldh_site[2]:8.3f})")
    print(f"\n  --site {hldh_site[0]:.3f} {hldh_site[1]:.3f} {hldh_site[2]:.3f}")

    # Sanity: identify the closest residues to the new site
    by_res = {}
    for line in HLDH.read_text().splitlines():
        if not line.startswith("ATOM"): continue
        try:
            r = int(line[22:26].strip())
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError: continue
        key = (line[21], r, line[17:20].strip())
        d_ = float(np.linalg.norm(np.array([x, y, z]) - hldh_site))
        if key not in by_res or d_ < by_res[key]:
            by_res[key] = d_
    closest = sorted(by_res.items(), key=lambda x: x[1])[:15]
    print(f"\nClosest 15 hLDH residues to new site:")
    for (chain, resseq, resn), d_ in closest:
        print(f"   {resn}{resseq}{chain}  {d_:.2f} Å")


if __name__ == "__main__":
    main()
