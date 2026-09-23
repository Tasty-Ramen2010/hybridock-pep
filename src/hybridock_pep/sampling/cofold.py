"""Co-folding as a pose prior for docking.

WHY THIS MODULE EXISTS
----------------------
Co-folding models (Boltz-2, AlphaFold 3, Chai-1) predict a complex from sequences alone.
They are not dockers: they take no receptor structure, so they cannot answer the question
this tool is built to answer -- "where does this peptide sit on *my* structure". But on the
one geometry class we measurably fail at, they are better than our sampler at the thing we
get wrong.

The measured failure (see docs/coventry_findings.md) is directional, not positional. Against
de novo designed repeat-groove binders our poses land 0.9 A from the right place with a 2.6 A
Kabsch shape error, and then thread the groove BACKWARDS in 79-81% of cases, because a
pseudo-symmetric repeat groove looks the same from both ends. Re-ranking cannot recover it:
ref2015 picks the forward pose only 41.0% of the time and the best of five geometric
descriptors reaches AUC 0.603. But when the direction is right, our scoring works -- in the
threading test the cognate sequence ranked #1 in 4 of 5 cases with 9-12 REU of margin.

So the useful thing to import from a co-folding model is not its coordinates. It is its
ANSWER TO WHICH WAY ROUND THE PEPTIDE GOES. That is a far lower bar than getting the pose
right, and it is exactly the bit we cannot get on our own.

HOW IT ENTERS THE PIPELINE
--------------------------
Co-folding happens in its own frame with its own receptor conformation, so its output cannot
be scored directly against the user's receptor. Three steps make it usable:

  1. transplant  -- superimpose the co-folded receptor onto the user's receptor and carry the
                    peptide across with the same rigid transform. Superposition is done on
                    POCKET residues by default (the co-folded receptor may differ elsewhere,
                    and only the pocket frame matters for the peptide).
  2. gate        -- if the two receptors do not agree on the fold (CA RMSD over the aligned
                    pocket), the transplant is meaningless and is dropped rather than quietly
                    contributing a wrong pose. Same if the peptide lands outside the user's
                    --site box: that is a disagreement to report, not to average away.
  3. use         -- either as one more candidate pose that competes on the same scoring
                    function as everything else (`pose` mode), or as a direction prior that
                    filters the RAPiDock pool (`prior` mode), or both.

The transplanted peptide was built against the co-folded receptor's sidechains and will clash
against the user's. It must be interface-repacked and minimised before scoring (the caller
does this; scripts/coventry_refine.py is the same operation).

HONEST STATUS
-------------
The evidence that motivated this is one complex: Boltz-2 co-folds 9CCE to 1.99 A backbone
RMSD where our sampler manages 5.66 A at N=500, with the reversed threading at 19.98 A, and
9CCE was released after Boltz's training cutoff so it is a genuine prediction. One complex is
not a result. `scripts/cofold_direction_bench.py` measures the direction-agreement rate that
this module's `prior` mode actually depends on; do not claim the prior works until that
benchmark reports it on a held-out set.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

AA3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M",
}

#: Residues within this distance of the peptide define the pocket used for superposition.
POCKET_RADIUS = 10.0
#: Above this pocket CA RMSD the two receptors are not the same structure and the transplant
#: is refused. 3.0 A is generous -- on 9CCE the Boltz/crystal receptor agreed to 1.26 A.
MAX_FOLD_RMSD = 3.0


@dataclass(frozen=True)
class Chain:
    """One polypeptide chain read from a PDB file."""

    chain_id: str
    seq: str
    ca: np.ndarray                      # (n_res, 3)
    resnums: list[int]
    atoms: list[tuple[str, np.ndarray]] = field(default_factory=list, repr=False)


@dataclass(frozen=True)
class Transplant:
    """A co-folded peptide moved into the user receptor's frame.

    Attributes:
        peptide_ca: Peptide CA coordinates in the user receptor's frame, (n_res, 3).
        peptide_atoms: All peptide atoms as (pdb_line_prefix, xyz) in the user's frame.
        fold_rmsd: CA RMSD over the residues used for superposition, in angstrom. The gate.
        n_aligned: How many residue pairs the superposition used.
        site_offset: Distance from the transplanted peptide centroid to the requested site,
            or None if no site was given.
        direction: Unit vector from peptide N-terminal CA to C-terminal CA, in the user frame.
        centroid: Peptide CA centroid in the user frame.
    """

    peptide_ca: np.ndarray
    peptide_atoms: list[tuple[str, np.ndarray]]
    fold_rmsd: float
    n_aligned: int
    site_offset: float | None
    direction: np.ndarray
    centroid: np.ndarray

    @property
    def accepted(self) -> bool:
        """Whether the fold agreement is good enough for the transplant to mean anything."""
        return self.fold_rmsd <= MAX_FOLD_RMSD


def read_chains(pdb: Path) -> dict[str, Chain]:
    """Read every protein chain of a PDB file.

    Args:
        pdb: Path to a PDB-format file. mmCIF is not accepted; convert first.

    Returns:
        Mapping of chain id to Chain. Chains with no standard residues are omitted.

    Raises:
        FileNotFoundError: If pdb does not exist.
    """
    per_chain: dict[str, dict] = {}
    for line in pdb.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        resname = line[17:20].strip()
        if resname not in AA3to1:
            continue
        cid = line[21]
        d = per_chain.setdefault(cid, {"seq": [], "ca": [], "res": [], "atoms": []})
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        d["atoms"].append((line[:30], xyz))
        if line[12:16].strip() == "CA":
            d["seq"].append(AA3to1[resname])
            d["ca"].append(xyz)
            d["res"].append(int(line[22:26]))
    return {
        cid: Chain(cid, "".join(d["seq"]), np.asarray(d["ca"]), d["res"], d["atoms"])
        for cid, d in per_chain.items()
        if d["ca"]
    }


def split_complex(pdb: Path, peptide_max_len: int = 30) -> tuple[Chain, Chain]:
    """Split a co-folded complex into (receptor, peptide).

    The peptide is the shortest chain, which is unambiguous for our inputs -- binders are
    104-261 aa and peptides are 8-25 aa.

    Args:
        pdb: Co-folded complex, receptor and peptide as separate chains.
        peptide_max_len: Reject if the shortest chain is longer than this, which means the
            file is not a protein-peptide complex and the caller has the wrong input.

    Returns:
        (receptor chain, peptide chain).

    Raises:
        ValueError: If the file has fewer than two protein chains, or the shortest chain is
            longer than peptide_max_len.
    """
    chains = read_chains(pdb)
    if len(chains) < 2:
        raise ValueError(f"{pdb} has {len(chains)} protein chain(s); need at least 2")
    order = sorted(chains.values(), key=lambda c: len(c.seq))
    pep, rec = order[0], order[-1]
    if len(pep.seq) > peptide_max_len:
        raise ValueError(
            f"{pdb}: shortest chain is {len(pep.seq)} aa, longer than peptide_max_len="
            f"{peptide_max_len}; this does not look like a protein-peptide complex"
        )
    return rec, pep


def align_residues(a: str, b: str) -> list[tuple[int, int]]:
    """Map residue indices of sequence a onto sequence b by global alignment.

    The co-folded receptor is built from the full sequence while the user's PDB may have
    gaps, disordered termini or different numbering, so index i of one is not index i of the
    other. A global alignment recovers the correspondence.

    Args:
        a: First sequence, one-letter codes.
        b: Second sequence, one-letter codes.

    Returns:
        List of (index in a, index in b) for aligned, non-gap positions.
    """
    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner()
    aligner.open_gap_score = -11.0
    aligner.extend_gap_score = -1.0
    aligner.substitution_matrix = __import__(
        "Bio.Align.substitution_matrices", fromlist=["load"]
    ).load("BLOSUM62")
    aln = aligner.align(a, b)[0]
    pairs: list[tuple[int, int]] = []
    for (a0, a1), (b0, b1) in zip(*aln.aligned):
        pairs.extend((a0 + k, b0 + k) for k in range(a1 - a0))
    return pairs


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Optimal rigid superposition of mobile onto target.

    Args:
        mobile: (n, 3) coordinates to move.
        target: (n, 3) reference coordinates, same length as mobile.

    Returns:
        (rotation 3x3, translation 3, RMSD after superposition).

    Raises:
        ValueError: If the two arrays differ in length or have fewer than 3 points.
    """
    if mobile.shape != target.shape or len(mobile) < 3:
        raise ValueError(f"kabsch needs >=3 matched points, got {mobile.shape}/{target.shape}")
    mc, tc = mobile.mean(0), target.mean(0)
    P, Q = mobile - mc, target - tc
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    moved = (R @ P.T).T
    rmsd = float(np.sqrt(((moved - Q) ** 2).sum(1).mean()))
    return R, tc - (R @ mc), rmsd


def transplant(
    cofolded: Path,
    receptor: Path,
    site: tuple[float, float, float] | None = None,
    pocket_radius: float = POCKET_RADIUS,
) -> Transplant:
    """Move a co-folded peptide into the user receptor's coordinate frame.

    Superposition uses only pocket residues -- those of the co-folded receptor within
    pocket_radius of its own peptide -- because the two receptors may legitimately differ in
    loops and termini, and only the frame around the binding site governs where the peptide
    should sit. A global superposition on a 260-residue binder can be dragged several
    angstrom off by a disordered tail and put the peptide in the wrong place.

    Args:
        cofolded: Co-folded complex PDB (receptor + peptide as separate chains).
        receptor: The user's receptor PDB -- the structure everything is scored against.
        site: Optional requested binding-site centre, for the site-agreement report.
        pocket_radius: Radius in angstrom defining the pocket used for superposition.

    Returns:
        Transplant carrying the peptide in the user's frame plus the gate diagnostics.
        Check `.accepted` before using the coordinates.

    Raises:
        ValueError: If the complex cannot be split, or fewer than 3 pocket residues align
            between the two receptors.
    """
    cf_rec, cf_pep = split_complex(cofolded)
    user = read_chains(receptor)
    if not user:
        raise ValueError(f"{receptor} has no standard protein residues")
    user_rec = max(user.values(), key=lambda c: len(c.seq))

    pairs = align_residues(cf_rec.seq, user_rec.seq)
    if len(pairs) < 3:
        raise ValueError(
            f"co-folded receptor ({len(cf_rec.seq)} aa) and user receptor "
            f"({len(user_rec.seq)} aa) share only {len(pairs)} aligned residues"
        )

    # Pocket = co-folded receptor residues near its own peptide. Restricting the
    # superposition to these is what makes the transplant robust to differences elsewhere.
    d = np.linalg.norm(cf_rec.ca[:, None, :] - cf_pep.ca[None, :, :], axis=-1).min(axis=1)
    pocket = set(np.flatnonzero(d <= pocket_radius).tolist())
    sel = [(i, j) for i, j in pairs if i in pocket]
    if len(sel) < 3:
        logger.warning(
            "only %d pocket residues aligned; falling back to global superposition", len(sel)
        )
        sel = pairs
    mob = cf_rec.ca[[i for i, _ in sel]]
    tgt = user_rec.ca[[j for _, j in sel]]
    R, t, rmsd = kabsch(mob, tgt)

    pep_ca = (R @ cf_pep.ca.T).T + t
    pep_atoms = [(pre, R @ xyz + t) for pre, xyz in cf_pep.atoms]
    centroid = pep_ca.mean(0)
    axis = pep_ca[-1] - pep_ca[0]
    norm = float(np.linalg.norm(axis))
    direction = axis / norm if norm > 1e-6 else np.zeros(3)
    offset = float(np.linalg.norm(centroid - np.asarray(site))) if site else None

    logger.info(
        "co-fold transplant: %d pocket residues, fold RMSD %.2f A%s",
        len(sel), rmsd,
        f", peptide centroid {offset:.1f} A from requested site" if offset is not None else "",
    )
    return Transplant(pep_ca, pep_atoms, rmsd, len(sel), offset, direction, centroid)


def write_transplanted_peptide(tr: Transplant, out: Path, chain: str = "B") -> Path:
    """Write the transplanted peptide alone, in the user receptor's frame.

    RAPiDock poses are peptide-only files and the refinement path (scripts/coventry_refine.py)
    appends the peptide to a freshly loaded receptor, so a co-folded pose has to be handed
    over in the same shape. Passing a two-chain complex there instead silently appends a
    SECOND copy of the receptor and produces interface energies in the thousands of REU.

    Args:
        tr: Accepted transplant.
        out: Destination path.
        chain: Chain id to stamp on the peptide.

    Returns:
        The path written.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, (pre, xyz) in enumerate(tr.peptide_atoms, 1):
        lines.append(
            f"{pre[:6]}{i:5d}{pre[11:21]}{chain}{pre[22:30]}"
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00"
        )
    out.write_text("\n".join(lines) + "\nTER\nEND\n")
    return out


def write_transplanted_pose(tr: Transplant, receptor: Path, out: Path) -> Path:
    """Write the transplanted peptide next to the user's receptor as a two-chain complex.

    The receptor is chain A and the peptide chain B, matching what the rest of the pipeline
    (pose_io.parse_poses, the refinement scripts) expects. The output still has the co-folded
    peptide's sidechain placement against a receptor it was not built for, so it must be
    interface-repacked before its energy means anything.

    Args:
        tr: Accepted transplant.
        receptor: The user's receptor PDB.
        out: Destination path.

    Returns:
        The path written.
    """
    lines, serial = [], 0
    for line in receptor.read_text().splitlines():
        if line.startswith("ATOM"):
            serial += 1
            lines.append(f"{line[:6]}{serial:5d}{line[11:21]}A{line[22:]}")
    lines.append("TER")
    for pre, xyz in tr.peptide_atoms:
        serial += 1
        lines.append(
            f"{pre[:6]}{serial:5d}{pre[11:21]}B{pre[22:30]}"
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00"
        )
    lines.append("TER")
    lines.append("END")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out
