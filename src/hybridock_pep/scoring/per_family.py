"""Per-family calibration dispatcher with similarity gate and confidence flag.

The v1.3 per-family ridge (``data/calibration_per_family.json``) carries one
ridge fit per receptor family plus a fallback ridge. At inference time the
driver passes a single receptor PDB; this module routes that receptor to its
nearest family by k-mer Jaccard similarity against each family's member
sequences, returns the chosen ridge (or the fallback if no family is close
enough), and emits a confidence band the driver writes to run_metadata.json.

Similarity gate (``MIN_FAMILY_SIM`` = 0.10) — when the maximum Jaccard to any
family member is below 0.10, the receptor is treated as out-of-distribution
and routed to the fallback ridge instead of being forced into the nearest
big family. Without this gate v1.3's intercept range of ~13 kcal/mol can
produce predictions 8+ kcal/mol off when the dispatcher guesses wrong (see
``docs/calibration_notes.md`` generalization study, 2026-06-03).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Literal, NamedTuple

logger = logging.getLogger(__name__)


K = 6  # k-mer size, matches scripts/cluster_and_fit_per_family.py
MIN_FAMILY_SIM: float = 0.10  # below this → fallback ridge
HIGH_CONF_SIM: float = 0.20   # ≥ this → "in-distribution"

# Three-aa-to-one map for sequence extraction.
_AA3to1: dict[str, str] = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}


ConfidenceBand = Literal["in_distribution", "borderline", "out_of_distribution"]


class DispatchResult(NamedTuple):
    """Outcome of routing a receptor to a per-family ridge.

    Attributes:
        ridge: The chosen ridge fit (a dict with ``w_vina``, ``w_contact``,
            ``w_s_ss_weighted``, ``intercept`` keys).
        family_id: Family identifier ("0", "1", …) for the best match, or
            ``"fallback"`` when the similarity gate rejected all big families.
        similarity: The k-mer Jaccard similarity used to pick the family
            (or to gate to fallback).
        confidence_band: ``in_distribution`` (sim ≥ 0.20), ``borderline``
            (0.10 ≤ sim < 0.20), or ``out_of_distribution`` (sim < 0.10).
    """

    ridge: dict
    family_id: str
    similarity: float
    confidence_band: ConfidenceBand


def _chain_sequences(pdb_path: Path) -> dict[str, str]:
    """Return chain_id → 1-letter sequence for every ATOM-record chain."""
    chains: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            res_seq = int(line[22:26].strip())
        except ValueError:
            continue
        chains[line[21]].append((res_seq, line[26], line[17:20].strip()))

    seqs: dict[str, str] = {}
    for cid, triples in chains.items():
        seen, last = [], None
        for r, ic, resn in triples:
            key = (r, ic)
            if key == last:
                continue
            last = key
            seen.append(_AA3to1.get(resn, "X"))
        seqs[cid] = "".join(seen)
    return seqs


def receptor_sequence(pdb_path: Path) -> str | None:
    """Extract the dominant receptor chain sequence from a PDB.

    Strategy: return the longest chain. For HybriDock-Pep usage the receptor
    has already been pocket-cropped + peptide-stripped before reaching this
    point (see ``prep/receptor.py::crop_to_pocket``), so the longest chain is
    the right one.

    Args:
        pdb_path: Receptor PDB file.

    Returns:
        Sequence string, or None if no parseable ATOM records.
    """
    if not pdb_path.exists():
        return None
    seqs = _chain_sequences(pdb_path)
    if not seqs:
        return None
    return max(seqs.values(), key=len)


def _kmer_set(seq: str, k: int = K) -> set[str]:
    """All length-k substrings of seq as a set."""
    if len(seq) < k:
        return set()
    return {seq[i:i+k] for i in range(len(seq) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _confidence_band(sim: float) -> ConfidenceBand:
    if sim >= HIGH_CONF_SIM:
        return "in_distribution"
    if sim >= MIN_FAMILY_SIM:
        return "borderline"
    return "out_of_distribution"


def dispatch_per_family(
    receptor_seq: str,
    cal: dict,
    family_member_kmers: dict[str, list[set[str]]] | None = None,
) -> DispatchResult:
    """Route a receptor sequence to its nearest big family's ridge.

    Similarity = max k-mer Jaccard over all members of each family; we pick
    the family with the highest similarity. If the winning similarity is
    below ``MIN_FAMILY_SIM``, we fall back to the per-family JSON's
    ``fallback`` ridge instead — the per-family intercepts span ~13 kcal/mol
    and routing to the wrong intercept is more harmful than using a single
    less-specific intercept.

    Args:
        receptor_seq: 1-letter sequence of the receptor.
        cal: Loaded per-family calibration dict (schema v3).
        family_member_kmers: Optional pre-computed mapping
            ``family_id → list-of-kmer-sets``. If None, this function
            recomputes from ``cal['families'][fam]['pdbs']`` — slow if
            called per-pose, so callers SHOULD cache.

    Returns:
        DispatchResult with the chosen ridge, family id, similarity, and band.

    Raises:
        ValueError: If ``cal`` does not look like a schema-v3 per-family file.
    """
    families = cal.get("families")
    fallback = cal.get("fallback")
    if not isinstance(families, dict) or not isinstance(fallback, dict):
        raise ValueError(
            "per-family calibration must have both 'families' and 'fallback' keys"
        )
    if family_member_kmers is None:
        raise ValueError(
            "dispatch_per_family requires pre-computed family_member_kmers; "
            "use build_family_kmer_index(cal, raw_pdbs_dir) first"
        )

    q = _kmer_set(receptor_seq)
    best_fam: str | None = None
    best_sim: float = -1.0
    for fam_id, member_kmers in family_member_kmers.items():
        if not member_kmers:
            continue
        sim = max((_jaccard(q, m) for m in member_kmers), default=0.0)
        if sim > best_sim:
            best_sim = sim
            best_fam = fam_id

    if best_fam is None or best_sim < MIN_FAMILY_SIM:
        logger.info(
            "Per-family dispatch: max Jaccard %.3f < %.2f gate → fallback ridge",
            best_sim if best_sim >= 0 else 0.0, MIN_FAMILY_SIM,
        )
        return DispatchResult(
            ridge=fallback,
            family_id="fallback",
            similarity=max(0.0, best_sim),
            confidence_band="out_of_distribution",
        )

    logger.info(
        "Per-family dispatch: family %s (Jaccard %.3f, %s)",
        best_fam, best_sim, _confidence_band(best_sim),
    )
    return DispatchResult(
        ridge=families[best_fam],
        family_id=best_fam,
        similarity=best_sim,
        confidence_band=_confidence_band(best_sim),
    )


def build_family_kmer_index(
    cal: dict,
    raw_pdbs_dir: Path,
) -> dict[str, list[set[str]]]:
    """Pre-compute k-mer sets for every member PDB of every big family.

    This is the heavy step (reads N PDBs once); cache the result and pass it
    to ``dispatch_per_family`` per inference call.

    Args:
        cal: Loaded per-family calibration (schema v3).
        raw_pdbs_dir: Directory containing the family-member PDBs named
            ``<PDB>.pdb`` (case-insensitive).

    Returns:
        Mapping ``family_id → list-of-kmer-sets``. Families whose PDBs cannot
        be found get an empty list (they will never win the max-sim contest).
    """
    out: dict[str, list[set[str]]] = {}
    for fam_id, fit in cal["families"].items():
        member_kmers: list[set[str]] = []
        for pdb in fit.get("pdbs", []):
            candidates = [
                raw_pdbs_dir / f"{pdb.upper()}.pdb",
                raw_pdbs_dir / f"{pdb.lower()}.pdb",
            ]
            pdb_path = next((c for c in candidates if c.exists()), None)
            if pdb_path is None:
                continue
            seq = receptor_sequence(pdb_path)
            if seq:
                member_kmers.append(_kmer_set(seq))
        out[fam_id] = member_kmers
    return out
