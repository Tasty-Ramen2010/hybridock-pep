#!/usr/bin/env python3
"""
prescreen_calibration.py — structural quality audit of the calibration dataset.

Runs BEFORE score_calibration_set.py to catch data quality issues that would
produce garbage scores or misleading calibration.

Checks per entry
────────────────
1.  Chain count & sizes           — flags 4-chain symmetric dimers
2.  Peptide chain validity        — actual PDB residues vs CSV seq length
3.  Chain-pairing correctness     — is the auto-detected peptide actually in
                                    contact with the specified receptor chain?
4.  Min inter-chain distance      — sanity: should be < 4 Å for a real complex
5.  Interface contacts (pep-side) — count before scoring
6.  n_contact_residues overflow   — contacts > peptide length (impossible)
7.  Affinity type flag            — IC50 != Kd; mixed types distort calibration
8.  Pre-existing score data       — cross-references training_scores_full.json
                                    if it exists

Verdict
───────
RED  — entry should be excluded from calibration; reason given
FLAG — entry may be usable but warrants manual inspection
PASS — no structural issues detected

Usage
─────
    python scripts/prescreen_calibration.py                     # uses defaults
    python scripts/prescreen_calibration.py --csv data/training_complexes_full.csv \\
        --output runs/screening_report.csv --scores data/training_scores_full.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

# ─── constants ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent

MAX_PEP_RESIDUES = 30          # Å — longer → it's a protein domain, not a peptide
MIN_PEP_RESIDUES = 4           # Shorter than this → probably not a real peptide
MAX_CONTACT_DIST_ANG = 4.0     # Å — receptor and peptide must be at least this close
CONTACT_CUTOFF_ANG = 4.5       # Å — contact residue definition (matches entropy.py)
MAX_LEN_MISMATCH = 3           # tolerated difference: CSV seq len vs actual PDB len
SAMPLE_ATOMS = 800             # max atoms to use in brute-force distance calculations
PDB_SEARCH_DIRS = [
    REPO / "datasets" / "raw_pdbs",
    REPO / "datasets" / "pdb_2019_2023" / "structures",
    REPO / "datasets" / "pdb_2010_2018" / "structures",
    REPO / "datasets" / "pdb_2024_2026" / "structures",
    REPO / "datasets" / "pdb_pre2010" / "structures",
]

# ─── PDB parsing helpers ───────────────────────────────────────────────────────

def _read_pdb(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="ignore") as fh:
            return fh.read()
    return path.read_text(errors="ignore")


def _find_pdb(pdb_id: str) -> Path | None:
    pid = pdb_id.upper()
    pid_lo = pdb_id.lower()
    candidates = []
    for d in PDB_SEARCH_DIRS:
        if not d.exists():
            continue
        for fname in (f"{pid}.pdb", f"{pid_lo}.pdb", f"{pid}.pdb.gz", f"{pid_lo}.pdb.gz"):
            p = d / fname
            if p.exists():
                candidates.append(p)
    return candidates[0] if candidates else None


def _chain_residues(pdb_text: str) -> dict[str, set[tuple]]:
    """Returns {chain: set_of_residue_ids} from ATOM records (ignores HETATM)."""
    chains: dict[str, set[tuple]] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if len(line) < 27:
            continue
        ch = line[21:22]
        try:
            resnum = int(line[22:26].strip())
        except ValueError:
            continue
        ins = line[26:27].strip()
        chains.setdefault(ch, set()).add((resnum, ins))
    return chains


def _heavy_atoms(pdb_text: str, chain: str) -> np.ndarray:
    """xyz array of non-H ATOM heavy atoms for a given chain."""
    pts: list[tuple[float, float, float]] = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if len(line) < 54:
            continue
        if line[21:22] != chain:
            continue
        atom_name = line[12:16].strip()
        if atom_name.startswith("H"):
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            continue
        pts.append((x, y, z))
    return np.array(pts, dtype=float) if pts else np.zeros((0, 3))


def _heavy_atoms_by_res(pdb_text: str, chain: str) -> dict[tuple, list[tuple]]:
    """Map residue_id → list of (x,y,z) for non-H ATOM atoms in chain."""
    out: dict[tuple, list[tuple]] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if len(line) < 54:
            continue
        if line[21:22] != chain:
            continue
        atom_name = line[12:16].strip()
        if atom_name.startswith("H"):
            continue
        try:
            resnum = int(line[22:26].strip())
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            continue
        ins = line[26:27].strip()
        key = (resnum, ins)
        out.setdefault(key, []).append((x, y, z))
    return out


def _min_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Min pairwise distance between two atom sets (sampled for speed)."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("inf")
    a2 = a[:SAMPLE_ATOMS]
    b2 = b[:SAMPLE_ATOMS]
    # Block computation to avoid huge memory use
    min_d = float("inf")
    block = 200
    for i in range(0, len(a2), block):
        diff = a2[i:i+block, np.newaxis, :] - b2[np.newaxis, :, :]
        d = float(np.sqrt((diff ** 2).sum(axis=2)).min())
        if d < min_d:
            min_d = d
    return min_d


def _count_contacts(rec_xyz: np.ndarray, pep_by_res: dict, cutoff: float = CONTACT_CUTOFF_ANG) -> int:
    """Number of peptide residues with ≥1 atom within cutoff of receptor."""
    if rec_xyz.shape[0] == 0:
        return 0
    rec2 = rec_xyz[:SAMPLE_ATOMS]
    cutoff_sq = cutoff ** 2
    n = 0
    for coords in pep_by_res.values():
        arr = np.array(coords, dtype=float)
        diff = arr[:, np.newaxis, :] - rec2[np.newaxis, :, :]
        if (diff ** 2).sum(axis=2).min() <= cutoff_sq:
            n += 1
    return n


# ─── chain-pairing logic ───────────────────────────────────────────────────────

def _candidate_peptide_chains(
    pdb_text: str,
    rec_chain: str,
    rec_xyz: np.ndarray,
) -> list[tuple[str, int, float, int]]:
    """
    For each non-receptor chain, return (chain_id, n_residues, min_dist_to_rec, n_contacts).
    Sorted by min_dist_to_rec ascending (closest first).
    """
    chain_res = _chain_residues(pdb_text)
    result = []
    for ch, res_set in chain_res.items():
        if ch == rec_chain:
            continue
        n_res = len(res_set)
        ch_xyz = _heavy_atoms(pdb_text, ch)
        dist = _min_dist(rec_xyz, ch_xyz)
        pep_by_res = _heavy_atoms_by_res(pdb_text, ch)
        n_contacts = _count_contacts(rec_xyz, pep_by_res)
        result.append((ch, n_res, dist, n_contacts))
    result.sort(key=lambda x: x[2])  # closest to receptor first
    return result


# ─── per-entry screening ───────────────────────────────────────────────────────

def screen_entry(
    row: dict[str, str],
    existing_scores: dict[str, dict],
) -> dict[str, Any]:
    """Screen one calibration entry. Returns a result dict."""
    pdb_id = row["pdb_id"]
    csv_seq = row.get("peptide_sequence", "").strip()
    csv_seq_len = len(csv_seq)
    rec_chain = (row.get("receptor_chain") or "").strip()
    affinity_type = row.get("affinity_type", "").strip()
    try:
        pkd = float(row.get("experimental_pkd", ""))
    except ValueError:
        pkd = float("nan")

    issues: list[str] = []
    flags: list[str] = []

    # baseline result
    result: dict[str, Any] = {
        "pdb_id": pdb_id,
        "rec_chain": rec_chain,
        "csv_seq_len": csv_seq_len,
        "affinity_type": affinity_type,
        "experimental_pkd": pkd,
        "pdb_found": False,
        "n_total_chains": 0,
        "chain_sizes": "",
        "pep_chain_best": "",
        "pep_actual_residues": 0,
        "len_mismatch": 0,
        "min_dist_to_rec_A": float("inf"),
        "n_interface_contacts": 0,
        "all_candidate_chains": "",
        # from scores JSON
        "vina_score": None,
        "ad4_score": None,
        "n_cr_from_scores": None,
        "cr_overflow": False,
        "vina_zero": False,
        "verdict": "SKIP",
        "reasons": "",
    }

    # ── 1. affinity type ──────────────────────────────────────────────────────
    if affinity_type.upper() in ("IC50", "INHIBITION"):
        flags.append(f"IC50 affinity type (not true Kd; may inflate noise)")

    # ── 2. find PDB ───────────────────────────────────────────────────────────
    struct = _find_pdb(pdb_id)
    if struct is None:
        issues.append("PDB file not found in search dirs")
        result["verdict"] = "RED"
        result["reasons"] = "; ".join(issues)
        return result

    result["pdb_found"] = True
    pdb_text = _read_pdb(struct)

    # ── 3. chain inventory ────────────────────────────────────────────────────
    chain_res = _chain_residues(pdb_text)
    n_chains = len(chain_res)
    result["n_total_chains"] = n_chains
    sizes = {ch: len(res) for ch, res in chain_res.items()}
    result["chain_sizes"] = " ".join(f"{ch}:{n}" for ch, n in sorted(sizes.items()))

    if not rec_chain or rec_chain not in sizes:
        issues.append(f"receptor_chain '{rec_chain}' not found in PDB chains: {sorted(sizes)}")
        result["verdict"] = "RED"
        result["reasons"] = "; ".join(issues)
        return result

    if n_chains >= 4:
        flags.append(f"{n_chains} chains — likely symmetric dimer; auto-pep-chain may be wrong")

    # ── 4. receptor atoms ─────────────────────────────────────────────────────
    rec_xyz = _heavy_atoms(pdb_text, rec_chain)

    # ── 5. evaluate all candidate peptide chains ──────────────────────────────
    candidates = _candidate_peptide_chains(pdb_text, rec_chain, rec_xyz)
    # Format for output: "B:25res/d=2.1/cr=10 C:10res/d=15.2/cr=0"
    cand_strs = [f"{ch}:{nr}res/d={d:.1f}/cr={nc}" for ch, nr, d, nc in candidates]
    result["all_candidate_chains"] = " ".join(cand_strs)

    # Best peptide chain: closest to receptor AND within peptide size range
    best_ch = None
    best_dist = float("inf")
    best_n_res = 0
    best_contacts = 0

    for ch, n_res, dist, n_contacts in candidates:
        if MIN_PEP_RESIDUES <= n_res <= MAX_PEP_RESIDUES:
            if dist < best_dist:
                best_dist = dist
                best_ch = ch
                best_n_res = n_res
                best_contacts = n_contacts

    # If nothing in range, take closest regardless
    if best_ch is None and candidates:
        best_ch, best_n_res, best_dist, best_contacts = candidates[0]
        issues.append(
            f"no candidate chain has {MIN_PEP_RESIDUES}–{MAX_PEP_RESIDUES} residues; "
            f"closest is {best_ch} ({best_n_res} residues, d={best_dist:.1f} Å)"
        )

    result["pep_chain_best"] = best_ch or ""
    result["pep_actual_residues"] = best_n_res
    result["min_dist_to_rec_A"] = round(best_dist, 2)
    result["n_interface_contacts"] = best_contacts

    # ── 6. verdict checks ─────────────────────────────────────────────────────

    # 6a. Peptide too large
    if best_n_res > MAX_PEP_RESIDUES:
        issues.append(
            f"best candidate chain {best_ch} has {best_n_res} residues "
            f"(> {MAX_PEP_RESIDUES}; not a peptide)"
        )

    # 6b. Not in contact with receptor
    if best_dist > MAX_CONTACT_DIST_ANG:
        issues.append(
            f"min receptor-peptide distance = {best_dist:.1f} Å "
            f"(> {MAX_CONTACT_DIST_ANG} Å; chains not in contact)"
        )

    # 6c. CSV seq length vs actual PDB chain length
    len_mismatch = abs(csv_seq_len - best_n_res)
    result["len_mismatch"] = len_mismatch
    if len_mismatch > MAX_LEN_MISMATCH:
        flags.append(
            f"length mismatch: CSV seq={csv_seq_len} aa vs PDB chain {best_ch}={best_n_res} res"
        )

    # 6d. Zero contacts despite being close
    if best_dist <= MAX_CONTACT_DIST_ANG and best_contacts == 0:
        flags.append("min_dist < 4 Å but 0 interface contacts detected (unusual)")

    # ── 7. cross-reference existing scores JSON ────────────────────────────────
    existing = existing_scores.get(pdb_id.upper())
    if existing:
        vina = float(existing.get("vina_score", 0))
        ad4 = float(existing.get("ad4_score", 0))
        n_cr = int(existing.get("n_contact_residues", 0))
        result["vina_score"] = round(vina, 3)
        result["ad4_score"] = round(ad4, 3)
        result["n_cr_from_scores"] = n_cr
        result["vina_zero"] = abs(vina) < 0.01

        if abs(vina) < 0.01:
            issues.append("Vina score=0 (crystal pose outside grid box during scoring)")

        if best_n_res > 0 and n_cr > best_n_res + 2:
            result["cr_overflow"] = True
            flags.append(
                f"n_contact_residues from scores ({n_cr}) > actual pep residues ({best_n_res})"
            )

    # ── 8. assemble verdict ───────────────────────────────────────────────────
    all_reasons = []
    if issues:
        result["verdict"] = "RED"
        all_reasons.extend(issues)
    else:
        result["verdict"] = "PASS"

    if flags:
        if result["verdict"] == "PASS":
            result["verdict"] = "FLAG"
        all_reasons.extend(flags)

    result["reasons"] = "; ".join(all_reasons)
    return result


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=str(REPO / "data/training_complexes_full.csv"),
                        help="Training complexes CSV (default: data/training_complexes_full.csv)")
    parser.add_argument("--scores", default=str(REPO / "data/training_scores_full.json"),
                        help="Pre-computed scores JSON (optional — for cross-reference)")
    parser.add_argument("--output", default=str(REPO / "runs/screening_report.csv"),
                        help="Output CSV path (default: runs/screening_report.csv)")
    args = parser.parse_args()

    # Load training CSV
    csv_path = Path(args.csv)
    if not csv_path.exists():
        _log.error("CSV not found: %s", csv_path)
        sys.exit(1)
    rows = list(csv.DictReader(csv_path.open()))
    _log.info("Loaded %d entries from %s", len(rows), csv_path)

    # Load existing scores (optional)
    existing_scores: dict[str, dict] = {}
    scores_path = Path(args.scores)
    if scores_path.exists():
        raw = json.loads(scores_path.read_text())
        existing_scores = {k.upper(): v for k, v in raw.items()}
        _log.info("Loaded %d existing scores from %s", len(existing_scores), scores_path)
    else:
        _log.info("No scores JSON found at %s — skipping cross-reference", scores_path)

    # Run screening
    results = []
    for i, row in enumerate(rows, 1):
        pdb = row["pdb_id"]
        _log.info("[%d/%d] %s", i, len(rows), pdb)
        try:
            result = screen_entry(row, existing_scores)
        except Exception as exc:
            _log.exception("Unhandled error for %s: %s", pdb, exc)
            result = {"pdb_id": pdb, "verdict": "RED", "reasons": f"EXCEPTION: {exc}"}
        results.append(result)

    # Write report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pdb_id", "verdict", "rec_chain", "pep_chain_best",
        "csv_seq_len", "pep_actual_residues", "len_mismatch",
        "n_total_chains", "chain_sizes",
        "min_dist_to_rec_A", "n_interface_contacts",
        "vina_score", "ad4_score", "n_cr_from_scores", "cr_overflow", "vina_zero",
        "affinity_type", "experimental_pkd",
        "all_candidate_chains", "reasons",
    ]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    _log.info("Written %d rows to %s", len(results), out_path)

    # ── summary ───────────────────────────────────────────────────────────────
    from collections import Counter
    verdict_counts = Counter(r.get("verdict", "?") for r in results)
    print("\n" + "="*60)
    print("CALIBRATION DATASET SCREENING REPORT")
    print("="*60)
    print(f"Total entries: {len(results)}")
    print(f"  PASS : {verdict_counts['PASS']:4d}")
    print(f"  FLAG : {verdict_counts['FLAG']:4d}  (inspect manually)")
    print(f"  RED  : {verdict_counts['RED']:4d}  (exclude from calibration)")
    print(f"  SKIP : {verdict_counts['SKIP']:4d}  (error during screening)")

    # Breakdown of RED reasons
    red = [r for r in results if r.get("verdict") == "RED"]
    if red:
        print(f"\n── RED entries ({'─'*45})")
        for r in sorted(red, key=lambda x: x["pdb_id"]):
            print(f"  {r['pdb_id']:6s}  {r.get('reasons','')[:90]}")

    # Breakdown of FLAG reasons
    flagged = [r for r in results if r.get("verdict") == "FLAG"]
    if flagged:
        print(f"\n── FLAG entries ({'─'*44})")
        for r in sorted(flagged, key=lambda x: x["pdb_id"]):
            print(f"  {r['pdb_id']:6s}  {r.get('reasons','')[:90]}")

    # Affinity type distribution
    aff_types = Counter(r.get("affinity_type","?") for r in results)
    print(f"\n── Affinity types ({'─'*43})")
    for t, n in sorted(aff_types.items(), key=lambda x: -x[1]):
        print(f"  {t:10s}: {n}")

    # Size distribution of best peptide chains
    sizes = [r.get("pep_actual_residues", 0) for r in results if r.get("pdb_found")]
    if sizes:
        arr = np.array(sizes)
        print(f"\n── Peptide chain size (actual PDB residues) ({'─'*17})")
        print(f"  mean={arr.mean():.1f}  std={arr.std():.1f}  min={arr.min()}  max={arr.max()}")
        print(f"  > {MAX_PEP_RESIDUES} residues (RED): {(arr > MAX_PEP_RESIDUES).sum()}")

    print(f"\nFull report: {out_path}")
    print("="*60)


if __name__ == "__main__":
    main()
