"""Analyze structural quality of the 284-entry calibration set.

For each PDB entry in data/training_complexes_full.csv, reads the structure
file and computes:

  - n_chains_total       : total number of protein chains in the ASU
  - chain_lengths        : lengths of all chains (e.g. "A:104, B:6")
  - pep_chain_len        : length of the assigned peptide chain
  - rec_chain_len        : length of the assigned receptor chain
  - min_dist_Å           : min heavy-atom distance between receptor and peptide
  - n_contacts           : receptor residues with any atom within 4.5 Å of peptide
  - clash_count          : atom pairs closer than 1.5 Å (should be 0)
  - flag                 : GREEN / YELLOW / RED / MISSING
  - issues               : human-readable list of problems

Flag thresholds:
  GREEN   : min_dist < 4.5 Å, n_contacts ≥ 3, no clashes
  YELLOW  : 4.5 ≤ min_dist < 8 Å  OR  n_contacts < 3
  RED     : min_dist ≥ 8 Å (not in contact) OR clash_count > 0 OR structure missing
  MISSING : structure file not on disk

Output:
  - Terminal table with colour coding
  - datasets/calibration_quality.csv  (all metrics, importable by pandas)
  - Summary counts per flag

Usage:
    python scripts/analyze_calibration_structures.py
    python scripts/analyze_calibration_structures.py --save-red-list bad_entries.txt
    python scripts/analyze_calibration_structures.py --min-contacts 5 --max-dist 5.0
"""
from __future__ import annotations

import argparse
import gzip
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"

# Thresholds
CLASH_DIST   = 1.5   # Å — any pair closer than this is a clash
CONTACT_DIST = 4.5   # Å — receptor residue counted as contacting if any atom this close
GREEN_DIST   = 4.5   # Å — min_dist below this → good contact
YELLOW_DIST  = 8.0   # Å — min_dist above this → not in contact (RED)
MIN_CONTACTS = 3     # n_contacts below this → YELLOW

# ANSI colour codes
_RED    = "\033[91m"
_YEL    = "\033[93m"
_GRN    = "\033[92m"
_DIM    = "\033[2m"
_RST    = "\033[0m"
_BOLD   = "\033[1m"


def _find_structure(pdb_id: str) -> Path | None:
    search_dirs = [
        REPO / "datasets" / ds
        for ds in [
            "raw_pdbs", "pdb_2024_2026/structures", "ppii_enriched/structures",
            "pdb_2019_2023/structures", "pdb_2010_2018/structures", "pdb_pre2010/structures",
            "family_targeted/structures", "ppii_extended/structures",
            "training_expanded_structures",
        ]
    ]
    uid = pdb_id.upper()
    for d in search_dirs:
        if not d.exists():
            continue
        for pat in [f"{uid}.pdb.gz", f"{uid}.pdb", f"{uid.lower()}.pdb"]:
            p = d / pat
            if p.exists() and p.stat().st_size > 500:
                return p
    return None


def _read_pdb(path: Path) -> str:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return f.read().decode("latin-1")
    return path.read_text("latin-1")


AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "HSD", "HSE", "HSP", "HIE", "HID", "HIP", "CYX", "CYM",
    "TPO", "SEP", "PTR", "MLY",
}


def _parse_chains(pdb_text: str) -> dict[str, list[tuple[int, float, float, float]]]:
    """Return chain → list of (resnum, x, y, z) for all heavy atoms of standard residues.

    Only MODEL 1 is parsed for NMR / multi-model structures.  When a second
    MODEL record is encountered all further ATOM/HETATM lines are ignored.
    """
    chains: dict[str, list] = {}
    in_model: bool = False   # True once we've entered MODEL 1
    skip_rest: bool = False  # True once ENDMDL for MODEL 1 is seen

    for line in pdb_text.splitlines():
        tag = line[:6].rstrip()

        if tag == "MODEL":
            if not in_model:
                in_model = True   # start of MODEL 1 (or first MODEL block)
            else:
                skip_rest = True  # second MODEL → stop
            continue

        if tag == "ENDMDL":
            if in_model:
                skip_rest = True  # end of MODEL 1 → stop
            continue

        if skip_rest:
            continue

        if tag not in ("ATOM", "HETATM"):
            continue
        if len(line) < 54:
            continue
        atom_name = line[12:16].strip()
        if atom_name.startswith("H"):  # skip hydrogens
            continue
        # ALTLOC indicator at column 16: only accept blank or 'A'
        altloc = line[16] if len(line) > 16 else " "
        if altloc not in (" ", "A"):
            continue
        resname = line[17:20].strip()
        if resname not in AA3:
            continue
        chain = line[21]
        try:
            resnum = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except (ValueError, IndexError):
            continue
        if chain not in chains:
            chains[chain] = []
        chains[chain].append((resnum, x, y, z))
    return chains


def _chain_residue_count(chain_atoms: list) -> int:
    return len({r for r, *_ in chain_atoms})


def _min_distance(chain_a: list, chain_b: list) -> float:
    """Minimum heavy-atom distance between two chains."""
    if not chain_a or not chain_b:
        return float("inf")
    arr_a = np.array([[x, y, z] for _, x, y, z in chain_a])
    arr_b = np.array([[x, y, z] for _, x, y, z in chain_b])
    # Compute all pairwise distances; use broadcasting
    # For large chains use batched approach
    batch = 500
    min_d = float("inf")
    for i in range(0, len(arr_a), batch):
        chunk = arr_a[i : i + batch]  # (batch, 3)
        # (batch, N, 3) - (1, N, 3)
        diff = chunk[:, None, :] - arr_b[None, :, :]
        d = np.sqrt((diff ** 2).sum(axis=2))  # (batch, N)
        min_d = min(min_d, d.min())
    return float(min_d)


def _clash_count(chain_a: list, chain_b: list, threshold: float = CLASH_DIST) -> int:
    """Count atom pairs closer than threshold (cross-chain only)."""
    if not chain_a or not chain_b:
        return 0
    arr_a = np.array([[x, y, z] for _, x, y, z in chain_a])
    arr_b = np.array([[x, y, z] for _, x, y, z in chain_b])
    count = 0
    batch = 200
    for i in range(0, len(arr_a), batch):
        chunk = arr_a[i : i + batch]
        diff = chunk[:, None, :] - arr_b[None, :, :]
        d = np.sqrt((diff ** 2).sum(axis=2))
        count += int((d < threshold).sum())
    return count


def _n_contacts(rec_atoms: list, pep_atoms: list, cutoff: float = CONTACT_DIST) -> int:
    """Number of receptor residues with ≥1 heavy atom within cutoff of any peptide atom."""
    if not rec_atoms or not pep_atoms:
        return 0
    pep_arr = np.array([[x, y, z] for _, x, y, z in pep_atoms])
    contacts: set[int] = set()
    batch = 200
    rec_arr = np.array([[x, y, z] for _, x, y, z in rec_atoms])
    rec_resnums = np.array([r for r, *_ in rec_atoms])
    for i in range(0, len(pep_arr), batch):
        chunk = pep_arr[i : i + batch]  # (batch, 3)
        diff = rec_arr[None, :, :] - chunk[:, None, :]  # (batch, N_rec, 3)
        d = np.sqrt((diff ** 2).sum(axis=2))  # (batch, N_rec)
        mask = (d < cutoff).any(axis=0)  # (N_rec,)
        contacts.update(rec_resnums[mask].tolist())
    return len(contacts)


def analyze_entry(pdb_id: str, rec_chain: str, pep_chain_given: str | None) -> dict:
    """Analyze one calibration entry. Returns a dict of metrics."""
    result: dict = {
        "pdb_id": pdb_id,
        "rec_chain_given": rec_chain,
        "pep_chain_given": pep_chain_given or "",
        "struct_path": "",
        "n_chains_total": 0,
        "chain_summary": "",
        "rec_chain_len": 0,
        "pep_chain_len": 0,
        "pep_chain_found": "",
        "min_dist_A": float("inf"),
        "n_contacts": 0,
        "clash_count": 0,
        "flag": "MISSING",
        "issues": [],
    }

    struct_path = _find_structure(pdb_id)
    if not struct_path:
        result["issues"].append("structure not on disk")
        return result

    result["struct_path"] = str(struct_path.relative_to(REPO))

    try:
        pdb_text = _read_pdb(struct_path)
    except Exception as exc:
        result["flag"] = "RED"
        result["issues"].append(f"read error: {exc}")
        return result

    chains = _parse_chains(pdb_text)
    if not chains:
        result["flag"] = "RED"
        result["issues"].append("no protein atoms parsed")
        return result

    result["n_chains_total"] = len(chains)

    # Build chain summary: sort by length descending
    chain_lens = sorted(
        [(ch, _chain_residue_count(atoms)) for ch, atoms in chains.items()],
        key=lambda x: -x[1],
    )
    result["chain_summary"] = "  ".join(f"{ch}:{n}" for ch, n in chain_lens)

    # Receptor chain
    if rec_chain not in chains:
        result["flag"] = "RED"
        result["issues"].append(f"receptor chain '{rec_chain}' not in structure")
        return result

    rec_atoms = chains[rec_chain]
    result["rec_chain_len"] = _chain_residue_count(rec_atoms)

    # Determine which chain is the peptide.
    # Priority:
    #   1. Explicitly given pep_chain (from CSV column)
    #   2. Among candidates in the 5–30 aa range, pick the one CLOSEST to the
    #      receptor geometrically (solves symmetric-dimer chain mixup).
    #   3. Fallback: any non-receptor chain, closest to receptor.
    if pep_chain_given and pep_chain_given in chains and pep_chain_given != rec_chain:
        pep_chain = pep_chain_given
    else:
        candidates = [
            ch for ch, atoms in chains.items()
            if ch != rec_chain and 5 <= _chain_residue_count(atoms) <= 30
        ]
        if not candidates:
            # Fallback: any chain ≠ rec_chain (even if length off)
            candidates = [ch for ch in chains if ch != rec_chain]
        if not candidates:
            result["flag"] = "RED"
            result["issues"].append("no peptide chain candidate found")
            return result
        if len(candidates) == 1:
            pep_chain = candidates[0]
        else:
            # For symmetric dimers (multiple peptide candidates) pick the one
            # geometrically CLOSEST to the receptor to avoid cross-dimer mixup.
            best_ch, best_d = None, float("inf")
            for ch in candidates:
                d = _min_distance(rec_atoms, chains[ch])
                if d < best_d:
                    best_d = d
                    best_ch = ch
            pep_chain = best_ch

    pep_atoms = chains[pep_chain]
    result["pep_chain_found"] = pep_chain
    result["pep_chain_len"] = _chain_residue_count(pep_atoms)

    if pep_chain == rec_chain:
        result["flag"] = "RED"
        result["issues"].append("peptide and receptor are the same chain")
        return result

    # Distance metrics
    min_d = _min_distance(rec_atoms, pep_atoms)
    n_cont = _n_contacts(rec_atoms, pep_atoms)
    clashes = _clash_count(rec_atoms, pep_atoms)

    result["min_dist_A"] = round(min_d, 2)
    result["n_contacts"] = n_cont
    result["clash_count"] = clashes

    # Flag assignment
    issues = []
    if clashes > 0:
        issues.append(f"{clashes} atom clashes (<{CLASH_DIST}Å)")
    if min_d >= YELLOW_DIST:
        issues.append(f"not in contact (min dist {min_d:.1f}Å ≥ {YELLOW_DIST}Å)")
    elif min_d >= GREEN_DIST:
        issues.append(f"marginal contact (min dist {min_d:.1f}Å)")
    if n_cont < MIN_CONTACTS:
        issues.append(f"few contacts ({n_cont} < {MIN_CONTACTS})")

    # Structural context warnings
    n_pep_candidates = sum(
        1 for ch, atoms in chains.items()
        if ch != rec_chain and 5 <= _chain_residue_count(atoms) <= 30
    )
    n_rec_candidates = sum(
        1 for ch, atoms in chains.items()
        if _chain_residue_count(atoms) >= 50
    )
    if n_rec_candidates > 2:
        issues.append(f"possible oligomer ({n_rec_candidates} receptor-length chains)")
    if n_pep_candidates > 2:
        issues.append(f"multiple peptide candidates ({n_pep_candidates})")
    if result["pep_chain_len"] < 5:
        issues.append(f"very short peptide ({result['pep_chain_len']} aa)")
    if result["rec_chain_len"] < 30:
        issues.append(f"short receptor ({result['rec_chain_len']} aa) — may be peptide")

    result["issues"] = issues

    if clashes > 0 or min_d >= YELLOW_DIST:
        result["flag"] = "RED"
    elif min_d >= GREEN_DIST or n_cont < MIN_CONTACTS:
        result["flag"] = "YELLOW"
    else:
        result["flag"] = "GREEN"

    return result


def _flag_colour(flag: str) -> str:
    return {
        "GREEN": _GRN,
        "YELLOW": _YEL,
        "RED": _RED,
        "MISSING": _RED + _BOLD,
    }.get(flag, "")


def main() -> None:
    global MIN_CONTACTS, GREEN_DIST
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--training-csv", type=Path, default=DATA_DIR / "training_complexes_full.csv")
    parser.add_argument("--output-csv", type=Path, default=REPO / "datasets" / "calibration_quality.csv")
    parser.add_argument("--save-red-list", type=Path, default=None,
                        help="Write RED/MISSING pdb_ids to this file (one per line)")
    parser.add_argument("--min-contacts", type=int, default=MIN_CONTACTS,
                        help=f"Contacts threshold for GREEN (default: {MIN_CONTACTS})")
    parser.add_argument("--max-dist", type=float, default=GREEN_DIST,
                        help=f"Max min-dist for GREEN in Å (default: {GREEN_DIST})")
    parser.add_argument("--show-green", action="store_true",
                        help="Also print GREEN entries (default: only YELLOW/RED)")
    parser.add_argument("--pep-chain-col", type=str, default=None,
                        help="Column in training CSV with peptide chain IDs (optional)")
    args = parser.parse_args()

    # Override module-level thresholds from CLI args
    MIN_CONTACTS = args.min_contacts
    GREEN_DIST = args.max_dist

    df = pd.read_csv(args.training_csv)
    _log.info("Analyzing %d calibration entries from %s", len(df), args.training_csv)

    has_pep_chain_col = args.pep_chain_col and args.pep_chain_col in df.columns

    results = []
    for i, row in df.iterrows():
        pdb_id = str(row["pdb_id"]).upper()
        rec_chain = str(row.get("receptor_chain", "") or "")
        pep_chain_hint = str(row[args.pep_chain_col]) if has_pep_chain_col else None

        if not rec_chain or rec_chain in ("nan", ""):
            rec_chain = ""

        res = analyze_entry(pdb_id, rec_chain, pep_chain_hint)
        res["experimental_pkd"] = float(row.get("experimental_pkd", float("nan")))
        res["affinity_type"] = str(row.get("affinity_type", ""))
        res["source"] = str(row.get("source", ""))
        results.append(res)

        if (i + 1) % 50 == 0:
            _log.info("  Processed %d / %d entries...", i + 1, len(df))

    # ---------------------------------------------------------------
    # Terminal output
    # ---------------------------------------------------------------
    by_flag = {"GREEN": [], "YELLOW": [], "RED": [], "MISSING": []}
    for r in results:
        by_flag.get(r["flag"], by_flag["RED"]).append(r)

    tty = sys.stdout.isatty()

    def c(text: str, flag: str) -> str:
        return (_flag_colour(flag) + text + _RST) if tty else text

    print(f"\n{'=' * 100}")
    print(f"  CALIBRATION STRUCTURE QUALITY REPORT — {args.training_csv.name}")
    print(f"{'=' * 100}")
    print(f"  {'PDB':6}  {'Flag':7}  {'Chains':3}  {'ChainMap':35}  {'RecLen':6}  {'PepLen':6}  {'MinDist':8}  {'Conts':5}  {'Clash':5}  Issues")
    print(f"  {'-'*6}  {'-'*7}  {'-'*3}  {'-'*35}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*5}  {'-'*5}  {'-'*40}")

    def _row(r: dict, show: bool) -> None:
        if not show:
            return
        flag = r["flag"]
        min_d = f"{r['min_dist_A']:.2f}" if r["min_dist_A"] != float("inf") else " ∞"
        issues_str = " | ".join(r["issues"][:3])
        chain_trunc = r["chain_summary"][:35]
        pep_found = r["pep_chain_found"]
        rec_given = r["rec_chain_given"]
        pep_marker = f"→pep:{pep_found}" if pep_found else "no-pep"
        line = (
            f"  {r['pdb_id']:6}  {flag:7}  {r['n_chains_total']:3d}  {chain_trunc:35}  "
            f"{r['rec_chain_len']:6d}  {r['pep_chain_len']:6d}  {min_d:>8}  "
            f"{r['n_contacts']:5d}  {r['clash_count']:5d}  {issues_str}"
        )
        print(c(line, flag))

    # Print RED first (problems to fix)
    if by_flag["RED"] or by_flag["MISSING"]:
        print(f"\n  {c('--- RED / MISSING ---', 'RED')}")
        for r in sorted(by_flag["RED"] + by_flag["MISSING"], key=lambda x: x["pdb_id"]):
            _row(r, True)

    if by_flag["YELLOW"]:
        print(f"\n  {c('--- YELLOW (marginal) ---', 'YELLOW')}")
        for r in sorted(by_flag["YELLOW"], key=lambda x: x["min_dist_A"]):
            _row(r, True)

    if args.show_green:
        print(f"\n  {c('--- GREEN (ok) ---', 'GREEN')}")
        for r in sorted(by_flag["GREEN"], key=lambda x: x["pdb_id"]):
            _row(r, True)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print(f"  SUMMARY")
    print(f"{'=' * 100}")
    n_total = len(results)
    for flag in ["GREEN", "YELLOW", "RED", "MISSING"]:
        n = len(by_flag[flag])
        pct = 100 * n / n_total if n_total else 0
        bar = "█" * int(pct / 2)
        print(c(f"  {flag:8}: {n:4d} / {n_total} ({pct:5.1f}%)  {bar}", flag))

    # Min-dist distribution for non-missing
    valid = [r for r in results if r["min_dist_A"] != float("inf")]
    if valid:
        dists = [r["min_dist_A"] for r in valid]
        print(f"\n  Min-dist distribution (n={len(dists)}):")
        bins = [0, 2, 3, 4, 5, 6, 8, 12, 20, 999]
        labels = ["0-2Å", "2-3Å", "3-4Å", "4-5Å", "5-6Å", "6-8Å", "8-12Å", "12-20Å", ">20Å"]
        for lo, hi, lab in zip(bins, bins[1:], labels):
            n = sum(1 for d in dists if lo <= d < hi)
            bar = "█" * n + "░" * max(0, 20 - n)
            flag = "GREEN" if hi <= GREEN_DIST else ("YELLOW" if hi <= YELLOW_DIST else "RED")
            print(c(f"    {lab:8}: {n:4d}  {bar}", flag))

    print(f"\n  Contacts distribution (n={len(valid)}):")
    cont_bins = [0, 1, 2, 3, 5, 8, 12, 20, 9999]
    cont_labels = ["0", "1", "2", "3-4", "5-7", "8-11", "12-19", "≥20"]
    for lo, hi, lab in zip(cont_bins, cont_bins[1:], cont_labels):
        n = sum(1 for r in valid if lo <= r["n_contacts"] < hi)
        bar = "█" * n + "░" * max(0, 20 - n)
        flag = "RED" if hi <= MIN_CONTACTS else ("YELLOW" if hi <= MIN_CONTACTS + 2 else "GREEN")
        print(c(f"    {lab:6}: {n:4d}  {bar}", flag))

    # ---------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------
    out_df = pd.DataFrame(results)
    out_df["issues"] = out_df["issues"].apply(lambda x: " | ".join(x) if x else "")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    _log.info("Saved quality report → %s", args.output_csv)

    # Save RED list
    if args.save_red_list:
        bad = [r["pdb_id"] for r in results if r["flag"] in ("RED", "MISSING")]
        args.save_red_list.write_text("\n".join(bad) + "\n")
        _log.info("Saved %d RED/MISSING IDs → %s", len(bad), args.save_red_list)

    print(f"\n  Full report saved → {args.output_csv}")
    if by_flag["RED"] or by_flag["MISSING"]:
        print(f"\n  {c('Action:', 'RED')} Review RED entries before using for calibration.")
        print(f"  Use --save-red-list bad.txt to get a list for exclusion.")

    return results


if __name__ == "__main__":
    main()
