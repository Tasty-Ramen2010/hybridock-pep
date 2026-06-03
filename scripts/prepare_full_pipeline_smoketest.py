"""Prepare run config for the 18-complex full-pipeline smoke test.

For each of the 10 entries in data/test_complexes.csv plus 8 cluster
representatives from data/calibration_per_family.json:

  1. Read the raw PDB from datasets/raw_pdbs/{ID}.pdb.
  2. Identify the peptide chain (matches the known peptide sequence).
  3. Write a receptor-only PDB (no HETATM, peptide chain removed) to
     runs/smoketest_jun02/inputs/{id}_receptor.pdb.
  4. Compute site coords (peptide Cα centroid) and box size (peptide extent
     + 10 Å margin, min 20 Å).
  5. Write a single run config YAML/CSV to
     runs/smoketest_jun02/run_plan.csv that the launcher will iterate.

Does NOT launch RAPiDock. Just builds the plan for review.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw_pdbs"
OUT = ROOT / "runs" / "smoketest_jun02"
INPUTS = OUT / "inputs"

AA3to1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q",
          "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
          "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}


def split_chains(pdb_path: Path) -> dict[str, list[str]]:
    chains: dict[str, list[str]] = {}
    for line in pdb_path.read_text().splitlines():
        if line.startswith("ATOM"):
            chains.setdefault(line[21], []).append(line)
    return chains


def chain_sequence(lines: list[str]) -> str:
    seen, last = [], None
    for line in lines:
        try:
            key = (int(line[22:26].strip()), line[26])
        except ValueError:
            continue
        if key == last:
            continue
        last = key
        seen.append(AA3to1.get(line[17:20].strip(), "X"))
    return "".join(seen)


def find_peptide_chain(chains: dict[str, list[str]], seq: str,
                       receptor_chain: str | None) -> str | None:
    """Find the chain matching the peptide.

    Tier 1: chain length ≤ len(seq)+5 AND contains seq exactly.
    Tier 2: chain length ≤ len(seq)+5 AND shares a 5+-mer with seq (handles
            crystal chains truncated relative to the CSV sequence — common
            when N/C-terminal residues are disordered).
    """
    seq = seq.upper()
    rc = (receptor_chain or "").strip().upper()
    max_len = len(seq) + 5

    # Tier 1: exact substring
    best, best_len = None, None
    for cid, lines in chains.items():
        if rc and cid.upper() == rc:
            continue
        s = chain_sequence(lines)
        if seq not in s or len(s) > max_len:
            continue
        if best is None or len(s) < best_len:
            best, best_len = cid, len(s)
    if best is not None:
        return best

    # Tier 2: fuzzy — shares a 5+-mer with target
    if len(seq) < 5:
        return None
    kmers = {seq[i:i+5] for i in range(len(seq) - 4)}
    for cid, lines in chains.items():
        if rc and cid.upper() == rc:
            continue
        s = chain_sequence(lines)
        if len(s) > max_len or len(s) < 5:
            continue
        if any(s[i:i+5] in kmers for i in range(len(s) - 4)):
            if best is None or len(s) < best_len:
                best, best_len = cid, len(s)
    return best


def ca_coords(lines: list[str]) -> np.ndarray:
    out = []
    for l in lines:
        if l[12:16].strip() == "CA":
            try:
                out.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
            except ValueError:
                continue
    return np.array(out) if out else np.zeros((0, 3))


def heavy_coords(lines: list[str]) -> np.ndarray:
    out = []
    for l in lines:
        if l[12:16].strip().startswith("H"):
            continue
        try:
            out.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
        except ValueError:
            continue
    return np.array(out) if out else np.zeros((0, 3))


def write_receptor(chains: dict[str, list[str]], pep_chain: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = ["REMARK   HybriDock-Pep smoke test: peptide chain removed, HETATM stripped"]
    serial = 1
    for cid, lines in chains.items():
        if cid == pep_chain:
            continue
        for l in lines:
            l = l[:6] + f"{serial:>5d}" + l[11:]
            body.append(l)
            serial += 1
        body.append(f"TER   {serial:>5d}      {chains[cid][-1][17:26]}")
        serial += 1
    body.append("END")
    out_path.write_text("\n".join(body) + "\n")


def write_crystal_peptide(lines: list[str], out_path: Path) -> None:
    """Write just the peptide chain ATOM lines to a separate PDB for RMSD ref."""
    out_path.write_text("\n".join(lines) + "\nEND\n")


def main() -> None:
    INPUTS.mkdir(parents=True, exist_ok=True)

    # 10 test complexes; 1G73 dropped — peptide is fused N-terminus of a
    # 157-residue chain (intramolecular interaction, not free-peptide docking)
    EXCLUDE = {"1G73"}
    with (ROOT / "data" / "test_complexes.csv").open() as f:
        test10 = [r for r in csv.DictReader(f) if r["pdb_id"] not in EXCLUDE]
    # 8 cluster reps + peptide sequences from training_complexes_full
    pf = json.loads((ROOT / "data" / "calibration_per_family.json").read_text())
    with (ROOT / "data" / "training_complexes_full.csv").open() as f:
        meta = {r["pdb_id"].lower(): r for r in csv.DictReader(f)}
    reps = []
    for c, fit in pf["families"].items():
        rep_pdb = fit["pdbs"][0]
        r = meta.get(rep_pdb.lower())
        if r:
            reps.append({
                "pdb_id": rep_pdb.upper(),
                "peptide_sequence": r["peptide_sequence"],
                "experimental_pkd": r["experimental_pkd"],
                "receptor_chain": r.get("receptor_chain", ""),
                "cluster": c,
            })

    rows = []
    for set_name, src in [("test10", test10), ("cluster_reps", reps)]:
        for r in src:
            pdb_id = r["pdb_id"].upper()
            seq = r["peptide_sequence"]
            raw_path = RAW / f"{pdb_id}.pdb"
            chains = split_chains(raw_path)
            pep_chain = find_peptide_chain(
                chains, seq, r.get("receptor_chain"))
            if pep_chain is None:
                # try receptor_chain hint inverted (sometimes peptide is on a
                # short chain not flagged in CSV)
                # fall back: shortest chain containing seq
                cands = [(c, chain_sequence(ls)) for c, ls in chains.items()
                         if seq in chain_sequence(ls)]
                if cands:
                    pep_chain = min(cands, key=lambda x: len(x[1]))[0]
            if pep_chain is None:
                print(f"  [SKIP] {pdb_id}: peptide chain not found")
                continue

            pep_ca = ca_coords(chains[pep_chain])
            if pep_ca.size == 0:
                print(f"  [SKIP] {pdb_id}: no Cα in peptide chain")
                continue
            site = pep_ca.mean(axis=0)
            pep_heavy = heavy_coords(chains[pep_chain])
            extent = float(np.linalg.norm(pep_heavy.max(0) - pep_heavy.min(0)))
            box = max(20.0, extent + 10.0)

            recep_path = INPUTS / f"{pdb_id}_receptor.pdb"
            crystal_pep_path = INPUTS / f"{pdb_id}_peptide_crystal.pdb"
            write_receptor(chains, pep_chain, recep_path)
            write_crystal_peptide(chains[pep_chain], crystal_pep_path)

            rows.append({
                "set": set_name,
                "pdb_id": pdb_id,
                "peptide": seq,
                "pkd": r["experimental_pkd"],
                "receptor_pdb": str(recep_path.relative_to(ROOT)),
                "crystal_peptide_pdb": str(crystal_pep_path.relative_to(ROOT)),
                "site_x": round(float(site[0]), 3),
                "site_y": round(float(site[1]), 3),
                "site_z": round(float(site[2]), 3),
                "box": round(box, 1),
                "n_pep_residues": len(seq),
                "n_receptor_atoms": sum(len(ls) for c, ls in chains.items() if c != pep_chain),
                "cluster": r.get("cluster", ""),
            })

    plan_path = OUT / "run_plan.csv"
    fieldnames = list(rows[0].keys())
    with plan_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {plan_path}  ({len(rows)} rows)")
    print(f"Receptor PDBs + crystal peptides in {INPUTS}")
    print("\nPer-complex summary:")
    print(f"  {'SET':<14} {'PDB':<6} {'PEP':<22} {'pKd':<6} {'BOX':<6} SITE")
    for r in rows:
        print(f"  {r['set']:<14} {r['pdb_id']:<6} {r['peptide'][:20]:<22} "
              f"{r['pkd']:<6} {r['box']:<6} "
              f"({r['site_x']}, {r['site_y']}, {r['site_z']})")


if __name__ == "__main__":
    main()
