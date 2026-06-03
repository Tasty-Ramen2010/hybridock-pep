"""Audit the held-out eval — were peptide chains correctly identified, do
crystal Vina scores look sane, and does the entropy actually come from the
bound peptide pose?
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw_pdbs"

AA3to1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q",
          "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
          "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

def split_chains(pdb_path):
    chains = {}
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        chains.setdefault(line[21], []).append(line)
    return chains

def chain_sequence(lines):
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

def heavy_xyz(lines):
    out = []
    for line in lines:
        if line[12:16].strip().startswith("H"):
            continue
        try:
            out.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
    return np.array(out) if out else np.zeros((0, 3))

def main():
    eval_data = json.loads((ROOT / "data" / "eval_holdout_calibrations.json").read_text())
    csv_meta = {r["pdb_id"].lower(): r for r in csv.DictReader(
        (ROOT / "data" / "training_complexes_full.csv").open()) if r.get("experimental_pkd")}

    issues = {
        "peptide_chain_too_long": [],  # found chain >>> peptide seq
        "wrong_chain_picked": [],
        "no_close_contact": [],  # peptide and receptor not within 5 Å
        "vina_zero": [],
        "tiny_peptide_chain": [],
    }
    audited = []

    for row in eval_data:
        pdb = row["pdb"]
        seq = csv_meta[pdb]["peptide_sequence"].upper()
        pdb_file = RAW / f"{pdb.upper()}.pdb"
        chains = split_chains(pdb_file)

        # Check what chains contain the sequence
        seq_matches = {cid: chain_sequence(lines) for cid, lines in chains.items()}
        containing = {cid: s for cid, s in seq_matches.items() if seq in s}

        # Our heuristic picks the shortest containing chain
        if not containing:
            continue
        picked = min(containing, key=lambda c: len(seq_matches[c]))
        picked_len = len(seq_matches[picked])

        # Quality checks
        chain_too_long = picked_len > len(seq) + 5  # picked a chain with peptide embedded
        multi_match = len(containing) > 1

        # Distance: peptide chain heavy atoms vs all other chains
        pep_xyz = heavy_xyz(chains[picked])
        rec_xyz = np.concatenate([heavy_xyz(ls) for c, ls in chains.items() if c != picked])
        if pep_xyz.size == 0 or rec_xyz.size == 0:
            continue
        # Min distance
        d = np.sqrt(((pep_xyz[:, None] - rec_xyz[None]) ** 2).sum(-1))
        min_dist = float(d.min())
        n_contact_at_5A = int((d.min(axis=1) < 5.0).sum())

        if row["vina"] == 0.0:
            issues["vina_zero"].append(pdb)
        if min_dist > 5.0:
            issues["no_close_contact"].append((pdb, round(min_dist, 2)))
        if chain_too_long:
            issues["peptide_chain_too_long"].append((pdb, picked, picked_len, len(seq)))
        if picked_len < len(seq):
            issues["tiny_peptide_chain"].append((pdb, picked, picked_len, len(seq)))

        audited.append({
            "pdb": pdb,
            "seq_len": len(seq),
            "picked_chain": picked,
            "picked_chain_len": picked_len,
            "min_dist_to_receptor": round(min_dist, 2),
            "n_within_5A": n_contact_at_5A,
            "n_contact_reported": row["n_contact"],
            "vina": row["vina"],
            "n_chains_containing_seq": len(containing),
            "containing_chains": list(containing.keys()),
        })

    print(f"Audited {len(audited)} of {len(eval_data)}\n")
    for k, v in issues.items():
        print(f"  {k}: {len(v)}")
        for x in v[:5]:
            print(f"    {x}")
        if len(v) > 5:
            print(f"    ... +{len(v) - 5} more")

    # Sanity: distribution of (n_within_5A - n_contact_reported)
    diffs = [a["n_within_5A"] - a["n_contact_reported"] for a in audited]
    print(f"\nn_contact agreement: median diff = {np.median(diffs):.1f}, "
          f"|diff|>3 in {sum(1 for d in diffs if abs(d) > 3)}/{len(diffs)} cases")

    # Picked-chain == peptide chain ratio
    exact = sum(1 for a in audited if a["picked_chain_len"] == a["seq_len"])
    embedded = sum(1 for a in audited if a["picked_chain_len"] > a["seq_len"] + 2)
    print(f"\nChain pick quality:")
    print(f"  exact length match (seq is whole chain): {exact}/{len(audited)}")
    print(f"  peptide embedded in longer chain:        {embedded}/{len(audited)}")

    # Dump first 10 audit rows
    print("\nSample rows:")
    for a in audited[:10]:
        print(f"  {a['pdb']}: chain={a['picked_chain']} len={a['picked_chain_len']} "
              f"(seq={a['seq_len']}) min_d={a['min_dist_to_receptor']}Å "
              f"contacts={a['n_within_5A']} (reported={a['n_contact_reported']}) "
              f"vina={a['vina']}")

    # Write full audit
    (ROOT / "data" / "audit_holdout_eval.json").write_text(json.dumps(audited, indent=2))
    print(f"\nFull audit -> data/audit_holdout_eval.json")

if __name__ == "__main__":
    main()
