#!/usr/bin/env python
"""Fetch the SKEMPI v2 structures needed for structure-based ddG.

WHY WE NEED STRUCTURES AT ALL.  The shipped ddG head (scoring/mutation_ddg.py) is deliberately
structure-FREE: it sees the two residue identities and a five-category interface-location label,
never the mutant structure.  That is what makes it instant, and it reaches r = 0.385 leave-
complex-out.  The June assessment named the next lever explicitly and it was never executed:
"CRUDE proxy (5 categories); full structure ddG (dock WT-vs-mut, OUR pipeline) is the lever to
beat 0.38."

That lever is now cheap, because the selectivity work already built the machinery: sel_features.py
mutates a residue in place, repacks the interface with a soft-repulsive pre-pass, minimises, and
reports per-term ref2015 interface energies plus typed contact chemistry.  Running it on the
wild-type and the mutant and taking the DIFFERENCE gives a structure-based ddG feature vector.

This also answers the congeneric blindness head-on.  The absolute scorer moves only 14% of a
feature sd across a one-residue change, and tracks near-twin differences at r = 0.182 while moving
only 35% as far as reality.  But those were peptide-level descriptors.  Per-term INTERFACE energies
after repacking the mutant are not blind to a single substitution by construction -- fa_elec and the
hbond terms change directly when a charge or a donor is swapped.  Whether that converts into skill
is the measurement.

Only complexes whose smaller partner chain is short are kept by default: SKEMPI is mostly
protein-protein, and our scope is peptides.

Usage: sel_skempi_fetch.py [--max-partner-len 40] [--workers 8]
Output: datasets/skempi/<PDB>.pdb, data/skempi_single_peptide.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
SRC = ROOT / "data/skempi_v2.csv"
DEST = ROOT / "datasets/skempi"
OUT = ROOT / "data/skempi_single_peptide.csv"
URL = "https://files.rcsb.org/download/{}.pdb"
AA3to1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
          "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
          "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M"}


def fetch(pdb: str) -> tuple[str, bool]:
    out = DEST / f"{pdb}.pdb"
    if out.exists() and out.stat().st_size > 1000:
        return pdb, True
    for attempt in range(3):
        try:
            with urllib.request.urlopen(URL.format(pdb), timeout=60) as r:
                data = r.read()
            if len(data) > 1000:
                out.write_bytes(data)
                return pdb, True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(1.5 * (attempt + 1))
    return pdb, False


def chain_lengths(pdb_file: Path) -> dict[str, int]:
    seen: dict[str, set] = {}
    for line in pdb_file.read_text(errors="ignore").splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA" \
                and line[17:20].strip() in AA3to1:
            seen.setdefault(line[21], set()).add(line[22:27])
    return {c: len(v) for c, v in seen.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-partner-len", type=int, default=40,
                    help="keep a complex only if its smaller partner chain is at most this long")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)

    rows = [r for r in csv.DictReader(open(SRC), delimiter=";")
            if "," not in r["Mutation(s)_PDB"]]
    pdbs = sorted({r["#Pdb"].split("_")[0] for r in rows})
    print(f"{len(rows)} single mutations over {len(pdbs)} complexes", flush=True)

    ok: set[str] = set()
    with ThreadPoolExecutor(args.workers) as ex:
        for k, (pdb, good) in enumerate(ex.map(fetch, pdbs), 1):
            if good:
                ok.add(pdb)
            if k % 40 == 0:
                print(f"  fetched {k}/{len(pdbs)} ({len(ok)} ok)", flush=True)
    print(f"downloaded {len(ok)}/{len(pdbs)} structures", flush=True)

    kept, skipped = [], {"nofile": 0, "chains": 0, "toolong": 0}
    for r in rows:
        pdb, c1, c2 = (r["#Pdb"].split("_") + ["", ""])[:3]
        if pdb not in ok:
            skipped["nofile"] += 1
            continue
        L = chain_lengths(DEST / f"{pdb}.pdb")
        g1 = [c for c in c1 if c in L]
        g2 = [c for c in c2 if c in L]
        if not g1 or not g2:
            skipped["chains"] += 1
            continue
        n1 = sum(L[c] for c in g1)
        n2 = sum(L[c] for c in g2)
        small, big = (g1, g2) if n1 <= n2 else (g2, g1)
        if min(n1, n2) > args.max_partner_len:
            skipped["toolong"] += 1
            continue
        mut = r["Mutation(s)_PDB"]
        kept.append({
            "pdb": pdb, "mutation": mut, "wt_aa": mut[0], "mut_chain": mut[1],
            "resnum": mut[2:-1], "mut_aa": mut[-1],
            "location": r["iMutation_Location(s)"],
            "small_chains": "".join(small), "big_chains": "".join(big),
            "small_len": min(n1, n2), "big_len": max(n1, n2),
            "affinity_mut": r["Affinity_mut_parsed"], "affinity_wt": r["Affinity_wt_parsed"],
            "temperature": r.get("Temperature", ""),
        })
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(kept[0].keys()))
        w.writeheader(); w.writerows(kept)
    print(f"\nkept {len(kept)} single mutations on complexes with a partner chain "
          f"<= {args.max_partner_len} aa, over "
          f"{len({k['pdb'] for k in kept})} complexes -> {OUT}")
    print(f"skipped: {skipped}")
    on_small = sum(1 for k in kept if k["mut_chain"] in k["small_chains"])
    print(f"mutations ON the short partner chain (the peptide-like regime): {on_small}")


if __name__ == "__main__":
    main()
