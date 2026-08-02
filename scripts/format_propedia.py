#!/usr/bin/env python3
"""format_propedia.py — Turn selected Propedia complexes into RAPiDock training format.

Input:  data/propedia_longsel.csv (pdb, chain=peptide chain, size, seq) +
        datasets/propedia/complex2_3.zip (files named complex/{pdb}_{recChain}_{pepChain}.pdb)
Output: datasets/propedia_formatted/{name}/{name}_peptide.pdb + {name}_protein_pocket.pdb
        and data/propedia_train.csv (complex_name, protein_description, peptide_description, source)

Pocket = receptor residues with any atom within POCKET_R (10 A) of any peptide atom
(matches how RefPepDB pockets were truncated). Peptide = the pepChain; receptor = all
other chains. Runs in rapidock env (Biopython).

Usage:
    python scripts/format_propedia.py --limit 3     # smoke
    python scripts/format_propedia.py               # full
"""
from __future__ import annotations

import argparse
import csv
import io
import warnings
import zipfile
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "datasets" / "propedia" / "complex2_3.zip"
SEL = ROOT / "data" / "propedia_longsel.csv"
OUTDIR = ROOT / "datasets" / "propedia_formatted"
OUTCSV = ROOT / "data" / "propedia_train.csv"
# MUST match RAPiDock's pocket_trunction.py threshold (20.0 A). Using 10.0 produced
# pockets 2.6x too small (median 70 vs RecentSet 186 res) → the model learned placement
# relative to tiny pockets and degraded badly at inference. See lengthcliff retrain post-mortem.
POCKET_R = 20.0

AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}


def format_one(zf, member, pep_chain, name, outdir):
    from Bio.PDB import PDBParser, PDBIO, Select, NeighborSearch
    raw = zf.read(member).decode("utf-8", "ignore")
    struct = PDBParser(QUIET=True).get_structure(name, io.StringIO(raw))
    model = struct[0]
    if pep_chain not in [c.id for c in model]:
        return None
    pep_atoms = [a for a in model[pep_chain].get_atoms()]
    if len(pep_atoms) < 8:
        return None
    ns = NeighborSearch([a for c in model for a in c.get_atoms() if c.id != pep_chain])
    pocket_res = set()
    for pa in pep_atoms:
        for hit in ns.search(pa.coord, POCKET_R):
            r = hit.get_parent()
            if r.get_parent().id != pep_chain and r.resname in AA3:
                pocket_res.add(r)
    if len(pocket_res) < 10:  # reject degenerate/solvent-exposed peptides
        return None

    d = outdir / name
    d.mkdir(parents=True, exist_ok=True)
    io_ = PDBIO()

    class PepSel(Select):
        def accept_chain(self, c): return c.id == pep_chain
        def accept_residue(self, r): return r.get_parent().id == pep_chain and r.resname in AA3

    class PockSel(Select):
        def accept_residue(self, r): return r in pocket_res

    io_.set_structure(struct)
    pep_path = d / f"{name}_peptide.pdb"
    pock_path = d / f"{name}_protein_pocket.pdb"
    io_.save(str(pep_path), PepSel())
    io_.save(str(pock_path), PockSel())
    return str(pock_path), str(pep_path), len(pocket_res)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sel", type=Path, default=SEL,
                    help="selection CSV (pdb,chain,size,resolution,seq)")
    ap.add_argument("--outcsv", type=Path, default=OUTCSV,
                    help="output training CSV path")
    args = ap.parse_args()

    sel = list(csv.DictReader(open(args.sel)))
    if args.limit:
        sel = sel[: args.limit]
    OUTDIR.mkdir(parents=True, exist_ok=True)

    zf = zipfile.ZipFile(ZIP)
    names = zf.namelist()
    # index: pdb -> list of members
    from collections import defaultdict
    idx = defaultdict(list)
    for m in names:
        base = Path(m).name  # {pdb}_{rec}_{pep}.pdb
        idx[base.split("_")[0].lower()].append(m)

    rows, ok, fail = [], 0, 0
    for i, r in enumerate(sel):
        pdb, chain = r["pdb"].lower(), r["chain"]
        # member whose peptide chain (last token before .pdb) == chain
        member = None
        for m in idx.get(pdb, []):
            toks = Path(m).stem.split("_")
            if len(toks) >= 3 and toks[-1] == chain:
                member = m
                break
        if member is None and idx.get(pdb):
            member = idx[pdb][0]
        if member is None:
            fail += 1
            continue
        name = f"propedia_{pdb}_{chain}"
        try:
            res = format_one(zf, member, chain, name, OUTDIR)
        except Exception as e:
            res = None
            if fail < 5:
                print(f"  [{name}] ERR {type(e).__name__}: {str(e)[:80]}")
        if res is None:
            fail += 1
            continue
        pock, pep, nres = res
        rows.append({"complex_name": name, "protein_description": pock,
                     "peptide_description": pep, "source": "propedia"})
        ok += 1
        if ok <= 5 or ok % 200 == 0:
            print(f"  [{ok}] {name} pocket={nres}res size={r['size']}", flush=True)

    with open(args.outcsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["complex_name", "protein_description",
                                          "peptide_description", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[format_propedia] formatted={ok} failed={fail} -> {args.outcsv}", flush=True)


if __name__ == "__main__":
    main()
