"""E115a — build clean idealized peptide PDBs from sequence (rapidock env, PeptideBuilder).

Free-state conformational entropy needs a CLEAN starting peptide (the mol2/crystal poses NaN in MD).
Build an extended peptide from each unique sequence in the pooled set → data/sfree_peptides/<hash>.pdb.
Then e115_md_sfree.py (score-env) runs the GPU MD on these. Resumable.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import Bio.PDB
import PeptideBuilder
from PeptideBuilder import Geometry

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data" / "sfree_peptides"
OUTDIR.mkdir(parents=True, exist_ok=True)
AA = set("ACDEFGHIKLMNPQRSTVWY")


def seqhash(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]


def collect_seqs():
    seqs = set()
    for fn in ["data/pooled_benchmark_train.csv", "data/pooled_benchmark_test.csv"]:
        for r in csv.DictReader(open(ROOT / fn)):
            s = r.get("seq", "").upper()
            if s and all(c in AA for c in s) and 2 <= len(s) <= 40:
                seqs.add(s)
    for ln in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines():
        s = json.loads(ln)["seq"].upper()
        if s and all(c in AA for c in s) and 2 <= len(s) <= 40:
            seqs.add(s)
    return sorted(seqs)


def build(seq, out):
    s = PeptideBuilder.initialize_res(Geometry.geometry(seq[0]))
    for aa in seq[1:]:
        PeptideBuilder.add_residue(s, Geometry.geometry(aa))
    io = Bio.PDB.PDBIO()
    io.set_structure(s)
    io.save(str(out))


def main():
    seqs = collect_seqs()
    index = {}
    print(f"=== E115a building {len(seqs)} unique peptides ===", flush=True)
    ok = 0
    for i, seq in enumerate(seqs):
        h = seqhash(seq)
        out = OUTDIR / f"{h}.pdb"
        index[seq] = h
        if out.exists():
            ok += 1
            continue
        try:
            build(seq, out)
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  build FAIL {seq[:20]}: {type(e).__name__}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(seqs)} built", flush=True)
    (OUTDIR / "index.json").write_text(json.dumps(index))
    print(f"=== built {ok}/{len(seqs)} → {OUTDIR} ===", flush=True)


if __name__ == "__main__":
    main()
