"""E228 — assemble the MD-pilot receptor set (CPU, no GPU). Group PDBbind-925 peptide complexes by RECEPTOR
(longest protein chain sequence), keep receptors that have >=MIN_PEP distinct peptide binders with an
affinity spread — these are the only complexes where a "receptor baseline" is even measurable and where
pocket-water MD could add signal the static pose can't. Emits the pilot manifest + sizes the experiment.

Run: python3 scripts/e228_pilot_assemble.py [--min-pep 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from Bio.PDB import PDBParser  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e180_protdcal_925 as e180  # noqa: E402

T2O = e180.T2O
_parser = PDBParser(QUIET=True)
OUT = ROOT / "data" / "e228_pilot_manifest.json"


def receptor_seq(pdb, pepseq):
    """Return (receptor_chain_seqs joined, peptide_chain_id) using the cached RCSB structure."""
    f = e180.fetch(pdb)
    if f is None:
        return None, None
    try:
        st = _parser.get_structure(pdb, str(f))
    except Exception:  # noqa: BLE001
        return None, None
    want = pepseq.upper()
    pep_ch = None
    chains = {}
    for ch in st[0]:
        seq = "".join(T2O.get(r.resname, "") for r in ch if r.id[0] == " ")
        if not seq:
            continue
        chains[ch.id] = seq
        if pep_ch is None and (want in seq or (seq and seq in want)) and abs(len(seq) - len(want)) <= max(3, 0.4 * len(want)):
            pep_ch = ch.id
    if pep_ch is None:
        return None, None
    rec = "/".join(sorted(s for cid, s in chains.items() if cid != pep_ch and len(s) >= 25))
    return (rec or None), pep_ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-pep", type=int, default=3)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out_path = Path(a.out)

    rows = [json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")]
    by_rec = defaultdict(list)
    n_ok = 0
    for i, r in enumerate(rows, 1):
        rec, pep_ch = receptor_seq(r["pdb"], r["seq"])
        if rec is None:
            continue
        n_ok += 1
        by_rec[rec].append({"pdb": r["pdb"], "seq": r["seq"], "y": float(r["y"]),
                            "L": r["length"], "pep_ch": pep_ch})
        if i % 100 == 0:
            print(f"  scanned {i}/{len(rows)}  receptors so far={len(by_rec)}", flush=True)

    # collapse near-identical receptors (same sequence) already grouped by exact string.
    multi = []
    for rec, peps in by_rec.items():
        uniq = {p["seq"]: p for p in peps}.values()      # distinct peptides
        if len(uniq) >= a.min_pep:
            ys = [p["y"] for p in uniq]
            multi.append({"receptor_len": len(rec.replace("/", "")), "n_pep": len(uniq),
                          "y_mean": float(np.mean(ys)), "y_std": float(np.std(ys)),
                          "y_min": float(min(ys)), "y_max": float(max(ys)),
                          "peptides": list(uniq), "rec_seq": rec})
    multi.sort(key=lambda d: (d["n_pep"], d["y_std"]), reverse=True)

    print(f"\n=== PILOT SIZING (min {a.min_pep} distinct peptides/receptor) ===")
    print(f"  complexes with a parsed receptor: {n_ok}/{len(rows)}")
    print(f"  multi-binder receptors: {len(multi)}")
    print(f"  total peptides covered:  {sum(d['n_pep'] for d in multi)}")
    tot = sum(d["n_pep"] for d in multi)
    if multi:
        bm = np.mean([d["y_std"] for d in multi])
        print(f"  mean within-receptor affinity spread (std): {bm:.2f} log-units")
        print(f"  receptor-baseline std (between receptors):  {np.std([d['y_mean'] for d in multi]):.2f}")
        print("\n  top multi-binder receptors (n_pep, spread, mean, rec_len):")
        for d in multi[:20]:
            print(f"    n={d['n_pep']:<3} std={d['y_std']:.2f} mean={d['y_mean']:+.2f} reclen={d['receptor_len']}")
    json.dump({"min_pep": a.min_pep, "n_receptors": len(multi), "n_peptides": tot, "receptors": multi},
              open(out_path, "w"))
    print(f"\n  wrote manifest → {out_path}")
    print(f"  VERDICT: {'ADEQUATELY POWERED' if len(multi) >= 15 else 'UNDERPOWERED — pilot needs PPIKB structures or more PDBbind'} "
          f"({len(multi)} receptors)")


if __name__ == "__main__":
    main()
