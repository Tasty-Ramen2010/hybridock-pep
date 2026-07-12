"""E180 — compute faithful ProtDCal-3D (37 PPI descriptors) on the 925 PDBbind peptides + T100, by
fetching full RCSB structures and extracting the peptide chain (seq-matched). Caches descriptors, then
trains SMOreg on 925 and predicts T100 — the real faithfulness gate (vs truth AND vs PPI shipped preds).
Contact config d=6.0,t=3 (best match-to-shipped from E179 sweep).
Resumable. CPU+network only (does not touch GPU campaigns).
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "3"
import numpy as np  # noqa: E402
from Bio.PDB import PDBParser  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e179_protdcal_3d as e179  # noqa: E402  (descriptors, invariant, parse, PROP, GROUPS)

T2O = e179.T2O
PDBDIR = ROOT / "data" / "rcsb_full"; PDBDIR.mkdir(exist_ok=True)
ALT = ROOT / "data" / "skempi_pdbs"
_parser = PDBParser(QUIET=True)
D_CUT, T_CUT = 6.0, 3
OUT = ROOT / "data" / "e180_protdcal3d.jsonl"


def fetch(pdb):
    for c in (PDBDIR / f"{pdb}.pdb", ALT / f"{pdb}.pdb"):
        if c.exists() and c.stat().st_size > 0:
            return c
    f = PDBDIR / f"{pdb}.pdb"
    try:
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb.upper()}.pdb", f)
        return f if f.stat().st_size > 0 else None
    except Exception:  # noqa: BLE001
        return None


def chain_residues(st, ch):
    out = []
    for r in st[0][ch]:
        if r.id[0] != " ":
            continue
        aa = T2O.get(r.resname)
        if aa is None:
            continue
        atom = r["CB"] if "CB" in r else (r["CA"] if "CA" in r else None)
        if atom is not None:
            out.append((aa, atom.coord))
    return out


def peptide_chain(pdb, want_seq):
    """find the chain whose sequence best matches the known peptide seq."""
    f = fetch(pdb)
    if f is None:
        return None
    try:
        st = _parser.get_structure(pdb, str(f))
    except Exception:  # noqa: BLE001
        return None
    want = want_seq.upper(); L = len(want)
    best, best_sc = None, -1
    for ch in st[0]:
        res = chain_residues(st, ch.id)
        if not (2 <= len(res) <= 60):
            continue
        seq = "".join(a for a, _ in res)
        # score: exact substring match, else length proximity + AA-composition overlap
        if want in seq or seq in want:
            sc = 100 - abs(len(seq) - L)
        else:
            sc = -abs(len(seq) - L) + sum((min(seq.count(c), want.count(c)) for c in set(want)))
        if sc > best_sc:
            best, best_sc = res, sc
    # require a reasonable match (peptide chain length close to seq)
    if best is None or abs(len(best) - L) > max(3, 0.4 * L):
        return None
    return best


def main():
    rows = [json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")]
    done = {json.loads(l)["pdb"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    todo = [r for r in rows if r["pdb"] not in done]
    print(f"=== E180: {len(done)} cached, {len(todo)} to do (925 PDBbind) ===", flush=True)
    t0 = time.time(); n = ok = 0
    for r in todo:
        n += 1
        res = peptide_chain(r["pdb"], r["seq"])
        rec = {"pdb": r["pdb"], "seq": r["seq"], "y": float(r["y"]), "length": r["length"]}
        if res is not None:
            rec["desc"] = e179.descriptors(res, D_CUT, T_CUT); rec["npep"] = len(res); ok += 1
        else:
            rec["desc"] = None
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if n % 25 == 0:
            print(f"  {n}/{len(todo)}  ok={ok}  {(time.time()-t0)/n:.2f}s/cplx", flush=True)
    print(f"=== done: {ok}/{n} got peptide structure ===", flush=True)


if __name__ == "__main__":
    main()
