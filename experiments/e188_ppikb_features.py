"""E188 — extract features for ALL PPIKB complexes (feeds both training-expansion AND selectivity).

For each PPIKB entry: fetch its PDB, extract the peptide chain (seq-matched) + pocket residues (non-peptide
chains within 8A). Compute: ProtDCal-3D contact descriptors (37, d=6 t=3) on the peptide + pocket ProtDCal
seq descriptors + peptide ProtDCal seq descriptors + charge/length. Writes data/ppikb_features.jsonl.
Resumable. CPU+network only (does not touch GPU). Reuses RCSB cache from e180.
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
import e179_protdcal_3d as e179  # noqa: E402
e150 = __import__("importlib").util.module_from_spec(
    __import__("importlib").util.spec_from_file_location("e150", ROOT / "experiments/e150_protdcal_descriptors.py"))
__import__("importlib").util.spec_from_file_location("e150", ROOT / "experiments/e150_protdcal_descriptors.py").loader.exec_module(e150)
SCALES = e150.SCALES
SN = list(SCALES.keys())
T2O = e179.T2O
PDBDIR = ROOT / "data" / "rcsb_full"; PDBDIR.mkdir(exist_ok=True)
ALT = ROOT / "data" / "skempi_pdbs"
_parser = PDBParser(QUIET=True)
OUT = ROOT / "data" / "ppikb_features.jsonl"


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


def chain_res(st, ch):
    out = []
    for r in st[0][ch]:
        if r.id[0] != " ":
            continue
        aa = T2O.get(r.resname)
        if aa is None:
            continue
        a = r["CB"] if "CB" in r else (r["CA"] if "CA" in r else None)
        if a is not None:
            out.append((aa, a.coord))
    return out


def extract(pdb, want_seq):
    f = fetch(pdb)
    if f is None:
        return None
    try:
        st = _parser.get_structure(pdb, str(f))
    except Exception:  # noqa: BLE001
        return None
    want = want_seq.upper(); L = len(want)
    best, best_sc, best_ch = None, -1e9, None
    for ch in st[0]:
        res = chain_res(st, ch.id)
        if not (2 <= len(res) <= 60):
            continue
        seq = "".join(a for a, _ in res)
        sc = (100 - abs(len(seq) - L)) if (want in seq or seq in want) else \
             (-abs(len(seq) - L) + sum(min(seq.count(c), want.count(c)) for c in set(want)))
        if sc > best_sc:
            best, best_sc, best_ch = res, sc, ch.id
    if best is None or abs(len(best) - L) > max(3, 0.4 * L):
        return None
    # pocket = non-peptide chain residues within 8A of peptide
    pep_xyz = np.array([c for _, c in best])
    pocket = []
    for ch in st[0]:
        if ch.id == best_ch:
            continue
        for r in ch:
            if r.id[0] != " ":
                continue
            aa = T2O.get(r.resname)
            if aa is None:
                continue
            rc = np.array([a.coord for a in r])
            if rc.size and ((rc[:, None, :] - pep_xyz[None, :, :]) ** 2).sum(-1).min() <= 64.0:
                pocket.append(aa)
    desc3d = e179.descriptors(best, 6.0, 3)
    pkf = [float(np.mean([SCALES[s].get(c, 0) for c in pocket])) for s in SN] if pocket else [0.0] * len(SN)
    return {"desc3d": desc3d, "pocket_pkf": pkf, "npep": len(best), "npocket": len(pocket)}


def main():
    rows = [json.loads(l) for l in open(ROOT / "data/ppikb_clean.jsonl")]
    done = {json.loads(l)["id"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    todo = [r for r in rows if r["id"] not in done and r["pdb"]]
    print(f"=== E188 PPIKB features: {len(done)} done, {len(todo)} to do ===", flush=True)
    t0 = time.time(); n = ok = 0
    for r in todo:
        n += 1
        e = extract(r["pdb"], r["seq"])
        rec = {"id": r["id"], "pdb": r["pdb"], "seq": r["seq"], "length": r["length"], "y": r["y"],
               "aff_type": r["aff_type"], "protein_seq": r["protein_seq"][:50], "net_charge": r["net_charge"]}
        if e is not None:
            rec.update(e); ok += 1
        else:
            rec["desc3d"] = None
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if n % 50 == 0:
            print(f"  {n}/{len(todo)} ok={ok} {(time.time()-t0)/n:.2f}s/entry", flush=True)
    print(f"=== done: {ok}/{n} got structure ===", flush=True)


if __name__ == "__main__":
    main()
