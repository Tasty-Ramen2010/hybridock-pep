"""E212 — compute the 16 production geometry features for PPIKB long/vlong complexes (the SHIP PATH for
the augmentation lever). For each PPIKB Kd long/vlong entry: fetch RCSB PDB, split into peptide chain (seq-
matched) + receptor, run compute_geometry_features → 16 geometry feats. Writes data/e212_ppikb_geom.jsonl
(adds geometry to the existing desc3d+pocket). Resumable, CPU+network only. Excludes T100 pdb+seq (no-leak).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from Bio.PDB import PDBIO, PDBParser, Select  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hybridock_pep.scoring.geometry_features import compute_geometry_features, GEOMETRY_FEATURE_KEYS  # noqa: E402
import e180_protdcal_925 as e180  # noqa: E402

T2O = e180.T2O
PDBDIR = ROOT / "data" / "rcsb_full"
WORK = ROOT / "runs" / "e212_geom"; WORK.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "data" / "e212_ppikb_geom.jsonl"
_parser = PDBParser(QUIET=True)


class ChainSel(Select):
    def __init__(self, ch):
        self.ch = ch

    def accept_chain(self, c):
        return c.id == self.ch

    def accept_residue(self, r):
        return r.id[0] == " "


class NotChainSel(Select):
    def __init__(self, ch):
        self.ch = ch

    def accept_chain(self, c):
        return c.id != self.ch

    def accept_residue(self, r):
        return r.id[0] == " "


def chain_seq(st, ch):
    return "".join(T2O.get(r.resname, "") for r in st[0][ch] if r.id[0] == " ") if ch in st[0] else ""


def geom_for(pdb, want):
    f = e180.fetch(pdb)
    if f is None:
        return None
    try:
        st = _parser.get_structure(pdb, str(f))
    except Exception:  # noqa: BLE001
        return None
    want = want.upper(); L = len(want); pep_ch = None
    for ch in st[0]:
        seq = chain_seq(st, ch.id)
        nstd = sum(1 for r in ch if r.id[0] == " ")
        if (2 <= nstd <= 60) and (want in seq or (seq and seq in want)) \
                and abs(len(seq) - L) <= max(3, 0.4 * L):
            pep_ch = ch.id; break
    if pep_ch is None:
        return None
    io = PDBIO(); io.set_structure(st)
    pp = WORK / f"{pdb}_pep.pdb"; rp = WORK / f"{pdb}_rec.pdb"
    io.save(str(pp), ChainSel(pep_ch)); io.save(str(rp), NotChainSel(pep_ch))
    try:
        g = compute_geometry_features(pp, rp)
    except Exception:  # noqa: BLE001
        g = None
    pp.unlink(missing_ok=True); rp.unlink(missing_ok=True)
    if g is None:
        return None
    return {k: float(g.get(k, 0.0)) for k in GEOMETRY_FEATURE_KEYS}


def main():
    man = {m["pdb"].lower() for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    seqc = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")}
    t100_seqs = {seqc[p]["seq"] for p in man if p in seqc}
    ours = {json.loads(l)["pdb"].lower() for l in open(ROOT / "data/pdbbind_peptides.jsonl")}
    ppikb = [json.loads(l) for l in open(ROOT / "data/ppikb_features.jsonl") if json.loads(l).get("desc3d")]
    # long+vlong Kd, no-leak, structure-clean
    pool = [r for r in ppikb if r["length"] >= 13 and r["aff_type"] in ("Kd", "KD", "pKd")
            and r["pdb"].lower() not in man and r["pdb"].lower() not in ours and r["seq"] not in t100_seqs
            and -18 < r["y"] < -2 and abs(r.get("npep", r["length"]) - r["length"]) <= 2 and r.get("npocket", 0) >= 8]
    done = {json.loads(l)["pdb"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    todo = [r for r in pool if r["pdb"] not in done]
    print(f"=== E212 PPIKB long/vlong geometry: {len(done)} done, {len(todo)} to do (pool {len(pool)}) ===", flush=True)
    t0 = time.time(); n = ok = 0
    for r in todo:
        n += 1
        g = geom_for(r["pdb"], r["seq"])
        rec = {"pdb": r["pdb"], "seq": r["seq"], "y": r["y"], "length": r["length"],
               "net_charge": r["net_charge"], "geom": g, "pocket_pkf": r.get("pocket_pkf")}
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if g is not None:
            ok += 1
        if n % 25 == 0:
            print(f"  {n}/{len(todo)} ok={ok} {(time.time()-t0)/n:.2f}s/cplx", flush=True)
    print(f"=== done: {ok}/{n} got geometry ===", flush=True)


if __name__ == "__main__":
    main()
