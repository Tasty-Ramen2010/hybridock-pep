"""E371 — compute the 16 STRUCT geometry features for PPIKB (Kd/Ki) so we can dual-train the HEADLINE model.

Uses the SAME extractor as PDBbind (src/hybridock_pep/scoring/geometry_features.compute_geometry_features) and the
SAME peptide-chain identification as e188, so the features are directly poolable with data/pdbbind_peptides.jsonl.
For each PPIKB Kd/Ki complex with a local structure: find the peptide chain, split peptide vs receptor into temp
PDBs, run compute_geometry_features, keep the 16 GEOMETRY_KEYS + seq + y. Caches incrementally (resumable).

Run: OMP_NUM_THREADS=1 LD_LIBRARY_PATH=$CONDA_PREFIX/lib python scripts/e371_ppikb_struct_extract.py
"""
from __future__ import annotations
import json, os, sys, tempfile
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import protein_letters_3to1 as _T2O

sys.path.insert(0, str(ROOT := str(Path(__file__).resolve().parents[1])))
from src.hybridock_pep.scoring.geometry_features import compute_geometry_features  # noqa: E402
from src.hybridock_pep.scoring.affinity_model import GEOMETRY_KEYS  # noqa: E402

ROOT = Path(ROOT)
PDBDIR = ROOT / "data/rcsb_full"
OUT = ROOT / "data/ppikb_struct_features.jsonl"
_parser = PDBParser(QUIET=True)
T2O = {k.upper(): v for k, v in _T2O.items()}


def chain_seq(st, ch):
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


def find_pep_chain(st, want):
    want = want.upper(); L = len(want)
    best_ch, best_sc, best_n = None, -1e9, 0
    for ch in st[0]:
        res = chain_seq(st, ch.id)
        if not (2 <= len(res) <= 60):
            continue
        seq = "".join(a for a, _ in res)
        sc = (100 - abs(len(seq) - L)) if (want in seq or seq in want) else \
             (-abs(len(seq) - L) + sum(min(seq.count(c), want.count(c)) for c in set(want)))
        if sc > best_sc:
            best_ch, best_sc, best_n = ch.id, sc, len(res)
    if best_ch is None or abs(best_n - L) > max(3, 0.4 * L):
        return None
    return best_ch


class _ChainSel(Select):
    def __init__(self, keep, exclude=False):
        self.keep, self.exclude = keep, exclude

    def accept_chain(self, ch):
        inkeep = ch.id == self.keep
        return (not inkeep) if self.exclude else inkeep


def main():
    done = set()
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["id"])
    rows = [json.loads(l) for l in (ROOT / "data/ppikb_features.jsonl").read_text().splitlines() if l.strip()]
    todo = [r for r in rows if r.get("aff_type") in ("Kd", "KD", "Ki") and r["id"] not in done and r.get("seq")]
    print(f"PPIKB Kd/Ki to extract: {len(todo)}  (already cached {len(done)})")

    io = PDBIO()
    ok = fail = 0
    with open(OUT, "a") as fout:
        for i, r in enumerate(todo, 1):
            pdb = r["pdb"].lower()
            f = PDBDIR / f"{pdb}.pdb"
            rec = {"id": r["id"], "pdb": pdb, "seq": r["seq"], "y": r["y"], "aff_type": r.get("aff_type")}
            try:
                if not f.exists():
                    raise FileNotFoundError
                st = _parser.get_structure(pdb, str(f))
                pep_ch = find_pep_chain(st, r["seq"])
                if pep_ch is None:
                    raise ValueError("no peptide chain")
                with tempfile.TemporaryDirectory() as td:
                    pep_p, rec_p = Path(td) / "pep.pdb", Path(td) / "rec.pdb"
                    io.set_structure(st); io.save(str(pep_p), _ChainSel(pep_ch, exclude=False))
                    io.set_structure(st); io.save(str(rec_p), _ChainSel(pep_ch, exclude=True))
                    g = compute_geometry_features(pep_p, rec_p)
                if not g:
                    raise ValueError("geometry None")
                for k in GEOMETRY_KEYS:
                    rec[k] = float(g.get(k, 0.0))
                ok += 1
            except Exception as e:  # noqa: BLE001
                rec["error"] = str(e)[:60]
                fail += 1
            fout.write(json.dumps(rec) + "\n")
            if i % 50 == 0:
                fout.flush()
                print(f"  {i}/{len(todo)}  ok={ok} fail={fail}", flush=True)
    print(f"DONE: ok={ok} fail={fail} → {OUT}")


if __name__ == "__main__":
    main()
