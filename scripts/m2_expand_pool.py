"""M2 — expand the labelled pool with new peptide-Kd complexes, run ensemble features, retest.

Ram's sparsity warning confirmed (~15-25 truly new). Fetch the new Kd PDBs, extract peptide (shortest
chain, 4-30 res) + receptor pocket, run the e49 ensemble pipeline (ff14SB+GBn2, GPU), add to the pool,
retest the INTENSIVE-only model (the Simpson-robust fix). Crystal poses used (peptide as deposited).
"""
from __future__ import annotations

import csv
import json
import math
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from Bio.PDB import PDBParser, PDBIO, Select  # noqa: E402
from hybridock_pep.scoring.interaction_entropy import (interaction_entropy,  # noqa: E402
                                                       sample_interaction_energies)
from hybridock_pep.scoring.mmgbsa import compute_mmgbsa_single  # noqa: E402

P = PDBParser(QUIET=True)
A3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
      "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
      "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
R_KCAL = 1.987e-3
WORK = Path("/tmp/m2_work"); WORK.mkdir(exist_ok=True)


def fetch(pdb):
    f = WORK / f"{pdb}.pdb"
    if not f.exists():
        try:
            urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb}.pdb", f)
        except Exception:
            return None
    return f if f.exists() else None


def chains_of(model):
    out = {}
    for ch in model:
        res = [r for r in ch if r.id[0] == " " and r.resname.upper() in A3]
        if res:
            out[ch.id] = res
    return out


def extract(pdb_file, pdb):
    """shortest protein chain (4-30) = peptide; receptor = pocket residues within 8Å."""
    model = P.get_structure(pdb, str(pdb_file))[0]
    chs = chains_of(model)
    if len(chs) < 2:
        return None
    pep_ch = min(chs, key=lambda c: len(chs[c]))
    if not (4 <= len(chs[pep_ch]) <= 30):
        return None
    pep_xyz = np.array([a.coord for r in chs[pep_ch] for a in r if a.element != "H"])
    pep_f = WORK / f"{pdb}_pep.pdb"; rec_f = WORK / f"{pdb}_rec.pdb"

    class Pep(Select):
        def accept_chain(self, c): return c.id == pep_ch
        def accept_residue(self, r): return r.id[0] == " " and r.resname.upper() in A3

    class Poc(Select):
        def accept_residue(self, r):
            if r.id[0] != " " or r.get_parent().id == pep_ch or r.resname.upper() not in A3:
                return False
            return any(np.min(((pep_xyz - a.coord) ** 2).sum(1)) <= 100.0
                       for a in r if a.element != "H")
    io = PDBIO()
    io.set_structure(model); io.save(str(pep_f), Pep())
    io.set_structure(model); io.save(str(rec_f), Poc())
    seq = "".join(A3.get(r.resname.upper(), "X") for r in chs[pep_ch])
    return pep_f, rec_f, seq


def main():
    rows = [r for r in csv.DictReader(open(ROOT / "data/rcsb_binding_affinity_bulk.csv"))
            if r["affinity_type"].strip().lower() == "kd"]
    have = set(r["pdb"].upper() for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text()))
    have98 = set(k.split("_")[0].upper() for k in json.loads(Path("/tmp/e28_feats.json").read_text()))
    # best (lowest) Kd per new PDB
    bykd = {}
    for r in rows:
        pdb = r["pdb_id"].upper()
        if pdb in have or pdb in have98:
            continue
        try:
            v = float(r["value"]); u = r["unit"].strip().lower()
            kd = v * {"nm": 1e-9, "um": 1e-6, "µm": 1e-6, "mm": 1e-3, "pm": 1e-12, "m": 1.0}.get(u, 1e-9)
        except Exception:
            continue
        if kd > 0 and (pdb not in bykd or kd < bykd[pdb]):
            bykd[pdb] = kd
    cache = Path("/tmp/m2_new.json")
    out = json.loads(cache.read_text()) if cache.exists() else {}
    print(f"=== M2: {len(bykd)} new Kd PDBs to try (GPU ensemble) ===", flush=True)
    for pdb, kd in bykd.items():
        if pdb in out:
            continue
        ff = fetch(pdb)
        if not ff:
            print(f"  {pdb} fetch fail", flush=True); continue
        ex = extract(ff, pdb)
        if not ex:
            print(f"  {pdb} not protein-peptide (skip)", flush=True); continue
        pep_f, rec_f, seq = ex
        dg = R_KCAL * 298.0 * math.log(kd)   # ΔG = RT ln Kd
        try:
            ei = sample_interaction_energies(pep_f.resolve(), rec_f.resolve(), n_frames=50,
                                             steps_between_frames=300, force_cpu=False)
            ds = compute_mmgbsa_single(pep_f.resolve(), rec_f.resolve(), force_cpu=False)
            out[pdb] = dict(y=dg, cf=sum(c in "DEKR" for c in seq) / len(seq), seq=seq, L=len(seq),
                            dg_single=float(ds), e_int_mean=float(ei.mean()), e_int_std=float(ei.std()),
                            minus_tds=float(interaction_entropy(ei)))
            cache.write_text(json.dumps(out))
            print(f"  {pdb} OK len={len(seq)} ΔG={dg:.1f} <E_int>={ei.mean():.1f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {pdb} score fail {type(e).__name__}: {str(e)[:40]}", flush=True)
    print(f"=== M2 done: {len(out)} new complexes scored ===", flush=True)


if __name__ == "__main__":
    main()
