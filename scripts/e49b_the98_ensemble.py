"""E49b — ensemble <E_int> features on the-98 (cross-dataset partner for M1 leave-dataset-out).

Same Langevin-MD ensemble as e49 (ff14SB+GBn2, GPU) on the-98 poses in /tmp/ppep_work. Stores
<E_int>, std, -TdS, dg_single + seq (for net-charge) so the M1 residual model has identical features
on both datasets. Cached/resumable.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybridock_pep.scoring.interaction_entropy import (interaction_entropy,  # noqa: E402
                                                       sample_interaction_energies)
from hybridock_pep.scoring.mmgbsa import compute_mmgbsa_single  # noqa: E402
from Bio.PDB import PDBParser  # noqa: E402

P = PDBParser(QUIET=True)
A3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
      "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
      "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
N_FRAMES, STEPS = 50, 300


def seq_of(pdb):
    return "".join(A3.get(r.resname.upper(), "X") for r in P.get_structure("p", str(pdb))[0].get_residues()
                   if r.id[0] == " ")


def cf(seq):
    return sum(c in "DEKR" for c in seq) / max(1, len(seq))


def main():
    e28 = json.loads(Path("/tmp/e28_feats.json").read_text())
    work = Path("/tmp/ppep_work")
    cache = Path("/tmp/e49b_the98.json")
    out = json.loads(cache.read_text()) if cache.exists() else {}
    items = [(k, work / f"{k}_pep.pdb", work / f"{k}_rec.pdb", e28[k]["y"])
             for k in e28 if (work / f"{k}_pep.pdb").exists() and (work / f"{k}_rec.pdb").exists()]
    print(f"=== e49b ensemble on the-98 ({len(items)} complexes, GPU) ===", flush=True)
    for k, pep, rec, y in items:
        if k in out:
            continue
        try:
            seq = seq_of(pep)
            dg_s = compute_mmgbsa_single(pep.resolve(), rec.resolve(), force_cpu=False)
            eint = sample_interaction_energies(pep.resolve(), rec.resolve(), n_frames=N_FRAMES,
                                               steps_between_frames=STEPS, force_cpu=False)
            out[k] = dict(y=y, cf=cf(seq), seq=seq, L=len(seq), dg_single=float(dg_s),
                          e_int_mean=float(eint.mean()), e_int_std=float(eint.std()),
                          minus_tds=float(interaction_entropy(eint)))
            cache.write_text(json.dumps(out))
            if len(out) % 10 == 0:
                print(f"  {len(out)}/{len(items)} done", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {k} FAIL {type(e).__name__}: {str(e)[:50]}", flush=True)
    print(f"done: {len(out)} complexes", flush=True)


if __name__ == "__main__":
    main()
