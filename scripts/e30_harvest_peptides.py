"""E30 — harvest PEPTIDE complexes from PPB-Affinity (3023 PDBs) to grow the diverse corpus.

PPB-Affinity is mostly protein-protein; filter to entries whose LIGAND chain is a peptide
(4-35 residues) and the receptor is larger. Download from RCSB, extract chains, compute
geometry+MJ features on the native pose. Build /tmp/e30_harvest.json — a diverse Kd corpus
to test whether MORE diverse training data lifts generalization toward PPI-Affinity (0.554).
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from Bio.PDB import PDBParser, PDBIO, Select  # noqa: E402
from hybridock_pep.scoring.geometry_features import compute_geometry_features  # noqa: E402

P = PDBParser(QUIET=True)
CACHE = Path("/tmp/ppb_pdb"); CACHE.mkdir(exist_ok=True)
WORK = Path("/tmp/ppb_work"); WORK.mkdir(exist_ok=True)
AA3 = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())


def fetch(pdb):
    f = CACHE / f"{pdb}.pdb"
    if not f.exists() or f.stat().st_size < 500:
        subprocess.run(["curl", "-sSL", "-o", str(f),
                        f"https://files.rcsb.org/download/{pdb}.pdb"], capture_output=True, timeout=40)
    return f if f.exists() and f.stat().st_size > 500 else None


class _Chain(Select):
    def __init__(self, ch): self.ch = ch
    def accept_chain(self, c): return c.id == self.ch
    def accept_residue(self, r): return r.id[0] == " " and r.resname.strip() in AA3


def chain_len(model, ch):
    if ch not in {c.id for c in model}:
        return 0
    return len([r for r in model[ch] if r.id[0] == " " and r.resname.strip() in AA3])


def main():
    rows = [r for r in csv.DictReader(open("/tmp/ppb.csv")) if not r["mutstr"].strip()]
    # unique (pdb, lig, rec) wild-type, prefer PDBbind (has peptides)
    seen = set(); cand = []
    for r in rows:
        key = (r["pdb"], r["ligand"], r["receptor"])
        if key in seen:
            continue
        seen.add(key)
        try:
            dg = float(r["dG"])
        except ValueError:
            continue
        if -25 < dg < -2:  # plausible single-site ΔG (drop extreme SKEMPI sums)
            cand.append((r["pdb"].upper(), r["ligand"], r["receptor"], dg))
    print(f"candidates: {len(cand)} unique WT entries", flush=True)
    out_path = Path("/tmp/e30_harvest.json")
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    done = set(out)
    n_pep = 0
    for i, (pdb, lig, rec, dg) in enumerate(cand):
        key = f"{pdb}_{lig}_{rec}"
        if key in done:
            n_pep += 1 if out.get(key) else 0
            continue
        f = fetch(pdb)
        if not f:
            continue
        try:
            m = P.get_structure(pdb, str(f))[0]
            ll, rl = chain_len(m, lig), chain_len(m, rec)
            if not (4 <= ll <= 35 and rl >= 40):  # peptide ligand + protein receptor
                out[key] = None  # mark as non-peptide so we skip next time
                continue
            io = PDBIO(); io.set_structure(P.get_structure("p", str(f)))
            pepf = WORK / f"{key}_pep.pdb"; io.save(str(pepf), _Chain(lig))
            io2 = PDBIO(); io2.set_structure(P.get_structure("r", str(f)))
            recf = WORK / f"{key}_rec.pdb"; io2.save(str(recf), _Chain(rec))
            feat = compute_geometry_features(pepf, recf)
            if not feat:
                out[key] = None; continue
            out[key] = dict(pdb=pdb, y=dg, L=ll, **feat)
            n_pep += 1
            if n_pep % 10 == 0:
                print(f"  peptides found: {n_pep} (scanned {i+1}/{len(cand)})", flush=True)
        except Exception:  # noqa: BLE001
            out[key] = None
        if (i + 1) % 50 == 0:
            out_path.write_text(json.dumps(out))
    out_path.write_text(json.dumps(out))
    real = {k: v for k, v in out.items() if v}
    print(f"DONE: {len(real)} peptide complexes harvested from {len(cand)} scanned", flush=True)


if __name__ == "__main__":
    main()
