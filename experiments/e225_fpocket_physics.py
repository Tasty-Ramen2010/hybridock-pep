"""E225 — fpocket apo-pocket PHYSICS descriptors (the cheap proxy for Ram's 500ns pocket-water MD): does
3D pocket physics break the receptor-baseline wall that sequence/ESM couldn't (0.15)?

For each PDBbind-925 complex: load RCSB structure → strip peptide → run fpocket on apo receptor → find the
pocket at the binding site (overlap with known binding-site residues) → extract 18 physics descriptors
(druggability, volume, SASA polar/apolar, hydrophobicity, polarity, charge, flexibility, alpha-sphere density).
Cache → data/e225_fpocket.jsonl. Then test: per-complex ΔG clustered-CV, baseline vs +pocket-physics.
"""
from __future__ import annotations

import json
import os
import subprocess
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
import e180_protdcal_925 as e180  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402
FPOCKET = "/home/igem/miniconda3/envs/score-env/bin/fpocket"
WORK = ROOT / "runs" / "e225_fp"; WORK.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "data" / "e225_fpocket.jsonl"
T2O = e180.T2O
_parser = PDBParser(QUIET=True)
# fpocket descriptor keys (order in *_info.txt)
DKEYS = ["Score", "Druggability Score", "Number of Alpha Spheres", "Total SASA", "Polar SASA", "Apolar SASA",
         "Volume", "Mean local hydrophobic density", "Mean alpha sphere radius", "Mean alp. sph. solvent access",
         "Apolar alpha sphere proportion", "Hydrophobicity score", "Volume score", "Polarity score",
         "Charge score", "Proportion of polar atoms", "Alpha sphere density",
         "Cent. of mass - Alpha Sphere max dist", "Flexibility"]


class ApoSel(Select):
    def __init__(self, pep):
        self.pep = pep

    def accept_chain(self, c):
        return c.id != self.pep

    def accept_residue(self, r):
        return r.id[0] == " "


def parse_info(p):
    """parse fpocket _info.txt → list of dicts per pocket."""
    pockets = []; cur = None
    for line in p.read_text().splitlines():
        if line.startswith("Pocket"):
            cur = {}; pockets.append(cur)
        elif ":" in line and cur is not None:
            k, _, v = line.strip().partition(":")
            k = k.strip()
            if k in DKEYS:
                try:
                    cur[k] = float(v.strip())
                except ValueError:
                    pass
    return pockets


def pocket_atoms_residues(pqr, pid):
    """residue ids belonging to a given pocket number from the pockets/pocketN_atm.pdb."""
    return None  # not needed; we match by spatial proximity below


def run_one(pdb, pepseq, site_xyz):
    f = e180.fetch(pdb)
    if f is None:
        return None
    try:
        st = _parser.get_structure(pdb, str(f))
    except Exception:  # noqa: BLE001
        return None
    want = pepseq.upper(); L = len(want); pep_ch = None
    for ch in st[0]:
        seq = "".join(T2O.get(r.resname, "") for r in ch if r.id[0] == " ")
        nstd = sum(1 for r in ch if r.id[0] == " ")
        if 2 <= nstd <= 60 and (want in seq or (seq and seq in want)) and abs(len(seq) - L) <= max(3, 0.4 * L):
            pep_ch = ch.id; break
    if pep_ch is None:
        return None
    apo = WORK / f"{pdb}.pdb"
    io = PDBIO(); io.set_structure(st); io.save(str(apo), ApoSel(pep_ch))
    try:
        subprocess.run([FPOCKET, "-f", str(apo)], capture_output=True, timeout=120)
    except Exception:  # noqa: BLE001
        return None
    outdir = WORK / f"{pdb}_out"
    info = outdir / f"{pdb}_info.txt"
    if not info.exists():
        return None
    pockets = parse_info(info)
    if not pockets:
        return None
    # pick the pocket nearest the binding site (peptide centroid)
    best, best_d = None, 1e18
    for i, pk in enumerate(pockets, 1):
        pf = outdir / "pockets" / f"pocket{i}_atm.pdb"
        if not pf.exists():
            continue
        try:
            coords = np.array([a.coord for a in _parser.get_structure("p", str(pf)).get_atoms()])
        except Exception:  # noqa: BLE001
            continue
        d = np.linalg.norm(coords.mean(0) - site_xyz)
        if d < best_d:
            best_d, best = d, pk
    import shutil
    shutil.rmtree(outdir, ignore_errors=True); apo.unlink(missing_ok=True)
    if best is None or best_d > 15:
        return None
    return [float(best.get(k, 0.0)) for k in DKEYS]


def build():
    rows = [json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")]
    done = {json.loads(l)["pdb"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    todo = [r for r in rows if r["pdb"] not in done]
    print(f"=== E225 fpocket: {len(done)} done, {len(todo)} to do ===", flush=True)
    t0 = time.time(); n = ok = 0
    for r in todo:
        n += 1
        # binding-site centroid = mean of peptide atoms — fetch from structure once
        f = e180.fetch(r["pdb"])
        site = None
        if f is not None:
            try:
                st = _parser.get_structure(r["pdb"], str(f))
                want = r["seq"].upper()
                for ch in st[0]:
                    seq = "".join(T2O.get(rr.resname, "") for rr in ch if rr.id[0] == " ")
                    if (want in seq or (seq and seq in want)) and abs(len(seq) - len(want)) <= max(3, 0.4 * len(want)):
                        site = np.mean([a.coord for rr in ch if rr.id[0] == " " for a in rr], 0); break
            except Exception:  # noqa: BLE001
                pass
        d = run_one(r["pdb"], r["seq"], site) if site is not None else None
        with open(OUT, "a") as fh:
            fh.write(json.dumps({"pdb": r["pdb"], "fp": d}) + "\n")
        if d:
            ok += 1
        if n % 50 == 0:
            print(f"  {n}/{len(todo)} ok={ok} {(time.time()-t0)/n:.2f}s/cplx", flush=True)
    print(f"=== done: {ok}/{n} ===", flush=True)


if __name__ == "__main__":
    build()
