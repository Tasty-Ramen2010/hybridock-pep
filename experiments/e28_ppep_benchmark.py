"""E28 — download the 100-complex protein-peptide benchmark (PPI-Affinity SI), score with
our ensemble on NATIVE poses, and compare head-to-head vs PPI-Affinity/PRODIGY/etc.

Downloads real RCSB structures, extracts receptor + peptide chains, computes geometry+MJ
features on the native (crystal) pose. Saves /tmp/e28_feats.json. Stage 2 evaluates.
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from Bio.PDB import PDBParser, PDBIO, Select  # noqa: E402
from hybridock_pep.scoring.geometry_features import compute_geometry_features  # noqa: E402

P = PDBParser(QUIET=True)
CACHE = Path("/tmp/ppep_pdb"); CACHE.mkdir(exist_ok=True)
WORK = Path("/tmp/ppep_work"); WORK.mkdir(exist_ok=True)
AA3 = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())


def fetch(pdb):
    f = CACHE / f"{pdb}.pdb"
    if not f.exists() or f.stat().st_size < 500:
        subprocess.run(["curl", "-sSL", "-o", str(f),
                        f"https://files.rcsb.org/download/{pdb}.pdb"],
                       capture_output=True, timeout=60)
    return f if f.exists() and f.stat().st_size > 500 else None


class _Chain(Select):
    def __init__(self, ch):
        self.ch = ch
    def accept_chain(self, c):
        return c.id == self.ch
    def accept_residue(self, r):
        return r.id[0] == " " and r.resname.strip() in AA3


def extract(struct, chain, out):
    io = PDBIO(); io.set_structure(struct)
    io.save(str(out), _Chain(chain))
    return out if out.exists() and out.stat().st_size > 100 else None


def peptide_ss(pep_pdb):
    """Rough SS of the peptide from backbone phi/psi: fraction helix/sheet."""
    try:
        from Bio.PDB.internal_coords import IC_Chain  # noqa
    except Exception:
        return "?"
    # simple: use phi/psi via Biopython vectors
    model = P.get_structure("p", str(pep_pdb))[0]
    res = [r for ch in model for r in ch if r.id[0] == " "]
    if len(res) < 4:
        return "short"
    import numpy as np
    def dih(p):
        b0, b1, b2 = p[0]-p[1], p[2]-p[1], p[3]-p[2]
        b1 /= np.linalg.norm(b1)+1e-9
        v = b0-np.dot(b0, b1)*b1; w = b2-np.dot(b2, b1)*b1
        return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))
    h = e = 0; n = 0
    for i in range(1, len(res)-1):
        try:
            phi = dih([res[i-1]["C"].coord, res[i]["N"].coord, res[i]["CA"].coord, res[i]["C"].coord])
            psi = dih([res[i]["N"].coord, res[i]["CA"].coord, res[i]["C"].coord, res[i+1]["N"].coord])
            n += 1
            if -100 < phi < -30 and -80 < psi < -5:
                h += 1
            elif -180 < phi < -40 and 90 < psi < 180:
                e += 1
        except Exception:
            pass
    if n == 0:
        return "?"
    if h/n > 0.4:
        return "HELIX"
    if e/n > 0.3:
        return "SHEET"
    return "LOOP"


def main():
    parsed = json.loads(Path("/tmp/ppep_parsed.json").read_text())
    out_path = Path("/tmp/e28_feats.json")
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    done = set(out)
    for pdb, rec_ch, pep_ch, dg in parsed:
        key = f"{pdb}_{rec_ch}_{pep_ch}"
        if key in done:
            continue
        f = fetch(pdb)
        if not f:
            print(f"  {pdb} fetch FAIL", flush=True); continue
        try:
            s = P.get_structure(pdb, str(f))[0]
            chains = {c.id for c in s}
            if rec_ch not in chains or pep_ch not in chains:
                print(f"  {key} chain missing (have {sorted(chains)})", flush=True); continue
            recf = extract(P.get_structure("r", str(f)), rec_ch, WORK/f"{key}_rec.pdb")
            pepf = extract(P.get_structure("p", str(f)), pep_ch, WORK/f"{key}_pep.pdb")
            if not recf or not pepf:
                print(f"  {key} extract FAIL", flush=True); continue
            npep = len([r for r in P.get_structure("x", str(pepf))[0].get_residues() if r.id[0] == " "])
            nrec = len([r for r in P.get_structure("y", str(recf))[0].get_residues() if r.id[0] == " "])
            if npep < 3 or npep > 50 or nrec < npep:  # sanity: peptide shorter than receptor
                print(f"  {key} bad sizes pep={npep} rec={nrec}", flush=True); continue
            feat = compute_geometry_features(pepf, recf)
            if not feat:
                print(f"  {key} no interface", flush=True); continue
            ss = peptide_ss(pepf)
            out[key] = dict(pdb=pdb, y=dg, L=npep, ss=ss, **feat)
            out_path.write_text(json.dumps(out))
            print(f"  {key}: L={npep} ss={ss} y={dg} ({len(out)}/100)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {key} FAIL {type(e).__name__}: {str(e)[:50]}", flush=True)
    print(f"done {len(out)}/100", flush=True)


if __name__ == "__main__":
    main()
