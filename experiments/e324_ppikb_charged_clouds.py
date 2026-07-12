"""E324 — charged PPIKB pose-cloud campaign (the "other ~2000") to further power up N2.

PPIKB has no local pockets (unlike PDBbind), but it has PDB codes and a local RCSB cache (data/rcsb_full) +
the proven E212 chain-split (seq-match the peptide chain, receptor = the rest). This campaign, for each charged
PPIKB Kd complex (|net q|>=2): fetch structure → split peptide/receptor → crop a 12 A pocket around the crystal
peptide → RAPiDock N=100 → inline ⟨V_elec⟩/Var over the cloud + geometry → append to the SAME N2 dataset
(data/e323_charged_clouds.jsonl, tagged source="ppikb"). Resumable.

MUST run AFTER e323 (single GPU, sequential per CLAUDE.md). Prep (fetch/split/crop) is CPU+network and safe to
test while e323 runs; only the RAPiDock call needs the GPU.

Prep-test:  python experiments/e324_ppikb_charged_clouds.py --prep-test 5
Full (chained after e323):  OMP_NUM_THREADS=1 python experiments/e324_ppikb_charged_clouds.py
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from Bio.PDB import PDBIO, PDBParser, Select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hybridock_pep.scoring.geometry_features import compute_geometry_features  # noqa: E402
from hybridock_pep.scoring.interaction_map import _formal_charge_atoms  # noqa: E402
from e323_charged_cloud_campaign import GEOM, velec  # noqa: E402  (reuse identical machinery)

PDBDIR = ROOT / "data" / "rcsb_full"
T2O = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
       "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
       "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def _fetch(pdb: str) -> Path | None:
    """Local RCSB cache hit, else download from RCSB (same behaviour as e180.fetch)."""
    f = PDBDIR / f"{pdb}.pdb"
    if f.exists() and f.stat().st_size > 0:
        return f
    try:
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb.upper()}.pdb", f)
        return f if f.exists() and f.stat().st_size > 0 else None
    except Exception:  # noqa: BLE001
        return None
WORK = ROOT / "runs" / "e324_ppikb_clouds"
OUT = ROOT / "data" / "e323_charged_clouds.jsonl"       # same N2 dataset
RAPIDOCK_PY = "/home/igem/miniconda3/envs/rapidock/bin/python3"
RUNNER = ROOT / "src/hybridock_pep/sampling/run_rapidock.py"
RDIR = ROOT / "third_party/RAPiDock"
MODELDIR = RDIR / "train_models/CGTensorProductEquivariantModel"
_parser = PDBParser(QUIET=True)


class ChainSel(Select):
    def __init__(self, ch): self.ch = ch
    def accept_chain(self, c): return c.id == self.ch
    def accept_residue(self, r): return r.id[0] == " "


class PocketSel(Select):
    """Receptor residues (any chain != peptide) with an atom within `cut` A of the peptide chain."""
    def __init__(self, pep_ch, keep_ids): self.ch = pep_ch; self.keep = keep_ids
    def accept_chain(self, c): return c.id != self.ch
    def accept_residue(self, r): return r.id[0] == " " and (r.get_parent().id, r.id[1]) in self.keep


def chain_seq(st, ch):
    return "".join(T2O.get(r.resname, "") for r in st[0][ch] if r.id[0] == " ") if ch in st[0] else ""


def prep(pdb: str, want: str) -> Path | None:
    """Fetch + split; write a 12 A pocket receptor. Returns the pocket path or None."""
    f = _fetch(pdb)
    if f is None:
        return None
    try:
        st = _parser.get_structure(pdb, str(f))
    except Exception:  # noqa: BLE001
        return None
    want = want.upper(); L = len(want); pep_ch = None
    for ch in st[0]:
        seq = chain_seq(st, ch.id); nstd = sum(1 for r in ch if r.id[0] == " ")
        if (2 <= nstd <= 60) and (want in seq or (seq and seq in want)) and abs(len(seq) - L) <= max(3, 0.4 * L):
            pep_ch = ch.id; break
    if pep_ch is None:
        return None
    pep_atoms = np.array([a.coord for r in st[0][pep_ch] if r.id[0] == " " for a in r])
    if pep_atoms.size == 0:
        return None
    keep = set()
    for ch in st[0]:
        if ch.id == pep_ch:
            continue
        for r in ch:
            if r.id[0] != " ":
                continue
            rc = np.array([a.coord for a in r])
            if rc.size and np.min(np.linalg.norm(pep_atoms[:, None, :] - rc[None, :, :], axis=2)) <= 12.0:
                keep.add((ch.id, r.id[1]))
    if len(keep) < 8:
        return None
    wd = WORK / pdb; wd.mkdir(parents=True, exist_ok=True)
    io = PDBIO(); io.set_structure(st)
    pocket = wd / "receptor.pdb"
    io.save(str(pocket), PocketSel(pep_ch, keep))
    return pocket


def candidates() -> list[dict]:
    rows = [json.loads(l) for l in open(ROOT / "data/ppikb_features.jsonl")]
    ours = {json.loads(l)["pdb"].lower() for l in open(ROOT / "data/pdbbind_peptides.jsonl")}
    pool = [r for r in rows if abs(r.get("net_charge", 0)) >= 2 and r.get("pdb")
            and str(r["aff_type"]).lower() in ("kd", "pkd") and -18 < r["y"] < -2
            and r["pdb"].lower() not in ours and 4 <= r["length"] <= 30]
    pool.sort(key=lambda r: r["length"])   # shorter first (faster)
    return pool


def run_one(c: dict) -> dict | None:
    pocket = prep(c["pdb"], c["seq"])
    if pocket is None:
        return None
    wd = pocket.parent
    raw = wd / "poses_raw"
    cmd = [RAPIDOCK_PY, str(RUNNER), "--peptide", c["seq"], "--receptor", str(pocket.resolve()),
           "--output-dir", str(raw.resolve()), "--n-samples", "100", "--rapidock-dir", str(RDIR.resolve()),
           "--model-dir", str(MODELDIR.resolve()), "--ckpt", "rapidock_local.pt",
           "--scoring-function", "none", "--seed", "42"]
    env = dict(os.environ, PATH="/usr/lib/wsl/lib:" + os.environ.get("PATH", ""))
    subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
    ranks = sorted(raw.glob("**/rank*.pdb"),
                   key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0))
    if not ranks:
        shutil.rmtree(wd, ignore_errors=True)
        return None
    rec_charges = _formal_charge_atoms(pocket)
    ve = np.array([v for v in (velec(_formal_charge_atoms(p), rec_charges) for p in ranks) if v == v])
    f1 = compute_geometry_features(ranks[0], pocket)
    rank1 = {k: float(f1[k]) for k in GEOM} if f1 else None
    top5 = [d for p in ranks[:5] if (f := compute_geometry_features(p, pocket)) and (d := {k: float(f[k]) for k in GEOM})]
    shutil.rmtree(wd, ignore_errors=True)
    if rank1 is None or not top5 or ve.size < 50:
        return None
    return {"pdb": c["pdb"], "seq": c["seq"], "y": c["y"], "length": c["length"],
            "q": abs(c["net_charge"]), "source": "ppikb", "n_poses": len(ranks),
            "mean_ve": float(ve.mean()), "var_ve": float(ve.var()), "std_ve": float(ve.std()),
            "rank1": rank1, "top5": top5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-test", type=int, default=0, help="test fetch/split/crop on N candidates (no GPU)")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    cands = candidates()
    if args.prep_test:
        ok = 0
        for c in cands[:args.prep_test * 4]:
            p = prep(c["pdb"], c["seq"])
            status = "OK" if p else "skip"
            if p:
                ok += 1
                print(f"  {c['pdb']} L={c['length']} q={c['net_charge']:+d} -> pocket "
                      f"{sum(1 for l in p.read_text().splitlines() if l.startswith('ATOM'))} atoms")
                shutil.rmtree(p.parent, ignore_errors=True)
            if ok >= args.prep_test:
                break
        print(f"prep-test: {ok}/{args.prep_test} charged PPIKB complexes prepped from {len(cands)} candidates")
        return
    done = {json.loads(l).get("pdb") for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    todo = [c for c in cands if c["pdb"] not in done]
    print(f"=== E324 charged PPIKB clouds: {len(todo)} to do (pool {len(cands)}) ===", flush=True)
    t0, n = time.time(), 0
    for c in todo:
        try:
            rec = run_one(c)
        except subprocess.TimeoutExpired:
            rec = None
        except Exception as exc:  # noqa: BLE001
            print(f"  [{c['pdb']}] FAIL {str(exc)[:100]}", flush=True)
            rec = None
        if rec:
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            n += 1
            if n % 5 == 0:
                print(f"  {n} new  {(time.time()-t0)/n:.0f}s/complex  last={c['pdb']}", flush=True)
    print(f"=== E324 complete: {n} new charged PPIKB clouds ===", flush=True)


if __name__ == "__main__":
    main()
