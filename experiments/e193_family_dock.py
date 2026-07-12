"""E193 — dock PPIKB selectivity families into a COMMON receptor frame, then measure within-family
selectivity tau on CONSISTENTLY-POSED structures (removes the cross-crystal artifact that killed E190).

For each branch family (>=4 peptides, >=2 kcal spread, has PDB_IDs): pick a representative receptor (the
member whose PDB has the most receptor atoms), extract its receptor chains ONCE, then RAPiDock-dock EVERY
family peptide's sequence into that SAME receptor → rank-1 pose → ProtDCal-3D + geometry features. Within
each family the receptor frame is identical, so descriptor differences reflect the PEPTIDE, not crystal
artifacts. Writes data/e193_family_dock.jsonl (resumable). GPU; runs AFTER the short campaign.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import e179_protdcal_3d as e179  # noqa: E402
from hybridock_pep.scoring.geometry_features import compute_geometry_features, GEOMETRY_FEATURE_KEYS  # noqa: E402

RAPIDOCK_PY = "/home/igem/miniconda3/envs/rapidock/bin/python3"
RUNNER = ROOT / "src/hybridock_pep/sampling/run_rapidock.py"
RDIR = ROOT / "third_party/RAPiDock"
MODELDIR = RDIR / "train_models/CGTensorProductEquivariantModel"
WORK = ROOT / "runs" / "e193_family"
OUT = ROOT / "data" / "e193_family_dock.jsonl"
NSAMPLES = "30"  # fewer per peptide — many peptides per family, want breadth not depth
_parser = PDBParser(QUIET=True)
T2O = e179.T2O
PDBDIR = ROOT / "data" / "rcsb_full"


class RecSel(Select):
    def __init__(self, pep_ch):
        self.pep = pep_ch

    def accept_chain(self, c):
        return c.id != self.pep

    def accept_residue(self, r):
        return r.id[0] == " "


def families():
    rows = [json.loads(l) for l in open(ROOT / "data/ppikb_clean.jsonl")]
    fam = defaultdict(list)
    for r in rows:
        if r["pdb"]:
            fam[r["protein_seq"][:50]].append(r)
    out = []
    for k, v in fam.items():
        seqs = {x["seq"]: x for x in v}
        if len(seqs) >= 4 and (max(x["y"] for x in v) - min(x["y"] for x in v)) >= 2.0:
            out.append((k, list(seqs.values())))
    return out


def get_receptor(members):
    """pick the member whose PDB yields the largest receptor; extract receptor chains once."""
    best = None
    for m in sorted(members, key=lambda x: -len(x["seq"])):  # longer peptide often = better-resolved complex
        f = PDBDIR / f"{m['pdb']}.pdb"
        if not f.exists():
            try:
                import urllib.request
                urllib.request.urlretrieve(f"https://files.rcsb.org/download/{m['pdb'].upper()}.pdb", f)
            except Exception:  # noqa: BLE001
                continue
        try:
            st = _parser.get_structure(m["pdb"], str(f))
        except Exception:  # noqa: BLE001
            continue
        # peptide chain = the one matching the member seq; receptor = the rest
        want = m["seq"].upper(); pep_ch = None
        for ch in st[0]:
            seq = "".join(T2O.get(r.resname, "") for r in ch if r.id[0] == " ")
            if want in seq or (seq and seq in want):
                pep_ch = ch.id; break
        if pep_ch is None:
            continue
        nrec = sum(1 for ch in st[0] if ch.id != pep_ch for r in ch if r.id[0] == " ")
        if best is None or nrec > best[2]:
            best = (st, pep_ch, nrec, m["pdb"])
    return best


def dock_score(seq, rec_pdb, tag):
    wd = WORK / tag; wd.mkdir(parents=True, exist_ok=True)
    raw = wd / "poses"
    cmd = [RAPIDOCK_PY, str(RUNNER), "--peptide", seq, "--receptor", str(rec_pdb.resolve()),
           "--output-dir", str(raw.resolve()), "--n-samples", NSAMPLES, "--rapidock-dir", str(RDIR.resolve()),
           "--model-dir", str(MODELDIR.resolve()), "--ckpt", "rapidock_local.pt",
           "--scoring-function", "none", "--seed", "42"]
    env = dict(os.environ, PATH="/usr/lib/wsl/lib:" + os.environ.get("PATH", ""))
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=1200, env=env)
    except Exception:  # noqa: BLE001
        shutil.rmtree(wd, ignore_errors=True); return None
    ranks = sorted(raw.glob("**/rank*.pdb"), key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
    if not ranks:
        shutil.rmtree(wd, ignore_errors=True); return None
    pose = ranks[0]
    res = e179.residue_seq_and_coords(pose)
    d3 = e179.descriptors(res, 6.0, 3) if res else None
    g = compute_geometry_features(pose, rec_pdb)
    geo = [float(g.get(k, 0.0)) for k in GEOMETRY_FEATURE_KEYS] if g else None
    shutil.rmtree(wd, ignore_errors=True)
    if d3 is None or geo is None:
        return None
    return {"desc3d": d3, "geo": geo}


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    fams = families()
    done = {json.loads(l)["fam"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    print(f"=== E193 family common-frame dock: {len(fams)} families, {len(done)} done ===", flush=True)
    t0 = time.time()
    for fi, (k, members) in enumerate(fams):
        if k in done:
            continue
        rec = get_receptor(members)
        if rec is None:
            continue
        st, pep_ch, _, rec_pdb_id = rec
        io = PDBIO(); io.set_structure(st)
        recf = WORK / f"receptor_{fi}.pdb"; io.save(str(recf), RecSel(pep_ch))
        scored = []
        for j, m in enumerate(members):
            r = dock_score(m["seq"], recf, f"f{fi}_p{j}")
            if r:
                scored.append({"seq": m["seq"], "y": m["y"], "net_charge": m["net_charge"], **r})
        recf.unlink(missing_ok=True)
        if len(scored) >= 4:
            with open(OUT, "a") as fh:
                fh.write(json.dumps({"fam": k, "rec_pdb": rec_pdb_id, "n": len(scored), "members": scored}) + "\n")
            el = time.time() - t0
            print(f"  fam {fi} ({rec_pdb_id}): {len(scored)} peptides docked  [{el/60:.1f}min elapsed]", flush=True)
    print("=== E193 done ===", flush=True)


if __name__ == "__main__":
    main()
