"""E171 — real-pose data campaign for the DATA-SPARSE bands (short ≤8, vlong ≥17). The deployment model
collapses on short (0.206, only 40 examples) and vlong (−0.034, 37 examples) purely from data sparsity
(crystal-925 short=0.456 with 305 examples). RAPiDock N=100 on PDBbind pockets in those bands, score rank-1
with the FULL deployment feature set this time — geometry + anchor (max_burial/buried_inert/pro_run) + SS
(phi/psi helix/sheet/ppii/turn) — so the rows are complete and never need pose regeneration.

Writes data/e171_realpose_shortvlong.jsonl. Resumable, auto-deletes poses. GPU (RTX 5070), ~60s/complex.
"""
from __future__ import annotations

import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybridock_pep.scoring.geometry_features import compute_geometry_features, GEOMETRY_FEATURE_KEYS  # noqa: E402
from hybridock_pep.scoring.anchor_features import compute_anchor_features, ANCHOR_STABLE_KEYS  # noqa: E402
from Bio.PDB import PDBParser, PPBuilder  # noqa: E402

RAPIDOCK_PY = "/home/igem/miniconda3/envs/rapidock/bin/python3"
RUNNER = ROOT / "src/hybridock_pep/sampling/run_rapidock.py"
RDIR = ROOT / "third_party/RAPiDock"
MODELDIR = RDIR / "train_models/CGTensorProductEquivariantModel"
WORK = ROOT / "runs" / "e171_realpose"
OUT = ROOT / "data" / "e171_realpose_shortvlong.jsonl"
POS, NEG = set("KR"), set("DE")
_parser = PDBParser(QUIET=True); _ppb = PPBuilder()


def ss_fracs(pose: Path):
    try:
        st = _parser.get_structure("p", str(pose))
    except Exception:  # noqa: BLE001
        return {"helix": 0.0, "sheet": 0.0, "ppii": 0.0, "turn": 0.0}
    h = s = p = t = tot = 0
    for ch in st[0]:
        for poly in _ppb.build_peptides(ch):
            for phi, psi in poly.get_phi_psi_list():
                if phi is None or psi is None:
                    continue
                phd, psd = math.degrees(phi), math.degrees(psi); tot += 1
                if -100 <= phd <= -30 and -80 <= psd <= -5:
                    h += 1
                elif -180 <= phd <= -90 and 90 <= psd <= 180:
                    s += 1
                elif -90 <= phd <= -45 and 120 <= psd <= 180:
                    p += 1
                elif 0 <= phd <= 90:
                    t += 1
    if tot == 0:
        return {"helix": 0.0, "sheet": 0.0, "ppii": 0.0, "turn": 0.0}
    return {"helix": h / tot, "sheet": s / tot, "ppii": p / tot, "turn": t / tot}


def candidates():
    rows = [json.loads(l) for l in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines()]
    cands = []
    for r in rows:
        if not (r["length"] <= 8 or r["length"] >= 17):
            continue
        d = next((Path(p).parent for p in glob.glob(str(ROOT / f"data/drive_pull/pl/P-L/*/{r['pdb']}/{r['pdb']}_pocket.pdb"))), None)
        if d is None:
            continue
        q = abs(sum(c in POS for c in r["seq"]) - sum(c in NEG for c in r["seq"]))
        cands.append({**r, "dir": str(d), "q": q})
    # vlong first (rarest, 31), then short
    cands.sort(key=lambda c: (0 if c["length"] >= 17 else 1, c["length"]))
    return cands


def run_one(c):
    wd = WORK / c["pdb"]; wd.mkdir(parents=True, exist_ok=True)
    d = Path(c["dir"]); rec_full = d / f"{c['pdb']}_protein.pdb"
    pocket = wd / "receptor.pdb"; shutil.copy(d / f"{c['pdb']}_pocket.pdb", pocket)
    raw = wd / "poses"
    cmd = [RAPIDOCK_PY, str(RUNNER), "--peptide", c["seq"], "--receptor", str(pocket.resolve()),
           "--output-dir", str(raw.resolve()), "--n-samples", "100", "--rapidock-dir", str(RDIR.resolve()),
           "--model-dir", str(MODELDIR.resolve()), "--ckpt", "rapidock_local.pt",
           "--scoring-function", "none", "--seed", "42"]
    env = dict(os.environ, PATH="/usr/lib/wsl/lib:" + os.environ.get("PATH", ""))
    subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
    ranks = sorted(raw.glob("**/rank*.pdb"), key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0))
    if not ranks:
        shutil.rmtree(wd, ignore_errors=True); return None
    pose = ranks[0]
    g = compute_geometry_features(pose, rec_full)
    if g is None:
        shutil.rmtree(wd, ignore_errors=True); return None
    a = compute_anchor_features(pose, rec_full, hb_count=float(g.get("hb_count", 0.0)))
    ss = ss_fracs(pose)
    shutil.rmtree(wd, ignore_errors=True)
    rank1 = {k: float(g[k]) for k in GEOMETRY_FEATURE_KEYS}
    if a:
        rank1.update({k: float(a[k]) for k in ANCHOR_STABLE_KEYS})
    rank1.update(ss)
    return {"pdb": c["pdb"], "seq": c["seq"], "y": c["y"], "length": c["length"], "q": c["q"], "rank1": rank1}


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["pdb"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    todo = [c for c in candidates() if c["pdb"] not in done]
    print(f"=== E171 short+vlong real-pose: {len(done)} done, {len(todo)} to do ===", flush=True)
    t0 = time.time(); n = 0
    for c in todo:
        try:
            rec = run_one(c)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{c['pdb']}] FAIL {str(exc)[:70]}", flush=True); rec = None
        if rec:
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            n += 1
            if n % 10 == 0:
                print(f"  {n} done  {(time.time()-t0)/n:.0f}s/complex  last={c['pdb']} L={c['length']}", flush=True)
    print(f"=== E171 done: {n} new ===", flush=True)


if __name__ == "__main__":
    main()
