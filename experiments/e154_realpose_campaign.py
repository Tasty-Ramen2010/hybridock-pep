"""E154 — real-pose training-data campaign: RAPiDock N=100 on PDBbind complexes to expand the real-pose
training set past 156 and push the deployment model past r=0.55 (the AI-haircut fix, E152).

For each PDBbind complex (prioritised: charged |q|≥2 first, then length-balanced, smallest pocket first for
speed): RAPiDock N=100 on the pocket (site-focused, ~62 s/complex), score rank-1 + top-5 poses with
geometry_features against the full protein, append {pdb, seq, y, length, rank1, top5} to
data/e154_realpose_pdbbind.jsonl (e93 format). Resumable; deletes pose PDBs after scoring (disk).

Runs in score-env (geometry_features); calls RAPiDock via the rapidock-env python subprocess. GPU ~4 GB.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybridock_pep.scoring.geometry_features import compute_geometry_features  # noqa: E402

RAPIDOCK_PY = "/home/igem/miniconda3/envs/rapidock/bin/python3"
RUNNER = ROOT / "src/hybridock_pep/sampling/run_rapidock.py"
RDIR = ROOT / "third_party/RAPiDock"
MODELDIR = RDIR / "train_models/CGTensorProductEquivariantModel"
WORK = ROOT / "runs" / "e154_realpose"
OUT = ROOT / "data" / "e154_realpose_pdbbind.jsonl"
POS, NEG = set("KR"), set("DE")
GEOM = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]


def band(L):
    return 0 if L <= 8 else 1 if L <= 12 else 2 if L <= 16 else 3


def candidates():
    rows = [json.loads(l) for l in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines()]
    cands = []
    for r in rows:
        d = next((Path(p).parent for p in glob.glob(str(ROOT / f"data/drive_pull/pl/P-L/*/{r['pdb']}/{r['pdb']}_pocket.pdb"))), None)
        if d is None:
            continue
        nat = sum(1 for ln in (d / f"{r['pdb']}_protein.pdb").read_text().splitlines() if ln.startswith("ATOM"))
        q = abs(sum(c in POS for c in r["seq"]) - sum(c in NEG for c in r["seq"]))
        cands.append({"pdb": r["pdb"], "seq": r["seq"], "y": r["y"], "length": r["length"],
                      "q": q, "nat": nat, "dir": str(d)})
    # priority: charged first, then round-robin length bands, smallest receptor first
    cands.sort(key=lambda c: (0 if c["q"] >= 2 else 1, band(c["length"]), c["nat"]))
    return cands


def score_pose(pose: Path, rec: Path):
    f = compute_geometry_features(pose, rec)
    return {k: float(f[k]) for k in GEOM} if f else None


def run_one(c):
    wd = WORK / c["pdb"]
    wd.mkdir(parents=True, exist_ok=True)
    d = Path(c["dir"])
    rec_full = d / f"{c['pdb']}_protein.pdb"
    pocket = wd / "receptor.pdb"
    shutil.copy(d / f"{c['pdb']}_pocket.pdb", pocket)
    raw = wd / "poses_raw"
    cmd = [RAPIDOCK_PY, str(RUNNER), "--peptide", c["seq"], "--receptor", str(pocket.resolve()),
           "--output-dir", str(raw.resolve()), "--n-samples", "100", "--rapidock-dir", str(RDIR.resolve()),
           "--model-dir", str(MODELDIR.resolve()), "--ckpt", "rapidock_local.pt",
           "--scoring-function", "none", "--seed", "42"]
    env = dict(os.environ, PATH="/usr/lib/wsl/lib:" + os.environ.get("PATH", ""))
    subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
    ranks = sorted(raw.glob("**/rank*.pdb"), key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0))
    if not ranks:
        shutil.rmtree(wd, ignore_errors=True)
        return None
    rank1 = score_pose(ranks[0], rec_full)
    top5 = [s for p in ranks[:5] if (s := score_pose(p, rec_full))]
    shutil.rmtree(wd, ignore_errors=True)   # free disk
    if rank1 is None or not top5:
        return None
    return {"pdb": c["pdb"], "seq": c["seq"], "y": c["y"], "length": c["length"], "q": c["q"],
            "n_poses": len(ranks), "rank1": rank1, "top5": top5}


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["pdb"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    cands = candidates()
    todo = [c for c in cands if c["pdb"] not in done]
    print(f"=== E154 real-pose campaign: {len(done)} done, {len(todo)} to do "
          f"(charged-first, ~62s each) ===", flush=True)
    t0 = time.time()
    n = 0
    for c in todo:
        try:
            rec = run_one(c)
        except subprocess.TimeoutExpired:
            rec = None
        except Exception as exc:  # noqa: BLE001
            print(f"  [{c['pdb']}] FAIL {str(exc)[:80]}", flush=True)
            rec = None
        if rec:
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            n += 1
            if n % 10 == 0:
                el = time.time() - t0
                print(f"  {n} done ({len(done)+n} total)  {el/n:.0f}s/complex  "
                      f"last={c['pdb']} q={c['q']} L={c['length']}", flush=True)
    print(f"=== E154 complete: {n} new real-pose complexes ===", flush=True)


if __name__ == "__main__":
    main()
