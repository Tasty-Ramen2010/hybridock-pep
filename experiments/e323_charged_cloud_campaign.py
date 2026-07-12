"""E323 — charged pose-cloud GPU campaign to power up N2 (ensemble ⟨V_elec⟩ over the generative cloud).

N2 (E318) was the first crack in the single-structure charged wall (⟨V_elec⟩ over RAPiDock's 100-pose cloud ~
charged residual r=−0.37) but underpowered (n=24). This campaign generates the cloud for EVERY charged PDBbind
complex (417 with pockets on disk), and — unlike E154 which discards poses — computes the electrostatic ENSEMBLE
statistics ⟨V_elec⟩ and Var(V_elec) over the cloud INLINE before deleting the poses (so disk stays flat).

Per charged complex (|net q|≥2), prioritised smallest-pocket-first for speed (~60-90 s each on RTX 5070):
  RAPiDock N=100 → pocket-receptor formal-charge V_elec per pose → {mean_ve, var_ve} + geometry rank1/top5 + y, q.
Appends to data/e323_charged_clouds.jsonl (resumable). Then rerun N2 on this set (n up to ~417).

Run (background):  OMP_NUM_THREADS=1 nohup /home/igem/miniconda3/envs/score-env/bin/python \
                     experiments/e323_charged_cloud_campaign.py > logs/e323_charged_clouds.log 2>&1 &
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybridock_pep.scoring.geometry_features import compute_geometry_features  # noqa: E402
from hybridock_pep.scoring.interaction_map import _formal_charge_atoms  # noqa: E402

RAPIDOCK_PY = "/home/igem/miniconda3/envs/rapidock/bin/python3"
RUNNER = ROOT / "src/hybridock_pep/sampling/run_rapidock.py"
RDIR = ROOT / "third_party/RAPiDock"
MODELDIR = RDIR / "train_models/CGTensorProductEquivariantModel"
WORK = ROOT / "runs" / "e323_charged_clouds"
OUT = ROOT / "data" / "e323_charged_clouds.jsonl"
POS, NEG = set("KR"), set("DE")
GEOM = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]


def band(L: int) -> int:
    return 0 if L <= 8 else 1 if L <= 12 else 2 if L <= 16 else 3


def candidates() -> list[dict]:
    rows = [json.loads(l) for l in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines()]
    cands = []
    for r in rows:
        q = abs(sum(c in POS for c in r["seq"]) - sum(c in NEG for c in r["seq"]))
        if q < 2:  # charged only
            continue
        d = next((Path(p).parent for p in
                  glob.glob(str(ROOT / f"data/drive_pull/pl/P-L/*/{r['pdb']}/{r['pdb']}_pocket.pdb"))), None)
        if d is None:
            continue
        nat = sum(1 for ln in (d / f"{r['pdb']}_protein.pdb").read_text().splitlines() if ln.startswith("ATOM"))
        cands.append({"pdb": r["pdb"], "seq": r["seq"], "y": r["y"], "length": r["length"],
                      "q": q, "nat": nat, "dir": str(d)})
    cands.sort(key=lambda c: (band(c["length"]), c["nat"]))  # smallest pocket first within length band
    return cands


def velec(pep_charges, rec_charges) -> float:
    e = 0.0
    for qp, xp in pep_charges:
        for qr, xr in rec_charges:
            r = float(np.linalg.norm(xp - xr))
            if r >= 1.0:
                e += qp * qr / r
    return e


def run_one(c: dict) -> dict | None:
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
    ranks = sorted(raw.glob("**/rank*.pdb"),
                   key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0))
    if not ranks:
        shutil.rmtree(wd, ignore_errors=True)
        return None
    rec_charges = _formal_charge_atoms(pocket)              # ensemble electrostatics on the pocket receptor
    ve = [velec(_formal_charge_atoms(p), rec_charges) for p in ranks]
    ve = np.array([v for v in ve if v == v], dtype=float)
    # geometry features (E93 format) on the full protein
    f1 = compute_geometry_features(ranks[0], rec_full)
    rank1 = {k: float(f1[k]) for k in GEOM} if f1 else None
    top5 = []
    for p in ranks[:5]:
        f = compute_geometry_features(p, rec_full)
        if f:
            top5.append({k: float(f[k]) for k in GEOM})
    shutil.rmtree(wd, ignore_errors=True)                  # free disk
    if rank1 is None or not top5 or ve.size < 50:
        return None
    return {"pdb": c["pdb"], "seq": c["seq"], "y": c["y"], "length": c["length"], "q": c["q"],
            "n_poses": len(ranks), "mean_ve": float(ve.mean()), "var_ve": float(ve.var()),
            "std_ve": float(ve.std()), "rank1": rank1, "top5": top5}


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    done = {json.loads(l)["pdb"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    cands = candidates()
    todo = [c for c in cands if c["pdb"] not in done]
    print(f"=== E323 charged-cloud campaign: {len(done)} done, {len(todo)} charged to do "
          f"(smallest-pocket-first, ~60-90s each) ===", flush=True)
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
                el = time.time() - t0
                print(f"  {n} new ({len(done)+n} total)  {el/n:.0f}s/complex  "
                      f"last={c['pdb']} q={c['q']} L={c['length']} mean_ve={rec['mean_ve']:.2f}", flush=True)
    print(f"=== E323 complete: {n} new charged clouds ===", flush=True)


if __name__ == "__main__":
    main()
