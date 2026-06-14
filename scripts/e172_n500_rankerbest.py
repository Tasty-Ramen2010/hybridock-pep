"""E172 — test Ram's hypothesis: score the PoseRanker-BEST pose (not RAPiDock's rank-1) and use N=500 samples
(not 100), to see if a better-selected pose from a deeper sample improves the affinity bands (short/vlong).

For each short/vlong PDBbind complex: RAPiDock N=500 on the pocket → score EVERY generated pose with our
PoseRanker (data/pose_ranker_ml.joblib, predicts native RMSD, lower=better) → pick the ranker-best pose.
Compute the full affinity feature set (geometry + anchor + SS) on BOTH the RAPiDock rank-1 AND the
ranker-best pose, so we can compare rank1 vs ranker-best head-to-head. Writes data/e172_n500_rankerbest.jsonl.
Resumable, auto-deletes poses. GPU; N=500 ~250-350s/complex.
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

import joblib  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybridock_pep.scoring.geometry_features import compute_geometry_features, GEOMETRY_FEATURE_KEYS  # noqa: E402
from hybridock_pep.scoring.anchor_features import compute_anchor_features, ANCHOR_STABLE_KEYS  # noqa: E402
from hybridock_pep.scoring.pose_ranker_ml import compute_features as ranker_feats  # noqa: E402
from Bio.PDB import PDBParser, PPBuilder  # noqa: E402

RAPIDOCK_PY = "/home/igem/miniconda3/envs/rapidock/bin/python3"
RUNNER = ROOT / "src/hybridock_pep/sampling/run_rapidock.py"
RDIR = ROOT / "third_party/RAPiDock"
MODELDIR = RDIR / "train_models/CGTensorProductEquivariantModel"
WORK = ROOT / "runs" / "e172_n500"
OUT = ROOT / "data" / "e172_n500_rankerbest.jsonl"
POS, NEG = set("KR"), set("DE")
_parser = PDBParser(QUIET=True); _ppb = PPBuilder()
_RANKER = joblib.load(ROOT / "data/pose_ranker_ml.joblib")
_PHI, _PSI, _RMODEL = _RANKER["phi_kde"], _RANKER["psi_kde"], _RANKER["model"]


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


def afeat(pose: Path, rec: Path):
    g = compute_geometry_features(pose, rec)
    if g is None:
        return None
    out = {k: float(g[k]) for k in GEOMETRY_FEATURE_KEYS}
    a = compute_anchor_features(pose, rec, hb_count=float(g.get("hb_count", 0.0)))
    if a:
        out.update({k: float(a[k]) for k in ANCHOR_STABLE_KEYS})
    out.update(ss_fracs(pose))
    return out


def ranker_score(pose: Path):
    f = ranker_feats(pose, _PHI, _PSI)
    if f is None:
        return None
    import numpy as np
    return float(_RMODEL.predict(np.array([f]))[0])  # predicted native RMSD; lower=better


def _existing_pids():
    """complexes we ALREADY have real-pose data for (e93 + e154) — exclude to add only NEW data."""
    have = {k.lower() for k in json.loads((ROOT / "data/e93_realpose_results.json").read_text())}
    if (ROOT / "data/e154_realpose_pdbbind.jsonl").exists():
        have |= {json.loads(l)["pdb"].lower() for l in (ROOT / "data/e154_realpose_pdbbind.jsonl").read_text().splitlines()}
    return have


def candidates():
    rows = [json.loads(l) for l in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines()]
    have = _existing_pids()
    out = []
    for r in rows:
        if not (r["length"] <= 8 or r["length"] >= 17):
            continue
        if r["pdb"].lower() in have:            # skip complexes we already have real poses for
            continue
        d = next((Path(p).parent for p in glob.glob(str(ROOT / f"data/drive_pull/pl/P-L/*/{r['pdb']}/{r['pdb']}_pocket.pdb"))), None)
        if d is None:
            continue
        out.append({**r, "dir": str(d)})
    out.sort(key=lambda c: (0 if c["length"] >= 17 else 1, c["length"]))   # vlong first
    return out


def run_one(c):
    wd = WORK / c["pdb"]; wd.mkdir(parents=True, exist_ok=True)
    d = Path(c["dir"]); rec = d / f"{c['pdb']}_protein.pdb"
    pocket = wd / "receptor.pdb"; shutil.copy(d / f"{c['pdb']}_pocket.pdb", pocket)
    raw = wd / "poses"
    cmd = [RAPIDOCK_PY, str(RUNNER), "--peptide", c["seq"], "--receptor", str(pocket.resolve()),
           "--output-dir", str(raw.resolve()), "--n-samples", "250", "--rapidock-dir", str(RDIR.resolve()),
           "--model-dir", str(MODELDIR.resolve()), "--ckpt", "rapidock_local.pt",
           "--scoring-function", "none", "--seed", "42"]
    env = dict(os.environ, PATH="/usr/lib/wsl/lib:" + os.environ.get("PATH", ""))
    subprocess.run(cmd, capture_output=True, text=True, timeout=3000, env=env)
    ranks = sorted(raw.glob("**/rank*.pdb"), key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0))
    if not ranks:
        shutil.rmtree(wd, ignore_errors=True); return None
    # ranker-best = lowest predicted RMSD across all generated poses
    best, best_s = None, 1e9
    for p in ranks:
        s = ranker_score(p)
        if s is not None and s < best_s:
            best_s, best = s, p
    rank1_f = afeat(ranks[0], rec)
    best_f = afeat(best, rec) if best is not None else None
    n_poses = len(ranks)
    shutil.rmtree(wd, ignore_errors=True)
    if rank1_f is None or best_f is None:
        return None
    return {"pdb": c["pdb"], "seq": c["seq"], "y": c["y"], "length": c["length"],
            "n_poses": n_poses, "ranker_best_rmsd_pred": round(best_s, 2),
            "rank1": rank1_f, "ranker_best": best_f}


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["pdb"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    todo = [c for c in candidates() if c["pdb"] not in done]
    print(f"=== E172 N=500 + ranker-best: {len(done)} done, {len(todo)} to do ===", flush=True)
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
            el = time.time() - t0
            print(f"  {n} done  {el/n:.0f}s/complex  {c['pdb']} L={c['length']} "
                  f"npose={rec['n_poses']} bestRMSDpred={rec['ranker_best_rmsd_pred']}", flush=True)
    print(f"=== E172 done: {n} new ===", flush=True)


if __name__ == "__main__":
    main()
