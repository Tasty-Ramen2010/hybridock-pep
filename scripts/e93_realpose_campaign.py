"""E93 — the proper real-pose deployment campaign: RAPiDock N=100 on all 65 Kd complexes, then score.

Re-confirms the deployment number LIVE (vs the documented 0.486/0.532). For each crystal-65 Kd complex we
generate 100 RAPiDock-Reloaded poses (the real AI poses a user gets — NOT the crystal), score rank-1 and
top-5-ensemble with the production geometry+ensemble model + length router, and correlate the real-pose
predicted ΔG with experiment. Crash-safe / resumable. Multi-hour GPU campaign.

Phase 1 (this script, --generate): loop 65 complexes, run RAPiDock N=100 → campaign/<pdb>/poses.
Phase 2 (--score): score generated poses, write data/e93_realpose_results.json, report real-pose r.
Run both: default (generate any missing, then score).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CAMP = ROOT / "runs" / "e93_realpose_campaign"
CAMP.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "data" / "e93_realpose_results.json"
RAPIDOCK_PY = "/home/igem/miniconda3/envs/rapidock/bin/python3"
RUNNER = ROOT / "src/hybridock_pep/sampling/run_rapidock.py"
RDIR = ROOT / "third_party/RAPiDock"
MODELDIR = RDIR / "train_models/CGTensorProductEquivariantModel"


def generate(complexes):
    """Phase 1: RAPiDock N=100 per complex (resumable)."""
    for i, r in enumerate(complexes):
        pdb = r["pdb"]
        outdir = CAMP / pdb
        poses = outdir / "poses"
        if poses.exists() and len(list(poses.glob("pose_*.pdb"))) >= 90:
            print(f"  [{i+1}/{len(complexes)}] {pdb}: already have poses, skip", flush=True)
            continue
        outdir.mkdir(parents=True, exist_ok=True)
        # harvest-first: if RAPiDock already wrote ranks (nested), just harvest them, don't regenerate
        raw_existing = outdir / "poses_raw"
        if raw_existing.exists():
            ranks = sorted([p for p in raw_existing.rglob("rank*.pdb")],
                           key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
            if len(ranks) >= 90:
                poses.mkdir(exist_ok=True)
                for j, rp in enumerate(ranks):
                    (poses / f"pose_{j}.pdb").write_text(rp.read_text())
                print(f"  [{i+1}/{len(complexes)}] {pdb}: harvested {len(ranks)} existing poses", flush=True)
                continue
        rec = str(Path(r["pocket_pdb"]).resolve())
        t0 = time.time()
        cmd = [RAPIDOCK_PY, str(RUNNER), "--peptide", r["peptide_seq"], "--receptor", rec,
               "--output-dir", str((outdir / "poses_raw").resolve()), "--n-samples", "100",
               "--rapidock-dir", str(RDIR.resolve()), "--model-dir", str(MODELDIR.resolve()),
               "--ckpt", "rapidock_local.pt", "--scoring-function", "none", "--seed", "42"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            # rename rank*.pdb -> pose_*.pdb into poses/ (runner nests under poses_raw/poses_raw/)
            raw = outdir / "poses_raw"
            poses.mkdir(exist_ok=True)
            ranks = sorted([p for p in raw.rglob("rank*.pdb")],
                           key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0)) if raw.exists() else []
            for j, rp in enumerate(ranks):
                (poses / f"pose_{j}.pdb").write_text(rp.read_text())
            n = len(list(poses.glob("pose_*.pdb")))
            print(f"  [{i+1}/{len(complexes)}] {pdb}: {n} poses ({time.time()-t0:.0f}s)", flush=True)
            if n == 0:
                print(f"      stderr: {res.stderr[-300:]}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"  [{i+1}/{len(complexes)}] {pdb}: TIMEOUT (>30min)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i+1}/{len(complexes)}] {pdb}: FAIL {str(e)[:80]}", flush=True)


def score(complexes):
    """Phase 2: score generated poses (rank-1 + top-5 mean) with production geometry features."""
    from hybridock_pep.scoring.geometry_features import compute_geometry_features
    from scipy.stats import pearsonr
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for i, r in enumerate(complexes):
        pdb = r["pdb"]
        if pdb in results:
            continue
        poses = sorted((CAMP / pdb / "poses").glob("pose_*.pdb"),
                       key=lambda p: int(p.stem.split("_")[1]))
        if not poses:
            continue
        rec = Path(r["pocket_pdb"]).resolve()
        feats_list = []
        for p in poses[:25]:  # top-25 by RAPiDock rank
            try:
                f = compute_geometry_features(p, rec)
                if f:
                    feats_list.append(f)
            except Exception:  # noqa: BLE001
                pass
        if not feats_list:
            continue
        # rank-1 features and top-5 mean features
        keys = [k for k in feats_list[0] if isinstance(feats_list[0][k], (int, float))]
        rank1 = {k: feats_list[0][k] for k in keys}
        top5 = {k: float(np.mean([fl[k] for fl in feats_list[:5]])) for k in keys}
        results[pdb] = dict(y=r["dg_exp"], seq=r["peptide_seq"], n_poses=len(poses),
                            rank1=rank1, top5=top5)
        OUT.write_text(json.dumps(results))
        print(f"  scored {pdb} ({len(feats_list)} poses featurized)", flush=True)

    # correlate: fit production geometry model LOO on rank-1 and top-5 features vs crystal
    rows = list(results.values())
    if len(rows) >= 20:
        report(rows)


def report(rows):
    from scipy.stats import pearsonr
    PROD = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb",
            "sasa_sb", "arom_cc", "hb_count", "mj_contact", "mean_burial", "rg_per_L"]
    y = np.array([r["y"] for r in rows])

    def loo(feat_key, cols):
        X = np.array([[rr[feat_key].get(c, np.nan) for c in cols] for rr in rows], float)
        ok = ~np.isnan(X).any(1)
        Xx, yy = X[ok], y[ok]
        pred = np.zeros(len(Xx))
        for i in range(len(Xx)):
            tr = np.arange(len(Xx)) != i
            mu, sd = Xx[tr].mean(0), Xx[tr].std(0) + 1e-9
            A = np.column_stack([np.ones(tr.sum()), (Xx[tr] - mu) / sd])
            R = np.eye(A.shape[1]); R[0, 0] = 0
            w = np.linalg.solve(A.T @ A + R, A.T @ yy[tr])
            pred[i] = np.r_[1.0, (Xx[i] - mu) / sd] @ w
        return pearsonr(pred, yy)[0], ok.sum()
    avail = [c for c in PROD if c in rows[0]["rank1"]]
    print(f"\n=== E93 REAL-POSE deployment r (n={len(rows)} Kd complexes, real RAPiDock poses) ===")
    r1, n1 = loo("rank1", avail)
    r5, n5 = loo("top5", avail)
    print(f"  rank-1 real pose:     r = {r1:+.3f}  (n={n1})")
    print(f"  top-5 ensemble pose:  r = {r5:+.3f}  (n={n5})")
    print(f"  (vs crystal-pose benchmark ~0.54; documented real-pose 0.486/0.532)")


def main():
    b = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    print(f"=== E93 real-pose campaign ({len(b)} Kd complexes), mode={mode} ===", flush=True)
    if mode in ("both", "--generate", "generate"):
        generate(b)
    if mode in ("both", "--score", "score"):
        score(b)


if __name__ == "__main__":
    main()
