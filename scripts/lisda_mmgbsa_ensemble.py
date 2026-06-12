"""Top-15 per-pose MM-GBSA ensemble for LISDAELEAIFEADC on PfLDH vs hLDH.

Fixes the n=1 cluster-centroid problem: runs MM-GBSA on the top-15 individual
poses (ranked by hybrid_score, excluding vina=0 failed poses) for each target,
then ensemble-averages to get ΔΔG with an error bar.

Matches the pipeline: same receptor_for_rapidock.pdb, same poses_scored PDBs.
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

from hybridock_pep.scoring.mmgbsa import compute_mmgbsa_single

ROOT = Path("/home/igem/unknown_software")
RUNS = ROOT / "runs" / "liu2019"
TOPK = 15


def top_poses(run_dir: Path, k: int) -> list[tuple[str, float]]:
    rows = list(csv.DictReader((run_dir / "ranked_poses.csv").open()))
    out = []
    for r in rows:
        v = r.get("vina_score")
        if v in (None, "") or float(v) == 0.0:
            continue  # skip failed/clashed poses
        out.append((r["pose_filename"], float(r["hybrid_score"])))
        if len(out) >= k:
            break
    return out


def run_target(name: str) -> dict:
    run_dir = RUNS / name
    receptor = run_dir / "receptor_for_rapidock.pdb"
    posedir = run_dir / "poses_scored"
    poses = top_poses(run_dir, TOPK)
    vals: list[float] = []
    print(f"\n=== {name}: MM-GBSA on top-{len(poses)} poses ===", flush=True)
    for i, (pf, hyb) in enumerate(poses, 1):
        pose_pdb = posedir / pf
        if not pose_pdb.exists():
            print(f"  [{i:2d}] {pf}: MISSING", flush=True)
            continue
        try:
            dg = compute_mmgbsa_single(pose_pdb, receptor)
            vals.append(dg)
            print(f"  [{i:2d}] {pf}: {dg:8.2f}  (hybrid {hyb:.2f})", flush=True)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  [{i:2d}] {pf}: FAILED {type(exc).__name__}: {exc}", flush=True)
    return {
        "name": name,
        "n": len(vals),
        "vals": vals,
        "mean": st.mean(vals) if vals else None,
        "sd": st.pstdev(vals) if len(vals) > 1 else 0.0,
        "best": min(vals) if vals else None,
    }


def main() -> int:
    pf = run_target("lisda_pfldh")
    hl = run_target("lisda_hldh")
    print("\n" + "=" * 56)
    print("MM-GBSA ENSEMBLE SUMMARY (kcal/mol, more negative = stronger)")
    for r in (pf, hl):
        if r["mean"] is not None:
            print(f"  {r['name']:14s} n={r['n']:2d}  mean {r['mean']:8.2f} ± {r['sd']:5.2f}  best {r['best']:8.2f}")
        else:
            print(f"  {r['name']:14s} n=0  ALL FAILED")
    if pf["mean"] is not None and hl["mean"] is not None:
        ddg = pf["mean"] - hl["mean"]
        # pooled SE of the difference
        se = ((pf["sd"] ** 2) / max(pf["n"], 1) + (hl["sd"] ** 2) / max(hl["n"], 1)) ** 0.5
        direction = "PfLDH-selective" if ddg < 0 else "hLDH-selective"
        print(f"\n  ΔΔG (PfLDH - hLDH) = {ddg:+.2f} ± {se:.2f} kcal/mol  -> {direction}")
        print(f"  |ΔΔG|/SE = {abs(ddg) / se:.2f}  (>2 ~ resolved, <1 ~ noise)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
