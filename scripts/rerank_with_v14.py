"""Re-rank an existing dock run using a different calibration.

Reads ranked_poses.csv from a previous run, recomputes the hybrid score with
the new calibration, and reports the new top-K vs the old top-K. Does NOT
re-run docking or scoring — only re-aggregates the existing per-pose
features (Vina, entropy, n_contact) under the new weights.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rerank(run_dir: Path, cal_path: Path, top_k: int = 10) -> None:
    cal = json.loads(cal_path.read_text())
    w_vina = float(cal["w_vina"])
    w_ad4 = float(cal.get("w_ad4", 0.0))
    w_contact = float(cal.get("w_contact", 0.0))
    w_s_ss = float(cal.get("w_s_ss_weighted", 0.0))
    intercept = float(cal["intercept"])

    # Need the per-pose s_ss_weighted too — read from training_scores or
    # recompute on the fly from poses_scored/
    # For now, derive from old hybrid: pose.s_ss_weighted = (old_hybrid - old_intercept) / w_s_ss_old
    # Better: parse the production-entropy JSON for per-pose entropy if available.
    ranked_path = run_dir / "ranked_poses.csv"
    if not ranked_path.exists():
        print(f"ERROR: {ranked_path} not found")
        sys.exit(1)
    rows = list(csv.DictReader(ranked_path.open()))

    # The old v1.2 weights were w_vina=0, w_ad4=0, w_contact=0,
    # w_s_ss_weighted=-0.4341, intercept=-3.9462. So:
    #   old_hybrid = entropy_correction + intercept_v12
    #   entropy_correction = w_s_ss_v12 * s_ss_weighted
    #   s_ss_weighted = entropy_correction / w_s_ss_v12
    v12 = json.loads((ROOT / "data" / "calibration_v1_2_production_entropy.json").read_text())
    w_s_ss_v12 = float(v12["w_s_ss_weighted"])
    intercept_v12 = float(v12["intercept"])

    rescored = []
    for r in rows:
        try:
            vina = float(r["vina_score"]) if r["vina_score"] else 0.0
            ad4 = float(r["ad4_score"]) if r["ad4_score"] else 0.0
            entropy_corr = float(r["entropy_correction"])
            n_contact = int(r["n_contact_residues"]) if r["n_contact_residues"] else 0
            # Recover s_ss_weighted (assumes v1.2 calibration was used originally)
            if w_s_ss_v12 != 0:
                s_ss_weighted = entropy_corr / w_s_ss_v12
            else:
                s_ss_weighted = 0.0
        except (ValueError, KeyError):
            continue

        new_hybrid = (w_vina * vina + w_ad4 * ad4 + w_contact * n_contact
                      + w_s_ss * s_ss_weighted + intercept)
        rescored.append({
            "pose": r["pose_filename"],
            "old_rank": int(r["rank"]),
            "old_hybrid": float(r["hybrid_score"]),
            "vina": vina,
            "n_contact": n_contact,
            "s_ss_weighted": round(s_ss_weighted, 2),
            "new_hybrid": new_hybrid,
        })
    rescored.sort(key=lambda x: x["new_hybrid"])
    for new_rank, r in enumerate(rescored, 1):
        r["new_rank"] = new_rank

    print(f"=== Re-rank: {run_dir.name} with {cal_path.name} ===")
    print(f"  w_vina={w_vina}, w_s_ss_weighted={w_s_ss:.3f}, intercept={intercept:.3f}\n")

    print(f"{'NEW':>4} {'POSE':<14} {'NEW_HYB':>8} {'VINA':>7} {'S_ss':>6} {'OLD':>4} {'OLD_HYB':>8}")
    print("-" * 60)
    for r in rescored[:top_k]:
        print(f"{r['new_rank']:>4d} {r['pose']:<14} "
              f"{r['new_hybrid']:>8.2f} {r['vina']:>7.2f} {r['s_ss_weighted']:>6.1f} "
              f"{r['old_rank']:>4d} {r['old_hybrid']:>8.2f}")

    # Find pose_84 in both
    p84 = next((r for r in rescored if r["pose"] == "pose_84.pdb"), None)
    p74 = next((r for r in rescored if r["pose"] == "pose_74.pdb"), None)
    if p84 and p74:
        print(f"\nSpecific check:")
        print(f"  pose_84 (vina=-10.14): old_rank={p84['old_rank']:3d}  →  new_rank={p84['new_rank']:3d}")
        print(f"  pose_74 (vina= -2.04): old_rank={p74['old_rank']:3d}  →  new_rank={p74['new_rank']:3d}")


if __name__ == "__main__":
    rerank(
        Path(sys.argv[1] if len(sys.argv) > 1 else "runs/pfldh_lisdaeleaifeadc_v3"),
        Path(sys.argv[2] if len(sys.argv) > 2 else "data/calibration_v1_4_balanced.json"),
        top_k=int(sys.argv[3]) if len(sys.argv) > 3 else 10,
    )
