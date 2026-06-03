"""Add per-residue + SS-weighted entropy features to existing production scores.

Reads ``data/training_scores_production.json`` (or any v2 production-pose
scores file with per-pose pose paths), iterates the minimized pose PDBs
already on disk under ``runs/calibration_production/{pdb}/poses_minimized/``,
computes the per-pose entropy sums via ``scoring.per_residue_entropy``,
aggregates top-K by Vina (matching the original aggregation), then writes
an augmented JSON.

Output schema additions per complex:
    s_sc_sum, s_bb_sum, s_ss_weighted  — aggregated over top-K
    ss_loop_frac, ss_helix_frac, ss_sheet_frac — same window

Optionally fits a ridge calibration on the new feature space and reports
in-sample / LOO-CV r alongside the existing baselines.

Usage:
    /path/to/score-env/python scripts/score_per_residue_entropy.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "src"))

from hybridock_pep.scoring.entropy import load_receptor_heavy_atom_coords  # noqa: E402
from hybridock_pep.scoring.per_residue_entropy import compute_entropy_sums  # noqa: E402

_RT_LN10 = 1.364


def augment_one(pdb_id: str, sequence: str, pepset_dir: Path, work_dir: Path,
                top_k: int = 10) -> dict:
    pose_dir = work_dir / pdb_id / "poses_minimized"
    if not pose_dir.exists():
        pose_dir = work_dir / pdb_id / "poses"
    if not pose_dir.exists():
        raise FileNotFoundError(f"No pose directory for {pdb_id}: {pose_dir}")

    pocket_pdb = pepset_dir / pdb_id / f"{pdb_id}_rec_unbound_pocket.pdb"
    if not pocket_pdb.exists():
        pocket_pdb = pepset_dir / pdb_id / f"{pdb_id}_rec_unbound.pdb"
    receptor_coords = load_receptor_heavy_atom_coords(pocket_pdb)

    per_pose: list[dict] = []
    for pose_pdb in sorted(pose_dir.glob("pose_*.pdb")):
        try:
            entry = compute_entropy_sums(pose_pdb, sequence, receptor_coords=receptor_coords)
            entry["pose"] = pose_pdb.name
            per_pose.append(entry)
        except Exception as exc:  # noqa: BLE001
            print(f"  {pdb_id} {pose_pdb.name}: skipped ({exc})")
    if not per_pose:
        raise RuntimeError(f"[{pdb_id}] no successful poses")

    # Sort by n_contact descending (largest interface first); take top-K
    sorted_by_contact = sorted(per_pose, key=lambda r: -r["n_contact"])[:top_k]
    aggregate = {
        "n_contact":     int(statistics.median(p["n_contact"]     for p in sorted_by_contact)),
        "s_sc_sum":      float(statistics.median(p["s_sc_sum"]    for p in sorted_by_contact)),
        "s_bb_sum":      float(statistics.median(p["s_bb_sum"]    for p in sorted_by_contact)),
        "s_ss_weighted": float(statistics.median(p["s_ss_weighted"] for p in sorted_by_contact)),
        "ss_loop_count": int(statistics.median(p["ss_loop_count"]  for p in sorted_by_contact)),
        "ss_helix_count": int(statistics.median(p["ss_helix_count"] for p in sorted_by_contact)),
        "ss_sheet_count": int(statistics.median(p["ss_sheet_count"] for p in sorted_by_contact)),
    }
    return {"aggregate": aggregate, "per_pose": per_pose, "n_poses": len(per_pose)}


def fit_and_report(rows: list[dict], features: list[str]) -> tuple[float, float, float, float]:
    """Return (in_sample_r, in_sample_rmse, loo_r, loo_rmse)."""
    X = np.column_stack([
        [r[f] if not f.startswith("-") else -r[f[1:]] for r in rows]
        for f in features
    ]).astype(float)
    y = np.array([-_RT_LN10 * r["pkd"] for r in rows])
    m = Ridge(alpha=0.1, positive=True).fit(X, y)
    pred_in = m.predict(X)
    r_in = float(pearsonr(y, pred_in).statistic)
    rmse_in = float(np.sqrt(((pred_in - y) ** 2).mean()))
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        preds[te] = Ridge(alpha=0.1, positive=True).fit(X[tr], y[tr]).predict(X[te])
    r_loo = float(pearsonr(y, preds).statistic)
    rmse_loo = float(np.sqrt(((preds - y) ** 2).mean()))
    return r_in, rmse_in, r_loo, rmse_loo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path,
                        default=_REPO / "data/training_scores_production.json")
    parser.add_argument("--training-csv", type=Path,
                        default=_REPO / "data/training_complexes.csv")
    parser.add_argument("--pepset-dir", type=Path, default=_REPO / "datasets/pepset")
    parser.add_argument("--work-dir", type=Path, default=_REPO / "runs/calibration_production")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path,
                        default=_REPO / "data/training_scores_production_entropy.json")
    args = parser.parse_args()

    scores = json.loads(args.scores.read_text())
    rows = list(csv.DictReader(args.training_csv.open()))

    augmented: dict[str, dict] = {}
    table_rows: list[dict] = []
    for r in rows:
        pdb = r["pdb_id"]
        if pdb not in scores:
            print(f"{pdb}: not in production scores, skipping")
            continue
        print(f"\n=== {pdb}  pKd={r['experimental_pkd']}  seq={r['peptide_sequence']} ===")
        aug = augment_one(pdb, r["peptide_sequence"], args.pepset_dir.resolve(),
                          args.work_dir.resolve(), args.top_k)
        aug["pkd"] = float(r["experimental_pkd"])
        aug["seq"] = r["peptide_sequence"]
        aug["vina_score"] = scores[pdb]["vina_score"]
        aug["ad4_score"] = scores[pdb].get("ad4_score", 0.0)
        aug.update(aug["aggregate"])  # flatten for fit_and_report
        augmented[pdb] = aug
        table_rows.append(aug)

    args.output.write_text(json.dumps(augmented, indent=2))
    print(f"\nWrote {args.output}")

    print("\n" + "=" * 90)
    print("Per-complex aggregates (median across top-K by N_contact):")
    print(f"{'pdb':6s} {'pKd':>5s} {'vina':>7s} {'nC':>3s} "
          f"{'s_sc':>6s} {'s_bb':>6s} {'s_ss':>6s} "
          f"{'L/H/E':>10s}")
    for a in table_rows:
        print(f"{[r['pdb_id'] for r in rows if r['pdb_id'] in augmented and augmented[r['pdb_id']] is a][0]:6s} "
              f"{a['pkd']:>5.2f} {a['vina_score']:>+7.2f} {a['n_contact']:>3d} "
              f"{a['s_sc_sum']:>6.2f} {a['s_bb_sum']:>6.2f} {a['s_ss_weighted']:>6.2f} "
              f"{a['ss_loop_count']}/{a['ss_helix_count']}/{a['ss_sheet_count']:>3d}")

    print("\n" + "=" * 90)
    print("Ridge LOO comparison (positive-constrained, λ=0.1):")
    print(f"{'feature set':40s} {'in r':>6s} {'in RMSE':>8s} {'LOO r':>7s} {'LOO RMSE':>9s}")
    feature_sets = [
        ("baseline: vina + (-n_contact)",    ["vina_score", "-n_contact"]),
        ("legacy: vina (only)",               ["vina_score"]),
        ("AA: vina + (-s_sc_sum)",            ["vina_score", "-s_sc_sum"]),
        ("AA: vina + (-s_bb_sum)",            ["vina_score", "-s_bb_sum"]),
        ("AA+SS: vina + (-s_ss_weighted)",    ["vina_score", "-s_ss_weighted"]),
        ("AA: vina + (-s_sc_sum) + (-s_bb_sum)", ["vina_score", "-s_sc_sum", "-s_bb_sum"]),
        ("AA+SS+legacy: vina + (-n_contact) + (-s_ss_weighted)",
                                              ["vina_score", "-n_contact", "-s_ss_weighted"]),
        ("entropy only: (-s_ss_weighted)",    ["-s_ss_weighted"]),
        ("entropy only: (-s_sc_sum)",         ["-s_sc_sum"]),
        ("entropy only: (-n_contact)",        ["-n_contact"]),
    ]
    for name, feats in feature_sets:
        try:
            r_in, rmse_in, r_loo, rmse_loo = fit_and_report(table_rows, feats)
        except Exception as exc:
            print(f"  {name:40s}  ERROR: {exc}")
            continue
        print(f"  {name:40s} {r_in:>+6.3f} {rmse_in:>8.2f} {r_loo:>+7.3f} {rmse_loo:>9.2f}")


if __name__ == "__main__":
    main()
