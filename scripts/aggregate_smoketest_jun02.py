"""Aggregate 17-complex smoke test → predicted ΔG vs experimental pKd
+ Cα RMSD-to-crystal where available.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "smoketest_jun02"
PLAN = RUN_DIR / "run_plan.csv"


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    s = np.eye(3)
    s[2, 2] = np.sign(np.linalg.det(u @ vt))
    return float(np.sqrt(((a @ u @ s @ vt - b) ** 2).sum(axis=1).mean()))


def ca_coords(pdb: Path, expected_len: int | None = None) -> np.ndarray:
    out = []
    seen = set()
    for line in pdb.read_text().splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        try:
            r = int(line[22:26].strip())
        except ValueError:
            continue
        if (r, line[21]) in seen:
            continue
        seen.add((r, line[21]))
        try:
            out.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
    a = np.array(out) if out else np.zeros((0, 3))
    if expected_len and len(a) != expected_len:
        return a[:expected_len]  # trim to match crystal length when peptides truncated
    return a


def best_top1_from_run(out_dir: Path) -> tuple[float, Path] | None:
    csv_path = out_dir / "ranked_poses.csv"
    if not csv_path.exists():
        return None
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    # Sort by hybrid_score (lowest = best)
    rows.sort(key=lambda r: float(r["hybrid_score"]) if r.get("hybrid_score") else 0.0)
    best = rows[0]
    best_pdb = out_dir / "poses_scored" / best["pose_file"] if "pose_file" in best else None
    if not best_pdb or not best_pdb.exists():
        best_pdb = out_dir / "best_pose.pdb"
    return float(best["hybrid_score"]), best_pdb


def main() -> None:
    with PLAN.open() as f:
        plan = {r["pdb_id"]: r for r in csv.DictReader(f)}

    results = []
    for pdb_id, p in plan.items():
        out_dir = RUN_DIR / pdb_id
        crystal_pep = ROOT / p["crystal_peptide_pdb"]
        pkd_str = p["pkd"]
        try:
            pkd = float(pkd_str) if pkd_str not in ("—", "") else None
        except ValueError:
            pkd = None
        dg_exp = -1.3633 * pkd if pkd is not None else None

        best = best_top1_from_run(out_dir)
        if best is None:
            results.append({
                "pdb": pdb_id, "set": p["set"], "peptide": p["peptide"],
                "pkd": pkd, "dg_exp": dg_exp,
                "dg_pred": None, "rmsd_to_crystal": None,
                "n_pep_residues": int(p["n_pep_residues"]),
                "status": "no_output",
            })
            continue
        dg_pred, best_pdb = best

        rmsd = None
        if crystal_pep.exists() and best_pdb and best_pdb.exists():
            crys = ca_coords(crystal_pep)
            pose = ca_coords(best_pdb)
            if crys.size and pose.size:
                n = min(len(crys), len(pose))
                if n >= 3:
                    rmsd = kabsch_rmsd(crys[:n], pose[:n])

        results.append({
            "pdb": pdb_id, "set": p["set"], "peptide": p["peptide"],
            "pkd": pkd, "dg_exp": dg_exp,
            "dg_pred": dg_pred,
            "rmsd_to_crystal": rmsd,
            "n_pep_residues": int(p["n_pep_residues"]),
            "status": "ok",
        })

    # Compute headline stats
    def stats(rows, key_pred="dg_pred", key_exp="dg_exp"):
        pairs = [(r[key_pred], r[key_exp]) for r in rows
                 if r[key_pred] is not None and r[key_exp] is not None]
        if len(pairs) < 3:
            return None, None, len(pairs)
        pred = np.array([p[0] for p in pairs])
        exp = np.array([p[1] for p in pairs])
        r = float(np.corrcoef(pred, exp)[0, 1])
        rmse = float(math.sqrt(((pred - exp) ** 2).mean()))
        return r, rmse, len(pairs)

    print("=" * 76)
    print("HybriDock-Pep Full-Pipeline Smoke Test — 17 complexes — 2026-06-03")
    print("=" * 76)
    print(f"\n{'PDB':<7}{'SET':<14}{'PEP':<22}{'pKd':>6}{'ΔGexp':>8}{'ΔGpred':>9}{'RMSD':>8}{'len':>5}")
    print("-" * 76)
    for r in results:
        dg_exp_s = f"{r['dg_exp']:.2f}" if r['dg_exp'] is not None else "—"
        dg_pred_s = f"{r['dg_pred']:.2f}" if r['dg_pred'] is not None else "FAIL"
        rmsd_s = f"{r['rmsd_to_crystal']:.2f}" if r['rmsd_to_crystal'] is not None else "—"
        pkd_s = f"{r['pkd']:.2f}" if r['pkd'] is not None else "—"
        print(f"{r['pdb']:<7}{r['set']:<14}{r['peptide'][:20]:<22}"
              f"{pkd_s:>6}{dg_exp_s:>8}{dg_pred_s:>9}{rmsd_s:>8}{r['n_pep_residues']:>5}")

    print("\n" + "=" * 76)
    print("Headline numbers")
    print("=" * 76)
    for label, predicate in [
        ("ALL with pKd", lambda r: r['pkd'] is not None),
        ("test10 only",  lambda r: r['set'] == 'test10' and r['pkd'] is not None),
        ("cluster_reps", lambda r: r['set'] == 'cluster_reps'),
    ]:
        rows = [r for r in results if predicate(r)]
        r_, rmse, n = stats(rows)
        if r_ is None:
            print(f"  {label} (n={n}): not enough data")
        else:
            print(f"  {label} (n={n}): Pearson r = {r_:+.3f}, RMSE = {rmse:.2f} kcal/mol")

    # RMSD distribution
    rmsds = [r['rmsd_to_crystal'] for r in results if r['rmsd_to_crystal'] is not None]
    if rmsds:
        rmsds_sorted = sorted(rmsds)
        print(f"\n  Top-1 Cα RMSD-to-crystal (n={len(rmsds)}):")
        print(f"    min  = {min(rmsds):.2f} Å")
        print(f"    median = {rmsds_sorted[len(rmsds)//2]:.2f} Å")
        print(f"    mean = {np.mean(rmsds):.2f} Å")
        print(f"    max  = {max(rmsds):.2f} Å")
        print(f"    fraction ≤ 2.5 Å (literature near-native): "
              f"{sum(1 for x in rmsds if x <= 2.5)}/{len(rmsds)} "
              f"= {100*sum(1 for x in rmsds if x <= 2.5)/len(rmsds):.0f}%")
        print(f"    fraction ≤ 5.0 Å: "
              f"{sum(1 for x in rmsds if x <= 5.0)}/{len(rmsds)} "
              f"= {100*sum(1 for x in rmsds if x <= 5.0)/len(rmsds):.0f}%")

    # Save JSON
    out = RUN_DIR / "aggregate_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nFull results: {out}")


if __name__ == "__main__":
    main()
