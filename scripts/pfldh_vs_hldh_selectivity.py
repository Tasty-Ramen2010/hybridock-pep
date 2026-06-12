"""Compare PfLDH (on-target) and hLDH (off-target) dock runs for LISDAELEAIFEADC.

Reports ΔΔG = ΔG_PfLDH − ΔG_hLDH (negative ⇒ peptide selective for PfLDH),
along with per-run cluster structure, contact-residue patterns, and the
MM-GBSA spread between the two runs.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PFLDH_RUN = ROOT / "runs" / "pfldh_lisdaeleaifeadc_v4"  # v4 = same poses as v3, scored with v1.4 calibration (matches hLDH)
HLDH_RUN = ROOT / "runs" / "hldh_lisdaeleaifeadc"


def topk_hybrid(run_dir: Path, k: int = 10) -> list[float]:
    rows = list(csv.DictReader((run_dir / "ranked_poses.csv").open()))
    return [float(r["hybrid_score"]) for r in rows[:k]
            if r["hybrid_score"] not in ("", None)]


def mmgbsa_dg(run_dir: Path) -> list[float]:
    rows = list(csv.DictReader((run_dir / "ranked_poses.csv").open()))
    return [float(r["mmgbsa_dg"]) for r in rows
            if r.get("mmgbsa_dg") not in ("", None)]


def best_pose_summary(run_dir: Path) -> dict:
    rows = list(csv.DictReader((run_dir / "ranked_poses.csv").open()))
    if not rows:
        return {}
    best = rows[0]
    return {
        "pose": best["pose_filename"],
        "hybrid": float(best["hybrid_score"]),
        "vina": float(best["vina_score"]) if best["vina_score"] else None,
        "mmgbsa": float(best["mmgbsa_dg"]) if best.get("mmgbsa_dg") else None,
        "n_contact": int(best["n_contact_residues"]) if best["n_contact_residues"] else 0,
    }


def cluster_summary(run_dir: Path) -> list[dict]:
    p = run_dir / "cluster_summary.csv"
    if not p.exists():
        return []
    return list(csv.DictReader(p.open()))


def contact_residues(run_dir: Path, pose_file: str, cutoff: float = 4.5) -> list[str]:
    """Get contact residues for one pose by recomputing against cropped receptor."""
    receptor = run_dir / "receptor_for_rapidock.pdb"
    pose = run_dir / "poses_scored" / pose_file
    if not receptor.exists() or not pose.exists():
        return []
    pep_xyz = []
    for line in pose.read_text().splitlines():
        if not line.startswith("ATOM"): continue
        if line[12:16].strip().startswith("H"): continue
        try:
            pep_xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError: continue
    pep = np.array(pep_xyz) if pep_xyz else np.zeros((0, 3))
    if pep.size == 0:
        return []
    by_res = defaultdict(list)
    for line in receptor.read_text().splitlines():
        if not line.startswith("ATOM"): continue
        if line[12:16].strip().startswith("H"): continue
        try:
            r = int(line[22:26].strip())
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        except ValueError: continue
        by_res[(line[21], r, line[17:20].strip())].append(xyz)
    out = []
    for key, atoms in by_res.items():
        a = np.array(atoms)
        d = np.sqrt(((pep[:, None] - a[None]) ** 2).sum(-1)).min()
        if d <= cutoff:
            out.append(f"{key[2]}{key[1]}{key[0]}")
    return sorted(out)


def bootstrap_ddg(t_dg: list[float], o_dg: list[float],
                  n_iter: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    """Paired bootstrap of mean ΔΔG. Returns (point_estimate, lo95, hi95)."""
    rng = np.random.default_rng(seed)
    t = np.array(t_dg); o = np.array(o_dg)
    diffs = np.empty(n_iter)
    for i in range(n_iter):
        ti = t[rng.integers(0, t.size, t.size)]
        oi = o[rng.integers(0, o.size, o.size)]
        diffs[i] = ti.mean() - oi.mean()
    return float(t.mean() - o.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> None:
    print("=" * 72)
    print("LISDAELEAIFEADC selectivity: PfLDH (target) vs hLDH (off-target)")
    print("=" * 72)

    pf_topk = topk_hybrid(PFLDH_RUN, k=10)
    hl_topk = topk_hybrid(HLDH_RUN, k=10)
    pf_best = best_pose_summary(PFLDH_RUN)
    hl_best = best_pose_summary(HLDH_RUN)
    pf_clusters = cluster_summary(PFLDH_RUN)
    hl_clusters = cluster_summary(HLDH_RUN)

    # Bootstrap ΔΔG over top-K hybrid
    ddg, lo, hi = bootstrap_ddg(pf_topk, hl_topk, n_iter=2000)

    print(f"\n--- Best-pose comparison ---")
    print(f"{'':<10} {'pose':<14} {'hybrid':>8} {'vina':>7} {'GBSA':>8} {'contacts':>10}")
    print(f"{'PfLDH':<10} {pf_best['pose']:<14} "
          f"{pf_best['hybrid']:>8.2f} "
          f"{pf_best['vina'] or 0:>7.2f} "
          f"{pf_best.get('mmgbsa') or 'n/a':>8} "
          f"{pf_best['n_contact']:>10}")
    print(f"{'hLDH':<10} {hl_best['pose']:<14} "
          f"{hl_best['hybrid']:>8.2f} "
          f"{hl_best['vina'] or 0:>7.2f} "
          f"{hl_best.get('mmgbsa') or 'n/a':>8} "
          f"{hl_best['n_contact']:>10}")

    print(f"\n--- Cluster structure ---")
    for label, clusters in [("PfLDH", pf_clusters), ("hLDH", hl_clusters)]:
        for c in clusters[:3]:
            print(f"  {label} cluster {c['cluster_id']}: n={c['n_poses']}, "
                  f"mean ΔG = {float(c['mean_hybrid_score']):.2f} ± "
                  f"{float(c['std_hybrid_score']):.2f}, best pose={c['best_pose_idx']}")

    print(f"\n--- Mean top-10 ΔG comparison ---")
    print(f"  PfLDH: {np.mean(pf_topk):.2f} ± {np.std(pf_topk):.2f} kcal/mol  (n={len(pf_topk)})")
    print(f"  hLDH:  {np.mean(hl_topk):.2f} ± {np.std(hl_topk):.2f} kcal/mol  (n={len(hl_topk)})")
    print(f"\n--- Selectivity (top-K hybrid) ---")
    print(f"  ΔΔG = ΔG_PfLDH − ΔG_hLDH = {ddg:+.2f} kcal/mol")
    print(f"  Bootstrap 95% CI: [{lo:+.2f}, {hi:+.2f}] kcal/mol")
    verdict = ("Selective for PfLDH" if hi < 0
               else "Selective for hLDH" if lo > 0
               else "Inconclusive (CI crosses zero)")
    print(f"  Verdict: {verdict}")

    # MM-GBSA comparison if available
    pf_mmgbsa = mmgbsa_dg(PFLDH_RUN)
    hl_mmgbsa = mmgbsa_dg(HLDH_RUN)
    if pf_mmgbsa and hl_mmgbsa:
        print(f"\n--- MM-GBSA spread ---")
        print(f"  PfLDH: best={min(pf_mmgbsa):+.2f}  mean={np.mean(pf_mmgbsa):+.2f} kcal/mol (n={len(pf_mmgbsa)})")
        print(f"  hLDH:  best={min(hl_mmgbsa):+.2f}  mean={np.mean(hl_mmgbsa):+.2f} kcal/mol (n={len(hl_mmgbsa)})")
        print(f"  ΔΔG_GBSA (best): {min(pf_mmgbsa) - min(hl_mmgbsa):+.2f} kcal/mol")

    # Contact residue overlap
    pf_contacts = contact_residues(PFLDH_RUN, pf_best['pose'])
    hl_contacts = contact_residues(HLDH_RUN, hl_best['pose'])
    print(f"\n--- Best-pose contact residues ---")
    print(f"  PfLDH ({len(pf_contacts)}): {', '.join(pf_contacts[:15])}{' ...' if len(pf_contacts) > 15 else ''}")
    print(f"  hLDH  ({len(hl_contacts)}): {', '.join(hl_contacts[:15])}{' ...' if len(hl_contacts) > 15 else ''}")


if __name__ == "__main__":
    main()
