"""Score the crystal-pose benchmark with MM-GBSA (+IE/+3-traj) — overhaul validation.

Runs the enhanced MM-GBSA path on each native crystal complex in
data/benchmark_crystal.json and correlates ΔG_pred with experimental ΔG. Because
the poses are crystallographic, this isolates the *scoring* function from docking
error — the honest test of whether εin / IE / 3-traj move the r≈0.42 Vina-docked
CV baseline.

CPU-only (force_cpu) so it never contends with the GPU production dock. Writes
results incrementally to a JSON so a long background run is resumable/monitorable.

Usage:
    python scripts/score_crystal_benchmark.py --limit 3            # smoke test
    python scripts/score_crystal_benchmark.py --ie --3traj         # full run
    python scripts/score_crystal_benchmark.py --eps 2.0            # tuned dielectric
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from hybridock_pep.scoring.mmgbsa import compute_mmgbsa_single
from hybridock_pep.scoring.interaction_entropy import (
    interaction_entropy,
    sample_interaction_energies,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "benchmark_crystal.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="Only score the first N (smoke test).")
    ap.add_argument("--ie", action="store_true", help="Add Interaction-Entropy −TΔS.")
    ap.add_argument("--3traj", dest="three_traj", action="store_true", help="Three-trajectory MM-GBSA.")
    ap.add_argument("--eps", type=float, default=1.0, help="GB internal dielectric εin.")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "benchmark_crystal_scored.json")
    args = ap.parse_args()

    rows = json.loads(MANIFEST.read_text())
    if args.limit:
        rows = rows[: args.limit]
    tag = f"εin={args.eps}{'+IE' if args.ie else ''}{'+3traj' if args.three_traj else ''}"
    print(f"Scoring {len(rows)} crystal complexes | {tag} | CPU\n")

    scored = []
    for i, r in enumerate(rows, 1):
        pocket = ROOT / r["pocket_pdb"]
        peptide = ROOT / r["peptide_pdb"]
        t0 = time.time()
        try:
            dh = compute_mmgbsa_single(
                pose_pdb=peptide, receptor_pdb=pocket, force_cpu=True,
                solute_dielectric=args.eps, three_traj=args.three_traj,
            )
            dg = dh
            ie = None
            if args.ie:
                e_int = sample_interaction_energies(
                    pose_pdb=peptide, receptor_pdb=pocket, force_cpu=True,
                    solute_dielectric=args.eps,
                )
                ie = interaction_entropy(e_int)
                dg = dh + ie
            rec = {**{k: r[k] for k in ("pdb", "pkd", "dg_exp", "peptide_len")},
                   "mmgbsa_dh": dh, "ie": ie, "mmgbsa_dg": dg}
            scored.append(rec)
            print(f"  [{i}/{len(rows)}] {r['pdb']} ΔG={dg:8.1f} "
                  f"(exp {r['dg_exp']:.1f}) {time.time()-t0:.0f}s", flush=True)
        except Exception as exc:  # noqa: BLE001 — benchmark harness: log + continue
            print(f"  [{i}/{len(rows)}] {r['pdb']} ERR {type(exc).__name__}: {str(exc)[:80]}", flush=True)
        # incremental write so a long run is recoverable
        args.out.write_text(json.dumps(scored, indent=2))

    if len(scored) >= 3:
        y = np.array([s["dg_exp"] for s in scored])
        p = np.array([s["mmgbsa_dg"] for s in scored])
        r = pearsonr(p, y).statistic
        rho = spearmanr(p, y).statistic
        # RMSE after slope+intercept refit (MM-GBSA absolute scale is arbitrary)
        A = np.vstack([p, np.ones_like(p)]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        rmse = float(np.sqrt(np.mean((m * p + b - y) ** 2)))
        print(f"\n=== {tag} | n={len(scored)} ===")
        print(f"  Pearson r = {r:+.3f}   Spearman = {rho:+.3f}   refit-RMSE = {rmse:.2f} kcal/mol")
        print(f"  (baseline to beat: Vina-docked CV r=+0.42)")
    print(f"\nWrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
