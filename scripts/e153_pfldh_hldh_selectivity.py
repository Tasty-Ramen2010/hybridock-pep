"""E153 — PfLDH vs hLDH selectivity for LISDAELEAIFEADC with the real-pose production model.

Score the 100 real RAPiDock poses for each receptor (the actual deployment scenario), take the top-5
ensemble mean ΔG, and report selectivity ΔΔG = ΔG(PfLDH) − ΔG(hLDH). Negative ⇒ PfLDH-selective (the
desired malaria-diagnostic direction). Uses the real-pose-trained affinity model (E152: crystal model
collapses on RAPiDock poses).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybridock_pep.scoring.affinity_model import predict_affinity  # noqa: E402
from hybridock_pep.scoring.geometry_features import compute_geometry_features  # noqa: E402

PEPTIDE = "LISDAELEAIFEADC"
RUNS = {"PfLDH (1T2D, target)": "pfldh_lisdaeleaifeadc", "hLDH (1I0Z, off-target)": "hldh_lisdaeleaifeadc"}


def score_run(run_dir: Path, n_poses: int = 25):
    rec = run_dir / "receptor_for_rapidock.pdb"
    poses = sorted((run_dir / "poses").glob("pose_*.pdb"),
                   key=lambda p: int(p.stem.split("_")[1]))[:n_poses]
    dgs = []
    for pose in poses:
        feats = compute_geometry_features(pose, rec)
        if feats is None:
            continue
        dg = predict_affinity(feats, PEPTIDE)
        if dg is not None:
            dgs.append(dg)
    return np.array(dgs)


def main():
    print(f"=== E153 PfLDH vs hLDH selectivity — {PEPTIDE} (real-pose model) ===\n")
    res = {}
    for label, rn in RUNS.items():
        d = ROOT / "runs" / rn
        dgs = score_run(d)
        if dgs.size == 0:
            print(f"  {label}: no scoreable poses"); continue
        top5 = np.sort(dgs)[:5].mean()
        res[label] = {"top5": top5, "rank1": dgs.min(), "mean": dgs.mean(), "n": dgs.size}
        print(f"  {label:<26} top-5 ens ΔG={top5:+.2f}  rank-1={dgs.min():+.2f}  "
              f"mean={dgs.mean():+.2f} kcal/mol  (n={dgs.size})")
    if len(res) == 2:
        pf = res["PfLDH (1T2D, target)"]["top5"]
        hl = res["hLDH (1I0Z, off-target)"]["top5"]
        ddg = pf - hl
        print(f"\n  SELECTIVITY ΔΔG (PfLDH − hLDH) = {ddg:+.2f} kcal/mol")
        print(f"  → {'PfLDH-SELECTIVE (desired)' if ddg < 0 else 'hLDH-leaning' if ddg > 0 else 'non-selective'}")
        print("\n  caveat: absolute ΔG on a charged 15-mer is near the FEP floor; selectivity ΔΔG is the more")
        print("  trustworthy signal (the floor cancels between two similar receptors). Real-pose-model scored.")


if __name__ == "__main__":
    main()
