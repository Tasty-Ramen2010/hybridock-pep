"""E94 — WHY doesn't the best-RMSD (oracle) pose score the highest affinity?

Documented paradox: selecting the lowest-RMSD pose per complex gives affinity r=0.467, WORSE than RAPiDock
rank-1 (0.564). A geometrically-better pose scores affinity WORSE. This autopsy finds the mechanism.

Hypotheses tested per complex (real RAPiDock poses, crystal reference):
  H1 POSE-INVARIANCE: the affinity model is dominated by POCKET features (receptor-derived, ~constant
     across poses of one complex). If predicted ΔG barely varies across poses, pose selection is moot.
  H2 OFF-MANIFOLD: the near-native pose has interface features (BSA, contacts) that are OUT-OF-DISTRIBUTION
     vs the docked-pose distribution the model was calibrated on → its prediction is extrapolated/worse.
  H3 NO WITHIN-COMPLEX RMSD→ΔG GRADIENT: within a complex, lower RMSD does NOT yield stronger predicted
     ΔG (corr ≈ 0), so 'better pose' carries no affinity signal.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CAMP = ROOT / "runs" / "e93_realpose_campaign"
from Bio.PDB import PDBParser  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

P = PDBParser(QUIET=True)


def ca(pdb):
    m = P.get_structure("x", str(pdb))[0]
    return np.array([a.coord for ch in m for r in ch if r.id[0] == " " for a in r if a.name == "CA"])


def rmsd(a, b):
    n = min(len(a), len(b))
    if n < 3:
        return np.nan
    a, b = a[:n] - a[:n].mean(0), b[:n] - b[:n].mean(0)
    H = a.T @ b
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.sqrt(((a @ R.T - b) ** 2).sum(1).mean()))


def main():
    from hybridock_pep.scoring.geometry_features import compute_geometry_features
    from hybridock_pep.scoring.ensemble import EnsembleCalibration, score as escore
    cal = EnsembleCalibration.load(ROOT / "data/ensemble_calibration.json")
    bench = {r["pdb"]: r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}

    complexes = sorted([d.name for d in CAMP.iterdir() if (d / "poses").exists()])
    print(f"=== E94 oracle-pose autopsy on {len(complexes)} complexes (real RAPiDock poses) ===\n")
    POCKET = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis"]
    IFACE = ["bsa_hyd", "sasa_hb", "sasa_sb", "arom_cc", "hb_count"]

    rows_summary = []
    feat_var_pocket, feat_var_iface, dg_var = [], [], []
    within_rmsd_dg = []
    for cx in complexes:
        meta = bench.get(cx)
        if not meta:
            continue
        xtal = ca(meta["peptide_pdb"])
        rec = Path(meta["pocket_pdb"]).resolve()
        poses = sorted((CAMP / cx / "poses").glob("pose_*.pdb"),
                       key=lambda p: int(p.stem.split("_")[1]))[:40]
        recs = []
        for p in poses:
            try:
                f = compute_geometry_features(p, rec)
                if not f:
                    continue
                dg = escore(f, 0.0, cal)  # geometry-only ΔG (vina=0 → geometry dominates the z-blend)
                recs.append((rmsd(ca(p), xtal), dg, f))
            except Exception:  # noqa: BLE001
                pass
        if len(recs) < 10:
            continue
        rms = np.array([r[0] for r in recs])
        dgs = np.array([r[1] for r in recs])
        # H1: variance of pocket vs interface features across poses (coeff of variation)
        def cv(key):
            v = np.array([r[2].get(key, np.nan) for r in recs])
            v = v[~np.isnan(v)]
            return np.std(v) / (abs(np.mean(v)) + 1e-9) if len(v) else np.nan
        cvp = np.nanmean([cv(k) for k in POCKET])
        cvi = np.nanmean([cv(k) for k in IFACE])
        feat_var_pocket.append(cvp); feat_var_iface.append(cvi); dg_var.append(np.std(dgs))
        # H3: within-complex corr(RMSD, predicted ΔG) — does lower RMSD → stronger ΔG?
        wc = pearsonr(rms, dgs)[0] if np.std(rms) > 0 and np.std(dgs) > 0 else np.nan
        within_rmsd_dg.append(wc)
        # H2: is the best-RMSD pose an outlier in ΔG? (z-score of its ΔG within the complex)
        best_i = int(np.argmin(rms))
        z_best = (dgs[best_i] - dgs.mean()) / (dgs.std() + 1e-9)
        rows_summary.append((cx, rms.min(), rms.mean(), np.std(dgs), wc, z_best))
        print(f"  {cx}: bestRMSD={rms.min():.1f}Å  ΔG_std={np.std(dgs):.2f}  "
              f"corr(RMSD,ΔG)={wc:+.2f}  best-pose ΔG z={z_best:+.2f}")

    print("\n=== MECHANISM ===")
    print(f"H1 POSE-INVARIANCE: pocket-feature CV={np.nanmean(feat_var_pocket):.3f}  "
          f"interface-feature CV={np.nanmean(feat_var_iface):.3f}")
    print(f"   predicted ΔG std across poses (per complex) ≈ {np.nanmean(dg_var):.2f} kcal/mol")
    print(f"H3 WITHIN-COMPLEX corr(RMSD, predicted ΔG) = {np.nanmean(within_rmsd_dg):+.3f} "
          f"(±{np.nanstd(within_rmsd_dg):.2f})")
    print(f"H2 best-RMSD pose mean ΔG z-score = {np.nanmean([r[5] for r in rows_summary]):+.3f} "
          f"(0 = typical, >0 = WEAKER than its peers)")
    print("\n  reading:")
    print("   - if pocket CV << interface CV: affinity rides the POSE-INVARIANT pocket → selection barely moves ΔG")
    print("   - if corr(RMSD,ΔG) ≈ 0: lower RMSD carries NO affinity signal (the decoupling, mechanistic)")
    print("   - if best-pose z > 0: the near-native pose scores WEAKER than typical → oracle selection HURTS")


if __name__ == "__main__":
    main()
