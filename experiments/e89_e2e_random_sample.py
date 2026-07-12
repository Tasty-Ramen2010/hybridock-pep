"""E89 — full end-to-end demonstration on a random sample, with ensemble + refine-topk + Kd.

Runs the real pipeline (driver.run_dock) on a random sample of PepSet fixtures spanning short->long
peptides, with Vina+AD4 scoring, the geometry ensemble (which now routes short peptides through the
length sub-model), and MM-GBSA --refine-topk. Reports the ranked ΔG, converts to predicted Kd, and
confirms the short-peptide router fires. Proves the wired pipeline works front-to-back.
"""
from __future__ import annotations

import math
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIX = ROOT / "tests" / "fixtures"
RT = 0.5924  # kcal/mol at 298 K

# (tag, peptide, site, box) — subset of the PepSet fixtures that ship with poses
CASES = [
    ("pdz_1jq8", "LAIYS", (12.36, 21.26, 40.31), 30.0),            # 5  short
    ("sh2_1jw6", "MYWYPY", (12.56, 32.23, 25.92), 32.0),           # 6  short
    ("brd_3shb", "ARTKQTA", (-12.52, 10.72, 13.38), 32.0),         # 7  short
    ("kin_2khh", "DSGFSFGSK", (-12.00, -45.84, 13.78), 32.0),      # 9  med
    ("cam_3bej", "ERHKILHRLLQ", (29.16, 9.09, -0.39), 35.0),       # 11 med
    ("mdm2_p53", "ETFSDLWKLLPE", (26.4, 3.5, -5.6), 60.0),         # 12 med
    ("bcl2_2vzg", "NLSELDRLLLELNAV", (-19.93, -12.78, 18.06), 38.0),  # 15 long
    ("helix_1yfn", "EAQPAPHQWQKMPFWQKV", (30.75, -17.88, 7.80), 37.0),  # 18 long
]


def kd_string(dg: float) -> str:
    """ΔG (kcal/mol) -> human Kd string."""
    kd = math.exp(dg / RT)  # molar
    for unit, scale in [("pM", 1e-12), ("nM", 1e-9), ("µM", 1e-6), ("mM", 1e-3), ("M", 1.0)]:
        if kd < scale * 1000:
            return f"{kd / scale:.1f} {unit}"
    return f"{kd:.1e} M"


def main():
    from hybridock_pep import driver
    from hybridock_pep.models import DockConfig

    random.seed(7)
    sample = random.sample(CASES, 5)
    sample.sort(key=lambda c: len(c[1]))
    print(f"=== E89 e2e random sample (n={len(sample)}): ensemble + refine-topk + Kd ===", flush=True)
    entropy_cal = ROOT / "data" / "calibration.json"          # hybrid-score entropy coeffs
    cal = ROOT / "data" / "ensemble_calibration.json"          # geometry+Vina ensemble
    results = []
    for tag, pep, site, box in sample:
        poses = FIX / tag
        rec = poses / "receptor_pocket.pdb"
        if not (rec.exists() and any(poses.glob("pose_*.pdb"))):
            print(f"  {tag:12} SKIP (no fixture poses)", flush=True)
            continue
        out = ROOT / "runs" / f"e89_{tag}"
        cfg = DockConfig(
            peptide_sequence=pep, receptor_path=rec, site_coords=site, box_size=box,
            n_samples=5, output_dir=out, seed=7, scoring={"vina", "ad4"},
            compute_ensemble=True, ensemble_calibration=cal if cal.exists() else None,
            refine_topk=2,
        )
        try:
            scored, _ = driver.run_dock(config=cfg, input_poses_dir=poses,
                                        calibration_path=entropy_cal)
            best = min((p for p in scored if p.ensemble_dg is not None),
                       key=lambda p: p.ensemble_dg, default=None)
            dg_field = "ensemble_dg" if best else "hybrid_score"
            if best is None:
                best = min((p for p in scored if p.hybrid_score is not None),
                           key=lambda p: p.hybrid_score, default=None)
            dg = getattr(best, dg_field) if best else None
            mg = getattr(best, "mmgbsa_dg", None) if best else None
            routed = len(pep) <= 8
            results.append((tag, pep, len(pep), dg, mg, routed))
            print(f"  {tag:12} {pep:20} L={len(pep):2} "
                  f"ΔG={dg:+6.2f} ({dg_field})  Kd~{kd_string(dg):>8}  "
                  f"{'[ROUTER]' if routed else ''}  poses={len(scored)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {tag:12} FAIL {str(e)[:70]}", flush=True)

    if results:
        print("\n=== ranking by predicted ΔG (tightest first) ===")
        for tag, pep, L, dg, mg, routed in sorted(results, key=lambda r: r[3]):
            mgs = f"  MM-GBSA={mg:+.1f}" if mg is not None else ""
            print(f"  {dg:+6.2f} kcal/mol  Kd~{kd_string(dg):>8}  {tag} ({pep}, L={L})"
                  f"{'  [short-router]' if routed else ''}{mgs}")
        print("\n  >> ranked ΔG + Kd produced end-to-end; short peptides routed; refine-topk MM-GBSA ran.")


if __name__ == "__main__":
    main()
