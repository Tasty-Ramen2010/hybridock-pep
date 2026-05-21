"""Quick scoring benchmark: run each PepSet fixture family through the pipeline
and print Vina, AD4, hybrid scores, entropy correction, and n_contact_residues.
Used to populate the comparison table in docs/dataset_analysis.md.

Usage:
    conda run --no-capture-output -n score-env python scripts/score_family_benchmark.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from hybridock_pep import driver
from hybridock_pep.models import DockConfig

FIXTURES = REPO / "tests" / "fixtures"
CAL = FIXTURES / "mdm2_calibration.json"

CASES = [
    ("pdz_1jq8",  "PDZ domain",              "LAIYS",               (12.36,  21.26,  40.31), 30.0),
    ("sh2_1jw6",  "SH2 domain",              "MYWYPY",              (12.56,  32.23,  25.92), 32.0),
    ("brd_3shb",  "Bromodomain",             "ARTKQTA",             (-12.52, 10.72,  13.38), 32.0),
    ("cam_3bej",  "Calmodulin / EF-hand",    "ERHKILHRLLQ",         (29.16,   9.09,  -0.39), 35.0),
    ("bcl2_2vzg", "BCL-2 / BH3",             "NLSELDRLLLELNAV",     (-19.93,-12.78,  18.06), 38.0),
    ("mdm2_1pmx", "MDM2 / MDMX",             "RNCFESVAALRRCMYG",   (-2.82,  -1.18,  -5.17), 29.0),
    ("kin_2khh",  "Kinase substrate",        "DSGFSFGSK",           (-12.00,-45.84,  13.78), 32.0),
    ("helix_1yfn","Amphipathic helix",       "EAQPAPHQWQKMPFWQKV",  (30.75, -17.88,   7.80), 37.0),
    ("arm_2cny",  "ARM / HEAT repeat",       "GSFLPNSEQQKSVDAVFSS", (30.75, -13.23,  44.33), 65.0),
    ("sh3_1a0n",  "SH3 / PPXP",             "PPRPLPVAPGSSKT",      (-5.83,  -4.45, -21.04), 38.0),
    ("ww_1ywi",   "WW domain",              "PPPLPP",               (0.90,    7.77,   0.59), 28.0),
]


def run_case(tag, family, peptide, site, box, tmpdir):
    out = Path(tmpdir) / tag
    config = DockConfig(
        peptide_sequence=peptide,
        receptor_path=FIXTURES / tag / "receptor_pocket.pdb",
        site_coords=site,
        box_size=box,
        n_samples=5,
        output_dir=out,
        seed=42,
        scoring={"vina", "ad4"},
    )
    try:
        scored_poses, _ = driver.run_dock(
            config=config,
            input_poses_dir=FIXTURES / tag,
            calibration_path=CAL,
        )
    except Exception as e:
        return {"family": family, "tag": tag, "error": str(e)}

    csv_path = out / "ranked_poses.csv"
    if not csv_path.exists():
        return {"family": family, "tag": tag, "error": "no ranked_poses.csv"}

    import csv
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return {"family": family, "tag": tag, "error": "empty ranked_poses.csv"}

    best = rows[0]
    return {
        "family": family,
        "tag": tag,
        "n_res": len(peptide),
        "vina": float(best["vina_score"]) if best["vina_score"] else None,
        "ad4":  float(best["ad4_score"])  if best["ad4_score"]  else None,
        "ec":   float(best["entropy_correction"]) if best["entropy_correction"] else None,
        "hybrid": float(best["hybrid_score"]) if best["hybrid_score"] else None,
        "n_contact": best.get("n_contact_residues", "?"),
        "ad4_anomaly": best.get("is_ad4_anomaly", "?"),
    }


def main():
    import logging
    logging.disable(logging.CRITICAL)

    print(f"{'Family':<28} {'Seq':>4} {'Vina':>8} {'AD4':>8} {'EC':>7} {'Hybrid':>9} {'Contacts':>9} {'AD4_anom':>9}")
    print("-" * 90)

    with tempfile.TemporaryDirectory() as tmpdir:
        for tag, family, peptide, site, box in CASES:
            r = run_case(tag, family, peptide, site, box, tmpdir)
            if "error" in r:
                print(f"{r['family']:<28} ERROR: {r['error']}")
            else:
                v  = f"{r['vina']:+.2f}"  if r['vina']   is not None else "  N/A"
                a  = f"{r['ad4']:+.2f}"   if r['ad4']    is not None else "  N/A"
                ec = f"{r['ec']:+.2f}"    if r['ec']     is not None else "  N/A"
                h  = f"{r['hybrid']:+.2f}" if r['hybrid'] is not None else "  N/A"
                print(f"{r['family']:<28} {r['n_res']:>4} {v:>8} {a:>8} {ec:>7} {h:>9} {r['n_contact']:>9} {r['ad4_anomaly']:>9}")


if __name__ == "__main__":
    main()
