"""E34 — 3-trajectory MM-GBSA (adds reorganization entropy) on a diverse subset.

Single-pose MM-GBSA flipped (-0.33/+0.27) because it's the size-extensive enthalpy half,
missing the size-compensating entropy. 3-traj relaxes the unbound peptide + receptor on their
own, capturing the conformational REORGANIZATION term (partial entropy/strain). Does adding it
fix the flip? Subset: 12 crystal-65 + 12 of the 98, spanning the ΔG range. ~1-2 min/complex.
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
from hybridock_pep.scoring.mmgbsa import compute_mmgbsa_single  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402


def subset(items, k):
    items = sorted(items, key=lambda t: t[3])
    idx = np.linspace(0, len(items) - 1, k).astype(int)
    return [items[i] for i in idx]


def main():
    e0 = {r["pdb"].upper(): r for r in json.loads(Path("/tmp/e0_rows.json").read_text())}
    geo = json.loads(Path("/tmp/e19_cr.json").read_text())
    cr = [(g["pdb"].upper(), e0[g["pdb"].upper()].get("pep_pdb"),
           e0[g["pdb"].upper()].get("poc_pdb"), g["y"])
          for g in geo if g["pdb"].upper() in e0 and e0[g["pdb"].upper()].get("pep_pdb")]
    b98 = json.loads(Path("/tmp/e28_feats.json").read_text())
    work = Path("/tmp/ppep_work")
    b9 = [(k, str(work / f"{k}_pep.pdb"), str(work / f"{k}_rec.pdb"), r["y"])
          for k, r in b98.items() if (work / f"{k}_pep.pdb").exists()]
    out_path = Path("/tmp/e34.json")
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for tag, items in [("cr", subset(cr, 12)), ("b98", subset(b9, 12))]:
        for key, pep, rec, y in items:
            kk = f"{tag}:{key}"
            if kk in out:
                continue
            try:
                dg = compute_mmgbsa_single(Path(pep), Path(rec), three_traj=True)
                out[kk] = dict(dg=float(dg), y=y, tag=tag)
                out_path.write_text(json.dumps(out))
                print(f"  {kk}: 3traj dG={dg:+.1f} (exp {y:.1f}) [{len(out)}]", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  {kk} FAIL {type(e).__name__}: {str(e)[:50]}", flush=True)
    cr_ = [v for v in out.values() if v["tag"] == "cr"]
    b9_ = [v for v in out.values() if v["tag"] == "b98"]
    if len(cr_) >= 5 and len(b9_) >= 5:
        rc = pearsonr([v["dg"] for v in cr_], [v["y"] for v in cr_]).statistic
        r9 = pearsonr([v["dg"] for v in b9_], [v["y"] for v in b9_]).statistic
        verdict = "UNIVERSAL — entropy fixed the flip!" if rc * r9 > 0 else "STILL FLIPS"
        print(f"\n3-traj MM-GBSA: crystal-65 r={rc:+.3f} | the-98 r={r9:+.3f} | {verdict}")
        print("  (single-pose was -0.33/+0.27 = flipped; reorganization entropy = partial fix?)")


if __name__ == "__main__":
    main()
