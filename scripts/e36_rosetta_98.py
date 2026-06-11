"""E36 — Rosetta ref2015 + FastRelax (FlexPepDock-style interface energy) on the 98 benchmark.

FlexPepDock's published r=0.59 is WITHIN-target (mutants of one peptide). This computes the
same physics (ref2015 reweighted interface energy after FastRelax) CROSS-target on the
independent 98 — directly comparable to PPI-Affinity 0.554 and ours 0.228 on the SAME data.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from rosetta_ref2015_eval import init_rosetta, score_complex  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

RELAX = "--norelax" not in sys.argv


def main():
    pr = init_rosetta()
    b98 = json.loads(Path("/tmp/e28_feats.json").read_text())
    work = Path("/tmp/ppep_work")
    out_path = Path("/tmp/e36_rosetta98.json")
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    t0 = time.time()
    for key, r in b98.items():
        if key in out:
            continue
        pep = work / f"{key}_pep.pdb"; rec = work / f"{key}_rec.pdb"
        if not pep.exists() or not rec.exists():
            continue
        try:
            s = score_complex(pr, str(pep), str(rec), relax=RELAX)
        except Exception as e:  # noqa: BLE001
            print(f"  {key} FAIL {type(e).__name__}", flush=True); continue
        out[key] = dict(y=r["y"], **s)
        out_path.write_text(json.dumps(out))
        if len(out) % 10 == 0:
            print(f"  {len(out)} done ({(time.time()-t0)/len(out):.1f}s/complex)", flush=True)
    y = np.array([v["y"] for v in out.values()])
    print(f"\n=== Rosetta ref2015+relax on the 98 (n={len(out)}, {(time.time()-t0)/max(1,len(out)):.0f}s/complex) ===")
    for f in ["ros_total", "ros_ifdG"]:
        v = np.array([vv[f] for vv in out.values()])
        if v.std() > 0:
            print(f"  raw corr({f}, ΔG) = {pearsonr(v, y).statistic:+.3f}")
    # fair LOO-fit (like ours + PPI-Affinity)
    def loo1(x, y):
        p = np.zeros(len(y))
        for i in range(len(y)):
            tr = [j for j in range(len(y)) if j != i]
            a, b = np.polyfit(x[tr], y[tr], 1); p[i] = a * x[i] + b
        return pearsonr(p, y).statistic, np.sqrt(((p - y) ** 2).mean())
    for f in ["ros_ifdG", "ros_total"]:
        v = np.array([vv[f] for vv in out.values()])
        if v.std() > 0:
            r, e = loo1(v, y)
            print(f"  Rosetta {f:<10} LOO-fit: r={r:+.3f} RMSE={e:.2f}")
    print("  [same data] PPI-Affinity 0.554 | ours geometry+MJ 0.228 | PRODIGY 0.127")


if __name__ == "__main__":
    main()
