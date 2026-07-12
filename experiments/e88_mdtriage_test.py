"""E88 — MD-triage test: does MM-GBSA (sampling) rank long/vlong better than the static geometry model?

Long/vlong static failure was diagnosed as conformational averaging (single pose != ensemble). If true,
MM-GBSA (minimized + implicit-solvent ΔG_bind, an ensemble-ish estimate) should rank these complexes
better than the static geometry prediction. Quick test on 10 diverse long/vlong the98 complexes.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybridock_pep.scoring.mmgbsa import compute_mmgbsa_single  # noqa: E402

OUT = ROOT / "data/e88_mdtriage.jsonl"
WORK = Path("/tmp/ppep_work")


def main():
    pick = json.load(open("/tmp/mdtriage_pick.json"))
    e78 = json.load(open("/tmp/e78_dewet.json"))
    done = {json.loads(l)["id"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    print(f"=== E88 MM-GBSA on {len(pick)} long/vlong the98 ===", flush=True)
    for k in pick:
        if k in done:
            continue
        y = e78["98_" + k]["y"]
        pep, rec = WORK / f"{k}_pep.pdb", WORK / f"{k}_rec.pdb"
        t0 = time.time()
        try:
            dg = compute_mmgbsa_single(pep, rec, entropy_penalty=True)
            row = dict(id=k, y=y, mmgbsa=float(dg))
            with OUT.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"  {k:12} y={y:+6.1f} mmgbsa={dg:+8.1f} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {k:12} FAIL {str(e)[:60]}", flush=True)
    analyze()


def analyze():
    if not OUT.exists():
        return
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    if len(rows) < 5:
        print(f"\n(only {len(rows)} rows)")
        return
    # static geometry prediction for the same complexes (pooled-LOO)
    src = open(ROOT / "experiments/e80_charged_gap.py").read().split("def main")[0]
    src = src.replace("Path(__file__).resolve().parents[1]", "Path('%s')" % ROOT)
    ns = {}; exec(src, ns)
    allrows = ns["load"](); PROD = ns["PROD"]
    pred = ns["loo_pred"](allrows, PROD)
    static_by_pdb = {r["pdb"]: p for r, p in zip(allrows, pred)}
    y = np.array([r["y"] for r in rows])
    mg = np.array([r["mmgbsa"] for r in rows])
    st = np.array([static_by_pdb.get(r["id"], np.nan) for r in rows])
    m = ~np.isnan(st)
    print(f"\n=== long/vlong (n={len(rows)}): MM-GBSA vs static geometry ===")
    print(f"  MM-GBSA   vs ΔG:  Pearson={pearsonr(mg, y)[0]:+.3f}  Spearman={spearmanr(mg, y).statistic:+.3f}")
    print(f"  static    vs ΔG:  Pearson={pearsonr(st[m], y[m])[0]:+.3f}  "
          f"Spearman={spearmanr(st[m], y[m]).statistic:+.3f}  (n={m.sum()})")
    print("  >> MM-GBSA Spearman clearly > static => MD-triage for long peptides justified.")


if __name__ == "__main__":
    main()
