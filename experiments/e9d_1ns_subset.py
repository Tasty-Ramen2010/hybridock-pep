"""E9d — 1 ns MD-LIE on a representative subset, to measure the 60ps vs 1ns delta.

Runs the same single-trajectory MM-GBSA+IE as e9 but at 1000 ps, on a fixed
subset spanning families, so we can compare per-complex against the 60ps results.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e9_md_ensemble_ie import run_complex  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUBSET = ["1NRL", "1Q8W", "1T5Z", "1ZGY", "3HQR", "1A1M", "2O02", "4N7H"]


def main():
    rows = json.loads(Path("/tmp/e0_rows.json").read_text())
    by = {r["pdb"].upper(): r for r in rows if r.get("pep_pdb")}
    pick = [by[p] for p in SUBSET if p in by][:8]
    if len(pick) < 8:
        # backfill with first available
        for r in rows:
            if r.get("pep_pdb") and r not in pick:
                pick.append(r)
            if len(pick) >= 8:
                break
    out = []
    t0 = time.time()
    for i, r in enumerate(pick):
        ts = time.time()
        try:
            res = run_complex(r["pep_pdb"], r["poc_pdb"], prod_ps=1000, frame_every_ps=10)
            res.update(pdb=r["pdb"], y=r["y"], L=r["L"], aff=r["aff"])
            out.append(res)
            print(f"[{i+1}/{len(pick)}] {r['pdb']}: dg_pred={res['dg_pred']:.1f} "
                  f"<E_int>={res['e_int_mean']:.1f} -TdS={res['minus_tds_ie']:.1f} "
                  f"(exp {r['y']:.1f}) {time.time()-ts:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{i+1}] {r['pdb']} FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
    Path("/tmp/e9d_1ns.json").write_text(json.dumps(out))
    print(f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
