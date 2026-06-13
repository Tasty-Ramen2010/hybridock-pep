"""E122 — prototype: 200ps per-residue free-MD on 50 DIVERSE peptides (proof before the full 6h run).

Picks 50 peptides spread across the length distribution, runs 200ps free-MD, saves per-residue dihedral
entropy + rmsf → data/sfree_proto.jsonl. Then e123 trains the context-aware per-residue surrogate on it
and reports whether the per-residue entropy is learnable (grouped CV). If yes → launch the full 922 run.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
ROOT = Path(__file__).resolve().parents[1]
PEPDIR = ROOT / "data" / "sfree_peptides"
OUT = ROOT / "data" / "sfree_proto.jsonl"
PS = 200
N = 50


def main():
    from e18v2_md import run_free_dynamics

    index = json.loads((PEPDIR / "index.json").read_text())
    items = sorted(index.items(), key=lambda kv: len(kv[0]))
    # spread 50 evenly across the length-sorted list (diverse lengths)
    pick = [items[int(i)] for i in np.linspace(0, len(items) - 1, N)]
    done = set()
    if OUT.exists():
        done = {json.loads(l)["hash"] for l in OUT.read_text().splitlines()}
    todo = [(s, h) for s, h in pick if h not in done]
    print(f"=== E122 proto MD {PS}ps on {len(todo)} peptides ({len(done)} cached) ===", flush=True)
    t0 = time.time()
    with open(OUT, "a") as fh:
        for i, (seq, h) in enumerate(todo):
            try:
                rmsf, ent = run_free_dynamics(PEPDIR / f"{h}.pdb", PS)
                ent = [float(x) if np.isfinite(x) else None for x in ent]
                fh.write(json.dumps({"hash": h, "seq": seq, "len": len(seq),
                                     "per_res_entropy": ent, "per_res_rmsf": [float(x) for x in rmsf]}) + "\n")
                fh.flush()
            except Exception as e:  # noqa: BLE001
                print(f"  {seq[:16]} FAIL {type(e).__name__}", flush=True)
            if (i + 1) % 10 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(todo)} ({el/(i+1):.0f}s/pep, ETA {el/(i+1)*(len(todo)-i-1)/60:.0f}min)", flush=True)
    print(f"=== done {(time.time()-t0)/60:.0f}min ===", flush=True)


if __name__ == "__main__":
    main()
