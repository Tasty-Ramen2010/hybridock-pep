"""E120 — GPU MD free-state entropy, saving PER-RESIDUE profiles (for Ram's entropy-surrogate model).

run_free_dynamics returns per-residue dihedral entropy + per-residue RMSF; the scalar wrapper averaged
them away. Per-residue entropy is strongly residue-type/neighbor determined (learnable) and enables the
decomposition: entropy_lost_on_binding = Σ per-residue entropy over the CONTACTING residues. Saves both
per-residue arrays AND scalar means (so e116 scalar grading still works). Resumable → data/sfree_perres.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
PEPDIR = ROOT / "data" / "sfree_peptides"
OUT = ROOT / "data" / "sfree_perres.jsonl"


def main():
    from e18v2_md import run_free_dynamics

    index = json.loads((PEPDIR / "index.json").read_text())  # seq -> hash
    done = set()
    if OUT.exists():
        for ln in OUT.read_text().splitlines():
            try:
                done.add(json.loads(ln)["hash"])
            except Exception:  # noqa: BLE001
                pass
    todo = [(seq, h) for seq, h in index.items() if h not in done and (PEPDIR / f"{h}.pdb").exists()]
    print(f"=== E120 per-residue GPU MD: {len(todo)} to do ({len(done)} cached) ===", flush=True)
    t0 = time.time()
    with open(OUT, "a") as fh:
        for i, (seq, h) in enumerate(todo):
            try:
                rmsf, ent = run_free_dynamics(PEPDIR / f"{h}.pdb", 200)  # per-residue arrays
                ent = [float(x) if np.isfinite(x) else None for x in ent]
                rmsf = [float(x) for x in rmsf]
                valid = [x for x in ent if x is not None]
                rec = {"hash": h, "seq": seq, "len": len(seq),
                       "s_free": float(np.mean(valid)) if valid else None,
                       "s_free_total": float(np.sum(valid)) if valid else None,
                       "rmsf": float(np.mean(rmsf)) if rmsf else None,
                       "per_res_entropy": ent, "per_res_rmsf": rmsf}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            except Exception as e:  # noqa: BLE001
                print(f"  {seq[:16]} FAIL {type(e).__name__}", flush=True)
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(todo)} ({el/(i+1):.1f}s/pep, ETA {el/(i+1)*(len(todo)-i-1)/60:.0f}min)", flush=True)
    print(f"=== done in {(time.time()-t0)/60:.0f}min ===", flush=True)


if __name__ == "__main__":
    main()
