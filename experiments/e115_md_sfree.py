"""E115b — GPU MD free-state conformational entropy (score-env) on the built peptides.

The missing physics the atlas (e114) diagnosed: conformational entropy (vlong residual grows with
extendedness/disorder). Compute it for real via 60ps free-peptide MD (OpenMM/GBn2/CUDA), ~8-15s/pep.
Resumable → data/sfree_results.jsonl (keyed by sequence hash). Then e116 grades whether adding s_free
recovers the failure regimes (vlong especially) — i.e., whether computing the missing physics fixes it.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PEPDIR = ROOT / "data" / "sfree_peptides"
OUT = ROOT / "data" / "sfree_results.jsonl"


def main():
    from hybridock_pep.scoring.free_entropy import compute_free_state_entropy

    index = json.loads((PEPDIR / "index.json").read_text())  # seq -> hash
    done = set()
    if OUT.exists():
        for ln in OUT.read_text().splitlines():
            try:
                done.add(json.loads(ln)["hash"])
            except Exception:  # noqa: BLE001
                pass
    todo = [(seq, h) for seq, h in index.items() if h not in done and (PEPDIR / f"{h}.pdb").exists()]
    print(f"=== E115b GPU MD s_free: {len(todo)} to do ({len(done)} cached) ===", flush=True)
    t0 = time.time()
    with open(OUT, "a") as fh:
        for i, (seq, h) in enumerate(todo):
            try:
                r = compute_free_state_entropy(PEPDIR / f"{h}.pdb", prod_ps=60)
                if r:
                    fh.write(json.dumps({"hash": h, "seq": seq, "len": len(seq), **r}) + "\n")
                    fh.flush()
            except Exception as e:  # noqa: BLE001
                print(f"  {seq[:16]} FAIL {type(e).__name__}", flush=True)
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(todo)}  ({el/(i+1):.1f}s/pep, ETA {el/(i+1)*(len(todo)-i-1)/60:.0f}min)", flush=True)
    print(f"=== done in {(time.time()-t0)/60:.0f}min ===", flush=True)


if __name__ == "__main__":
    main()
