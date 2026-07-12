"""E326 — NEUTRAL PDBbind pose-cloud CONTROL set for N2.

N2's claim is that ensemble ⟨V_elec⟩ over the generative cloud carries the *charged* residual specifically. The
control: on NEUTRAL complexes (|net q|<2) the same ⟨V_elec⟩ should NOT track the residual (there is little
charged signal to carry). This generates the neutral clouds (508 PDBbind, pockets on disk) with the identical
inline ⟨V_elec⟩/Var machinery as e323, writing to a separate file so e325 can compare charged vs neutral.

Reuses e323.run_one verbatim (only the candidate filter + output file differ). Runs AFTER e323+e324 on the
single GPU (chain_e326_after_e324.sh). Resumable.

Run:  OMP_NUM_THREADS=1 python experiments/e326_neutral_pdbbind_clouds.py
"""
from __future__ import annotations
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e323_charged_cloud_campaign as e323  # noqa: E402  (reuse run_one/velec/GEOM/band verbatim)

POS, NEG = set("KR"), set("DE")
OUT = ROOT / "data" / "e323_neutral_clouds.jsonl"


def candidates() -> list[dict]:
    rows = [json.loads(l) for l in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines()]
    cands = []
    for r in rows:
        q = abs(sum(c in POS for c in r["seq"]) - sum(c in NEG for c in r["seq"]))
        if q >= 2:  # NEUTRAL only (the control)
            continue
        d = next((Path(p).parent for p in
                  glob.glob(str(ROOT / f"data/drive_pull/pl/P-L/*/{r['pdb']}/{r['pdb']}_pocket.pdb"))), None)
        if d is None:
            continue
        nat = sum(1 for ln in (d / f"{r['pdb']}_protein.pdb").read_text().splitlines() if ln.startswith("ATOM"))
        cands.append({"pdb": r["pdb"], "seq": r["seq"], "y": r["y"], "length": r["length"],
                      "q": q, "nat": nat, "dir": str(d)})
    cands.sort(key=lambda c: (e323.band(c["length"]), c["nat"]))
    return cands


def main() -> None:
    e323.WORK.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["pdb"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    todo = [c for c in candidates() if c["pdb"] not in done]
    print(f"=== E326 neutral-control clouds: {len(done)} done, {len(todo)} to do ===", flush=True)
    t0, n = time.time(), 0
    for c in todo:
        try:
            rec = e323.run_one(c)
        except subprocess.TimeoutExpired:
            rec = None
        except Exception as exc:  # noqa: BLE001
            print(f"  [{c['pdb']}] FAIL {str(exc)[:100]}", flush=True)
            rec = None
        if rec:
            rec["source"] = "pdbbind_neutral"
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            n += 1
            if n % 5 == 0:
                print(f"  {n} new  {(time.time()-t0)/n:.0f}s/complex  last={c['pdb']}", flush=True)
    print(f"=== E326 complete: {n} new neutral clouds ===", flush=True)


if __name__ == "__main__":
    main()
